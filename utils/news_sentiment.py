"""
utils/news_sentiment.py — Real-Time News Sentiment Filter
==========================================================
Scans RSS feeds for market-moving keywords.
Pauses trading on unscheduled shock events.

KEYWORDS THAT TRIGGER PAUSE:
  Critical (immediate halt): rate hike surprise, RBI emergency,
  war declared, exchange circuit, market halt, force majeure
  
  High (30-min caution): inflation data miss, GDP shock,
  crude oil spike, dollar crash, FII selling massive

SOURCES:
  - NSE press releases (nseindia.com/rss)
  - Moneycontrol markets (moneycontrol.com/rss)
  - Economic Times markets (economictimes.com/rss)
  - BSE announcements

USAGE:
    filter = NewsFilter()
    check = await filter.scan()
    if check.halt_trading:
        return  # stop all entries
    if check.caution:
        lots = max(1, lots // 2)  # reduce size
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from config.settings import TELEGRAM_ENABLED
except ImportError:
    TELEGRAM_ENABLED = False

# ── Keyword dictionaries ──────────────────────────────────────────────────────

HALT_KEYWORDS = [
    "circuit breaker", "trading halt", "market suspended",
    "exchange closed", "force majeure", "war declared",
    "emergency rate", "rbi emergency",
    "black swan", "stock exchange halt",
]

HALT_PATTERNS = [
    r"\bnuclear\s+(war|attack|strike|missile|weapon|threat|fallout|radiation)\b",
    r"\b(radiation|missile)\s+(leak|attack|strike|threat)\b",
]

CAUTION_KEYWORDS = [
    "rate hike", "rate cut surprise", "inflation shock",
    "gdp miss", "recession", "crude spike", "oil shock",
    "fii massive selling", "dollar crash", "rupee crash",
    "bank crisis", "npa crisis", "systemic risk",
    "geopolitical", "sanctions india", "terror attack",
]

RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/business.xml",
]

SCAN_INTERVAL_MIN = 5    # scan every 5 minutes
HALT_DURATION_MIN = 60   # halt trading for 60 min on critical news
CAUTION_DURATION_MIN = 30
ALERT_DEDUP_MIN = 90
STATE_FILE = Path(__file__).resolve().parents[1] / "logs" / "news_filter_state.json"


@dataclass
class NewsCheck:
    halt_trading:  bool
    caution:       bool
    triggered_by:  str
    keywords_found:list[str]
    source:        str
    timestamp:     str
    note:          str


class NewsFilter:
    """
    Scans RSS feeds every 5 minutes for market-moving keywords.
    Triggers halt or caution mode automatically.
    """

    def __init__(self) -> None:
        self._last_scan:    Optional[datetime] = None
        self._halt_until:   Optional[datetime] = None
        self._caution_until:Optional[datetime] = None
        self._last_result:  Optional[NewsCheck] = None
        import os
        if os.getenv("TRADING_MODE") == "BACKTEST":
            self._alert_state = {"alerts": {}}
        else:
            self._alert_state = self._load_alert_state()

    async def scan(self) -> NewsCheck:
        """
        Scan RSS feeds for keywords.
        Returns cached result if scanned within last SCAN_INTERVAL_MIN.
        """
        import os
        if os.getenv("TRADING_MODE") == "BACKTEST":
            return NewsCheck(
                halt_trading  = False, caution=False,
                triggered_by  = "backtest_bypass",
                keywords_found= [],
                source        = "backtest",
                timestamp     = datetime.now(IST).isoformat(),
                note          = "News halt bypassed in backtesting mode",
            )
        now = datetime.now(IST)

        # Return cached result if available and refresh in background
        if self._last_result:
            if not self._last_scan or (now - self._last_scan).total_seconds() >= SCAN_INTERVAL_MIN * 60:
                asyncio.create_task(self._refresh_background(now))
            return self._last_result

        # First-time scan if no cache exists yet
        result = await self._do_scan()
        self._last_scan   = now
        self._last_result = result

        # Set halt/caution timers
        if result.halt_trading:
            self._halt_until = now + timedelta(minutes=HALT_DURATION_MIN)
            logger.warning(
                f"[NewsFilter] 🚨 HALT TRADING | "
                f"Keywords: {result.keywords_found} | "
                f"Source: {result.source}"
            )
            if self._should_send_alert(result, now):
                await self._send_alert(result)
            else:
                logger.info(
                    f"[NewsFilter] Duplicate halt alert suppressed | "
                    f"keywords={result.keywords_found} | source={result.source}"
                )
        elif result.caution:
            self._caution_until = now + timedelta(minutes=CAUTION_DURATION_MIN)
            logger.warning(
                f"[NewsFilter] ⚠️ CAUTION | "
                f"Keywords: {result.keywords_found}"
            )

        return result

    async def _refresh_background(self, now: datetime) -> None:
        try:
            result = await self._do_scan()
            self._last_scan = now
            self._last_result = result
            if result.halt_trading:
                self._halt_until = now + timedelta(minutes=HALT_DURATION_MIN)
                logger.warning(
                    f"[NewsFilter] 🚨 HALT TRADING | "
                    f"Keywords: {result.keywords_found} | "
                    f"Source: {result.source}"
                )
                if self._should_send_alert(result, now):
                    await self._send_alert(result)
            elif result.caution:
                self._caution_until = now + timedelta(minutes=CAUTION_DURATION_MIN)
                logger.warning(
                    f"[NewsFilter] ⚠️ CAUTION | "
                    f"Keywords: {result.keywords_found}"
                )
        except Exception as exc:
            logger.debug(f"[NewsFilter] Background refresh error: {exc}")

    async def _do_scan(self) -> NewsCheck:
        """Fetch and parse RSS feeds."""
        clear = NewsCheck(
            halt_trading=False, caution=False,
            triggered_by="", keywords_found=[],
            source="", timestamp=datetime.now(IST).isoformat(),
            note="clear",
        )

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for feed_url in RSS_FEEDS:
                    try:
                        async with session.get(
                            feed_url, timeout=aiohttp.ClientTimeout(total=8)
                        ) as resp:
                            if resp.status != 200:
                                continue
                            text = (await resp.text()).lower()
                            result = self._check_keywords(text, feed_url)
                            if result.halt_trading or result.caution:
                                return result
                    except Exception as e:
                        logger.debug(f"[NewsFilter] Feed {feed_url}: {e}")
                        continue
        except ImportError:
            # Fallback: requests
            try:
                import requests
                for feed_url in RSS_FEEDS:
                    try:
                        resp = requests.get(feed_url, timeout=8)
                        if resp.status_code == 200:
                            text   = resp.text.lower()
                            result = self._check_keywords(text, feed_url)
                            if result.halt_trading or result.caution:
                                return result
                    except Exception:
                        continue
            except ImportError:
                logger.debug("[NewsFilter] Neither aiohttp nor requests available")

        return clear

    def _check_keywords(self, text: str, source: str) -> NewsCheck:
        """Scan text for halt/caution keywords."""
        found_halt    = [k for k in HALT_KEYWORDS if k in text]
        found_halt.extend(
            match.group(0)
            for pattern in HALT_PATTERNS
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        )
        found_caution = [k for k in CAUTION_KEYWORDS if k in text]

        if found_halt:
            return NewsCheck(
                halt_trading  = True, caution=False,
                triggered_by  = "halt_keyword",
                keywords_found= found_halt[:3],
                source        = self._source_host(source),
                timestamp     = datetime.now(IST).isoformat(),
                note          = f"CRITICAL keywords: {', '.join(found_halt[:3])}",
            )
        if found_caution:
            return NewsCheck(
                halt_trading  = False, caution=True,
                triggered_by  = "caution_keyword",
                keywords_found= found_caution[:3],
                source        = self._source_host(source),
                timestamp     = datetime.now(IST).isoformat(),
                note          = f"Caution keywords: {', '.join(found_caution[:3])}",
            )

        return NewsCheck(False, False, "", [], "", datetime.now(IST).isoformat(), "clear")

    async def _send_alert(self, result: NewsCheck) -> None:
        if not TELEGRAM_ENABLED:
            return
        try:
            from utils.telegram_notifier import get_notifier
            await get_notifier().send_text(
                f"🚨 *NEWS HALT TRIGGERED*\n"
                f"Keywords: {', '.join(result.keywords_found)}\n"
                f"Source: {result.source}\n"
                f"Trading halted for {HALT_DURATION_MIN} min",
                target="LIVE",
            )
        except Exception:
            pass

    def _should_send_alert(self, result: NewsCheck, now: datetime) -> bool:
        key = self._alert_key(result)
        raw_ts = str((self._alert_state.get("alerts") or {}).get(key, "") or "")
        if raw_ts:
            try:
                last_ts = datetime.fromisoformat(raw_ts)
                if last_ts.tzinfo is None:
                    last_ts = IST.localize(last_ts)
                if now - last_ts < timedelta(minutes=ALERT_DEDUP_MIN):
                    return False
            except Exception:
                pass
        alerts = dict(self._alert_state.get("alerts") or {})
        alerts[key] = now.isoformat()
        cutoff = now - timedelta(hours=24)
        cleaned = {}
        for alert_key, ts_text in alerts.items():
            try:
                ts = datetime.fromisoformat(str(ts_text))
                if ts.tzinfo is None:
                    ts = IST.localize(ts)
                if ts >= cutoff:
                    cleaned[alert_key] = ts.isoformat()
            except Exception:
                continue
        self._alert_state["alerts"] = cleaned
        self._save_alert_state()
        return True

    @staticmethod
    def _source_host(source: str) -> str:
        parts = str(source or "").split("/")
        if len(parts) > 2 and parts[2]:
            return parts[2]
        return str(source or "")

    @staticmethod
    def _alert_key(result: NewsCheck) -> str:
        payload = "|".join([
            str(result.triggered_by or ""),
            str(result.source or ""),
            ",".join(sorted(str(k).lower() for k in result.keywords_found)),
        ])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _load_alert_state() -> dict:
        try:
            if STATE_FILE.exists():
                return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
        return {"alerts": {}}

    def _save_alert_state(self) -> None:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._alert_state, indent=2, sort_keys=True))
            tmp.replace(STATE_FILE)
        except Exception as exc:
            logger.debug(f"[NewsFilter] Alert state save skipped: {exc}")

    def clear_halt(self) -> None:
        """Manually clear halt (e.g. after news is confirmed benign)."""
        self._halt_until    = None
        self._caution_until = None
        logger.info("[NewsFilter] Halt/caution cleared manually")


# ── Singleton ─────────────────────────────────────────────────────────────────
_news_filter: NewsFilter | None = None

def get_news_filter() -> NewsFilter:
    global _news_filter
    if _news_filter is None:
        _news_filter = NewsFilter()
    return _news_filter

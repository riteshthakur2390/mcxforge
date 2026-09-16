"""
agents_code/agent1_data/fetcher.py  Data Fetcher Agent
=========================================================
Owns ALL market data ingestion. No other agent touches Kite API directly.

Timeline:
  09:00   pre-market bias (VIX + gap %)
  09:15   candle loop starts (timeframe-aligned fetches)
  09:30   ORB formed and published
  09:30+  CANDLES_READY every configured interval until 15:35

agents_code/agent1_data/fetcher.py  Data Fetcher Agent
=========================================================
Now uses broker abstraction  works with ANY broker.
Set BROKER=yfinance for testing without any account.
Set BROKER=kite/upstox/groww for live production.

=========================================================
Now uses broker abstraction  works with ANY broker.
Set BROKER in .env  no code changes needed anywhere.
Supports: yfinance (free), upstox, groww, kite  all via BROKER= in .env.
"""
from __future__ import annotations

import asyncio
import os
import json
import time
import pytz
import pandas as pd
from datetime import date, datetime, timedelta
from pathlib import Path
from loguru import logger
import re

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bus import get_bus, Topic
from config.settings import (
    LIVE_TIMEFRAME, CANDLE_LOOKBACK, DATA_CACHE_DIR,
    MARKET_OPEN_TIME, ORB_END_TIME, DASHBOARD_TICK_INTERVAL_SEC,
    HISTORICAL_ARCHIVE_INTERVALS, JOURNAL_DIR,
)
from broker.factory import get_broker
from agents_code.agent1_data.orb import compute_orb
from utils.cache_manager import ensure_backtest_cache
from data.oi_recorder import OIRecorder
from data.historical_store import HistoricalCandleStore
from data.option_volume import OptionVolumeRecorder
from utils.market_calendar import is_trading_day, latest_expected_trading_day

IST = pytz.timezone("Asia/Kolkata")
OPTION_SYMBOL_RE = re.compile(r"^([A-Z]+)(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$")


class DataFetcherAgent:
    NAME = "DataFetcherAgent"
    _symbol: str = os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM")).upper()

    def __init__(self) -> None:
        self.bus = get_bus()
        self.broker = get_broker()
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        self._running = False
        self._orb_high: float | None = None
        self._orb_low: float | None = None
        self._orb_published = False
        self._last_published_candle_ts: datetime | None = None
        self._session_date = datetime.now(IST).date()
        self._latest_ltp: float = 0.0
        self._latest_aux_ltps: dict[str, float] = {}
        self._last_quote_heartbeat: float = 0.0
        self._last_oi_bucket: datetime | None = None
        self._last_df: pd.DataFrame | None = None
        self._premarket_published = False
        self._premarket_gap_ready = False
        self._historical_store = HistoricalCandleStore()
        self._orb_dir = Path(JOURNAL_DIR) / "orb"
        self._orb_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{self.NAME}] Initialized | broker={self.broker.broker_name.upper()}")
        self._symbol = os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM")).upper()
        self._oi_recorder = OIRecorder(self.broker)
        self._option_volume_recorder = OptionVolumeRecorder(
            self.broker,
            self._historical_store,
            cache_dir=DATA_CACHE_DIR,
            symbol=self._symbol,
        )

    async def start(self) -> None:
        logger.info(f"[{self.NAME}] Starting...")
        await self._publish_premarket_bias()
        await self._restore_orb_state_if_available(datetime.now(IST))
        
        # Ensure fresh history for backtesting/archival immediately on boot
        asyncio.create_task(asyncio.to_thread(ensure_backtest_cache, (self._symbol,)))
        
        self._running = True
        await asyncio.gather(
            self._cache_maintenance_loop(),
            self._quote_loop(),
            self._candle_loop(),
        )

    def stop(self) -> None:
        self._running = False

    def get_option_ltp(self, option_symbol: str) -> float:
        val = 0.0
        try:
            val = float(self.broker.get_option_ltp(option_symbol) or 0.0)
        except Exception:
            val = 0.0
        if val > 0:
            return val
        return self.get_option_candle_close(option_symbol, datetime.now(IST), interval="5minute")

    def get_option_candle_close(self, option_symbol: str, timestamp, interval: str = "1minute") -> float:
        match = OPTION_SYMBOL_RE.match(str(option_symbol or "").upper().strip())
        if not match:
            return 0.0
        symbol, yy, mon, dd, strike, option_type = match.groups()
        try:
            expiry = date(int(f"20{yy}"), datetime.strptime(mon, "%b").month, int(dd)).isoformat()
            candle_ts = pd.Timestamp(timestamp)
            if candle_ts.tzinfo is None:
                candle_ts = candle_ts.tz_localize(IST)
            else:
                candle_ts = candle_ts.tz_convert(IST)
            for active_interval, window_minutes in ((interval, 1), ("5minute", 5)):
                end_ts = candle_ts + pd.Timedelta(minutes=window_minutes)
                df = self._historical_store.load_option_candles(
                    symbol=symbol,
                    interval=active_interval,
                    broker=getattr(self.broker, "broker_name", None),
                    start=candle_ts.isoformat(),
                    end=end_ts.isoformat(),
                )
                if df.empty:
                    continue
                rows = df[
                    (pd.to_numeric(df["strike"], errors="coerce") == int(strike))
                    & (df["expiry"].astype(str) == expiry)
                    & (df["option_type"].astype(str).str.upper() == option_type)
                ]
                if rows.empty:
                    continue
                close = float(pd.to_numeric(rows.iloc[-1].get("close"), errors="coerce") or 0.0)
                if close > 0:
                    return round(close, 2)
            return 0.0
        except Exception as exc:
            logger.debug(f"[{self.NAME}] option candle close lookup failed for {option_symbol}: {exc}")
            return 0.0

    def get_ltp(self) -> float:
        return self._latest_ltp

    def get_nifty_ltp(self) -> float:
        return self._latest_ltp

    def get_latest_ltps(self) -> dict[str, float]:
        ltps = dict(self._latest_aux_ltps)
        if self._latest_ltp > 0:
            ltps[self.symbol] = self._latest_ltp
        return ltps

    def get_oi_recorder(self) -> OIRecorder:
        return self._oi_recorder

    def get_recent_candles(self, n: int = 60) -> pd.DataFrame | None:
        if self._last_df is not None and not self._last_df.empty:
            return self._last_df.tail(n).copy()
        return self._fetch_candles(n)

    async def _publish_premarket_bias(self) -> None:
        try:
            vix = 14.0
            session = self._resolve_premarket_session_values()
            prev_close = session["prev_close"]
            today_open = session["today_open"]
            prev_close_source = session.get("prev_close_source", "")
            today_open_source = session.get("today_open_source", "")

            if prev_close and today_open:
                gap_pct = (today_open - prev_close) / prev_close * 100
                bias = self._classify_gap_bias(gap_pct)
            else:
                bias, gap_pct = "NEUTRAL", 0.0

            payload = {
                "symbol": self._symbol,
                "bias": bias, "india_vix": vix,
                "gap_pct": round(gap_pct, 3),
                "nifty_prev_close": prev_close or 0,
                "prev_close": prev_close or 0,
                "today_open": today_open or 0,
                "gap_ready": bool(prev_close and today_open),
                "prev_close_source": prev_close_source,
                "today_open_source": today_open_source,
                "broker": self.broker.broker_name,
                "timestamp": datetime.now(IST).isoformat(),
            }
            await self.bus.publish(Topic.PREMARKET_BIAS, payload, self.NAME)
            self._premarket_published = True
            self._premarket_gap_ready = bool(prev_close and today_open)
            logger.info(
                f"[{self.NAME}] Pre-market: bias={bias} | VIX={vix:.1f} | "
                f"prev_close={float(prev_close or 0):.2f} | "
                f"today_open={float(today_open or 0):.2f} | "
                f"gap={gap_pct:+.2f}% | ready={bool(prev_close and today_open)} | "
                f"prev_close_source={prev_close_source or 'n/a'} | "
                f"today_open_source={today_open_source or 'n/a'}"
            )
        except Exception as e:
            logger.warning(f"[{self.NAME}] Pre-market error: {e}")
            # RC3 FIX: fallback to VIX=16 (safe value well below threshold 30)
            # Was 14.0 before  both are fine but making it explicit
            logger.warning(f"[{self.NAME}] Pre-market fetch failed  using safe defaults (VIX=16)")
            await self.bus.publish(Topic.PREMARKET_BIAS, {
                "bias": "NEUTRAL", "india_vix": 16.0, "gap_pct": 0.0,
                "prev_close": 0.0, "today_open": 0.0, "gap_ready": False,
                "broker": self.broker.broker_name,
                "timestamp": datetime.now(IST).isoformat(),
            }, self.NAME)

    async def _candle_loop(self) -> None:
        logger.info(f"[{self.NAME}] Candle loop started (ultra-low latency fast-polling enabled).")
        try:
            interval_minutes = int(str(LIVE_TIMEFRAME).replace("minute", ""))
        except Exception:
            interval_minutes = 5
        interval_minutes = max(interval_minutes, 1)

        while self._running:
            now = datetime.now(IST)
            loop_started = time.time()
            self._reset_session_state_if_needed(now)
            now_str = now.strftime("%H:%M")
            published_new_candle = False

            if MARKET_OPEN_TIME <= now_str <= "23:45" and is_trading_day(now.date()):
                try:
                    df = self._fetch_candles(CANDLE_LOOKBACK)
                    if df is not None and len(df) >= 5:
                        self._last_df = df.copy()
                        fresh, fresh_reason = self._candles_are_fresh(df, now)
                        if not fresh:
                            logger.debug(
                                f"[{self.NAME}] Waiting for fresh candle | reason={fresh_reason}"
                            )
                        else:
                            if (not self._premarket_gap_ready) and self._premarket_gap_ready_from_df(df):
                                await self._publish_premarket_bias()
                            candle_close = float(df["close"].iloc[-1]) if len(df) else 0.0
                            latest_candle_ts = df.index[-1]
                            if latest_candle_ts.tz is None:
                                latest_candle_ts = IST.localize(latest_candle_ts.to_pydatetime())
                            else:
                                latest_candle_ts = latest_candle_ts.tz_convert(IST)

                            if self._last_published_candle_ts != latest_candle_ts:
                                ltp = self._latest_ltp if self._latest_ltp > 0 else candle_close
                                df_for_publish = self._enrich_with_option_volume(df)
                                opt_total_latest = 0.0
                                if "opt_total_volume" in df_for_publish.columns and len(df_for_publish):
                                    opt_total_latest = float(
                                        pd.to_numeric(df_for_publish["opt_total_volume"], errors="coerce")
                                        .fillna(0.0)
                                        .iloc[-1]
                                    )

                                vix_live = 0.0
                                inst_name = self._symbol
                                gold_ltp = self._best_effort_quote("GOLD")
                                crude_ltp = self._best_effort_quote("CRUDEOIL")
                                natgas_ltp = self._best_effort_quote("NATURALGAS")
                                ltps = {
                                    inst_name: ltp,
                                    "SILVERM": ltp,
                                    "SILVERMIC": ltp,
                                    "GOLD": gold_ltp,
                                    "CRUDEOIL": crude_ltp,
                                    "NATURALGAS": natgas_ltp,
                                    "NATGAS": natgas_ltp,
                                }

                                if now_str >= ORB_END_TIME and not self._orb_published:
                                    restored_orb = await self._restore_orb_state_if_available(now)
                                    if restored_orb:
                                        logger.info(
                                            f"[{self.NAME}] ORB restored from disk | "
                                            f"H={self._orb_high} L={self._orb_low}"
                                        )
                                    else:
                                        self._orb_high, self._orb_low = compute_orb(
                                            df,
                                            orb_start=MARKET_OPEN_TIME,
                                            orb_end=ORB_END_TIME,
                                        )
                                        if self._orb_high and self._orb_low:
                                            orb_payload = {
                                                "symbol": self._symbol,
                                                "orb_high": self._orb_high,
                                                "orb_low": self._orb_low,
                                                "orb_range": round(self._orb_high - self._orb_low, 2),
                                                "session_date": latest_candle_ts.date().isoformat(),
                                                "formed_at": latest_candle_ts.isoformat(),
                                            }
                                            self._persist_orb_state(orb_payload)
                                            await self.bus.publish(Topic.ORB_FORMED, orb_payload, self.NAME)
                                            self._orb_published = True

                                candles_list = [
                                    self._candle_payload_row(idx, r)
                                    for idx, r in df_for_publish.iterrows()
                                ]
                                await self.bus.publish(Topic.CANDLES_READY, {
                                    "symbol": inst_name, "timeframe": LIVE_TIMEFRAME,
                                    "candles": candles_list, "ltp": ltp,
                                    "ltps": ltps,
                                    "option_volume": {},
                                    "vix": vix_live,
                                    "orb_high": self._orb_high, "orb_low": self._orb_low,
                                    "broker": self.broker.broker_name,
                                    "timestamp": latest_candle_ts.isoformat(),
                                    "server_timestamp": now.isoformat(),
                                }, self.NAME)
                                self._last_published_candle_ts = latest_candle_ts
                                published_new_candle = True
                                logger.info(
                                    f"[{self.NAME}] CANDLES_READY | "
                                    f"market_ts={latest_candle_ts.strftime('%H:%M:%S')} | "
                                    f"LTP={ltp:.2f} | close={candle_close:.2f} | "
                                    f"opt_vol={opt_total_latest:.0f}"
                                )
                                # Non-blocking background persistence & enrichment
                                asyncio.create_task(self._async_background_enrichment(latest_candle_ts, ltp, df))
                except Exception as e:
                    logger.error(f"[{self.NAME}] Candle loop error: {e}")
            elif now_str > "23:45":
                await asyncio.sleep(60)
                continue

            loop_elapsed = time.time() - loop_started
            if loop_elapsed >= 15.0:
                logger.warning(
                    f"[{self.NAME}] Candle loop slow cycle | "
                    f"elapsed={loop_elapsed:.1f}s | now={now.strftime('%Y-%m-%d %H:%M:%S IST')}"
                )

            # Ultra-low latency scheduling:
            # 1. If we just published a fresh candle, sleep until the next boundary.
            # 2. If at boundary (:00, :05, :10...) and waiting for broker candle, poll every 1s (up to 45s).
            # 3. Otherwise sleep until near the next boundary.
            if published_new_candle:
                sleep_secs = self._secs_to_next_candle(now)
            else:
                is_boundary_window = (now.minute % interval_minutes == 0) and (now.second < 45)
                if is_boundary_window:
                    sleep_secs = 1.0  # Fast retry until broker releases closed candle
                else:
                    sleep_secs = min(5.0, self._secs_to_next_candle(now))

            await asyncio.sleep(sleep_secs)

    async def _async_background_enrichment(self, candle_ts: datetime, underlying_ltp: float, df: pd.DataFrame) -> None:
        """Run non-critical persistence and option snapshots in background without blocking candle loop."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._record_oi_if_due, candle_ts)
            await loop.run_in_executor(None, self._persist_recent_history, df)
            await loop.run_in_executor(None, self._record_option_volume_snapshot, candle_ts, underlying_ltp)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Background enrichment skipped: {exc}")

    def _record_oi_if_due(self, candle_ts: datetime) -> None:
        if self._latest_ltp <= 0:
            return
        bucket = candle_ts.replace(
            minute=(candle_ts.minute // 5) * 5,
            second=0,
            microsecond=0,
        )
        if self._last_oi_bucket == bucket:
            return
        self._last_oi_bucket = bucket
        snapshot = self._oi_recorder.record(
            nifty_ltp=self._latest_ltp,
            timestamp=bucket,
        )
        if snapshot:
            logger.debug(
                f"[{self.NAME}] OI snapshot recorded | "
                f"ts={bucket.strftime('%Y-%m-%d %H:%M IST')} | "
                f"ATM={snapshot.get('atm_strike')} | PCR={snapshot.get('pcr', 0):.3f}"
            )

    def _record_option_volume_snapshot(self, candle_ts: datetime, underlying_ltp: float) -> dict[str, float]:
        if underlying_ltp <= 0:
            return {}
        try:
            snapshot = self._option_volume_recorder.record_snapshot(
                underlying_spot=underlying_ltp,
                timestamp=candle_ts,
            )
            if snapshot:
                logger.debug(
                    f"[{self.NAME}] Option volume snapshot recorded | "
                    f"ts={candle_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"total={float(snapshot.get('opt_total_volume', 0.0) or 0.0):.0f} | "
                    f"atm={float(snapshot.get('opt_atm_volume', 0.0) or 0.0):.0f}"
                )
            else:
                logger.debug(
                    f"[{self.NAME}] Option volume snapshot empty | "
                    f"ts={candle_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"underlying={underlying_ltp:.2f}"
                )
            return snapshot
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Option volume snapshot skipped: {exc}")
            return {}

    def _enrich_with_option_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            return self._option_volume_recorder.enrich_candles(
                df,
                interval=LIVE_TIMEFRAME,
                broker=self.broker.broker_name,
            )
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Option volume enrichment skipped: {exc}")
            return df

    @staticmethod
    def _candle_payload_row(idx, row: pd.Series) -> dict:
        payload = {
            "datetime": str(idx),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row.get("volume", 0) or 0),
        }
        for col, value in row.items():
            if col in payload or col in {"open", "high", "low", "close", "volume"}:
                continue
            if pd.isna(value):
                continue
            if isinstance(value, (int, float)):
                payload[str(col)] = float(value)
            else:
                try:
                    payload[str(col)] = float(value)
                except Exception:
                    payload[str(col)] = value
        return payload

    @staticmethod
    def _resample_intraday_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        normalized = df.copy()
        idx = pd.to_datetime(normalized.index)
        if idx.tz is None:
            idx = idx.tz_localize(IST)
        else:
            idx = idx.tz_convert(IST)
        normalized.index = idx

        agg: dict[str, str] = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        for col in normalized.columns:
            if col not in agg:
                agg[col] = "last"
        out = normalized.resample(
            rule,
            origin="start_day",
            offset="15min",
            label="right",
            closed="right",
        ).agg(agg)
        return out.dropna(subset=["open", "high", "low", "close"]).sort_index()

    def _persist_recent_history(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        if LIVE_TIMEFRAME not in HISTORICAL_ARCHIVE_INTERVALS:
            return
        try:
            # Keep recent candles synced into the local long-horizon store.
            self._historical_store.upsert_candles(
                df.tail(3),
                symbol=self.symbol,
                interval=LIVE_TIMEFRAME,
                broker=self.broker.broker_name,
                source="live_fetcher",
            )
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Local history persist skipped: {exc}")

    def _candles_are_fresh(self, df: pd.DataFrame, now: datetime) -> tuple[bool, str]:
        if df is None or df.empty:
            return False, "empty_candles"
        latest_ts = df.index[-1]
        if latest_ts.tzinfo is None:
            latest_ts = IST.localize(latest_ts.to_pydatetime())
        else:
            latest_ts = latest_ts.tz_convert(IST)
        expected_day = latest_expected_trading_day(now.date())
        if latest_ts.date() != expected_day:
            return False, f"stale_candle_day={latest_ts.date()} expected={expected_day}"
        timeframe_minutes = {
            "1minute": 1,
            "5minute": 5,
            "15minute": 15,
            "30minute": 30,
            "day": 24 * 60,
        }.get(LIVE_TIMEFRAME, 5)
        max_age_minutes = max(timeframe_minutes * 3, 15)
        age_minutes = max(int((now - latest_ts).total_seconds() // 60), 0)
        if age_minutes > max_age_minutes:
            return False, f"stale_candle_age={age_minutes}m max={max_age_minutes}m"
        return True, ""

    async def _quote_loop(self) -> None:
        logger.info(
            f"[{self.NAME}] Quote loop started | interval={DASHBOARD_TICK_INTERVAL_SEC}s"
        )
        while self._running:
            now = datetime.now(IST)
            now_str = now.strftime("%H:%M")
            if MARKET_OPEN_TIME <= now_str <= "23:30" and is_trading_day(now.date()):
                try:
                    payload = self._tick_payload(now=now)
                    await self.bus.publish(
                        Topic.TICK_UPDATE,
                        payload,
                        self.NAME,
                    )
                except Exception as e:
                    logger.error(f"[{self.NAME}] Quote loop error: {e}")
            await asyncio.sleep(max(DASHBOARD_TICK_INTERVAL_SEC, 5))

    def get_orb_state(self, for_date=None) -> dict | None:
        target_date = for_date or self._session_date
        if (
            self._orb_high is not None
            and self._orb_low is not None
            and target_date == self._session_date
        ):
            return {
                "orb_high": self._orb_high,
                "orb_low": self._orb_low,
                "orb_range": round(self._orb_high - self._orb_low, 2),
                "session_date": self._session_date.isoformat(),
            }
        
        state = self._load_orb_state(target_date)
        if state and target_date == self._session_date:
            try:
                if self._orb_high is None:
                    self._orb_high = float(state.get("orb_high", 0) or 0)
                    self._orb_low = float(state.get("orb_low", 0) or 0)
                    if self._orb_high > 0:
                        self._orb_published = True
            except Exception:
                pass
        return state

    async def _cache_maintenance_loop(self) -> None:
        logger.info(f"[{self.NAME}] Cache maintenance loop started.")
        while self._running:
            try:
                await asyncio.to_thread(ensure_backtest_cache)
            except Exception as e:
                logger.warning(f"[{self.NAME}] Cache maintenance error: {e}")
            await asyncio.sleep(60 * 60 * 6)

    def _fetch_candles(self, n: int = 200) -> pd.DataFrame | None:
        try:
            end = datetime.now(IST)
            buffer_days = max(5, int(n / 60) + 2)
            start = end - timedelta(days=buffer_days)
            requested_interval = LIVE_TIMEFRAME
            source_interval = "1minute" if requested_interval == "3minute" else requested_interval
            df = self.broker.get_historical_data(
                symbol=self._symbol, interval=source_interval,
                from_date=start.strftime("%Y-%m-%d"),
                to_date=end.strftime("%Y-%m-%d"),
            )
            if df is None or df.empty:
                return None
            df = df.sort_index()
            if requested_interval == "3minute":
                df = self._resample_intraday_ohlcv(df, "3min")
            return df.tail(n)
        except Exception as e:
            logger.error(f"[{self.NAME}] _fetch_candles: {e}")
            return None

    def _fetch_daily_candles(self, days: int = 30) -> pd.DataFrame | None:
        try:
            end = datetime.now(IST)
            start = end - timedelta(days=max(days, 10))
            df = self.broker.get_historical_data(
                symbol=self._symbol,
                interval="day",
                from_date=start.strftime("%Y-%m-%d"),
                to_date=end.strftime("%Y-%m-%d"),
            )
            if df is None or df.empty:
                return None
            return df.sort_index()
        except Exception as e:
            logger.error(f"[{self.NAME}] _fetch_daily_candles: {e}")
            return None

    def fetch_historical(self, weeks: int = 100, interval: str = "5minute", force: bool = False) -> pd.DataFrame:
        cache = Path(DATA_CACHE_DIR) / f"{self._symbol}_{interval}_{self.broker.broker_name}.parquet"
        if cache.exists() and not force:
            return pd.read_parquet(cache)
        end = datetime.now(IST)
        start = end - timedelta(weeks=weeks)
        source_interval = "1minute" if interval == "3minute" else interval
        df = self.broker.get_historical_data(
            self._symbol, source_interval,
            start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            raise RuntimeError(f"No data via {self.broker.broker_name}")
        if interval == "3minute":
            df = self._resample_intraday_ohlcv(df, "3min")
        df.to_parquet(cache)
        logger.success(f"[{self.NAME}] Saved {len(df)} rows  {cache}")
        return df

    @staticmethod
    def _index_dates(df: pd.DataFrame) -> pd.Series:
        index = pd.DatetimeIndex(df.index)
        if index.tz is None:
            index = index.tz_localize(IST)
        else:
            index = index.tz_convert(IST)
        return pd.Series(index.date, index=index)

    def _get_prev_close_from_cache(self, as_of_date=None) -> float | None:
        as_of_date = as_of_date or datetime.now(IST).date()
        try:
            candidates = [
                f"{self._symbol}_day_{self.broker.broker_name}.parquet",
                *[str(p.name) for p in Path(DATA_CACHE_DIR).glob(f"{self._symbol}_day_*.parquet")],
                *[str(p.name) for p in Path(DATA_CACHE_DIR).glob("*_day_*.parquet")],
            ]
            seen = set()
            for fname in candidates:
                if fname in seen:
                    continue
                seen.add(fname)
                p = Path(DATA_CACHE_DIR) / fname
                if p.exists():
                    df = pd.read_parquet(p).sort_index()
                    if df.empty:
                        continue
                    index_dates = self._index_dates(df)
                    prior_df = df[index_dates.to_numpy() < as_of_date]
                    if not prior_df.empty:
                        return float(prior_df["close"].iloc[-1])
        except Exception:
            pass
        return None

    def _resolve_premarket_session_values(self) -> dict:
        as_of_date = datetime.now(IST).date()
        prev_close = None
        today_open = None
        prev_close_source = ""
        today_open_source = ""
        df = self._fetch_candles(max(CANDLE_LOOKBACK, 120))
        if df is not None and not df.empty:
            index_dates = self._index_dates(df)
            today_df = df[index_dates.to_numpy() == as_of_date]
            if not today_df.empty:
                today_open = float(today_df["open"].iloc[0])
                today_open_source = f"{self.broker.broker_name}:intraday_current_session"

            prior_df = df[index_dates.to_numpy() < as_of_date]
            if not prior_df.empty:
                prev_close = float(prior_df["close"].iloc[-1])
                prev_close_source = f"{self.broker.broker_name}:intraday_prior_session"

        daily_df = self._fetch_daily_candles(30)
        if daily_df is not None and not daily_df.empty:
            daily_dates = self._index_dates(daily_df)
            prior_daily = daily_df[daily_dates.to_numpy() < as_of_date]
            if not prior_daily.empty:
                prev_close = float(prior_daily["close"].iloc[-1])
                prev_close_source = f"{self.broker.broker_name}:day_prior_close"

            today_daily = daily_df[daily_dates.to_numpy() == as_of_date]
            if today_open is None and not today_daily.empty:
                today_open = float(today_daily["open"].iloc[0])
                today_open_source = f"{self.broker.broker_name}:day_current_open"

        if not prev_close:
            prev_close = self._get_prev_close_from_cache(as_of_date=as_of_date)
            if prev_close:
                prev_close_source = "cache:day_prior_close"
        return {
            "prev_close": prev_close,
            "today_open": today_open,
            "prev_close_source": prev_close_source,
            "today_open_source": today_open_source,
        }

    def _reset_session_state_if_needed(self, now: datetime) -> None:
        session_date = now.date()
        if session_date == self._session_date:
            return
        self._session_date = session_date
        self._orb_high = None
        self._orb_low = None
        self._orb_published = False
        self._last_published_candle_ts = None
        self._last_oi_bucket = None
        self._premarket_published = False
        self._premarket_gap_ready = False

    def _orb_state_path(self, session_date) -> Path:
        if hasattr(session_date, "isoformat"):
            session_date = session_date.isoformat()
        sym = getattr(self, "_symbol", "")
        if sym:
            sym_path = self._orb_dir / f"orb_{sym}_{session_date}.json"
            legacy_path = self._orb_dir / f"orb_{session_date}.json"
            if legacy_path.exists() and not sym_path.exists():
                return legacy_path
            return sym_path
        return self._orb_dir / f"orb_{session_date}.json"

    def _persist_orb_state(self, payload: dict) -> None:
        session_date = payload.get("session_date") or self._session_date.isoformat()
        path = self._orb_state_path(session_date)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception as exc:
            logger.warning(f"[{self.NAME}] ORB persist failed: {exc}")

    def _load_orb_state(self, session_date) -> dict | None:
        path = self._orb_state_path(session_date)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except Exception as exc:
            logger.warning(f"[{self.NAME}] ORB load failed: {exc}")
            return None

    async def _restore_orb_state_if_available(self, now: datetime) -> bool:
        if self._orb_published or now.strftime("%H:%M") < ORB_END_TIME:
            return False
        orb_state = self._load_orb_state(self._session_date)
        if not orb_state:
            return False
        cached_sym = str(orb_state.get("symbol") or "")
        if cached_sym and cached_sym != self._symbol:
            logger.info(f"[{self.NAME}] Discarding stale ORB for {cached_sym} != {self._symbol}")
            return False
        try:
            orb_high = float(orb_state.get("orb_high", 0.0) or 0.0)
            orb_low = float(orb_state.get("orb_low", 0.0) or 0.0)
        except Exception:
            return False
        if orb_high <= 0 or orb_low <= 0:
            return False

        self._orb_high = orb_high
        self._orb_low = orb_low
        self._orb_published = True
        payload = {
            "orb_high": orb_high,
            "orb_low": orb_low,
            "orb_range": float(orb_state.get("orb_range", round(orb_high - orb_low, 2)) or 0.0),
            "session_date": str(orb_state.get("session_date") or self._session_date.isoformat()),
            "formed_at": str(orb_state.get("formed_at") or now.isoformat()),
        }
        if "symbol" in orb_state:
            payload["symbol"] = orb_state["symbol"]
        await self.bus.publish(Topic.ORB_FORMED, payload, self.NAME)
        return True

    @staticmethod
    def _classify_gap_bias(gap_pct: float) -> str:
        if gap_pct > 0.15:
            return "BULLISH"
        if gap_pct < -0.15:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _premarket_gap_ready_from_df(df: pd.DataFrame | None) -> bool:
        if df is None or df.empty:
            return False
        session_dates = sorted({idx.date() for idx in df.index})
        if len(session_dates) < 2:
            return False
        latest_day = session_dates[-1]
        latest_df = df[[idx.date() == latest_day for idx in df.index]]
        return not latest_df.empty

    def _best_effort_quote(self, symbol: str, default: float = 0.0) -> float:
        if symbol in ("INDIA VIX", "VIX"):
            return 0.0

        try:
            latest = float(self.broker.get_ltp(symbol) or 0.0)
        except Exception:
            latest = 0.0
        if latest > 0:
            self._latest_aux_ltps[symbol] = latest
            return latest
        if symbol in self._latest_aux_ltps:
            return self._latest_aux_ltps[symbol]
        return default

    def _tick_payload(self, now: datetime | None = None) -> dict:
        now = now or datetime.now(IST)
        needed_symbols = list({self._symbol, "SILVERM", "GOLD", "CRUDEOIL", "NATURALGAS"})
        if hasattr(self.broker, "get_quotes"):
            try:
                batch_quotes = self.broker.get_quotes(needed_symbols)
                for s, p in batch_quotes.items():
                    if p > 0:
                        self._latest_aux_ltps[s] = p
            except Exception:
                pass

        inst_ltp = self._best_effort_quote(self._symbol, default=self._latest_ltp)
        if inst_ltp > 0:
            self._latest_ltp = inst_ltp
        silver_ltp = inst_ltp if "SILVER" in self._symbol else self._best_effort_quote("SILVERM", default=self._latest_ltp)
        gold_ltp = self._best_effort_quote("GOLD")
        crude_ltp = self._best_effort_quote("CRUDEOIL")
        natgas_ltp = self._best_effort_quote("NATURALGAS")
        return {
            "symbol": self._symbol,
            "ltp": self._latest_ltp,
            "ltps": {
                self._symbol: self._latest_ltp,
                "SILVERM": silver_ltp,
                "SILVERMIC": silver_ltp,
                "GOLD": gold_ltp,
                "CRUDEOIL": crude_ltp,
                "NATURALGAS": natgas_ltp,
                "NATGAS": natgas_ltp,
            },
            "vix": 0.0,
            "broker": self.broker.broker_name,
            "timestamp": now.isoformat(),
            "server_timestamp": now.isoformat(),
        }

    @staticmethod
    def _secs_to_next_candle(now: datetime) -> float:
        try:
            interval_minutes = int(str(LIVE_TIMEFRAME).replace("minute", ""))
        except Exception:
            interval_minutes = 5
        interval_minutes = max(interval_minutes, 1)
        rem = interval_minutes - (now.minute % interval_minutes)
        if rem == 0:
            rem = interval_minutes
        # Target second 1 past the next boundary so broker has formed the candle
        target = (now + timedelta(minutes=rem)).replace(second=1, microsecond=0)
        diff = (target - now).total_seconds()
        return max(diff, 1.0)

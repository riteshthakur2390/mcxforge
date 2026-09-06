"""
Cache maintenance helpers for backtesting and ML training.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading

import pandas as pd
import pytz
from loguru import logger

from broker.factory import get_active_broker_name, get_broker
from config.settings import (
    DATA_CACHE_DIR,
    RECENT_MARKET_WINDOW_DAYS,
    HISTORICAL_ARCHIVE_INTERVALS,
    SUPPORTED_INDEX_SYMBOLS,
)
from utils.market_calendar import latest_expected_trading_day
from data.historical_store import HistoricalCandleStore

IST = pytz.timezone("Asia/Kolkata")
INTRADAY_INTERVALS = ("1minute", "3minute", "5minute", "15minute", "30minute")
_CACHE_REFRESH_LOCK = threading.Lock()
_CACHE_REFRESH_IN_PROGRESS: set[tuple[str, str, str]] = set()
_BACKTEST_CACHE_LOCK = threading.Lock()


def _cache_symbol(symbol: str | None) -> str:
    key = str(symbol or "NIFTY").upper().replace(" ", "")
    if key in {"NIFTY50", "NIFTY_50"}:
        return "NIFTY"
    return key


def interval_capability_days(broker_name: str, interval: str) -> int:
    broker_name = str(broker_name or "").lower().strip()
    interval = str(interval or "").strip()

    if broker_name == "dhan":
        if interval in INTRADAY_INTERVALS:
            return 31
        if interval == "day":
            return 365
        return 90
    if broker_name == "upstox":
        if interval in {"1minute", "3minute", "5minute", "15minute"}:
            # Upstox V3 historical docs: minute intervals 1-15 are limited to ~1 month.
            return 31
        if interval == "30minute":
            # Upstox V3 historical docs: minute intervals >15 can reach one quarter.
            return 90
    if broker_name == "yfinance":
        if interval in {"1minute", "3minute"}:
            return 7
        if interval in {"5minute", "15minute", "30minute"}:
            return 60
    if interval in {"1minute", "3minute"}:
        return 60
    if interval in {"5minute", "15minute", "30minute"}:
        return 60
    if interval in {"60minute", "day"}:
        return 3650
    return 90


def recent_source_window_days(broker_name: str, interval: str) -> int:
    capability = interval_capability_days(broker_name, interval)
    # Use a safety margin: archive anything older than 20 days or half the capability, 
    # whichever is smaller, to ensure overlap between broker cache and local DB.
    margin_days = min(20, max(1, capability // 2))
    return max(1, min(int(RECENT_MARKET_WINDOW_DAYS), margin_days))


def inspect_cache(interval: str, broker_name: str | None = None, symbol: str = "NIFTY") -> dict | None:
    broker_name = broker_name or get_active_broker_name()
    symbol_key = _cache_symbol(symbol)
    cache_path = Path(DATA_CACHE_DIR) / f"{symbol_key}_{interval}_{broker_name}.parquet"
    if not cache_path.exists():
        return None

    df = pd.read_parquet(cache_path)
    if df.empty:
        return None

    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)

    earliest = idx.min().date()
    latest = idx.max().date()
    return {
        "path": cache_path,
        "earliest_date": earliest,
        "latest_date": latest,
        "span_days": (latest - earliest).days + 1,
        "rows": len(df),
        "mtime": datetime.fromtimestamp(cache_path.stat().st_mtime, tz=IST),
    }


def ensure_cache(interval: str, force: bool = False, symbol: str = "NIFTY") -> dict | None:
    broker = get_broker()
    broker_name = str(getattr(broker, "broker_name", get_active_broker_name()) or "").lower().strip()
    symbol_key = _cache_symbol(symbol)
    refresh_key = (broker_name, symbol_key, str(interval))
    capability_days = interval_capability_days(broker_name, interval)
    expected_latest = latest_expected_trading_day(datetime.now(IST).date())
    info = inspect_cache(interval, broker_name=broker_name, symbol=symbol_key)

    stale = info is None or info["latest_date"] < expected_latest
    short_span = info is None or info["span_days"] < max(2, capability_days - 2)
    refresh_needed = force or stale or short_span

    if not refresh_needed:
        return info

    if not hasattr(broker, "download_and_cache"):
        return info

    with _CACHE_REFRESH_LOCK:
        if refresh_key in _CACHE_REFRESH_IN_PROGRESS:
            logger.debug(
                f"[cache] Refresh already running for {symbol_key} {interval} "
                f"| broker={broker_name}"
            )
            return info
        _CACHE_REFRESH_IN_PROGRESS.add(refresh_key)

    try:
        logger.info(
            f"[cache] Refreshing {symbol_key} {interval} cache | "
            f"broker={broker_name} | stale={stale} | short_span={short_span}"
        )
        broker.download_and_cache(symbol=symbol_key, interval=interval)
        return inspect_cache(interval, broker_name=broker_name, symbol=symbol_key)
    except Exception as exc:
        logger.warning(f"[cache] Refresh failed for {symbol_key} {interval}: {exc}")
        return info
    finally:
        with _CACHE_REFRESH_LOCK:
            _CACHE_REFRESH_IN_PROGRESS.discard(refresh_key)


def ensure_backtest_cache(symbols: tuple[str, ...] | None = None) -> dict:
    if not _BACKTEST_CACHE_LOCK.acquire(blocking=False):
        logger.debug("[cache] Backtest cache refresh already running; skipping duplicate request")
        return {}
    try:
        return _ensure_backtest_cache(symbols)
    finally:
        _BACKTEST_CACHE_LOCK.release()


def _ensure_backtest_cache(symbols: tuple[str, ...] | None = None) -> dict:
    broker_name = get_active_broker_name()
    intervals = ["1minute", "5minute", "day"]
    symbols = tuple(_cache_symbol(s) for s in (symbols or SUPPORTED_INDEX_SYMBOLS or ("NIFTY",)))
    results: dict[str, dict | None] = {}
    for symbol in symbols:
        for interval in intervals:
            results[f"{symbol}:{interval}"] = ensure_cache(interval, symbol=symbol)

    try:
        store = HistoricalCandleStore()
        cache_root = Path(DATA_CACHE_DIR)
        for interval in HISTORICAL_ARCHIVE_INTERVALS:
            window_days = recent_source_window_days(broker_name, interval)
            for symbol in symbols:
                for path in sorted(cache_root.glob(f"{symbol}_{interval}_*.parquet")):
                    if path.name.startswith("instruments_"):
                        continue
                    store.import_cache_file(
                        path,
                        archive_only=True,
                        window_days=window_days,
                        archive_intervals=(interval,),
                    )
            store.prune_recent_candles(
                intervals=(interval,),
                window_days=window_days,
            )
    except Exception as exc:
        logger.debug(f"[cache] Local history backfill skipped: {exc}")
    return results

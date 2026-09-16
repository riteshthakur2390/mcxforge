"""
scripts/sync_commodity_historical_data.py — Multi-Commodity Historical Data Synchronization
===========================================================================================
Synchronizes 5-year daily and rolling intraday 5-minute OHLCV candles for:
  - SILVERM (and SILVERMIC)
  - GOLDM (and GOLD)
  - CRUDEOIL (and CRUDEOILM)
  - NATGAS (and NATGASM / NATURALGAS)

Market Protection Rule:
  By default, multi-commodity synchronization is BLOCKED during active MCX market hours
  (09:00 - 23:30 IST) to prevent consuming broker API quota or interfering with live
  SILVERM tracking. It is scheduled to run post-market close (EOD at 23:35 IST), or
  manually with --force.

Broker Abstraction:
  Uses the standard broker interface (broker.get_historical_data) via broker.factory.
  Fully compatible with both Dhan (default) and Upstox without code changes.
"""

import os
import sys
import time
import argparse
from datetime import datetime, date, timedelta, time as dt_time
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
import pytz
from loguru import logger

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from broker.factory import get_broker, BROKER_REGISTRY
from broker.base_broker import BaseBroker
from data.historical_store import HistoricalCandleStore

IST = pytz.timezone("Asia/Kolkata")

# Multi-commodity specification
COMMODITY_CONFIGS: List[Dict[str, Any]] = [
    {
        "symbol": "SILVERM",
        "aliases": ["SILVERMIC", "SILVER"],
        "name": "Silver Mini / Micro",
        "default_daily_start": "2021-01-01",
    },
    {
        "symbol": "GOLDM",
        "aliases": ["GOLD"],
        "name": "Gold Mini / Mega",
        "default_daily_start": "2021-01-01",
    },
    {
        "symbol": "CRUDEOIL",
        "aliases": ["CRUDEOILM"],
        "name": "Crude Oil / Mini",
        "default_daily_start": "2021-01-01",
    },
    {
        "symbol": "NATGAS",
        "aliases": ["NATGASM", "NATURALGAS", "NATGASMINI"],
        "name": "Natural Gas / Mini",
        "default_daily_start": "2021-01-01",
    },
]


def is_mcx_market_open(now_dt: Optional[datetime] = None) -> bool:
    """
    Checks if MCX is currently open for live trading.
    Standard MCX hours: 09:00 to 23:30 IST, Monday through Friday.
    """
    now = now_dt or datetime.now(IST)
    if now.weekday() >= 5:  # Saturday or Sunday
        return False
    market_open = dt_time(9, 0)
    market_close = dt_time(23, 30)
    return market_open <= now.time() <= market_close


def _merge_and_save_csv(
    df_new: pd.DataFrame,
    csv_paths: List[Path],
    symbol: str,
    interval_label: str,
    broker_name: str,
) -> pd.DataFrame:
    """
    Monotonically merges new candles with existing CSV, deduplicating on timestamp.
    Saves to all target paths.
    """
    if df_new.empty:
        # Load from primary if exists
        for p in csv_paths:
            if p.exists():
                return pd.read_csv(p)
        return pd.DataFrame()

    df_prepared = df_new.copy()
    if not isinstance(df_prepared.index, pd.DatetimeIndex):
        ts_cols = [c for c in df_prepared.columns if "time" in c.lower() or "date" in c.lower()]
        if ts_cols:
            df_prepared.index = pd.to_datetime(df_prepared[ts_cols[0]], format="mixed", utc=True)
    
    df_prepared.index.name = "timestamp"
    df_prepared = df_prepared.reset_index()
    df_prepared["symbol"] = symbol
    df_prepared["data_origin"] = f"{broker_name.upper()}_REAL_HISTORICAL"

    primary_path = csv_paths[0]
    if primary_path.exists():
        try:
            existing_df = pd.read_csv(primary_path)
            prev_cnt = len(existing_df)
            merged = pd.concat([existing_df, df_prepared], ignore_index=True)
            ts_col = [c for c in merged.columns if "time" in c.lower() or "date" in c.lower()][0]
            dt_series = pd.to_datetime(merged[ts_col], format="mixed", utc=True).dt.tz_convert("Asia/Kolkata")
            merged["_dt_sort"] = dt_series
            merged = merged.sort_values(by="_dt_sort").drop_duplicates(subset=["_dt_sort"], keep="last")
            merged[ts_col] = merged["_dt_sort"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
                r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True
            )
            merged = merged.drop(columns=["_dt_sort"])
            result_df = merged
            logger.info(
                f"[{symbol} {interval_label}] Monotonically accumulated: {prev_cnt} -> {len(result_df)} candles"
            )
        except Exception as e:
            logger.warning(f"Failed to merge existing CSV {primary_path}: {e}. Writing fresh.")
            result_df = df_prepared
    else:
        ts_col = [c for c in df_prepared.columns if "time" in c.lower() or "date" in c.lower()][0]
        dt_series = pd.to_datetime(df_prepared[ts_col], format="mixed", utc=True).dt.tz_convert("Asia/Kolkata")
        df_prepared["_dt_sort"] = dt_series
        df_prepared = df_prepared.sort_values(by="_dt_sort").drop_duplicates(subset=["_dt_sort"], keep="last")
        df_prepared[ts_col] = df_prepared["_dt_sort"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(
            r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True
        )
        df_prepared = df_prepared.drop(columns=["_dt_sort"])
        result_df = df_prepared
        logger.info(f"[{symbol} {interval_label}] Saved initial {len(result_df)} candles")

    # Enforce physical OHLC laws (High >= max(Open, Close) and Low <= min(Open, Close))
    if all(col in result_df.columns for col in ["open", "high", "low", "close"]):
        result_df["high"] = result_df[["high", "open", "close"]].max(axis=1)
        result_df["low"] = result_df[["low", "open", "close"]].min(axis=1)

    # Save to all target CSV paths
    for p in csv_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(p, index=False)

    return result_df


def sync_single_commodity(
    commodity: Dict[str, Any],
    broker: BaseBroker,
    broker_name: str,
    output_dir: Path,
    store: HistoricalCandleStore,
    intraday_days: int = 180,
    daily_start_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Downloads and stores daily and 5-minute historical candles for a single commodity.
    """
    symbol = commodity["symbol"]
    aliases = commodity.get("aliases", [])
    all_syms = [symbol] + [a for a in aliases if a != symbol]
    
    today = datetime.now(IST).date()
    today_str = today.strftime("%Y-%m-%d")
    daily_start = daily_start_date or commodity.get("default_daily_start", "2021-01-01")
    intraday_start = (today - timedelta(days=intraday_days)).strftime("%Y-%m-%d")

    logger.info(f"==> Syncing {commodity['name']} ({symbol}) via {broker_name.upper()}...")

    # 1. Fetch Daily Data
    logger.info(f"[{symbol}] Fetching daily candles: {daily_start} to {today_str}...")
    try:
        df_daily = broker.get_historical_data(symbol, "day", daily_start, today_str)
    except Exception as e:
        logger.error(f"[{symbol}] Failed to fetch daily data: {e}")
        df_daily = pd.DataFrame()

    time.sleep(0.6)  # Pacing to avoid 429 rate limit

    # Daily target paths
    daily_paths = [
        output_dir / f"{symbol}_{broker_name}_1d.csv",
        output_dir / f"{symbol}_1d.csv",
    ]
    # For SILVERM, also maintain SILVERMIC compatibility files
    if symbol in ("SILVERM", "SILVERMIC"):
        daily_paths.extend([
            output_dir / f"SILVERMIC_{broker_name}_1d.csv",
            output_dir / "SILVERMIC_dhan_1d.csv",
            output_dir / "SILVERMIC_1d.csv",
        ])

    df_daily_merged = _merge_and_save_csv(df_daily, daily_paths, symbol, "1d", broker_name)

    # 2. Fetch Intraday 5-Minute Data
    logger.info(f"[{symbol}] Fetching 5-minute intraday candles: {intraday_start} to {today_str}...")
    try:
        df_5m = broker.get_historical_data(symbol, "5minute", intraday_start, today_str)
    except Exception as e:
        logger.error(f"[{symbol}] Failed to fetch 5m data: {e}")
        df_5m = pd.DataFrame()

    time.sleep(0.6)  # Pacing

    # 5m target paths
    intraday_paths = [
        output_dir / f"{symbol}_{broker_name}_5m.csv",
        output_dir / f"{symbol}_5m.csv",
    ]
    if symbol in ("SILVERM", "SILVERMIC"):
        intraday_paths.extend([
            output_dir / f"SILVERMIC_{broker_name}_5m.csv",
            output_dir / "SILVERMIC_dhan_5m.csv",
            output_dir / "SILVERMIC_5m.csv",
        ])

    df_5m_merged = _merge_and_save_csv(df_5m, intraday_paths, symbol, "5m", broker_name)

    # 3. Synchronize to SQLite HistoricalCandleStore
    try:
        if not df_daily_merged.empty:
            df_store_d = df_daily_merged.copy()
            ts_c = [c for c in df_store_d.columns if "time" in c.lower() or "date" in c.lower()][0]
            df_store_d.set_index(pd.to_datetime(df_store_d[ts_c], format="mixed"), inplace=True)
            for sym in all_syms:
                store.upsert_candles(
                    symbol=sym,
                    interval="day",
                    broker=broker_name,
                    df=df_store_d,
                    source="real_historical_sync",
                )
            logger.info(f"[{symbol}] Synced {len(df_store_d)} daily candles to SQLite for {all_syms}")

        if not df_5m_merged.empty:
            df_store_5m = df_5m_merged.copy()
            ts_c = [c for c in df_store_5m.columns if "time" in c.lower() or "date" in c.lower()][0]
            df_store_5m.set_index(pd.to_datetime(df_store_5m[ts_c], format="mixed"), inplace=True)
            for sym in all_syms:
                store.upsert_candles(
                    symbol=sym,
                    interval="5minute",
                    broker=broker_name,
                    df=df_store_5m,
                    source="real_historical_sync",
                )
            logger.info(f"[{symbol}] Synced {len(df_store_5m)} 5m candles to SQLite for {all_syms}")

    except Exception as e:
        logger.warning(f"[{symbol}] SQLite sync warning: {e}")

    return {
        "symbol": symbol,
        "daily_count": len(df_daily_merged) if not df_daily_merged.empty else 0,
        "intraday_5m_count": len(df_5m_merged) if not df_5m_merged.empty else 0,
        "primary_csv_5m": str(intraday_paths[0]),
    }


def sync_all_commodities_historical(
    broker: Optional[BaseBroker] = None,
    broker_name: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    output_dir: str = "data/historical",
    allow_market_hours: bool = False,
    intraday_days: int = 180,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point to synchronize historical data for all supported MCX commodities.
    Enforces market hours safety rule unless allow_market_hours or force is True.
    """
    now = datetime.now(IST)
    if is_mcx_market_open(now) and not (allow_market_hours or force):
        msg = (
            f"🚫 [MarketProtection] MCX live market is currently OPEN ({now.strftime('%H:%M:%S')} IST). "
            f"Multi-commodity data download is deferred until post-market close (23:35 IST) "
            f"to protect Dhan/Upstox API quotas and avoid interfering with live trading. "
            f"Use --force to override if intentional."
        )
        logger.warning(msg)
        return {"status": "BLOCKED_MARKET_OPEN", "message": msg, "results": {}}

    # Resolve Broker (Dhan / Upstox)
    if broker_name is None:
        broker_name = os.getenv("BROKER", "dhan").lower().strip()
    if broker is None:
        if broker_name == "dhan":
            from broker.dhan_broker import DhanBroker
            broker = DhanBroker()
        elif broker_name == "upstox":
            from broker.upstox_broker import UpstoxBroker
            broker = UpstoxBroker()
        else:
            broker = get_broker()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    store = HistoricalCandleStore()

    target_configs = COMMODITY_CONFIGS
    if symbols:
        filter_syms = {s.upper().strip() for s in symbols}
        target_configs = [
            c for c in COMMODITY_CONFIGS
            if c["symbol"] in filter_syms or any(a in filter_syms for a in c.get("aliases", []))
        ]

    logger.info(
        f"Starting multi-commodity historical synchronization via {broker_name.upper()} "
        f"for {[c['symbol'] for c in target_configs]}..."
    )

    results = {}
    for commodity in target_configs:
        try:
            res = sync_single_commodity(
                commodity=commodity,
                broker=broker,
                broker_name=broker_name,
                output_dir=out_path,
                store=store,
                intraday_days=intraday_days,
            )
            results[commodity["symbol"]] = res
        except Exception as e:
            logger.error(f"Failed sync for {commodity['symbol']}: {e}")
            results[commodity["symbol"]] = {"error": str(e)}

    logger.success(f"Multi-commodity sync completed. Synced {len(results)} instruments.")
    return {"status": "SUCCESS", "results": results}


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize multi-commodity historical data from Dhan / Upstox."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols to sync (default: SILVERM,GOLDM,CRUDEOIL,NATGAS)",
    )
    parser.add_argument(
        "--broker",
        type=str,
        default=None,
        choices=["dhan", "upstox"],
        help="Broker to use (default: env BROKER or dhan)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/historical",
        help="Directory to store historical CSVs (default: data/historical)",
    )
    parser.add_argument(
        "--intraday-days",
        type=int,
        default=180,
        help="Days of intraday 5m lookback to fetch (default: 180)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even during active MCX market hours",
    )
    args = parser.parse_args()

    sym_list = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    sync_all_commodities_historical(
        broker_name=args.broker,
        symbols=sym_list,
        output_dir=args.output_dir,
        intraday_days=args.intraday_days,
        force=args.force,
    )


if __name__ == "__main__":
    main()

"""
scripts/download_dhan_historical.py — Download real historical MCX data from Dhan
================================================================================
Fetches 5-year daily and full intraday 5-minute historical OHLCV data for
MCX SILVERMIC (and other commodities) directly from Dhan HQ API.
"""

import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from broker.dhan_broker import DhanBroker
from instruments import SILVERMIC_CONFIG
from scripts.sync_commodity_historical_data import sync_all_commodities_historical


def download_all_commodities_data(
    output_dir: str = "data/historical",
    broker_name: str = "dhan",
    force: bool = False,
) -> dict:
    """Download historical data for all 4 commodities (SILVERM, GOLDM, CRUDEOIL, NATGAS)."""
    return sync_all_commodities_historical(
        output_dir=output_dir,
        broker_name=broker_name,
        force=force,
    )


def download_silvermic_data(output_dir: str = "data/historical") -> dict:
    """
    Downloads both daily data and intraday 5-minute data from Dhan.
    Merges monotonically with existing local CSVs and SQLite historical store
    so historical days are never lost as new live sessions accumulate (134, 135, 136...).
    """
    os.makedirs(output_dir, exist_ok=True)
    broker = DhanBroker()
    
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")
    
    # 1. Download daily data (2021 to present)
    start_5y = "2021-01-01"
    logger.info(f"Downloading daily SILVERMIC data from Dhan: {start_5y} to {today_str}...")
    df_daily = broker.get_historical_data("SILVERMIC", "day", start_5y, today_str)
    
    daily_file = os.path.join(output_dir, "SILVERMIC_dhan_1d.csv")
    if not df_daily.empty:
        df_daily.index.name = "timestamp"
        df_daily = df_daily.reset_index()
        df_daily["symbol"] = "SILVERMIC"
        df_daily["data_origin"] = "DHAN_REAL_HISTORICAL"
        
        if os.path.exists(daily_file):
            existing_daily = pd.read_csv(daily_file)
            prev_cnt = len(existing_daily)
            merged_daily = pd.concat([existing_daily, df_daily], ignore_index=True)
            ts_col = [c for c in merged_daily.columns if "time" in c.lower() or "date" in c.lower()][0]
            dt_series = pd.to_datetime(merged_daily[ts_col], format="mixed", utc=True).dt.tz_convert("Asia/Kolkata")
            merged_daily["_dt_sort"] = dt_series
            merged_daily = merged_daily.sort_values(by="_dt_sort").drop_duplicates(subset=["_dt_sort"], keep="last")
            merged_daily[ts_col] = merged_daily["_dt_sort"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)
            merged_daily = merged_daily.drop(columns=["_dt_sort"])
            merged_daily.to_csv(daily_file, index=False)
            logger.success(f"Accumulated daily candles: {prev_cnt} -> {len(merged_daily)} rows saved to {daily_file}")
            df_daily = merged_daily
        else:
            df_daily.to_csv(daily_file, index=False)
            logger.success(f"Saved {len(df_daily)} daily candles to {daily_file}")
    else:
        logger.warning("No new daily data fetched from Dhan (retaining existing).")

    # 2. Download full available 5-minute intraday data (rolling lookback)
    start_intraday = (today - timedelta(days=180)).strftime("%Y-%m-%d")
    logger.info(f"Downloading 5-minute SILVERMIC intraday data from Dhan: {start_intraday} to {today_str}...")
    df_5m = broker.get_historical_data("SILVERMIC", "5minute", start_intraday, today_str)
    
    intraday_file = os.path.join(output_dir, "SILVERMIC_dhan_5m.csv")
    if not df_5m.empty:
        df_5m.index.name = "timestamp"
        df_5m = df_5m.reset_index()
        df_5m["symbol"] = "SILVERMIC"
        df_5m["data_origin"] = "DHAN_REAL_HISTORICAL"
        
        if os.path.exists(intraday_file):
            existing_5m = pd.read_csv(intraday_file)
            prev_cnt = len(existing_5m)
            ts_col_old = [c for c in existing_5m.columns if "time" in c.lower() or "date" in c.lower()][0]
            prev_days = pd.to_datetime(existing_5m[ts_col_old], format="mixed", utc=True).dt.tz_convert("Asia/Kolkata").dt.date.nunique()
            
            merged_5m = pd.concat([existing_5m, df_5m], ignore_index=True)
            ts_col = [c for c in merged_5m.columns if "time" in c.lower() or "date" in c.lower()][0]
            dt_series_5m = pd.to_datetime(merged_5m[ts_col], format="mixed", utc=True).dt.tz_convert("Asia/Kolkata")
            merged_5m["_dt_sort"] = dt_series_5m
            merged_5m = merged_5m.sort_values(by="_dt_sort").drop_duplicates(subset=["_dt_sort"], keep="last")
            new_days = merged_5m["_dt_sort"].dt.date.nunique()
            merged_5m[ts_col] = merged_5m["_dt_sort"].dt.strftime("%Y-%m-%d %H:%M:%S%z").str.replace(r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True)
            merged_5m = merged_5m.drop(columns=["_dt_sort"])
            
            merged_5m.to_csv(intraday_file, index=False)
            logger.success(
                f"Accumulated 5-minute candles: {prev_cnt} -> {len(merged_5m)} candles "
                f"({prev_days} -> {new_days} trading sessions) saved to {intraday_file}"
            )
            df_5m = merged_5m
        else:
            df_5m.to_csv(intraday_file, index=False)
            logger.success(f"Saved {len(df_5m)} 5-minute candles to {intraday_file}")
    else:
        logger.warning("No new 5-minute data fetched from Dhan (retaining existing).")

    # 3. Sync to SQLite HistoricalCandleStore
    try:
        from data.historical_store import HistoricalCandleStore
        store = HistoricalCandleStore()
        if os.path.exists(intraday_file):
            df_for_store = pd.read_csv(intraday_file)
            ts_c = [c for c in df_for_store.columns if "time" in c.lower() or "date" in c.lower()][0]
            df_for_store.set_index(pd.to_datetime(df_for_store[ts_c], format="mixed"), inplace=True)
            for sym in ("SILVERMIC", "SILVERM"):
                store.upsert_candles(
                    symbol=sym,
                    interval="5minute",
                    broker="dhan",
                    df=df_for_store,
                    source="real_historical_merge",
                )
            logger.info("Successfully synced 5-minute candles to market_history.sqlite3 for SILVERMIC & SILVERM")
        if os.path.exists(daily_file):
            df_for_store_d = pd.read_csv(daily_file)
            ts_cd = [c for c in df_for_store_d.columns if "time" in c.lower() or "date" in c.lower()][0]
            df_for_store_d.set_index(pd.to_datetime(df_for_store_d[ts_cd], format="mixed"), inplace=True)
            for sym in ("SILVERMIC", "SILVERM"):
                store.upsert_candles(
                    symbol=sym,
                    interval="day",
                    broker="dhan",
                    df=df_for_store_d,
                    source="real_historical_merge",
                )
            logger.info("Successfully synced daily candles to market_history.sqlite3 for SILVERMIC & SILVERM")
    except Exception as e:
        logger.warning(f"Could not sync to SQLite store: {e}")

    return {
        "daily_file": daily_file,
        "daily_candles": len(df_daily) if not df_daily.empty else 0,
        "intraday_file": intraday_file,
        "intraday_candles": len(df_5m) if not df_5m.empty else 0,
    }


if __name__ == "__main__":
    download_silvermic_data()

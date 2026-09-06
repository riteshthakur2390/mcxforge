#!/usr/bin/env python3
"""
One-time backfill of existing parquet cache files into the local SQLite store.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import (
    DATA_CACHE_DIR,
    DATA_HIST_DB_PATH,
    RECENT_MARKET_WINDOW_DAYS,
    HISTORICAL_ARCHIVE_INTERVALS,
    SUPPORTED_INDEX_SYMBOLS,
)
from data.historical_store import HistoricalCandleStore
from broker.factory import get_active_broker_name
from utils.cache_manager import recent_source_window_days


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Import all data, not just archive")
    args = parser.parse_args()

    store = HistoricalCandleStore()
    active_broker = get_active_broker_name()
    
    # We'll import files one by one to use correct window_days for each interval
    imported = {}
    total_deleted = 0
    cache_root = Path(DATA_CACHE_DIR)
    
    for interval in HISTORICAL_ARCHIVE_INTERVALS:
        window_days = recent_source_window_days(active_broker, interval)
        for symbol in SUPPORTED_INDEX_SYMBOLS:
            for path in sorted(cache_root.glob(f"{symbol}_{interval}_*.parquet")):
                if path.name.startswith("instruments_"):
                    continue
                rows = store.import_cache_file(
                    path,
                    archive_only=not args.all,
                    window_days=window_days,
                    archive_intervals=(interval,),
                )
                imported[path.name] = rows
            
        if not args.all:
            deleted = store.prune_recent_candles(
                intervals=(interval,),
                window_days=window_days,
            )
            total_deleted += deleted

    print()
    print("SignalForge local history backfill")
    print(f"SQLite: {DATA_HIST_DB_PATH}")
    print(f"Cache dir: {DATA_CACHE_DIR}")
    print(f"Active broker for windows: {active_broker}")
    print(f"Archive intervals: {', '.join(HISTORICAL_ARCHIVE_INTERVALS)}")
    print()

    total_rows = 0
    for name, rows in imported.items():
        total_rows += int(rows or 0)
        print(f"{name}: {rows}")

    print()
    print(f"Total imported rows: {total_rows}")
    print(f"Pruned recent rows from local archive: {total_deleted}")
    print("Coverage:")
    for symbol in SUPPORTED_INDEX_SYMBOLS:
        for row in store.coverage(symbol=symbol):
            print(
                f"- {row['symbol']} {row['interval']}: "
                f"{row['earliest_date']} -> {row['latest_date']} "
                f"({row['rows']} rows, {row['brokers']} broker sources)"
            )


if __name__ == "__main__":
    main()

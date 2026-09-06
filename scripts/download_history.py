#!/usr/bin/env python3
"""
scripts/download_history.py — Seed Local Data Cache
======================================================
Downloads NIFTY/SENSEX historical data using yfinance and saves
to data/cache/ as Parquet files.

Run this ONCE before starting SignalForge with BROKER=yfinance.
After downloading, data is cached — no re-download needed.

Usage:
    python scripts/download_history.py
    python scripts/download_history.py --symbol NIFTY
    python scripts/download_history.py --symbol SENSEX
    python scripts/download_history.py --symbol NIFTY --interval day
    python scripts/download_history.py --interval 5minute

What gets downloaded:
    5-minute candles : last 60 days  (yfinance limit)
    Daily candles    : last 10 years (for strategy development)

Re-running this script refreshes the cache file in data/cache/.
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

IST = pytz.timezone("Asia/Kolkata")


def main():
    parser = argparse.ArgumentParser(
        description="Download index historical data via yfinance"
    )
    parser.add_argument(
        "--symbol", default="NIFTY",
        help="Symbol to download (default: NIFTY)"
    )
    parser.add_argument(
        "--interval", default="all",
        choices=["5minute", "day", "all"],
        help="Candle interval (default: all — downloads 5-minute and day)"
    )
    args = parser.parse_args()

    print()
    print("━" * 54)
    print("  SignalForge — Historical Data Download")
    print(f"  Broker: yfinance (free, no account needed)")
    print("━" * 54)

    # ── Check yfinance installed ───────────────────────────────────────────
    try:
        import yfinance as yf
        print(f"\n  ✅ yfinance {yf.__version__} ready")
    except ImportError:
        print("\n  ❌ yfinance not installed.")
        print("     Run: pip install yfinance")
        sys.exit(1)

    # ── Import broker ──────────────────────────────────────────────────────
    from broker.yfinance_broker import YFinanceBroker
    broker = YFinanceBroker()

    intervals_to_download = (
        ["5minute", "day"] if args.interval == "all"
        else [args.interval]
    )

    results = {}

    for interval in intervals_to_download:
        print()
        print(f"  Downloading {args.symbol} — {interval} candles...")

        try:
            df = broker.download_and_cache(
                symbol   = args.symbol,
                interval = interval,
            )

            if df.empty:
                print(f"  ⚠️  No data returned for {interval}")
                results[interval] = 0
                continue

            results[interval] = len(df)
            start = df.index[0].strftime("%Y-%m-%d")
            end   = df.index[-1].strftime("%Y-%m-%d")

            print(f"  ✅ {len(df):,} candles  ({start} → {end})")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[interval] = 0

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("━" * 54)
    print("  Download Summary")
    print("━" * 54)

    from config.settings import DATA_CACHE_DIR
    symbol_key = str(args.symbol or "NIFTY").upper().replace(" ", "")
    if symbol_key in {"NIFTY50", "NIFTY_50"}:
        symbol_key = "NIFTY"
    for interval, count in results.items():
        cache_file = Path(DATA_CACHE_DIR) / f"{symbol_key}_{interval}_yfinance.parquet"
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {interval:12} → {count:>6,} candles  [{cache_file.name}]")

    print()

    if all(v > 0 for v in results.values()):
        print("  ✅ All data downloaded successfully!")
        print()
        print("  Next steps:")
        print("  1. Set BROKER=yfinance in .env")
        print("  2. docker-compose up")
        print("  3. open http://localhost:5050")
        print()
        print("  Note: yfinance data is delayed ~15 min.")
        print("  For live data, switch to BROKER=kite/upstox/groww")
    else:
        print("  ⚠️  Some downloads failed. Check internet connection and retry.")

    print("━" * 54)
    print()


if __name__ == "__main__":
    main()

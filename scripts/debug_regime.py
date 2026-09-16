#!/usr/bin/env python3
"""
scripts/debug_regime.py — Diagnose why signals are being suppressed
=====================================================================
Run this to see exactly what ADX, Chop values are on your cached data.
Helps tune regime thresholds.

Usage:
    python scripts/debug_regime.py
    python scripts/debug_regime.py --days 10
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()

    from config.settings import DATA_CACHE_DIR, ADX_TREND_THRESHOLD, ADX_CHOP_THRESHOLD, CHOP_INDEX_THRESHOLD
    from pathlib import Path

    # Load cached data
    cache = Path(DATA_CACHE_DIR) / "NIFTY_5minute_yfinance.parquet"
    if not cache.exists():
        print(f"❌ Cache not found: {cache}")
        print("   Run: python scripts/download_history.py")
        sys.exit(1)

    df = pd.read_parquet(cache)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Last N days
    cutoff = pd.Timestamp.now(tz=df.index.tz) - pd.Timedelta(days=args.days)
    df = df[df.index >= cutoff]

    print(f"\n{'━'*60}")
    print(f"  MCXForge — Regime Debug")
    print(f"  Candles: {len(df)} | Days: {args.days}")
    print(f"  Thresholds: ADX trend>{ADX_TREND_THRESHOLD} | choppy<{ADX_CHOP_THRESHOLD} | chop>{CHOP_INDEX_THRESHOLD}")
    print(f"{'━'*60}\n")

    try:
        import pandas_ta as ta
    except ImportError:
        print("❌ pandas_ta not installed")
        sys.exit(1)

    from agents_code.agent9_regime.classifier import MarketRegimeAgent
    agent = MarketRegimeAgent()

    # Compute indicators on full dataset
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx_df is None:
        print("❌ ADX computation failed")
        sys.exit(1)

    adx_vals = adx_df["ADX_14"].dropna()

    print(f"  ADX Statistics:")
    print(f"    Min:    {adx_vals.min():.1f}")
    print(f"    Max:    {adx_vals.max():.1f}")
    print(f"    Mean:   {adx_vals.mean():.1f}")
    print(f"    Median: {adx_vals.median():.1f}")
    print()
    print(f"  Regime Distribution:")
    trending = (adx_vals > ADX_TREND_THRESHOLD).sum()
    ranging  = ((adx_vals >= ADX_CHOP_THRESHOLD) & (adx_vals <= ADX_TREND_THRESHOLD)).sum()
    choppy   = (adx_vals < ADX_CHOP_THRESHOLD).sum()
    total    = len(adx_vals)
    print(f"    TRENDING (ADX>{ADX_TREND_THRESHOLD}):  {trending:3d} candles ({trending/total*100:.0f}%)")
    print(f"    RANGING  ({ADX_CHOP_THRESHOLD}-{ADX_TREND_THRESHOLD}):   {ranging:3d} candles ({ranging/total*100:.0f}%)")
    print(f"    CHOPPY   (ADX<{ADX_CHOP_THRESHOLD}):  {choppy:3d} candles ({choppy/total*100:.0f}%)")
    print()

    # Choppiness index on last window
    chop = agent._chop_index(df.tail(50), 14)
    print(f"  Choppiness Index (last 50 candles): {chop:.1f}")
    print(f"    (>61.8 = choppy, <38.2 = trending)")
    print()

    # Per-day breakdown
    print(f"  Per-Day ADX Breakdown:")
    print(f"  {'Date':<12} {'Candles':>8} {'ADX Min':>9} {'ADX Max':>9} {'ADX Mean':>10} {'Regime':>10}")
    print(f"  {'-'*58}")

    df["_adx"] = adx_df["ADX_14"]
    df["_date"] = [i.date() for i in df.index]

    for day, group in df.groupby("_date"):
        adx_day = group["_adx"].dropna()
        if len(adx_day) == 0:
            continue
        mean_adx = adx_day.mean()
        regime = (
            "TRENDING" if mean_adx > ADX_TREND_THRESHOLD
            else "RANGING" if mean_adx >= ADX_CHOP_THRESHOLD
            else "CHOPPY"
        )
        print(f"  {str(day):<12} {len(group):>8} {adx_day.min():>9.1f} {adx_day.max():>9.1f} {mean_adx:>10.1f} {regime:>10}")

    print()

    # Recommendation
    if trending / total < 0.2:
        print(f"  ⚠️  Only {trending/total*100:.0f}% of candles are TRENDING.")
        print(f"     Current market may genuinely be choppy/ranging.")
        print(f"     Or try lowering ADX_TREND_THRESHOLD in config/settings/:")
        print(f"     Current: ADX_TREND_THRESHOLD = {ADX_TREND_THRESHOLD}")
        suggested = max(18, int(adx_vals.quantile(0.60)))
        print(f"     Suggested: ADX_TREND_THRESHOLD = {suggested}")
        print()
        print(f"     To test with lower threshold temporarily:")
        print(f"     Edit config/settings/ → ADX_TREND_THRESHOLD = {suggested}")
        print(f"     Then: docker-compose restart")
    else:
        print(f"  ✅ {trending/total*100:.0f}% of candles are TRENDING — thresholds look good.")

    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
scripts/debug_strategies.py — Strategy Eligibility Diagnostics
================================================================
Shows exactly which strategies run at each candle count,
in both backtest and live mode.

Run this to understand why you're getting low signal counts.

Usage:
    python scripts/debug_strategies.py
    python scripts/debug_strategies.py --candles 60
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")


def make_synthetic_df(n_candles: int) -> pd.DataFrame:
    """Create synthetic NIFTY-like candle data for testing."""
    np.random.seed(42)
    base   = 22000
    closes = base + np.cumsum(np.random.normal(0, 30, n_candles))
    highs  = closes + np.random.uniform(5, 40, n_candles)
    lows   = closes - np.random.uniform(5, 40, n_candles)
    opens  = np.roll(closes, 1)
    opens[0] = base
    vols   = np.random.randint(50000, 300000, n_candles)

    now   = datetime.now(IST).replace(hour=9, minute=30, second=0, microsecond=0)
    idx   = [now + timedelta(minutes=5*i) for i in range(n_candles)]

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    }, index=pd.DatetimeIndex(idx))


def run_diagnostics(candle_count: int = 60):
    from agents_code.agent2_strategy.runner import StrategyAgent, STRATEGY_REGISTRY

    agent = StrategyAgent()

    print()
    print("━" * 70)
    print(f"  SignalForge — Strategy Diagnostics")
    print(f"  Testing with {candle_count} candles")
    print("━" * 70)
    print()

    # Show eligibility report
    report = agent.get_eligibility_report(candle_count)
    print(f"  BACKTEST MODE ({len(report['backtest_eligible'])}/{report.get('total', '?')} eligible):")
    for name in report["backtest_eligible"]:
        print(f"    ✅ {name}")
    print()
    if report.get("skipped_backtest"):
        print(f"  Skipped in backtest:")
        for name in report["skipped_backtest"]:
            print(f"    ⏭  {name}")
    print()

    print(f"  LIVE MODE ({len(report['live_eligible'])}/{report.get('total', '?')} eligible):")
    for name in report["live_eligible"]:
        print(f"    ✅ {name}")
    print()

    # Test each strategy with synthetic data
    df = make_synthetic_df(candle_count)
    print(f"  INDIVIDUAL STRATEGY TEST (synthetic data, {candle_count} candles):")
    print(f"  {'Strategy':<20} {'Status':<12} {'Direction':<12} {'Conf':<8} {'ms'}")
    print(f"  {'-'*65}")

    import time
    from core.models import Direction
    for meta in STRATEGY_REGISTRY:
        t0 = time.perf_counter()
        try:
            result = meta.instance.evaluate(df, orb_high=22200, orb_low=21800)
            ms     = round((time.perf_counter() - t0) * 1000, 1)
            dir_   = result.get("direction", Direction.NONE)
            conf   = result.get("confidence", 0.0)
            if dir_ != Direction.NONE:
                status = "🔥 FIRED"
                dir_str = dir_.value if hasattr(dir_, "value") else str(dir_)
            else:
                status = "  NONE"
                dir_str = "-"
            print(f"  {meta.name:<20} {status:<12} {dir_str:<12} {conf:<8.3f} {ms}ms")
        except Exception as e:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            print(f"  {meta.name:<20} {'❌ ERROR':<12} {'-':<12} {0:<8.3f} {ms}ms  [{e}]")

    print()
    print("━" * 70)
    print()
    print("  To increase signal count, try running with more candles:")
    for n in [30, 50, 60, 80, 100]:
        agent2 = StrategyAgent()
        r      = agent2.get_eligibility_report(n)
        print(f"    {n:3d} candles → {len(r['backtest_eligible']):2d}/{r.get('total', '?')} eligible in backtest")
    print()
    print("  Current settings (config/settings/):")
    from config.settings import (
        VIX_HIGH_THRESHOLD,
        ADX_TREND_THRESHOLD, ONE_SIGNAL_AT_A_TIME, CANDLE_LOOKBACK
    )
    from config.settings.strategy import MIN_STRATEGY_VOTES
    print(f"    MIN_STRATEGY_VOTES  = {MIN_STRATEGY_VOTES}")
    print(f"    VIX_HIGH_THRESHOLD  = {VIX_HIGH_THRESHOLD}")
    print(f"    ADX_TREND_THRESHOLD = {ADX_TREND_THRESHOLD}")
    print(f"    ONE_SIGNAL_AT_A_TIME= {ONE_SIGNAL_AT_A_TIME}")
    print(f"    CANDLE_LOOKBACK     = {CANDLE_LOOKBACK}")
    print("━" * 70)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles", type=int, default=60)
    args = parser.parse_args()
    run_diagnostics(args.candles)

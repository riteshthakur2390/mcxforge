#!/usr/bin/env python3
"""
scripts/walk_forward_backtest.py — Walk-Forward Backtest
==========================================================
Replaces: Run backtest → retrain on same data → false confidence

Correct workflow:
  1. Split 66 days into 4 expanding windows
  2. Train ML on window N, test on window N+1
  3. Report out-of-sample performance ONLY
  4. Tell you whether ML is actually helping

Usage:
    python scripts/walk_forward_backtest.py
    python scripts/walk_forward_backtest.py --train-days 30 --test-days 10
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from pathlib import Path
from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days",  type=int, default=10)
    parser.add_argument("--step-days",  type=int, default=10)
    args = parser.parse_args()

    from ml.training.walk_forward import WalkForwardValidator
    from ml.training.trainer      import SignalForgeTrainer
    from config.settings           import DATA_CACHE_DIR, JOURNAL_DIR

    print()
    print("━" * 60)
    print("  SignalForge Walk-Forward Validation")
    print("━" * 60)
    print(f"  Train window: {args.train_days} days")
    print(f"  Test window:  {args.test_days} days")
    print(f"  Step:         {args.step_days} days")
    print()

    # Load candles
    trainer = SignalForgeTrainer(min_samples=5)
    try:
        candles = trainer.load_candles()
        print(f"  Candles loaded: {len(candles)} rows")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    # Load journal (all available labeled trades)
    try:
        journal = trainer.load_journal()
        print(f"  Journal loaded: {len(journal)} labeled trades")
        print(f"  Date range: {journal['date'].min()} → {journal['date'].max()}")
    except (FileNotFoundError, ValueError) as e:
        print(f"  ⚠️  Journal issue: {e}")
        print("  Running in synthetic mode — using backtest CSV files instead")
        # Fall back to backtest CSV files
        import glob
        csv_files = glob.glob("backtesting/results/backtest_trades_*.csv")
        if not csv_files:
            print("  ❌ No trade data found. Run backtest first.")
            sys.exit(1)
        dfs = [pd.read_csv(f) for f in csv_files]
        journal = pd.concat(dfs, ignore_index=True)
        journal["label"] = (journal["outcome_eod"] == "WIN").astype(int)
        journal = journal[journal["outcome_eod"].isin(["WIN", "LOSS"])].copy()
        print(f"  Loaded {len(journal)} trades from backtest CSVs")

    if len(journal) < 15:
        print()
        print("  ⚠️  INSUFFICIENT DATA FOR WALK-FORWARD")
        print(f"  Only {len(journal)} labeled trades found.")
        print("  Walk-forward needs ~50+ trades to be meaningful.")
        print()
        print("  RECOMMENDATION:")
        print("  1. Run OBSERVE mode for 2-4 weeks to collect real trade outcomes")
        print("  2. Then retrain ML on actual live trade data")
        print("  3. Current model is trained on backtest artifacts → unreliable")
        print()
        print("  For now, DISABLE ML filtering and use strategy votes only:")
        print("  → Set ML_MIN_CONFIDENCE = 0.0 in config/settings/")
        print("  → This lets all strategy-confirmed signals through without ML gate")
        sys.exit(0)

    # Run walk-forward validation
    wfv     = WalkForwardValidator(args.train_days, args.test_days, args.step_days)
    results = wfv.run(candles, journal)
    summary = wfv.report(results)

    # Final recommendation
    print("  ACTIONS:")
    if not summary.get("has_edge", False):
        print("  1. Disable ML gate (ML_MIN_CONFIDENCE = 0.0)")
        print("  2. Rely on MIN_STRATEGY_VOTES = 2 for signal filtering")
        print("  3. Re-enable ML after collecting 100+ live trades")
    else:
        print(f"  1. ML has edge (OOS AUC={summary['avg_oos_auc']:.3f}) — keep enabled")
        print("  2. Set ML_MIN_CONFIDENCE = 0.50 (use OOS precision as guide)")
        print("  3. Retrain monthly as new trades accumulate")
    print()


if __name__ == "__main__":
    main()

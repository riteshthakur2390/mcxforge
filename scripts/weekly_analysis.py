#!/usr/bin/env python3
"""
scripts/weekly_analysis.py — Trade Journal Feedback Loop
==========================================================
NEW FILE. Analyzes completed trades from journal CSVs and finds
patterns that expose which strategies, hours, and regimes work.

WHAT PROFESSIONAL ALGO TRADERS DO EVERY WEEK:
  1. By-hour win rate: are morning trades better than afternoon?
  2. By-strategy win rate: which strategies add real edge?
  3. Exit analysis: are TIME_DECAY exits a bug in logic or a market truth?
  4. Losing trade pattern: do losers share a common feature? (time, regime, VIX)
  5. Expectancy by setup: which combination of strategies has best EV?

This script generates a plain-text report that feeds back into settings.

USAGE:
    python scripts/weekly_analysis.py
    python scripts/weekly_analysis.py --days 30
"""

import sys, os, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_trades(days: int = 60) -> pd.DataFrame:
    """Load all backtest and journal trade CSVs."""
    files = (
        glob.glob("backtesting/results/backtest_trades_*.csv") +
        glob.glob("journal/signals_*.csv")
    )
    if not files:
        print("No trade files found. Run backtest first.")
        sys.exit(1)

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                cutoff = datetime.now() - timedelta(days=days)
                df = df[df["date"] >= cutoff]
            dfs.append(df)
        except Exception as e:
            print(f"  Skip {f}: {e}")

    if not dfs:
        print("No trades loaded.")
        sys.exit(1)

    full = pd.concat(dfs, ignore_index=True)
    full = full[full.get("outcome_eod", full.get("pnl_pct", pd.Series(dtype=float))).notna()]

    # Normalize win/loss column
    if "outcome_eod" in full.columns:
        full["win"] = (full["outcome_eod"] == "WIN").astype(int)
    elif "pnl_pct" in full.columns:
        full["win"] = (full["pnl_pct"].astype(float) > 0).astype(int)
    else:
        full["win"] = 0

    if "pnl_pct" not in full.columns:
        full["pnl_pct"] = 0.0
    full["pnl_pct"] = full["pnl_pct"].astype(float)

    return full


def report(df: pd.DataFrame) -> None:
    total  = len(df)
    wins   = df["win"].sum()
    wr     = wins / total * 100 if total > 0 else 0
    avg_pnl = df["pnl_pct"].mean()

    print()
    print("━" * 65)
    print("  SignalForge — Weekly Trade Analysis")
    print("━" * 65)
    print(f"  Trades: {total} | Wins: {wins} | Win Rate: {wr:.1f}% | Avg PnL: {avg_pnl:.1f}%")
    print()

    # ── 1. By hour ────────────────────────────────────────────────────────
    if "time" in df.columns:
        df["hour"] = df["time"].astype(str).str[:2].astype(int, errors="ignore")
        hourly = df.groupby("hour").agg(
            trades=("win", "count"),
            win_rate=("win", "mean"),
            avg_pnl=("pnl_pct", "mean"),
        ).reset_index()
        print("  BY HOUR:")
        for _, row in hourly.iterrows():
            bar = "█" * int(row["win_rate"] * 20)
            print(f"    {int(row['hour']):02d}:xx  {row['trades']:3.0f} trades | "
                  f"WR={row['win_rate']:.0%} {bar} | avg_pnl={row['avg_pnl']:+.1f}%")
        print()

        # Actionable insight
        best_hour = hourly.loc[hourly["win_rate"].idxmax()]
        worst_hour = hourly.loc[hourly["win_rate"].idxmin()]
        print(f"  💡 Best hour:  {int(best_hour['hour']):02d}:xx (WR={best_hour['win_rate']:.0%})")
        print(f"  ⚠️  Worst hour: {int(worst_hour['hour']):02d}:xx (WR={worst_hour['win_rate']:.0%})")
        if worst_hour["win_rate"] < 0.35 and worst_hour["trades"] >= 3:
            print(f"  → Consider: SIGNAL_START_TIME or NO_NEW_SIGNAL_AFTER adjustment")
        print()

    # ── 2. By strategy ────────────────────────────────────────────────────
    if "strategies_fired" in df.columns:
        strat_rows = []
        for _, row in df.iterrows():
            strats = row["strategies_fired"]
            if isinstance(strats, str):
                for s in strats.split("|"):
                    strat_rows.append({"strategy": s.strip(), "win": row["win"],
                                        "pnl_pct": row["pnl_pct"]})
        if strat_rows:
            sdf = pd.DataFrame(strat_rows)
            by_strat = sdf.groupby("strategy").agg(
                count=("win", "count"),
                win_rate=("win", "mean"),
                avg_pnl=("pnl_pct", "mean"),
            ).sort_values("win_rate", ascending=False)

            print("  BY STRATEGY (appears in trade):")
            for strat, row in by_strat.iterrows():
                flag = "✅" if row["win_rate"] >= 0.55 else "❌" if row["win_rate"] < 0.40 else "⚠️ "
                print(f"    {flag} {strat:<22} {row['count']:3.0f} trades | "
                      f"WR={row['win_rate']:.0%} | avg_pnl={row['avg_pnl']:+.1f}%")
            print()

    # ── 3. Exit reason analysis ───────────────────────────────────────────
    if "exit_reason" in df.columns:
        exits = df.groupby("exit_reason").agg(
            count=("win", "count"),
            win_rate=("win", "mean"),
            avg_pnl=("pnl_pct", "mean"),
        )
        print("  BY EXIT REASON:")
        for reason, row in exits.iterrows():
            flag = "✅" if row["avg_pnl"] > 5 else "❌" if row["avg_pnl"] < 0 else "⚠️ "
            print(f"    {flag} {reason:<20} {row['count']:3.0f} trades | "
                  f"WR={row['win_rate']:.0%} | avg_pnl={row['avg_pnl']:+.1f}%")

        if "TIME_DECAY" in exits.index:
            td = exits.loc["TIME_DECAY"]
            if td["avg_pnl"] < -2 and td["count"] >= 3:
                print()
                print("  💡 TIME_DECAY exits are losing — consider:")
                print("     1. Wider target (TARGET_PCT = 80 instead of 60)")
                print("     2. Or shorter hold time (reduce time cap to exit sooner)")
        print()

    # ── 4. Holding time vs PnL ────────────────────────────────────────────
    if "holding_minutes" in df.columns:
        df["hold_bucket"] = pd.cut(
            df["holding_minutes"].astype(float),
            bins=[0, 30, 60, 120, 240, 9999],
            labels=["<30m", "30-60m", "1-2h", "2-4h", ">4h"],
        )
        hold_analysis = df.groupby("hold_bucket").agg(
            count=("win", "count"),
            win_rate=("win", "mean"),
            avg_pnl=("pnl_pct", "mean"),
        )
        print("  BY HOLD TIME:")
        for bucket, row in hold_analysis.iterrows():
            if row["count"] > 0:
                print(f"    {str(bucket):<10} {row['count']:3.0f} trades | "
                      f"WR={row['win_rate']:.0%} | avg_pnl={row['avg_pnl']:+.1f}%")
        print()

    # ── 5. Actionable recommendations ────────────────────────────────────
    print("  RECOMMENDATIONS:")
    if total < 20:
        print(f"  ⚠️  Only {total} trades — analysis not statistically reliable.")
        print(f"     Run OBSERVE mode for 4+ weeks to collect 30+ trades.")
        print(f"     Keep ML_MIN_CONFIDENCE=0.0 until then (bypass ML gate).")
    else:
        print(f"  ✅ {total} trades — analysis is meaningful.")
        if wr < 45:
            print(f"  → Win rate {wr:.0f}% is low. Check: MTF alignment enabled?")
            print(f"     Set ADX_CHOP_THRESHOLD=18 to reduce ranging-day signals.")
        elif wr >= 60:
            print(f"  ✅ Win rate {wr:.0f}% is strong. Consider increasing position size.")
    print()
    print("━" * 65)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60, help="Days of history to analyse")
    args = parser.parse_args()

    df = load_trades(days=args.days)
    report(df)
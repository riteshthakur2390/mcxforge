"""
scripts/backtest_rca.py — Quick RCA for closed-trade backtest CSVs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _bucket_time(series: pd.Series) -> pd.Series:
    hour = series.astype(str).str.slice(0, 2)
    return hour.where(hour.str.isdigit(), "NA")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a backtest trade CSV")
    parser.add_argument("csv_path", type=Path, help="Path to backtest_trades_*.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if df.empty:
        print("No trades found.")
        return

    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct"), errors="coerce").fillna(0.0)
    df["holding_minutes"] = pd.to_numeric(df.get("holding_minutes"), errors="coerce").fillna(0).astype(int)
    time_col = df["time"] if "time" in df.columns else pd.Series(["NA"] * len(df))
    df["hour_bucket"] = _bucket_time(time_col)
    if "peak_pnl_pct" in df.columns:
        df["peak_pnl_pct"] = pd.to_numeric(df["peak_pnl_pct"], errors="coerce")
    else:
        df["peak_pnl_pct"] = pd.Series([pd.NA] * len(df))

    losses = df[df["pnl_pct"] < 0].copy()
    win_rate = round((df["pnl_pct"] > 0).mean() * 100, 1)

    print(f"Trades: {len(df)} | Win rate: {win_rate}% | Total PnL: {df['pnl_pct'].sum():.2f}%")
    print("\nExit reason summary")
    print(df.groupby("exit_reason")["pnl_pct"].agg(["count", "mean", "median", "min", "max"]).round(2).to_string())

    if losses.empty:
        print("\nNo losses in this run.")
        return

    print("\nLoss clusters by setup")
    print(losses.groupby("setup_type")["pnl_pct"].agg(["count", "mean", "median", "min"]).round(2).to_string())

    print("\nLoss clusters by hour")
    print(losses.groupby("hour_bucket")["pnl_pct"].agg(["count", "mean", "median"]).round(2).to_string())

    print("\nLoss clusters by holding bucket")
    hold_bucket = pd.cut(
        losses["holding_minutes"],
        bins=[0, 30, 60, 90, 9999],
        labels=["<=30", "31-60", "61-90", "90+"],
        include_lowest=True,
    )
    print(losses.groupby(hold_bucket)["pnl_pct"].agg(["count", "mean", "median"]).round(2).to_string())

    peak_cols = ["date", "time", "setup_type", "exit_reason", "pnl_pct", "peak_pnl_pct", "holding_minutes"]
    peak_cols = [c for c in peak_cols if c in losses.columns]
    print("\nWorst losses")
    print(losses.sort_values("pnl_pct").head(10)[peak_cols].to_string(index=False))

    if losses["peak_pnl_pct"].notna().any():
        recovered = losses[
            (losses["peak_pnl_pct"] >= 15.0) & (losses["pnl_pct"] < 0)
        ]
        print("\nLosses that touched 15%+ peak first")
        if recovered.empty:
            print("None")
        else:
            print(recovered[peak_cols].sort_values("peak_pnl_pct", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

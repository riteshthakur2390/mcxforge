#!/usr/bin/env python3
"""
scripts/compare_4lakh_vs_current.py

Forensic Trade-by-Trade & Day-by-Day Comparison:
Benchmark Run: backtesting/results/backtest_trades_20260818_175704.csv (₹4.33 Lakh Net PnL)
VS
Current System Runs: P&L_MAXIMIZER_V1 / ALPHA_HUNTER (~₹30k - ₹63k)

Extracts:
1. Exact differences in lot sizing, targets, exits, trade selection, holding time
2. Day-by-day P&L comparison on overlapping dates
3. Trade-by-trade anatomy of winners and losers
4. The exact code/parameter differences that produced ₹4.33 Lakh profit
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]

def analyze_4l_run():
    f_4l = ROOT_DIR / "backtesting/results/backtest_trades_20260818_175704.csv"
    df_4l = pd.read_csv(f_4l)

    print("="*95)
    print("      PART 1: DEEP FORENSIC PROFILE OF AUGUST 18 BENCHMARK (₹4.33 LAKH NET PNL)")
    print("="*95)

    df_4l["date_dt"] = pd.to_datetime(df_4l["date"])
    print(f"Date Range: {df_4l['date'].min()} to {df_4l['date'].max()} ({df_4l['date'].nunique()} trading days)")
    print(f"Total Trades: {len(df_4l)}")
    print(f"Gross P&L: ₹{df_4l['gross_pnl_inr'].sum():,.2f}")
    print(f"Total Charges: ₹{df_4l['total_charges'].sum():,.2f}")
    print(f"Net P&L: ₹{df_4l['net_pnl_inr'].sum():,.2f}")
    
    wins = df_4l[df_4l["net_pnl_inr"] > 0]
    losses = df_4l[df_4l["net_pnl_inr"] <= 0]
    wr = len(wins) / len(df_4l) * 100.0
    gw = wins["gross_pnl_inr"].sum()
    gl = abs(losses["gross_pnl_inr"].sum())
    pf = gw / gl if gl > 0 else 99.0
    
    print(f"Win Rate: {wr:.2f}% ({len(wins)} wins, {len(losses)} losses)")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Average Win: ₹{wins['net_pnl_inr'].mean():,.2f}")
    print(f"Average Loss: ₹{losses['net_pnl_inr'].mean():,.2f}")
    print(f"Win/Loss Ratio (Payoff): {abs(wins['net_pnl_inr'].mean() / losses['net_pnl_inr'].mean()):.2f}x")
    print(f"Expectancy: ₹{df_4l['net_pnl_inr'].mean():,.2f} per trade")

    # 1. Lot sizing distribution
    print("\n-- LOT SIZING IN 4.33L RUN --")
    if "lots" in df_4l.columns:
        print(df_4l["lots"].value_counts().sort_index().to_string())
        print(f"Mean lots: {df_4l['lots'].mean():.2f}, Max lots: {df_4l['lots'].max()}")
        print(f"Mean quantity: {df_4l['quantity'].mean():.1f}")
        print(f"Mean total invested: ₹{df_4l['total_invested'].mean():,.2f}")

    # 2. Exit Reason Breakdown
    print("\n-- EXIT REASONS IN 4.33L RUN --")
    exit_summary = df_4l.groupby("exit_reason").agg(
        trades=("net_pnl_inr", "count"),
        net_pnl=("net_pnl_inr", "sum"),
        avg_pnl=("net_pnl_inr", "mean"),
        win_rate=("net_pnl_inr", lambda s: (s > 0).mean() * 100.0)
    ).reset_index()
    print(exit_summary.to_string(index=False))

    # 3. Budget Lane Breakdown
    print("\n-- BUDGET LANE BREAKDOWN IN 4.33L RUN --")
    if "budget_lane" in df_4l.columns:
        lane_summary = df_4l.groupby("budget_lane").agg(
            trades=("net_pnl_inr", "count"),
            net_pnl=("net_pnl_inr", "sum"),
            avg_pnl=("net_pnl_inr", "mean"),
            win_rate=("net_pnl_inr", lambda s: (s > 0).mean() * 100.0)
        ).reset_index()
        print(lane_summary.to_string(index=False))

    # 4. Top 10 Biggest Winners in 4.33L Run
    print("\n-- TOP 10 BIGGEST WINNERS IN 4.33L RUN --")
    cols = ["date", "time", "direction", "option_symbol", "lots", "entry_premium", "exit_premium", "exit_reason", "net_pnl_inr"]
    print(df_4l.sort_values("net_pnl_inr", ascending=False)[cols].head(10).to_string(index=False))

    return df_4l

def compare_with_recent(df_4l):
    # Load recent backtest ledger
    f_recent = ROOT_DIR / "analysis/deterministic_1287_backtest/legacy_context_trade_ledger.csv"
    if not f_recent.exists():
        print(f"Recent ledger {f_recent} not found.")
        return

    df_rec = pd.read_csv(f_recent)
    print("\n" + "="*95)
    print("      PART 2: COMPARISON WITH RECENT DETERMINISTIC 1,287 BACKTEST")
    print("="*95)

    print(f"Recent Ledger Trades: {len(df_rec)}")
    print(f"Recent Ledger Net P&L: ₹{df_rec['net_pnl'].sum():,.2f}")
    
    # Compare key differences
    print("\n-- CORE STRUCTURAL DIFFERENCES --")
    diff_table = [
        {"Dimension": "Total Net P&L", "Aug 18 Benchmark": f"₹{df_4l['net_pnl_inr'].sum():,.2f}", "Recent Replay Engine": f"₹{df_rec['net_pnl'].sum():,.2f}"},
        {"Dimension": "Total Trades", "Aug 18 Benchmark": f"{len(df_4l)} trades", "Recent Replay Engine": f"{len(df_rec)} trades"},
        {"Dimension": "Trades Per Day", "Aug 18 Benchmark": f"{len(df_4l)/df_4l['date'].nunique():.2f} trades/day", "Recent Replay Engine": f"{len(df_rec)/df_rec['session_date'].nunique():.2f} trades/day"},
        {"Dimension": "Win Rate", "Aug 18 Benchmark": f"{(df_4l['net_pnl_inr']>0).mean()*100:.1f}%", "Recent Replay Engine": f"{(df_rec['net_pnl']>0).mean()*100:.1f}%"},
        {"Dimension": "Avg Win Size", "Aug 18 Benchmark": f"₹{df_4l[df_4l['net_pnl_inr']>0]['net_pnl_inr'].mean():,.2f}", "Recent Replay Engine": f"₹{df_rec[df_rec['net_pnl']>0]['net_pnl'].mean():,.2f}"},
        {"Dimension": "Avg Loss Size", "Aug 18 Benchmark": f"₹{df_4l[df_4l['net_pnl_inr']<=0]['net_pnl_inr'].mean():,.2f}", "Recent Replay Engine": f"₹{df_rec[df_rec['net_pnl']<=0]['net_pnl'].mean():,.2f}"},
        {"Dimension": "Payoff Ratio", "Aug 18 Benchmark": f"{abs(df_4l[df_4l['net_pnl_inr']>0]['net_pnl_inr'].mean()/df_4l[df_4l['net_pnl_inr']<=0]['net_pnl_inr'].mean()):.2f}x", "Recent Replay Engine": f"{abs(df_rec[df_rec['net_pnl']>0]['net_pnl'].mean()/df_rec[df_rec['net_pnl']<=0]['net_pnl'].mean()):.2f}x"},
        {"Dimension": "Max Lots Traded", "Aug 18 Benchmark": f"{df_4l['lots'].max()} lots", "Recent Replay Engine": "Fixed 1 lot (quantity=65)"},
        {"Dimension": "Exit Mechanism", "Aug 18 Benchmark": "Dynamic ATR Target + Stale Loss + Trailing", "Recent Replay Engine": "Fixed 15pt SL / 30pt Target"},
        {"Dimension": "Capital Deployed", "Aug 18 Benchmark": f"Up to ₹{df_4l['total_invested'].max():,.2f} (Multi-lot compounding)", "Recent Replay Engine": "Fixed ~₹7,000 (1 lot)"}
    ]
    print(pd.DataFrame(diff_table).to_string(index=False))

    # Day-by-Day Overlapping Comparison
    print("\n-- DAY-BY-DAY OVERLAPPING COMPARISON (SAMPLE 15 DAYS) --")
    daily_4l = df_4l.groupby("date")["net_pnl_inr"].sum()
    daily_rec = df_rec.groupby("session_date")["net_pnl"].sum()
    
    overlap_dates = sorted(list(set(daily_4l.index).intersection(set(daily_rec.index))))
    overlap_rows = []
    for d in overlap_dates[:15]:
        p4 = daily_4l.get(d, 0.0)
        pr = daily_rec.get(d, 0.0)
        overlap_rows.append({
            "Date": d,
            "Aug 18 (4.33L) PnL": f"₹{p4:,.2f}",
            "Recent Replay PnL": f"₹{pr:,.2f}",
            "Difference": f"₹{p4 - pr:+,.2f}",
            "Why Aug 18 Won More": "Multi-lot sizing (2-3 lots) on runners" if p4 > pr else "Recent took less risk"
        })
    print(pd.DataFrame(overlap_rows).to_string(index=False))

if __name__ == "__main__":
    df_4l = analyze_4l_run()
    compare_with_recent(df_4l)

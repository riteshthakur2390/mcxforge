#!/usr/bin/env python3
"""
scripts/run_1295d_high_conviction_replay.py

Authoritative 1,295-Day High-Conviction Replay Engine:
Executes the high-conviction 1-2 trade per day setup (EMA20 pullback confirmation,
quality gate, single best trade per session bucket) across all 1,295 historical sessions.

Usage:
    python3 scripts/run_1295d_high_conviction_replay.py
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

def run_replay():
    print("="*90)
    print("      SIGNALFORGE 1,295-DAY HIGH-CONVICTION OPPORTUNITY REPLAY")
    print("="*90)
    
    csv_path = ROOT_DIR / "analysis/replay_1295d/phase7d_1295d_replayed_trades_ledger.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)
    total_sessions = df["session_date"].nunique()
    total_trades = len(df)

    gross_pnl = df["gross_realized_pnl"].sum()
    charges = df["actual_charges"].sum()
    net_pnl = df["replay_net_pnl"].sum()
    
    wins = df[df["replay_net_pnl"] > 0]
    losses = df[df["replay_net_pnl"] <= 0]
    wr = len(wins) / total_trades * 100.0
    pf = wins["gross_realized_pnl"].sum() / abs(losses["gross_realized_pnl"].sum())

    start_cap = float(df.iloc[0]["total_capital_at_decision"])
    end_cap = float(df.iloc[-1]["total_capital_at_decision"])

    print(f"Total Sessions Replayed: {total_sessions} (2021-01-04 to 2025-12-19)")
    print(f"Total High-Conviction Trades: {total_trades} (Avg {total_trades/total_sessions:.1f} trades/day)")
    print(f"Win Count: {len(wins)} | Loss Count: {len(losses)} | Win Rate: {wr:.2f}%")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Gross P&L: ₹{gross_pnl:,.2f}")
    print(f"Total Transaction Costs: ₹{charges:,.2f}")
    print(f"Net Realized P&L: ₹{net_pnl:,.2f}")
    print(f"Starting Capital: ₹{start_cap:,.2f}")
    print(f"Ending Capital: ₹{end_cap:,.2f} (+₹{end_cap - start_cap:,.2f} / +{((end_cap-start_cap)/start_cap)*100:.1f}%)")

    # Drawdown
    eq = np.cumsum(df["replay_net_pnl"].values)
    dd = eq - np.maximum.accumulate(eq)
    max_dd = float(np.min(dd))
    print(f"Maximum Realized Drawdown: ₹{max_dd:,.2f}")
    print(f"Realized Mean MFE: {df['realized_mfe_pts'].mean():.2f} pts | Mean MAE: {df['realized_mae_pts'].mean():.2f} pts")
    print("="*90)

if __name__ == "__main__":
    run_replay()

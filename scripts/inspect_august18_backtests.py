#!/usr/bin/env python3
"""
scripts/inspect_august18_backtests.py

Inspect all August 18, 2026 backtest runs in backtesting/results/
Calculates Total Trades, Net PnL, Gross PnL, Win Rate, Profit Factor, Charges.
"""

import sys
import glob
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

def inspect_runs():
    files = sorted(glob.glob(str(ROOT_DIR / "backtesting/results/backtest_trades_20260818_*.csv")))
    print(f"Found {len(files)} backtest runs from August 18.")

    results = []
    for f in files:
        fname = Path(f).name
        try:
            df = pd.read_csv(f)
            pnl_col = "net_pnl_inr" if "net_pnl_inr" in df.columns else ("realized_pnl" if "realized_pnl" in df.columns else None)
            gross_col = "gross_pnl_inr" if "gross_pnl_inr" in df.columns else None
            charges_col = "total_charges" if "total_charges" in df.columns else None
            
            if pnl_col:
                net_pnl = df[pnl_col].sum()
                gross_pnl = df[gross_col].sum() if gross_col else 0.0
                charges = df[charges_col].sum() if charges_col else 0.0
                wins = df[df[pnl_col] > 0]
                losses = df[df[pnl_col] <= 0]
                wr = len(wins) / len(df) * 100.0 if len(df) > 0 else 0.0
                
                results.append({
                    "Filename": fname,
                    "Trades": len(df),
                    "WinRate": f"{wr:.1f}%",
                    "GrossPnL": f"₹{gross_pnl:,.2f}",
                    "Charges": f"₹{charges:,.2f}",
                    "NetPnL": f"₹{net_pnl:,.2f}",
                    "RawNet": net_pnl
                })
        except Exception as e:
            results.append({"Filename": fname, "Trades": 0, "WinRate": "0", "GrossPnL": "0", "Charges": "0", "NetPnL": f"Error: {e}", "RawNet": 0})

    df_out = pd.DataFrame(results).sort_values("RawNet", ascending=False)
    print(df_out[["Filename", "Trades", "WinRate", "GrossPnL", "Charges", "NetPnL"]].to_string(index=False))

if __name__ == "__main__":
    inspect_runs()

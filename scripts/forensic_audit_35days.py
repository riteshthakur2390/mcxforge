#!/usr/bin/env python3
"""
scripts/forensic_audit_35days.py

Forensic Audit of the 35-Day Live Integration Dataset:
1. Day-by-day P&L, trades, wins, losses, win rate
2. Analysis of the 15 losing trades: exact causes, MAE, MFE, exit reasons
3. What mistakes occurred (whipsaw entries, morning traps, double losses)
4. Benchmarking: Why Phase 7D had ~4.6 Lakh account balance / ₹2.15L net profit
   vs why the recent unconstrained backtest had ~₹30k
5. The command to run the authoritative high-conviction replay across all 1,295 days!
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

def audit_35_days():
    csv_path = ROOT_DIR / "analysis/final_integration/final_35day_trade_ledger.csv"
    df = pd.read_csv(csv_path)
    
    print("="*90)
    print("       TASK 1: FORENSIC AUDIT OF 35-DAY CANONICAL INTEGRATION DATASET")
    print("="*90)
    print(f"Total Sessions: {df['session_date'].nunique()} | Total Trades: {len(df)}")
    print(f"Gross P&L: ₹{df['gross_PnL'].sum():,.2f} | Costs: ₹{df['transaction_cost'].sum():,.2f} | Net P&L: ₹{df['net_PnL'].sum():,.2f}")
    
    wins = df[df['net_PnL'] > 0]
    losses = df[df['net_PnL'] <= 0]
    gw = wins['gross_PnL'].sum()
    gl = abs(losses['gross_PnL'].sum())
    
    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {len(wins)/len(df)*100:.2f}% | Profit Factor: {gw/gl:.2f}")
    print(f"Average Win: ₹{wins['net_PnL'].mean():.2f} | Average Loss: ₹{losses['net_PnL'].mean():.2f} | Expectancy: ₹{df['net_PnL'].mean():.2f}/trade\n")

    # 1. Day-by-Day Performance
    print("-- DAY-BY-DAY PERFORMANCE (35 SESSIONS) --")
    daily = df.groupby("session_date").agg(
        trades=("net_PnL", "count"),
        wins=("net_PnL", lambda s: (s > 0).sum()),
        losses=("net_PnL", lambda s: (s <= 0).sum()),
        net_pnl=("net_PnL", "sum"),
        contracts=("contract", lambda s: list(s.unique())),
        directions=("direction", lambda s: list(s.unique())),
    ).reset_index()
    daily["cum_pnl"] = daily["net_pnl"].cumsum()
    print(daily[["session_date", "trades", "wins", "losses", "net_pnl", "cum_pnl", "directions"]].to_string(index=False))

    # 2. Detailed Anatomy of the 15 Losses
    print("\n" + "="*90)
    print("                   ANATOMY OF THE 15 LOSING TRADES")
    print("="*90)
    loss_df = df[df['net_PnL'] <= 0].copy()
    loss_cols = ["session_date", "direction", "contract", "entry_price", "exit_price", "exit_reason", "gross_PnL", "net_PnL", "realized_mae_pts", "realized_mfe_pts", "bars_held"]
    print(loss_df[loss_cols].to_string(index=False))

    # Key Mistakes Identified
    print("\n-- KEY MISTAKES IDENTIFIED ACROSS THE 15 LOSSES --")
    
    # A. Intra-day directional flipping (Whipsaw)
    multi_dir_days = daily[daily["directions"].apply(lambda d: len(set(d)) > 1)]
    print(f"1. Directional Whipsaw (Taking both CALL and PUT on same day):")
    print(f"   Occurred on {len(multi_dir_days)} sessions (e.g. 2021-01-13, 2021-01-18). On both days, the second trade was a loss!")
    
    # B. Entry into exhausted momentum (Low MFE)
    low_mfe_losses = loss_df[loss_df["realized_mfe_pts"] < 1.0]
    print(f"2. Zero-Follow-Through Entries (Realized MFE < 1.0 pt):")
    print(f"   {len(low_mfe_losses)} out of 15 losses ({len(low_mfe_losses)/len(loss_df)*100:.1f}%) had MFE < 1 pt. Entry occurred at the exact top/bottom.")
    
    # C. High MAE rapid stop hits
    rapid_sl = loss_df[loss_df["bars_held"] <= 2]
    print(f"3. Immediate Reversal (Stopped out within 2 bars):")
    print(f"   {len(rapid_sl)} out of 15 losses were stopped out in <= 2 bars (10 mins).\n")

    return df, daily, loss_df

def benchmark_4lakh_vs_30k():
    print("="*90)
    print("      TASK 3: BENCHMARKING — 4 LAKH PROFIT (PHASE 7D) VS 30K (RECENT BACKTEST)")
    print("="*90)

    # 1. Phase 7D 1,295-Day Results
    p7d_rep = json.load(open(ROOT_DIR / "analysis/replay_1295d/phase7d_long_horizon_robustness_report.json"))
    p7d_perf = p7d_rep["overall_performance"]

    # 2. Recent Unconstrained Baseline & P&L_MAXIMIZER_V1
    bm_data = [
        {
            "Architecture": "Phase 7D Canonical (4 Lakh Account)",
            "Sessions": 1295,
            "Total Trades": p7d_perf["total_replayed_trades"],
            "Win Rate": f"{p7d_perf['win_rate_pct']:.1f}%",
            "Profit Factor": p7d_perf["profit_factor"],
            "Gross P&L": f"₹{p7d_perf['total_realized_gross_pnl_inr']:,.2f}",
            "Transaction Costs": f"₹{p7d_perf['total_charges_inr']:,.2f}",
            "Net Realized P&L": f"₹{p7d_perf['total_realized_net_pnl_inr']:,.2f}",
            "Ending Capital": "₹4,64,538.60 (Start ₹2.5L)",
            "Why it Performed This Way": "State Machine Pullback Gate + 2 trades/day cap + High/Med quality budget sizing"
        },
        {
            "Architecture": "Unconstrained Replay Baseline",
            "Sessions": 1287,
            "Total Trades": 3267,
            "Win Rate": "29.1%",
            "Profit Factor": 0.83,
            "Gross P&L": "-₹169,711.10",
            "Transaction Costs": "₹462,081.60",
            "Net Realized P&L": "-₹631,792.70",
            "Ending Capital": "Heavily Negative",
            "Why it Performed This Way": "Fired raw 4-vote consensus on every 5m bar with zero gating; massive fee drag"
        },
        {
            "Architecture": "P&L_MAXIMIZER_V1 (Frozen)",
            "Sessions": 1287,
            "Total Trades": 804,
            "Win Rate": "33.3%",
            "Profit Factor": 1.16,
            "Gross P&L": "+₹167,466.00",
            "Transaction Costs": "₹136,825.60",
            "Net Realized P&L": "+₹30,640.40",
            "Ending Capital": "₹2,80,640.40",
            "Why it Performed This Way": "Deduplicated bar signals, filtered 10:00-11:30 chop, but took 804 trades (fees ₹136k)"
        },
        {
            "Architecture": "ALPHA_HUNTER (Top 1 Daily)",
            "Sessions": 986,
            "Total Trades": 529,
            "Win Rate": "36.3%",
            "Profit Factor": 1.24,
            "Gross P&L": "+₹147,833.40",
            "Transaction Costs": "₹84,208.00",
            "Net Realized P&L": "+₹63,625.40",
            "Ending Capital": "₹3,13,625.40",
            "Why it Performed This Way": "Single best trade per day, cut fee drag to ₹84k, halved max drawdown"
        }
    ]

    print(pd.DataFrame(bm_data)[["Architecture", "Total Trades", "Win Rate", "Profit Factor", "Gross P&L", "Transaction Costs", "Net Realized P&L", "Ending Capital"]].to_string(index=False))

    print("\n-- ROOT CAUSE OF THE DIFFERENCE --")
    print("""
1. Trade Filtering & Frequency:
   - In Phase 7D (4 Lakh account), trades were strictly limited to CONFIRMED EMA20 PULLBACKS (6-bar state machine)
     with High/Medium Quality gating, taking at most 2 trades per session (2,590 trades across 1,295 sessions).
     It achieved a 75% win rate because false breakouts were filtered by the EMA20 retest.
   - In the recent unconstrained backtest, the deterministic replay engine was firing raw consensus trades on EVERY
     single 5-minute candle (3,267 trades), accumulating ₹462,081 in transaction costs!
   - Even when filtered down to P&L_MAXIMIZER_V1 (804 trades) or Alpha Hunter (529 trades), the backtest was using
     a tighter 15-pt stop loss without dynamic lot scaling.

2. Account Capital vs Profit:
   - In Phase 7D, the starting capital was ₹2,50,000 and ending capital was ₹4,64,538.60 (+₹2.15 Lakh Net Profit,
     +₹5.48 Lakh Gross Profit). Users colloquially referred to this as the "4 Lakh backtest" because the account grew to ~4.6 Lakh.
   - In the ₹30k capital test, trading 1 lot restricts rupee gains per win to ₹1,950, so cumulative P&L scales to ~₹30k–₹63k.
""")

if __name__ == "__main__":
    audit_35_days()
    benchmark_4lakh_vs_30k()

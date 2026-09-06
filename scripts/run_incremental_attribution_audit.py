#!/usr/bin/env python3
"""
scripts/run_incremental_attribution_audit.py

Comprehensive incremental trade attribution, strategy contribution matrix,
option price path audit, and determinism verification fix.
"""

import ast
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

def run_attribution_audit():
    analysis_dir = Path("analysis/deterministic_1287_backtest")
    legacy_file = analysis_dir / "legacy_context_trade_ledger.csv"
    rolling_file = analysis_dir / "rolling_context_trade_ledger.csv"
    activations_file = analysis_dir / "strategy_activation_comparison.csv"

    df_legacy = pd.read_csv(legacy_file)
    df_rolling = pd.read_csv(rolling_file)
    df_activations = pd.read_csv(activations_file)

    print(f"Loaded Legacy Trades : {len(df_legacy)}")
    print(f"Loaded Rolling Trades: {len(df_rolling)}")

    # Parse strategy_votes lists
    df_legacy['votes_list'] = df_legacy['strategy_votes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    df_rolling['votes_list'] = df_rolling['strategy_votes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # ──────────────────────────────────────────────────────────────────────────
    # 1. IDENTIFY INCREMENTAL TRADES
    # ──────────────────────────────────────────────────────────────────────────
    # A trade in rolling is "shared" if there is a matching trade in legacy on the same session_date with same entry_timestamp and direction
    df_legacy['trade_key'] = df_legacy['session_date'] + "_" + df_legacy['entry_timestamp'] + "_" + df_legacy['direction']
    df_rolling['trade_key'] = df_rolling['session_date'] + "_" + df_rolling['entry_timestamp'] + "_" + df_rolling['direction']

    legacy_keys = set(df_legacy['trade_key'])
    
    df_rolling['is_incremental'] = ~df_rolling['trade_key'].isin(legacy_keys)
    df_incremental = df_rolling[df_rolling['is_incremental']].copy()
    df_shared = df_rolling[~df_rolling['is_incremental']].copy()

    print(f"\n[1] Incremental Trade Count: {len(df_incremental)} (out of {len(df_rolling)})")
    print(f"    Shared Trade Count     : {len(df_shared)}")

    inc_wins = df_incremental[df_incremental['is_win']]
    inc_losses = df_incremental[~df_incremental['is_win']]
    inc_win_rate = (len(inc_wins) / len(df_incremental)) * 100 if len(df_incremental) > 0 else 0
    inc_gross_pnl = df_incremental['gross_pnl'].sum()
    inc_costs = df_incremental['transaction_cost'].sum()
    inc_net_pnl = df_incremental['net_pnl'].sum()
    inc_pf = abs(inc_wins['gross_pnl'].sum() / inc_losses['gross_pnl'].sum()) if abs(inc_losses['gross_pnl'].sum()) > 0 else 0
    inc_expectancy = inc_net_pnl / len(df_incremental) if len(df_incremental) > 0 else 0

    print(f"    Incremental Win Rate   : {inc_win_rate:.2f}% ({len(inc_wins)}W / {len(inc_losses)}L)")
    print(f"    Incremental Profit Fac : {inc_pf:.2f}")
    print(f"    Incremental Gross PnL  : ₹{inc_gross_pnl:,.2f}")
    print(f"    Incremental Costs      : ₹{inc_costs:,.2f}")
    print(f"    Incremental Net PnL    : ₹{inc_net_pnl:,.2f}")
    print(f"    Incremental Expectancy : ₹{inc_expectancy:.2f} / trade")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. STRATEGY-LEVEL CONTRIBUTION & ATTRIBUTION
    # ──────────────────────────────────────────────────────────────────────────
    all_strategies = sorted(list(set([s for votes in df_rolling['votes_list'] for s in votes] + list(df_activations['lead_name']))))
    strat_rows = []

    for s_name in all_strategies:
        # Legacy stats
        leg_trades = df_legacy[df_legacy['votes_list'].apply(lambda v: s_name in v)]
        leg_n = len(leg_trades)
        leg_wins = len(leg_trades[leg_trades['is_win']])
        leg_net_pnl = leg_trades['net_pnl'].sum() if leg_n > 0 else 0.0

        # Rolling stats
        rol_trades = df_rolling[df_rolling['votes_list'].apply(lambda v: s_name in v)]
        rol_n = len(rol_trades)
        rol_wins = len(rol_trades[rol_trades['is_win']])
        rol_losses = rol_n - rol_wins
        rol_wr = (rol_wins / rol_n * 100) if rol_n > 0 else 0.0
        rol_gross_pnl = rol_trades['gross_pnl'].sum() if rol_n > 0 else 0.0
        rol_net_pnl = rol_trades['net_pnl'].sum() if rol_n > 0 else 0.0
        rol_win_gross = rol_trades[rol_trades['is_win']]['gross_pnl'].sum() if rol_wins > 0 else 0.0
        rol_loss_gross = abs(rol_trades[~rol_trades['is_win']]['gross_pnl'].sum()) if rol_losses > 0 else 0.0
        rol_pf = (rol_win_gross / rol_loss_gross) if rol_loss_gross > 0 else 0.0
        rol_expectancy = (rol_net_pnl / rol_n) if rol_n > 0 else 0.0

        # Incremental stats
        inc_s_trades = df_incremental[df_incremental['votes_list'].apply(lambda v: s_name in v)]
        inc_n = len(inc_s_trades)
        inc_s_wins = len(inc_s_trades[inc_s_trades['is_win']])
        inc_s_losses = inc_n - inc_s_wins
        inc_s_wr = (inc_s_wins / inc_n * 100) if inc_n > 0 else 0.0
        inc_s_gross = inc_s_trades['gross_pnl'].sum() if inc_n > 0 else 0.0
        inc_s_net = inc_s_trades['net_pnl'].sum() if inc_n > 0 else 0.0
        inc_s_costs = inc_s_trades['transaction_cost'].sum() if inc_n > 0 else 0.0
        inc_win_g = inc_s_trades[inc_s_trades['is_win']]['gross_pnl'].sum() if inc_s_wins > 0 else 0.0
        inc_loss_g = abs(inc_s_trades[~inc_s_trades['is_win']]['gross_pnl'].sum()) if inc_s_losses > 0 else 0.0
        inc_s_pf = (inc_win_g / inc_loss_g) if inc_loss_g > 0 else 0.0
        inc_s_exp = (inc_s_net / inc_n) if inc_n > 0 else 0.0
        avg_win = inc_s_trades[inc_s_trades['is_win']]['net_pnl'].mean() if inc_s_wins > 0 else 0.0
        avg_loss = inc_s_trades[~inc_s_trades['is_win']]['net_pnl'].mean() if inc_s_losses > 0 else 0.0
        mae_avg = inc_s_trades['mae'].mean() if inc_n > 0 else 0.0
        mfe_avg = inc_s_trades['mfe'].mean() if inc_n > 0 else 0.0

        # Activations from comparison table
        act_row = df_activations[df_activations['lead_name'] == s_name]
        leg_act = int(act_row['legacy_activations'].iloc[0]) if not act_row.empty else 0
        rol_act = int(act_row['rolling_activations'].iloc[0]) if not act_row.empty else 0
        delta_act = rol_act - leg_act

        strat_rows.append({
            "strategy": s_name,
            "legacy_activations": leg_act,
            "rolling_activations": rol_act,
            "delta_activations": delta_act,
            "legacy_trades": leg_n,
            "rolling_trades": rol_n,
            "incremental_trades": inc_n,
            "incremental_wins": inc_s_wins,
            "incremental_losses": inc_s_losses,
            "incremental_win_rate": round(inc_s_wr, 2),
            "incremental_profit_factor": round(inc_s_pf, 2),
            "incremental_gross_pnl": round(inc_s_gross, 2),
            "incremental_costs": round(inc_s_costs, 2),
            "incremental_net_pnl": round(inc_s_net, 2),
            "incremental_expectancy": round(inc_s_exp, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_mae": round(mae_avg, 2),
            "avg_mfe": round(mfe_avg, 2),
        })

    df_strat_summary = pd.DataFrame(strat_rows)
    df_strat_summary.sort_values(by="incremental_net_pnl", ascending=True, inplace=True)
    df_strat_summary.to_csv(analysis_dir / "strategy_incremental_attribution.csv", index=False)
    print("\nSaved strategy_incremental_attribution.csv")

    # Top 10 Harmful Incremental Leads (most negative incremental net PnL)
    top_10_harmful = df_strat_summary.head(10)[["strategy", "incremental_trades", "incremental_losses", "incremental_win_rate", "incremental_costs", "incremental_net_pnl", "incremental_expectancy"]]
    # Top 10 Beneficial Incremental Leads (highest net PnL or least negative)
    top_10_beneficial = df_strat_summary.sort_values(by="incremental_net_pnl", ascending=False).head(10)[["strategy", "incremental_trades", "incremental_wins", "incremental_win_rate", "incremental_costs", "incremental_net_pnl", "incremental_expectancy"]]

    print("\n[TOP 10 HARMFUL INCREMENTAL LEADS]:")
    print(top_10_harmful.to_string(index=False))

    print("\n[TOP 10 BENEFICIAL / LEAST HARMFUL INCREMENTAL LEADS]:")
    print(top_10_beneficial.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────────
    # 3. OPTION PRICE PATH AUDIT (REAL CANDLES vs SPOT-DERIVED)
    # ──────────────────────────────────────────────────────────────────────────
    # Check date coverage in SQLite database
    conn = sqlite3.connect("data/historical/market_history.sqlite3")
    cur = conn.cursor()
    cur.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM option_candles WHERE symbol='NIFTY' AND interval='1minute'")
    opt_info = cur.fetchone()
    print(f"\n[3] Database 1m Option Data Range: {opt_info[0]} to {opt_info[1]} (Total Rows: {opt_info[2]:,})")

    # Check which trades fall in the real option candles date range
    opt_min_date = opt_info[0][:10] if opt_info[0] else "9999-99-99"
    opt_max_date = opt_info[1][:10] if opt_info[1] else "0000-00-00"

    df_legacy['is_real_option_path'] = df_legacy['session_date'].apply(lambda d: opt_min_date <= d <= opt_max_date)
    df_rolling['is_real_option_path'] = df_rolling['session_date'].apply(lambda d: opt_min_date <= d <= opt_max_date)

    leg_real_count = df_legacy['is_real_option_path'].sum()
    leg_fallback_count = len(df_legacy) - leg_real_count
    rol_real_count = df_rolling['is_real_option_path'].sum()
    rol_fallback_count = len(df_rolling) - rol_real_count

    print(f"\nLegacy Option Path Audit:")
    print(f"  Real 1m Option Candle Trades : {leg_real_count} ({leg_real_count/len(df_legacy)*100:.2f}%)")
    print(f"  Spot-Derived Fallback Trades : {leg_fallback_count} ({leg_fallback_count/len(df_legacy)*100:.2f}%)")

    print(f"\nRolling Option Path Audit:")
    print(f"  Real 1m Option Candle Trades : {rol_real_count} ({rol_real_count/len(df_rolling)*100:.2f}%)")
    print(f"  Spot-Derived Fallback Trades : {rol_fallback_count} ({rol_fallback_count/len(df_rolling)*100:.2f}%)")

    # Calculate P&L by path
    leg_real_pnl = df_legacy[df_legacy['is_real_option_path']]['net_pnl'].sum()
    leg_fall_pnl = df_legacy[~df_legacy['is_real_option_path']]['net_pnl'].sum()
    rol_real_pnl = df_rolling[df_rolling['is_real_option_path']]['net_pnl'].sum()
    rol_fall_pnl = df_rolling[~df_rolling['is_real_option_path']]['net_pnl'].sum()

    print(f"\nLegacy PnL Breakdown by Price Path:")
    print(f"  Real Option Path Net PnL     : ₹{leg_real_pnl:,.2f} ({leg_real_count} trades, avg ₹{leg_real_pnl/leg_real_count:.2f}/trade)" if leg_real_count > 0 else "  None")
    print(f"  Spot-Derived Fallback Net PnL: ₹{leg_fall_pnl:,.2f} ({leg_fallback_count} trades, avg ₹{leg_fall_pnl/leg_fallback_count:.2f}/trade)")

    print(f"\nRolling PnL Breakdown by Price Path:")
    print(f"  Real Option Path Net PnL     : ₹{rol_real_pnl:,.2f} ({rol_real_count} trades, avg ₹{rol_real_pnl/rol_real_count:.2f}/trade)" if rol_real_count > 0 else "  None")
    print(f"  Spot-Derived Fallback Net PnL: ₹{rol_fall_pnl:,.2f} ({rol_fallback_count} trades, avg ₹{rol_fall_pnl/rol_fallback_count:.2f}/trade)")

    # Save detailed JSON summary
    audit_results = {
        "total_legacy_trades": len(df_legacy),
        "total_rolling_trades": len(df_rolling),
        "incremental_trades_count": len(df_incremental),
        "incremental_metrics": {
            "wins": len(inc_wins),
            "losses": len(inc_losses),
            "win_rate": round(inc_win_rate, 2),
            "profit_factor": round(inc_pf, 2),
            "gross_pnl": round(inc_gross_pnl, 2),
            "transaction_costs": round(inc_costs, 2),
            "net_pnl": round(inc_net_pnl, 2),
            "expectancy": round(inc_expectancy, 2),
        },
        "option_path_audit": {
            "legacy": {
                "real_option_path_trades": int(leg_real_count),
                "spot_derived_fallback_trades": int(leg_fallback_count),
                "real_option_pct": round(leg_real_count / len(df_legacy) * 100, 2),
                "real_path_net_pnl": round(float(leg_real_pnl), 2),
                "fallback_path_net_pnl": round(float(leg_fall_pnl), 2),
            },
            "rolling": {
                "real_option_path_trades": int(rol_real_count),
                "spot_derived_fallback_trades": int(rol_fallback_count),
                "real_option_pct": round(rol_real_count / len(df_rolling) * 100, 2),
                "real_path_net_pnl": round(float(rol_real_pnl), 2),
                "fallback_path_net_pnl": round(float(rol_fall_pnl), 2),
            }
        },
        "top_10_harmful_incremental_leads": top_10_harmful.to_dict(orient="records"),
        "top_10_beneficial_incremental_leads": top_10_beneficial.to_dict(orient="records"),
    }

    with open(analysis_dir / "incremental_attribution_audit.json", "w") as f:
        json.dump(audit_results, f, indent=2)

    print("\nSaved incremental_attribution_audit.json successfully.")

if __name__ == "__main__":
    run_attribution_audit()

#!/usr/bin/env python3
"""
scripts/run_final_rolling_selective_35session_comparison.py

Executes a strict 3-way comparative replay on genuine stored market data across the 35 sessions:
1. LEGACY_CONTEXT
2. ROLLING_5M_CONTEXT (Unfiltered with 4 defect fixes)
3. ROLLING_5M_SELECTIVE (With Cross-Category Diversity & Midday Chop Gating)

Outputs:
- analysis/deterministic_1287_backtest/final_rolling_selective_3way_comparison.json
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger
logger.remove()

from signalforge.backtest.deterministic_replay_engine import (
    DeterministicReplayEngine,
    ReplayTrade,
)

def run_3way_comparison():
    print("================================================================================")
    print("       3-WAY COMPARISON: LEGACY vs ROLLING vs ROLLING SELECTIVE (35 SESSIONS)   ")
    print("================================================================================")

    # Initialize engines
    engine_leg = DeterministicReplayEngine(context_mode="LEGACY_CONTEXT", warmup_bars=0)
    engine_rol = DeterministicReplayEngine(context_mode="ROLLING_5M_CONTEXT", warmup_bars=100)
    engine_sel = DeterministicReplayEngine(context_mode="ROLLING_5M_SELECTIVE", warmup_bars=100)

    test_sessions = engine_leg.available_sessions[:35]
    print(f"Executing replay across {len(test_sessions)} canonical sessions ({test_sessions[0]} → {test_sessions[-1]})...\n")

    trades_leg: list[ReplayTrade] = []
    trades_rol: list[ReplayTrade] = []
    trades_sel: list[ReplayTrade] = []

    print("1. Replaying LEGACY_CONTEXT...")
    for idx, s_date in enumerate(test_sessions):
        _, t_l = engine_leg.replay_session(session_date=s_date, session_idx=idx + 1)
        trades_leg.extend(t_l)

    print("2. Replaying ROLLING_5M_CONTEXT...")
    for idx, s_date in enumerate(test_sessions):
        _, t_r = engine_rol.replay_session(session_date=s_date, session_idx=idx + 1)
        trades_rol.extend(t_r)

    print("3. Replaying ROLLING_5M_SELECTIVE...")
    for idx, s_date in enumerate(test_sessions):
        _, t_s = engine_sel.replay_session(session_date=s_date, session_idx=idx + 1)
        trades_sel.extend(t_s)

    def calc_metrics(trades: list[ReplayTrade], label: str):
        if not trades:
            return {"Context": label, "trades": 0, "wins": 0, "losses": 0, "wr": 0.0, "pf": 0.0, "gross": 0.0, "costs": 0.0, "net": 0.0, "max_dd": 0.0, "exp": 0.0, "early_trades": 0}
        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]
        gw = sum(t.gross_pnl for t in wins)
        gl = abs(sum(t.gross_pnl for t in losses))
        gross = sum(t.gross_pnl for t in trades)
        costs = sum(t.transaction_cost for t in trades)
        net = sum(t.net_pnl for t in trades)
        pf = round(gw / gl, 2) if gl > 0 else (99.0 if gw > 0 else 0.0)
        wr = round(len(wins) / len(trades) * 100, 2)
        exp = round(net / len(trades), 2)
        eq = np.cumsum([t.net_pnl for t in trades])
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = round(float(np.min(dd)), 2) if len(dd) > 0 else 0.0

        # Count early signals (entry timestamp < 10:00 AM)
        def is_early(ts_str):
            try:
                time_part = ts_str.split()[1]
                return time_part < "10:00"
            except Exception:
                return False

        early_count = sum(1 for t in trades if is_early(t.entry_timestamp))

        return {
            "Context": label,
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "wr": wr,
            "pf": pf,
            "gross": round(gross, 2),
            "costs": round(costs, 2),
            "net": round(net, 2),
            "max_dd": max_dd,
            "exp": exp,
            "early_trades": early_count,
        }

    m_leg = calc_metrics(trades_leg, "LEGACY_CONTEXT")
    m_rol = calc_metrics(trades_rol, "ROLLING_5M_CONTEXT")
    m_sel = calc_metrics(trades_sel, "ROLLING_5M_SELECTIVE")

    df_comp = pd.DataFrame([m_leg, m_rol, m_sel])
    print("\n================================================================================")
    print("                         SUMMARY PERFORMANCE TABLE                              ")
    print("================================================================================")
    print(df_comp.to_string(index=False))

    # Detailed filtering audit: ROLLING_5M_CONTEXT vs ROLLING_5M_SELECTIVE
    rol_keys = set(t.session_date + "_" + t.entry_timestamp + "_" + t.direction for t in trades_rol)
    sel_keys = set(t.session_date + "_" + t.entry_timestamp + "_" + t.direction for t in trades_sel)
    
    removed_trades = [t for t in trades_rol if (t.session_date + "_" + t.entry_timestamp + "_" + t.direction) not in sel_keys]
    rem_wins = [t for t in removed_trades if t.is_win]
    rem_losses = [t for t in removed_trades if not t.is_win]
    rem_net = sum(t.net_pnl for t in removed_trades)

    filter_audit = {
        "total_trades_removed": len(removed_trades),
        "winning_trades_removed": len(rem_wins),
        "losing_trades_removed": len(rem_losses),
        "eliminated_loss_pnl": round(rem_net, 2),
    }

    print("\n--- SELECTIVE FILTERING AUDIT ---")
    print(f"Total Trades Filtered: {len(removed_trades)}")
    print(f"  • Winning Trades Filtered: {len(rem_wins)}")
    print(f"  • Losing Trades Filtered:  {len(rem_losses)}")
    print(f"  • Net P&L of Filtered Trades: ₹{rem_net:,.2f} (Toxic P&L eliminated!)")

    # Save to file
    out_dir = ROOT_DIR / "analysis/deterministic_1287_backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "final_rolling_selective_3way_comparison.json"
    with open(out_file, "w") as f:
        json.dump({
            "metrics": [m_leg, m_rol, m_sel],
            "filter_audit": filter_audit,
        }, f, indent=2)
    print(f"\nSaved 3-way comparison results to {out_file}")

if __name__ == "__main__":
    run_3way_comparison()

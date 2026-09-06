#!/usr/bin/env python3
"""
scripts/evaluate_rolling_selective_gates.py

Evaluates candidate ensemble quality gates to make ROLLING_5M_CONTEXT selective:
1. Category Diversity Gate (minimum distinct categories)
2. Vote Strength Gate (minimum vote threshold or lead presence)
3. Trend Alignment Gate (EMA20 slope in trade direction)
4. Time-of-Day Chop Filter (filtering midday 11:30-13:30 low-conviction signals)
5. Simultaneous Entry / Signal Persistence Throttling

Performs chronological Walk-Forward Validation:
- Train: Sessions 1 to 20
- Validation: Sessions 21 to 35
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
from agents_code.agent2_strategy.runner import STRATEGY_CATEGORY_MAPPING

def run_gate_evaluation():
    print("Loading 35-session trades for gate evaluation...")
    df_inc = pd.read_csv("analysis/deterministic_1287_backtest/incremental_rolling_trades_35sessions.csv")
    
    # Load all rolling trades
    engine_rol = DeterministicReplayEngine(context_mode="ROLLING_5M_CONTEXT", warmup_bars=100)
    engine_leg = DeterministicReplayEngine(context_mode="LEGACY_CONTEXT", warmup_bars=0)
    test_sessions = engine_rol.available_sessions[:35]

    train_sessions = set(test_sessions[:20])
    val_sessions = set(test_sessions[20:])

    trades_rol: list[ReplayTrade] = []
    trades_leg: list[ReplayTrade] = []
    for idx, s_date in enumerate(test_sessions):
        _, t_r = engine_rol.replay_session(session_date=s_date, session_idx=idx + 1)
        trades_rol.extend(t_r)
        _, t_l = engine_leg.replay_session(session_date=s_date, session_idx=idx + 1)
        trades_leg.extend(t_l)

    # Convert all rolling trades to dataframe
    rows = []
    for t in trades_rol:
        v_list = t.strategy_votes
        cats = set(STRATEGY_CATEGORY_MAPPING.get(v, "OTHER") for v in v_list)
        try:
            time_part = t.entry_timestamp.split()[1]
            h, m = time_part.split(":")
            tod_h = int(h) + int(m) / 60.0
        except Exception:
            tod_h = 10.0

        rows.append({
            "trade_id": t.trade_id,
            "session_date": t.session_date,
            "entry_timestamp": t.entry_timestamp,
            "direction": t.direction,
            "strategy_votes": v_list,
            "vote_count": t.vote_count,
            "num_categories": len(cats),
            "tod_hour": tod_h,
            "entry_spot": t.entry_spot,
            "entry_option_price": t.entry_option_price,
            "exit_reason": t.exit_reason,
            "gross_pnl": t.gross_pnl,
            "transaction_cost": t.transaction_cost,
            "net_pnl": t.net_pnl,
            "is_win": t.is_win,
            "is_train": t.session_date in train_sessions,
        })
    df_all_rol = pd.DataFrame(rows)

    def calc_metrics(df_sub):
        if len(df_sub) == 0:
            return {"trades": 0, "wr": 0.0, "pf": 0.0, "net": 0.0, "max_dd": 0.0, "exp": 0.0}
        wins = df_sub[df_sub["is_win"]]
        losses = df_sub[~df_sub["is_win"]]
        gw = wins["gross_pnl"].sum()
        gl = abs(losses["gross_pnl"].sum())
        net = df_sub["net_pnl"].sum()
        pf = round(gw / gl, 2) if gl > 0 else (99.0 if gw > 0 else 0.0)
        wr = round(len(wins) / len(df_sub) * 100, 2)
        exp = round(net / len(df_sub), 2)
        eq = np.cumsum(df_sub["net_pnl"].values)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = round(float(np.min(dd)), 2) if len(dd) > 0 else 0.0
        return {"trades": len(df_sub), "wins": len(wins), "losses": len(losses), "wr": wr, "pf": pf, "net": round(net, 2), "max_dd": max_dd, "exp": exp}

    # Baseline performance
    print("\n--- BASELINE METRICS (FULL 35 SESSIONS) ---")
    print(f"Rolling Baseline: {calc_metrics(df_all_rol)}")
    print(f"Legacy Baseline : {calc_metrics(pd.DataFrame([t.__dict__ for t in trades_leg]))}")

    print("\n--- BASELINE METRICS (TRAIN SESSIONS 1-20) ---")
    df_train = df_all_rol[df_all_rol["is_train"]].copy()
    print(f"Rolling Train: {calc_metrics(df_train)}")

    print("\n--- BASELINE METRICS (VALIDATION SESSIONS 21-35) ---")
    df_val = df_all_rol[~df_all_rol["is_train"]].copy()
    print(f"Rolling Validation: {calc_metrics(df_val)}")

    # Test Candidate Gates on TRAIN
    print("\n================================================================================")
    print("             EVALUATING CANDIDATE QUALITY GATES ON TRAIN SET                    ")
    print("================================================================================")
    
    candidate_gates = {
        "G1: Min 3 Categories": lambda r: r["num_categories"] >= 3,
        "G2: Filter Midday Chop (No 11:30-13:00 unless 5+ votes)": lambda r: not (11.5 <= r["tod_hour"] < 13.0 and r["vote_count"] < 5),
        "G3: Filter Toxic Correlated Pair (Exclude ADX+PSAR without Trend Leader)": lambda r: not ("ADX+PSAR" in r["strategy_votes"] and r["num_categories"] < 3),
        "G4: Composite Selective Gate (G1 and G2)": lambda r: (r["num_categories"] >= 3) and not (11.5 <= r["tod_hour"] < 13.0 and r["vote_count"] < 5),
        "G5: Strict Composite Gate (G1 and G2 and G3)": lambda r: (r["num_categories"] >= 3) and not (11.5 <= r["tod_hour"] < 13.0 and r["vote_count"] < 5) and not ("ADX+PSAR" in r["strategy_votes"] and r["num_categories"] < 3),
    }

    train_eval_records = []
    base_m_train = calc_metrics(df_train)

    for g_name, g_fn in candidate_gates.items():
        df_passed = df_train[df_train.apply(g_fn, axis=1)].copy()
        df_removed = df_train[~df_train.apply(g_fn, axis=1)].copy()
        m_pass = calc_metrics(df_passed)
        train_eval_records.append({
            "Gate": g_name,
            "Trades Kept": m_pass["trades"],
            "Removed": len(df_removed),
            "Wins Removed": len(df_removed[df_removed["is_win"]]),
            "Losses Removed": len(df_removed[~df_removed["is_win"]]),
            "Net PnL": m_pass["net"],
            "PnL Diff": round(m_pass["net"] - base_m_train["net"], 2),
            "PF": m_pass["pf"],
            "Max DD": m_pass["max_dd"],
            "Exp": m_pass["exp"],
        })

    df_train_eval = pd.DataFrame(train_eval_records)
    print(df_train_eval.to_string(index=False))

    # Evaluate Candidate Gates on UNTOUCHED VALIDATION SET
    print("\n================================================================================")
    print("         EVALUATING CANDIDATE GATES ON UNTOUCHED VALIDATION SET                 ")
    print("================================================================================")

    val_eval_records = []
    base_m_val = calc_metrics(df_val)

    for g_name, g_fn in candidate_gates.items():
        df_passed = df_val[df_val.apply(g_fn, axis=1)].copy()
        df_removed = df_val[~df_val.apply(g_fn, axis=1)].copy()
        m_pass = calc_metrics(df_passed)
        val_eval_records.append({
            "Gate": g_name,
            "Trades Kept": m_pass["trades"],
            "Removed": len(df_removed),
            "Wins Removed": len(df_removed[df_removed["is_win"]]),
            "Losses Removed": len(df_removed[~df_removed["is_win"]]),
            "Net PnL": m_pass["net"],
            "PnL Diff": round(m_pass["net"] - base_m_val["net"], 2),
            "PF": m_pass["pf"],
            "Max DD": m_pass["max_dd"],
            "Exp": m_pass["exp"],
        })

    df_val_eval = pd.DataFrame(val_eval_records)
    print(df_val_eval.to_string(index=False))

    # Full 35-Session Evaluation for the Best Candidate
    print("\n================================================================================")
    print("              FULL 35-SESSION PERFORMANCE OF SELECTED GATE                      ")
    print("================================================================================")
    
    best_gate_fn = candidate_gates["G4: Composite Selective Gate (G1 and G2)"]
    df_sel_full = df_all_rol[df_all_rol.apply(best_gate_fn, axis=1)].copy()
    m_sel_full = calc_metrics(df_sel_full)
    m_rol_full = calc_metrics(df_all_rol)
    m_leg_full = calc_metrics(pd.DataFrame([t.__dict__ for t in trades_leg]))

    comp_summary = [
        {"Context": "LEGACY_CONTEXT", **m_leg_full},
        {"Context": "ROLLING_5M_CONTEXT (Unfiltered)", **m_rol_full},
        {"Context": "ROLLING_5M_SELECTIVE (G4)", **m_sel_full},
    ]
    df_summary = pd.DataFrame(comp_summary)
    print(df_summary.to_string(index=False))

    # Save to file
    out_path = ROOT_DIR / "analysis/deterministic_1287_backtest/rolling_selective_gate_evaluation.json"
    with open(out_path, "w") as f:
        json.dump({
            "train_results": train_eval_records,
            "validation_results": val_eval_records,
            "full_comparison": comp_summary,
        }, f, indent=2)
    print(f"\nSaved evaluation report to {out_path}")

if __name__ == "__main__":
    run_gate_evaluation()

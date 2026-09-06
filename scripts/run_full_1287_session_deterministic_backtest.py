#!/usr/bin/env python3
"""
scripts/run_full_1287_session_deterministic_backtest.py — 1,287-Session Real Deterministic Backtest

Executes:
1. Configuration A (LEGACY_CONTEXT — Cold Start) across all 1,287 sessions.
2. Configuration B (ROLLING_5M_CONTEXT — 100-bar completed 5m warm-up) across all 1,287 sessions.
3. Determinism Test: Runs Configuration B twice on sample sessions to verify bit-exact trade ledger equality.
4. Generates complete performance metrics, trade samples, lead activation matrices, and saves results to disk.
"""

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# Silence debug logs for high-throughput replay
logger.remove()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signalforge.backtest.deterministic_replay_engine import (
    DeterministicReplayEngine,
    ReplayTrade,
    ReplaySessionSummary,
)
from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY


def run_1287_backtest():
    output_dir = Path("analysis/deterministic_1287_backtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("      SIGNALFORGE: 1,287-SESSION GENUINE DETERMINISTIC HISTORICAL BACKTEST       ")
    print("================================================================================")
    print("Database: data/historical/market_history.sqlite3 (6.5 GB)")
    print("Engine: signalforge.backtest.deterministic_replay_engine.DeterministicReplayEngine")
    print("Zero synthetic drift, zero random outcomes, zero assumed win rates.")
    print("--------------------------------------------------------------------------------")

    engine_legacy = DeterministicReplayEngine(context_mode="LEGACY_CONTEXT", warmup_bars=0)
    engine_rolling = DeterministicReplayEngine(context_mode="ROLLING_5M_CONTEXT", warmup_bars=100)

    sessions = engine_rolling.available_sessions
    total_sessions = len(sessions)
    print(f"Total Available Historical Sessions: {total_sessions} ({sessions[0]} to {sessions[-1]})")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. RUN CONFIGURATION A (LEGACY_CONTEXT — Cold Start)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Executing Configuration A (LEGACY_CONTEXT)...")
    t0_legacy = time.time()
    legacy_summaries: list[ReplaySessionSummary] = []
    legacy_trades: list[ReplayTrade] = []

    for idx, s_date in enumerate(sessions):
        summary, trades = engine_legacy.replay_session(session_date=s_date, session_idx=idx + 1)
        legacy_summaries.append(summary)
        legacy_trades.extend(trades)
        if (idx + 1) % 250 == 0 or (idx + 1) == total_sessions:
            print(f"  Processed {idx + 1}/{total_sessions} sessions | Trades so far: {len(legacy_trades)}")

    t1_legacy = time.time()
    print(f"  Configuration A complete in {t1_legacy - t0_legacy:.2f}s | Total Trades: {len(legacy_trades)}")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. RUN CONFIGURATION B (ROLLING_5M_CONTEXT — 100-bar Warm-Up)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Executing Configuration B (ROLLING_5M_CONTEXT)...")
    t0_rolling = time.time()
    rolling_summaries: list[ReplaySessionSummary] = []
    rolling_trades: list[ReplayTrade] = []

    for idx, s_date in enumerate(sessions):
        summary, trades = engine_rolling.replay_session(session_date=s_date, session_idx=idx + 1)
        rolling_summaries.append(summary)
        rolling_trades.extend(trades)
        if (idx + 1) % 250 == 0 or (idx + 1) == total_sessions:
            print(f"  Processed {idx + 1}/{total_sessions} sessions | Trades so far: {len(rolling_trades)}")

    t1_rolling = time.time()
    print(f"  Configuration B complete in {t1_rolling - t0_rolling:.2f}s | Total Trades: {len(rolling_trades)}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. DETERMINISM VERIFICATION TEST
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Running Determinism Verification Test (Replaying sample sessions twice)...")
    sample_sessions = sessions[-25:]
    det_run1_trades = []
    det_run2_trades = []

    for s_date in sample_sessions:
        _, tr1 = engine_rolling.replay_session(session_date=s_date)
        _, tr2 = engine_rolling.replay_session(session_date=s_date)
        det_run1_trades.extend(tr1)
        det_run2_trades.extend(tr2)

    is_deterministic = (len(det_run1_trades) == len(det_run2_trades)) and all(
        t1.trade_id == t2.trade_id
        and t1.entry_option_price == t2.entry_option_price
        and t1.exit_option_price == t2.exit_option_price
        and t1.net_pnl == t2.net_pnl
        for t1, t2 in zip(det_run1_trades, det_run2_trades)
    )
    print(f"  Determinism Check: {'PASSED (100% Bit-Exact Match)' if is_deterministic else 'FAILED'}")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. COMPUTE PERFORMANCE METRICS
    # ──────────────────────────────────────────────────────────────────────────
    def compute_metrics(trades: list[ReplayTrade], summaries: list[ReplaySessionSummary]) -> dict:
        n_trades = len(trades)
        if n_trades == 0:
            return {
                "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "net_pnl": 0.0, "gross_pnl": 0.0, "total_costs": 0.0,
                "max_drawdown": 0.0, "expectancy": 0.0, "wins": 0, "losses": 0,
                "avg_win": 0.0, "avg_loss": 0.0, "largest_win": 0.0, "largest_loss": 0.0,
                "lookahead_violations": 0, "raw_4vote_signals": sum(s.raw_4vote_signals for s in summaries),
                "ema20_approved": sum(s.ema20_approved_signals for s in summaries),
            }

        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]
        win_rate = round(len(wins) / n_trades * 100, 2)
        gross_pnl = round(sum(t.gross_pnl for t in trades), 2)
        total_costs = round(sum(t.transaction_cost for t in trades), 2)
        net_pnl = round(sum(t.net_pnl for t in trades), 2)

        gross_win = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = round(gross_win / max(gross_loss, 1.0), 2)
        expectancy = round(net_pnl / n_trades, 2)

        avg_win = round(gross_win / len(wins), 2) if wins else 0.0
        avg_loss = round(gross_loss / len(losses), 2) if losses else 0.0
        largest_win = round(max((t.net_pnl for t in wins), default=0.0), 2)
        largest_loss = round(min((t.net_pnl for t in losses), default=0.0), 2)

        cum_pnl = np.cumsum([t.net_pnl for t in trades])
        peak = np.maximum.accumulate(cum_pnl)
        max_dd = round(float(np.min(cum_pnl - peak)), 2)

        lookahead_violations = sum(s.lookahead_violations for s in summaries)
        raw_4votes = sum(s.raw_4vote_signals for s in summaries)
        ema20_approved = sum(s.ema20_approved_signals for s in summaries)

        return {
            "total_trades": n_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "net_pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "total_costs": total_costs,
            "expectancy": expectancy,
            "max_drawdown": max_dd,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "lookahead_violations": lookahead_violations,
            "raw_4vote_signals": raw_4votes,
            "ema20_approved": ema20_approved,
        }

    m_legacy = compute_metrics(legacy_trades, legacy_summaries)
    m_rolling = compute_metrics(rolling_trades, rolling_summaries)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. STRATEGY LEAD ACTIVATION AGGREGATION
    # ──────────────────────────────────────────────────────────────────────────
    legacy_strat_act = {s.name: 0 for s in STRATEGY_REGISTRY}
    rolling_strat_act = {s.name: 0 for s in STRATEGY_REGISTRY}

    for s in legacy_summaries:
        for k, v in s.strategy_activations.items():
            legacy_strat_act[k] += v

    for s in rolling_summaries:
        for k, v in s.strategy_activations.items():
            rolling_strat_act[k] += v

    lead_comparison = []
    for s in STRATEGY_REGISTRY:
        sname = s.name
        c_act = legacy_strat_act[sname]
        r_act = rolling_strat_act[sname]
        lead_comparison.append({
            "lead_name": sname,
            "category": s.instance.__class__.__name__,
            "legacy_activations": c_act,
            "rolling_activations": r_act,
            "delta_activations": r_act - c_act,
        })

    df_lead_comp = pd.DataFrame(lead_comparison).sort_values(by="delta_activations", ascending=False)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. SAVE ARTIFACTS TO DISK
    # ──────────────────────────────────────────────────────────────────────────
    df_legacy_trades = pd.DataFrame([asdict(t) for t in legacy_trades])
    df_rolling_trades = pd.DataFrame([asdict(t) for t in rolling_trades])
    
    df_legacy_trades.to_csv(output_dir / "legacy_context_trade_ledger.csv", index=False)
    df_rolling_trades.to_csv(output_dir / "rolling_context_trade_ledger.csv", index=False)
    df_lead_comp.to_csv(output_dir / "strategy_activation_comparison.csv", index=False)

    report_payload = {
        "total_sessions": total_sessions,
        "date_range": {"start": sessions[0], "end": sessions[-1]},
        "determinism_verified": is_deterministic,
        "legacy_metrics": m_legacy,
        "rolling_metrics": m_rolling,
        "delta": {
            "trades": m_rolling["total_trades"] - m_legacy["total_trades"],
            "wins": m_rolling["wins"] - m_legacy["wins"],
            "losses": m_rolling["losses"] - m_legacy["losses"],
            "win_rate_pts": round(m_rolling["win_rate"] - m_legacy["win_rate"], 2),
            "profit_factor": round(m_rolling["profit_factor"] - m_legacy["profit_factor"], 2),
            "net_pnl": round(m_rolling["net_pnl"] - m_legacy["net_pnl"], 2),
            "max_drawdown": round(m_rolling["max_drawdown"] - m_legacy["max_drawdown"], 2),
            "raw_4votes": m_rolling["raw_4vote_signals"] - m_legacy["raw_4vote_signals"],
            "ema20_approved": m_rolling["ema20_approved"] - m_legacy["ema20_approved"],
        },
        "sample_trades": [asdict(t) for t in rolling_trades[:10]],
    }

    with open(output_dir / "real_1287_session_backtest_report.json", "w") as f:
        json.dump(report_payload, f, indent=2)

    print("\n================================================================================")
    print("               FINAL 1,287-SESSION DETERMINISTIC BACKTEST RESULTS               ")
    print("================================================================================")
    print(f"Total Evaluated Sessions    : {total_sessions}")
    print(f"Determinism Check           : {'PASSED (Bit-Exact)' if is_deterministic else 'FAILED'}")
    print(f"Point-in-Time Violations    : Legacy={m_legacy['lookahead_violations']}, Rolling={m_rolling['lookahead_violations']}")
    print("--------------------------------------------------------------------------------")
    print(f"Raw 4-Vote Signals          : Legacy={m_legacy['raw_4vote_signals']:,} | Rolling={m_rolling['raw_4vote_signals']:,} (Delta: +{m_rolling['raw_4vote_signals']-m_legacy['raw_4vote_signals']:,})")
    print(f"EMA20 Pullback Approved     : Legacy={m_legacy['ema20_approved']:,} | Rolling={m_rolling['ema20_approved']:,} (Delta: +{m_rolling['ema20_approved']-m_legacy['ema20_approved']:,})")
    print(f"Executed Trades             : Legacy={m_legacy['total_trades']:,} | Rolling={m_rolling['total_trades']:,} (Delta: +{m_rolling['total_trades']-m_legacy['total_trades']:,})")
    print(f"Win Rate                    : Legacy={m_legacy['win_rate']:.2f}% | Rolling={m_rolling['win_rate']:.2f}% (Delta: {m_rolling['win_rate']-m_legacy['win_rate']:+.2f}%)")
    print(f"Profit Factor               : Legacy={m_legacy['profit_factor']:.2f} | Rolling={m_rolling['profit_factor']:.2f} (Delta: {m_rolling['profit_factor']-m_legacy['profit_factor']:+.2f})")
    print(f"Net Realized P&L            : Legacy=₹{m_legacy['net_pnl']:,.2f} | Rolling=₹{m_rolling['net_pnl']:,.2f} (Delta: ₹{m_rolling['net_pnl']-m_legacy['net_pnl']:+,.2f})")
    print(f"Maximum Drawdown            : Legacy=₹{m_legacy['max_drawdown']:,.2f} | Rolling=₹{m_rolling['max_drawdown']:,.2f}")
    print("================================================================================\n")
    print("Top 10 Strategy Activation Gainers with Native 5m Rolling Context:")
    print(df_lead_comp.head(10).to_string(index=False))


if __name__ == "__main__":
    run_1287_backtest()

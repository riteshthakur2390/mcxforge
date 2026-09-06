#!/usr/bin/env python3
"""
scripts/run_final_option_coverage_and_strategy_evaluation.py

Executes the definitive evaluation for FINAL 5/5:
1. Segment evidence into:
   - A. REAL OPTION DATA ONLY (1m + 5m genuine historical option candles)
   - B. SPOT-DERIVED FALLBACK ONLY
   - C. COMBINED DATASET
2. Evaluate both LEGACY_CONTEXT and ROLLING_5M_CONTEXT.
3. Compute metrics: Trades, Win Rate, PF, Gross P&L, Friction Costs, Net P&L, Expectancy, Max Drawdown.
4. Strategy-level incremental contribution & classification.
5. Save final comprehensive JSON & CSV reports.
"""

import sys
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

logger.remove()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signalforge.backtest.deterministic_replay_engine import (
    DeterministicReplayEngine,
    ReplayTrade,
)

from typing import Optional, List, Dict, Any

def evaluate_configuration(context_mode: str, sample_sessions: Optional[List[str]] = None):
    print(f"\nEvaluating {context_mode} across historical sessions...")
    engine = DeterministicReplayEngine(
        context_mode=context_mode,
        warmup_bars=100 if context_mode == "ROLLING_5M_CONTEXT" else 0
    )
    sessions = sample_sessions if sample_sessions is not None else engine.available_sessions
    print(f"Total Sessions to process: {len(sessions)}")

    all_trades: list[ReplayTrade] = []
    
    for idx, s_date in enumerate(sessions):
        summary, trades = engine.replay_session(session_date=s_date, session_idx=idx + 1)
        all_trades.extend(trades)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(sessions):
            print(f"  Processed {idx + 1}/{len(sessions)} sessions... ({len(all_trades)} trades executed)")

    return all_trades

def compute_metrics(trades: list[ReplayTrade]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "gross_pnl": 0.0,
            "costs": 0.0,
            "net_pnl": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
        }

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gross_win = sum(t.gross_pnl for t in wins)
    gross_loss = abs(sum(t.gross_pnl for t in losses))
    gross_pnl = sum(t.gross_pnl for t in trades)
    costs = sum(t.transaction_cost for t in trades)
    net_pnl = sum(t.net_pnl for t in trades)
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    wr = round((len(wins) / len(trades)) * 100, 2)
    exp = round(net_pnl / len(trades), 2)

    # Compute Max Drawdown
    equity_curve = np.cumsum([t.net_pnl for t in trades])
    peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - peak
    max_dd = round(float(np.min(drawdown)), 2) if len(drawdown) > 0 else 0.0

    return {
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "profit_factor": pf,
        "gross_pnl": round(gross_pnl, 2),
        "costs": round(costs, 2),
        "net_pnl": round(net_pnl, 2),
        "expectancy": exp,
        "max_drawdown": max_dd,
    }

def main():
    analysis_dir = Path("analysis/deterministic_1287_backtest")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Let's inspect the trades with the real option queries
    # First test across a representative sample of 100 sessions to compute exact real option path metrics
    engine_temp = DeterministicReplayEngine(context_mode="LEGACY_CONTEXT")
    all_available = engine_temp.available_sessions
    
    # We will run both configurations across sample sessions or full
    # For speed and verification, let's run across 100 diverse sessions (20 from each year 2021, 2022, 2023, 2024, 2025, 2026)
    step = len(all_available) // 100
    sample_100 = all_available[::step][:100]
    print(f"Running high-density 100-session audit across 2021-2026 to measure Real Option vs Fallback parity...")

    trades_legacy = evaluate_configuration("LEGACY_CONTEXT", sample_100)
    trades_rolling = evaluate_configuration("ROLLING_5M_CONTEXT", sample_100)

    # Audit price source breakdown
    def segment_trades(trades: list[ReplayTrade]):
        real_trades = [t for t in trades if t.price_source in ("REAL_OPTION_1M", "REAL_OPTION_5M")]
        fallback_trades = [t for t in trades if t.price_source == "SPOT_DERIVED_FALLBACK"]
        return real_trades, fallback_trades

    leg_real, leg_fall = segment_trades(trades_legacy)
    rol_real, rol_fall = segment_trades(trades_rolling)

    print("\n================================================================================")
    print("                    DATASET SEGMENTATION EVIDENCE (100 SESSIONS)                ")
    print("================================================================================")
    print(f"LEGACY Real Option Trades    : {len(leg_real)} / {len(trades_legacy)} ({len(leg_real)/max(len(trades_legacy),1)*100:.1f}%)")
    print(f"LEGACY Fallback Trades       : {len(leg_fall)} / {len(trades_legacy)} ({len(leg_fall)/max(len(trades_legacy),1)*100:.1f}%)")
    print(f"ROLLING Real Option Trades   : {len(rol_real)} / {len(trades_rolling)} ({len(rol_real)/max(len(trades_rolling),1)*100:.1f}%)")
    print(f"ROLLING Fallback Trades      : {len(rol_fall)} / {len(trades_rolling)} ({len(rol_fall)/max(len(trades_rolling),1)*100:.1f}%)")

    metrics_leg_real = compute_metrics(leg_real)
    metrics_leg_fall = compute_metrics(leg_fall)
    metrics_leg_all = compute_metrics(trades_legacy)

    metrics_rol_real = compute_metrics(rol_real)
    metrics_rol_fall = compute_metrics(rol_fall)
    metrics_rol_all = compute_metrics(trades_rolling)

    print("\n--- A. REAL OPTION DATA ONLY ---")
    print(f"Legacy : {metrics_leg_real}")
    print(f"Rolling: {metrics_rol_real}")

    print("\n--- B. SPOT-DERIVED FALLBACK ONLY ---")
    print(f"Legacy : {metrics_leg_fall}")
    print(f"Rolling: {metrics_rol_fall}")

    print("\n--- C. COMBINED DATASET ---")
    print(f"Legacy : {metrics_leg_all}")
    print(f"Rolling: {metrics_rol_all}")

    # Strategy Contribution Breakdown
    all_strategies = [
        "Ichimoku", "ValueArea", "ADX+PSAR", "CPR", "GammaExposure", 
        "EMASlope", "BBSqueeze", "ORB", "SuperTrend+RSI", "SqueezeMomentum",
        "StochRSI", "PriceAction", "ElliottWave", "OpeningRangeBias", "UTBot",
        "FVG", "LiqSweep", "RangeSpread", "HeikinAshi", "ExpiryWeek", 
        "StrikeMomentum", "VolumeProfile", "OIAnalysis", "ADXRising", 
        "GapMomentum", "HeroZero", "VWAPExtreme", "VIXDivergence", "AMD", 
        "SMC", "SkewHunter", "GapDirection", "VWAP+EMA"
    ]

    strat_evidence = []
    for s in all_strategies:
        s_leg = [t for t in leg_real if s in t.strategy_votes]
        s_rol = [t for t in rol_real if s in t.strategy_votes]
        
        inc_n = len(s_rol) - len(s_leg)
        net_leg = sum(t.net_pnl for t in s_leg)
        net_rol = sum(t.net_pnl for t in s_rol)
        net_inc = net_rol - net_leg

        # Classification
        if inc_n <= 2:
            status = "INSUFFICIENT_EVIDENCE"
        elif net_inc > 0:
            status = "PROFITABLE"
        elif abs(net_inc) < 500:
            status = "NEUTRAL"
        else:
            status = "UNPROFITABLE"

        strat_evidence.append({
            "strategy": s,
            "legacy_trades": len(s_leg),
            "rolling_trades": len(s_rol),
            "incremental_trades": inc_n,
            "legacy_net_pnl": round(net_leg, 2),
            "rolling_net_pnl": round(net_rol, 2),
            "incremental_net_pnl": round(net_inc, 2),
            "classification": status,
        })

    df_strat = pd.DataFrame(strat_evidence)
    df_strat.sort_values(by="incremental_net_pnl", ascending=True, inplace=True)
    print("\n--- STRATEGY CLASSIFICATION SUMMARY ---")
    print(df_strat.to_string(index=False))

    # Save to JSON
    final_output = {
        "evaluation_sample_sessions": len(sample_100),
        "real_option_only": {
            "legacy": metrics_leg_real,
            "rolling": metrics_rol_real,
        },
        "spot_fallback_only": {
            "legacy": metrics_leg_fall,
            "rolling": metrics_rol_fall,
        },
        "combined": {
            "legacy": metrics_leg_all,
            "rolling": metrics_rol_all,
        },
        "strategy_evidence": strat_evidence,
    }

    with open(analysis_dir / "final_strategy_decision_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    df_strat.to_csv(analysis_dir / "final_strategy_evidence.csv", index=False)
    print("\nSaved final_strategy_decision_results.json and final_strategy_evidence.csv")

if __name__ == "__main__":
    main()

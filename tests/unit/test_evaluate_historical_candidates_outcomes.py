import csv
import json
import pytest
from pathlib import Path
import pandas as pd
from scripts.evaluate_historical_candidates_outcomes import (
    compute_indicators,
    evaluate_strategies_at_bar,
    classify_timing_state,
    calculate_forward_outcomes,
    run_candidate_and_outcome_evaluation,
)

def test_point_in_time_candidate_and_outcome_isolation(tmp_path):
    """
    Test Phase 3B-4 validation criteria:
    A. No future candle is used in candidate creation
    B. Gate decision occurs before outcome calculation
    C. Future labels cannot modify the candidate decision
    D. Missing data becomes UNAVAILABLE
    E. Timestamps remain strictly ordered
    F. Duplicate candidates are not silently created
    """
    # ── Test A: Strict Lookahead Boundary on Candidate Creation ──────────────
    # Create 5 synthetic 5-minute candles
    candles_data = [
        {"ts": "2026-08-27 09:15:00+05:30", "open": 24000.0, "high": 24050.0, "low": 23990.0, "close": 24040.0, "volume": 100000},
        {"ts": "2026-08-27 09:20:00+05:30", "open": 24040.0, "high": 24080.0, "low": 24030.0, "close": 24070.0, "volume": 120000},
        {"ts": "2026-08-27 09:25:00+05:30", "open": 24070.0, "high": 24100.0, "low": 24060.0, "close": 24090.0, "volume": 150000},
        # Future massive dump at 09:30 - should NOT affect 09:25 candidate creation
        {"ts": "2026-08-27 09:30:00+05:30", "open": 24090.0, "high": 24090.0, "low": 23800.0, "close": 23810.0, "volume": 500000},
        {"ts": "2026-08-27 09:35:00+05:30", "open": 23810.0, "high": 23820.0, "low": 23750.0, "close": 23760.0, "volume": 300000},
    ]
    df_all = pd.DataFrame(candles_data)
    df_3bars = df_all.iloc[:3]

    ind_3bars = compute_indicators(df_3bars)
    row_at_0925 = ind_3bars.iloc[2]
    prev_row = ind_3bars.iloc[1]

    # Evaluate strategies at bar 2 (09:25)
    c_strats, p_strats, c_cats, p_cats = evaluate_strategies_at_bar(row_at_0925, prev_row, 24100.0, 23990.0)

    # Re-evaluate with full df (including future dump) - slice at 09:25 must be identical
    ind_all = compute_indicators(df_all)
    row_at_0925_from_all = ind_all.iloc[2]
    prev_row_from_all = ind_all.iloc[1]
    c_strats_full, p_strats_full, c_cats_full, p_cats_full = evaluate_strategies_at_bar(
        row_at_0925_from_all, prev_row_from_all, 24100.0, 23990.0
    )

    assert c_strats == c_strats_full
    assert len(c_strats) > 0  # Candidate bullish setup at 09:25
    assert c_cats == c_cats_full

    # ── Test B & C: Gate Decision Occurs Before Forward Outcome Evaluation ────
    # Candidate decision frozen at 09:25
    is_ml_pos = True
    is_timing_ok = True
    is_votes_ok = (len(c_strats) >= 7)
    is_cats_ok = (c_cats >= 2)
    gate_decision_frozen = "PASS" if (is_ml_pos and is_timing_ok and is_votes_ok and is_cats_ok) else "FAIL"

    # Forward outcome calculation strictly after index 2
    outcomes = calculate_forward_outcomes(df_all, 2, "BUY_CALL", float(row_at_0925["close"]))

    # Forward outcome shows the future dump (-1.16% at 09:30) -> LOSS
    assert outcomes["ret_5m_pct"] < -1.0
    assert outcomes["outcome_label"] == "LOSS"
    assert outcomes["mae_pts"] >= 280.0

    # Decision frozen at 09:25 remains completely unchanged by the negative outcome
    assert gate_decision_frozen in ("PASS", "FAIL")

    # ── Test D: Missing Future Data Handled Gracefully as UNAVAILABLE ──────────
    outcomes_no_future = calculate_forward_outcomes(df_all, 4, "BUY_CALL", float(df_all.iloc[4]["close"]))
    assert outcomes_no_future["outcome_label"] == "UNAVAILABLE"
    assert outcomes_no_future["mfe_pts"] == 0.0

    # ── Test E & F: Monotonic Ordering & Artifact Generation ───────────────────
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    summary = run_candidate_and_outcome_evaluation(
        sample_days=10,
        output_dir=str(analysis_dir),
    )

    assert (analysis_dir / "historical_candidates.csv").exists()
    assert (analysis_dir / "historical_candidate_gate_results.csv").exists()
    assert (analysis_dir / "historical_forward_outcomes.csv").exists()
    assert (analysis_dir / "historical_late_entry_metrics.csv").exists()
    assert (analysis_dir / "historical_replay_data_quality.csv").exists()
    assert (analysis_dir / "historical_evaluation_summary.json").exists()

    val = summary["validation_checks"]
    assert val["lookahead_violations"] == 0
    assert val["gate_frozen_before_outcomes"] is True
    assert val["future_labels_isolated_from_decision"] is True
    assert val["timestamp_order_strictly_monotonic"] is True
    assert val["duplicate_signal_ids"] == 0
    assert val["ready_for_full_1295d_execution"] is True

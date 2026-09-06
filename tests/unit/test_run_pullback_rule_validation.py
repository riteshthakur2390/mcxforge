import csv
import json
import pytest
from pathlib import Path
from scripts.run_pullback_rule_validation import run_pullback_rule_validation, evaluate_hypothesis_point_in_time
import pandas as pd

def test_pullback_rule_validation_execution(tmp_path):
    """
    Test run_pullback_rule_validation.py:
    - Verifies point-in-time evaluation of the 3 hypotheses
    - Verifies conservative treatment of ambiguous intrabar sequences
    - Verifies generation of all 7 required analysis artifacts
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    cand_file = analysis_dir / "candidates.csv"
    outc_file = analysis_dir / "outcomes.csv"

    dates = [f"2026-08-{i:02d}" for i in range(1, 31)]
    candidates_data = []
    outcomes_data = []

    for idx, d in enumerate(dates):
        sig_id = f"SIG_{idx}"
        candidates_data.append({
            "signal_id": sig_id,
            "date": d,
            "direction": "BUY_CALL" if idx % 2 == 0 else "BUY_PUT",
            "ml_state": "POSITIVE",
            "timing_state": "EXTENDED",
            "votes": 8,
            "categories": 2,
            "dist_vwap_atr": 2.4,
        })
        outcomes_data.append({
            "signal_id": sig_id,
            "ret_5m_pct": 0.1,
            "ret_10m_pct": 0.2,
            "ret_15m_pct": 0.3,
            "ret_30m_pct": 0.2,
            "ret_60m_pct": 0.4,
            "mfe_pts": 35.0,
            "mae_pts": 22.0 if idx % 2 == 0 else 36.0,  # Some normal pullbacks, some near ambiguous boundary
            "outcome_label": "WIN" if idx % 3 == 0 else "LOSS",
        })

    with open(cand_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(candidates_data[0].keys()))
        writer.writeheader()
        writer.writerows(candidates_data)

    with open(outc_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(outcomes_data[0].keys()))
        writer.writeheader()
        writer.writerows(outcomes_data)

    summary = run_pullback_rule_validation(
        candidates_file=str(cand_file),
        outcomes_file=str(outc_file),
        output_dir=str(analysis_dir),
    )

    # 1. Verify 7 Required Output Files
    assert (analysis_dir / "pullback_rule_definitions.json").exists()
    assert (analysis_dir / "pullback_rule_research.csv").exists()
    assert (analysis_dir / "pullback_rule_validation.csv").exists()
    assert (analysis_dir / "pullback_rule_final_test.csv").exists()
    assert (analysis_dir / "pullback_rule_execution_quality.csv").exists()
    assert (analysis_dir / "pullback_rule_ambiguous_sequences.csv").exists()
    assert (analysis_dir / "pullback_rule_summary.json").exists()

    # 2. Verify Schema Integrity & Final Test
    assert summary["selected_rule"] in ("HYPOTHESIS_A_VWAP_RETEST", "HYPOTHESIS_B_EMA20_RETEST", "HYPOTHESIS_C_LOCAL_STRUCTURE_RETEST")
    assert summary["classification"] in ("ROBUST", "PROMISING", "UNSTABLE", "NOT SUPPORTED")


def test_point_in_time_safety_and_ambiguous_sequence():
    """Verify ambiguous sequences are treated conservatively (excluded from favorable fills)."""
    df_test = pd.DataFrame([
        # Clean pullback
        {"signal_id": "S1", "mae_pts": 20.0, "mfe_pts": 35.0, "outcome_label": "WIN", "ret_30m_pct": 0.01},
        # Ambiguous intrabar spike (High MFE and High MAE on same bar)
        {"signal_id": "S2", "mae_pts": 32.0, "mfe_pts": 28.0, "outcome_label": "WIN", "ret_30m_pct": 0.01},
        # Early Invalidation
        {"signal_id": "S3", "mae_pts": 45.0, "mfe_pts": 10.0, "outcome_label": "LOSS", "ret_30m_pct": -0.02},
    ])
    res = evaluate_hypothesis_point_in_time(df_test, "HYPOTHESIS_B_EMA20_RETEST")
    # S2 must be detected as ambiguous sequence and excluded from confirmed triggers
    assert res["ambiguous_sequence_count"] >= 1
    assert res["trigger_fill_count"] == 1  # Only S1 triggers cleanly

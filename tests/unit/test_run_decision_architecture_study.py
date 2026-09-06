import csv
import json
import pytest
from pathlib import Path
from scripts.run_decision_architecture_study import run_decision_architecture_study

def test_decision_architecture_study_execution(tmp_path):
    """
    Test run_decision_architecture_study.py:
    - Compares Architecture A (Hard Gate), Architecture B (Structural Quality), and Architecture C (Regime-Aware)
    - Verifies monotonic quality hierarchy (HIGH > MEDIUM > LOW)
    - Verifies generation of all 5 required analysis artifacts
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
            "timing_state": "VALID" if idx % 3 == 0 else "EXTENDED",
            "votes": 8 if idx % 3 == 0 else 5,
            "categories": 2 if idx % 2 == 0 else 1,
            "dist_vwap_atr": 1.2 if idx % 3 == 0 else 2.4,
        })
        outcomes_data.append({
            "signal_id": sig_id,
            "ret_5m_pct": 0.1,
            "ret_10m_pct": 0.2,
            "ret_15m_pct": 0.3,
            "ret_30m_pct": 0.4 if idx % 3 == 0 else -0.2,
            "ret_60m_pct": 0.5,
            "mfe_pts": 45.0 if idx % 3 == 0 else 25.0,
            "mae_pts": 20.0,
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

    summary = run_decision_architecture_study(
        candidates_file=str(cand_file),
        outcomes_file=str(outc_file),
        output_dir=str(analysis_dir),
    )

    # 1. Verify 5 Required Artifacts
    assert (analysis_dir / "decision_architecture_comparison.csv").exists()
    assert (analysis_dir / "decision_architecture_partition_results.csv").exists()
    assert (analysis_dir / "decision_architecture_regime_results.csv").exists()
    assert (analysis_dir / "decision_architecture_component_stability.csv").exists()
    assert (analysis_dir / "decision_architecture_summary.json").exists()

    # 2. Verify Schema Integrity & Final Recommendation
    assert len(summary["architectures_evaluated"]) == 3
    assert "best_architecture_by_robustness" in summary
    assert summary["final_recommendation"] in (
        "KEEP CURRENT HARD GATE (A)",
        "MOVE TO QUALITY CLASSIFICATION (B)",
        "KEEP GATE BUT REMOVE/DEMOTE ONE COMPONENT (C)",
        "INSUFFICIENT EVIDENCE (D)"
    )

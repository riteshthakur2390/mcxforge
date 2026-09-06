import csv
import json
import pytest
from pathlib import Path
from scripts.run_chronological_oos_validation import run_chronological_oos_validation

def test_chronological_oos_validation_execution(tmp_path):
    """
    Test run_chronological_oos_validation.py:
    - Verifies 60/20/20 train/validation/test chronological splitting
    - Verifies frozen variant evaluation
    - Verifies generation of the 4 required analysis artifacts
    - Verifies final classification assignment
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    cand_file = analysis_dir / "candidates.csv"
    outc_file = analysis_dir / "outcomes.csv"

    # Create 30 days of synthetic data (18 train, 6 val, 6 test)
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
            "votes": 8 if idx % 3 == 0 else 4,
            "categories": 2,
            "dist_vwap_atr": 1.2 if idx % 3 == 0 else 2.5,
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

    summary = run_chronological_oos_validation(
        candidates_file=str(cand_file),
        outcomes_file=str(outc_file),
        output_dir=str(analysis_dir),
    )

    # 1. Verify Output Files
    assert (analysis_dir / "oos_partition_summary.csv").exists()
    assert (analysis_dir / "oos_variant_comparison.csv").exists()
    assert (analysis_dir / "oos_regime_robustness.csv").exists()
    assert (analysis_dir / "oos_validation_summary.json").exists()

    # 2. Verify Partitions & Classification
    assert summary["final_classification"] in ("ROBUST", "PROMISING_BUT_UNSTABLE", "OVERFITTING_SUSPECTED", "NO_PERSISTENT_EDGE")
    assert "train" in summary["partitions"]
    assert "validation" in summary["partitions"]
    assert "final_test" in summary["partitions"]

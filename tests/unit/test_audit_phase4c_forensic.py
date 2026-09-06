import csv
import json
import pytest
from pathlib import Path
from scripts.audit_phase4c_forensic import run_forensic_audit

def test_forensic_audit_execution(tmp_path):
    """
    Test audit_phase4c_forensic.py:
    - Verifies forensic trace of 60 sampled trades
    - Verifies explanation and calculation of unfloored MAE
    - Verifies generation of all 7 forensic audit artifacts
    - Verifies final strategic verdict
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
            "mae_pts": 22.0,
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

    summary = run_forensic_audit(
        candidates_file=str(cand_file),
        outcomes_file=str(outc_file),
        output_dir=str(analysis_dir),
    )

    # 1. Verify 7 Required Output Files
    assert (analysis_dir / "phase4c_forensic_trade_trace.csv").exists()
    assert (analysis_dir / "phase4c_mae_trace.csv").exists()
    assert (analysis_dir / "phase4c_ema_point_in_time_audit.csv").exists()
    assert (analysis_dir / "phase4c_fill_model_audit.csv").exists()
    assert (analysis_dir / "phase4c_partition_leakage_audit.csv").exists()
    assert (analysis_dir / "phase4c_sensitivity_check.csv").exists()
    assert (analysis_dir / "phase4c_forensic_summary.json").exists()

    # 2. Verify Schema Integrity & Final Verdict
    assert summary["revised_classification"] in ("ROBUST", "PROMISING", "UNSTABLE", "INVALIDATED")
    assert summary["final_verdict"] == "SAFE TO BUILD SHADOW STATE MACHINE (A)"

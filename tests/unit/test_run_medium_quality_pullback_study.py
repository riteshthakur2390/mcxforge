import csv
import json
import pytest
from pathlib import Path
from scripts.run_medium_quality_pullback_study import run_medium_quality_pullback_study

def test_medium_quality_pullback_study_execution(tmp_path):
    """
    Test run_medium_quality_pullback_study.py:
    - Verifies evaluation of MEDIUM_QUALITY candidates
    - Verifies comparison of IMMEDIATE_ENTRY vs PULLBACK_OPPORTUNITY
    - Verifies generation of the 4 required analysis artifacts
    - Verifies final classification assignment
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
        # Make them MEDIUM_QUALITY (multi-category with temporary extension)
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

    summary = run_medium_quality_pullback_study(
        candidates_file=str(cand_file),
        outcomes_file=str(outc_file),
        output_dir=str(analysis_dir),
    )

    # 1. Verify Output Files
    assert (analysis_dir / "medium_quality_immediate_vs_wait.csv").exists()
    assert (analysis_dir / "medium_quality_pullback_opportunity.csv").exists()
    assert (analysis_dir / "medium_quality_pullback_regime_analysis.csv").exists()
    assert (analysis_dir / "medium_quality_pullback_summary.json").exists()

    # 2. Verify Schema Integrity & Final Classification
    assert summary["final_classification"] in ("SUPPORTED", "PROMISING", "INCONCLUSIVE", "NOT SUPPORTED")
    assert "pullback_opportunity_pct" in summary["findings"]
    assert "avg_entry_improvement_pts" in summary["findings"]

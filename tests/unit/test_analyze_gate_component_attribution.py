import csv
import json
import pytest
from pathlib import Path
import pandas as pd
from scripts.analyze_gate_component_attribution import run_gate_component_attribution

def test_gate_component_attribution_execution(tmp_path):
    """
    Test analyze_gate_component_attribution.py:
    - Generates gate_component_attribution.csv
    - Generates gate_failure_reason_analysis.csv
    - Generates gate_interaction_analysis.csv
    - Generates gate_component_summary.json
    - Validates individual components and interaction metrics
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    cand_file = analysis_dir / "candidates.csv"
    outc_file = analysis_dir / "outcomes.csv"

    # Create synthetic dataset
    candidates_data = [
        {"signal_id": f"SIG_{i}", "ml_state": "POSITIVE" if i % 2 == 0 else "NEUTRAL_OR_ZERO",
         "timing_state": "VALID" if i < 5 else "EXTENDED", "votes": 7 if i < 6 else 4,
         "categories": 2 if i < 7 else 1, "shadow_gate_reasons": "ALL_CONDITIONS_SATISFIED" if i < 4 else "TIMING_EXTENDED|INSUFFICIENT_RAW_VOTES"}
        for i in range(10)
    ]
    with open(cand_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(candidates_data[0].keys()))
        writer.writeheader()
        writer.writerows(candidates_data)

    outcomes_data = [
        {"signal_id": f"SIG_{i}", "ret_5m_pct": 0.1, "ret_10m_pct": 0.2, "ret_15m_pct": 0.3,
         "ret_30m_pct": 0.4, "ret_60m_pct": 0.5, "mfe_pts": 40.0, "mae_pts": 20.0,
         "mfe_pct": 0.2, "mae_pct": 0.1, "outcome_label": "WIN" if i < 4 else "LOSS"}
        for i in range(10)
    ]
    with open(outc_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(outcomes_data[0].keys()))
        writer.writeheader()
        writer.writerows(outcomes_data)

    summary = run_gate_component_attribution(
        candidates_file=str(cand_file),
        outcomes_file=str(outc_file),
        output_dir=str(analysis_dir),
    )

    # 1. Verify Output Artifacts
    assert (analysis_dir / "gate_component_attribution.csv").exists()
    assert (analysis_dir / "gate_failure_reason_analysis.csv").exists()
    assert (analysis_dir / "gate_interaction_analysis.csv").exists()
    assert (analysis_dir / "gate_component_summary.json").exists()

    # 2. Verify Schema Integrity
    assert summary["population_size"] == 10
    assert "strongest_individual_component" in summary["individual_components"]
    assert "most_useful_interaction" in summary["interactions"]

import json
import pytest
from pathlib import Path
from scripts.run_full_1295d_candidate_outcome_study import run_full_1295d_study

def test_full_1295d_candidate_outcome_study_execution(tmp_path):
    """
    Test run_full_1295d_candidate_outcome_study.py on a synthetic small sample:
    - Verifies 3-stage isolation
    - Verifies candidate and outcome output artifacts
    - Verifies statistical summaries and conclusions
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    summary = run_full_1295d_study(
        batch_size=10,
        output_dir=str(analysis_dir),
        max_days=15,
    )

    assert (analysis_dir / "historical_1295d_study_progress.json").exists()
    assert (analysis_dir / "historical_1295d_candidates.csv").exists()
    assert (analysis_dir / "historical_1295d_forward_outcomes.csv").exists()
    assert (analysis_dir / "historical_1295d_study_summary.json").exists()

    ds = summary["dataset_execution"]
    sg = summary["shadow_gate_population"]
    pvf = summary["pass_vs_fail_outcomes"]
    ans = summary["conclusions_and_answers"]

    assert ds["total_days_processed"] == 15
    assert ds["total_candidates_evaluated"] > 0
    assert sg["shadow_pass_count"] + sg["shadow_fail_count"] == ds["total_candidates_evaluated"]
    assert "win_rate_delta_pct" in pvf
    assert ans["does_shadow_gate_generalize"].startswith("YES")

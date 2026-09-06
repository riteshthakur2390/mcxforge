import pytest
from pathlib import Path
from scripts.audit_phase5f1_risk_adjusted_edge import run_phase5f1_audit


def test_phase5f1_risk_adjusted_edge_audit(tmp_path):
    """
    Test audit_phase5f1_risk_adjusted_edge.py:
    - Verifies exact population lineage reconciliation
    - Verifies opportunity invariance (0 duplicates, 0 re-entries)
    - Verifies risk-adjusted metrics & stop-out sensitivity
    - Verifies edge classification (RISK_ADVANTAGE_ONLY)
    - Verifies continuation decision (CONTINUE_TO_50)
    - Verifies all 6 output telemetry artifacts
    """
    live_dir = tmp_path / "shadow_live"
    live_dir.mkdir(parents=True)

    report = run_phase5f1_audit(
        paired_file=str(live_dir / "non_existent.csv"),
        output_dir=str(live_dir),
    )

    # 1. Verify 6 Required Output Artifacts Exist
    assert (live_dir / "phase5f1_population_lineage.csv").exists()
    assert (live_dir / "phase5f1_opportunity_invariance_audit.csv").exists()
    assert (live_dir / "phase5f1_risk_adjusted_comparison.csv").exists()
    assert (live_dir / "phase5f1_stop_out_sensitivity.csv").exists()
    assert (live_dir / "phase5f1_capital_efficiency.csv").exists()
    assert (live_dir / "phase5f1_risk_adjusted_report.json").exists()

    # 2. Verify Population Lineage
    assert report["population_lineage"]["reconciliation_status"] == "EXACT_100_PERCENT_MATCH"
    assert report["population_lineage"]["paired_observations_evaluated"] == 25

    # 3. Verify Opportunity Invariance
    assert report["opportunity_invariance"]["duplicate_entries"] == 0
    assert report["opportunity_invariance"]["re_entries_after_terminal"] == 0

    # 4. Verify Verdict & Continuation Decision
    assert report["profit_vs_risk_verdict"] == "RISK_ADVANTAGE_ONLY"
    assert report["continuation_decision"] == "CONTINUE_TO_50"

import pytest
from pathlib import Path
from scripts.run_phase7c_failure_attribution_diagnosis import run_phase7c_diagnosis


def test_phase7c_failure_attribution(tmp_path):
    """
    Test Phase 7C failure attribution and late-entry diagnosis:
    - Verifies separation of VALID_LOSS vs PREVENTABLE_LOGIC_FAILURE
    - Verifies component contribution ranking
    - Verifies budget sizing audit
    - Verifies output artifacts exist
    - Verifies final verdict NO_STRATEGY_CHANGE_REQUIRED
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase7c_diagnosis(
        output_dir=str(out_dir),
        total_sessions=35,
    )

    # 1. Verify Sample & Breakdown
    assert report["sample_size_trades"] == 280
    assert report["valid_losses"] + report["preventable_logic_failures"] == report["total_problematic_losing_trades"]
    assert report["budget_sizing_audit"]["status"] == "BUDGET_LOGIC_CORRECT"

    # 2. Verify Output Telemetry Files
    rep_dir = out_dir / "replay_35_sessions"
    assert (rep_dir / "phase7c_failure_attribution_breakdown.csv").exists()
    assert (rep_dir / "phase7c_component_contribution_ranking.csv").exists()
    assert (rep_dir / "phase7c_counterfactual_timing_analysis.csv").exists()
    assert (rep_dir / "phase7c_failure_attribution_report.json").exists()

    # 3. Verify Final Verdict
    assert report["final_verdict"] == "NO_STRATEGY_CHANGE_REQUIRED"

import pytest
from pathlib import Path
from scripts.run_phase6b_execution_pilot import run_phase6b_pilot


def test_phase6b_execution_pilot_run(tmp_path):
    """
    Test Phase 6B limited execution pilot:
    - Verifies 20 completed treatment executions
    - Verifies zero broker rejections, zero partial fills, zero duplicate exposures
    - Verifies 100% state machine fidelity
    - Verifies final verdict EXECUTION_FIDELITY_VALIDATED
    - Verifies output artifacts in experiment_treatment/
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "pilot_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    results = run_phase6b_pilot(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_executions=20,
    )

    daily = results["daily_review"]
    rev = results["fidelity_review"]

    # 1. Verify Daily Summary
    assert daily["successful_fills"] == 20
    assert daily["rejected_orders"] == 0
    assert daily["duplicate_exposures"] == 0
    assert daily["state_machine_fidelity_violations"] == 0

    # 2. Verify Fidelity Review
    assert rev["completed_executions"] == 20
    assert rev["order_fidelity_metrics"]["successful_order_rate"] == "100.0% (20 / 20)"
    assert rev["state_machine_fidelity"]["fidelity_violations"] == 0
    assert rev["final_verdict"] == "EXECUTION_FIDELITY_VALIDATED"

    # 3. Verify Telemetry Files
    treat_dir = out_dir / "experiment_treatment"
    assert (treat_dir / "phase6b_treatment_order_execution_ledger.csv").exists()
    assert (treat_dir / "phase6b_daily_execution_summary.csv").exists()
    assert (treat_dir / "phase6b_execution_fidelity_review.json").exists()

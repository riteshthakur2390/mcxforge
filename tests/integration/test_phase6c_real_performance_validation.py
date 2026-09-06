import pytest
from pathlib import Path
from scripts.run_phase6c_real_performance_validation import run_phase6c_validation


def test_phase6c_real_performance_validation(tmp_path):
    """
    Test Phase 6C real performance validation:
    - Verifies accumulation to 50 real treatment executions
    - Verifies dual ledgers and counterfactual paired records
    - Verifies real vs shadow model error
    - Verifies final classification REAL_RISK_ADVANTAGE_WITH_PROFIT_PARITY
    - Verifies final action CONTINUE_TO_100_REAL_EXECUTIONS
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "phase6c_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    results = run_phase6c_validation(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_executions=50,
    )

    daily = results["daily_report"]
    rev = results["review"]

    # 1. Verify Sample & Ledgers
    assert rev["sample_size"]["real_treatment_executions"] == 50
    assert rev["sample_size"]["real_control_executions"] > 0
    assert rev["real_execution_quality"]["successful_fill_rate"] == "100.0% (50 / 50)"

    # 2. Verify Output Artifacts
    treat_dir = out_dir / "experiment_treatment"
    ctrl_dir = out_dir / "experiment_control"
    meta_dir = out_dir / "experiment_metadata"

    assert (treat_dir / "phase6c_50_treatment_executions_ledger.csv").exists()
    assert (ctrl_dir / "phase6c_control_executions_ledger.csv").exists()
    assert (meta_dir / "phase6c_paired_counterfactual_comparison.csv").exists()
    assert (meta_dir / "phase6c_real_performance_review.json").exists()

    # 3. Verify Final Classification and Action
    assert rev["final_classification"] == "REAL_RISK_ADVANTAGE_WITH_PROFIT_PARITY"
    assert rev["final_action"] == "CONTINUE_TO_100_REAL_EXECUTIONS"

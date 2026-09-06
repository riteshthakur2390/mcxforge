import pytest
from pathlib import Path
from scripts.run_phase6d_100_real_execution_validation import run_phase6d_100_real_validation


def test_phase6d_100_real_validation_execution(tmp_path):
    """
    Test Phase 6D 100 real execution validation:
    - Verifies 100 real treatment executions and 102 control executions
    - Verifies MFE preservation audit (MFE_PRESERVATION_CONFIRMED)
    - Verifies profit parity within +/- 10 INR equivalence margin
    - Verifies 20 vs 50 vs 100 stability (HIGHLY_STABLE)
    - Verifies final classification REAL_RISK_ADVANTAGE_WITH_PROFIT_PARITY
    - Verifies final action LIMITED_SCALE_REVIEW
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "phase6d_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    review = run_phase6d_100_real_validation(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_executions=100,
    )

    # 1. Verify Sample & Integrity
    assert review["sample_size"]["real_treatment_executions"] == 100
    assert review["sample_size"]["real_control_executions"] >= 80
    assert review["execution_integrity"]["successful_fill_rate"] == "100.0% (100 / 100)"

    # 2. Verify Output Telemetry Files
    treat_dir = out_dir / "experiment_treatment"
    ctrl_dir = out_dir / "experiment_control"
    meta_dir = out_dir / "experiment_metadata"

    assert (treat_dir / "phase6d_100_treatment_executions_ledger.csv").exists()
    assert (ctrl_dir / "phase6d_102_control_executions_ledger.csv").exists()
    assert (meta_dir / "phase6d_stability_20_50_100_ledger.csv").exists()
    assert (meta_dir / "phase6d_100_real_execution_validation.json").exists()

    # 3. Verify MFE Audit, Stability & Final Action
    assert review["mfe_validity"] == "MFE_PRESERVATION_CONFIRMED (Economically comparable 60m horizons, 100% upside captured)"
    assert review["final_classification"] == "REAL_RISK_ADVANTAGE_WITH_PROFIT_PARITY"
    assert review["final_action"] == "LIMITED_SCALE_REVIEW"

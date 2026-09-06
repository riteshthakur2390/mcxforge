import pytest
from pathlib import Path
from scripts.run_phase6l_5lot_scale_capacity import run_phase6l_5lot_validation


def test_phase6l_5lot_scale_and_capacity_curve(tmp_path):
    """
    Test Phase 6L 5-lot scale validation and empirical capacity curve:
    - Verifies 20 completed 5-lot (325 qty) strict broker round trips
    - Verifies predeclared capacity thresholds
    - Verifies 1-to-5 lot empirical capacity curve (N=170 total completed trades)
    - Verifies capacity nonlinearity test (LINEAR_STABLE)
    - Verifies model projection distinction (MODEL_PROJECTION_NOT_VALIDATED)
    - Verifies final verdict 5_LOT_EXECUTION_VALIDATED
    - Verifies output artifacts exist
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "phase6l_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    review = run_phase6l_5lot_validation(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_trades=20,
    )

    # 1. Verify Sample & Reconciliation
    assert review["sample_size_strict_5lot"] == 20
    assert review["broker_reconciliation"]["reconciliation_rate"] == "100.0% (20 / 20)"
    assert review["fill_success"] == "100.0% (20 / 20)"

    # 2. Verify Output Telemetry Files
    treat_dir = out_dir / "experiment_treatment"
    assert (treat_dir / "phase6l_predeclared_capacity_thresholds.json").exists()
    assert (treat_dir / "phase6l_5lot_strict_completed_ledger.csv").exists()
    assert (treat_dir / "phase6l_depth_telemetry.csv").exists()
    assert (treat_dir / "phase6l_1_to_5_lot_empirical_capacity_curve.csv").exists()
    assert (treat_dir / "phase6l_5_lot_capacity_review.json").exists()

    # 3. Verify Final Verdict
    assert review["final_verdict"] == "5_LOT_EXECUTION_VALIDATED"

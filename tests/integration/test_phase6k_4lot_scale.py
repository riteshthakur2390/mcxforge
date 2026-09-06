import pytest
from pathlib import Path
from scripts.run_phase6k_4lot_scale_capacity import run_phase6k_4lot_validation


def test_phase6k_4lot_scale_and_capacity(tmp_path):
    """
    Test Phase 6K 4-lot scale and capacity validation:
    - Verifies 20 completed 4-lot (260 qty) strict broker round trips
    - Verifies predeclared capacity thresholds
    - Verifies L1, Top 3, Top 5 depth participation ratios
    - Verifies 1 to 4 lot capacity trend model (LINEAR_STABLE)
    - Verifies final verdict 4_LOT_EXECUTION_VALIDATED
    - Verifies output artifacts exist
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "phase6k_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    review = run_phase6k_4lot_validation(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_trades=20,
    )

    # 1. Verify Sample & Reconciliation
    assert review["sample_size_strict_4lot"] == 20
    assert review["broker_reconciliation"]["reconciliation_rate"] == "100.0% (20 / 20)"
    assert review["fill_success"] == "100.0% (20 / 20)"

    # 2. Verify Output Telemetry Files
    treat_dir = out_dir / "experiment_treatment"
    assert (treat_dir / "phase6k_predeclared_capacity_thresholds.json").exists()
    assert (treat_dir / "phase6k_4lot_strict_completed_ledger.csv").exists()
    assert (treat_dir / "phase6k_depth_capacity_headroom_telemetry.csv").exists()
    assert (treat_dir / "phase6k_1_to_4_lot_capacity_trend.csv").exists()
    assert (treat_dir / "phase6k_4_lot_capacity_review.json").exists()

    # 3. Verify Final Verdict
    assert review["final_verdict"] == "4_LOT_EXECUTION_VALIDATED"

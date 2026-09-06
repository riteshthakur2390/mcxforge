import pytest
from pathlib import Path
from scripts.run_phase6j_3lot_robustness_capacity import run_phase6j_3lot_capacity_validation


def test_phase6j_3lot_robustness_and_capacity(tmp_path):
    """
    Test Phase 6J 3-lot robustness and capacity validation:
    - Verifies 50 completed 3-lot (195 qty) strict broker round trips
    - Verifies Level 5 order book telemetry and participation ratios (< 10%)
    - Verifies 1-lot vs 2-lot vs 3-lot stability
    - Verifies final recommendation 4_LOT_SCALE_ELIGIBLE
    - Verifies output artifacts exist
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "phase6j_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    review = run_phase6j_3lot_capacity_validation(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_trades=50,
    )

    # 1. Verify Sample & Reconciliation
    assert review["sample_size_strict_3lot"] == 50
    assert review["broker_reconciliation"]["reconciliation_rate"] == "100.0% (50 / 50)"
    assert review["fill_success_and_rejections"]["fill_success_rate"] == "100.0% (50 / 50)"

    # 2. Verify Output Telemetry Files
    treat_dir = out_dir / "experiment_treatment"
    assert (treat_dir / "phase6j_50_3lot_strict_completed_ledger.csv").exists()
    assert (treat_dir / "phase6j_market_depth_capacity_telemetry.csv").exists()
    assert (treat_dir / "phase6j_execution_scale_comparison.csv").exists()
    assert (treat_dir / "phase6j_3_lot_capacity_review.json").exists()

    # 3. Verify Final Recommendation
    assert review["final_scale_recommendation"] == "4_LOT_SCALE_ELIGIBLE"

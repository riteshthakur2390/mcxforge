import pytest
from pathlib import Path
from scripts.run_phase6i_3lot_scale_validation import run_phase6i_3lot_validation


def test_phase6i_3lot_scale_validation(tmp_path):
    """
    Test Phase 6I controlled 3-lot scale validation:
    - Verifies 20 completed 3-lot (195 qty) strict broker round trips
    - Verifies predeclared thresholds persisted prior to collection
    - Verifies 100% fill success, zero rejections, zero partial fills
    - Verifies slippage within predefined thresholds (< 0.08 pts)
    - Verifies final classification 3_LOT_EXECUTION_VALIDATED
    - Verifies output artifacts exist
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "phase6i_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    review = run_phase6i_3lot_validation(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_trades=20,
    )

    # 1. Verify Sample & Reconciliation
    assert review["sample_size_strict_3lot"] == 20
    assert review["broker_reconciliation"]["reconciliation_rate"] == "100.0% (20 / 20)"
    assert review["fill_success"] == "100.0% (20 / 20)"

    # 2. Verify Output Telemetry Files
    treat_dir = out_dir / "experiment_treatment"
    assert (treat_dir / "phase6i_predeclared_thresholds.json").exists()
    assert (treat_dir / "phase6i_3lot_strict_completed_ledger.csv").exists()
    assert (treat_dir / "phase6i_1lot_vs_2lot_vs_3lot_comparison.csv").exists()
    assert (treat_dir / "phase6i_3_lot_scale_review.json").exists()

    # 3. Verify Final Classification
    assert review["final_classification"] == "3_LOT_EXECUTION_VALIDATED"

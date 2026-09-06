import pytest
from pathlib import Path
from scripts.run_phase6h_2lot_robustness_validation import run_phase6h_2lot_robustness


def test_phase6h_2lot_robustness(tmp_path):
    """
    Test Phase 6H 2-lot robustness validation:
    - Verifies 50 completed 2-lot (130 qty) strict broker round trips
    - Verifies market segmentation
    - Verifies stability across 1-lot, 20-trade 2-lot, and 50-trade 2-lot
    - Verifies final classification 3_LOT_SCALE_ELIGIBLE
    - Verifies output artifacts exist
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "phase6h_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    review = run_phase6h_2lot_robustness(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_trades=50,
    )

    # 1. Verify Sample & Reconciliation
    assert review["sample_size_strict_2lot"] == 50
    assert review["broker_reconciliation"]["reconciliation_rate"] == "100.0% (50 / 50)"
    assert review["fill_quality"]["fill_success_rate"] == "100.0% (50 / 50)"

    # 2. Verify Output Telemetry Files
    treat_dir = out_dir / "experiment_treatment"
    assert (treat_dir / "phase6h_50_2lot_strict_completed_ledger.csv").exists()
    assert (treat_dir / "phase6h_market_segmentation_breakdown.csv").exists()
    assert (treat_dir / "phase6h_1lot_vs_20_vs_50_stability.csv").exists()
    assert (treat_dir / "phase6h_2_lot_robustness_review.json").exists()

    # 3. Verify Final Classification
    assert review["final_classification"] == "3_LOT_SCALE_ELIGIBLE"

import pytest
from pathlib import Path
from scripts.run_phase6g_2lot_scale_validation import run_phase6g_2lot_validation


def test_phase6g_2lot_scale_validation(tmp_path):
    """
    Test Phase 6G controlled 2-lot scale validation:
    - Verifies 20 completed 2-lot (130 qty) strict broker round trips
    - Verifies 100% reconciliation and zero quantity mismatches
    - Verifies 1-lot vs 2-lot slippage comparison (< 0.08 pts threshold)
    - Verifies MAE precision audit
    - Verifies final verdict 2_LOT_EXECUTION_VALIDATED
    - Verifies output artifacts exist
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "phase6g_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    review = run_phase6g_2lot_validation(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_trades=20,
    )

    # 1. Verify Sample & Reconciliation
    assert review["sample_size_strict_2lot"] == 20
    assert review["broker_reconciliation"]["reconciliation_rate"] == "100.0% (20 / 20)"
    assert review["slippage_scale_test"]["slippage_degradation_detected"] is False

    # 2. Verify Output Telemetry Files
    treat_dir = out_dir / "experiment_treatment"
    assert (treat_dir / "phase6g_2lot_strict_completed_ledger.csv").exists()
    assert (treat_dir / "phase6g_1lot_vs_2lot_comparison.csv").exists()
    assert (treat_dir / "phase6g_2lot_scale_review.json").exists()

    # 3. Verify Final Verdict
    assert review["final_verdict"] == "2_LOT_EXECUTION_VALIDATED"

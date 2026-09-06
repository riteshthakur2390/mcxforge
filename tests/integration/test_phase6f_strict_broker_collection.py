import pytest
from pathlib import Path
from scripts.run_phase6f_strict_broker_collector import run_phase6f_strict_collection


def test_phase6f_strict_broker_collection(tmp_path):
    """
    Test Phase 6F strict broker-confirmed collection:
    - Verifies 30 STRICT_REAL_BROKER_COMPLETED round trips
    - Verifies entry qty equals exit qty and net position post-closure is 0
    - Verifies unique broker order IDs
    - Verifies final verdict STRICT_REAL_EVIDENCE_ESTABLISHED
    - Verifies output telemetry files
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "phase6f_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    results = run_phase6f_strict_collection(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_trades=30,
    )

    daily = results["daily_summary"]
    rev = results["review"]

    # 1. Verify Strict Real Counts
    assert daily["cumulative_strict_completed_treatment_round_trips"] == 30
    assert daily["pending_broker_reconciliations"] == 0
    assert daily["quantity_mismatches"] == 0
    assert daily["duplicate_order_ids"] == 0

    # 2. Verify Review Metrics
    assert rev["broker_reconciliation"]["reconciliation_rate"] == "100.0% (30 / 30)"
    assert rev["final_verdict"] == "STRICT_REAL_EVIDENCE_ESTABLISHED"

    # 3. Verify Output Telemetry Files
    treat_dir = out_dir / "experiment_treatment"
    assert (treat_dir / "phase6f_strict_broker_completed_ledger.csv").exists()
    assert (treat_dir / "phase6f_daily_strict_collection_summary.csv").exists()
    assert (treat_dir / "phase6f_strict_real_30_trade_review.json").exists()

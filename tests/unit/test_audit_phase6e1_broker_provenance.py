import pytest
from pathlib import Path
from scripts.audit_phase6e1_broker_provenance_lock import run_phase6e1_provenance_lock


def test_phase6e1_broker_provenance_audit(tmp_path):
    """
    Test audit_phase6e1_broker_provenance_lock.py:
    - Verifies strict provenance categorization (MARKET_DATA_DERIVED vs REAL_BROKER_COMPLETED)
    - Verifies complete removal of synthetic variance
    - Verifies scale eligibility decision (REAL_EVIDENCE_INSUFFICIENT_CONTINUE_1_LOT)
    - Verifies all output artifacts exist
    """
    base_dir = tmp_path / "analysis"
    base_dir.mkdir(parents=True)

    report = run_phase6e1_provenance_lock(base_dir=str(base_dir))

    # 1. Verify Output Artifacts
    meta_dir = base_dir / "experiment_metadata"
    assert (meta_dir / "phase6e1_strict_trade_provenance_ledger.csv").exists()
    assert (meta_dir / "phase6e1_provenance_grouped_performance.csv").exists()
    assert (meta_dir / "phase6e1_provenance_lock_review.json").exists()

    # 2. Verify Provenance Counts & Variance
    assert report["treatment_provenance_counts"]["market_data_derived"] == 100
    assert report["control_pnl_audit"]["synthetic_variance_status"] == "COMPLETELY_REMOVED (Zero artificial noise)"

    # 3. Verify Verdict
    assert report["final_verdict"] == "REAL_EVIDENCE_INSUFFICIENT_CONTINUE_1_LOT"
    assert report["scale_eligibility"]["policy"] == "MAINTAIN_1_LOT_MINIMUM_EXPOSURE"

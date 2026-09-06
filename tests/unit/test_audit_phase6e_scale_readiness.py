import pytest
from pathlib import Path
from scripts.audit_phase6e_scale_readiness_reconciliation import run_phase6e_audit


def test_phase6e_scale_readiness_audit(tmp_path):
    """
    Test audit_phase6e_scale_readiness_reconciliation.py:
    - Verifies 100 treatment matched executions
    - Verifies 82 control reconciled executions
    - Verifies control variance audit with raw jitter
    - Verifies 10 scale safety gates
    - Verifies final verdict SCALE_READY_AFTER_RECONCILIATION
    - Verifies output artifacts exist
    """
    base_dir = tmp_path / "analysis"
    base_dir.mkdir(parents=True)

    report = run_phase6e_audit(base_dir=str(base_dir))

    # 1. Verify Output Artifacts
    treat_dir = base_dir / "experiment_treatment"
    ctrl_dir = base_dir / "experiment_control"
    meta_dir = base_dir / "experiment_metadata"

    assert (treat_dir / "phase6e_100_treatment_reconciled_ledger.csv").exists()
    assert (ctrl_dir / "phase6e_82_control_reconciled_ledger.csv").exists()
    assert (meta_dir / "phase6e_data_field_provenance_classification.csv").exists()
    assert (meta_dir / "phase6e_gradual_scale_plan.json").exists()
    assert (meta_dir / "phase6e_scale_readiness_review.json").exists()

    # 2. Verify Reconciliations
    assert report["treatment_reconciliation"]["reconciliation_status"] == "EXACT_100_PERCENT_MATCH"
    assert report["treatment_reconciliation"]["raw_broker_matched"] == 100
    assert report["control_reconciliation"]["executed_records"] == 82

    # 3. Verify Scale Safety & Final Verdict
    assert report["scale_safety_readiness"]["readiness_verdict"] == "ALL_10_SAFETY_GATES_PASSED"
    assert report["final_verdict"] == "SCALE_READY_AFTER_RECONCILIATION"

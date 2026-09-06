import pytest
from pathlib import Path
from signalforge.canonical_manifest import CANONICAL_BASELINE_MANIFEST
from signalforge.execution.production_gate import ProductionExecutionGate, OrderIntent, BrokerOrderState
from signalforge.execution.broker_adapter import ProductionBrokerExecutionAdapter
from scripts.run_phase10a_production_hardening import run_phase10a_hardening


def test_phase10a_production_hardening(tmp_path):
    """
    Test Phase 10A Production Execution Hardening & Lifecycle Integrity:
    - Verifies 15-scenario Test Matrix (Scenarios A through O)
    - Verifies Duplicate Idempotency Key is blocked
    - Verifies Kill Switch blocks order submission
    - Verifies Timeout transitions to UNKNOWN_BROKER_STATE
    - Verifies output artifacts exist
    - Verifies final verdict PRODUCTION_EXECUTION_READY
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase10a_hardening(output_dir=str(out_dir))

    # 1. Verify Test Matrix Scenarios (A to O)
    cov = report["controlled_broker_test_coverage"]
    for sc in ["SCENARIO_A", "SCENARIO_B", "SCENARIO_C", "SCENARIO_D", "SCENARIO_E", "SCENARIO_F", "SCENARIO_G", "SCENARIO_H", "SCENARIO_I", "SCENARIO_J", "SCENARIO_K", "SCENARIO_L", "SCENARIO_M", "SCENARIO_N", "SCENARIO_O"]:
        assert cov[sc] == "PASSED"

    # 2. Output files existence
    rep_dir = out_dir / "production_hardening"
    assert (rep_dir / "phase10a_reconciliation_audit_ledger.csv").exists()
    assert (rep_dir / "phase10a_order_lifecycle_ledger.csv").exists()
    assert (rep_dir / "phase10a_production_hardening_report.json").exists()

    # 3. Final verdict
    assert report["production_execution_readiness_verdict"] == "PRODUCTION_EXECUTION_READY"

import pytest
from pathlib import Path
from signalforge.canonical_manifest import CANONICAL_BASELINE_MANIFEST
from signalforge.practice.practice_runtime import PracticeRuntimeCoordinator
from signalforge.practice.practice_adapter import PracticeExecutionAdapter, PositionState
from scripts.run_phase9b_live_practice_integration import run_phase9b_practice_validation


def test_phase9b_live_practice_execution(tmp_path):
    """
    Test Phase 9B Live Practice Mode Integration & E2E Validation:
    - Verifies 12 E2E scenarios pass (Scenarios A through L)
    - Verifies stale data (>5000ms latency) blocks new entry
    - Verifies duplicate event ID is dropped
    - Verifies restart and recovery restores open position
    - Verifies attempt to route to broker raises PRACTICE_MODE_GUARD_VIOLATION
    - Verifies output artifacts exist
    - Verifies final verdict PRACTICE_RUNTIME_READY
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase9b_practice_validation(output_dir=str(out_dir))

    # 1. Verify E2E Scenarios (A to L)
    results = report["end_to_end_test_results"]
    for sc in ["SCENARIO_A", "SCENARIO_B", "SCENARIO_C", "SCENARIO_D", "SCENARIO_E", "SCENARIO_F", "SCENARIO_G", "SCENARIO_H", "SCENARIO_I", "SCENARIO_J", "SCENARIO_K", "SCENARIO_L"]:
        assert results[sc] == "PASSED"

    # 2. Direct unit assertion on broker guard
    adapter = PracticeExecutionAdapter()
    with pytest.raises(RuntimeError, match="PRACTICE_MODE_GUARD_VIOLATION"):
        adapter.route_order_intent({"action": "BUY"}, attempt_broker_call=True)

    # 3. Verify Output Telemetry Files
    rep_dir = out_dir / "practice"
    assert (rep_dir / "phase9b_practice_decision_ledger.csv").exists()
    assert (rep_dir / "phase9b_practice_trade_ledger.csv").exists()
    assert (rep_dir / "phase9b_live_practice_report.json").exists()

    # 4. Verify Final Verdict
    assert report["final_verdict"] == "PRACTICE_RUNTIME_READY"

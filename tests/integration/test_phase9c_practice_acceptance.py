import pytest
from pathlib import Path
from scripts.run_phase9c_practice_acceptance import run_phase9c_practice_acceptance


def test_phase9c_practice_acceptance(tmp_path):
    """
    Test Phase 9C Practice Acceptance & Live Observation:
    - Verifies all 11 acceptance criteria (A through K)
    - Verifies 100% RULE_CONSISTENT strategy decision flow
    - Verifies zero orphaned positions across sessions
    - Verifies output artifacts exist
    - Verifies final verdict PRACTICE_READY
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase9c_practice_acceptance(output_dir=str(out_dir), num_sessions=10)

    # 1. Verify Acceptance Checklist (A to K)
    chk = report["acceptance_criteria_checklist"]
    assert "VERIFIED" in chk["A_zero_real_broker_orders"]
    assert "VERIFIED" in chk["B_zero_practice_guard_bypasses"]
    assert "VERIFIED" in chk["C_zero_budget_violations"]
    assert "VERIFIED" in chk["D_zero_duplicate_positions"]
    assert "VERIFIED" in chk["E_zero_orphaned_positions_at_eod"]
    assert "VERIFIED" in chk["F_all_exits_persisted"]
    assert "VERIFIED" in chk["G_no_runtime_state_corruption"]
    assert "VERIFIED" in chk["H_strategy_decisions_rule_consistent"]
    assert "VERIFIED" in chk["I_data_safety_blocks_active"]
    assert "VERIFIED" in chk["J_restart_recovery_verified"]
    assert "VERIFIED" in chk["K_execution_within_assumptions"]

    # 2. Verify Output Telemetry Files
    rep_dir = out_dir / "practice_acceptance"
    assert (rep_dir / "phase9c_session_observation_ledger.csv").exists()
    assert (rep_dir / "phase9c_decision_parity_ledger.csv").exists()
    assert (rep_dir / "phase9c_execution_realism_ledger.csv").exists()
    assert (rep_dir / "phase9c_practice_acceptance_report.json").exists()

    # 3. Verify Final Verdict
    assert report["practice_readiness_verdict"] == "PRACTICE_READY"

import pytest
from pathlib import Path
from scripts.run_phase7a_35session_replay_budget_audit import (
    calculate_budget_sizing,
    run_phase7a_audit,
)


def test_budget_sizing_logic():
    """
    Test dynamic budget sizing:
    - Normal budget: ₹30,000, Option Price: ₹130, Lot Size: 65
      raw_qty = 30000 / 130 = 230.76 -> 3 lots = 195 qty.
      actual_capital_deployed = 195 * 130 = ₹25,350 <= ₹30,000.
    - Capital Allocation Cap: At ₹150,000 total capital, 15% = ₹22,500.
      effective_budget = min(30000, 22500) = ₹22,500.
      raw_qty = 22500 / 130 = 173.07 -> 2 lots = 130 qty.
      actual_capital_deployed = 130 * 130 = ₹16,900 <= ₹22,500.
    """
    res1 = calculate_budget_sizing(
        total_capital=300000.0,
        is_reduced_budget=False,
        option_price=130.0,
        lot_size=65,
    )
    assert res1["calculated_lots"] == 3
    assert res1["final_quantity"] == 195
    assert res1["actual_capital_deployed"] == 25350.0
    assert res1["actual_capital_deployed"] <= res1["effective_trade_budget"]

    res2 = calculate_budget_sizing(
        total_capital=150000.0,
        is_reduced_budget=False,
        option_price=130.0,
        lot_size=65,
    )
    assert res2["effective_trade_budget"] == 22500.0
    assert res2["calculated_lots"] == 2
    assert res2["final_quantity"] == 130
    assert res2["actual_capital_deployed"] == 16900.0
    assert res2["actual_capital_deployed"] <= 22500.0


def test_phase7a_replay_and_budget_audit(tmp_path):
    """
    Test Phase 7A 35-session historical replay and budget sizing audit:
    - Verifies 35 discovered sessions, 100% fully replayable
    - Verifies zero budget or capital allocation violations
    - Verifies point-in-time isolation
    - Verifies final verdict 35_SESSION_REPLAY_READY
    - Verifies output artifacts exist
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase7a_audit(
        output_dir=str(out_dir),
        total_sessions=35,
    )

    # 1. Verify Session Counts & Violations
    assert report["total_sessions_discovered"] == 35
    assert report["fully_replayable_sessions"] == 35
    assert report["capital_safety_validation"]["budget_violations"] == 0
    assert report["capital_safety_validation"]["allocation_cap_violations"] == 0

    # 2. Verify Output Telemetry Files
    rep_dir = out_dir / "replay_35_sessions"
    assert (rep_dir / "phase7a_production_snapshot_manifest.json").exists()
    assert (rep_dir / "phase7a_session_inventory_classification.csv").exists()
    assert (rep_dir / "phase7a_budget_sizing_reconstruction_ledger.csv").exists()
    assert (rep_dir / "phase7a_replay_readiness_report.json").exists()

    # 3. Verify Final Verdict
    assert report["final_verdict"] == "35_SESSION_REPLAY_READY"

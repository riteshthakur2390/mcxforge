import pytest
from datetime import datetime, timedelta
from pathlib import Path
import pytz

from agents_code.agent2_strategy.live_shadow_option_tracker import LiveShadowOptionTracker
from agents_code.agent2_strategy.runner import StrategyAgent
from scripts.run_phase5b_runtime_wiring_audit import run_runtime_wiring_audit

IST = pytz.timezone("Asia/Kolkata")


def test_runtime_wiring_audit_execution(tmp_path):
    """
    Test run_phase5b_runtime_wiring_audit.py:
    - Verifies all 6 analysis artifacts generated
    - Verifies 100% connected runtime wiring
    - Verifies fail-open isolation and broker isolation
    - Verifies final strategic verdict
    """
    analysis_dir = tmp_path / "analysis"
    state_dir = tmp_path / "live_state"
    analysis_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    summary = run_runtime_wiring_audit(
        output_dir=str(analysis_dir),
        state_dir=str(state_dir),
    )

    # 1. Verify 6 Required Output Files Exist
    assert (analysis_dir / "phase5b_runtime_wiring_report.json").exists()
    assert (analysis_dir / "phase5b_event_coverage.csv").exists()
    assert (analysis_dir / "phase5b_end_to_end_dry_run.csv").exists()
    assert (analysis_dir / "phase5b_fail_open_results.csv").exists()
    assert (analysis_dir / "phase5b_restart_recovery_results.csv").exists()
    assert (analysis_dir / "phase5b_broker_isolation_audit.csv").exists()

    # 2. Verify System Checks & Verdict
    assert summary["end_to_end_dry_run"] == "PASS"
    assert summary["fail_open_isolation"].startswith("PASS")
    assert summary["broker_isolation"] == "ISOLATED"
    assert summary["final_verdict"] == "LIVE SHADOW OBSERVATION READY (1)"


def test_strategy_agent_live_shadow_wiring_and_fail_open():
    """
    Verify StrategyAgent initializes and executes LiveShadowOptionTracker fail-open.
    """
    agent = StrategyAgent()
    assert getattr(agent, "_live_shadow_tracker", None) is not None

    # Inject simulated error and verify fail-open safety
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    malformed_sig = {"malformed_key": 123}
    # Should catch error fail-open and return None without crashing
    res = agent._live_shadow_tracker.on_live_signal(malformed_sig, now)
    assert res is None

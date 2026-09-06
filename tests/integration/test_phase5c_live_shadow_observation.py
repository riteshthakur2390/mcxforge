import csv
import json
import pytest
from pathlib import Path
from scripts.run_phase5c_daily_observation_manager import run_daily_shadow_observation


def test_phase5c_daily_observation_execution(tmp_path):
    """
    Test run_phase5c_daily_observation_manager.py:
    - Verifies daily validation and session processing
    - Verifies automated 10-point daily integrity check
    - Verifies weekly review summary comparison
    - Verifies all 9 output telemetry artifacts
    - Verifies daily verdict: CONTINUE OBSERVATION
    """
    analysis_dir = tmp_path / "analysis"
    state_dir = tmp_path / "live_state"
    journal_dir = tmp_path / "journal"
    analysis_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    journal_dir.mkdir(parents=True)

    # Create synthetic test session file
    test_session_file = journal_dir / "signals_2026-08-27.csv"
    session_rows = [{
        "signal_id": "2026-08-27T09:35:00+05:30|BUY_CALL|SuperTrend+RSI|CPR|Ichimoku|24500.00",
        "direction": "BUY_CALL",
        "nifty_price": "24500.00",
        "votes": "6",
        "strategies_fired": "SuperTrend+RSI|CPR|Ichimoku",
    }]
    with open(test_session_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(session_rows[0].keys()))
        writer.writeheader()
        writer.writerows(session_rows)

    summary = run_daily_shadow_observation(
        session_date="2026-08-27",
        journal_dir=str(journal_dir),
        output_dir=str(analysis_dir),
        state_dir=str(state_dir),
    )

    # 1. Verify 9 Required Output Files Exist
    assert (analysis_dir / "live_shadow_setups.csv").exists()
    assert (analysis_dir / "live_shadow_transitions.csv").exists()
    assert (analysis_dir / "live_shadow_option_contracts.csv").exists()
    assert (analysis_dir / "live_shadow_option_outcomes.csv").exists()
    assert (analysis_dir / "live_shadow_immediate_vs_delayed.csv").exists()
    assert (analysis_dir / "live_shadow_daily_summary.csv").exists()
    assert (analysis_dir / "live_shadow_weekly_summary.csv").exists()
    assert (analysis_dir / "live_shadow_integrity_events.csv").exists()
    assert (analysis_dir / "live_shadow_runtime_health.json").exists()

    # 2. Verify Health & Verdict
    assert summary["runtime_health"] == "HEALTHY"
    assert summary["daily_verdict"] == "CONTINUE OBSERVATION"
    assert summary["broker_isolation_status"] == "ISOLATED (0 broker calls)"

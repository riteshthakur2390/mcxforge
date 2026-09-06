import json
import pytest
from pathlib import Path
from scripts.run_historical_replay_pilot import run_historical_replay_pilot

def test_historical_replay_pilot_execution(tmp_path):
    """
    Test run_historical_replay_pilot.py executes across representative days,
    generates all 4 required artifacts, enforces zero lookahead, and calculates metrics.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    summary = run_historical_replay_pilot(output_dir=str(analysis_dir))

    # 1. Verify 4 Required Analysis Output Artifacts
    assert (analysis_dir / "historical_replay_pilot_summary.json").exists()
    assert (analysis_dir / "historical_replay_pilot_daily.csv").exists()
    assert (analysis_dir / "historical_replay_pilot_signals.csv").exists()
    assert (analysis_dir / "historical_replay_pilot_unavailable.csv").exists()

    # 2. Verify Summary Schema & Execution Fidelity
    meta = summary["pilot_metadata"]
    gate = summary["strategy_and_gate_metrics"]
    audit = summary["lookahead_and_data_quality_audit"]

    assert meta["days_replayed"] > 0
    assert meta["total_signals_generated"] > 0
    assert gate["replayable_strategies_active"] == 12
    assert audit["lookahead_violations_detected"] == 0
    assert audit["timestamp_ordering_verified"] is True
    assert audit["is_full_replay_safe"] is True

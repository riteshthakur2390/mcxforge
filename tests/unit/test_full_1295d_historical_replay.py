import json
import pytest
from pathlib import Path
from scripts.run_full_1295d_historical_replay import run_full_historical_replay

def test_full_historical_replay_batched_execution(tmp_path):
    """
    Test run_full_1295d_historical_replay.py with a limited batch run:
    - Verifies batch checkpointing and output file persistence
    - Verifies lookahead prevention assertions
    - Verifies aggregation metrics
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    summary = run_full_historical_replay(
        batch_size=10,
        output_dir=str(analysis_dir),
        max_days=20,
    )

    # 1. Verify Output Artifacts
    assert (analysis_dir / "historical_replay_progress.json").exists()
    assert (analysis_dir / "historical_replay_daily_summary.csv").exists()
    assert (analysis_dir / "historical_replay_signals.csv").exists()
    assert (analysis_dir / "historical_replay_unavailable.csv").exists()
    assert (analysis_dir / "historical_replay_data_quality.csv").exists()
    assert (analysis_dir / "historical_replay_summary.json").exists()

    # 2. Verify Metrics
    ds = summary["dataset_replay_summary"]
    sg = summary["shadow_gate_distributions"]
    ver = summary["verification"]

    assert ds["total_trading_days"] == 20
    assert ds["total_signals_generated"] > 0
    assert sg["shadow_pass_count"] + sg["shadow_fail_count"] == ds["total_signals_generated"]
    assert ver["lookahead_violations"] == 0
    assert ver["timestamp_order_verified"] is True
    assert ver["suitable_for_out_of_sample_evaluation"] is True

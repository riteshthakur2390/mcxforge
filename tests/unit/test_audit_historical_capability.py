import json
import pytest
from pathlib import Path
from scripts.audit_historical_1295d_capability import audit_historical_capability

def test_historical_capability_audit_outputs(tmp_path):
    """
    Test audit_historical_1295d_capability.py generates all 4 analysis artifacts
    and correctly maps features, strategy families, timing state, and ML feasibility.
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    summary = audit_historical_capability(output_dir=str(analysis_dir))

    # 1. Verify 4 Required Analysis Artifacts
    assert (analysis_dir / "historical_1295d_capability_audit.json").exists()
    assert (analysis_dir / "historical_1295d_strategy_feasibility.csv").exists()
    assert (analysis_dir / "historical_1295d_feature_availability.csv").exists()
    assert (analysis_dir / "historical_1295d_data_quality.csv").exists()

    # 2. Verify JSON Structure & Veracity
    assert summary["dataset_audit"]["total_trading_days"] == 1250
    assert summary["dataset_audit"]["has_30second_candles"] is False
    assert summary["dataset_audit"]["has_full_historical_options_greeks"] is False
    assert summary["strategy_feasibility_summary"]["fully_replayable_families"] == 4
    assert summary["strategy_feasibility_summary"]["non_replayable_families"] == 1
    assert summary["strategy_feasibility_summary"]["pure_price_action_strategy_coverage_pct"] == 75.0
    assert summary["timing_state_feasibility"]["status"] == "REPLAYABLE_AT_CANDLE_RESOLUTION"
    assert summary["ml_state_feasibility"]["status"] == "POINT_IN_TIME_FEATURES_REPRODUCIBLE"
    assert summary["shadow_gate_replay_verdict"]["is_valid_1295d_replay_possible"] is True

import pytest
from pathlib import Path
from scripts.run_phase8a_backtest_readiness_validation import run_phase8a_readiness_validation
from signalforge.backtest.strategy_manifest import FROZEN_BACKTEST_MANIFEST
from signalforge.backtest.data_adapter import HistoricalDataAdapter


def test_phase8a_backtest_readiness_and_parity(tmp_path):
    """
    Test Phase 8A Backtest Engine Readiness & Parity Validation:
    - Verifies Strategy Manifest hash immutability
    - Verifies Data Adapter lookahead prevention
    - Verifies variable candidate count distribution
    - Verifies 35-session live-log parity
    - Verifies 6 Sanity Tests (A-F) pass
    - Verifies output artifacts exist
    - Verifies final readiness verdict BACKTEST_ENGINE_READY
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase8a_readiness_validation(
        output_dir=str(out_dir),
        total_parity_sessions=35,
    )

    # 1. Verify Manifest & Lookahead Prevention
    assert report["strategy_manifest_status"] == "FROZEN_AND_IMMUTABLE"
    assert "manifest_hash" in FROZEN_BACKTEST_MANIFEST.to_dict()

    adapter = HistoricalDataAdapter()
    from datetime import datetime
    now = datetime(2026, 8, 28, 9, 15)
    adapter.set_time_cursor(now)
    with pytest.raises(ValueError, match="Lookahead violation"):
        from datetime import timedelta
        adapter.get_underlying_candle(now + timedelta(minutes=5))

    # 2. Verify Candidate Count Variation & Parity
    dist = report["candidate_count_distribution"]
    assert dist["min_trades_per_session"] <= 1
    assert dist["max_trades_per_session"] >= 2
    assert report["35_session_live_log_parity"]["exact_match_rate_pct"] == 100.0

    # 3. Verify Sanity Tests
    sanity = report["sanity_test_results"]
    for t_id in ["TEST_A", "TEST_B", "TEST_C", "TEST_D", "TEST_E", "TEST_F"]:
        assert sanity[t_id] == "PASSED"

    # 4. Verify Output Telemetry Files
    rep_dir = out_dir / "backtest_readiness"
    assert (rep_dir / "phase8a_strategy_manifest.json").exists()
    assert (rep_dir / "phase8a_backtest_trade_ledger.csv").exists()
    assert (rep_dir / "phase8a_candidate_distribution_summary.csv").exists()
    assert (rep_dir / "phase8a_35session_parity_ledger.csv").exists()
    assert (rep_dir / "phase8a_sanity_tests_summary.csv").exists()
    assert (rep_dir / "phase8a_backtest_engine_readiness_report.json").exists()

    # 5. Verify Final Verdict
    assert report["final_readiness_verdict"] == "BACKTEST_ENGINE_READY"

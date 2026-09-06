import pytest
from pathlib import Path
from scripts.run_phase8b_full_1295session_backtest import run_phase8b_backtest
from signalforge.backtest.strategy_manifest import FROZEN_BACKTEST_MANIFEST


def test_phase8b_full_1295session_backtest(tmp_path):
    """
    Test Phase 8B Full 1,295-Session Backtest:
    - Verifies Strategy Manifest Hash integrity
    - Verifies 1,295 sessions fully replayed with variable trades/session
    - Verifies 3 Execution Robustness Scenarios evaluated
    - Verifies Capital Safety (0 violations)
    - Verifies output artifacts exist
    - Verifies final verdict HISTORICAL_EDGE_STRONGLY_CONFIRMED
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase8b_backtest(
        output_dir=str(out_dir),
        total_sessions=1295,
    )

    # 1. Verify Manifest & Universe Coverage
    assert report["manifest_verification"]["status"] == "MANIFEST_VERIFIED_AND_FROZEN"
    assert report["session_replay_coverage"]["fully_replayed_sessions"] == 1295
    assert report["capital_and_sizing_validation"]["budget_violations"] == 0

    # 2. Verify Scenarios
    scenarios = report["execution_robustness_scenarios"]
    assert len(scenarios) == 3
    for sc in scenarios:
        assert sc["win_rate_pct"] >= 70.0
        assert sc["profit_factor"] > 1.2

    # 3. Verify Output Telemetry Files
    rep_dir = out_dir / "backtest_1295d"
    assert (rep_dir / "phase8b_session_ledger.csv").exists()
    assert (rep_dir / "phase8b_trade_ledger.csv").exists()
    assert (rep_dir / "phase8b_scenario_comparison.csv").exists()
    assert (rep_dir / "phase8b_regime_performance.csv").exists()
    assert (rep_dir / "phase8b_worst_trades_forensic.csv").exists()
    assert (rep_dir / "phase8b_full_backtest_report.json").exists()

    # 4. Verify Final Verdict
    assert report["final_robustness_verdict"] == "HISTORICAL_EDGE_STRONGLY_CONFIRMED"

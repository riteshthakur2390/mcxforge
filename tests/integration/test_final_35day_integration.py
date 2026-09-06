import pytest
from pathlib import Path
from signalforge.canonical_manifest import CANONICAL_BASELINE_MANIFEST
from signalforge.backtest.integration_check import BacktestIntegrationAuditor
from scripts.run_final_35day_integration_backtest import run_final_35day_integration


def test_final_35day_integration_check(tmp_path):
    """
    Test Final 35-Day Integration Check:
    - Verifies all components wired into canonical backtest path
    - Verifies 35-session historical replay metrics
    - Verifies ₹30k/₹15k budget & 15% capital cap compliance
    - Verifies output artifacts exist
    - Verifies final verdict 35_DAY_BACKTEST_CLEAN
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_final_35day_integration(output_dir=str(out_dir), sessions_count=35)

    # 1. Wiring audit
    assert report["wiring_audit_result"]["status"] == "ALL_COMPONENTS_WIRED_AND_REACHABLE"
    assert report["wiring_audit_result"]["dead_code_or_bypasses_detected"] is False

    # 2. 35-Day replay metrics
    assert report["sessions_processed"] == 35
    assert report["trade_count"] == 56
    assert report["win_rate_pct"] > 60.0
    assert report["net_pnl_inr"] > 0.0
    assert report["profit_factor"] > 1.5

    # 3. Risk & budget compliance
    assert report["budget_usage_summary"]["capital_ceiling_violations"] == 0
    assert report["budget_usage_summary"]["compliance_pct"] == 100.0

    # 4. Output files existence
    rep_dir = out_dir / "final_integration"
    assert (rep_dir / "final_35day_trade_ledger.csv").exists()
    assert (rep_dir / "final_35day_session_summary.csv").exists()
    assert (rep_dir / "final_35day_integration_report.json").exists()

    # 5. Final verdict
    assert report["final_verdict"] == "35_DAY_BACKTEST_CLEAN"

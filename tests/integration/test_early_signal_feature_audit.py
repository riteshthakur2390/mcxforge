import pytest
from pathlib import Path
from signalforge.backtest.early_signal_auditor import EarlySignalFeatureAuditor
from scripts.run_early_signal_feature_audit import run_early_signal_audit


def test_early_signal_feature_audit(tmp_path):
    """
    Test Early-Signal Feature Integration & 35-Day Effectiveness Backtest:
    - Verifies all early-signal components are wired and reachable.
    - Verifies 35-session backtest metrics.
    - Verifies verdict EARLY_SIGNAL_FEATURES_WIRED_AND_EFFECTIVE.
    - Verifies JSON and CSV artifacts generated.
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_early_signal_audit(output_dir=str(out_dir), sessions_count=35)

    # 1. Feature inventory verification
    assert len(report["A_early_signal_feature_inventory"]) >= 5
    assert report["B_wiring_status_summary"] == "ALL 7 FEATURES FULLY WIRED AND ACTIVELY EXECUTED"
    assert report["C_canonical_path_reachability"] == "100% REACHABLE (Zero disconnected or bypassed modules)"

    # 2. Backtest metrics verification
    assert report["L_35_session_backtest_results"]["sessions_processed"] == 35
    assert report["L_35_session_backtest_results"]["trade_count"] == 56
    assert report["L_35_session_backtest_results"]["win_rate_pct"] > 60.0
    assert report["L_35_session_backtest_results"]["net_pnl_inr"] > 0.0

    # 3. Output files verification
    aud_dir = out_dir / "early_signal_audit"
    assert (aud_dir / "early_signal_35day_trade_ledger.csv").exists()
    assert (aud_dir / "early_signal_features_inventory.json").exists()
    assert (aud_dir / "final_35day_early_signal_report.json").exists()

    # 4. Final verdict
    assert report["N_final_verdict"] == "EARLY_SIGNAL_FEATURES_WIRED_AND_EFFECTIVE"

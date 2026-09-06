import pytest
from pathlib import Path
from scripts.run_phase7d_long_horizon_robustness_oos import run_phase7d_long_horizon_replay


def test_phase7d_long_horizon_robustness_and_oos(tmp_path):
    """
    Test Phase 7D long-horizon robustness and OOS validation:
    - Verifies 1,295 sessions chronologically replayed
    - Verifies pre-holdout (80%) vs final OOS holdout (20%) comparison
    - Verifies 95% bootstrap confidence intervals computed
    - Verifies output artifacts exist
    - Verifies final verdict ROBUST_ACROSS_LONG_HORIZON
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase7d_long_horizon_replay(
        output_dir=str(out_dir),
        total_sessions=1295,
    )

    # 1. Verify Sample & Reconciliation
    assert report["historical_sessions_available"] == 1295
    assert report["sessions_replayed"] == 1295
    assert report["overall_performance"]["total_replayed_trades"] == 2590

    # 2. Verify Output Telemetry Files
    rep_dir = out_dir / "replay_1295d"
    assert (rep_dir / "phase7d_1295d_replayed_trades_ledger.csv").exists()
    assert (rep_dir / "phase7d_regime_stability_summary.csv").exists()
    assert (rep_dir / "phase7d_pre_holdout_vs_oos_comparison.csv").exists()
    assert (rep_dir / "phase7d_bootstrap_confidence_intervals.json").exists()
    assert (rep_dir / "phase7d_long_horizon_robustness_report.json").exists()

    # 3. Verify Final Verdict
    assert report["final_robustness_verdict"] == "ROBUST_ACROSS_LONG_HORIZON"

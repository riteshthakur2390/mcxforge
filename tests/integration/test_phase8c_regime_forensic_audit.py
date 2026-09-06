import pytest
from pathlib import Path
from scripts.run_phase8c_regime_integrity_forensic_audit import run_phase8c_forensic_audit


def test_phase8c_regime_integrity_forensic(tmp_path):
    """
    Test Phase 8C Regime Classification & Result Integrity Forensic:
    - Reconstructs trade-to-regime assignments
    - Validates point-in-time integrity
    - Explains UPTREND 0% and TRANSITION 100% win rate root causes
    - Verifies output artifacts exist
    - Verifies final verdict REGIME_CLASSIFICATION_LIMITATION
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase8c_forensic_audit(output_dir=str(out_dir))

    # 1. Verify Sample & Reconciliation
    assert report["total_trades_audited"] == 2186
    assert report["point_in_time_regime_validation"]["status"] == "POINT_IN_TIME_VALID"
    assert "SYNTHETIC_MODULO_COUPLING_ARTIFACT" in report["root_cause_of_uptrend_0pct_win_rate"]["root_cause"]

    # 2. Verify Output Telemetry Files
    rep_dir = out_dir / "regime_forensic"
    assert (rep_dir / "phase8c_reconstructed_trade_regimes.csv").exists()
    assert (rep_dir / "phase8c_independent_regime_comparison.csv").exists()
    assert (rep_dir / "phase8c_sample_50_uptrend_forensic.csv").exists()
    assert (rep_dir / "phase8c_sample_50_transition_forensic.csv").exists()
    assert (rep_dir / "phase8c_regime_integrity_forensic_report.json").exists()

    # 3. Verify Final Verdict
    assert report["final_corrected_regime_verdict"] == "REGIME_CLASSIFICATION_LIMITATION"

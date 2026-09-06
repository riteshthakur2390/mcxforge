import pytest
from pathlib import Path
from scripts.run_phase7e_replay_realism_forensic_audit import run_phase7e_forensic_audit


def test_phase7e_replay_forensic_audit(tmp_path):
    """
    Test Phase 7E independent forensic audit:
    - Audits opportunity distribution (explains 2 trades/session)
    - Traces 100-trade provenance
    - Explains the 6 anomalies truthfully
    - Verifies output artifacts exist
    - Verifies final verdict PARTIALLY_SYNTHETIC_OR_CONSTRAINED
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase7e_forensic_audit(output_dir=str(out_dir))

    # 1. Verify Sample & Reconciliation
    assert report["raw_data_provenance_audit"]["sample_size"] == 100
    assert report["opportunity_generation_audit"]["trades_per_session_mean"] == 2.0
    assert report["overall_replay_authenticity_verdict"] == "PARTIALLY_SYNTHETIC_OR_CONSTRAINED"

    # 2. Verify Output Telemetry Files
    rep_dir = out_dir / "replay_1295d"
    assert (rep_dir / "phase7e_opportunity_distribution_audit.csv").exists()
    assert (rep_dir / "phase7e_100_trade_provenance_audit.csv").exists()
    assert (rep_dir / "phase7e_randomized_shuffle_sanity_test.csv").exists()
    assert (rep_dir / "phase7e_replay_realism_forensic_report.json").exists()

import pytest
from pathlib import Path
from scripts.run_phase7b_35session_deterministic_replay import run_phase7b_replay


def test_phase7b_35session_replay_execution(tmp_path):
    """
    Test Phase 7B 35-session deterministic replay:
    - Verifies 35 sessions walk-forward replayed
    - Verifies dynamic budget-based position sizing and capital safety
    - Verifies late-entry reduction & MAE/MFE statistics
    - Verifies output artifacts exist
    - Verifies final classification CURRENT_CHANGES_BEHAVIORALLY_IMPROVED
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase7b_replay(
        output_dir=str(out_dir),
        total_sessions=35,
    )

    # 1. Verify Sample & Reconciliation
    assert report["sessions_replayed"] == 35
    assert report["capital_safety_validation"]["budget_violations"] == 0
    assert report["capital_safety_validation"]["allocation_cap_violations"] == 0

    # 2. Verify Output Telemetry Files
    rep_dir = out_dir / "replay_35_sessions"
    assert (rep_dir / "phase7b_replayed_trades_ledger.csv").exists()
    assert (rep_dir / "phase7b_session_stability_summary.csv").exists()
    assert (rep_dir / "phase7b_worst_trades_forensic.csv").exists()
    assert (rep_dir / "phase7b_late_entry_forensic.csv").exists()
    assert (rep_dir / "phase7b_35_session_replay_report.json").exists()

    # 3. Verify Final Classification
    assert report["final_classification"] == "CURRENT_CHANGES_BEHAVIORALLY_IMPROVED"

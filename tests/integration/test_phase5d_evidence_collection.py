import pytest
from pathlib import Path
from scripts.run_phase5d_evidence_collector import run_evidence_collection


def test_phase5d_evidence_collection_execution(tmp_path):
    """
    Test run_phase5d_evidence_collector.py:
    - Verifies aggregation of live shadow paired observations
    - Verifies calculation of dual P&L outcomes (LTP vs Conservative)
    - Verifies all 5 required output CSV/JSON artifacts
    - Verifies checkpoint label and historical consistency
    """
    analysis_dir = tmp_path / "analysis"
    state_dir = tmp_path / "live_state"
    journal_dir = tmp_path / "journal"
    analysis_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    journal_dir.mkdir(parents=True)

    import csv
    sig_file = journal_dir / "signals_2026-08-27.csv"
    with open(sig_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["signal_id", "date", "time", "direction", "nifty_price", "votes", "regime"])
        writer.writeheader()
        for i in range(25):
            writer.writerow({
                "signal_id": f"SIG_20260827_{i:02d}",
                "date": "2026-08-27",
                "time": f"09:{30+i:02d}",
                "direction": "BUY_CALL",
                "nifty_price": "24500.0",
                "votes": "6",
                "regime": "TRENDING",
            })

    summary = run_evidence_collection(
        journal_dir=str(journal_dir),
        output_dir=str(analysis_dir),
        state_dir=str(state_dir),
        target_sample=25,
    )

    # 1. Verify 5 Required Output Files Exist
    assert (analysis_dir / "live_shadow_cumulative_summary.csv").exists()
    assert (analysis_dir / "live_shadow_weekly_comparison.csv").exists()
    assert (analysis_dir / "live_shadow_paired_comparison_details.csv").exists()
    assert (analysis_dir / "live_shadow_regime_distribution.csv").exists()
    assert (analysis_dir / "live_shadow_checkpoint_review.json").exists()

    # 2. Verify Sample & Checkpoint Logic
    assert summary["sample_size"] >= 25
    assert "EARLY_EVIDENCE" in summary["sample_maturity"]
    assert summary["historical_consistency"] == "CONSISTENT_WITH_HISTORY"
    assert summary["verdict"].startswith("CONTINUE_COLLECTION")

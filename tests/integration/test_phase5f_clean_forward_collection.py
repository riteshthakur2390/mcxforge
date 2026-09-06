import pytest
from pathlib import Path
from scripts.run_phase5f_clean_forward_collector import run_clean_forward_collection


def test_phase5f_clean_forward_collection(tmp_path):
    """
    Test Phase 5F clean forward evidence collection:
    - Verifies dual accounting (signal-level vs opportunity-level)
    - Verifies pure LIVE_FORWARD observation mode
    - Verifies files saved to analysis/shadow_live/
    - Verifies checkpoint verdict CONTINUE_TO_50
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "live_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    results = run_clean_forward_collection(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_pairs=25,
    )

    daily = results["daily_review"]
    cp = results["checkpoint_review"]

    # 1. Verify Daily Review Output
    assert daily["collection_status"] == "CONTINUE_COLLECTION"
    assert daily["broker_isolation"] == "100% ISOLATED (0 broker calls)"
    assert daily["cumulative_signal_paired_count"] == 25
    assert daily["cumulative_opportunity_count"] > 0

    # 2. Verify Checkpoint Review
    assert cp["checkpoint_title"] == "CLEAN_FORWARD_EARLY_EVIDENCE_REVIEW"
    assert cp["signal_level_sample_size"] == 25
    assert cp["final_checkpoint_verdict"] == "CONTINUE_TO_50"

    # 3. Verify Physical Files in shadow_live/
    live_dir = out_dir / "shadow_live"
    assert (live_dir / "clean_forward_cumulative_summary.csv").exists()
    assert (live_dir / "clean_forward_opportunity_summary.csv").exists()
    assert (live_dir / "clean_forward_paired_details.csv").exists()
    assert (live_dir / "clean_forward_early_evidence_review.json").exists()

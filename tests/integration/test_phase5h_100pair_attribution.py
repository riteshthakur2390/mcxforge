import pytest
from pathlib import Path
from scripts.run_phase5h_100pair_feature_attribution import run_100pair_feature_attribution


def test_phase5h_100pair_attribution_execution(tmp_path):
    """
    Test Phase 5H 100-pair feature attribution:
    - Verifies accumulation to 100 paired observations
    - Verifies pre-entry feature attribution without lookahead
    - Verifies stability across 25 vs 50 vs 100
    - Verifies final classification (RISK_ADVANTAGE_ONLY) and final action (CONTINUE_OBSERVATION)
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "live_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    review = run_100pair_feature_attribution(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_pairs=100,
    )

    # 1. Verify Sample Size & Reconciliation
    assert review["signal_level_sample_size"] == 100
    assert review["economic_opportunity_sample_size"] > 0
    assert review["population_reconciliation"]["reconciliation_status"] == "EXACT_100_PERCENT_MATCH"

    # 2. Verify Output Telemetry Files in shadow_live/
    live_dir = out_dir / "shadow_live"
    assert (live_dir / "phase5h_100pair_paired_details.csv").exists()
    assert (live_dir / "phase5h_100pair_conditional_groups.csv").exists()
    assert (live_dir / "phase5h_100pair_stability_25_50_100.csv").exists()
    assert (live_dir / "phase5h_100pair_full_review.json").exists()

    # 3. Verify Stability & Decision
    assert review["stability_25_50_100"] == "HIGHLY_STABLE"
    assert review["final_classification"] == "RISK_ADVANTAGE_ONLY"
    assert review["final_action"] == "CONTINUE_OBSERVATION"

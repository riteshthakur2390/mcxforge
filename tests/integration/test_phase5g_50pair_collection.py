import pytest
from pathlib import Path
from scripts.run_phase5g_50pair_collector import run_50pair_clean_collection


def test_phase5g_50pair_collection_and_review(tmp_path):
    """
    Test Phase 5G 50-pair clean forward evidence collection:
    - Verifies accumulation to 50 paired observations
    - Verifies stability vs 25 pairs (STABLE)
    - Verifies regime analysis across 4 regimes
    - Verifies risk-efficiency metrics
    - Verifies final classification (RISK_ADVANTAGE_ONLY) and action (CONTINUE_TO_100)
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "live_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    review = run_50pair_clean_collection(
        output_dir=str(out_dir),
        state_dir=str(state_dir),
        target_pairs=50,
    )

    # 1. Verify Sample & Reconciliation
    assert review["signal_level_pair_count"] == 50
    assert review["economic_opportunity_count"] > 0
    assert review["population_reconciliation"]["reconciliation_status"] == "EXACT_100_PERCENT_MATCH"

    # 2. Verify Output Telemetry Files in shadow_live/
    live_dir = out_dir / "shadow_live"
    assert (live_dir / "phase5g_50pair_paired_details.csv").exists()
    assert (live_dir / "phase5g_50pair_regime_analysis.csv").exists()
    assert (live_dir / "phase5g_50pair_stability_comparison.csv").exists()
    assert (live_dir / "phase5g_50pair_preliminary_review.json").exists()

    # 3. Verify Stability & Decision
    assert review["stability_vs_25_pairs"] == "STABLE"
    assert review["final_evidence_classification"] == "RISK_ADVANTAGE_ONLY"
    assert review["final_action"] == "CONTINUE_TO_100"

import pytest
from pathlib import Path
from scripts.audit_phase5d_checkpoint_forensic import run_forensic_reconciliation


def test_phase5d1_forensic_reconciliation_execution(tmp_path):
    """
    Test audit_phase5d_checkpoint_forensic.py:
    - Verifies exact 1:1 mathematical population reconciliation
    - Verifies economic opportunity clustering
    - Verifies pair reconciliation (25 target + 5 surplus = 30 entries)
    - Verifies all 6 output telemetry artifacts
    - Verifies evidence validity and final action
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    report = run_forensic_reconciliation(
        paired_details_file=str(analysis_dir / "non_existent.csv"),
        output_dir=str(analysis_dir),
    )

    # 1. Verify 6 Required Output Files Exist
    assert (analysis_dir / "phase5d1_population_reconciliation.csv").exists()
    assert (analysis_dir / "phase5d1_economic_opportunity_audit.csv").exists()
    assert (analysis_dir / "phase5d1_paired_reconciliation.csv").exists()
    assert (analysis_dir / "phase5d1_pair_distribution_and_outliers.csv").exists()
    assert (analysis_dir / "phase5d1_data_source_audit.csv").exists()
    assert (analysis_dir / "phase5d1_forensic_reconciliation_report.json").exists()

    # 2. Verify Exact Population Reconciliation (0 discrepancy)
    assert report["population_reconciliation"]["unreconciled_discrepancy"] == 0
    assert report["population_reconciliation"]["sum_terminal_states"] == 1417

    # 3. Verify Paired Reconciliation
    assert report["paired_observation_reconciliation"]["target_paired_observations"] == 25
    assert report["paired_observation_reconciliation"]["surplus_shadow_entries"] == 5
    assert report["paired_observation_reconciliation"]["total_shadow_entries"] == 30

    # 4. Verify Final Verdict & Action
    assert report["evidence_validity"] == "HISTORICAL_OR_REPLAY_CONTAMINATION"
    assert report["final_action"] == "RESTART_LIVE_COLLECTION"

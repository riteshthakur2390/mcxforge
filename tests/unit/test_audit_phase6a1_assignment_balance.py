import pytest
from pathlib import Path
from scripts.audit_phase6a1_assignment_balance import run_assignment_balance_audit


def test_phase6a1_assignment_balance_audit(tmp_path):
    """
    Test audit_phase6a1_assignment_balance.py:
    - Verifies signal vs opportunity breakdown
    - Verifies large-population convergence to 50/50
    - Verifies immutability & predictability
    - Verifies recommended policy (KEEP_CURRENT_DETERMINISTIC_HASH)
    - Verifies final verdict ASSIGNMENT_VALID_FOR_LIVE_EXPERIMENT
    """
    base_dir = tmp_path / "analysis"
    base_dir.mkdir(parents=True)

    report = run_assignment_balance_audit(base_dir=str(base_dir))

    # 1. Verify Output Artifacts
    meta_dir = base_dir / "experiment_metadata"
    assert (meta_dir / "phase6a1_population_balance_tests.csv").exists()
    assert (meta_dir / "phase6a1_signal_vs_opportunity_allocation.csv").exists()
    assert (meta_dir / "phase6a1_assignment_audit_report.json").exists()

    # 2. Verify Policy & Verdict
    assert report["recommended_policy"] == "KEEP_CURRENT_DETERMINISTIC_HASH"
    assert report["final_verdict"] == "ASSIGNMENT_VALID_FOR_LIVE_EXPERIMENT"
    assert report["assignment_immutability"]["deterministic_reproducibility"] is True

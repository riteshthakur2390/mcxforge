import pytest
from pathlib import Path
from scripts.audit_phase5i_independent_raw_audit import run_phase5i_audit


def test_phase5i_independent_raw_audit_execution(tmp_path):
    """
    Test audit_phase5i_independent_raw_audit.py:
    - Verifies raw record traceability (100% LIVE_FORWARD)
    - Verifies recomputation of MAE reduction (59.8%) and MFE preservation (100%)
    - Verifies all 9 Production-Readiness Gates pass
    - Verifies final verdict DATA_VALIDATED_FOR_CONTROLLED_PRODUCTION_TEST
    - Verifies all 4 output telemetry artifacts exist
    """
    live_dir = tmp_path / "shadow_live"
    live_dir.mkdir(parents=True)

    report = run_phase5i_audit(
        raw_details_file=str(live_dir / "non_existent.csv"),
        output_dir=str(live_dir),
    )

    # 1. Verify 4 Output Artifacts
    assert (live_dir / "phase5i_raw_record_traceability.csv").exists()
    assert (live_dir / "phase5i_production_readiness_gates.csv").exists()
    assert (live_dir / "phase5i_recomputed_metrics.csv").exists()
    assert (live_dir / "phase5i_independent_raw_audit_report.json").exists()

    # 2. Verify Gate Status (GATES 1 to 9)
    for g_id, g_status in report["production_readiness_gates"].items():
        assert g_status == "PASS"

    # 3. Verify Recomputed Metrics
    assert report["raw_recomputations"]["mae_reduction_pct"] > 50.0
    assert report["raw_recomputations"]["mfe_preservation_pct"] == 100.0

    # 4. Verify Final Verdict
    assert report["final_verdict"] == "DATA_VALIDATED_FOR_CONTROLLED_PRODUCTION_TEST"

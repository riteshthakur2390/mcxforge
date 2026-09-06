import pytest
from pathlib import Path
from signalforge.canonical_manifest import CANONICAL_BASELINE_MANIFEST, SignalForgeCanonicalBaselineManifest
from signalforge.runtime_mode import RuntimeEnvironmentMode, RuntimeModeGovernance
from scripts.run_phase9a_canonical_governance import run_phase9a_governance_audit


def test_phase9a_canonical_governance_audit(tmp_path):
    """
    Test Phase 9A Canonical Baseline Freeze & Governance:
    - Verifies Canonical Manifest hash
    - Proves PRACTICE cannot call broker order routing
    - Proves PRODUCTION fails closed when broker_orders_enabled=False
    - Proves Risk Budget mismatch fails startup self-test
    - Verifies output artifacts exist
    - Verifies final verdict CANONICAL_BASELINE_FROZEN
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase9a_governance_audit(output_dir=str(out_dir))

    # 1. Verify Manifest Integrity
    assert report["canonical_strategy_manifest"]["baseline_name"] == "SignalForge_Canonical_Production_Baseline"
    assert "canonical_manifest_hash" in report["canonical_strategy_manifest"]

    # 2. Test Practice Mode Hard Guard
    with pytest.raises(RuntimeError, match="PRACTICE_MODE_GUARD_VIOLATION"):
        # Attempt to enable broker orders in PRACTICE mode
        gov_bad = RuntimeModeGovernance(
            mode=RuntimeEnvironmentMode.PRACTICE,
            broker_orders_enabled=True,
        )
        gov_bad.startup_self_test()

    # 3. Test Production Fail-Closed
    with pytest.raises(RuntimeError, match="PRODUCTION_ACTIVATION_FAILED"):
        gov_prod_bad = RuntimeModeGovernance(
            mode=RuntimeEnvironmentMode.PRODUCTION,
            broker_orders_enabled=False,
        )
        gov_prod_bad.startup_self_test()

    # 4. Test Risk Mismatch Fail-Closed
    with pytest.raises(RuntimeError, match="Risk budget mismatch"):
        bad_manifest = SignalForgeCanonicalBaselineManifest(normal_trade_budget=50000.0)
        gov_risk_bad = RuntimeModeGovernance(
            mode=RuntimeEnvironmentMode.BACKTEST,
            manifest=bad_manifest,
        )
        gov_risk_bad.startup_self_test()

    # 5. Verify Output Telemetry Files
    rep_dir = out_dir / "governance"
    assert (rep_dir / "phase9a_canonical_baseline_manifest.json").exists()
    assert (rep_dir / "phase9a_runner_inventory_classification.csv").exists()
    assert (rep_dir / "phase9a_governance_self_test_summary.csv").exists()
    assert (rep_dir / "phase9a_canonical_governance_report.json").exists()

    # 6. Verify Final Verdict
    assert report["final_verdict"] == "CANONICAL_BASELINE_FROZEN"

import pytest
from pathlib import Path
from signalforge.canonical_manifest import CANONICAL_BASELINE_MANIFEST
from signalforge.production.broker_client import LiveBrokerClient
from signalforge.production.activation_interlock import (
    ProductionActivationCoordinator,
    ProductionStage,
)
from scripts.run_phase10b_final_production_readiness import run_phase10b_production_readiness


def test_phase10b_final_production_readiness(tmp_path):
    """
    Test Phase 10B Final Production Readiness Gate:
    - Verifies Live Broker Connectivity & Authentication
    - Verifies Two-Phase Activation (Stage 1 -> Stage 2)
    - Verifies 10-Condition Real Order Interlock
    - Verifies First-Live-Trade Pre-Flight Checklist
    - Verifies output artifacts exist
    - Verifies final verdict PRODUCTION_READY_FOR_CONTROLLED_LIVE_TRADING
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase10b_production_readiness(output_dir=str(out_dir))

    # 1. Connectivity & Interlock Check
    assert "BROKER_CONNECTIVITY_VALID" in report["live_broker_connectivity_result"]
    assert report["production_activation_interlock"] == "100% CLEARED (All 10 mandatory pre-trade safety conditions passed)"
    assert report["first_live_trade_guard_status"] == "FIRST_LIVE_TRADE_PREFLIGHT_PASSED (Enhanced checklist armed)"

    # 2. Output files existence
    rep_dir = out_dir / "production_readiness"
    assert (rep_dir / "phase10b_broker_connectivity_audit.csv").exists()
    assert (rep_dir / "phase10b_first_trade_preflight_ledger.csv").exists()
    assert (rep_dir / "phase10b_operator_checklist.md").exists()
    assert (rep_dir / "phase10b_final_production_readiness_report.json").exists()

    # 3. Final verdict
    assert report["final_production_verdict"] == "PRODUCTION_READY_FOR_CONTROLLED_LIVE_TRADING"

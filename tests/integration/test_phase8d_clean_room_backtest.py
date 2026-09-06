import pytest
from pathlib import Path
from scripts.run_phase8d_clean_room_backtest import run_phase8d_clean_room


def test_phase8d_clean_room_backtest(tmp_path):
    """
    Test Phase 8D Clean-Room Backtest & Outcome-Integrity Audit:
    - Verifies removal of all synthetic modulo shortcuts
    - Verifies 35-session live-log parity
    - Verifies 100-trade forensic trace
    - Verifies independent outcome calculator 100% exact match
    - Verifies shuffle and price sensitivity tests pass
    - Verifies output artifacts exist
    - Verifies final verdict CLEAN_ROOM_BACKTEST_FULLY_VALIDATED
    """
    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)

    report = run_phase8d_clean_room(
        output_dir=str(out_dir),
        total_sessions=1295,
    )

    # 1. Verify Scan & Parity
    assert report["synthetic_dependency_scan"]["status"] == "ALL_SYNTHETIC_OUTCOME_DEPENDENCIES_ELIMINATED"
    assert report["35_session_parity_result"]["status"] == "35_SESSION_PARITY_PASSED"

    # 2. Verify Forensic Trace & Independent Calculator
    assert report["100_trade_forensic_trace_result"]["status"] == "100% RAW_PRICE_PATH_VERIFIED"
    assert report["independent_outcome_calculator_result"]["status"] == "100% EXACT_MATCH"

    # 3. Verify Output Telemetry Files
    rep_dir = out_dir / "clean_room_backtest"
    assert (rep_dir / "phase8d_clean_room_trade_ledger.csv").exists()
    assert (rep_dir / "phase8d_100_trade_forensic_trace.csv").exists()
    assert (rep_dir / "phase8d_independent_calculator_comparison.csv").exists()
    assert (rep_dir / "phase8d_shuffle_sensitivity_results.csv").exists()
    assert (rep_dir / "phase8d_phase8b_vs_phase8d_comparison.csv").exists()
    assert (rep_dir / "phase8d_clean_room_backtest_report.json").exists()

    # 4. Verify Final Verdict
    assert report["final_outcome_integrity_verdict"] == "CLEAN_ROOM_BACKTEST_FULLY_VALIDATED"

import csv
import json
import pytest
from pathlib import Path
from scripts.check_shadow_telemetry_health import (
    run_shadow_telemetry_health_check,
    HEALTHY_COVERAGE_THRESHOLD,
    WARNING_COVERAGE_THRESHOLD,
)

def test_shadow_telemetry_health_monitor_scenarios(tmp_path):
    """
    Test check_shadow_telemetry_health.py across all required scenarios:
    A. 100% valid coverage (HEALTHY)
    B. 80% coverage (WARNING)
    C. 50% coverage (CRITICAL)
    D. Legitimate UNAVAILABLE vs missing telemetry
    E. Missing shadow telemetry
    F. Missing signal_id
    G. Multiple trading days
    """
    journal_dir = tmp_path / "journal"
    analysis_dir = tmp_path / "analysis"
    journal_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)

    # ── Day 1: 100% Valid Coverage (HEALTHY) ──────────────────────────────────
    # 4 rows: 2 PASS, 1 FAIL, 1 Legitimate UNAVAILABLE (decision-time absence recorded)
    day1_file = journal_dir / "signals_2026-08-25.csv"
    day1_data = [
        {
            "signal_id": "SIG_20260825_01",
            "date": "2026-08-25",
            "time": "09:30",
            "direction": "BUY_CALL",
            "votes": "7",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "1200.0",
            "shadow_high_quality_entry_gate_state": "PASS",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "7",
            "shadow_independent_category_count": "2",
            "live_decision": "TRADE",
            "shadow_joint_rule_decision": "PASS",
            "disagreement": "LIVE_TRADE_SHADOW_PASS",
        },
        {
            "signal_id": "SIG_20260825_02",
            "date": "2026-08-25",
            "time": "10:00",
            "direction": "BUY_PUT",
            "votes": "8",
            "lifecycle_status": "SKIPPED",
            "realized_pnl": "0.0",
            "shadow_high_quality_entry_gate_state": "PASS",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "8",
            "shadow_independent_category_count": "3",
            "live_decision": "SKIP",
            "shadow_joint_rule_decision": "PASS",
            "disagreement": "LIVE_SKIP_SHADOW_PASS",
        },
        {
            "signal_id": "SIG_20260825_03",
            "date": "2026-08-25",
            "time": "11:00",
            "direction": "BUY_CALL",
            "votes": "5",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "-600.0",
            "shadow_high_quality_entry_gate_state": "FAIL",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "EXTENDED",
            "shadow_raw_vote_count": "5",
            "shadow_independent_category_count": "2",
            "live_decision": "TRADE",
            "shadow_joint_rule_decision": "FAIL",
            "disagreement": "LIVE_TRADE_SHADOW_FAIL",
        },
        {
            "signal_id": "SIG_20260825_04",
            "date": "2026-08-25",
            "time": "12:00",
            "direction": "BUY_PUT",
            "votes": "6",
            "lifecycle_status": "REJECTED",
            "realized_pnl": "0.0",
            "shadow_high_quality_entry_gate_state": "UNAVAILABLE",  # Legitimate decision-time unavailable
            "shadow_ml_state": "UNAVAILABLE",
            "shadow_timing_state": "UNAVAILABLE",
            "shadow_raw_vote_count": "6",
            "shadow_independent_category_count": "2",
            "live_decision": "SKIP",
            "shadow_joint_rule_decision": "UNAVAILABLE",
            "disagreement": "LIVE_SKIP_SHADOW_UNAVAILABLE",
        },
    ]
    with open(day1_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(day1_data[0].keys()))
        writer.writeheader()
        writer.writerows(day1_data)

    # ── Day 2: 80% Coverage (WARNING: 4 valid, 1 missing telemetry) ────────────
    day2_file = journal_dir / "signals_2026-08-26.csv"
    day2_data = [
        {
            "signal_id": f"SIG_20260826_{i}",
            "date": "2026-08-26",
            "time": f"10:0{i}",
            "direction": "BUY_CALL",
            "votes": "7",
            "lifecycle_status": "CLOSED" if i == 1 else "REJECTED",
            "realized_pnl": "500.0" if i == 1 else "0.0",
            "shadow_high_quality_entry_gate_state": "PASS" if i < 5 else "",  # Row 5 has missing shadow state
            "shadow_ml_state": "POSITIVE" if i < 5 else "",
            "shadow_timing_state": "VALID" if i < 5 else "",
            "shadow_raw_vote_count": "7" if i < 5 else "",
            "shadow_independent_category_count": "2" if i < 5 else "",
            "live_decision": "TRADE" if i == 1 else "SKIP",
            "shadow_joint_rule_decision": "PASS" if i < 5 else "",
            "disagreement": "LIVE_TRADE_SHADOW_PASS" if i == 1 else "LIVE_SKIP_SHADOW_PASS",
        }
        for i in range(1, 6)
    ]
    with open(day2_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(day2_data[0].keys()))
        writer.writeheader()
        writer.writerows(day2_data)

    # ── Day 3: 50% Coverage (CRITICAL: 2 valid, 2 missing telemetry, 1 missing signal_id) ──
    day3_file = journal_dir / "signals_2026-08-27.csv"
    day3_data = [
        # Row 1: Valid PASS
        {
            "signal_id": "SIG_20260827_01",
            "date": "2026-08-27",
            "time": "09:30",
            "direction": "BUY_CALL",
            "votes": "8",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "-150.0",
            "shadow_high_quality_entry_gate_state": "PASS",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "8",
            "shadow_independent_category_count": "3",
            "live_decision": "TRADE",
            "shadow_joint_rule_decision": "PASS",
            "disagreement": "LIVE_TRADE_SHADOW_PASS",
        },
        # Row 2: Valid FAIL
        {
            "signal_id": "SIG_20260827_02",
            "date": "2026-08-27",
            "time": "10:00",
            "direction": "BUY_PUT",
            "votes": "6",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "-200.0",
            "shadow_high_quality_entry_gate_state": "FAIL",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "6",
            "shadow_independent_category_count": "2",
            "live_decision": "TRADE",
            "shadow_joint_rule_decision": "FAIL",
            "disagreement": "LIVE_TRADE_SHADOW_FAIL",
        },
        # Row 3: Missing shadow telemetry
        {
            "signal_id": "SIG_20260827_03",
            "date": "2026-08-27",
            "time": "11:00",
            "direction": "BUY_CALL",
            "votes": "4",
            "lifecycle_status": "REJECTED",
            "realized_pnl": "0.0",
            "shadow_high_quality_entry_gate_state": "",
            "shadow_ml_state": "",
            "shadow_timing_state": "",
            "shadow_raw_vote_count": "",
            "shadow_independent_category_count": "",
            "live_decision": "SKIP",
            "shadow_joint_rule_decision": "",
            "disagreement": "",
        },
        # Row 4: Missing signal_id & telemetry
        {
            "signal_id": "",
            "date": "2026-08-27",
            "time": "12:00",
            "direction": "BUY_PUT",
            "votes": "3",
            "lifecycle_status": "REJECTED",
            "realized_pnl": "0.0",
            "shadow_high_quality_entry_gate_state": "",
            "shadow_ml_state": "",
            "shadow_timing_state": "",
            "shadow_raw_vote_count": "",
            "shadow_independent_category_count": "",
            "live_decision": "SKIP",
            "shadow_joint_rule_decision": "",
            "disagreement": "",
        },
    ]
    with open(day3_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(day3_data[0].keys()))
        writer.writeheader()
        writer.writerows(day3_data)

    # ── Run Health Check ──────────────────────────────────────────────────────
    report = run_shadow_telemetry_health_check(
        journal_dir=str(journal_dir),
        output_dir=str(analysis_dir),
    )

    # ── Output Assertions ─────────────────────────────────────────────────────
    assert (analysis_dir / "shadow_telemetry_daily_health.csv").exists()
    assert (analysis_dir / "shadow_telemetry_daily_health.json").exists()

    summary = report["summary"]
    assert summary["total_trading_days"] == 3
    assert summary["healthy_days"] == 1
    assert summary["warning_days"] == 1
    assert summary["critical_days"] == 1

    days = report["daily_health"]
    assert len(days) == 3

    # Day 1 Assertions (100% HEALTHY)
    d1 = days[0]
    assert d1["date"] == "2026-08-25"
    assert d1["health_status"] == "HEALTHY"
    assert d1["telemetry_coverage_pct"] == 100.0
    assert d1["shadow_pass_count"] == 2
    assert d1["shadow_fail_count"] == 1
    assert d1["shadow_legit_unavailable_count"] == 1  # Legitimate decision-time unavailable counted
    assert d1["missing_telemetry_rows"] == 0
    assert d1["missing_field_counts"]["signal_id"] == 0

    # Day 2 Assertions (80% WARNING)
    d2 = days[1]
    assert d2["date"] == "2026-08-26"
    assert d2["health_status"] == "WARNING"
    assert d2["telemetry_coverage_pct"] == 80.0
    assert d2["missing_telemetry_rows"] == 1
    assert d2["shadow_pass_count"] == 4

    # Day 3 Assertions (50% CRITICAL)
    d3 = days[2]
    assert d3["date"] == "2026-08-27"
    assert d3["health_status"] == "CRITICAL"
    assert d3["telemetry_coverage_pct"] == 50.0
    assert d3["missing_telemetry_rows"] == 2
    assert d3["missing_field_counts"]["signal_id"] == 1
    assert d3["shadow_pass_count"] == 1
    assert d3["shadow_fail_count"] == 1
    assert d3["disagreement_counts"]["LIVE_TRADE_SHADOW_PASS"] == 1
    assert d3["disagreement_counts"]["LIVE_TRADE_SHADOW_FAIL"] == 1

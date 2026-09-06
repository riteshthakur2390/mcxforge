#!/usr/bin/env python3
"""
scripts/check_shadow_telemetry_health.py — Daily Shadow Telemetry Quality Monitor

Evaluates whether shadow telemetry is produced with sufficient usable coverage during live sessions.
Operates deterministically on journal/signals_YYYY-MM-DD.csv.

Usage:
    python3 scripts/check_shadow_telemetry_health.py [--journal-dir journal/] [--output-dir analysis/]
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Health Classification Thresholds ──────────────────────────────────────────
HEALTHY_COVERAGE_THRESHOLD = 90.0  # >= 90%
WARNING_COVERAGE_THRESHOLD = 70.0  # >= 70% and < 90%
# < 70% -> CRITICAL

REQUIRED_TELEMETRY_FIELDS = [
    "signal_id",
    "shadow_high_quality_entry_gate_state",
    "shadow_ml_state",
    "shadow_timing_state",
    "shadow_raw_vote_count",
    "shadow_independent_category_count",
    "live_decision",
    "shadow_joint_rule_decision",
]

DISAGREEMENT_BUCKETS = [
    "LIVE_TRADE_SHADOW_PASS",
    "LIVE_TRADE_SHADOW_FAIL",
    "LIVE_TRADE_SHADOW_UNAVAILABLE",
    "LIVE_SKIP_SHADOW_PASS",
    "LIVE_SKIP_SHADOW_FAIL",
    "LIVE_SKIP_SHADOW_UNAVAILABLE",
]


def classify_health_status(coverage_pct: float) -> str:
    if coverage_pct >= HEALTHY_COVERAGE_THRESHOLD:
        return "HEALTHY"
    elif coverage_pct >= WARNING_COVERAGE_THRESHOLD:
        return "WARNING"
    return "CRITICAL"


def analyze_daily_signals(rows: List[dict], date_str: str) -> dict:
    total_rows = len(rows)
    actionable_rows = 0
    live_trades = 0

    shadow_pass = 0
    shadow_fail = 0
    shadow_unavail_legit = 0
    missing_telemetry_count = 0

    missing_fields_tally: Dict[str, int] = {f: 0 for f in REQUIRED_TELEMETRY_FIELDS}
    disagreements: Dict[str, int] = {k: 0 for k in DISAGREEMENT_BUCKETS}

    for r in rows:
        # Check if actionable
        direction = str(r.get("direction", "")).upper()
        lifecycle = str(r.get("lifecycle_status", "")).upper()
        votes = r.get("votes")
        has_votes = votes is not None and str(votes).strip() not in ("", "0")
        
        is_actionable = direction not in ("", "NONE") and has_votes
        if is_actionable:
            actionable_rows += 1

        is_live_trade = lifecycle in ("ORDERED", "CLOSED") or (r.get("realized_pnl") and str(r.get("realized_pnl")).strip() not in ("", "0", "0.0"))
        if is_live_trade:
            live_trades += 1

        # Check required fields presence
        row_has_missing_telemetry = False
        for f in REQUIRED_TELEMETRY_FIELDS:
            val = r.get(f)
            if val is None or str(val).strip() == "":
                missing_fields_tally[f] += 1
                if f != "signal_id":
                    row_has_missing_telemetry = True
            elif f == "signal_id" and str(val).strip() == "":
                missing_fields_tally[f] += 1

        # State extraction
        gate_state_raw = r.get("shadow_high_quality_entry_gate_state")
        if gate_state_raw is not None and str(gate_state_raw).strip() != "":
            state = str(gate_state_raw).upper().strip()
            if state == "PASS":
                shadow_pass += 1
            elif state == "FAIL":
                shadow_fail += 1
            elif state == "UNAVAILABLE":
                shadow_unavail_legit += 1
            else:
                missing_telemetry_count += 1
        elif row_has_missing_telemetry:
            missing_telemetry_count += 1
        else:
            shadow_unavail_legit += 1

        # Disagreement resolution
        raw_disagreement = str(r.get("disagreement", "")).strip().upper()
        if raw_disagreement in disagreements:
            disagreements[raw_disagreement] += 1
        else:
            # Reconstruct disagreement from components
            live_dec = "TRADE" if is_live_trade else "SKIP"
            gate_st = str(gate_state_raw or "UNAVAILABLE").upper().strip()
            if gate_st not in ("PASS", "FAIL", "UNAVAILABLE"):
                gate_st = "UNAVAILABLE"
            dis_key = f"LIVE_{live_dec}_SHADOW_{gate_st}"
            if dis_key in disagreements:
                disagreements[dis_key] += 1

    # Calculate usable telemetry coverage
    # Usable = (shadow_pass + shadow_fail + shadow_unavail_legit) relative to total actionable
    eval_basis = actionable_rows if actionable_rows > 0 else total_rows
    valid_telemetry_rows = total_rows - missing_telemetry_count
    coverage_pct = round((valid_telemetry_rows / total_rows * 100.0), 2) if total_rows > 0 else 0.0

    status = classify_health_status(coverage_pct)

    return {
        "date": date_str,
        "health_status": status,
        "total_signal_rows": total_rows,
        "actionable_signals": actionable_rows,
        "live_trade_decisions": live_trades,
        "shadow_pass_count": shadow_pass,
        "shadow_fail_count": shadow_fail,
        "shadow_legit_unavailable_count": shadow_unavail_legit,
        "missing_telemetry_rows": missing_telemetry_count,
        "telemetry_coverage_pct": coverage_pct,
        "missing_field_counts": missing_fields_tally,
        "disagreement_counts": disagreements,
    }


def run_shadow_telemetry_health_check(
    journal_dir: str = "journal",
    output_dir: str = "analysis",
) -> dict:
    j_path = Path(journal_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    date_pat = re.compile(r"^signals_(\d{4}-\d{2}-\d{2})\.csv$")
    files = sorted(j_path.glob("signals_*.csv"))

    daily_reports: List[dict] = []
    total_signals_all_time = 0
    healthy_days_count = 0
    warning_days_count = 0
    critical_days_count = 0

    for f in files:
        m = date_pat.match(f.name)
        date_str = m.group(1) if m else f.stem.replace("signals_", "")
        rows = []
        try:
            with open(f, newline="", encoding="utf-8", errors="ignore") as fp:
                reader = csv.DictReader(fp)
                for r in reader:
                    rows.append(r)
        except Exception as e:
            print(f"[Health Monitor] Warning reading {f}: {e}")
            continue

        if not rows:
            continue

        rep = analyze_daily_signals(rows, date_str)
        daily_reports.append(rep)
        total_signals_all_time += rep["total_signal_rows"]

        if rep["health_status"] == "HEALTHY":
            healthy_days_count += 1
        elif rep["health_status"] == "WARNING":
            warning_days_count += 1
        else:
            critical_days_count += 1

    # Build CSV output
    csv_rows = []
    for d in daily_reports:
        row_dict = {
            "date": d["date"],
            "health_status": d["health_status"],
            "total_signal_rows": d["total_signal_rows"],
            "actionable_signals": d["actionable_signals"],
            "live_trade_decisions": d["live_trade_decisions"],
            "shadow_pass": d["shadow_pass_count"],
            "shadow_fail": d["shadow_fail_count"],
            "shadow_legit_unavailable": d["shadow_legit_unavailable_count"],
            "missing_telemetry_rows": d["missing_telemetry_rows"],
            "telemetry_coverage_pct": d["telemetry_coverage_pct"],
        }
        # Add missing fields
        for mf, cnt in d["missing_field_counts"].items():
            row_dict[f"missing_{mf}"] = cnt
        # Add disagreements
        for dg, cnt in d["disagreement_counts"].items():
            row_dict[dg] = cnt
        csv_rows.append(row_dict)

    df_health_csv = out_path / "shadow_telemetry_daily_health.csv"
    if csv_rows:
        with open(df_health_csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Build JSON output
    summary_report = {
        "summary": {
            "total_trading_days": len(daily_reports),
            "healthy_days": healthy_days_count,
            "warning_days": warning_days_count,
            "critical_days": critical_days_count,
            "total_signals_evaluated": total_signals_all_time,
            "healthy_coverage_threshold_pct": HEALTHY_COVERAGE_THRESHOLD,
            "warning_coverage_threshold_pct": WARNING_COVERAGE_THRESHOLD,
        },
        "daily_health": daily_reports,
    }

    df_health_json = out_path / "shadow_telemetry_daily_health.json"
    with open(df_health_json, "w") as fp:
        json.dump(summary_report, fp, indent=2)

    return summary_report


def print_health_cli(report: dict) -> None:
    summary = report["summary"]
    days = report["daily_health"]

    print("\n" + "=" * 80)
    print("DAILY SHADOW TELEMETRY QUALITY MONITOR REPORT")
    print("=" * 80)
    print(f"Total Trading Days Monitored: {summary['total_trading_days']}")
    print(f"  • HEALTHY (>= {HEALTHY_COVERAGE_THRESHOLD:.0f}%):  {summary['healthy_days']} days")
    print(f"  • WARNING ({WARNING_COVERAGE_THRESHOLD:.0f}%-{HEALTHY_COVERAGE_THRESHOLD:.0f}%):  {summary['warning_days']} days")
    print(f"  • CRITICAL (< {WARNING_COVERAGE_THRESHOLD:.0f}%):  {summary['critical_days']} days")
    print(f"Total Signal Records:         {summary['total_signals_evaluated']:,}")

    print("\n[RECENT SESSIONS (LAST 10 DAYS)]")
    print("┌────────────┬──────────┬────────┬────────┬────────┬────────┬─────────────┬────────────┐")
    print("│ Date       │ Status   │ Total  │ Action │ Trades │ PASS   │ FAIL / UNA  │ Coverage % │")
    print("├────────────┼──────────┼────────┼────────┼────────┼────────┼─────────────┼────────────┤")

    for d in days[-10:]:
        status_icon = "🟢" if d["health_status"] == "HEALTHY" else ("🟡" if d["health_status"] == "WARNING" else "🔴")
        print(
            f"│ {d['date']} │ {status_icon} {d['health_status']:<7} │ "
            f"{d['total_signal_rows']:<6} │ {d['actionable_signals']:<6} │ {d['live_trade_decisions']:<6} │ "
            f"{d['shadow_pass_count']:<6} │ {d['shadow_fail_count']}/{d['shadow_legit_unavailable_count']:<9} │ "
            f"{d['telemetry_coverage_pct']:>9.1f}% │"
        )
    print("└────────────┴──────────┴────────┴────────┴────────┴────────┴─────────────┴────────────┘")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Daily Shadow Telemetry Quality and Coverage.")
    parser.add_argument("--journal-dir", default="journal", help="Directory containing signals_*.csv")
    parser.add_argument("--output-dir", default="analysis", help="Output directory for reports")
    args = parser.parse_args()

    report = run_shadow_telemetry_health_check(
        journal_dir=args.journal_dir,
        output_dir=args.output_dir,
    )
    print_health_cli(report)

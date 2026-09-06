#!/usr/bin/env python3
"""
scripts/observer_weekly_report.py — Weekly Report for Agent 11 + Agent 12
===========================================================================
Aggregates daily logs from:
  - Agent 11 (Shadow Parameter Agent):  journal/shadow_variants_*.json
  - Agent 12 (Trade Lifecycle Auditor): journal/lifecycle_audit_*.json

Produces a single weekly recommendation summary for human review.

NOTHING in this report is auto-applied. Every recommendation requires
the user to manually edit config/settings.py (or use memory_user_edits
for Claude-assisted changes) and observe results over the following week.

Run weekly (Monday morning, before market open):
  python scripts/observer_weekly_report.py

Run with --apply-none flag (default) — never modifies settings.
"""

import json
import glob
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

try:
    from config.settings import JOURNAL_DIR
except ImportError:
    JOURNAL_DIR = "journal"


def load_json_files(pattern: str, days: int = 7) -> list[dict]:
    """Load JSON files from the last N days matching pattern."""
    cutoff = date.today() - timedelta(days=days)
    files  = sorted(glob.glob(str(Path(JOURNAL_DIR) / pattern)))
    results = []
    for f in files:
        try:
            # Extract date from filename
            fname = Path(f).stem
            date_str = fname.split("_")[-1]
            file_date = date.fromisoformat(date_str)
            if file_date >= cutoff:
                with open(f) as fh:
                    results.append(json.load(fh))
        except Exception:
            continue
    return results


def shadow_summary(reports: list[dict]) -> dict:
    """Aggregate Agent 11 shadow variant reports across days."""
    if not reports:
        return {"status": "no data"}

    aggregated: dict[str, dict] = {}
    for r in reports:
        for key, v in r.get("variants", {}).items():
            if key not in aggregated:
                aggregated[key] = {
                    "param_name": v["param_name"],
                    "variant_value": v["variant_value"],
                    "extra_signals": 0, "extra_wins": 0,
                    "extra_losses": 0, "extra_pnl_sum": 0.0,
                }
            a = aggregated[key]
            a["extra_signals"] += v.get("extra_signals", 0)
            a["extra_wins"]    += v.get("extra_wins", 0)
            a["extra_losses"]  += v.get("extra_losses", 0)
            a["extra_pnl_sum"] += v.get("extra_pnl_sum", 0.0)

    # Rank by net P&L
    ranked = sorted(aggregated.items(), key=lambda x: x[1]["extra_pnl_sum"], reverse=True)

    recs = []
    for key, v in ranked[:3]:
        n = v["extra_wins"] + v["extra_losses"]
        if n < 5:
            continue
        wr = v["extra_wins"] / max(n, 1) * 100
        recs.append(
            f"{v['param_name']} → {v['variant_value']}: "
            f"{v['extra_signals']} extra signals over {len(reports)} days, "
            f"WR={wr:.0f}% ({v['extra_wins']}W/{v['extra_losses']}L), "
            f"cumulative {v['extra_pnl_sum']:+.1f}%"
        )

    return {
        "days_analyzed": len(reports),
        "top_variants":  recs or ["Insufficient data (need >= 5 resolved signals per variant)"],
        "current_settings": reports[-1].get("current_settings", {}),
    }


def lifecycle_summary(reports: list[dict]) -> dict:
    """Aggregate Agent 12 lifecycle audit reports across days."""
    if not reports:
        return {"status": "no data"}

    totals = defaultdict(int)
    all_recs = []
    strategy_issues = defaultdict(lambda: {"total": 0, "premature_loss": 0, "late_exit": 0})

    for r in reports:
        s = r.get("summary", {})
        for k, v in s.items():
            if isinstance(v, (int, float)) and k != "avg_recovery_missed_pct":
                totals[k] += v

        for rec in r.get("recommendations", []):
            if "REQUIRES YOUR REVIEW" in rec:
                all_recs.append(rec)

        for strat, issue in r.get("strategy_issues", {}).items():
            si = strategy_issues[strat]
            si["total"]          += issue.get("total", 0)
            si["premature_loss"] += issue.get("premature_loss", 0)
            si["late_exit"]      += issue.get("late_exit", 0)

    total_exits = totals.get("correct_sl",0) + totals.get("premature_loss_exit",0) + \
                  totals.get("good_exit",0) + totals.get("premature_profit_exit",0) + \
                  totals.get("late_exit",0)

    persistent_issues = []
    for strat, si in strategy_issues.items():
        if si["total"] >= 5:
            bad_rate = (si["premature_loss"] + si["late_exit"]) / si["total"]
            if bad_rate > 0.35:
                persistent_issues.append(
                    f"{strat}: {si['premature_loss']+si['late_exit']}/{si['total']} "
                    f"problematic exits ({bad_rate:.0%}) over {len(reports)} days"
                )

    return {
        "days_analyzed":      len(reports),
        "total_exits":        total_exits,
        "breakdown":          dict(totals),
        "recurring_warnings": list(set(all_recs))[:5],
        "persistent_strategy_issues": persistent_issues or ["None detected"],
    }


def main():
    print(f"\n{'='*60}")
    print(f"  WEEKLY OBSERVER REPORT — Agent 11 + Agent 12")
    print(f"  Generated: {date.today().isoformat()}")
    print(f"{'='*60}\n")

    shadow_reports    = load_json_files("shadow_variants_*.json")
    lifecycle_reports = load_json_files("lifecycle_audit_*.json")

    print("─── AGENT 11: SHADOW PARAMETER ANALYSIS ───\n")
    ss = shadow_summary(shadow_reports)
    if ss.get("status") == "no data":
        print("  No shadow data yet. Agent runs during live trading.\n")
    else:
        print(f"  Days analyzed: {ss['days_analyzed']}")
        print(f"  Current settings: {ss['current_settings']}")
        print(f"\n  Top parameter variants by cumulative P&L:")
        for rec in ss["top_variants"]:
            print(f"    • {rec}")
        print()

    print("─── AGENT 12: TRADE LIFECYCLE AUDIT ───\n")
    ls = lifecycle_summary(lifecycle_reports)
    if ls.get("status") == "no data":
        print("  No lifecycle data yet. Agent runs during live trading.\n")
    else:
        print(f"  Days analyzed: {ls['days_analyzed']}")
        print(f"  Total exits tracked: {ls['total_exits']}")
        print(f"  Breakdown: {ls['breakdown']}")
        print(f"\n  Recurring warnings:")
        for w in ls["recurring_warnings"]:
            print(f"    ⚠️  {w}")
        print(f"\n  Persistent strategy issues:")
        for issue in ls["persistent_strategy_issues"]:
            print(f"    • {issue}")
        print()

    print(f"{'='*60}")
    print("  IMPORTANT: All recommendations require YOUR manual review.")
    print("  No settings have been changed automatically.")
    print(f"{'='*60}\n")

    # Save combined report
    out = {
        "date":      date.today().isoformat(),
        "shadow":    ss,
        "lifecycle": ls,
    }
    out_path = Path(JOURNAL_DIR) / f"weekly_observer_report_{date.today().isoformat()}.json"
    Path(JOURNAL_DIR).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: {out_path}\n")


if __name__ == "__main__":
    main()

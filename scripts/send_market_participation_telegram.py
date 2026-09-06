#!/usr/bin/env python3
"""Send market participation snapshot to Telegram."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IST = pytz.timezone("Asia/Kolkata")
SNAPSHOT_PATH = PROJECT_ROOT / "data" / "market_participation" / "latest.json"


def _fmt_num(value: Any, digits: int = 0, signed: bool = False) -> str:
    try:
        if value is None or value == "":
            return "-"
        num = float(value)
    except Exception:
        return "-"
    sign = "+" if signed and num > 0 else ""
    if digits <= 0:
        return f"{sign}{num:,.0f}"
    return f"{sign}{num:,.{digits}f}"


def _load_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"FAIL market participation snapshot not found: {path}")
    data = json.loads(path.read_text(errors="replace"))
    if not isinstance(data, dict):
        raise SystemExit(f"FAIL invalid snapshot: {path}")
    return data


def _refresh_snapshot(args: argparse.Namespace) -> None:
    command = [
        str(PROJECT_ROOT / "scripts" / "run_upstox_market_participation.sh"),
        "--source",
        args.source,
        "--nse-lookback-days",
        str(args.nse_lookback_days),
        "--timeout",
        str(args.timeout),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _carry_line(name: str, row: dict[str, Any], *, net_label: str = "OI") -> str:
    bias = str(row.get("bias") or "-")
    return (
        f"{name:<6} "
        f"C {_fmt_num(row.get('call_buy'))}/{_fmt_num(row.get('call_sell'))} "
        f"Net {_fmt_num(row.get('net_call'), signed=True)} | "
        f"P {_fmt_num(row.get('put_buy'))}/{_fmt_num(row.get('put_sell'))} "
        f"Net {_fmt_num(row.get('net_put'), signed=True)} | "
        f"{net_label} {_fmt_num(row.get('net_oi'), signed=True)} | {bias}"
    )


def _build_message(data: dict[str, Any], label: str) -> str:
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    score = summary.get("market_score", {}) if isinstance(summary.get("market_score"), dict) else {}
    activity = summary.get("daily_index_option_activity", {}) if isinstance(summary.get("daily_index_option_activity"), dict) else {}
    carry = summary.get("options_carryforward", {}) if isinstance(summary.get("options_carryforward"), dict) else {}
    direction_view = summary.get("participant_direction_view", {}) if isinstance(summary.get("participant_direction_view"), dict) else {}
    notes = [str(note) for note in summary.get("notes", []) if str(note).strip()]
    fetched_at = data.get("fetched_at")
    try:
        fetched_label = datetime.fromisoformat(str(fetched_at)).astimezone(IST).strftime("%d %b %H:%M IST")
    except Exception:
        fetched_label = datetime.now(IST).strftime("%d %b %H:%M IST")

    header = "Pre-market" if label == "morning" else "Post-market"
    source = str(data.get("active_source") or data.get("source") or "-").upper()
    report_date = str((data.get("nse") or {}).get("report_date") or data.get("date") or "-")
    lines = [
        f"SignalForge Market Intelligence - {header}",
        f"Snapshot: {fetched_label} | Source: {source} | Report: {report_date}",
        f"Market: {summary.get('market_side', 'UNKNOWN')} | Score: {_fmt_num(score.get('percent'), 1)}% | {score.get('recommendation', '-')}",
        "",
    ]
    lines.extend(notes[:8])
    lines.extend([
        "",
        "Derived participant direction",
        f"FII: {((direction_view.get('FII') or {}).get('direction') or '-')}",
        f"DII: {((direction_view.get('DII') or {}).get('direction') or '-')}",
        f"PRO: {((direction_view.get('PRO') or {}).get('direction') or '-')}",
        f"CLIENT: {((direction_view.get('CLIENT') or {}).get('direction') or '-')}",
    ])
    lines.extend([
        "",
        "Index option daily buy/sell contracts",
        _carry_line("FII", activity.get("FII", {}) if isinstance(activity.get("FII"), dict) else {}, net_label="Net"),
        _carry_line("DII", activity.get("DII", {}) if isinstance(activity.get("DII"), dict) else {}, net_label="Net"),
        _carry_line("PRO", activity.get("PRO", {}) if isinstance(activity.get("PRO"), dict) else {}, net_label="Net"),
        _carry_line("CLIENT", activity.get("CLIENT", {}) if isinstance(activity.get("CLIENT"), dict) else {}, net_label="Net"),
        "",
        "Index option carry-forward net OI",
        _carry_line("FII", carry.get("FII", {}) if isinstance(carry.get("FII"), dict) else {}),
        _carry_line("DII", carry.get("DII", {}) if isinstance(carry.get("DII"), dict) else {}),
        _carry_line("PRO", carry.get("PRO", {}) if isinstance(carry.get("PRO"), dict) else {}),
        _carry_line("CLIENT", carry.get("CLIENT", {}) if isinstance(carry.get("CLIENT"), dict) else {}),
        "",
        "Dashboard context only; strategies do not read this feed.",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Send market participation snapshot to Telegram.")
    parser.add_argument("--label", choices=("morning", "evening"), default="morning")
    parser.add_argument("--snapshot", default=str(SNAPSHOT_PATH))
    parser.add_argument("--refresh", action="store_true", help="Refresh snapshot before sending")
    parser.add_argument("--source", choices=("auto", "upstox", "nse"), default="auto")
    parser.add_argument("--nse-lookback-days", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=12.0)
    args = parser.parse_args()

    if args.refresh:
        _refresh_snapshot(args)

    data = _load_snapshot(Path(args.snapshot).expanduser().resolve())
    message = _build_message(data, args.label)
    from utils.telegram_notifier import get_notifier

    ok = asyncio.run(get_notifier().send_text(message, target="LIVE", parse_mode=None))
    print("telegram_sent" if ok else "telegram_skipped_or_failed")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

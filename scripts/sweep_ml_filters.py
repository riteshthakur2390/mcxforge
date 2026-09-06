"""
Sweep ML thresholds and strategy-pair filter presets against the existing backtest.

Goal:
Find configurations that increase approved/planned trade count while keeping
approved-trade precision and realized PnL acceptable.

Example:
    ./venv/bin/python scripts/sweep_ml_filters.py \
        --start-date 2026-02-03 \
        --end-date 2026-04-02 \
        --focused
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "backtesting" / "results"


PAIR_PRESETS = {
    "strict_trend": "SuperTrend+RSI|ADX+PSAR",
    "balanced_trend": (
        "SuperTrend+RSI|ADX+PSAR,"
        "SuperTrend+RSI|BBSqueeze"
    ),
    "expanded_trend": (
        "SuperTrend+RSI|ADX+PSAR,"
        "SuperTrend+RSI|BBSqueeze,"
        "ADX+PSAR|BBSqueeze"
    ),
}

BLOCK_PRESETS = {
    "default_block": "BBSqueeze|ADX+PSAR",
    "no_blocks": "",
}

FOCUSED_GRID = {
    "ML_THRESHOLD_OVERRIDE": ["0.54", "0.56", "0.58"],
    "ML_SECONDARY_THRESHOLD": ["0.50", "0.52"],
    "ML_SECONDARY_MIN_RAW_CONFIDENCE": ["0.72", "0.74"],
    "SIGNAL_APPROVAL_DAILY_BUDGET": ["1", "2"],
    "MAX_TRADES_PER_DAY": ["1", "2"],
    "PAIR_PRESET": ["strict_trend", "balanced_trend"],
    "BLOCK_PRESET": ["default_block", "no_blocks"],
}

FULL_GRID = {
    "ML_THRESHOLD_OVERRIDE": ["0.52", "0.54", "0.56", "0.58", "0.60"],
    "ML_SECONDARY_THRESHOLD": ["0.48", "0.50", "0.52", "0.54"],
    "ML_SECONDARY_MIN_RAW_CONFIDENCE": ["0.70", "0.72", "0.74"],
    "SIGNAL_APPROVAL_DAILY_BUDGET": ["1", "2", "3"],
    "MAX_TRADES_PER_DAY": ["1", "2"],
    "PAIR_PRESET": ["strict_trend", "balanced_trend", "expanded_trend"],
    "BLOCK_PRESET": ["default_block", "no_blocks"],
}


def run_case(args, overrides: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(overrides)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "backtest_runner.py"),
        "--json",
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--speed",
        "0",
    ]
    if args.ignore_regime:
        cmd.append("--ignore-regime")

    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "backtest failed")

    payload = json.loads(result.stdout)
    summary = payload.get("summary", {})
    return {
        **overrides,
        "raw_signals": int(summary.get("raw_signals", 0) or 0),
        "approved": int(summary.get("approved", 0) or 0),
        "planned": int(summary.get("planned", 0) or 0),
        "closed": int(summary.get("closed", 0) or 0),
        "wins": int(summary.get("wins", 0) or 0),
        "losses": int(summary.get("losses", 0) or 0),
        "approved_precision_pct": float(summary.get("approved_precision_pct", 0.0) or 0.0),
        "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
        "approved_to_planned_rate": float(summary.get("approved_to_planned_rate", 0.0) or 0.0),
        "planned_to_closed_rate": float(summary.get("planned_to_closed_rate", 0.0) or 0.0),
        "capital_return_pct": float(summary.get("capital_return_pct", 0.0) or 0.0),
        "aggregate_trade_pnl_pct": float(summary.get("aggregate_trade_pnl_pct", 0.0) or 0.0),
        "net_realized_pnl": float(summary.get("net_realized_pnl", 0.0) or 0.0),
        "transaction_costs": float(summary.get("transaction_costs", 0.0) or 0.0),
        "selected_trades": int(summary.get("selected_trades", 0) or 0),
    }


def build_cases(grid: dict[str, list[str]]) -> list[dict[str, str]]:
    keys = list(grid.keys())
    cases: list[dict[str, str]] = []
    for values in product(*(grid[key] for key in keys)):
        raw_case = dict(zip(keys, values))
        pair_preset = raw_case.pop("PAIR_PRESET")
        block_preset = raw_case.pop("BLOCK_PRESET")
        raw_case["ML_SECONDARY_ALLOWED_STRATEGY_PAIRS"] = PAIR_PRESETS[pair_preset]
        raw_case["ML_BLOCKED_STRATEGY_PAIRS"] = BLOCK_PRESETS[block_preset]
        raw_case["_pair_preset"] = pair_preset
        raw_case["_block_preset"] = block_preset
        cases.append(raw_case)
    return cases


def qualifies(row: dict, min_precision: float, target_approved: int) -> bool:
    return (
        row["approved_precision_pct"] >= min_precision
        and row["approved"] >= target_approved
        and row["planned"] >= target_approved
        and row["closed"] >= target_approved
        and row["net_realized_pnl"] > 0
    )


def case_sort_key(row: dict, min_precision: float, target_approved: int):
    quality_hit = qualifies(row, min_precision=min_precision, target_approved=target_approved)
    return (
        int(quality_hit),
        row["closed"],
        row["approved_precision_pct"],
        row["net_realized_pnl"],
        row["capital_return_pct"],
        row["approved"],
    )


def save_rows(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ml_filter_sweep_{timestamp}.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ML/filter settings for SignalForge")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--ignore-regime", action="store_true")
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--min-precision", type=float, default=55.0)
    parser.add_argument("--target-approved", type=int, default=10)
    args = parser.parse_args()

    grid = FOCUSED_GRID if args.focused else FULL_GRID
    cases = build_cases(grid)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    rows: list[dict] = []
    for index, overrides in enumerate(cases, start=1):
        printable = {
            "thr": overrides["ML_THRESHOLD_OVERRIDE"],
            "sec_thr": overrides["ML_SECONDARY_THRESHOLD"],
            "sec_raw": overrides["ML_SECONDARY_MIN_RAW_CONFIDENCE"],
            "daily": overrides["SIGNAL_APPROVAL_DAILY_BUDGET"],
            "max_trades": overrides["MAX_TRADES_PER_DAY"],
            "pairs": overrides["_pair_preset"],
            "blocks": overrides["_block_preset"],
        }
        try:
            row = run_case(args, overrides)
            rows.append(row)
            print(
                f"[{index}/{len(cases)}] {printable} -> "
                f"approved={row['approved']} planned={row['planned']} closed={row['closed']} "
                f"precision={row['approved_precision_pct']:.1f}% "
                f"net_inr={row['net_realized_pnl']:.0f}"
            )
        except Exception as exc:
            print(f"[{index}/{len(cases)}] {printable} -> ERROR: {exc}")

    if not rows:
        print("No successful sweep rows.")
        return

    rows.sort(
        key=lambda row: case_sort_key(
            row,
            min_precision=args.min_precision,
            target_approved=args.target_approved,
        ),
        reverse=True,
    )
    out_path = save_rows(rows)
    qualifying = [
        row for row in rows
        if qualifies(
            row,
            min_precision=args.min_precision,
            target_approved=args.target_approved,
        )
    ]

    print("\nTop configurations")
    for row in rows[: max(args.top, 1)]:
        print(
            f"pairs={row['_pair_preset']} blocks={row['_block_preset']} | "
            f"thr={row['ML_THRESHOLD_OVERRIDE']} sec={row['ML_SECONDARY_THRESHOLD']} "
            f"sec_raw={row['ML_SECONDARY_MIN_RAW_CONFIDENCE']} | "
            f"approved={row['approved']} planned={row['planned']} closed={row['closed']} | "
            f"precision={row['approved_precision_pct']:.1f}% | "
            f"capital={row['capital_return_pct']:.2f}% | "
            f"trade_sum={row['aggregate_trade_pnl_pct']:.2f}% | "
            f"net_inr={row['net_realized_pnl']:.0f}"
        )

    print(
        f"\nQualifying rows: {len(qualifying)} / {len(rows)} | "
        f"target approved >= {args.target_approved} | "
        f"precision >= {args.min_precision:.1f}%"
    )
    print(f"Saved results: {out_path}")


if __name__ == "__main__":
    main()

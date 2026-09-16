"""
Parameter sweep for exit/risk settings using the existing backtest runner.

Example:
    ./venv/bin/python scripts/tune_exit_params.py --start-date 2026-02-03 --end-date 2026-04-02
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


GRID = {
    "ATR_STOP_MULTIPLIER": ["1.00", "1.15", "1.30"],
    "ATR_TARGET_MULTIPLIER": ["1.80", "2.10", "2.40"],
    "BREAKEVEN_R_TRIGGER": ["0.80", "1.00", "1.20"],
    "TIME_STOP_MINUTES": ["20", "25", "35"],
    "TIME_STOP_MIN_PNL_PCT": ["4.0", "6.0", "8.0"],
}

FOCUSED_GRID = {
    "ATR_STOP_MULTIPLIER": ["1.00", "1.15"],
    "ATR_TARGET_MULTIPLIER": ["2.10", "2.40"],
    "BREAKEVEN_R_TRIGGER": ["0.80", "1.00"],
    "TIME_STOP_MINUTES": ["20", "25"],
    "TIME_STOP_MIN_PNL_PCT": ["4.0", "6.0"],
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
        "approved": int(summary.get("approved", 0) or 0),
        "wins": int(summary.get("wins", 0) or 0),
        "losses": int(summary.get("losses", 0) or 0),
        "win_rate": float(summary.get("win_rate", 0.0) or 0.0),
        "capital_return_pct": float(summary.get("capital_return_pct", 0.0) or 0.0),
        "net_realized_pnl": float(summary.get("net_realized_pnl", 0.0) or 0.0),
        "transaction_costs": float(summary.get("transaction_costs", 0.0) or 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep exit parameters for MCXForge")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--ignore-regime", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--focused", action="store_true")
    args = parser.parse_args()

    grid = FOCUSED_GRID if args.focused else GRID
    keys = list(grid.keys())
    rows = []
    for values in product(*(grid[k] for k in keys)):
        overrides = dict(zip(keys, values))
        try:
            row = run_case(args, overrides)
            rows.append(row)
            print(
                f"{overrides} -> trades={row['approved']} "
                f"wr={row['win_rate']:.1f}% pnl={row['net_realized_pnl']:.0f}"
            )
        except Exception as exc:
            print(f"{overrides} -> ERROR: {exc}")

    rows.sort(
        key=lambda r: (
            r["net_realized_pnl"],
            r["capital_return_pct"],
            r["win_rate"],
            r["approved"],
        ),
        reverse=True,
    )

    print("\nTop configurations")
    for row in rows[: max(args.top, 1)]:
        print(
            f"{ {k: row[k] for k in keys} } | "
            f"approved={row['approved']} | "
            f"wr={row['win_rate']:.1f}% | "
            f"capital={row['capital_return_pct']:.2f}% | "
            f"net_inr={row['net_realized_pnl']:.0f} | "
            f"costs={row['transaction_costs']:.0f}"
        )


if __name__ == "__main__":
    main()

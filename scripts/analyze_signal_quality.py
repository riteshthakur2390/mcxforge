"""
Analyze SignalForge journal quality at the signal level.

Focuses on deduplicated CLOSED rows so replay reruns and lifecycle noise
do not distort strategy win-rate analysis.
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from dataclasses import dataclass


DEFAULT_PATTERNS = [
    "journal/signals_2026-*.csv",
    "journal/old_csv/signals_2026-*.csv",
]


@dataclass
class BucketStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_sum: float = 0.0
    realized_sum: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0

    def add(self, *, is_win: bool, pnl_pct: float, realized_pnl: float) -> None:
        self.trades += 1
        self.wins += int(is_win)
        self.losses += int(not is_win)
        self.pnl_sum += pnl_pct
        self.realized_sum += realized_pnl
        if realized_pnl > 0:
            self.gross_profit += realized_pnl
        elif realized_pnl < 0:
            self.gross_loss += abs(realized_pnl)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100.0) if self.trades else 0.0

    @property
    def avg_pnl(self) -> float:
        return (self.pnl_sum / self.trades) if self.trades else 0.0

    @property
    def avg_realized(self) -> float:
        return (self.realized_sum / self.trades) if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss <= 0:
            return 999.0 if self.gross_profit > 0 else 0.0
        return self.gross_profit / self.gross_loss


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _row_key(row: dict) -> str:
    signal_id = (row.get("signal_id") or "").strip()
    if signal_id:
        return f"id:{signal_id}"
    return "|".join(
        [
            str(row.get("date", "")),
            str(row.get("time", "")),
            str(row.get("direction", "")),
            str(row.get("option_symbol", "")),
            str(row.get("strategies_fired", "")),
        ]
    )


def _strategy_parts(raw: str) -> list[str]:
    parts = [part.strip() for part in str(raw or "").split("|") if part.strip()]
    return parts or ["(blank)"]


def load_closed_rows(patterns: list[str]) -> list[dict]:
    rows: list[dict] = []
    files: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path not in seen:
                files.append(path)
                seen.add(path)

    for path in files:
        with open(path, newline="") as fh:
            rows.extend(csv.DictReader(fh))

    closed = [
        row for row in rows
        if row.get("outcome_eod") in {"WIN", "LOSS"}
        and row.get("lifecycle_status", "") in {"", "CLOSED"}
    ]

    latest: dict[str, dict] = {}
    for row in closed:
        latest[_row_key(row)] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("date", "")),
            str(row.get("time", "")),
            str(row.get("option_symbol", "")),
        ),
    )


def accumulate(rows: list[dict], explode_strategies: bool = False) -> dict[str, BucketStats]:
    buckets: dict[str, BucketStats] = defaultdict(BucketStats)
    for row in rows:
        pnl_pct = _safe_float(row.get("pnl_pct"))
        realized_pnl = _safe_float(row.get("realized_pnl"))
        is_win = row.get("outcome_eod") == "WIN"
        keys = (
            _strategy_parts(row.get("strategies_fired", ""))
            if explode_strategies
            else [str(row.get("strategies_fired", "") or "(blank)")]
        )
        for key in keys:
            buckets[key].add(
                is_win=is_win,
                pnl_pct=pnl_pct,
                realized_pnl=realized_pnl,
            )
    return buckets


def print_table(
    title: str,
    buckets: dict[str, BucketStats],
    *,
    total_losses: int,
    min_trades: int,
    limit: int,
) -> None:
    print()
    print(title)
    print("-" * len(title))
    ranked = [
        (name, stats)
        for name, stats in buckets.items()
        if stats.trades >= min_trades
    ]
    ranked.sort(
        key=lambda item: (
            -item[1].avg_realized,
            -item[1].profit_factor,
            -item[1].win_rate,
            -item[1].trades,
            item[0],
        )
    )
    if not ranked:
        print("No rows after filters.")
        return

    print(
        "name | n | wr | avg_pnl% | avg_inr | pf | loss_share | pnl_sum | inr_sum"
    )
    shown = 0
    for name, stats in ranked:
        loss_share = (stats.losses / total_losses * 100.0) if total_losses else 0.0
        print(
            f"{name} | {stats.trades} | {stats.win_rate:5.1f}% | "
            f"{stats.avg_pnl:7.2f}% | {stats.avg_realized:7.2f} | "
            f"{stats.profit_factor:4.2f} | {loss_share:6.1f}% | "
            f"{stats.pnl_sum:7.2f}% | {stats.realized_sum:7.2f}"
        )
        shown += 1
        if limit and shown >= limit:
            break


def print_focus(rows: list[dict], focus: list[str]) -> None:
    focus_set = {item.strip() for item in focus if item.strip()}
    if not focus_set:
        return

    filtered = [
        row for row in rows
        if focus_set.intersection(_strategy_parts(row.get("strategies_fired", "")))
    ]
    if not filtered:
        print()
        print(f"Focus {sorted(focus_set)}")
        print("----------------")
        print("No matching closed trades found.")
        return

    total_losses = sum(1 for row in filtered if row.get("outcome_eod") == "LOSS")
    print_table(
        f"Focus combos for {', '.join(sorted(focus_set))}",
        accumulate(filtered, explode_strategies=False),
        total_losses=total_losses,
        min_trades=1,
        limit=20,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SignalForge journal quality")
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        help="Glob pattern for journal CSV files. Repeatable. Defaults to current and archived journals.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=2,
        help="Minimum trades required to show a strategy or combo bucket",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum rows to print per table",
    )
    parser.add_argument(
        "--focus",
        action="append",
        default=[],
        help="Specific strategy name to highlight, e.g. --focus VolumeProfile --focus LiqSweep",
    )
    args = parser.parse_args()

    patterns = args.patterns or DEFAULT_PATTERNS
    rows = load_closed_rows(patterns)
    if not rows:
        print("No closed rows found.")
        return

    wins = sum(1 for row in rows if row.get("outcome_eod") == "WIN")
    losses = len(rows) - wins
    pnl_sum = sum(_safe_float(row.get("pnl_pct")) for row in rows)
    realized_sum = sum(_safe_float(row.get("realized_pnl")) for row in rows)
    gross_profit = sum(max(_safe_float(row.get("realized_pnl")), 0.0) for row in rows)
    gross_loss = sum(abs(min(_safe_float(row.get("realized_pnl")), 0.0)) for row in rows)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

    print(
        f"closed={len(rows)} | win_rate={wins / len(rows) * 100:.2f}% | "
        f"pnl_sum={pnl_sum:.2f}% | realized_sum={realized_sum:.2f} | "
        f"profit_factor={profit_factor:.2f}"
    )

    print_table(
        "Per-strategy contribution",
        accumulate(rows, explode_strategies=True),
        total_losses=losses,
        min_trades=args.min_trades,
        limit=args.limit,
    )
    print_table(
        "Per-combo contribution",
        accumulate(rows, explode_strategies=False),
        total_losses=losses,
        min_trades=args.min_trades,
        limit=args.limit,
    )
    exit_buckets: dict[str, BucketStats] = defaultdict(BucketStats)
    for row in rows:
        exit_buckets[str(row.get("exit_reason", "") or "(blank)")].add(
            is_win=row.get("outcome_eod") == "WIN",
            pnl_pct=_safe_float(row.get("pnl_pct")),
            realized_pnl=_safe_float(row.get("realized_pnl")),
        )
    print_table(
        "Exit reason contribution",
        exit_buckets,
        total_losses=losses,
        min_trades=args.min_trades,
        limit=args.limit,
    )

    print_focus(rows, args.focus)


if __name__ == "__main__":
    main()

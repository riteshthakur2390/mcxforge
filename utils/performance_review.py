"""
Shared live-vs-backtest performance review utilities.
"""

from __future__ import annotations

import csv
import glob
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from config.settings import (
    AUTOMATION_MAX_DRAWDOWN_PCT,
    AUTOMATION_MAX_WIN_RATE_GAP_PCT,
    AUTOMATION_MIN_BACKTEST_TRADES,
    AUTOMATION_MIN_LIVE_TRADES,
    AUTOMATION_MIN_PROFIT_FACTOR,
    AUTOMATION_MIN_WIN_RATE_PCT,
    PAPER_TRADING_CAPITAL,
)


@dataclass
class ReviewMetrics:
    source: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    pnl_sum_pct: float
    net_realized_pnl: float
    avg_trade_pct: float
    avg_realized_pnl: float
    profit_factor: float
    max_drawdown_pct: float
    avg_holding_minutes: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate_pct": self.win_rate_pct,
            "pnl_sum_pct": self.pnl_sum_pct,
            "net_realized_pnl": self.net_realized_pnl,
            "avg_trade_pct": self.avg_trade_pct,
            "avg_realized_pnl": self.avg_realized_pnl,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_holding_minutes": self.avg_holding_minutes,
        }


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(float(value or 0))
    except Exception:
        return 0


def _row_key(row: dict) -> str:
    signal_id = str(row.get("signal_id", "") or "").strip()
    if signal_id:
        return signal_id
    return "|".join(
        [
            str(row.get("date", "")),
            str(row.get("time", "")),
            str(row.get("direction", "")),
            str(row.get("option_symbol", "")),
            str(row.get("strategies_fired", "")),
        ]
    )


def _load_rows(patterns: Iterable[str]) -> list[dict]:
    files: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if path not in seen:
                files.append(path)
                seen.add(path)

    rows: list[dict] = []
    for path in files:
        try:
            with open(path, newline="") as fh:
                rows.extend(csv.DictReader(fh))
        except Exception:
            continue
    return rows


def _closed_rows(patterns: Iterable[str]) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in _load_rows(patterns):
        if str(row.get("lifecycle_status", "")).upper() != "CLOSED":
            continue
        if str(row.get("outcome_eod", "")).upper() not in {"WIN", "LOSS"}:
            continue
        latest[_row_key(row)] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("date", "")),
            str(row.get("time", "")),
            str(row.get("option_symbol", "")),
        ),
    )


def _compute_metrics(source: str, rows: list[dict]) -> ReviewMetrics:
    if not rows:
        return ReviewMetrics(
            source=source,
            trades=0,
            wins=0,
            losses=0,
            win_rate_pct=0.0,
            pnl_sum_pct=0.0,
            net_realized_pnl=0.0,
            avg_trade_pct=0.0,
            avg_realized_pnl=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
            avg_holding_minutes=0.0,
        )

    df = pd.DataFrame(rows).copy()
    df["realized_pnl"] = pd.to_numeric(df.get("realized_pnl"), errors="coerce").fillna(0.0)
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct"), errors="coerce").fillna(0.0)
    df["holding_minutes"] = pd.to_numeric(df.get("holding_minutes"), errors="coerce").fillna(0.0)
    wins = int((df["outcome_eod"].astype(str) == "WIN").sum())
    losses = len(df) - wins
    gross_profit = float(df.loc[df["realized_pnl"] > 0, "realized_pnl"].sum())
    gross_loss = abs(float(df.loc[df["realized_pnl"] < 0, "realized_pnl"].sum()))
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    exit_ts = pd.to_datetime(df.get("exit_time"), errors="coerce")
    if exit_ts.notna().any():
        daily = (
            df.assign(exit_date=exit_ts.dt.date)
            .groupby("exit_date", sort=True)["realized_pnl"]
            .sum()
        )
        baseline = max(float(PAPER_TRADING_CAPITAL or 0.0), 1.0)
        equity = daily.cumsum() + baseline
        peak = equity.cummax()
        drawdown_pct = ((equity - peak) / peak.replace(0, pd.NA) * 100.0).fillna(0.0)
        max_drawdown_pct = round(abs(float(drawdown_pct.min() or 0.0)), 2)
    else:
        max_drawdown_pct = 0.0

    return ReviewMetrics(
        source=source,
        trades=len(df),
        wins=wins,
        losses=losses,
        win_rate_pct=round(wins / len(df) * 100, 2),
        pnl_sum_pct=round(float(df["pnl_pct"].sum()), 2),
        net_realized_pnl=round(float(df["realized_pnl"].sum()), 2),
        avg_trade_pct=round(float(df["pnl_pct"].mean()), 2),
        avg_realized_pnl=round(float(df["realized_pnl"].mean()), 2),
        profit_factor=profit_factor,
        max_drawdown_pct=max_drawdown_pct,
        avg_holding_minutes=round(float(df["holding_minutes"].mean()), 1),
    )


def _top_breakdowns(rows: list[dict], key: str, limit: int = 5) -> list[dict]:
    buckets: dict[str, dict] = defaultdict(lambda: {"trades": 0, "wins": 0, "net_realized_pnl": 0.0})
    for row in rows:
        name = str(row.get(key, "") or "(blank)")
        buckets[name]["trades"] += 1
        buckets[name]["wins"] += int(str(row.get("outcome_eod", "")).upper() == "WIN")
        buckets[name]["net_realized_pnl"] += _safe_float(row.get("realized_pnl"))
    ranked = sorted(
        buckets.items(),
        key=lambda item: (item[1]["net_realized_pnl"], item[1]["wins"], item[1]["trades"]),
        reverse=True,
    )
    return [
        {
            "name": name,
            "trades": stats["trades"],
            "win_rate_pct": round(stats["wins"] / stats["trades"] * 100, 1) if stats["trades"] else 0.0,
            "net_realized_pnl": round(float(stats["net_realized_pnl"]), 2),
        }
        for name, stats in ranked[:limit]
    ]


def build_live_vs_backtest_report(
    *,
    live_patterns: list[str] | None = None,
    backtest_patterns: list[str] | None = None,
) -> dict:
    live_patterns = live_patterns or [
        "journal/signals_*.csv",
        "journal/old_csv/signals_*.csv",
    ]
    backtest_patterns = backtest_patterns or [
        "backtesting/results/backtest_trades_*.csv",
    ]

    live_rows = _closed_rows(live_patterns)
    backtest_rows = _closed_rows(backtest_patterns)
    live = _compute_metrics("live", live_rows)
    backtest = _compute_metrics("backtest", backtest_rows)

    gaps = {
        "trade_count_gap": live.trades - backtest.trades,
        "win_rate_gap_pct": round(live.win_rate_pct - backtest.win_rate_pct, 2),
        "profit_factor_gap": round(live.profit_factor - backtest.profit_factor, 3),
        "net_realized_pnl_gap": round(live.net_realized_pnl - backtest.net_realized_pnl, 2),
    }

    readiness_checks = [
        {
            "rule": f"Live trades >= {AUTOMATION_MIN_LIVE_TRADES}",
            "passed": live.trades >= AUTOMATION_MIN_LIVE_TRADES,
        },
        {
            "rule": f"Backtest trades >= {AUTOMATION_MIN_BACKTEST_TRADES}",
            "passed": backtest.trades >= AUTOMATION_MIN_BACKTEST_TRADES,
        },
        {
            "rule": f"Live win rate >= {AUTOMATION_MIN_WIN_RATE_PCT:.1f}%",
            "passed": live.win_rate_pct >= AUTOMATION_MIN_WIN_RATE_PCT,
        },
        {
            "rule": f"Live profit factor >= {AUTOMATION_MIN_PROFIT_FACTOR:.2f}",
            "passed": live.profit_factor >= AUTOMATION_MIN_PROFIT_FACTOR,
        },
        {
            "rule": f"Live max drawdown <= {AUTOMATION_MAX_DRAWDOWN_PCT:.1f}",
            "passed": live.max_drawdown_pct <= AUTOMATION_MAX_DRAWDOWN_PCT,
        },
        {
            "rule": f"Live/backtest win-rate gap <= {AUTOMATION_MAX_WIN_RATE_GAP_PCT:.1f}%",
            "passed": abs(gaps["win_rate_gap_pct"]) <= AUTOMATION_MAX_WIN_RATE_GAP_PCT,
        },
    ]

    return {
        "generated_at": datetime.now().isoformat(),
        "live": live.to_dict(),
        "backtest": backtest.to_dict(),
        "gaps": gaps,
        "top_live_combos": _top_breakdowns(live_rows, "strategies_fired"),
        "top_backtest_combos": _top_breakdowns(backtest_rows, "strategies_fired"),
        "automation_readiness": {
            "ready": all(item["passed"] for item in readiness_checks),
            "checks": readiness_checks,
            "policy": "Manual-first. Enable auto-execution only after live metrics stay close to backtest and pass all readiness gates.",
        },
    }

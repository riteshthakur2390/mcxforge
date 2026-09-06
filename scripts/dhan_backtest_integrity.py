#!/usr/bin/env python3
"""Dhan backtest data integrity and benchmark parity checks."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")
ROOT = Path(__file__).resolve().parents[1]


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return _normalize_df(df)


def _read_git_parquet(ref: str, path: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
        tmp.write(subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT))
        tmp.flush()
        return _normalize_df(pd.read_parquet(tmp.name))


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "timestamp" in work.columns:
        idx = pd.to_datetime(work.pop("timestamp"), format="mixed", errors="coerce")
    else:
        idx = pd.to_datetime(work.index, format="mixed", errors="coerce")
    work = work.loc[~idx.isna()].copy()
    idx = pd.DatetimeIndex(idx[~idx.isna()])
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)
    work.index = idx
    return work.sort_index()


def _load_sqlite(
    db_path: Path,
    *,
    broker: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    query = """
        SELECT ts, open, high, low, close, volume, oi, source
        FROM candles
        WHERE symbol='NIFTY'
          AND interval='5minute'
          AND broker=?
          AND ts >= ?
          AND ts < ?
        ORDER BY ts
    """
    with sqlite3.connect(db_path, timeout=30) as conn:
        df = pd.read_sql_query(query, conn, params=(broker, start, end))
    if df.empty:
        return df
    idx = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), utc=True)).tz_convert(IST)
    df.index = idx
    return df.sort_index()


def _coverage(label: str, df: pd.DataFrame) -> None:
    print(f"\n{label}")
    if df.empty:
        print("  missing")
        return
    days = sorted({idx.date() for idx in df.index})
    print(f"  rows={len(df)} days={len(days)} range={df.index.min()} -> {df.index.max()}")
    if "source" in df.columns:
        print(f"  sources={dict(Counter(df['source'].astype(str)).most_common(8))}")


def _row_at(df: pd.DataFrame, ts: pd.Timestamp) -> dict[str, float | int | str] | None:
    if df.empty or ts not in df.index:
        return None
    row = df.loc[ts]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    out: dict[str, float | int | str] = {}
    for col in ("open", "high", "low", "close", "volume", "source"):
        if col in row:
            value = row[col]
            out[col] = str(value) if col == "source" else float(value)
    return out


def _print_timestamp_parity(sources: dict[str, pd.DataFrame], timestamps: Iterable[str]) -> None:
    print("\nTimestamp parity")
    for raw in timestamps:
        ts = pd.Timestamp(raw, tz=IST)
        print(f"\n{raw}")
        base_close = None
        for label, df in sources.items():
            row = _row_at(df, ts)
            if row is None:
                print(f"  {label}: missing")
                continue
            if base_close is None:
                base_close = float(row.get("close", 0.0) or 0.0)
            delta = float(row.get("close", 0.0) or 0.0) - base_close
            print(f"  {label}: {row} close_delta_vs_first={delta:.2f}")


def _trade_key(row: dict) -> tuple[str, str, str]:
    time_value = str(row.get("entry_time") or row.get("time") or "")
    return (str(row.get("date", "")), time_value[11:16], str(row.get("direction", "")))


def _read_trades(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _trade_diff(benchmark_csv: Path, current_csv: Path) -> None:
    old = _read_trades(benchmark_csv)
    new = _read_trades(current_csv)
    if not old or not new:
        print("\nTrade diff skipped: missing CSV")
        return
    min_new = min(row["date"] for row in new)
    max_new = max(row["date"] for row in new)
    old = [row for row in old if min_new <= row["date"] <= max_new]
    old_by = {_trade_key(row): row for row in old}
    new_by = {_trade_key(row): row for row in new}
    print(f"\nTrade diff window {min_new} -> {max_new}")
    for label, rows in (("benchmark", old), ("current", new)):
        pnl = sum(float(row.get("realized_pnl") or 0.0) for row in rows)
        print(f"  {label}: trades={len(rows)} pnl={pnl:.2f}")
    missing = [old_by[key] for key in sorted(set(old_by) - set(new_by))]
    extra = [new_by[key] for key in sorted(set(new_by) - set(old_by))]
    print("\nMissing benchmark winners")
    for row in sorted(missing, key=lambda r: float(r.get("realized_pnl") or 0.0), reverse=True)[:12]:
        print(
            f"  {row['date']} {str(row.get('entry_time') or row.get('time'))[11:16]} "
            f"{row['direction']} {row['exit_reason']} lots={row.get('lots')} "
            f"pnl={float(row.get('realized_pnl') or 0.0):.2f} {row.get('strategies_fired')}"
        )
    print("\nExtra current trades")
    for row in sorted(extra, key=lambda r: float(r.get("realized_pnl") or 0.0), reverse=True)[:12]:
        print(
            f"  {row['date']} {str(row.get('entry_time') or row.get('time'))[11:16]} "
            f"{row['direction']} {row['exit_reason']} lots={row.get('lots')} "
            f"pnl={float(row.get('realized_pnl') or 0.0):.2f} {row.get('strategies_fired')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-05-12T00:00:00+05:30")
    parser.add_argument("--end", default="2026-06-19T00:00:00+05:30")
    parser.add_argument("--benchmark-ref", default="b12e94e")
    parser.add_argument("--benchmark-csv", type=Path, default=ROOT / "backtesting/results/backtest_trades_20260615_231537.csv")
    parser.add_argument("--current-csv", type=Path, default=ROOT / "backtesting/results/backtest_trades_20260620_143724.csv")
    parser.add_argument("--db", type=Path, default=ROOT / "data/historical/market_history.sqlite3")
    parser.add_argument(
        "--timestamps",
        default="2026-05-14 11:30,2026-05-18 10:55,2026-05-29 13:00,2026-06-12 13:25",
    )
    args = parser.parse_args()

    sources = {
        "benchmark_upstox_parquet": _read_git_parquet(args.benchmark_ref, "data/cache/NIFTY_5minute_upstox.parquet"),
        "current_dhan_parquet": _read_parquet(ROOT / "data/cache/NIFTY_5minute_dhan.parquet"),
        "current_upstox_parquet": _read_parquet(ROOT / "data/cache/NIFTY_5minute_upstox.parquet"),
        "sqlite_dhan": _load_sqlite(args.db, broker="dhan", start=args.start, end=args.end),
        "sqlite_upstox": _load_sqlite(args.db, broker="upstox", start=args.start, end=args.end),
    }

    print("Coverage")
    for label, df in sources.items():
        _coverage(label, df)
    timestamps = [value.strip() for value in args.timestamps.split(",") if value.strip()]
    _print_timestamp_parity(sources, timestamps)
    _trade_diff(args.benchmark_csv, args.current_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

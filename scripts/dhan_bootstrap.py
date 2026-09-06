#!/usr/bin/env python3
"""Resumable Dhan-only historical data bootstrap for SignalForge."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import DATA_HIST_DB_PATH
from data.historical_store import HistoricalCandleStore

IST = pytz.timezone("Asia/Kolkata")
BASE_URL = "https://api.dhan.co/v2"
NIFTY_SECURITY_ID = 13


def date_chunks(start: date, end: date, days: int):
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=days), end)
        yield current, chunk_end
        current = chunk_end


class Bootstrap:
    def __init__(self, db_path: str, transport: str = "requests") -> None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        self.client_id = os.getenv("DHAN_CLIENT_ID", "").strip()
        self.token = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
        if not self.client_id or not self.token:
            raise SystemExit("DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN are required in .env")
        self.headers = {
            "access-token": self.token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.transport = transport
        self.resolve_ip = os.getenv("DHAN_BOOTSTRAP_RESOLVE_IP", "").strip()
        self.store = HistoricalCandleStore(db_path)
        self.db_path = Path(db_path)
        self._ensure_metadata()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=60)
        conn.execute("PRAGMA busy_timeout=60000")
        return conn

    def _ensure_metadata(self) -> None:
        with self._connect() as conn:
            existing = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name IN ('ingestion_jobs', 'data_coverage')
                    """
                ).fetchall()
            }
            if existing == {"ingestion_jobs", "data_coverage"}:
                return
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    dataset TEXT NOT NULL,
                    job_key TEXT NOT NULL,
                    range_start TEXT NOT NULL,
                    range_end TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rows_written INTEGER NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (dataset, job_key, range_start, range_end)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS data_coverage (
                    dataset TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    earliest_ts TEXT NOT NULL,
                    latest_ts TEXT NOT NULL,
                    rows INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    PRIMARY KEY (dataset, symbol, interval, source)
                )
                """
            )

    def _completed(self, dataset: str, key: str, start: date, end: date) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT status FROM ingestion_jobs
                WHERE dataset=? AND job_key=? AND range_start=? AND range_end=?
                """,
                (dataset, key, start.isoformat(), end.isoformat()),
            ).fetchone()
        return bool(row and row[0] == "complete")

    def _record_job(
        self,
        dataset: str,
        key: str,
        start: date,
        end: date,
        status: str,
        rows: int = 0,
        error: str = "",
    ) -> None:
        for attempt in range(8):
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO ingestion_jobs (
                            dataset, job_key, range_start, range_end, status,
                            rows_written, attempts, error, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                        ON CONFLICT(dataset, job_key, range_start, range_end) DO UPDATE SET
                            status=excluded.status,
                            rows_written=excluded.rows_written,
                            attempts=ingestion_jobs.attempts + 1,
                            error=excluded.error,
                            updated_at=excluded.updated_at
                        """,
                        (
                            dataset, key, start.isoformat(), end.isoformat(), status,
                            rows, error[:500], datetime.now(IST).isoformat(),
                        ),
                    )
                return
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if not any(
                    text in message
                    for text in ("locked", "locking protocol", "disk i/o error", "busy")
                ) or attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))

    def _post_with_curl(self, endpoint: str, payload: dict) -> dict:
        curl = shutil.which("curl")
        if not curl:
            raise RuntimeError("curl transport requested but curl was not found")
        cmd = [
            curl,
            "--silent",
            "--show-error",
            "--fail-with-body",
            "--location",
            "--max-time",
            "45",
            "--connect-timeout",
            "15",
        ]
        if self.resolve_ip:
            cmd.extend(["--resolve", f"api.dhan.co:443:{self.resolve_ip}"])
        cmd.extend([
            "--request",
            "POST",
            f"{BASE_URL}/{endpoint}",
            "--header",
            f"access-token: {self.token}",
            "--header",
            f"client-id: {self.client_id}",
            "--header",
            "Content-Type: application/json",
            "--header",
            "Accept: application/json",
            "--data",
            json.dumps(payload, separators=(",", ":")),
        ])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            body = (result.stdout or result.stderr or "").strip()
            raise RuntimeError(f"curl failed ({result.returncode}): {body[:300]}")
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"curl returned non-JSON response: {(result.stdout or '')[:300]}") from exc

    def _post(self, endpoint: str, payload: dict, retries: int = 5) -> dict:
        for attempt in range(retries):
            try:
                if self.transport == "curl":
                    return self._post_with_curl(endpoint, payload)
                response = requests.post(
                    f"{BASE_URL}/{endpoint}",
                    headers=self.headers,
                    json=payload,
                    timeout=45,
                )
            except requests.RequestException as exc:
                if attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 60))
                    continue
                raise RuntimeError(f"Dhan request failed: {exc}") from exc
            except RuntimeError:
                if attempt < retries - 1:
                    time.sleep(min(2 ** attempt, 60))
                    continue
                raise
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:
                wait = max(float(response.headers.get("Retry-After", 0) or 0), 2 ** attempt)
                time.sleep(min(wait, 60))
                continue
            if response.status_code in {500, 502, 503, 504} and attempt < retries - 1:
                time.sleep(min(2 ** attempt, 60))
                continue
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        raise RuntimeError("Dhan retries exhausted")

    @staticmethod
    def _parallel_arrays(data: dict) -> pd.DataFrame:
        timestamps = data.get("timestamp") or []
        if not timestamps:
            return pd.DataFrame()
        size = len(timestamps)

        def values(name: str, default=0):
            raw = data.get(name) or []
            return [raw[i] if i < len(raw) else default for i in range(size)]

        idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(IST)
        frame = pd.DataFrame(
            {
                "open": values("open"),
                "high": values("high"),
                "low": values("low"),
                "close": values("close"),
                "volume": values("volume"),
                "oi": values("open_interest"),
            },
            index=idx,
        )
        return frame.apply(pd.to_numeric, errors="coerce").fillna(0)

    def underlying(
        self,
        start: date,
        end: date,
        intervals: list[str],
        sleep_seconds: float,
    ) -> None:
        interval_map = {"1minute": "1", "5minute": "5", "15minute": "15", "60minute": "60"}
        for interval in intervals:
            if interval != "day" and interval not in interval_map:
                raise SystemExit(f"Unsupported underlying interval: {interval}")
            chunk_days = 365 if interval == "day" else 90
            for chunk_start, chunk_end in date_chunks(start, end, chunk_days):
                key = f"NIFTY:{interval}"
                if self._completed("underlying", key, chunk_start, chunk_end):
                    continue
                if interval == "day":
                    try:
                        source = self.store.load_candles(
                            symbol="NIFTY",
                            interval="5minute",
                            broker="dhan",
                            start=f"{chunk_start.isoformat()} 09:15:00",
                            end=f"{chunk_end.isoformat()} 15:30:00",
                        )
                        if source.empty:
                            raise RuntimeError("missing local Dhan 5minute source candles")
                        grouped = source.groupby(source.index.normalize())
                        frame = pd.DataFrame(
                            {
                                "open": grouped["open"].first(),
                                "high": grouped["high"].max(),
                                "low": grouped["low"].min(),
                                "close": grouped["close"].last(),
                                "volume": grouped["volume"].sum(),
                                "oi": grouped["oi"].last(),
                            }
                        ).dropna()
                        rows = self.store.upsert_candles(
                            frame,
                            symbol="NIFTY",
                            interval="day",
                            broker="dhan",
                            source="dhan_5minute_resample",
                        )
                        self._record_job("underlying", key, chunk_start, chunk_end, "complete", rows)
                        print(f"underlying day {chunk_start}..{chunk_end}: {rows} rows from local 5minute")
                    except Exception as exc:
                        self._record_job("underlying", key, chunk_start, chunk_end, "failed", error=str(exc))
                        print(f"FAILED underlying day {chunk_start}..{chunk_end}: {exc}")
                    time.sleep(sleep_seconds)
                    continue
                endpoint = "charts/intraday"
                payload = {
                    "securityId": str(NIFTY_SECURITY_ID),
                    "exchangeSegment": "IDX_I",
                    "instrument": "INDEX",
                    "interval": interval_map[interval],
                    "oi": False,
                    "fromDate": f"{chunk_start.isoformat()} 09:15:00",
                    "toDate": f"{chunk_end.isoformat()} 15:30:00",
                }
                try:
                    raw = self._post(endpoint, payload)
                    frame = self._parallel_arrays(raw.get("data", raw))
                    rows = self.store.upsert_candles(
                        frame,
                        symbol="NIFTY",
                        interval=interval,
                        broker="dhan",
                        source="dhan_bootstrap",
                    )
                    self._record_job("underlying", key, chunk_start, chunk_end, "complete", rows)
                    print(f"underlying {interval} {chunk_start}..{chunk_end}: {rows} rows")
                except Exception as exc:
                    self._record_job("underlying", key, chunk_start, chunk_end, "failed", error=str(exc))
                    print(f"FAILED underlying {interval} {chunk_start}..{chunk_end}: {exc}")
                time.sleep(sleep_seconds)

    @staticmethod
    def _classification(offset: int, option_type: str) -> str:
        if offset == 0:
            return "ATM"
        if option_type == "CE":
            return "ITM" if offset < 0 else "OTM"
        return "ITM" if offset > 0 else "OTM"

    def options(
        self,
        start: date,
        end: date,
        offsets: int,
        interval: str,
        expiry_flags: list[str],
        sleep_seconds: float,
    ) -> None:
        for expiry_flag in expiry_flags:
            expiry_codes = range(1, 5) if expiry_flag == "WEEK" else range(1, 3)
            for expiry_code in expiry_codes:
                for offset in range(-offsets, offsets + 1):
                    strike_label = "ATM" if offset == 0 else f"ATM{offset:+d}"
                    for option_type, api_type in (("CE", "CALL"), ("PE", "PUT")):
                        key = f"{expiry_flag}:{expiry_code}:{strike_label}:{option_type}:{interval}"
                        for chunk_start, chunk_end in date_chunks(start, end, 30):
                            if self._completed("rolling_options", key, chunk_start, chunk_end):
                                continue
                            payload = {
                                "exchangeSegment": "NSE_FNO",
                                "interval": interval,
                                "securityId": NIFTY_SECURITY_ID,
                                "instrument": "OPTIDX",
                                "expiryFlag": expiry_flag,
                                "expiryCode": expiry_code,
                                "strike": strike_label,
                                "drvOptionType": api_type,
                                "requiredData": [
                                    "open", "high", "low", "close", "iv",
                                    "volume", "strike", "oi", "spot",
                                ],
                                "fromDate": chunk_start.isoformat(),
                                "toDate": chunk_end.isoformat(),
                            }
                            try:
                                raw = self._post("charts/rollingoption", payload)
                                side = (raw.get("data") or {}).get(option_type.lower()) or {}
                                timestamps = side.get("timestamp") or []
                                rows = []
                                for i, ts in enumerate(timestamps):
                                    def value(name, default=0):
                                        values = side.get(name) or []
                                        return values[i] if i < len(values) else default

                                    rows.append(
                                        {
                                            "timestamp": pd.to_datetime(ts, unit="s", utc=True).tz_convert(IST),
                                            "strike": int(float(value("strike", 0) or 0)),
                                            "expiry": f"{expiry_flag}:{expiry_code}",
                                            "option_type": option_type,
                                            "classification": self._classification(offset, option_type),
                                            "open": value("open"),
                                            "high": value("high"),
                                            "low": value("low"),
                                            "close": value("close"),
                                            "volume": value("volume"),
                                            "oi": value("oi"),
                                            "iv": value("iv"),
                                            "underlying_spot": value("spot"),
                                        }
                                    )
                                frame = pd.DataFrame(rows)
                                written = self.store.upsert_option_candles(
                                    frame,
                                    symbol="NIFTY",
                                    interval=f"{interval}minute",
                                    broker="dhan",
                                    source=f"dhan_rolling:{key}",
                                )
                                self._record_job(
                                    "rolling_options", key, chunk_start, chunk_end,
                                    "complete", written,
                                )
                                print(f"options {key} {chunk_start}..{chunk_end}: {written} rows")
                            except Exception as exc:
                                self._record_job(
                                    "rolling_options", key, chunk_start, chunk_end,
                                    "failed", error=str(exc),
                                )
                                print(f"FAILED options {key} {chunk_start}..{chunk_end}: {exc}")
                            time.sleep(sleep_seconds)

    def refresh_coverage(self) -> None:
        now = datetime.now(IST).isoformat()
        with self._connect() as conn:
            conn.execute("DELETE FROM data_coverage")
            for table, dataset in (("candles", "underlying"), ("option_candles", "options")):
                rows = conn.execute(
                    f"""
                    SELECT symbol, interval, MIN(ts), MAX(ts), COUNT(*), broker
                    FROM {table}
                    GROUP BY symbol, interval, broker
                    """
                ).fetchall()
                conn.executemany(
                    """
                    INSERT INTO data_coverage (
                        dataset, symbol, interval, earliest_ts, latest_ts,
                        rows, source, checked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(dataset, *row, now) for row in rows],
                )

    def status(self) -> None:
        with self._connect() as conn:
            print("\nCoverage")
            coverage_rows = []
            for table, dataset in (("candles", "underlying"), ("option_candles", "options")):
                coverage_rows.extend(
                    (dataset, *row)
                    for row in conn.execute(
                        f"""
                        SELECT symbol, interval, broker, MIN(ts), MAX(ts), COUNT(*)
                        FROM {table}
                        GROUP BY symbol, interval, broker
                        """
                    ).fetchall()
                )
            for row in sorted(coverage_rows):
                print(" | ".join(str(value) for value in row))
            print("\nJobs")
            for row in conn.execute(
                """
                SELECT dataset, status, COUNT(*), SUM(rows_written)
                FROM ingestion_jobs GROUP BY dataset, status ORDER BY dataset, status
                """
            ):
                print(" | ".join(str(value) for value in row))

    def validate(self, years: int, option_offsets: int) -> int:
        cutoff = date.today() - timedelta(days=365 * years)
        errors = []
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT MIN(ts), MAX(ts), COUNT(*) FROM candles
                WHERE symbol='NIFTY' AND interval='5minute' AND broker='dhan'
                """
            ).fetchone()
            if not row or not row[0] or date.fromisoformat(row[0][:10]) > cutoff + timedelta(days=14):
                errors.append(f"5minute Dhan coverage does not reach {cutoff}")
            option_rows = conn.execute(
                """
                SELECT COUNT(*) FROM option_candles
                WHERE symbol='NIFTY' AND interval='5minute' AND broker='dhan'
                """
            ).fetchone()[0]
            if option_rows <= 0:
                errors.append("No Dhan 5minute rolling option rows")
            failed = conn.execute(
                "SELECT COUNT(*) FROM ingestion_jobs WHERE status='failed'"
            ).fetchone()[0]
            if failed:
                errors.append(f"{failed} failed ingestion jobs remain")
        if errors:
            print("VALIDATION FAILED")
            for error in errors:
                print(f"- {error}")
            return 1
        print(
            f"VALIDATION PASSED: {years}y underlying and option data present "
            f"(requested ATM +/-{option_offsets})"
        )
        return 0


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DATA_HIST_DB_PATH)
    parser.add_argument(
        "--transport",
        choices=["requests", "curl"],
        default=os.getenv("DHAN_BOOTSTRAP_TRANSPORT", "requests").strip().lower() or "requests",
        help="HTTP transport. Use curl when Python requests cannot resolve api.dhan.co.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status")
    status.set_defaults(action="status")

    validate = sub.add_parser("validate")
    validate.add_argument("--years", type=int, default=5)
    validate.add_argument("--option-offsets", type=int, default=3)

    for name in ("underlying", "options", "all"):
        command = sub.add_parser(name)
        command.add_argument("--start", type=parse_date, default=date.today() - timedelta(days=365 * 5))
        command.add_argument("--end", type=parse_date, default=date.today() + timedelta(days=1))
        command.add_argument("--sleep", type=float, default=1.1)
        if name in ("underlying", "all"):
            command.add_argument(
                "--intervals", default="1minute,5minute",
                help="Comma-separated: 1minute,5minute,15minute,60minute",
            )
        if name in ("options", "all"):
            command.add_argument("--offsets", type=int, default=3)
            command.add_argument("--interval", choices=["1", "5", "15", "25", "60"], default="5")
            command.add_argument("--expiry-flags", default="WEEK,MONTH")

    args = parser.parse_args()
    bootstrap = Bootstrap(args.db, transport=args.transport)
    if args.command == "status":
        bootstrap.status()
        return 0
    if args.command == "validate":
        return bootstrap.validate(args.years, args.option_offsets)
    if args.command in ("underlying", "all"):
        bootstrap.underlying(
            args.start,
            args.end,
            [value.strip() for value in args.intervals.split(",") if value.strip()],
            args.sleep,
        )
    if args.command in ("options", "all"):
        bootstrap.options(
            args.start,
            args.end,
            args.offsets,
            args.interval,
            [value.strip().upper() for value in args.expiry_flags.split(",") if value.strip()],
            max(args.sleep, 1.1),
        )
    bootstrap.status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

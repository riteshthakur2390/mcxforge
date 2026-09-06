"""
utils/latency_fill_monitor.py
=============================
Tracks whether execution quality preserves the backtested edge.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Optional

import pytz

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class FillQualityRecord:
    timestamp: str
    symbol: str
    mode: str
    latency_ms: float
    expected_premium: float
    fill_premium: float
    slippage_pct: float
    quantity: int
    order_id: str
    source_topic: str


class LatencyAndFillQualityMonitor:
    """Append-only fill-quality tracker with rolling stats."""

    def __init__(self, path: Optional[str] = None, max_memory: int = 500) -> None:
        journal_dir = Path(os.getenv("JOURNAL_DIR", "journal"))
        self.path = Path(path) if path else journal_dir / "latency_fill_quality.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_memory = max_memory
        self._records: list[FillQualityRecord] = []

    def record_order_event(self, payload: dict, *, topic: str, event_ts: Optional[str] = None) -> FillQualityRecord:
        fill = float(payload.get("entry_premium", payload.get("fill_premium", payload.get("est_premium", 0))) or 0)
        expected = float(payload.get("quoted_premium", payload.get("est_premium", fill)) or fill)
        latency_ms = self._latency_ms(payload, event_ts)
        slippage_pct = ((fill - expected) / expected * 100.0) if expected > 0 else 0.0
        record = FillQualityRecord(
            timestamp=self._normalize_ts(event_ts).isoformat(),
            symbol=str(payload.get("option_symbol", "")),
            mode=str(payload.get("mode", payload.get("execution", ""))),
            latency_ms=round(latency_ms, 2),
            expected_premium=round(expected, 4),
            fill_premium=round(fill, 4),
            slippage_pct=round(slippage_pct, 4),
            quantity=int(payload.get("quantity", 0) or 0),
            order_id=str(payload.get("order_id", "")),
            source_topic=str(topic),
        )
        self._append(record)
        return record

    def get_stats(self, lookback: int = 100) -> dict:
        rows = self._records[-lookback:]
        if not rows:
            return {"count": 0}
        latencies = sorted(r.latency_ms for r in rows)
        slips = [r.slippage_pct for r in rows]
        p95_idx = min(len(latencies) - 1, int(len(latencies) * 0.95))
        stats = {
            "count": len(rows),
            "avg_latency_ms": round(mean(latencies), 2),
            "p95_latency_ms": round(latencies[p95_idx], 2),
            "avg_slippage_pct": round(mean(slips), 4),
            "max_slippage_pct": round(max(slips), 4),
        }
        stats["warning"] = stats["p95_latency_ms"] > 5000 or stats["avg_slippage_pct"] > 1.0
        return stats

    def _append(self, record: FillQualityRecord) -> None:
        self._records.append(record)
        if len(self._records) > self.max_memory:
            self._records = self._records[-self.max_memory:]
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")
        if record.latency_ms > 5000 or record.slippage_pct > 1.0:
            logger.warning(
                f"[LatencyFillMonitor] execution_quality_warning | "
                f"{record.symbol} latency={record.latency_ms:.0f}ms slippage={record.slippage_pct:.2f}%"
            )

    def _latency_ms(self, payload: dict, event_ts: Optional[str]) -> float:
        start_raw = (
            payload.get("signal_timestamp")
            or (payload.get("signal") or {}).get("timestamp")
            or payload.get("planned_at")
            or payload.get("timestamp")
        )
        start = self._normalize_ts(start_raw)
        end = self._normalize_ts(event_ts or payload.get("order_timestamp") or payload.get("timestamp"))
        return max((end - start).total_seconds() * 1000.0, 0.0)

    @staticmethod
    def _normalize_ts(raw: Optional[str]) -> datetime:
        if not raw:
            return datetime.now(IST)
        try:
            ts = datetime.fromisoformat(str(raw))
            return IST.localize(ts) if ts.tzinfo is None else ts.astimezone(IST)
        except Exception:
            return datetime.now(IST)

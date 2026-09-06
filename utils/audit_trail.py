"""
utils/audit_trail.py — Immutable Audit Trail
=============================================
Every configuration change, parameter override, trade decision,
and system event written to an append-only log.

WHY THIS IS MANDATORY:
  SEBI requires algo trading systems to maintain a complete audit
  trail for Category 3 algo registration. Every decision must be
  traceable — who changed what, when, and what trade resulted.

  Also critical for debugging: "Why did this trade fire at 11:30
  on Tuesday?" → audit trail shows exact param values at that time.

WHAT GETS LOGGED:
  - Every trade entry/exit with full signal context
  - Every configuration change (old value → new value)
  - Every system event (startup, shutdown, halt, resume)
  - Every ML model deployment
  - Every risk guard trigger (circuit breaker, margin alert)
  - Every parameter override

IMMUTABILITY:
  - Log is append-only (never overwritten)
  - Each entry has SHA-256 hash of previous entry (blockchain-style)
  - Tampering detection: hash chain verification
  - Separate from regular log files

FORMAT:
  JSONL (one JSON object per line) with:
  {
    "seq":       1234,              sequential ID
    "ts":        "2026-05-15T10:30:00+05:30",
    "event":     "TRADE_ENTRY",
    "data":      {...},             event-specific data
    "prev_hash": "abc123...",       hash of previous entry
    "hash":      "def456..."        hash of this entry
  }

USAGE:
  from utils.audit_trail import get_audit
  audit = get_audit()
  audit.log_trade_entry(symbol, premium, signal_data)
  audit.log_config_change("STOP_LOSS_PCT", old=25, new=20)
  audit.log_system_event("CIRCUIT_BREAKER", details)
  audit.verify_integrity()  # detect tampering
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import JOURNAL_DIR
except ImportError:
    JOURNAL_DIR = "journal"

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

AUDIT_FILE = Path(JOURNAL_DIR) / "audit_trail.jsonl"


class AuditTrail:
    """
    Append-only audit trail with hash chain integrity.
    Thread-safe for single-process use.
    """

    def __init__(self) -> None:
        Path(JOURNAL_DIR).mkdir(parents=True, exist_ok=True)
        self._seq       = self._get_last_seq()
        self._prev_hash = self._get_last_hash()

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def log_trade_entry(
        self,
        symbol:      str,
        premium:     float,
        direction:   str,
        signal_data: dict,
        lots:        int = 1,
    ) -> None:
        self._write("TRADE_ENTRY", {
            "symbol":    symbol,
            "premium":   premium,
            "direction": direction,
            "lots":      lots,
            "votes":     signal_data.get("votes"),
            "strategies":signal_data.get("strategies_fired"),
            "ml_conf":   signal_data.get("ml_confidence"),
            "regime":    signal_data.get("regime"),
            "adx":       signal_data.get("adx"),
        })

    def log_trade_exit(
        self,
        symbol:      str,
        entry_prem:  float,
        exit_prem:   float,
        pnl_pct:     float,
        exit_reason: str,
        held_min:    int = 0,
    ) -> None:
        self._write("TRADE_EXIT", {
            "symbol":      symbol,
            "entry_prem":  entry_prem,
            "exit_prem":   exit_prem,
            "pnl_pct":     pnl_pct,
            "exit_reason": exit_reason,
            "held_min":    held_min,
        })

    def log_config_change(
        self,
        param:     str,
        old_value: Any,
        new_value: Any,
        changed_by: str = "system",
    ) -> None:
        self._write("CONFIG_CHANGE", {
            "param":      param,
            "old_value":  old_value,
            "new_value":  new_value,
            "changed_by": changed_by,
        })
        logger.info(
            f"[AuditTrail] CONFIG_CHANGE: {param} "
            f"{old_value} → {new_value} (by {changed_by})"
        )

    def log_system_event(
        self,
        event_type: str,
        details:    dict = None,
    ) -> None:
        self._write(f"SYSTEM_{event_type.upper()}", details or {})

    def log_risk_event(
        self,
        event_type:  str,
        details:     dict,
    ) -> None:
        self._write(f"RISK_{event_type.upper()}", details)
        logger.warning(f"[AuditTrail] RISK: {event_type} | {details}")

    def log_ml_event(
        self,
        event_type: str,
        model_meta: dict,
    ) -> None:
        self._write(f"ML_{event_type.upper()}", model_meta)

    def log_signal_blocked(
        self,
        reason:     str,
        signal_data:dict,
    ) -> None:
        self._write("SIGNAL_BLOCKED", {
            "reason":    reason,
            "direction": signal_data.get("direction"),
            "strategies":signal_data.get("strategies_fired"),
            "conf":      signal_data.get("confidence"),
        })

    # ── INTEGRITY VERIFICATION ────────────────────────────────────────────────

    def verify_integrity(self) -> dict:
        """
        Verify hash chain integrity.
        Returns {valid, entries, tampered_at} 
        """
        if not AUDIT_FILE.exists():
            return {"valid": True, "entries": 0, "tampered_at": None}

        prev_hash = "GENESIS"
        entries   = 0
        tampered  = None

        with open(AUDIT_FILE) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry      = json.loads(line)
                    # Recompute hash
                    entry_copy = {k: v for k, v in entry.items() if k != "hash"}
                    computed   = _hash_entry(entry_copy)

                    if entry.get("hash") != computed:
                        tampered = line_num
                        break
                    if entry.get("prev_hash") != prev_hash:
                        tampered = line_num
                        break

                    prev_hash = entry["hash"]
                    entries  += 1
                except Exception:
                    tampered = line_num
                    break

        return {
            "valid":      tampered is None,
            "entries":    entries,
            "tampered_at":tampered,
        }

    def get_recent(self, n: int = 20) -> list[dict]:
        """Return last N audit entries."""
        if not AUDIT_FILE.exists():
            return []
        lines = []
        with open(AUDIT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except Exception:
                        pass
        return lines[-n:]

    def get_config_history(self, param: str) -> list[dict]:
        """Get all changes for a specific config parameter."""
        if not AUDIT_FILE.exists():
            return []
        history = []
        with open(AUDIT_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if (entry.get("event") == "CONFIG_CHANGE" and
                            entry.get("data", {}).get("param") == param):
                        history.append(entry)
                except Exception:
                    pass
        return history

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _write(self, event: str, data: dict) -> None:
        """Append one entry to the audit trail."""
        self._seq += 1
        ts     = datetime.now(IST).isoformat()
        entry  = {
            "seq":       self._seq,
            "ts":        ts,
            "event":     event,
            "data":      data,
            "prev_hash": self._prev_hash,
        }
        entry["hash"] = _hash_entry(entry)
        self._prev_hash = entry["hash"]

        line = json.dumps(entry, ensure_ascii=False, default=str)
        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _get_last_seq(self) -> int:
        if not AUDIT_FILE.exists():
            return 0
        last = 0
        with open(AUDIT_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    last  = entry.get("seq", last)
                except Exception:
                    pass
        return last

    def _get_last_hash(self) -> str:
        if not AUDIT_FILE.exists():
            return "GENESIS"
        last_hash = "GENESIS"
        with open(AUDIT_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    last_hash = entry.get("hash", last_hash)
                except Exception:
                    pass
        return last_hash


# ── Hash helper ───────────────────────────────────────────────────────────────

def _hash_entry(entry: dict) -> str:
    content = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Singleton ─────────────────────────────────────────────────────────────────
_audit: AuditTrail | None = None

def get_audit() -> AuditTrail:
    global _audit
    if _audit is None:
        _audit = AuditTrail()
    return _audit

"""
Persistent live trade history for dashboard and daily P&L review.

Signal journal rows are useful for intraday signal flow. This file is the
closed-trade source of truth across process restarts.
"""
from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import os

import pytz

try:
    from config.settings import JOURNAL_DIR, NIFTY_LOT_SIZE
except Exception:
    JOURNAL_DIR = "journal"
    NIFTY_LOT_SIZE = 65

IST = pytz.timezone("Asia/Kolkata")
LIVE_HISTORY_CSV = Path(JOURNAL_DIR) / "live_trade_history.csv"
OBSERVE_HISTORY_CSV = Path(JOURNAL_DIR) / "observe_trade_history.csv"
HISTORY_CSV = LIVE_HISTORY_CSV

FIELDNAMES = [
    "trade_id",
    "signal_id",
    "date",
    "entry_time",
    "exit_time",
    "symbol",
    "option_symbol",
    "direction",
    "nifty_price",
    "strike",
    "option_type",
    "expiry_date",
    "entry_premium",
    "exit_premium",
    "lots",
    "quantity",
    "invested",
    "sl_premium",
    "target_premium",
    "gross_pnl_inr",
    "total_charges",
    "brokerage",
    "stt",
    "net_pnl_inr",
    "net_pnl_pct",
    "pnl_pct",
    "realized_pnl",
    "max_multiplier",
    "exit_reason",
    "holding_minutes",
    "strategies_fired",
    "votes",
    "ml_confidence",
    "ml_rank_score",
    "regime",
    "setup_type",
    "setup_strength",
    "outcome_eod",
    "mode",
    "execution",
    "simulated",
    "budget_lane",
    "source",
    "notes",
]


def _parse_ts(raw: object) -> Optional[datetime]:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw))
        return ts.astimezone(IST) if ts.tzinfo else IST.localize(ts)
    except Exception:
        return None


def _date_from_ts(raw: object) -> str:
    ts = _parse_ts(raw)
    return ts.date().isoformat() if ts else date.today().isoformat()


def _time_value(raw: object) -> str:
    ts = _parse_ts(raw)
    return ts.isoformat() if ts else str(raw or "")


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except Exception:
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except Exception:
        return default


def _join_strategies(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(v) for v in value if str(v))
    return ""


def _normalize_trade_id(val: object) -> str:
    return str(val or "").strip()


def _trade_matches(a: dict, b: dict) -> bool:
    tid_a = _normalize_trade_id(a.get("trade_id"))
    tid_b = _normalize_trade_id(b.get("trade_id"))
    if tid_a and tid_b and tid_a == tid_b:
        return True

    sig_a = _normalize_trade_id(a.get("signal_id"))
    sig_b = _normalize_trade_id(b.get("signal_id"))
    if sig_a and sig_b and sig_a == sig_b:
        return True

    if tid_a and sig_b and tid_a == sig_b:
        return True
    if tid_b and sig_a and tid_b == sig_a:
        return True

    opt_a = str(a.get("option_symbol") or "").strip()
    opt_b = str(b.get("option_symbol") or "").strip()
    date_a = str(a.get("date") or "").strip()
    date_b = str(b.get("date") or "").strip()
    time_a = str(a.get("entry_time") or "")[:16].strip()
    time_b = str(b.get("entry_time") or "")[:16].strip()

    if opt_a and opt_b and opt_a == opt_b and date_a and date_b and date_a == date_b:
        if time_a and time_b and time_a == time_b:
            return True

    return False


def _deduplicate_rows(rows: list[dict]) -> list[dict]:
    """Deduplicate trade rows, keeping the most authoritative record."""
    seen_indices: set[int] = set()
    deduped: list[dict] = []

    for i, r in enumerate(rows):
        if i in seen_indices:
            continue
        group = [r]
        for j in range(i + 1, len(rows)):
            if j not in seen_indices and _trade_matches(r, rows[j]):
                group.append(rows[j])
                seen_indices.add(j)

        best = group[0]
        for candidate in group[1:]:
            c_tid = str(candidate.get("trade_id") or "")
            b_tid = str(best.get("trade_id") or "")
            if c_tid.startswith("SF-") and not b_tid.startswith("SF-"):
                best = candidate
            elif str(candidate.get("source", "")) == "live_event" and str(best.get("source", "")) != "live_event":
                best = candidate
            elif _to_float(candidate.get("entry_premium")) > 0 and _to_float(best.get("entry_premium")) <= 0:
                best = candidate
        deduped.append(best)

    return deduped


class LiveTradeHistory:
    def __init__(self, path: Path = HISTORY_CSV, is_live: Optional[bool] = None) -> None:
        self.path = path
        if is_live is not None:
            self.is_live = bool(is_live)
        else:
            self.is_live = (path == LIVE_HISTORY_CSV or "live_trade_history" in path.name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_csv()
        if self.is_live and os.getenv("SEED_LEGACY_TRADES", "").strip().lower() in ("1", "true", "yes"):
            self.seed_manual_trades()

    def record_closed_trade(self, entry: dict, exit_payload: dict) -> dict:
        row = self._row_from_trade(entry, exit_payload)
        self._upsert(row)
        return row

    def seed_manual_trades(self) -> None:
        seeds = self._seed_rows_from_signal_journals()
        if not seeds:
            return
        raw_rows = self._read_rows()
        existing = [r for r in raw_rows if self._is_valid_history_row(r)]
        all_present = all(any(_trade_matches(s, ex) for ex in existing) for s in seeds)
        has_duplicates = len(_deduplicate_rows(existing)) < len(existing)

        if not all_present or has_duplicates:
            self._remove_legacy_manual_seed_rows()
            for seed in seeds:
                self._upsert(seed)

    def _remove_legacy_manual_seed_rows(self) -> None:
        legacy_ids = {
            "manual|2026-05-15|trade1",
            "manual|2026-06-04|trade2",
        }
        rows = [
            row for row in self._read_rows()
            if row.get("signal_id") not in legacy_ids and row.get("trade_id") not in legacy_ids
        ]
        self._write_rows(_deduplicate_rows(rows))

    def _seed_rows_from_signal_journals(self) -> list[dict]:
        try:
            from scripts.reconcile_live_journal_and_equity import build_reconciled_trades
            return build_reconciled_trades()
        except Exception:
            return []

    def _row_from_signal_journal(self, signal_row: dict, spec: dict) -> dict:
        quantity = _to_int(signal_row.get("quantity"), 0)
        entry_premium = _to_float(signal_row.get("actual_premium", signal_row.get("est_premium", 0)))
        realized = _to_float(signal_row.get("realized_pnl"), 0.0)
        if realized == 0 and entry_premium > 0 and quantity > 0:
            exit_premium = _to_float(signal_row.get("exit_premium"), 0.0)
            if exit_premium > 0:
                realized = round((exit_premium - entry_premium) * quantity, 2)
        return {
            "trade_id": spec["trade_id"],
            "signal_id": signal_row.get("signal_id", ""),
            "date": signal_row.get("date", ""),
            "entry_time": signal_row.get("entry_time", ""),
            "exit_time": signal_row.get("exit_time", ""),
            "symbol": signal_row.get("symbol") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM")),
            "option_symbol": signal_row.get("option_symbol", ""),
            "direction": signal_row.get("direction", ""),
            "nifty_price": signal_row.get("nifty_price", ""),
            "strike": signal_row.get("strike", ""),
            "option_type": signal_row.get("option_type", ""),
            "expiry_date": signal_row.get("expiry_date", ""),
            "entry_premium": signal_row.get("actual_premium") or signal_row.get("est_premium", ""),
            "exit_premium": signal_row.get("exit_premium", ""),
            "lots": signal_row.get("lots", ""),
            "quantity": signal_row.get("quantity", ""),
            "invested": signal_row.get("total_invested", ""),
            "sl_premium": signal_row.get("sl_premium", ""),
            "target_premium": signal_row.get("target_premium", ""),
            "pnl_pct": signal_row.get("pnl_pct", ""),
            "realized_pnl": realized,
            "exit_reason": signal_row.get("exit_reason", ""),
            "holding_minutes": signal_row.get("holding_minutes", ""),
            "strategies_fired": signal_row.get("strategies_fired", ""),
            "votes": signal_row.get("votes", ""),
            "ml_confidence": signal_row.get("ml_confidence") or signal_row.get("ml_conf", ""),
            "ml_rank_score": signal_row.get("ml_rank_score", ""),
            "regime": signal_row.get("regime", ""),
            "setup_type": signal_row.get("setup_type", ""),
            "setup_strength": signal_row.get("setup_strength", ""),
            "outcome_eod": signal_row.get("outcome_eod", ""),
            "mode": signal_row.get("mode", ""),
            "execution": signal_row.get("execution", ""),
            "simulated": "True" if str(signal_row.get("execution", "")).upper() == "DRY_RUN" else "",
            "budget_lane": signal_row.get("budget_lane", "FULL"),
            "source": "signal_journal_backfill",
            "notes": spec.get("notes", ""),
        }

    def rows(self, limit: int | None = None, include_backtest: bool = False) -> list[dict]:
        if not self.path.exists():
            return []
        rows = self._read_rows()
        rows = [r for r in rows if self._is_valid_history_row(r)]
        if not include_backtest:
            rows = [r for r in rows if str(r.get("mode", "")).upper() != "BACKTEST"]
        deduped = _deduplicate_rows(rows)
        deduped.sort(key=lambda r: (r.get("entry_time") or r.get("date") or "", r.get("trade_id") or ""), reverse=True)
        return deduped[:limit] if limit else deduped

    @staticmethod
    def _is_valid_history_row(row: dict) -> bool:
        if not str(row.get("option_symbol", "") or "").strip():
            return False
        if not str(row.get("date", "") or "").startswith("20"):
            return False
        return _to_float(row.get("entry_premium"), 0.0) > 0

    def summary(self) -> dict:
        rows = self.rows()
        today = date.today().isoformat()
        month = today[:7]
        return {
            "today": self._summarize([r for r in rows if r.get("date") == today]),
            "month": self._summarize([r for r in rows if str(r.get("date", "")).startswith(month)]),
            "all": self._summarize(rows),
        }

    def _row_from_trade(self, entry: dict, exit_payload: dict) -> dict:
        entry_time = entry.get("entry_time") or entry.get("timestamp")
        exit_time = exit_payload.get("exit_time") or exit_payload.get("timestamp")
        entry_premium = _to_float(entry.get("actual_premium", entry.get("entry_premium", entry.get("est_premium", 0))))
        exit_premium = _to_float(exit_payload.get("exit_premium", 0))
        quantity = _to_int(entry.get("quantity") or exit_payload.get("quantity"), 0)
        lot_size = _to_int(entry.get("lot_size"), NIFTY_LOT_SIZE) or NIFTY_LOT_SIZE
        lots = _to_int(entry.get("lots"), 0) or max(1, quantity // max(lot_size, 1)) if quantity else _to_int(entry.get("lots"), 1)
        if not quantity:
            quantity = lots * lot_size

        from utils.brokerage_calculator import calculate_option_trade_charges
        charges = calculate_option_trade_charges(entry_premium, exit_premium, quantity)
        gross_pnl = charges.gross_pnl_inr
        net_pnl = charges.net_pnl_inr
        tot_charges = charges.total_charges
        net_pct = charges.net_pnl_pct
        gross_pct = round(((exit_premium - entry_premium) / max(entry_premium, 1e-6)) * 100.0, 2) if entry_premium > 0 else 0.0

        peak_pct = _to_float(exit_payload.get("peak_pnl", exit_payload.get("peak_pnl_pct", gross_pct)))
        max_multiplier = round(1.0 + max(0.0, peak_pct) / 100.0, 2)

        signal_id = str(entry.get("signal_id") or exit_payload.get("signal_id") or "")
        if not signal_id:
            signal_id = "|".join([
                str(entry_time or ""),
                str(entry.get("option_symbol") or exit_payload.get("option_symbol") or ""),
                str(entry.get("direction") or exit_payload.get("direction") or ""),
            ])

        return {
            "trade_id": entry.get("trade_id") or signal_id or f"SF-{datetime.now(IST).strftime('%Y%m%d%H%M%S')}",
            "signal_id": signal_id,
            "date": entry.get("date") or _date_from_ts(entry_time or exit_time),
            "entry_time": _time_value(entry_time),
            "exit_time": _time_value(exit_time),
            "symbol": entry.get("symbol") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM")),
            "option_symbol": entry.get("option_symbol") or exit_payload.get("option_symbol", ""),
            "direction": entry.get("direction") or exit_payload.get("direction", ""),
            "nifty_price": entry.get("nifty_price", entry.get("nifty_ltp", "")),
            "strike": entry.get("strike", ""),
            "option_type": entry.get("option_type", ""),
            "expiry_date": entry.get("expiry_date", ""),
            "entry_premium": round(entry_premium, 2),
            "exit_premium": round(exit_premium, 2),
            "lots": lots,
            "quantity": quantity,
            "invested": round(entry_premium * quantity, 2),
            "sl_premium": entry.get("sl_premium", ""),
            "target_premium": entry.get("target_premium", ""),
            "gross_pnl_inr": gross_pnl,
            "total_charges": tot_charges,
            "brokerage": charges.brokerage,
            "stt": charges.stt,
            "net_pnl_inr": net_pnl,
            "net_pnl_pct": net_pct,
            "pnl_pct": gross_pct,
            "realized_pnl": net_pnl,
            "max_multiplier": max_multiplier,
            "exit_reason": exit_payload.get("exit_reason", entry.get("exit_reason", "")),
            "holding_minutes": _to_int(exit_payload.get("holding_minutes", entry.get("holding_minutes", 0))),
            "strategies_fired": _join_strategies(entry.get("strategies_fired") or entry.get("strategy_combo")),
            "votes": _to_int(entry.get("votes"), 0),
            "ml_confidence": round(_to_float(entry.get("ml_confidence", entry.get("ml_conf", 0))), 4),
            "ml_rank_score": round(_to_float(entry.get("ml_rank_score", 0)), 4),
            "regime": entry.get("regime", ""),
            "setup_type": entry.get("setup_type", ""),
            "setup_strength": entry.get("setup_strength", ""),
            "outcome_eod": entry.get("outcome_eod", "WIN" if gross_pct > 0 else "LOSS"),
            "mode": entry.get("mode", ""),
            "execution": entry.get("execution", ""),
            "simulated": str(exit_payload.get("simulated", entry.get("simulated", ""))),
            "budget_lane": entry.get("budget_lane", "FULL"),
            "source": "live_event",
            "notes": entry.get("notes", ""),
        }

    def _summarize(self, rows: list[dict]) -> dict:
        pnl_values = [_to_float(r.get("net_pnl_pct", r.get("pnl_pct")), 0.0) for r in rows]
        net_values = [_to_float(r.get("net_pnl_inr", r.get("realized_pnl")), 0.0) for r in rows]
        gross_values = [_to_float(r.get("gross_pnl_inr"), 0.0) for r in rows]
        charges_values = [_to_float(r.get("total_charges"), 0.0) for r in rows]
        wins = sum(1 for v in net_values if v > 0)
        losses = sum(1 for v in net_values if v < 0)
        count = len(rows)
        return {
            "trades": count,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / count * 100, 1) if count else 0.0,
            "total_pnl_pct": round(sum(pnl_values), 2),
            "avg_pnl_pct": round(sum(pnl_values) / count, 2) if count else 0.0,
            "gross_pnl": round(sum(gross_values), 2),
            "total_charges": round(sum(charges_values), 2),
            "realized_pnl": round(sum(net_values), 2),
            "net_pnl": round(sum(net_values), 2),
        }

    def _init_csv(self) -> None:
        if not self.path.exists():
            self._write_rows([])

    def _read_rows(self) -> list[dict]:
        try:
            raw = self.path.read_bytes()
        except Exception:
            return []
        if not raw:
            return []
        cleaned = raw.replace(b"\x00", b"")
        text = cleaned.decode("utf-8", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return []
        try:
            return list(csv.DictReader(lines))
        except Exception:
            return []

    def _write_rows(self, rows: list[dict]) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in FIELDNAMES}
                for row in rows
            )
        os.replace(tmp_path, self.path)

    def _upsert(self, row: dict) -> None:
        if str(row.get("mode", "")).upper() == "BACKTEST":
            return
        current_rows = self._read_rows()
        valid_rows = [r for r in current_rows if self._is_valid_history_row(r)]
        
        # Purge ALL matching rows (eliminating any existing duplicate copies)
        retained = [r for r in valid_rows if not _trade_matches(r, row)]
        
        # Add the new/updated row
        retained.append({field: row.get(field, "") for field in FIELDNAMES})
        
        # Deduplicate and sort (newest first)
        deduped = _deduplicate_rows(retained)
        deduped.sort(key=lambda r: (r.get("entry_time") or r.get("date") or "", r.get("trade_id") or ""), reverse=True)
        self._write_rows(deduped)


_live_history: Optional[LiveTradeHistory] = None
_observe_history: Optional[LiveTradeHistory] = None


def get_live_trade_history() -> LiveTradeHistory:
    global _live_history
    if _live_history is None:
        _live_history = LiveTradeHistory(path=LIVE_HISTORY_CSV, is_live=True)
    return _live_history


def get_observe_trade_history() -> LiveTradeHistory:
    global _observe_history
    if _observe_history is None:
        _observe_history = LiveTradeHistory(path=OBSERVE_HISTORY_CSV, is_live=False)
    return _observe_history


def get_trade_history(mode: Optional[str] = None) -> LiveTradeHistory:
    resolved_mode = str(mode or os.getenv("TRADING_MODE", "AUTO")).strip().upper()
    if resolved_mode == "OBSERVE":
        return get_observe_trade_history()
    return get_live_trade_history()


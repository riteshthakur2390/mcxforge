"""
agents_code/agent7_analytics/journal.py  Analytics & Journal Agent
"""
import asyncio
import csv
import json
import os
import re
from datetime import date, datetime
from typing import Optional
from pathlib import Path
from loguru import logger
import pytz

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bus import get_bus, Topic, Message
from config.settings.journal_thresholds import *
from config.settings import (
    JOURNAL_DIR, TRADING_MODE, LLM_ENABLED,
)
from utils.llm import (
    TaskType, call_llm_async, build_morning_outlook_prompt, build_eod_analysis_prompt,
    normalize_llm_sentences,
)
from utils.report_formatter import format_rich_report
from utils.performance_analytics import PerformanceAnalytics
from utils.signal_conflict_resolver import get_resolver
from utils.audit_trail import get_audit
from utils.latency_fill_monitor import LatencyAndFillQualityMonitor
from utils.market_calendar import is_trading_day

IST = pytz.timezone("Asia/Kolkata")


class AnalyticsAgent:
    NAME = "AnalyticsAgent"

    def __init__(self, backtest_mode: bool = False) -> None:
        self.bus     = get_bus()
        self.backtest_mode = backtest_mode
        self._today  = date.today().isoformat()
        self._brief_published_date: Optional[str] = None
        self._journal: list[dict] = []
        self._seen_suppressed_ids: set[str] = set()
        self._stats  = {
            "signals": 0, "approved": 0, "suppressed": 0,
            "traded": 0, "wins": 0, "losses": 0, "total_pnl_pct": 0.0,
        }
        self._premarket: dict = {}
        self._latest_plan: dict = {}
        self._performance = PerformanceAnalytics()
        self._resolver = get_resolver()
        self._audit = get_audit()
        self._fill_quality = LatencyAndFillQualityMonitor()
        os.makedirs(JOURNAL_DIR, exist_ok=True)

    def _mode_value(self, payload: Optional[dict] = None) -> str:
        payload = payload or {}
        if payload.get("mode"):
            return str(payload["mode"])
        if self.backtest_mode:
            return "BACKTEST"
        return TRADING_MODE

    def register(self) -> None:
        self.bus.subscribe(Topic.PREMARKET_BIAS,   self.on_premarket)
        self.bus.subscribe(Topic.RAW_SIGNAL,       self._on_raw_signal)
        self.bus.subscribe(Topic.SIGNAL_APPROVED,  self._on_approved)
        self.bus.subscribe(Topic.SIGNAL_REJECTED,  self._on_rejected)
        self.bus.subscribe(Topic.SIGNAL_SUPPRESSED,self._on_suppressed)
        self.bus.subscribe(Topic.TRADE_PLAN_READY, self._on_plan)
        self.bus.subscribe(Topic.ORDER_DRY_RUN,    self._on_order)
        self.bus.subscribe(Topic.ORDER_PLACED,     self._on_order)
        self.bus.subscribe(Topic.ORDER_SKIPPED,    self._on_skipped)
        self.bus.subscribe(Topic.POSITION_CLOSED,  self._on_closed)
        self.bus.subscribe("TRADING_HALTED",       self._on_trading_halted)
        logger.info(f"[{self.NAME}] Registered.")

    @staticmethod
    def _entry_timing_bucket(ts: datetime) -> str:
        mins = ts.hour * 60 + ts.minute
        if mins <= 10 * 60:
            return "early"
        if mins >= 14 * 60:
            return "late"
        return "mid"

    @staticmethod
    def _ml_rank_bucket(rank_score: float) -> str:
        if rank_score < JOURNAL_RANK_SCORE_THRESH_0_52:
            return "<0.52"
        if rank_score < JOURNAL_RANK_SCORE_THRESH_0_56:
            return "0.52-0.56"
        if rank_score < JOURNAL_RANK_SCORE_THRESH_0_6:
            return "0.56-0.60"
        return ">=0.60"

    @staticmethod
    def _normalize_list(value) -> list[str]:
        if isinstance(value, str):
            return [part.strip() for part in value.split("|") if part.strip()]
        if isinstance(value, (list, tuple, set)):
            return [str(part).strip() for part in value if str(part).strip()]
        return []

    def _signal_context(self, payload: dict, contract_snapshot: Optional[dict] = None) -> dict:
        metadata = (payload.get("metadata") or {}) if isinstance(payload, dict) else {}
        context = (metadata.get("_context") or {}) if isinstance(metadata, dict) else {}
        if context:
            return context
        snapshot = contract_snapshot or payload.get("contract_snapshot", {}) or {}
        if isinstance(snapshot, str):
            try:
                snapshot = json.loads(snapshot)
            except Exception:
                snapshot = {}
        setup = (snapshot.get("setup") or {}) if isinstance(snapshot, dict) else {}
        return setup.get("context", {}) or {}

    def _diagnostics_from_payload(self, payload: dict, ts: datetime) -> dict:
        contract_snapshot = payload.get("contract_snapshot", {}) or {}
        context = self._signal_context(payload, contract_snapshot)
        setup = (context.get("setup") or {}) if context else {}
        if not setup and isinstance(contract_snapshot, dict):
            setup = contract_snapshot.get("setup", {}) or {}
        market_structure = (
            (context.get("market_structure") or {})
            or (setup.get("context", {}) or {}).get("market_structure", {})
            or {}
        )
        strategies = self._normalize_list(payload.get("strategies_fired", []))
        ml_rank = float(payload.get("ml_rank_score", payload.get("ml_confidence", 0.0)) or 0.0)
        strategy_conf = float(payload.get("confidence", payload.get("strategy_conf", 0.0)) or 0.0)
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        momentum_strength = round(
            max(0.0, min(1.0, ml_rank * 0.45 + setup_strength * 0.35 + strategy_conf * 0.20)),
            4,
        )
        return {
            "strategy_combo": "|".join(strategies),
            "setup_type": str(setup.get("setup_type", "unknown") or "unknown"),
            "setup_strength": round(setup_strength, 4),
            "momentum_strength": momentum_strength,
            "entry_timing": self._entry_timing_bucket(ts),
            "ml_score_bucket": self._ml_rank_bucket(ml_rank),
            "market_regime_detail": str(
                ((context.get("regime_info") or {}).get("label"))
                or payload.get("regime", "")
                or "UNKNOWN"
            ),
            "structure_bias": str(
                ((market_structure.get("structure_state") or {}).get("bias", "UNKNOWN"))
            ),
            "liquidity_event": str(
                ((market_structure.get("liquidity_event") or {}).get("type", "NONE"))
            ),
            "entry_validation": str(
                "valid"
                if not (market_structure.get("entry_validation", {}) or {}).get("avoid", False)
                else "avoid"
            ),
        }

    def _log_trade_diagnostics(self, event: str, entry: dict) -> None:
        logger.info(
            f"[{self.NAME}] {event} | signal_id={entry.get('signal_id')} | "
            f"combo={entry.get('strategy_combo', '')} | votes={entry.get('votes', 0)} | "
            f"ml_rank={float(entry.get('ml_rank_score', 0.0) or 0.0):.3f} | "
            f"ml_conf={float(entry.get('ml_confidence', 0.0) or 0.0):.3f} | "
            f"regime={entry.get('regime', '')}/{entry.get('market_regime_detail', '')} | "
            f"momentum={float(entry.get('momentum_strength', 0.0) or 0.0):.3f} | "
            f"timing={entry.get('entry_timing', '')} | setup={entry.get('setup_type', '')}:"
            f"{float(entry.get('setup_strength', 0.0) or 0.0):.2f} | "
            f"outcome={entry.get('outcome_eod', entry.get('lifecycle_status', ''))}"
        )

    def _post_trade_diagnostics(self) -> dict:
        closed = [row for row in self._journal if str(row.get("lifecycle_status", "")).upper() == "CLOSED"]
        grouped: dict[str, dict[str, dict[str, float]]] = {
            "by_regime": {},
            "by_strategy_combo": {},
            "by_ml_score_bucket": {},
        }
        for label, key in (
            ("by_regime", "regime"),
            ("by_strategy_combo", "strategy_combo"),
            ("by_ml_score_bucket", "ml_score_bucket"),
        ):
            buckets: dict[str, list[dict]] = {}
            for row in closed:
                bucket_key = str(row.get(key, "") or "UNKNOWN")
                buckets.setdefault(bucket_key, []).append(row)
            grouped[label] = {
                bucket_key: {
                    "count": len(rows),
                    "wins": sum(float(r.get("realized_pnl", 0.0) or 0.0) > 0 for r in rows),
                    "losses": sum(float(r.get("realized_pnl", 0.0) or 0.0) <= 0 for r in rows),
                    "win_rate": round(
                        (
                            sum(float(r.get("realized_pnl", 0.0) or 0.0) > 0 for r in rows)
                            / len(rows) * 100.0
                        ),
                        1,
                    ) if rows else 0.0,
                    "pnl": round(sum(float(r.get("realized_pnl", 0.0) or 0.0) for r in rows), 2),
                }
                for bucket_key, rows in buckets.items()
            }
        return grouped

    def _closed_trades_for_day(self, day: str) -> list[dict]:
        return [
            row for row in self._journal
            if str(row.get("lifecycle_status", "")).upper() == "CLOSED"
            and str(row.get("date", "")).strip() == str(day).strip()
        ]

    def _load_persisted_journal_for_day(self, day: str) -> list[dict]:
        path = Path(JOURNAL_DIR) / f"signals_{day}.csv"
        if not path.exists():
            return []
        rows: list[dict] = []
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader((line.replace("\x00", "") for line in handle))
                for row in reader:
                    if str(row.get("date", "")).strip() == str(day).strip():
                        rows.append(dict(row))
        except Exception as exc:
            logger.error(f"[{self.NAME}] Failed loading persisted journal for {day}: {exc}")
            return []
        return rows

    @staticmethod
    def _derive_daily_stats_from_journal(rows: list[dict]) -> dict:
        closed = [row for row in rows if str(row.get("lifecycle_status", "")).upper() == "CLOSED"]
        wins = sum(float(row.get("realized_pnl", 0.0) or 0.0) > 0 for row in closed)
        losses = sum(float(row.get("realized_pnl", 0.0) or 0.0) <= 0 for row in closed)
        traded = len([row for row in rows if str(row.get("lifecycle_status", "")).upper() in {"ORDERED", "CLOSED"}])
        approved = len([row for row in rows if str(row.get("lifecycle_status", "")).upper() in {"APPROVED", "PLANNED", "ORDERED", "CLOSED", "SKIPPED"}])
        rejected = len([row for row in rows if str(row.get("lifecycle_status", "")).upper() == "REJECTED"])
        suppressed = len([row for row in rows if str(row.get("lifecycle_status", "")).upper() == "SUPPRESSED"])
        total_pnl_pct = round(sum(float(row.get("pnl_pct", 0.0) or 0.0) for row in closed), 2)
        return {
            "signals": len(rows),
            "approved": approved,
            "suppressed": suppressed,
            "rejected": rejected,
            "traded": traded,
            "wins": wins,
            "losses": losses,
            "total_pnl_pct": total_pnl_pct,
            "closed": closed,
        }

    def _reconcile_equity_curve_day(
        self,
        *,
        day: str,
        closed_trades: list[dict],
        wins: int,
        trades: int,
    ) -> dict:
        net_pnl = round(
            sum(float(entry.get("realized_pnl", entry.get("net_pnl_inr", 0.0)) or 0.0) for entry in closed_trades),
            2,
        )
        gross_pnl = round(
            sum(float(entry.get("gross_pnl_inr", entry.get("realized_pnl", 0.0)) or 0.0) for entry in closed_trades),
            2,
        )
        total_charges = round(
            sum(float(entry.get("total_charges", 0.0) or 0.0) for entry in closed_trades),
            2,
        )
        from utils.equity_curve import get_equity_curve
        active_mode = self._mode_value()
        return get_equity_curve(mode=active_mode).record_day(
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_charges=total_charges,
            trades=trades,
            wins=wins,
            mode=active_mode,
            day=day,
        )

    #  Pre-market 
    async def on_premarket(self, msg: Message) -> None:
        self._premarket = msg.payload
        ts_raw = msg.payload.get("timestamp")
        now_dt = datetime.now(IST)
        if ts_raw:
            try:
                now_dt = datetime.fromisoformat(str(ts_raw))
            except Exception:
                pass
        self._today = now_dt.date().isoformat()

        # Guard against duplicate sends on the same day if process/container restarts
        if self._brief_published_date == self._today:
            logger.info(f"[{self.NAME}] Market brief already published for {self._today} — skipping duplicate.")
            return

        is_open = is_trading_day(now_dt.date())
        day_note = "" if is_open else " ⏸️ (Weekend / Market Closed)"

        bias    = msg.payload.get("bias", "NEUTRAL")
        raw_vix = float(msg.payload.get("india_vix") or msg.payload.get("vix") or 0.0)
        vix     = raw_vix if 8.0 <= raw_vix <= 80.0 else 14.0
        vix_note = "" if 8.0 <= raw_vix <= 80.0 else " (fallback)"
        gap_pct = float(msg.payload.get("gap_pct", 0))
        prev_close = float(msg.payload.get("prev_close", msg.payload.get("nifty_prev_close", 0)) or 0)
        today_open = float(msg.payload.get("today_open", 0) or 0)
        gap_ready = bool(msg.payload.get("gap_ready", False))
        mode = TRADING_MODE

        brief = (
            f"🛡️ *MCXForge Daily Market Brief*\n"
            f"📅 {now_dt.strftime('%d %b %Y')}{day_note}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Commodity: *SILVERM* (Silver Micro Futures)\n"
            f"⚙️ Execution Mode: *{mode}* (Simulated Paper Trading)\n"
            f"🌐 Bias: *{bias}* | Gap: {gap_pct:+.2f}%\n"
            f"📊 Prev Close: {prev_close:.2f} | Open: {today_open:.2f}\n"
            f"⚡ India VIX: {vix:.1f}{vix_note}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌅 Morning Session: 09:00 – 17:00 IST (Base & Structure)\n"
            f"🌙 Evening Session: 17:00 – 23:00 IST (🔥 Primary Trading Window)\n"
            f"🛑 EOD Cutoff: 23:15 IST | Close: 23:30 IST\n"
            f"📌 Strategy Note: High-probability setups & volume trigger mostly in the Evening Session in {mode} mode."
        )
        await self.bus.publish(Topic.ALERT,
            {"type": "morning_brief", "text": brief}, self.NAME)
        self._brief_published_date = self._today

        if LLM_ENABLED and is_open:
            asyncio.create_task(self._morning_llm(bias, vix, gap_pct))

    async def _morning_llm(self, bias: str, vix: float, gap_pct: float) -> None:
        safe_vix = float(vix or 0.0)
        if not (8.0 <= safe_vix <= 80.0):
            safe_vix = 14.0
        text = await call_llm_async(
            build_morning_outlook_prompt(bias, safe_vix, gap_pct),
            task_type=TaskType.MORNING_OUTLOOK,
            max_tokens=160,
        )
        text = normalize_llm_sentences(text, expected_sentences=3)
        text = self._sanitize_morning_llm_text(text, safe_vix)
        if text:
            await self.bus.publish(Topic.ALERT,
                {"type": "morning_llm_outlook", "text": text}, self.NAME)

    @staticmethod
    def _sanitize_morning_llm_text(text: str, safe_vix: float) -> str:
        cleaned = str(text or "")
        if not cleaned:
            return cleaned
        safe_label = f"VIX {safe_vix:.1f}"
        patterns = [
            r"\bA\s+VIX\s+(?:reading\s+)?(?:of\s+)?0(?:\.0+)?\b",
            r"\bVIX\s+(?:reading\s+)?(?:of\s+)?0(?:\.0+)?\b",
            r"\bIndia\s+VIX\s+(?:reading\s+)?(?:of\s+)?0(?:\.0+)?\b",
            r"\bIndia\s+VIX\s+(?:stands\s+at\s+|is\s+)?0(?:\.0+)?\b",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, safe_label, cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            rf"\bVIX\s+{safe_vix:.1f}\s+indicate\b",
            f"VIX {safe_vix:.1f} indicates",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned

    #  Signal tracking 
    async def _on_raw_signal(self, msg: Message) -> None:
        entry = self._base_entry(msg.payload, status="RAW")
        try:
            from utils.option_chain_snapshot import OptionChainSnapshot
            spot = float(msg.payload.get("nifty_ltp", 0) or 0)
            if spot > 0:
                chain = OptionChainSnapshot()
                snapshot = await chain.capture(
                    spot=spot,
                    signal_direction=str(msg.payload.get("direction", "")),
                    dte=int(msg.payload.get("days_to_expiry", 5) or 5),
                )
                entry["option_chain_snapshot"] = chain.save(snapshot, entry.get("signal_id", ""))
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Option-chain snapshot skipped: {exc}")
        self._upsert_entry(entry)

    async def _on_approved(self, msg: Message) -> None:
        self._stats["approved"] += 1
        entry = self._base_entry(msg.payload, status="APPROVED")
        self._merge_signal_fields(entry, msg.payload)
        self._upsert_entry(entry)

    async def _on_rejected(self, msg: Message) -> None:
        entry = self._base_entry(msg.payload, status="REJECTED")
        self._merge_signal_fields(entry, msg.payload)
        entry["rejection_reason"] = msg.payload.get("rejection_reason", "")
        self._upsert_entry(entry)

    async def _on_suppressed(self, msg: Message) -> None:
        entry = self._base_entry(msg.payload, status="SUPPRESSED")
        signal_id = str(entry.get("signal_id", "") or "")
        if signal_id not in self._seen_suppressed_ids:
            self._seen_suppressed_ids.add(signal_id)
            self._stats["suppressed"] += 1
        entry["regime"] = msg.payload.get("regime", "")
        entry["suppression_reason"] = msg.payload.get("reason", "")
        self._upsert_entry(entry)

    async def _on_plan(self, msg: Message) -> None:
        self._stats["signals"] += 1
        self._latest_plan = msg.payload
        sig = msg.payload.get("signal", {})
        entry = self._base_entry(sig, status="PLANNED")
        self._merge_signal_fields(entry, sig)
        self._merge_plan_fields(entry, msg.payload)
        self._upsert_entry(entry)

    async def _on_order(self, msg: Message) -> None:
        self._stats["traded"] += 1
        data = msg.payload
        sig  = data.get("signal", {})
        ts   = self._resolve_ts(
            data.get("timestamp")
            or sig.get("timestamp")
            or msg.timestamp
        )
        strategy_list = sig.get("strategies_fired", [])
        direction = sig.get("direction", "")
        option_symbol = data.get("option_symbol", "")
        contract_snapshot = data.get("contract_snapshot", {}) or {}
        selection_notes = data.get("selection_notes", []) or []
        entry_reason = (
            f"{direction} via {', '.join(strategy_list) or 'unknown strategy'} | "
            f"Regime={sig.get('regime', '')} | Votes={sig.get('votes', 0)}"
        )

        entry = self._base_entry(sig, status="ORDERED")
        self._merge_signal_fields(entry, sig)
        self._merge_plan_fields(entry, data)
        entry.update({
            "time": ts.strftime("%H:%M"),
            "entry_time": ts.isoformat(),
            "actual_premium": data.get("entry_premium", data.get("est_premium", 0)),
            "total_invested": round(
                float(data.get("entry_premium", data.get("est_premium", 0)) or 0)
                * int(data.get("quantity", entry.get("quantity", 0)) or 0),
                2,
            ),
            "selection_notes": selection_notes,
            "entry_reason": entry_reason,
            "mode": data.get("mode", TRADING_MODE),
            "execution": "DRY_RUN" if data.get("simulated") else "PLACED",
            "contract_snapshot": contract_snapshot,
            "llm_rationale": data.get("llm_rationale", ""),
            "lifecycle_status": "ORDERED",
        })
        entry.update(self._diagnostics_from_payload({**sig, **data}, ts))
        fill_quality = self._fill_quality.record_order_event(data, topic=msg.topic, event_ts=msg.timestamp)
        entry["latency_ms"] = fill_quality.latency_ms
        entry["slippage_pct"] = fill_quality.slippage_pct
        self._upsert_entry(entry)
        self._log_trade_diagnostics("TRADE_ENTRY", entry)
        try:
            from utils.telegram_notifier import get_notifier
            await get_notifier().send_trade_opened({**data, **entry})
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Telegram trade-open notification skipped: {exc}")

    async def _on_skipped(self, msg: Message) -> None:
        self._stats["signals"] += 1
        sig = msg.payload.get("signal", {})
        entry = self._base_entry(sig, status="SKIPPED")
        self._merge_signal_fields(entry, sig)
        self._merge_plan_fields(entry, msg.payload)
        entry["execution"] = "SKIPPED"
        entry["rejection_reason"] = msg.payload.get("reason", "")
        self._upsert_entry(entry)

    async def _on_closed(self, msg: Message) -> None:
        pnl = float(msg.payload.get("pnl_pct", 0))
        exit_ts = self._resolve_ts(
            msg.payload.get("exit_time")
            or msg.payload.get("timestamp")
            or msg.timestamp
        )
        close_signal_id = str(msg.payload.get("signal_id", "") or "")
        self._stats["total_pnl_pct"] += pnl
        if pnl > 0:
            self._stats["wins"] += 1
        else:
            self._stats["losses"] += 1

        sym = msg.payload.get("option_symbol", "")
        target_entry = None

        # 1. Primary strategy: Match by signal_id against open orders
        if close_signal_id:
            for e in reversed(self._journal):
                if e.get("lifecycle_status") == "ORDERED":
                    e_sig = str(e.get("signal_id", "") or "")
                    if e_sig == close_signal_id:
                        target_entry = e
                        break
                    # Handle microsecond vs minute-level timestamp differences
                    e_prefix = e_sig.split(".")[0] if "." in e_sig else e_sig
                    c_prefix = close_signal_id.split(".")[0] if "." in close_signal_id else close_signal_id
                    if e_prefix and e_prefix == c_prefix and e.get("option_symbol") == sym:
                        target_entry = e
                        break
        
        # 2. Fallback strategy: Match last open position for the same symbol
        if not target_entry and sym:
            for e in reversed(self._journal):
                if e.get("option_symbol") == sym and e.get("lifecycle_status") == "ORDERED":
                    logger.warning(
                        f"[{self.NAME}] Closing trade via fallback (symbol='{sym}', "
                        f"original_signal_id='{close_signal_id}')"
                    )
                    target_entry = e
                    break
        
        if target_entry:
            target_entry["outcome_eod"] = "WIN" if pnl > 0 else "LOSS"
            target_entry["pnl_pct"]     = pnl
            target_entry["peak_pnl"]    = float(msg.payload.get("peak_pnl", 0.0) or 0.0)
            target_entry["peak_pnl_pct"] = float(msg.payload.get("peak_pnl", 0.0) or 0.0)
            target_entry["realized_pnl"] = float(msg.payload.get("realized_pnl", 0.0))
            if "lots" in msg.payload:
                target_entry["lots"] = int(msg.payload["lots"])
            if "quantity" in msg.payload:
                target_entry["quantity"] = int(msg.payload["quantity"])
                entry_prem = float(target_entry.get("entry_premium", msg.payload.get("entry_premium", 0)) or 0)
                target_entry["total_invested"] = round(entry_prem * target_entry["quantity"], 2)
            target_entry["exit_premium"] = float(msg.payload.get("exit_premium", 0.0))
            target_entry["exit_reason"] = msg.payload.get("exit_reason", "")
            target_entry["exit_time"] = exit_ts.isoformat()
            entry_ts = self._resolve_ts(target_entry.get("entry_time"))
            target_entry["holding_minutes"] = max(int((exit_ts - entry_ts).total_seconds() // 60), 0)
            target_entry["lifecycle_status"] = "CLOSED"

            # Query Upstox (or active broker) tradebook for exact execution prices only if LIVE/AUTO
            t_mode = str(target_entry.get("mode") or msg.payload.get("mode") or self._mode_value(target_entry)).strip().upper()
            if t_mode not in ("OBSERVE", "BACKTEST", "PAPER", "SIMULATED"):
                try:
                    from broker.upstox_broker import UpstoxBroker
                    upstox_b = UpstoxBroker()
                    sym_to_check = str(target_entry.get("option_symbol") or sym or "")
                    b_fill, s_fill = upstox_b.get_fills_for_symbol(sym_to_check)
                    if b_fill and b_fill > 0:
                        target_entry["actual_premium"] = b_fill
                        target_entry["entry_premium"] = b_fill
                        logger.info(f"[{self.NAME}] Reconciled Upstox Buy fill: ₹{b_fill}")
                    if s_fill and s_fill > 0:
                        target_entry["exit_premium"] = s_fill
                        msg.payload["exit_premium"] = s_fill
                        logger.info(f"[{self.NAME}] Reconciled Upstox Sell fill: ₹{s_fill}")
                except Exception as exc:
                    logger.debug(f"[{self.NAME}] Upstox fill sync attempt: {exc}")

            # Compute accurate brokerage & statutory charges
            try:
                from utils.brokerage_calculator import calculate_option_trade_charges
                _ep = float(
                    target_entry.get("actual_premium")
                    or target_entry.get("entry_premium")
                    or msg.payload.get("entry_premium")
                    or target_entry.get("est_premium")
                    or 0.0
                )
                _xp = float(msg.payload.get("exit_premium", 0.0) or target_entry.get("exit_premium", 0.0) or 0.0)
                _qt = int(target_entry.get("quantity") or msg.payload.get("quantity") or 65)
                target_entry["entry_premium"] = _ep
                msg.payload["entry_premium"] = _ep
                charges = calculate_option_trade_charges(_ep, _xp, _qt)
                target_entry["gross_pnl_inr"] = charges.gross_pnl_inr
                target_entry["brokerage"]     = charges.brokerage
                target_entry["stt"]           = charges.stt
                target_entry["total_charges"] = charges.total_charges
                target_entry["realized_pnl"]  = charges.net_pnl_inr
                target_entry["net_pnl_inr"]   = charges.net_pnl_inr
                target_entry["net_pnl_pct"]   = charges.net_pnl_pct
                gross_pct = round(((_xp - _ep) / max(_ep, 1e-6)) * 100.0, 2)
                target_entry["pnl_pct"]       = gross_pct
                # Update payload for downstream alerts
                msg.payload["gross_pnl_inr"]  = charges.gross_pnl_inr
                msg.payload["total_charges"]  = charges.total_charges
                msg.payload["realized_pnl"]   = charges.net_pnl_inr
                msg.payload["pnl_pct"]        = gross_pct
                msg.payload["net_pnl_pct"]    = charges.net_pnl_pct
            except Exception as exc:
                logger.debug(f"[{self.NAME}] Brokerage calculation failed: {exc}")

            self._upsert_entry(target_entry)
            self._log_trade_diagnostics("TRADE_EXIT", target_entry)
            self._resolver.record_outcome(
                direction=str(target_entry.get("direction", "")),
                pnl_pct=pnl,
                strategies=self._normalize_list(target_entry.get("strategies_fired", "")),
                candle_time=str(target_entry.get("time", "")),
                current_time=exit_ts,
            )
            self._audit.log_trade_exit(
                symbol=sym,
                entry_prem=float(target_entry.get("entry_premium", msg.payload.get("entry_premium", 0)) or 0),
                exit_prem=float(msg.payload.get("exit_premium", 0) or 0),
                pnl_pct=pnl,
                exit_reason=str(msg.payload.get("exit_reason", "")),
                held_min=int(target_entry.get("holding_minutes", 0) or 0),
            )
            try:
                from utils.trade_ledger import get_ledger
                get_ledger().record_trade(target_entry, msg.payload)
            except Exception as exc:
                logger.error(f"[{self.NAME}] Trade ledger write failed: {exc}")
            try:
                from utils.live_trade_history import get_trade_history
                history_row = get_trade_history(mode=t_mode).record_closed_trade(target_entry, msg.payload)
                await self.bus.publish("LIVE_TRADE_HISTORY_UPDATED", history_row, self.NAME)
            except Exception as exc:
                logger.error(f"[{self.NAME}] Trade history write failed: {exc}")
            try:
                from utils.telegram_notifier import get_notifier
                await get_notifier().send_trade_closed({**target_entry, **msg.payload})
            except Exception as exc:
                logger.debug(f"[{self.NAME}] Telegram trade-close notification skipped: {exc}")
        else:
            logger.error(
                f"[{self.NAME}] Could not find matching trade to close for "
                f"signal_id='{close_signal_id}' or symbol='{sym}'."
            )

    async def _on_trading_halted(self, msg: Message) -> None:
        try:
            from utils.telegram_notifier import get_notifier
            await get_notifier().send_circuit_breaker(msg.payload)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Telegram circuit-breaker notification skipped: {exc}")

    #  EOD report 
    async def generate_eod_report(self) -> None:
        day_rows = [row for row in self._journal if str(row.get("date", "")).strip() == str(self._today).strip()]
        if not day_rows:
            day_rows = self._load_persisted_journal_for_day(self._today)
        stats = self._derive_daily_stats_from_journal(day_rows) if day_rows else {**self._stats, "closed": self._closed_trades_for_day(self._today)}
        s   = {
            "signals": stats.get("signals", self._stats["signals"]),
            "approved": stats.get("approved", self._stats["approved"]),
            "suppressed": stats.get("suppressed", self._stats["suppressed"]),
            "traded": stats.get("traded", self._stats["traded"]),
            "wins": stats.get("wins", self._stats["wins"]),
            "losses": stats.get("losses", self._stats["losses"]),
            "total_pnl_pct": stats.get("total_pnl_pct", self._stats["total_pnl_pct"]),
        }
        t   = s["wins"] + s["losses"]
        wr  = round(s["wins"] / t * 100, 1) if t else 0

        report = {
            "date":          self._today,
            "mode":          TRADING_MODE,
            "signals_total": s["signals"],
            "approved":      s["approved"],
            "suppressed":    s["suppressed"],
            "traded":        s["traded"],
            "wins":          s["wins"],
            "losses":        s["losses"],
            "win_rate":      wr,
            "total_pnl_pct": round(s["total_pnl_pct"], 2),
            "journal":       day_rows or self._journal,
            "diagnostics":   self._post_trade_diagnostics() if not day_rows else {},
        }
        closed_trades = stats.get("closed", self._closed_trades_for_day(self._today))
        gross_list = [float(e.get("gross_pnl_inr", e.get("realized_pnl", 0.0)) or 0.0) for e in closed_trades]
        charges_list = [float(e.get("total_charges", 0.0) or 0.0) for e in closed_trades]
        realized = [float(e.get("realized_pnl", 0.0) or 0.0) for e in closed_trades]
        gross_pnl = round(sum(gross_list), 2)
        total_charges = round(sum(charges_list), 2)
        net_pnl = round(sum(realized), 2)
        try:
            perf = self._performance.full_report(closed_trades)
            mc = self._performance.monte_carlo(closed_trades, simulations=1000)
            report["performance_analytics"] = {
                "summary": perf.__dict__,
                "monte_carlo": mc.__dict__,
            }
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Performance analytics skipped: {exc}")
        try:
            curve_row = self._reconcile_equity_curve_day(
                day=self._today,
                closed_trades=closed_trades,
                wins=s["wins"],
                trades=t,
            )
            report["equity_curve"] = curve_row
        except Exception as exc:
            logger.error(f"[{self.NAME}] Equity curve update failed: {exc}")
        
        # Generate rich report
        rich_report_clean = format_rich_report(day_rows or self._journal, use_colors=False)
        rich_report_color = format_rich_report(day_rows or self._journal, use_colors=True)
        report["rich_report"] = rich_report_clean
        
        p = Path(JOURNAL_DIR) / f"eod_{self._today}.json"
        p.write_text(json.dumps(report, indent=2, default=str))

        # Save rich report as text file as well
        p_txt = Path(JOURNAL_DIR) / f"report_{self._today}.txt"
        p_txt.write_text(rich_report_clean)

        eod_text = (
            f"📊 *MCXForge EOD — {self._today}*\n"
            f"```\n{rich_report_clean}\n```\n"
            f"[FEED] Signals: {s['signals']} | [OK] Approved: {s['approved']}\n"
            f"[PROGRESS] Traded: {s['traded']} | [GREEN] {s['wins']} wins | [RED] {s['losses']} losses"
        )
        await self.bus.publish(Topic.EOD_REPORT_READY, report, self.NAME)
        await self.bus.publish(Topic.ALERT,
            {"type": "eod_report", "text": eod_text}, self.NAME)
        try:
            from utils.telegram_notifier import get_notifier
            from config.settings import TOTAL_FUND, DEPLOYED_CAPITAL
            await get_notifier().send_eod_summary({
                "trades": t,
                "wins": s["wins"],
                "losses": s["losses"],
                "win_rate": wr,
                "total_fund": TOTAL_FUND,
                "deployed_capital": DEPLOYED_CAPITAL,
                "gross_pnl": gross_pnl,
                "total_charges": total_charges,
                "net_pnl": net_pnl,
                "ending_equity": (report.get("equity_curve") or {}).get("ending_equity", TOTAL_FUND),
                "drawdown_pct": (report.get("equity_curve") or {}).get("current_drawdown_pct", 0),
                "risk_signal": (report.get("equity_curve") or {}).get("risk_signal", "NORMAL"),
            })
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Telegram EOD notification skipped: {exc}")

        if LLM_ENABLED and s["traded"] > 0:
            asyncio.create_task(self._eod_llm(report))

        logger.success(f"[{self.NAME}] EOD report saved  {p}")

    async def _eod_llm(self, report: dict) -> None:
        summary = "\n".join([
            f"  {e.get('time', '')} | {e.get('direction', '')} | {e.get('strategies_fired', '')} | "
            f"outcome={e.get('outcome_eod', '')} | pnl={float(e.get('pnl_pct') or 0.0):+.1f}%"
            for e in report.get("journal", [])
        ]) or "  No trades today."
        text = await call_llm_async(
            build_eod_analysis_prompt(
                report["date"], report["signals_total"],
                report["suppressed"], report["wins"],
                report["losses"], report["win_rate"], summary,
            ),
            task_type=TaskType.EOD_ANALYSIS,
            max_tokens=200,
        )
        text = normalize_llm_sentences(text, expected_sentences=3)
        if text:
            await self.bus.publish(Topic.ALERT,
                {"type": "eod_llm_analysis", "text": text}, self.NAME)

    def _upsert_entry(self, entry: dict) -> None:
        existing = self._find_entry(entry["signal_id"])
        if existing is None:
            self._journal.append(entry)
        else:
            status_order = {
                "RAW": 0, "APPROVED": 1, "REJECTED": 2, "SUPPRESSED": 3,
                "PLANNED": 4, "ORDERED": 5, "SKIPPED": 6, "CLOSED": 7
            }
            existing_status = existing.get("lifecycle_status", "RAW")
            new_status = entry.get("lifecycle_status", "RAW")
            
            existing.update(entry)
            
            if status_order.get(existing_status, 0) > status_order.get(new_status, 0):
                existing["lifecycle_status"] = existing_status
                
            entry = existing
        if not self.backtest_mode:
            self._upsert_csv_row(entry)

    def _upsert_csv_row(self, entry: dict) -> None:
        csv_path = Path(JOURNAL_DIR) / f"signals_{entry.get('date', self._today)}.csv"
        default_fieldnames = list(self._blank_entry().keys())

        try:
            rows = []
            fieldnames = default_fieldnames
            if csv_path.exists():
                with open(csv_path, newline="") as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        fieldnames = list(rows[0].keys())
        except Exception as e:
            logger.error(f"[{self.NAME}] Journal read error: {e}")
            return

        updated = False
        for row in rows:
            if row.get("signal_id") == str(entry.get("signal_id", "")):
                for key, value in entry.items():
                    row[key] = value
                updated = True
                break

        if not updated:
            rows.append({key: entry.get(key, "") for key in fieldnames})

        for key in entry.keys():
            if key not in fieldnames:
                fieldnames.append(key)
        normalized_rows = [{key: row.get(key, "") for key in fieldnames} for row in rows]

        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(normalized_rows)
        except Exception as e:
            logger.error(f"[{self.NAME}] Journal update error: {e}")

    def _find_entry(self, signal_id: str) -> Optional[dict]:
        for entry in reversed(self._journal):
            if entry.get("signal_id") == signal_id:
                return entry
        return None

    def _base_entry(self, payload: dict, *, status: str) -> dict:
        ts = self._resolve_ts(payload.get("timestamp"))
        date_str = ts.date().isoformat()
        if status == "SUPPRESSED":
            signal_id = self._suppressed_signal_id(payload, ts)
        else:
            signal_id = self._signal_id(payload, ts)
        entry = self._blank_entry()
        entry.update({
            "signal_id": signal_id,
            "date": date_str,
            "time": ts.strftime("%H:%M"),
            "entry_time": payload.get("timestamp", ""),
            "symbol": payload.get("symbol", "NIFTY"),
            "direction": payload.get("direction", ""),
            "mode": self._mode_value(payload),
            "regime": payload.get("regime", ""),
            "lifecycle_status": status,
        })
        entry.update(self._diagnostics_from_payload(payload, ts))
        return entry

    def _merge_signal_fields(self, entry: dict, signal: dict) -> None:
        ctx = self._signal_context(signal, {}) or {}
        hq_gate = (
            signal.get("shadow_high_quality_entry_gate")
            or ctx.get("shadow_high_quality_entry_gate")
            or {}
        )
        reasons = (
            signal.get("shadow_high_quality_entry_gate_reasons")
            or hq_gate.get("shadow_high_quality_entry_gate_reasons")
            or []
        )
        if isinstance(reasons, list):
            reasons_str = "|".join(str(r) for r in reasons)
        else:
            reasons_str = str(reasons)

        entry.update({
            "symbol": signal.get("symbol", entry["symbol"]),
            "direction": signal.get("direction", entry["direction"]),
            "nifty_price": signal.get("nifty_ltp", entry["nifty_price"]),
            "strategies_fired": "|".join(signal.get("strategies_fired", [])),
            "votes": signal.get("votes", entry["votes"]),
            "strategy_conf": round(float(signal.get("confidence", entry["strategy_conf"]) or 0), 4),
            "regime": signal.get("regime", entry["regime"]),
            "ml_rank_score": round(float(signal.get("ml_rank_score", entry["ml_rank_score"]) or 0), 4),
            "ml_rank_tier": str(signal.get("ml_rank_tier", entry["ml_rank_tier"]) or entry["ml_rank_tier"]),
            "ml_decision_reason": str(
                signal.get("ml_decision_reason", entry["ml_decision_reason"])
                or entry["ml_decision_reason"]
            ),
            "shadow_high_quality_entry_gate_state": str(
                signal.get("shadow_high_quality_entry_gate_state")
                or hq_gate.get("shadow_high_quality_entry_gate_state")
                or entry.get("shadow_high_quality_entry_gate_state", "")
            ),
            "shadow_high_quality_entry_gate_reasons": reasons_str,
            "shadow_ml_state": str(
                signal.get("shadow_ml_state")
                or hq_gate.get("shadow_ml_state")
                or entry.get("shadow_ml_state", "")
            ),
            "shadow_timing_state": str(
                signal.get("shadow_timing_state")
                or hq_gate.get("shadow_timing_state")
                or entry.get("shadow_timing_state", "")
            ),
            "shadow_raw_vote_count": int(
                signal.get("shadow_raw_vote_count")
                or hq_gate.get("shadow_raw_vote_count")
                or signal.get("votes", entry.get("shadow_raw_vote_count", 0))
                or 0
            ),
            "shadow_independent_category_count": int(
                signal.get("shadow_independent_category_count")
                or hq_gate.get("shadow_independent_category_count")
                or signal.get("independent_category_count", entry.get("shadow_independent_category_count", 0))
                or 0
            ),
            "shadow_decision": str(
                signal.get("shadow_decision")
                or hq_gate.get("shadow_decision")
                or entry.get("shadow_decision", "")
            ),
            "live_decision": str(
                signal.get("live_decision")
                or hq_gate.get("live_decision")
                or entry.get("live_decision", "")
            ),
            "shadow_joint_rule_decision": str(
                signal.get("shadow_joint_rule_decision")
                or hq_gate.get("shadow_joint_rule_decision")
                or entry.get("shadow_joint_rule_decision", "")
            ),
            "disagreement": str(
                signal.get("disagreement")
                or hq_gate.get("disagreement")
                or entry.get("disagreement", "")
            ),
            "quality_classification": str(
                signal.get("quality_classification")
                or ctx.get("quality_classification")
                or signal.get("quality_tier")
                or ctx.get("quality_tier")
                or entry.get("quality_classification", "")
            ),
            "quality_classification_reasons": (
                "|".join(str(r) for r in (signal.get("quality_classification_reasons") or ctx.get("quality_classification_reasons") or []))
                if isinstance(signal.get("quality_classification_reasons") or ctx.get("quality_classification_reasons"), list)
                else str(signal.get("quality_classification_reasons") or ctx.get("quality_classification_reasons") or entry.get("quality_classification_reasons", ""))
            ),
        })

    def _merge_plan_fields(self, entry: dict, payload: dict) -> None:
        contract_snapshot = payload.get("contract_snapshot", {}) or {}
        ml_conf = round(float(payload.get("ml_confidence", entry["ml_conf"]) or 0), 4)
        entry.update({
            "ml_conf": ml_conf,
            "ml_confidence": ml_conf,
            "ml_decision": payload.get("ml_decision", entry["ml_decision"]),
            "ml_decision_reason": payload.get("ml_decision_reason", entry["ml_decision_reason"]),
            "ml_rank_score": round(float(payload.get("ml_rank_score", entry["ml_rank_score"]) or 0), 4),
            "ml_rank_tier": payload.get("ml_rank_tier", entry["ml_rank_tier"]),
            "option_symbol": payload.get("option_symbol", entry["option_symbol"]),
            "strike": payload.get("strike", contract_snapshot.get("strike", entry["strike"])),
            "option_type": payload.get("option_type", contract_snapshot.get("option_type", entry["option_type"])),
            "expiry_date": payload.get("expiry_date", contract_snapshot.get("expiry_date", entry["expiry_date"])),
            "est_premium": payload.get("est_premium", entry["est_premium"]),
            "sl_premium": payload.get("sl_premium", entry["sl_premium"]),
            "target_premium": payload.get("target_premium", entry["target_premium"]),
            "risk_reward": payload.get("risk_reward", entry["risk_reward"]),
            "premium_source": payload.get("premium_source", entry["premium_source"]),
            "contract_score": round(float(payload.get("contract_score", entry["contract_score"]) or 0), 4),
            "contract_snapshot": contract_snapshot or entry["contract_snapshot"],
            "selection_notes": payload.get("selection_notes", entry["selection_notes"]),
            "execution": payload.get("execution", entry["execution"]),
            "lots": int(payload.get("desired_lots", payload.get("lots", 1))),
            "lot_size": int(payload.get("lot_size", entry.get("lot_size", 0)) or 0),
            "quantity": int(payload.get("quantity", entry.get("quantity", 0)) or 0),
            "budget_lane": (
                "HERO_ZERO"
                if (
                    bool((payload.get("metadata") or {}).get("hero_zero_lane", False))
                    or bool(((payload.get("signal") or {}).get("metadata") or {}).get("hero_zero_lane", False))
                    or str(payload.get("budget_lane", "")).upper() == "HERO_ZERO"
                    or "HeroZero" in [str(s) for s in ((payload.get("signal") or {}).get("strategies_fired", []) or payload.get("strategies_fired", []))]
                )
                else (
                    "REDUCED"
                    if (
                        bool((payload.get("metadata") or {}).get("reduced_budget_lane", False))
                        or bool(((payload.get("signal") or {}).get("metadata") or {}).get("reduced_budget_lane", False))
                        or bool(payload.get("reduced_budget_lane", False))
                        or "REDUCED" in str(payload.get("ml_decision", "")).upper()
                        or "REDUCED" in str(payload.get("ml_decision_reason", "")).upper()
                    )
                    else "FULL"
                )
            ),
            "total_invested": round(
                float(
                    payload.get(
                        "total_invested",
                        float(payload.get("est_premium", entry["est_premium"]) or 0)
                        * int(payload.get("quantity", entry.get("quantity", 0)) or 0),
                    )
                    or 0
                ),
                2,
            ),
        })

    @staticmethod
    def _signal_id(payload: dict, ts: datetime) -> str:
        strategies = "|".join(payload.get("strategies_fired", []))
        price = round(float(payload.get("nifty_ltp", 0) or 0), 2)
        return "|".join([
            ts.isoformat(),
            str(payload.get("direction", "")),
            strategies,
            f"{price:.2f}",
        ])

    @staticmethod
    def _suppressed_signal_id(payload: dict, ts: datetime) -> str:
        return "|".join([
            ts.isoformat(),
            "SUPPRESSED",
            str(payload.get("regime", "")),
            str(payload.get("reason", "")),
        ])

    @staticmethod
    def _blank_entry() -> dict:
        return {
            "signal_id": "",
            "date": "",
            "time": "",
            "entry_time": "",
            "symbol": "NIFTY",
            "direction": "",
            "nifty_price": 0,
            "strategies_fired": "",
            "votes": 0,
            "strategy_conf": 0.0,
            "ml_conf": 0.0,
            "ml_confidence": 0.0,
            "ml_decision": "",
            "ml_decision_reason": "",
            "ml_rank_score": 0.0,
            "ml_rank_tier": "",
            "regime": "",
            "market_regime_detail": "",
            "strategy_combo": "",
            "setup_type": "",
            "setup_strength": 0.0,
            "momentum_strength": 0.0,
            "entry_timing": "",
            "ml_score_bucket": "",
            "structure_bias": "",
            "liquidity_event": "",
            "entry_validation": "",
            "lifecycle_status": "",
            "rejection_reason": "",
            "suppression_reason": "",
            "option_symbol": "",
            "strike": 0,
            "option_type": "",
            "expiry_date": "",
            "est_premium": 0,
            "actual_premium": 0,
            "lots": 0,
            "lot_size": 0,
            "quantity": 0,
            "total_invested": 0.0,
            "sl_premium": 0,
            "target_premium": 0,
            "risk_reward": 0,
            "premium_source": "",
            "contract_score": 0.0,
            "contract_snapshot": {},
            "selection_notes": [],
            "entry_reason": "",
            "mode": "",
            "execution": "",
            "outcome_eod": "",
            "pnl_pct": 0.0,
            "realized_pnl": 0.0,
            "exit_time": "",
            "exit_reason": "",
            "holding_minutes": 0,
            "llm_rationale": "",
            "budget_lane": "FULL",
            "shadow_high_quality_entry_gate_state": "",
            "shadow_high_quality_entry_gate_reasons": "",
            "shadow_ml_state": "",
            "shadow_timing_state": "",
            "shadow_raw_vote_count": 0,
            "shadow_independent_category_count": 0,
            "shadow_decision": "",
            "live_decision": "",
            "shadow_joint_rule_decision": "",
            "disagreement": "",
            "quality_classification": "",
            "quality_classification_reasons": "",
        }

    def get_daily_stats(self) -> dict:
        s   = self._stats
        t   = s["wins"] + s["losses"]
        return {
            **s,
            "date":     self._today,
            "mode":     self._mode_value(),
            "win_rate": round(s["wins"] / t * 100, 1) if t else 0,
        }

    @staticmethod
    def _resolve_ts(raw: Optional[str]) -> datetime:
        if raw:
            ts = datetime.fromisoformat(str(raw))
            return ts if ts.tzinfo else IST.localize(ts)
        return datetime.now(IST)

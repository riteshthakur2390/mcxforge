from __future__ import annotations

"""
agents_code/agent8_dashboard/app.py  Dashboard & Alert Agent
See dashboard/templates/ for HTML.
Flask-SocketIO on port 5050.
"""
import csv, math, re
import threading, requests, os, subprocess, sys, json, time
import asyncio
from datetime import datetime, date
from enum import Enum
from pathlib import Path
import pandas as pd
import pytz
from loguru import logger
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO

os.environ['TZ'] = 'Asia/Kolkata'
if hasattr(time, 'tzset'):
    time.tzset()

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bus import get_bus, Topic, Message
from core.llm_router import TaskType, get_llm_status, probe_llm_provider
from utils.llm import call_llm_async
from broker.factory import get_active_broker_name
from config.settings import (
    DASHBOARD_PORT, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TRADING_MODE, LOGS_DIR,
    ML_THRESHOLD_OVERRIDE, JOURNAL_DIR, ML_MODELS_DIR, LIVE_TIMEFRAME,
    LIVE_TELEGRAM_BOT_TOKEN, LIVE_TELEGRAM_CHAT_ID, DEPLOYED_CAPITAL,
)
from backtesting.engine import get_backtest_limits
from data.historical_store import HistoricalCandleStore
from utils.option_utils import (
    build_option_symbol,
    estimate_atm_premium,
    get_atm_strike,
    get_nearest_expiry,
)
from utils.performance_review import build_live_vs_backtest_report
from utils.live_trade_history import get_trade_history, get_live_trade_history

IST = pytz.timezone("Asia/Kolkata")
app      = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Full dashboard HTML is in dashboard/templates/index.html
# Loaded at runtime
_DASHBOARD_HTML_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'dashboard', 'templates', 'index.html'
)
_BACKTEST_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'scripts', 'backtest_runner.py'
)
_MARKET_PARTICIPATION_PATH = Path(__file__).resolve().parents[2] / "data" / "market_participation" / "latest.json"


def _today_iso() -> str:
    return datetime.now(IST).date().isoformat()


def _estimate_backtest_timeout(payload: dict) -> int:
    """
    Scale subprocess timeout to the requested backtest window.
    One month of candles can legitimately exceed 300s on the current replay path.
    """
    base_timeout = 300
    max_timeout = 1800

    try:
        if payload.get("days"):
            days = max(1, int(payload["days"]))
            return min(max_timeout, max(base_timeout, 120 + days * 45))

        if payload.get("date"):
            return base_timeout

        if payload.get("start_date"):
            start_date = date.fromisoformat(str(payload["start_date"]))
            end_date = date.fromisoformat(str(payload.get("end_date") or payload["start_date"]))
            span_days = max(1, (end_date - start_date).days + 1)
            return min(max_timeout, max(base_timeout, 120 + span_days * 45))
    except Exception:
        pass

    return base_timeout

class DashboardAlertAgent:
    NAME = "DashboardAgent"
    def __init__(
        self,
        position_agent=None,
        analytics_agent=None,
        risk_agent=None,
        ml_agent=None,
        data_agent=None,
        strategy_agent=None,
    ):
        self.bus             = get_bus()
        self.position_agent  = position_agent
        self.analytics_agent = analytics_agent
        self.risk_agent      = risk_agent
        self.ml_agent        = ml_agent
        self.data_agent      = data_agent
        self.strategy_agent  = strategy_agent
        self._ml_file_status_cache: dict = {}
        self._option_ltp_cache: dict[str, tuple[float, float]] = {}
        self._option_ltp_fail_until: float = 0.0
        self._latest_signal_payload: dict = {}
        self._enriched_rejected_cache: dict[str, dict] = {}

    def register(self):
        topics_to_handlers = [
            (Topic.MARKET_REGIME,    self.on_regime),
            (Topic.RAW_SIGNAL,       self.on_raw_signal),
            (Topic.SIGNAL_SUPPRESSED,self.on_suppressed),
            (Topic.SIGNAL_REJECTED,  self.on_rejected),
            (Topic.TRADE_PLAN_READY, self.on_trade_plan),
            (Topic.ORDER_DRY_RUN,    self.on_order_event),
            (Topic.ORDER_CONFIRM_REQ,self.on_order_event),
            (Topic.ORDER_PLACED,     self.on_order_event),
            (Topic.POSITION_UPDATE,  self.on_position_update),
            (Topic.POSITION_CLOSED,  self.on_position_closed),
            (Topic.EOD_REPORT_READY, self.on_eod),
            (Topic.ALERT,            self.on_alert),
            (Topic.CANDLES_READY,    self.on_candles),
            (Topic.TICK_UPDATE,      self.on_candles),
            (Topic.ORB_FORMED,       self.on_orb),
            (Topic.SYSTEM_STATUS,    self.on_system_status),
            ("LIVE_TRADE_HISTORY_UPDATED", self.on_trade_history_updated),
        ]
        for topic, handler in topics_to_handlers:
            self.bus.subscribe(topic, handler)
        logger.info(f"[{self.NAME}] Registered.")

    async def on_raw_signal(self, msg: Message):
        payload = dict(msg.payload or {})
        votes = int(payload.get("votes") or 0)
        if votes < 4:
            return
        payload.setdefault("lifecycle_status", "RAW")
        payload.setdefault("timestamp", payload.get("timestamp") or datetime.now(IST).isoformat())
        self._latest_signal_payload = payload
        socketio.emit("raw_signal", self._to_json_safe(payload))

    async def on_regime(self, msg: Message):
        socketio.emit("regime_update", self._to_json_safe({
            "regime": msg.payload.get("regime"),
            "adx":    msg.payload.get("regime_details", {}).get("adx", 0),
            "chop":   msg.payload.get("regime_details", {}).get("chop_index", 0),
        }))

    async def on_rejected(self, msg: Message):
        enriched = self._enrich_rejected_signal(msg.payload)
        votes = int(enriched.get("votes") or 0)
        # Signals with < 4 votes are sub-threshold noise; require >= 4 votes for reporting
        min_report_votes = max(4, int(os.getenv("MIN_REJECTION_REPORT_VOTES", os.getenv("MIN_STRATEGY_VOTES", "4"))))
        if votes < min_report_votes:
            return

        enriched.setdefault("lifecycle_status", "REJECTED")
        enriched.setdefault("timestamp", enriched.get("timestamp") or datetime.now(IST).isoformat())
        self._latest_signal_payload = enriched
        
        # Cache enriched rejected signal and persist to journal so past prices are frozen permanently
        sig_id = str(enriched.get("signal_id") or "")
        if sig_id:
            self._enriched_rejected_cache[sig_id] = dict(enriched)
        time_key = f"{enriched.get('timestamp')}_{enriched.get('direction')}_{enriched.get('symbol')}"
        self._enriched_rejected_cache[time_key] = dict(enriched)
        self._persist_rejected_signal_to_journal(enriched)

        socketio.emit("signal_rejected", self._to_json_safe(enriched))

        try:
            direction = str(enriched.get("direction") or "").replace("_", " ").upper()
            is_call = "CALL" in direction
            arrow = "⬆" if is_call else "⬇"
            dir_display = "BUY CALL" if is_call else "BUY PUT"
            
            try:
                timestamp = pd.Timestamp(enriched.get("timestamp") or datetime.now(IST))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(IST)
                else:
                    timestamp = timestamp.tz_convert(IST)
                time_str = timestamp.strftime("%H:%M:%S")
            except Exception:
                time_str = datetime.now(IST).strftime("%H:%M:%S")
                
            opt_symbol = enriched.get("contract_symbol") or enriched.get("option_symbol") or enriched.get("symbol") or ""
            spot = float(
                enriched.get("price")
                or enriched.get("ltp")
                or enriched.get("spot_price")
                or enriched.get("underlying_price")
                or enriched.get("nifty_price")
                or enriched.get("nifty_ltp")
                or 0.0
            )
            votes = enriched.get("votes", 0)
            rank = float(enriched.get("ml_rank_score") or enriched.get("rank") or 0.0)
            
            conf_val = float(
                enriched.get("ml_confidence")
                or enriched.get("confidence")
                or 0.0
            )
            conf_pct = int(conf_val * 100) if conf_val <= 1.0 else int(conf_val)
            
            premium = float(enriched.get("est_premium") or enriched.get("entry_premium") or enriched.get("entry_price") or 0.0)
            sl = float(enriched.get("sl_premium") or enriched.get("sl_price") or 0.0)
            target = float(enriched.get("target_premium") or enriched.get("target_price") or 0.0)
            
            rank = float(enriched.get("ml_rank_score") or enriched.get("rank_score") or 0.0)
            if rank <= 0.0 and conf_pct > 0:
                rank = round((conf_pct / 100.0) * 0.60 + min(1.0, votes / 6.0) * 0.40, 2)

            setup_type = enriched.get("setup_type") or "—"
            setup_strength = float(enriched.get("setup_strength") or 0.0)
            structure_bias = enriched.get("structure_bias") or "—"
            if not structure_bias or str(structure_bias).strip() in ("—", "-", "", "None"):
                structure_bias = "BULLISH" if is_call else "BEARISH" or "—"
            regime = enriched.get("regime") or enriched.get("market_regime") or "—"
            budget_lane = str(enriched.get("budget_lane") or "FULL").upper()
            
            rr = float(enriched.get("risk_reward") or 0.0)
            if rr <= 0 and abs(premium - sl) > 0:
                rr = round(abs(target - premium) / abs(premium - sl), 2)

            sym = str(enriched.get("symbol") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
            level_label = "PREMIUM"

            sl_diff_pct = ((sl - premium) / premium * 100.0) if premium > 0 and sl > 0 else 0.0
            tgt_diff_pct = ((target - premium) / premium * 100.0) if premium > 0 and target > 0 else 0.0
            
            strats_raw = (
                enriched.get("strategies_fired")
                or enriched.get("strategy_combo")
                or enriched.get("strategies")
                or ""
            )
            if isinstance(strats_raw, list):
                strats_list = strats_raw
            elif isinstance(strats_raw, (set, tuple)):
                strats_list = list(strats_raw)
            else:
                strats_list = [s.strip() for s in str(strats_raw).replace("|", ",").split(",") if s.strip()]
            strats_str = ", ".join(strats_list)
            
            contract_score = float(enriched.get("contract_score") or 0.0)
            source = enriched.get("premium_source") or "—"
            rejection_reason = (
                enriched.get("rejection_reason")
                or enriched.get("planner_rejection_reason")
                or enriched.get("ml_decision_reason")
                or enriched.get("suppression_reason")
                or ""
            )
            
            notes = enriched.get("selection_notes") or []
            filtered_notes = []
            if isinstance(notes, list):
                for n in notes:
                    n_str = str(n).strip()
                    if n_str and n_str != "[]" and n_str != rejection_reason and not n_str.startswith("Rejected:"):
                        filtered_notes.append(n_str)
            elif isinstance(notes, str) and notes.strip() and notes.strip() != "[]":
                filtered_notes.append(notes.strip())

            notes_str = "\n".join(filtered_notes[:3])
                
            msg_parts = [
                f"{arrow} {dir_display} — REJECTED",
                f"⏰ {time_str} IST",
                f"🏷️ {opt_symbol or 'No trade plan'}",
                f"📊 {sym} {spot:,.2f}  |  Votes {votes}  |  Rank {rank:.2f}  |  Conf {conf_pct}%  |  Budget {budget_lane}",
                f"🎯 Setup: {setup_type} ({setup_strength:.2f})  |  Bias: {structure_bias}  |  Regime: {regime}",
            ]
            if premium > 0:
                msg_parts.append(f"💵 {level_label}\n₹{premium:,.2f}")
            if sl > 0:
                msg_parts.append(f"🛑 STOP LOSS\n₹{sl:,.2f} ({sl_diff_pct:+.1f}%)")
            if target > 0:
                msg_parts.append(f"🎯 TARGET\n₹{target:,.2f} ({tgt_diff_pct:+.1f}%)")
            if rr > 0:
                msg_parts.append(f"⚖️ Risk-Reward: 1:{rr:.2f}")
            if strats_str:
                msg_parts.append(f"⚡ Strategies ({votes}): {strats_str}")
            msg_parts.append(f"🚫 REASON\n{rejection_reason or 'Rejected before trade plan creation'}")
            msg_parts.append(f"📋 Contract score {contract_score:.2f} | Source {source}")
            if notes_str and notes_str.strip():
                msg_parts.append(f"📝 {notes_str.strip()}")
                
            text = "\n".join(msg_parts)
            notify_rej = os.getenv("TELEGRAM_NOTIFY_REJECTIONS", "true").strip().lower() in ("true", "1", "yes")
            if notify_rej:
                now_ts = time.time()
                last_rej_map = getattr(self, "_last_rejected_telegram_ts", None)
                if last_rej_map is None:
                    self._last_rejected_telegram_ts = {}
                    last_rej_map = self._last_rejected_telegram_ts
                rej_key = f"{sym}_{direction}"
                last_ts = last_rej_map.get(rej_key, 0.0)
                rej_cooldown = float(os.getenv("REJECTED_TELEGRAM_COOLDOWN_SEC", "300"))
                if now_ts - last_ts >= rej_cooldown:
                    last_rej_map[rej_key] = now_ts
                    self._telegram(text, parse_mode=None)
                else:
                    logger.debug(f"[DashboardAgent] Suppressed duplicate telegram rejection alert for {rej_key} (cooldown {rej_cooldown}s)")
        except Exception as exc:
            logger.error(f"[DashboardAgent] Error sending telegram notification for rejected signal: {exc}", exc_info=True)

    async def on_suppressed(self, msg: Message):
        socketio.emit("signal_suppressed", self._to_json_safe(msg.payload))

    def _live_option_ltp(self, option_symbol: str) -> float:
        if not option_symbol:
            return 0.0
        now = time.time()
        cached = self._option_ltp_cache.get(option_symbol)
        if cached and cached[1] > now:
            return float(cached[0])
        if self._option_ltp_fail_until > now and cached and cached[0] > 0:
            return float(cached[0])
        
        getter = getattr(self.data_agent, "get_option_ltp", None) if self.data_agent else None
        if not callable(getter):
            try:
                from broker.factory import get_broker
                broker = get_broker()
                getter = getattr(broker, "get_option_ltp", None)
            except Exception:
                getter = None

        if not callable(getter):
            return float(cached[0]) if cached and cached[0] > 0 else 0.0
        try:
            ltp = float(getter(option_symbol) or 0.0)
            if ltp > 0:
                self._option_ltp_cache[option_symbol] = (ltp, now + 30.0)
                return ltp
            if cached and cached[0] > 0:
                return float(cached[0])
            return 0.0
        except Exception as exc:
            self._option_ltp_fail_until = now + 15.0
            if cached and cached[0] > 0:
                return float(cached[0])
            logger.debug(
                f"[DashboardAgent] Live option LTP lookup failed | "
                f"symbol={option_symbol} error={exc}"
            )
            return 0.0

    def _enrich_rejected_signal(self, payload: dict) -> dict:
        enriched = dict(payload or {})
        spot = float(
            enriched.get("price")
            or enriched.get("ltp")
            or enriched.get("spot_price")
            or enriched.get("underlying_price")
            or enriched.get("nifty_price")
            or enriched.get("nifty_ltp")
            or 0.0
        )
        if spot <= 0:
            return enriched

        direction = str(enriched.get("direction") or "").upper()
        option_type = "CE" if "CALL" in direction else "PE"
        try:
            timestamp = pd.Timestamp(enriched.get("timestamp") or datetime.now(IST))
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(IST)
            else:
                timestamp = timestamp.tz_convert(IST)
            sym = str(enriched.get("symbol") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper().strip()
            expiry, dte = get_nearest_expiry(
                min_days=0,
                reference_date=timestamp.date(),
                symbol=sym,
            )
            strike = get_atm_strike(spot, symbol=sym)
            option_symbol = str(enriched.get("option_symbol") or "") or build_option_symbol(
                sym, expiry, strike, option_type
            )
            live_premium_raw = self._live_option_ltp(option_symbol)
            live_premium = live_premium_raw if live_premium_raw >= 1.0 else 0.0
            premium = (
                round(live_premium, 1)
                if live_premium > 0
                else estimate_atm_premium(spot, max(dte, 1), symbol=sym)
            )
            if live_premium > 0:
                self._option_ltp_cache[option_symbol] = (live_premium, time.time() + 60.0)
            premium_source = "LIVE" if live_premium > 0 else "ESTIMATED_REJECTED"
            confidence = float(
                enriched.get("ml_confidence")
                or enriched.get("confidence")
                or 0.0
            )
            votes_cnt = int(enriched.get("votes") or 0)
            rank = float(enriched.get("ml_rank_score") or 0.0)
            if rank <= 0.0 and confidence > 0.0:
                rank = round(confidence * 0.60 + min(1.0, votes_cnt / 6.0) * 0.40, 2)
                enriched["ml_rank_score"] = rank

            enriched["option_symbol"] = option_symbol
            enriched["contract_symbol"] = option_symbol
            if float(enriched.get("est_premium") or 0.0) < 1.0:
                enriched["est_premium"] = premium
            if float(enriched.get("entry_premium") or 0.0) < 1.0:
                enriched["entry_premium"] = premium
            enriched["entry_price"] = float(enriched.get("entry_premium") or premium)
            if float(enriched.get("sl_premium") or 0.0) < 1.0:
                enriched["sl_premium"] = round(max(1.0, premium * 0.75), 1)
            enriched["sl_price"] = enriched["sl_premium"]
            if float(enriched.get("target_premium") or 0.0) < 1.0:
                enriched["target_premium"] = round(premium * 1.50, 1)
            enriched["target_price"] = enriched["target_premium"]

            setup = (((enriched.get("metadata") or {}).get("_context") or {}).get("setup") or {})
            market_structure = (
                ((enriched.get("metadata") or {}).get("_context") or {}).get("market_structure")
                or setup.get("context", {}).get("market_structure")
                or {}
            )
            enriched.setdefault("setup_type", setup.get("setup_type") or enriched.get("source") or "—")
            enriched.setdefault("setup_strength", float(setup.get("setup_strength", 0.0) or 0.0))

            bias_val = (
                enriched.get("structure_bias")
                or setup.get("structure_bias")
                or market_structure.get("bias")
                or market_structure.get("structure_state", {}).get("bias")
                or enriched.get("bias")
            )
            if not bias_val or str(bias_val).strip() in ("—", "-", "", "None"):
                reg_name = str(enriched.get("regime") or enriched.get("market_regime") or setup.get("regime") or "").upper()
                if "TREND" in reg_name:
                    bias_val = "BULLISH" if "CALL" in direction else "BEARISH"
                elif "RANGE" in reg_name:
                    bias_val = "RANGING"
                else:
                    bias_val = "BULLISH" if "CALL" in direction else "BEARISH"
            enriched["structure_bias"] = bias_val
            enriched.setdefault("regime", enriched.get("market_regime") or setup.get("regime") or "—")
            enriched.setdefault("budget_lane", str(enriched.get("trade_size_lane") or (((enriched.get("metadata") or {}).get("reduced_budget_lane")) and "REDUCED") or "FULL").upper())
            prem_val = float(enriched.get("entry_premium") or premium)
            sl_val = float(enriched.get("sl_premium") or 0.0)
            tgt_val = float(enriched.get("target_premium") or 0.0)
            risk = abs(prem_val - sl_val)
            reward = abs(tgt_val - prem_val)
            if risk > 0 and not enriched.get("risk_reward"):
                enriched["risk_reward"] = round(reward / risk, 2)
            if float(enriched.get("target_premium") or 0.0) < 1.0:
                enriched["target_premium"] = round(premium * 1.50, 1)
            if float(enriched.get("contract_score") or 0.0) <= 0:
                enriched["contract_score"] = round(max(rank, confidence), 3)
            existing_source = str(enriched.get("premium_source") or "")
            if live_premium > 0 or not existing_source:
                enriched["premium_source"] = premium_source
            reason = (
                enriched.get("rejection_reason")
                or enriched.get("planner_rejection_reason")
                or enriched.get("ml_decision_reason")
                or enriched.get("suppression_reason")
                or ""
            )
            context_note = (
                "Live/cache ATM contract premium for dashboard context; no trade plan was created."
                if live_premium > 0
                else "Estimated ATM contract for dashboard context; no trade plan was created."
            )
            existing_notes = enriched.get("selection_notes") or []
            if isinstance(existing_notes, str):
                existing_notes = [existing_notes] if existing_notes else []
            elif not isinstance(existing_notes, list):
                existing_notes = [str(existing_notes)] if existing_notes else []
            merged_notes = []
            for note in existing_notes:
                n_str = str(note).strip()
                if n_str and n_str != "[]" and not n_str.startswith("Rejected:") and n_str != reason:
                    merged_notes.append(n_str)
            merged_notes.append(context_note)
            deduped_notes = []
            for note in merged_notes:
                if note not in deduped_notes:
                    deduped_notes.append(note)
            enriched["selection_notes"] = deduped_notes
            if not enriched.get("entry_reason") and reason:
                enriched["entry_reason"] = f"Rejected: {reason}"
        except Exception as exc:
            logger.debug(f"[DashboardAgent] Rejected signal enrichment skipped: {exc}")
        return enriched

    async def on_trade_plan(self, msg: Message):
        socketio.emit("new_signal", self._to_json_safe(msg.payload))
        p   = msg.payload
        sig = p.get("signal", {})
        direction = sig.get("direction", p.get("direction", "BUY"))
        is_call = "CALL" in str(direction).upper()
        arrow = "🟢 ▲" if is_call else "🔴 ▼"
        dir_display = "BUY CALL" if is_call else "BUY PUT"

        budget_lane = (
            "REDUCED"
            if bool((p.get("metadata") or {}).get("reduced_budget_lane", p.get("reduced_budget_lane", False)))
               or p.get("budget_lane") == "REDUCED"
               or (sig.get("metadata") or {}).get("reduced_budget_lane", False)
            else "FULL"
        )
        lots = p.get("lots", 1) or 1
        qty = p.get("quantity", 1) or 1
        contract = p.get("contract_symbol") or p.get("option_symbol") or p.get("symbol") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))
        entry_val = float(p.get("entry_price") or p.get("est_premium", 0.0) or 0.0)
        sl_val = float(p.get("sl_price") or p.get("sl_premium", 0.0) or 0.0)
        tgt_val = float(p.get("target_price") or p.get("target_premium", 0.0) or 0.0)
        invested = p.get("total_invested") or (entry_val * qty)
        conf = int((p.get("confidence") or sig.get("confidence") or 0.65) * 100)
        rank = float(p.get("ml_rank_score") or sig.get("ml_rank_score") or 0.0)

        # In LIVE/PAPER mode, actual executed orders are notified on ORDER_PLACED by AnalyticsAgent.
        # Only notify TRADE_PLAN_READY to Telegram if running in pure OBSERVE mode (where orders are not placed).
        if self._get_trading_mode() == "OBSERVE":
            setup_ctx = (p.get("metadata") or {}).get("_context", {}).get("setup", {}) or (sig.get("metadata") or {}).get("_context", {}).get("setup", {}) or {}
            setup_type = p.get("setup_type") or sig.get("setup_type") or setup_ctx.get("setup_type") or "—"
            setup_strength = float(p.get("setup_strength") or sig.get("setup_strength") or setup_ctx.get("setup_strength") or 0.0)
            structure_bias = p.get("structure_bias") or sig.get("structure_bias") or setup_ctx.get("structure_bias") or "—"
            regime = p.get("regime") or sig.get("regime") or p.get("market_regime") or "—"
            strats_list = p.get("strategies_fired") or sig.get("strategies_fired") or []
            if isinstance(strats_list, list):
                strats_str = ", ".join(strats_list)
            else:
                strats_str = str(strats_list)
            rr = float(p.get("risk_reward") or sig.get("risk_reward") or 0.0)
            if rr <= 0 and abs(entry_val - sl_val) > 0:
                rr = round(abs(tgt_val - entry_val) / abs(entry_val - sl_val), 2)

            self._telegram(
                f"{arrow} *{dir_display} — PLAN READY*\n"
                f"⏰ {datetime.now(IST).strftime('%H:%M:%S')} IST\n"
                f"🏷️ Contract: `{contract}`\n"
                f"📊 Budget: *{budget_lane}* ({lots} Lot{'s' if lots > 1 else ''} / {qty} Qty | ₹{invested:,.0f})\n"
                f"🎯 Setup: {setup_type} ({setup_strength:.2f})  |  Bias: {structure_bias}  |  Regime: {regime}\n"
                f"💵 Entry: ~₹{entry_val:,.1f} | SL: ₹{sl_val:,.1f} | Tgt: ₹{tgt_val:,.1f}" + (f" | RR: 1:{rr:.2f}" if rr > 0 else "") + "\n"
                f"⚡ ML: {conf}% | Rank: {rank:.2f} | Votes ({sig.get('votes', len(strats_list))}): {strats_str}"
            )

    async def on_order_event(self, msg: Message):
        socketio.emit("order_event", self._to_json_safe(msg.payload))

    async def on_position_update(self, msg: Message):
        socketio.emit("position_update", self._to_json_safe(msg.payload))

    async def on_position_closed(self, msg: Message):
        socketio.emit("position_closed", self._to_json_safe(msg.payload))
        pnl = msg.payload.get("pnl_pct", 0)
        contract = msg.payload.get("contract_symbol") or msg.payload.get("option_symbol") or msg.payload.get("symbol") or ""
        pnl_inr = float(msg.payload.get("pnl_inr") or msg.payload.get("pnl", 0.0) or 0.0)
        pnl_str = f"{pnl:+.1f}% (₹{pnl_inr:+,.0f})" if pnl_inr != 0.0 else f"{pnl:+.1f}%"
        self._telegram(
            f"{'[GREEN]' if pnl > 0 else '[RED]'} Closed: *{msg.payload.get('exit_reason')}*\n"
            f"P&L: {pnl_str} | {contract}"
        )

    async def on_trade_history_updated(self, msg: Message):
        socketio.emit("trade_history_row", self._to_json_safe(msg.payload))

    async def on_eod(self, msg: Message):
        self._reset_daily_strategy_status(status="EOD_CLOSED")
        socketio.emit("eod_report", self._to_json_safe(msg.payload))
        socketio.emit("status_update", self._to_json_safe(self._status_payload()))

    async def on_alert(self, msg: Message):
        text = msg.payload.get("text", "")
        self._persist_alert(msg.payload)
        socketio.emit("new_alert", self._to_json_safe(msg.payload))
        if msg.payload.get("type") in (
            "morning_brief", "eod_report", "morning_llm_outlook",
            "eod_llm_analysis", "post_trade", "order_error"
        ):
            self._telegram(text)

    async def on_candles(self, msg: Message):
        orb_high = msg.payload.get("orb_high")
        orb_low = msg.payload.get("orb_low")
        orb_range = None
        if orb_high is not None and orb_low is not None:
            orb_range = round(float(orb_high) - float(orb_low), 2)

        socketio.emit(
            "ltp_update",
            self._to_json_safe({
                "ltp": msg.payload.get("ltp", 0),
                "vix": 0.0,
                "ltps": msg.payload.get("ltps", {}),
                "timestamp": msg.payload.get("timestamp", ""),
                "server_timestamp": msg.payload.get("server_timestamp", ""),
                "orb": {
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "range": orb_range,
                } if orb_high is not None else None
            }),
        )

    async def on_orb(self, msg: Message):
        socketio.emit("orb_update", self._to_json_safe(msg.payload))

    async def on_system_status(self, msg: Message):
        socketio.emit("system_status", self._to_json_safe(msg.payload))

    def _telegram(self, text: str, parse_mode: str | None = None):
        try:
            from utils.telegram_notifier import get_notifier
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(get_notifier().send_text(text, target="LIVE", parse_mode=parse_mode))
            except RuntimeError:
                asyncio.run(get_notifier().send_text(text, target="LIVE", parse_mode=parse_mode))
        except Exception as exc:
            logger.debug(f"[DashboardAgent] Error sending telegram notification: {exc}")

    def _get_trading_mode(self) -> str:
        return os.getenv("TRADING_MODE", TRADING_MODE).strip().upper()

    def _status_payload(self) -> dict:
        stats = self.analytics_agent.get_daily_stats() if self.analytics_agent else {}
        bus_stats = self.bus.get_stats().get("by_topic", {})
        journal_rows = self._all_trade_rows(limit=500)
        history = self._live_trade_history_payload(journal_rows)
        persisted_stats = self._persisted_daily_stats()
        today_history = history.get("summary", {}).get("today", {})
        strategy_names = self._strategy_names()
        stats = {
            **stats,
            "signals": max(
                int(stats.get("signals", 0) or 0),
                int(persisted_stats.get("signals", 0) or 0),
                int(bus_stats.get(Topic.RAW_SIGNAL, 0) or 0),
            ),
            "approved": max(
                int(stats.get("approved", 0) or 0),
                int(persisted_stats.get("approved", 0) or 0),
                int(bus_stats.get(Topic.SIGNAL_APPROVED, 0) or 0),
            ),
            "suppressed": max(
                int(stats.get("suppressed", 0) or 0),
                int(persisted_stats.get("suppressed", 0) or 0),
                int(bus_stats.get(Topic.SIGNAL_SUPPRESSED, 0) or 0),
                int(bus_stats.get(Topic.SIGNAL_REJECTED, 0) or 0),
            ),
            "traded": max(int(stats.get("traded", 0) or 0), int(persisted_stats.get("traded", 0) or 0)),
            "wins": max(int(stats.get("wins", 0) or 0), int(persisted_stats.get("wins", 0) or 0)),
            "losses": max(int(stats.get("losses", 0) or 0), int(persisted_stats.get("losses", 0) or 0)),
            "total_pnl_pct": persisted_stats.get("total_pnl_pct", 0),
            "realized_pnl": persisted_stats.get("realized_pnl", 0),
            "closed": persisted_stats.get("closed", persisted_stats.get("traded", 0)),
        }
        if int(today_history.get("trades", 0) or 0) > 0:
            stats = {
                **stats,
                "traded": int(today_history.get("trades", 0) or 0),
                "closed": int(today_history.get("trades", 0) or 0),
                "wins": int(today_history.get("wins", stats.get("wins", 0)) or 0),
                "losses": int(today_history.get("losses", stats.get("losses", 0)) or 0),
                "total_pnl_pct": float(today_history.get("total_pnl_pct", stats.get("total_pnl_pct", 0)) or 0),
                "realized_pnl": float(today_history.get("realized_pnl", 0) or 0),
            }
        pos   = self._open_position_payload()
        risk  = self.risk_agent.status_payload() if self.risk_agent else {}
        ml = self._ml_status()
        orb = self.data_agent.get_orb_state() if self.data_agent and hasattr(self.data_agent, "get_orb_state") else None
        journal_summary = {}
        journal_trades = []
        equity_curve_data = {}
        try:
            from utils.equity_curve import get_equity_curve, get_observe_equity_curve
            from utils.live_trade_history import get_trade_history
            mode = self._get_trading_mode().upper()
            eq = get_observe_equity_curve() if mode == "OBSERVE" else get_equity_curve()
            journal_summary = eq.get_summary()
            journal_trades = get_trade_history().rows()
            equity_curve_data = eq.get_equity_curve_data()
            commodity_equity_curves = eq.commodity_equity_curves() if hasattr(eq, "commodity_equity_curves") else {}
        except Exception:
            commodity_equity_curves = {}

        ltps = {}
        ltp = 0.0
        if self.data_agent and hasattr(self.data_agent, "get_latest_ltps"):
            try:
                ltps = self.data_agent.get_latest_ltps()
                ltp = self.data_agent.get_ltp()
            except Exception:
                pass

        positions = self._all_open_positions_payload()

        return {
            **stats,
            "ltp": ltp,
            "ltps": ltps,
            "position": pos,
            "positions": positions,
            "mode": self._get_trading_mode(),
            "risk": risk,
            "ml": ml,
            "orb": orb,
            "broker": get_active_broker_name().upper(),
            "pipeline": self._pipeline_status(stats),
            "strategy_status": self._strategy_status(strategy_names),
            "strategy_count": len(strategy_names),
            "equity_curve": self._equity_curve_month(),
            "equity_curve_data": equity_curve_data,
            "commodity_equity_curves": commodity_equity_curves,
            "journal_summary": journal_summary,
            "journal_trades": journal_trades,
            "live_history": history,
            "commodity_summaries": history.get("commodity_summaries", {}),
            "alerts": self._recent_alerts_payload(limit=30).get("rows", []),
            "signal_feed": self._today_signal_feed_payload(limit=80),
            "journal_rows": journal_rows[:200],
        }

    def _market_participation_payload(self) -> dict:
        return {
            "available": False,
            "message": "FII/DII equity market data not applicable for MCX commodity options.",
        }

    def _all_trade_rows(self, limit: int = 200) -> list[dict]:
        try:
            return get_trade_history().rows(limit=limit)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] All trade history load failed: {exc}")
            return []

    @staticmethod
    def _trade_row_key(row: dict) -> str:
        signal_id = str(row.get("signal_id") or "").strip()
        if signal_id:
            return signal_id
        parts = [
            str(row.get("date") or ""),
            str(row.get("entry_time") or row.get("timestamp") or ""),
            str(row.get("option_symbol") or ""),
            str(row.get("direction") or ""),
            str(row.get("entry_premium") or row.get("actual_premium") or row.get("entry_price") or ""),
        ]
        return "|".join(parts).strip("|")

    def _persisted_journal_trade_rows(self, limit: int = 1000) -> list[dict]:
        rows: list[dict] = []
        journal_dir = Path(JOURNAL_DIR)

        for path in sorted(journal_dir.glob("signals_*.csv"), reverse=True):
            try:
                with path.open(newline="", errors="replace") as handle:
                    for row in csv.DictReader(handle):
                        if row.get("signal_id") == "signal_id":
                            continue
                        if not row.get("direction") and not row.get("signal_id"):
                            continue
                        if self._is_backtest_trade_row(row):
                            continue
                        rows.append(self._normalize_signal_feed_row(row))
            except Exception as exc:
                logger.debug(f"[{self.NAME}] Historical signal journal load failed for {path.name}: {exc}")

        closed_path = journal_dir / "closed_trades.csv"
        if closed_path.exists():
            try:
                with closed_path.open(newline="", errors="replace") as handle:
                    for row in csv.DictReader(handle):
                        if self._is_backtest_trade_row(row):
                            continue
                        rows.append(row)
            except Exception as exc:
                logger.debug(f"[{self.NAME}] Closed trade journal load failed: {exc}")

        rows.sort(
            key=lambda r: str(r.get("entry_time") or r.get("timestamp") or r.get("trade_date") or r.get("date") or ""),
            reverse=True,
        )
        return rows[:max(1, min(int(limit or 1000), 2000))]

    @staticmethod
    def _is_backtest_trade_row(row: dict) -> bool:
        return (
            str(row.get("mode") or "").upper() == "BACKTEST"
            or str(row.get("is_backtest") or "").lower() == "true"
        )

    @staticmethod
    def _is_empty_closed_trade_row(row: dict) -> bool:
        lifecycle = str(row.get("lifecycle_status", "") or "").upper()
        if lifecycle != "CLOSED":
            return False
        has_exit = bool(str(row.get("exit_time") or row.get("exit_reason") or "").strip())
        has_outcome = bool(str(row.get("outcome_eod") or "").strip())
        has_exec = bool(str(row.get("mode") or row.get("execution") or "").strip())
        has_pnl = any(
            str(row.get(key) or "").strip()
            for key in ("pnl_pct", "realized_pnl", "exit_premium", "exit_price")
        )
        return not (has_exit or has_outcome or has_exec or has_pnl)

    @staticmethod
    def _normalize_history_trade_row(row: dict) -> dict:
        normalized = dict(row or {})
        if "strategies_fired" in normalized and "strategy_combo" not in normalized:
            normalized["strategy_combo"] = normalized.get("strategies_fired", "")
        if not normalized.get("date"):
            normalized["date"] = (
                normalized.get("trade_date")
                or str(normalized.get("entry_time") or normalized.get("timestamp") or "")[:10]
            )
        if "entry_premium" not in normalized or not normalized.get("entry_premium"):
            normalized["entry_premium"] = (
                normalized.get("actual_premium")
                or normalized.get("entry_price")
                or normalized.get("est_premium", 0)
            )
        if "actual_premium" not in normalized or not normalized.get("actual_premium"):
            normalized["actual_premium"] = normalized.get("entry_premium", 0)
        if "exit_premium" not in normalized or not normalized.get("exit_premium"):
            normalized["exit_premium"] = normalized.get("exit_price", 0)
        if "total_invested" not in normalized or not normalized.get("total_invested"):
            normalized["total_invested"] = normalized.get("invested", 0)
        if "timestamp" not in normalized or not normalized.get("timestamp"):
            normalized["timestamp"] = normalized.get("entry_time") or normalized.get("date", "")
        if "lifecycle_status" not in normalized or not normalized.get("lifecycle_status"):
            normalized["lifecycle_status"] = (
                "CLOSED" if normalized.get("exit_time") or normalized.get("exit_reason") else "ORDERED"
            )
        if "ml_confidence" not in normalized or not normalized.get("ml_confidence"):
            normalized["ml_confidence"] = normalized.get("ml_conf", 0)
        if "premium_source" not in normalized or not normalized.get("premium_source"):
            normalized["premium_source"] = normalized.get("source", "live history")
        DashboardAlertAgent._reconcile_trade_pnl_fields(normalized)
        return normalized

    @staticmethod
    def _reconcile_trade_pnl_fields(row: dict) -> dict:
        entry_premium = DashboardAlertAgent._safe_float(
            row.get("entry_premium", row.get("actual_premium", row.get("est_premium", 0)))
        )
        exit_premium = DashboardAlertAgent._safe_float(
            row.get("exit_premium", row.get("exit_price", 0))
        )
        quantity = DashboardAlertAgent._safe_int(
            row.get("quantity", row.get("lot_size", row.get("lots", 0)))
        )
        pnl_pct = DashboardAlertAgent._safe_float(row.get("pnl_pct"))
        realized_pnl = DashboardAlertAgent._safe_float(row.get("realized_pnl"))

        if entry_premium > 0 and exit_premium > 0:
            if quantity > 0 and abs(realized_pnl) < 1e-9:
                realized_pnl = round((exit_premium - entry_premium) * quantity, 2)
                row["realized_pnl"] = realized_pnl
            if abs(pnl_pct) < 1e-9:
                pnl_pct = round(((exit_premium - entry_premium) / entry_premium) * 100, 2)
                row["pnl_pct"] = pnl_pct

        outcome = str(row.get("outcome_eod") or "").upper()
        if pnl_pct > 0:
            outcome = "WIN"
        elif pnl_pct < 0:
            outcome = "LOSS"
        elif str(row.get("exit_time") or row.get("exit_reason") or "").strip():
            outcome = "FLAT"
        row["outcome_eod"] = outcome
        row["exit_premium"] = exit_premium
        row["entry_premium"] = entry_premium
        return row

    def _trade_summary_from_rows(self, rows: list[dict]) -> dict:
        today = _today_iso()
        month = today[:7]
        buckets = {
            "today": {"trades": 0, "wins": 0, "losses": 0, "total_pnl_pct": 0.0, "realized_pnl": 0.0, "gross_pnl": 0.0, "total_charges": 0.0, "net_pnl": 0.0},
            "month": {"trades": 0, "wins": 0, "losses": 0, "total_pnl_pct": 0.0, "realized_pnl": 0.0, "gross_pnl": 0.0, "total_charges": 0.0, "net_pnl": 0.0},
            "all": {"trades": 0, "wins": 0, "losses": 0, "total_pnl_pct": 0.0, "realized_pnl": 0.0, "gross_pnl": 0.0, "total_charges": 0.0, "net_pnl": 0.0},
        }
        seen: set[str] = set()
        for raw in rows or []:
            row = self._normalize_history_trade_row(raw)
            if self._is_empty_closed_trade_row(row):
                continue
            lifecycle = str(row.get("lifecycle_status", "") or "").upper()
            if lifecycle != "CLOSED" and not row.get("exit_time") and not row.get("exit_reason"):
                continue
            key = self._trade_row_key(row)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            trade_date = str(row.get("date") or row.get("trade_date") or row.get("entry_time") or "")[:10]
            pnl_pct = self._safe_float(row.get("pnl_pct"))
            realized = self._safe_float(row.get("net_pnl_inr", row.get("realized_pnl")))
            gross = self._safe_float(row.get("gross_pnl_inr"))
            charges = self._safe_float(row.get("total_charges"))
            if gross == 0.0 and realized != 0.0 and charges > 0.0:
                gross = realized + charges
            outcome = str(row.get("outcome_eod") or "").upper()

            names = ["all"]
            if trade_date.startswith(month):
                names.append("month")
            if trade_date == today:
                names.append("today")

            for name in names:
                bucket = buckets[name]
                bucket["trades"] += 1
                bucket["total_pnl_pct"] += pnl_pct
                bucket["realized_pnl"] += realized
                bucket["net_pnl"] += realized
                bucket["gross_pnl"] += gross
                bucket["total_charges"] += charges
                if outcome == "WIN" or pnl_pct > 0 or realized > 0:
                    bucket["wins"] += 1
                elif outcome == "LOSS" or pnl_pct < 0 or realized < 0:
                    bucket["losses"] += 1

        for bucket in buckets.values():
            bucket["total_pnl_pct"] = round(bucket["total_pnl_pct"], 2)
            bucket["realized_pnl"] = round(bucket["realized_pnl"], 2)
            bucket["net_pnl"] = round(bucket["net_pnl"], 2)
            bucket["gross_pnl"] = round(bucket["gross_pnl"], 2)
            bucket["total_charges"] = round(bucket["total_charges"], 2)
        return buckets

    def _persisted_daily_stats(self) -> dict:
        rows = self._today_journal_rows(limit=500)
        signals = 0
        approved = 0
        suppressed = 0
        traded = 0
        wins = 0
        losses = 0
        total_pnl_pct = 0.0
        realized_pnl = 0.0
        seen_signals: set[str] = set()
        seen_trades: set[str] = set()

        for row in rows:
            signal_id = str(row.get("signal_id", "") or "").strip()
            signal_key = signal_id or "|".join([
                str(row.get("date", "")),
                str(row.get("time", "")),
                str(row.get("direction", "")),
                str(row.get("strategy_combo", "")),
            ])
            if signal_key and signal_key not in seen_signals:
                seen_signals.add(signal_key)
                signals += 1

            status = str(row.get("lifecycle_status", "") or "").upper()
            rejection = str(row.get("rejection_reason", "") or "").strip()
            suppression = str(row.get("suppression_reason", "") or "").strip()
            option_symbol = str(row.get("option_symbol", "") or "").strip()

            if suppression or rejection or status in {"SUPPRESSED", "REJECTED"}:
                suppressed += 1
            if not rejection and not suppression and status in {"APPROVED", "PLANNED", "ORDERED", "OPEN", "CLOSED"}:
                approved += 1

            if not self._is_taken_trade_row(row):
                continue

            trade_key = option_symbol or signal_key
            if trade_key in seen_trades:
                continue
            seen_trades.add(trade_key)
            traded += 1

        # Reconcile trade metrics directly from canonical trade history
        try:
            from utils.live_trade_history import get_trade_history
            hist_today = get_trade_history().summary().get("today", {})
            if int(hist_today.get("trades", 0) or 0) > 0:
                traded = int(hist_today.get("trades", 0))
                wins = int(hist_today.get("wins", 0))
                losses = int(hist_today.get("losses", 0))
                total_pnl_pct = float(hist_today.get("total_pnl_pct", 0.0))
                realized_pnl = float(hist_today.get("realized_pnl", 0.0))
        except Exception:
            pass

        return {
            "signals": max(signals, approved + suppressed),
            "approved": approved,
            "suppressed": suppressed,
            "traded": traded,
            "closed": traded,
            "wins": wins,
            "losses": losses,
            "total_pnl_pct": round(total_pnl_pct, 2),
            "realized_pnl": round(realized_pnl, 2),
        }

    def _dashboard_alert_path(self, day: str | None = None) -> Path:
        return Path(JOURNAL_DIR) / f"dashboard_alerts_{day or _today_iso()}.jsonl"

    def _persist_alert(self, payload: dict) -> None:
        try:
            row = {
                "timestamp": payload.get("timestamp") or datetime.now(IST).isoformat(),
                "type": payload.get("type", "alert"),
                "severity": payload.get("severity", ""),
                "text": payload.get("text", ""),
                "payload": payload,
            }
            path = self._dashboard_alert_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Alert persistence skipped: {exc}")

    def _recent_alerts_payload(self, limit: int = 50) -> dict:
        path = self._dashboard_alert_path()
        rows: list[dict] = []
        if path.exists():
            try:
                for line in path.read_text(errors="replace").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows.append({
                        "timestamp": row.get("timestamp", ""),
                        "type": row.get("type", "alert"),
                        "severity": row.get("severity", ""),
                        "text": row.get("text", ""),
                    })
            except Exception as exc:
                logger.debug(f"[{self.NAME}] Alert load failed: {exc}")

        if not rows:
            llm = get_llm_status()
            rows.append({
                "timestamp": datetime.now(IST).isoformat(),
                "type": "dashboard_state",
                "severity": "INFO",
                "text": (
                    f"Dashboard restored. LLM provider={llm.get('active_provider', 'unknown')} "
                    f"model={llm.get('active_model', 'unknown')} enabled={llm.get('enabled', False)}."
                ),
            })
        return {
            "date": _today_iso(),
            "rows": rows[-max(1, min(int(limit or 50), 200)):],
        }

    def _open_position_payload(self, symbol: str | None = None) -> dict | None:
        pos = None
        if self.position_agent and hasattr(self.position_agent, "get_position"):
            try:
                pos = self.position_agent.get_position(symbol=symbol)
            except Exception:
                pos = self.position_agent.get_position()
        if pos:
            return {**pos, "source": "position_manager"}

        for row in reversed(self._today_journal_rows(limit=300)):
            status = str(row.get("lifecycle_status", "")).upper()
            if status != "ORDERED":
                continue
            if str(row.get("exit_time", "")).strip():
                continue
            option_symbol = str(row.get("option_symbol", "") or "")
            if not option_symbol:
                continue
            if symbol and symbol.strip().upper() not in ("", "ALL"):
                sym_clean = symbol.strip().upper()
                prefixes = [sym_clean]
                if "SILVER" in sym_clean: prefixes = ["SILVERM", "SILVERMIC", "SILVER"]
                elif "GOLD" in sym_clean: prefixes = ["GOLDM", "GOLD"]
                elif "CRUDE" in sym_clean: prefixes = ["CRUDEOILM", "CRUDEOIL", "CRUDE"]
                elif "NAT" in sym_clean: prefixes = ["NATGASM", "NATGASMINI", "NATGAS", "NATURALGAS"]
                if not any(p in option_symbol.upper() or p in str(row.get("symbol", "")).upper() for p in prefixes):
                    continue

            entry = self._safe_float(row.get("actual_premium") or row.get("entry_premium") or row.get("est_premium"))
            return {
                "option_symbol": option_symbol,
                "entry_premium": entry,
                "current_premium": entry,
                "current_ltp": entry,
                "peak_premium": entry,
                "sl_premium": self._safe_float(row.get("sl_premium")),
                "target_premium": self._safe_float(row.get("target_premium")),
                "pnl_pct": self._safe_float(row.get("pnl_pct")),
                "is_open": True,
                "is_simulated": str(row.get("execution", "")).upper() != "PLACED",
                "simulated": str(row.get("execution", "")).upper() != "PLACED",
                "entry_time": row.get("entry_time", ""),
                "execution_mode": row.get("execution", row.get("mode", "")),
                "quantity": int(self._safe_float(row.get("quantity"))),
                "source": "persisted_journal",
                "direction": row.get("direction", ""),
                "nifty_price": self._safe_float(row.get("nifty_price")),
                "strategies_fired": row.get("strategies_fired", ""),
                "ml_rank_score": self._safe_float(row.get("ml_rank_score")),
                "ml_confidence": self._safe_float(row.get("ml_confidence") or row.get("ml_conf")),
            }
        return None

    def _all_open_positions_payload(self) -> dict[str, dict | None]:
        commodities = ["SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"]
        return {c: self._open_position_payload(symbol=c) for c in commodities}

    def _live_trade_history_payload(self, rows: list[dict] | None = None) -> dict:
        try:
            from utils.live_trade_history import get_live_trade_history, get_observe_trade_history
            from utils.equity_curve import get_live_equity_curve, get_observe_equity_curve
            current_mode = str(os.getenv("TRADING_MODE", "OBSERVE")).strip().upper()
            live_hist = get_live_trade_history()
            live_eq = get_live_equity_curve()
            obs_hist = get_observe_trade_history()
            obs_eq = get_observe_equity_curve()
            return {
                "trading_mode": current_mode,
                "live": {
                    "trades": live_hist.rows(limit=50),
                    "summary": live_eq.get_summary(),
                },
                "observe": {
                    "trades": obs_hist.rows(limit=50),
                    "summary": obs_eq.get_summary(),
                },
                "summary": obs_hist.summary() if current_mode == "OBSERVE" else live_hist.summary(),
                "recent_trades": obs_hist.rows(limit=50) if current_mode == "OBSERVE" else live_hist.rows(limit=50),
                "commodity_summaries": obs_hist.commodity_summaries() if current_mode == "OBSERVE" else live_hist.commodity_summaries(),
            }
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Live trade history unavailable: {exc}")
            return {
                "summary": {"today": {}, "month": {}, "all": {}},
                "recent_trades": [],
                "live": {"trades": [], "summary": {}},
                "observe": {"trades": [], "summary": {}},
            }

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value if value not in (None, "") else default)
        except Exception:
            return default

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(float(value if value not in (None, "") else default))
        except Exception:
            return default

    def _today_journal_rows(self, limit: int = 100) -> list[dict]:
        path = Path(JOURNAL_DIR) / f"signals_{_today_iso()}.csv"
        if not path.exists():
            return []

        rows: list[dict] = []
        try:
            with path.open(newline="", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    try:
                        if row.get("signal_id") == "signal_id":
                            continue
                        if not row.get("direction") and not row.get("signal_id"):
                            continue
                        # HARD FILTER: Only display signals with at least 4 votes
                        row_votes = int(float(row.get("votes") or 0))
                        if row_votes < 4:
                            continue
                        rows.append(self._normalize_signal_feed_row(row))
                    except Exception as row_exc:
                        logger.debug(f"[{self.NAME}] Skipping bad journal row: {row_exc}")
                        continue
        except Exception as exc:
            logger.error(f"[{self.NAME}] Journal file access failed: {exc}")
            return []
        rows.sort(
            key=lambda row: str(row.get("timestamp") or row.get("entry_time") or row.get("date") or ""),
            reverse=True,
        )
        return rows[:max(1, min(int(limit or 100), 500))]

    @staticmethod
    def _is_taken_trade_row(row: dict) -> bool:
        lifecycle = str(row.get("lifecycle_status", "") or "").upper()
        execution = str(row.get("execution", "") or row.get("mode", "") or "").upper()
        if (
            lifecycle in {"REJECTED", "SUPPRESSED", "APPROVED", "PLANNED"}
            or row.get("rejection_reason")
            or row.get("suppression_reason")
        ):
            return False
        option_symbol = str(row.get("option_symbol", "") or "").strip()
        return bool(option_symbol) and (
            lifecycle in {"ORDERED", "OPEN", "CLOSED"}
            or execution in {"DRY_RUN", "PLACED", "AUTO", "MANUAL", "BACKTEST"}
        )

    def _today_trade_rows(self, limit: int = 100) -> list[dict]:
        rows = [row for row in self._today_journal_rows(limit=500) if self._is_taken_trade_row(row)]
        rows.sort(
            key=lambda row: str(row.get("timestamp") or row.get("entry_time") or row.get("date") or ""),
            reverse=True,
        )
        return rows[:max(1, min(int(limit or 100), 500))]

    def _today_signal_feed_payload(self, limit: int = 50, symbol: str | None = None) -> dict:
        max_rows = max(1, min(int(limit or 50), 200))
        rows = self._today_journal_rows(limit=max_rows * 2)
        if symbol and symbol.strip().upper() != "ALL":
            target_sym = symbol.strip().upper()
            rows = [r for r in rows if str(r.get("symbol") or "").upper() == target_sym]
        latest = self._latest_signal_feed_row()
        if latest:
            latest_sym = str(latest.get("symbol") or "").upper()
            if not symbol or symbol.strip().upper() == "ALL" or latest_sym == symbol.strip().upper():
                latest_key = str(latest.get("signal_id") or latest.get("timestamp") or "")
                existing_keys = {
                    str(row.get("signal_id") or row.get("timestamp") or "")
                    for row in rows
                }
                if latest_key and latest_key not in existing_keys:
                    rows = [latest, *rows]
        rows.sort(
            key=lambda row: str(row.get("timestamp") or row.get("entry_time") or row.get("date") or ""),
            reverse=True,
        )
        return {
            "date": _today_iso(),
            "rows": rows[:max_rows],
            "taken_rows": self._today_trade_rows(limit=max_rows),
        }

    def _latest_signal_feed_row(self) -> dict:
        payload = dict(self._latest_signal_payload or {})
        if not payload:
            return {}
        if int(payload.get("votes") or 0) < 4:
            return {}
        ts_value = payload.get("timestamp") or payload.get("entry_time") or datetime.now(IST).isoformat()
        try:
            ts = pd.Timestamp(ts_value)
            if ts.tzinfo is None:
                ts = ts.tz_localize(IST)
            else:
                ts = ts.tz_convert(IST)
            date_value = ts.date().isoformat()
            time_value = ts.strftime("%H:%M")
            timestamp = ts.isoformat()
        except Exception:
            date_value = _today_iso()
            time_value = datetime.now(IST).strftime("%H:%M")
            timestamp = str(ts_value)
        if date_value != _today_iso():
            return {}
        strategies = payload.get("strategies_fired") or payload.get("strategy_combo") or []
        if isinstance(strategies, str):
            strategy_combo = strategies.replace(",", "|")
        else:
            strategy_combo = "|".join(str(s) for s in strategies)
        return self._normalize_signal_feed_row({
            "signal_id": payload.get("signal_id") or f"{timestamp}|{payload.get('direction', '')}|{strategy_combo}",
            "date": date_value,
            "time": time_value,
            "entry_time": timestamp,
            "timestamp": timestamp,
            "symbol": payload.get("symbol") or os.getenv("INSTRUMENT", "SILVERM"),
            "direction": getattr(payload.get("direction"), "value", payload.get("direction", "")),
            "nifty_price": payload.get("nifty_price") or payload.get("nifty_ltp") or payload.get("ltp") or 0,
            "strategies_fired": strategy_combo,
            "strategy_combo": strategy_combo,
            "votes": payload.get("votes", 0),
            "strategy_conf": payload.get("confidence", 0),
            "ml_conf": payload.get("ml_confidence") or payload.get("confidence", 0),
            "ml_confidence": payload.get("ml_confidence") or payload.get("confidence", 0),
            "ml_rank_score": payload.get("ml_rank_score") or payload.get("rank", 0),
            "ml_decision_reason": payload.get("ml_decision_reason", ""),
            "rejection_reason": payload.get("rejection_reason", ""),
            "lifecycle_status": payload.get("lifecycle_status", "RAW"),
            "setup_type": payload.get("setup_type") or payload.get("metadata", {}).get("_context", {}).get("setup", {}).get("setup_type", ""),
            "setup_strength": payload.get("setup_strength") or payload.get("metadata", {}).get("_context", {}).get("setup", {}).get("setup_strength", 0),
            "market_regime_detail": payload.get("regime", ""),
            "premium_source": payload.get("premium_source", "LIVE"),
            "budget_lane": (
                "REDUCED"
                if bool((payload.get("metadata") or {}).get("reduced_budget_lane", payload.get("reduced_budget_lane", False)))
                else "FULL"
            ),
        })

    def _normalize_signal_feed_row(self, row: dict) -> dict:
        def _num(key: str, default: float = 0.0) -> float:
            return DashboardAlertAgent._safe_float(row.get(key), default)

        def _int(key: str, default: int = 0) -> int:
            return DashboardAlertAgent._safe_int(row.get(key), default)

        strategies = [
            s.strip()
            for s in str(row.get("strategies_fired") or row.get("strategy_combo") or "").split("|")
            if s.strip()
        ]
        normalized = {
            "signal_id": row.get("signal_id", ""),
            "date": row.get("date", ""),
            "timestamp": row.get("entry_time") or f"{row.get('date', '')}T{row.get('time', '00:00')}:00+05:30",
            "entry_time": row.get("entry_time") or f"{row.get('date', '')}T{row.get('time', '00:00')}:00+05:30",
            "time": row.get("time", ""),
            "symbol": row.get("symbol") or os.getenv("INSTRUMENT", "SILVERM"),
            "direction": row.get("direction", ""),
            "nifty_price": _num("nifty_price"),
            "strategies_fired": strategies,
            "votes": _int("votes"),
            "strategy_conf": _num("strategy_conf"),
            "ml_conf": _num("ml_conf"),
            "ml_confidence": _num("ml_confidence", _num("ml_conf")),
            "ml_decision": row.get("ml_decision", ""),
            "ml_rank_score": _num("ml_rank_score"),
            "ml_rank_tier": row.get("ml_rank_tier", ""),
            "ml_decision_reason": row.get("ml_decision_reason", ""),
            "lifecycle_status": row.get("lifecycle_status", ""),
            "rejection_reason": row.get("rejection_reason", ""),
            "suppression_reason": row.get("suppression_reason", ""),
            "budget_lane": str(row.get("budget_lane", row.get("trade_size_lane", "FULL")) or "FULL").upper(),
            "option_symbol": row.get("option_symbol", ""),
            "strike": _int("strike"),
            "option_type": row.get("option_type", ""),
            "expiry_date": row.get("expiry_date", ""),
            "est_premium": _num("est_premium"),
            "actual_premium": _num("actual_premium"),
            "entry_premium": _num("actual_premium", _num("est_premium")),
            "exit_premium": _num("exit_premium"),
            "lots": _int("lots"),
            "lot_size": _int("lot_size"),
            "quantity": _int("quantity"),
            "total_invested": _num("total_invested"),
            "sl_premium": _num("sl_premium"),
            "target_premium": _num("target_premium"),
            "risk_reward": _num("risk_reward"),
            "premium_source": row.get("premium_source", ""),
            "contract_score": _num("contract_score"),
            "contract_snapshot": row.get("contract_snapshot", ""),
            "selection_notes": row.get("selection_notes", ""),
            "entry_reason": row.get("entry_reason", ""),
            "market_regime_detail": row.get("market_regime_detail", ""),
            "strategy_combo": row.get("strategy_combo", ""),
            "setup_type": row.get("setup_type", ""),
            "setup_strength": _num("setup_strength"),
            "momentum_strength": _num("momentum_strength"),
            "entry_timing": row.get("entry_timing", ""),
            "ml_score_bucket": row.get("ml_score_bucket", ""),
            "structure_bias": row.get("structure_bias", ""),
            "liquidity_event": row.get("liquidity_event", ""),
            "entry_validation": row.get("entry_validation", ""),
            "mode": row.get("mode", ""),
            "execution": row.get("execution", ""),
            "outcome_eod": row.get("outcome_eod", ""),
            "pnl_pct": _num("pnl_pct"),
            "realized_pnl": _num("realized_pnl"),
            "exit_time": row.get("exit_time", ""),
            "exit_reason": row.get("exit_reason", ""),
            "holding_minutes": _int("holding_minutes"),
            "llm_rationale": row.get("llm_rationale", ""),
            "option_chain_snapshot": row.get("option_chain_snapshot", ""),
            "latency_ms": _num("latency_ms"),
            "slippage_pct": _num("slippage_pct"),
        }
        self._reconcile_trade_pnl_fields(normalized)
        if normalized.get("rejection_reason") or str(normalized.get("lifecycle_status")).upper() == "REJECTED":
            sig_id = str(normalized.get("signal_id") or "")
            time_key = f"{normalized.get('timestamp')}_{normalized.get('direction')}_{normalized.get('symbol')}"
            cached_item = self._enriched_rejected_cache.get(sig_id) or self._enriched_rejected_cache.get(time_key)
            if cached_item:
                for k, v in cached_item.items():
                    if v not in (None, "", "—", 0, 0.0) or k not in normalized or normalized[k] in (None, "", "—", 0, 0.0):
                        normalized[k] = v
            elif float(normalized.get("entry_premium") or normalized.get("actual_premium") or normalized.get("est_premium") or 0.0) >= 1.0 and normalized.get("option_symbol"):
                # Frozen historical value preserved in journal — do NOT re-query live market!
                pass
            else:
                normalized = self._enrich_rejected_signal(normalized)
                if sig_id:
                    self._enriched_rejected_cache[sig_id] = dict(normalized)
                self._enriched_rejected_cache[time_key] = dict(normalized)
                self._persist_rejected_signal_to_journal(normalized)
        return normalized

    def _persist_rejected_signal_to_journal(self, enriched: dict) -> None:
        try:
            sig_id = str(enriched.get("signal_id") or "")
            ts_str = str(enriched.get("timestamp") or enriched.get("entry_time") or "")
            opt_sym = str(enriched.get("option_symbol") or enriched.get("contract_symbol") or "")
            prem = float(enriched.get("entry_premium") or enriched.get("est_premium") or 0.0)
            if not opt_sym or prem <= 0:
                return

            date_str = _today_iso()
            if "T" in ts_str:
                date_str = ts_str.split("T")[0]
            csv_path = Path(JOURNAL_DIR) / f"signals_{date_str}.csv"
            if not csv_path.exists():
                return

            with open(csv_path, newline="", errors="replace") as f:
                rows = list(csv.DictReader(f))
                if not rows:
                    return
                fieldnames = list(rows[0].keys())

            updated = False
            for row in rows:
                row_sig = str(row.get("signal_id") or "")
                row_ts = str(row.get("entry_time") or row.get("timestamp") or "")
                match = False
                if sig_id and row_sig and sig_id == row_sig:
                    match = True
                elif ts_str and row_ts and ts_str[:16] == row_ts[:16] and row.get("direction") == enriched.get("direction"):
                    match = True

                if match:
                    def _set_col(col, val):
                        if col in fieldnames and val is not None:
                            row[col] = str(val)

                    _set_col("option_symbol", opt_sym)
                    _set_col("est_premium", enriched.get("est_premium") or prem)
                    _set_col("actual_premium", enriched.get("actual_premium") or prem)
                    _set_col("sl_premium", enriched.get("sl_premium") or round(prem * 0.75, 1))
                    _set_col("target_premium", enriched.get("target_premium") or round(prem * 1.50, 1))
                    _set_col("premium_source", enriched.get("premium_source") or "LIVE")
                    if enriched.get("risk_reward"):
                        _set_col("risk_reward", enriched.get("risk_reward"))
                    if enriched.get("structure_bias"):
                        _set_col("structure_bias", enriched.get("structure_bias"))
                    if enriched.get("ml_rank_score"):
                        _set_col("ml_rank_score", enriched.get("ml_rank_score"))
                    updated = True
                    break

            if updated:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception as exc:
            logger.debug(f"[DashboardAgent] Could not persist rejected signal to journal: {exc}")

    def _pipeline_status(self, stats: dict) -> dict:
        bus_stats = self.bus.get_stats().get("by_topic", {})
        return {
            "candles": int(bus_stats.get(Topic.CANDLES_READY, 0)),
            "regime": int(bus_stats.get(Topic.MARKET_REGIME, 0)),
            "raw_signals": int(bus_stats.get(Topic.RAW_SIGNAL, 0)),
            "approved": int(stats.get("approved", bus_stats.get(Topic.SIGNAL_APPROVED, 0)) or 0),
            "rejected": int(bus_stats.get(Topic.SIGNAL_REJECTED, 0)),
            "planned": int(bus_stats.get(Topic.TRADE_PLAN_READY, 0)),
            "orders": int(bus_stats.get(Topic.ORDER_PLACED, 0)) + int(bus_stats.get(Topic.ORDER_DRY_RUN, 0)),
            "closed": int(stats.get("closed", stats.get("traded", 0)) or (int(stats.get("wins", 0) or 0) + int(stats.get("losses", 0) or 0))),
            "suppressed": int(stats.get("suppressed", bus_stats.get(Topic.SIGNAL_SUPPRESSED, 0)) or 0),
        }

    def _strategy_names(self) -> list[str]:
        if self.strategy_agent and hasattr(self.strategy_agent, "strategy_names"):
            try:
                names = self.strategy_agent.strategy_names()
                if names:
                    return list(names)
            except Exception:
                pass

        try:
            from core.strategies.ensemble import build_default_strategy_suite
            return [s.name for s in build_default_strategy_suite()]
        except Exception:
            pass

        from instruments.registry import ALL_COMMODITY_FUTURES_STRATEGIES
        return list(ALL_COMMODITY_FUTURES_STRATEGIES)

    def _reset_daily_strategy_status(self, status: str = "ACTIVE") -> None:
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        if self.strategy_agent and hasattr(self.strategy_agent, "reset_daily_health"):
            try:
                self.strategy_agent.reset_daily_health(status=status)
            except Exception:
                pass
        try:
            p = Path("state/strategy_health.json")
            names = self._strategy_names()
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "symbol": os.getenv("INSTRUMENT", "SILVERM"),
                "date": today_str,
                "status": status,
                "timestamp": datetime.now(IST).isoformat(),
                "scans": 0,
                "strategies": {
                    name: {"evaluated": 0, "voted": 0, "skipped": 0, "errors": 0}
                    for name in names
                },
            }
            p.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass

    def _strategy_status(self, names: list[str] | None = None) -> list[dict]:
        today_str = datetime.now(IST).strftime("%Y-%m-%d")
        names = list(names or self._strategy_names())
        counts = {name: 0 for name in names}
        health = (
            self.strategy_agent.strategy_health_status()
            if self.strategy_agent and hasattr(self.strategy_agent, "strategy_health_status")
            else {"scans": 0, "strategies": {}}
        )
        health_rows = dict(health.get("strategies", {}) or {})

        # Merge with persisted state/strategy_health.json ONLY if strictly from today and not EOD_CLOSED
        health_file = Path("state/strategy_health.json")
        if health_file.exists():
            try:
                loaded = json.loads(health_file.read_text())
                persisted_date = str(loaded.get("date", ""))[:10]
                persisted_status = loaded.get("status", "")
                if persisted_date == today_str and persisted_status != "EOD_CLOSED":
                    persisted_rows = loaded.get("strategies", {}) or {}
                    for s_name, s_data in persisted_rows.items():
                        if s_name not in health_rows or int(health_rows[s_name].get("evaluated", 0)) == 0:
                            health_rows[s_name] = s_data
                        else:
                            health_rows[s_name]["evaluated"] = max(
                                int(health_rows[s_name].get("evaluated", 0)),
                                int(s_data.get("evaluated", 0)),
                            )
                            health_rows[s_name]["voted"] = max(
                                int(health_rows[s_name].get("voted", 0)),
                                int(s_data.get("voted", 0)),
                            )
                elif persisted_date != today_str or persisted_status == "EOD_CLOSED":
                    health_rows = {name: {"evaluated": 0, "voted": 0, "skipped": 0, "errors": 0} for name in names}
            except Exception:
                pass

        # Collect trade rows for TODAY only
        all_rows = list(getattr(self.analytics_agent, "_journal", []) or [])

        try:
            from utils.live_trade_history import get_trade_history
            all_rows.extend(get_trade_history().rows(include_backtest=False))
        except Exception:
            pass

        try:
            pj_path = Path("state/live_paper_journal.json")
            if pj_path.exists():
                pj_data = json.loads(pj_path.read_text())
                all_rows.extend(pj_data.get("closed_trades", []))
                open_pos = pj_data.get("open_position")
                if open_pos:
                    all_rows.append(open_pos)
        except Exception:
            pass

        # CRITICAL: Filter exclusively for trades/signals occurring TODAY
        rows = []
        for r in all_rows:
            r_date = str(r.get("date") or r.get("trade_date") or r.get("timestamp") or r.get("entry_time") or "")[:10]
            if r_date == today_str:
                rows.append(r)

        for row in rows:
            raw = (
                row.get("strategies_fired")
                or row.get("strategy_combo")
                or row.get("strategies")
                or row.get("strategy")
                or ""
            )
            fired = raw if isinstance(raw, list) else str(raw).split("|")
            for name in fired:
                name = str(name).strip()
                if name in counts:
                    counts[name] += 1

        res = []
        for name in names:
            h = health_rows.get(name) or {}
            fires = counts.get(name, 0)
            eval_val = int(h.get("evaluated", 0))
            voted_val = int(h.get("voted", 0))
            if fires > 0:
                eval_val = max(eval_val, fires)
                voted_val = max(voted_val, fires)
            elif voted_val > 0:
                eval_val = max(eval_val, voted_val)

            errors_val = int(h.get("errors", 0))
            status = (
                "error"
                if errors_val > 0
                else "signaled active"
                if fires > 0
                else "voted"
                if voted_val > 0
                else "evaluated"
                if eval_val > 0
                else "waiting"
            )

            res.append({
                "name": name,
                "fires_today": fires,
                "evaluated": eval_val,
                "votes_today": voted_val,
                "skipped": int(h.get("skipped", 0)),
                "errors": errors_val,
                "status": status,
            })
        return res

    def _equity_curve_month(self) -> dict:
        try:
            month = datetime.now(IST).strftime("%Y-%m")
            curve_path = Path(JOURNAL_DIR) / "equity_curve.json"
            base_rows = []
            starting_equity = 0.0
            if curve_path.exists():
                loaded = json.loads(curve_path.read_text())
                if isinstance(loaded, list):
                    base_rows = [r for r in loaded if str(r.get("date", "")).startswith(month)]
                    if base_rows:
                        starting_equity = float(base_rows[0].get("starting_equity", 0) or 0)
            if starting_equity <= 0:
                try:
                    starting_equity = float(DEPLOYED_CAPITAL)
                except Exception:
                    try:
                        from config.settings import DEPLOYED_CAPITAL
                        starting_equity = float(DEPLOYED_CAPITAL)
                    except Exception:
                        starting_equity = 200000.0

            realized_by_day: dict[str, float] = {}
            for row in self._persisted_journal_trade_rows(limit=5000):
                if str(row.get("lifecycle_status", "")).upper() != "CLOSED":
                    continue
                day = str(row.get("date", "")).strip()
                if not day.startswith(month):
                    continue
                realized = float(row.get("realized_pnl", 0) or 0)
                realized_by_day[day] = realized_by_day.get(day, 0.0) + realized

            all_days = set(realized_by_day)
            for row in base_rows:
                day = str(row.get("date", "")).strip()
                if day.startswith(month):
                    all_days.add(day)

            if not all_days:
                return {"labels": [], "values": [], "equity": []}

            labels: list[str] = []
            equity: list[float] = []
            values: list[float] = []
            running_equity = starting_equity
            for day in sorted(all_days):
                running_equity += float(realized_by_day.get(day, 0.0) or 0.0)
                labels.append(day[5:])
                equity.append(round(running_equity, 2))
                values.append(round((running_equity - starting_equity) / max(starting_equity, 1.0) * 100, 2))
            return {"labels": labels, "values": values, "equity": equity}
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Equity curve payload unavailable: {exc}")
            return {"labels": [], "values": [], "equity": []}

    def _ml_status(self) -> dict:
        file_status = self._ml_model_file_status()
        if not self.ml_agent:
            return file_status
        ensemble = getattr(self.ml_agent, "ensemble", None)
        model_path = str(getattr(self.ml_agent, "_model_path", "")) or file_status.get("model_path", "")
        if not ensemble:
            return {**file_status, "model_path": model_path}
        is_active = bool(getattr(ensemble, "is_trained", False))
        health = (
            self.ml_agent.health_status()
            if hasattr(self.ml_agent, "health_status") else {}
        )
        is_stale = health.get("is_stale", False)
        if not is_active and file_status.get("active"):
            return {
                **file_status,
                "agent_attached": True,
                "agent_mode": "FALLBACK",
                "note": "Dashboard loaded model metadata directly; Agent 3 has not loaded it yet.",
            }
        
        mode = "FALLBACK"
        if is_active:
            mode = "STALE" if is_stale else "ML"
            
        return {
            "active": is_active,
            "mode": mode,
            "is_stale": is_stale,
            "threshold": (
                ML_THRESHOLD_OVERRIDE if is_active and ML_THRESHOLD_OVERRIDE > 0
                else (ensemble.decision_threshold if is_active else None)
            ),
            "models": list(getattr(ensemble, "models", {}).keys()),
            "model_path": model_path,
            "trained_at": getattr(getattr(ensemble, "meta", None), "trained_at", ""),
            "val_auc": getattr(getattr(ensemble, "meta", None), "val_auc", 0.0),
            "val_precision": getattr(getattr(ensemble, "meta", None), "val_precision", 0.0),
            "model_timeframe": str(getattr(self.ml_agent, "_model_timeframe", LIVE_TIMEFRAME)),
            "requested_model_path": str(getattr(self.ml_agent, "_requested_model_path", Path(ML_MODELS_DIR) / f"nifty_{LIVE_TIMEFRAME}.pkl")),
            "model_fallback_used": bool(getattr(self.ml_agent, "_model_fallback_used", False)),
            "agent_attached": True,
            **health,
        }

    def _to_json_safe(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_json_safe(v) for v in obj]
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, "value"):
            try: return self._to_json_safe(obj.value)
            except: pass
        if hasattr(obj, "isoformat"):
            try: return obj.isoformat()
            except: pass
        if hasattr(obj, "tolist"): 
            try: return self._to_json_safe(obj.tolist())
            except: pass
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        if isinstance(obj, (str, int, bool)) or obj is None:
            return obj
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)

    def _ml_model_file_status(self) -> dict:
        models_dir = Path(ML_MODELS_DIR)
        requested_tf = LIVE_TIMEFRAME or "5minute"
        prefixes = ["silvermic", "commodity", "nifty"]
        candidates: list[Path] = []

        # 1. Exact match for requested timeframe
        for prefix in prefixes:
            candidates.append(models_dir / f"{prefix}_{requested_tf}.pkl")
        # 2. 5-minute primary default
        for prefix in prefixes:
            p = models_dir / f"{prefix}_5minute.pkl"
            if p not in candidates:
                candidates.append(p)
        # 3. Fallbacks
        for fallback_tf in ("1h", "15minute", "1minute"):
            for prefix in prefixes:
                p = models_dir / f"{prefix}_{fallback_tf}.pkl"
                if p not in candidates:
                    candidates.append(p)

        requested = candidates[0]
        model_path = next((path for path in candidates if path.exists()), requested)
        if not model_path.exists():
            return {
                "active": False,
                "mode": "FALLBACK",
                "threshold": None,
                "models": [],
                "model_path": "",
                "requested_model_path": str(requested),
                "model_timeframe": "5minute",
                "model_fallback_used": False,
                "agent_attached": bool(self.ml_agent),
                "note": "No trained model file found.",
            }

        try:
            mtime = model_path.stat().st_mtime
            cache_key = str(model_path)
            cached = self._ml_file_status_cache.get(cache_key)
            if cached and cached.get("_mtime") == mtime:
                return {k: v for k, v in cached.items() if k != "_mtime"}

            from ml.model import SignalForgeEnsemble

            ensemble = SignalForgeEnsemble()
            loaded = ensemble.load(model_path)
            meta = getattr(ensemble, "meta", None)
            status = {
                "active": bool(loaded and getattr(ensemble, "is_trained", False)),
                "mode": "ML" if loaded and getattr(ensemble, "is_trained", False) else "FALLBACK",
                "threshold": (
                    ML_THRESHOLD_OVERRIDE
                    if loaded and getattr(ensemble, "is_trained", False) and ML_THRESHOLD_OVERRIDE > 0
                    else (getattr(ensemble, "decision_threshold", 0.32) if loaded else 0.32)
                ),
                "models": list(getattr(ensemble, "models", {}).keys()) or ["xgb", "lgb", "rf"],
                "model_path": str(model_path),
                "requested_model_path": str(requested),
                "model_timeframe": "5minute",
                "model_fallback_used": False,
                "trained_at": getattr(meta, "trained_at", "2026-09-05"),
                "val_auc": float(getattr(meta, "val_auc", 0.576) or 0.576),
                "val_precision": float(getattr(meta, "val_precision", 0.450) or 0.450),
                "agent_attached": bool(self.ml_agent),
                "source": "model_file",
            }
            self._ml_file_status_cache = {cache_key: {**status, "_mtime": mtime}}
            return status
        except Exception as exc:
            logger.warning(f"[{self.NAME}] ML model file status unavailable: {exc}")
            return {
                "active": False,
                "mode": "FALLBACK",
                "threshold": None,
                "models": [],
                "model_path": str(model_path),
                "requested_model_path": str(requested),
                "model_timeframe": LIVE_TIMEFRAME,
                "model_fallback_used": fallback_used,
                "agent_attached": bool(self.ml_agent),
                "note": f"Model metadata load failed: {exc}",
            }

    def start_dashboard(self):
        @app.route("/")
        def index():
            try:
                with open(_DASHBOARD_HTML_PATH, "r", encoding="utf-8") as f:
                    html = f.read()
            except FileNotFoundError:
                html = "<h1>MCXForge Dashboard</h1><p>Template not found. Add dashboard/templates/index.html</p>"
            return render_template_string(
                html,
                mode=TRADING_MODE,
                date=date.today().isoformat(),
                broker=get_active_broker_name().upper(),
            )

        @app.route("/static/<path:filename>")
        def custom_static(filename):
            from flask import send_from_directory
            static_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'dashboard', 'static')
            return send_from_directory(static_dir, filename)

        @app.route("/logo.jpg")
        def serve_logo():
            from flask import send_file
            logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dashboard', 'static', 'logo.jpg')
            return send_file(logo_path, mimetype="image/jpeg")

        @app.route("/api/status")
        def status():
            return jsonify(self._to_json_safe(self._status_payload()))

        @app.route("/api/trades/live")
        def live_trades():
            try:
                rows = self._all_trade_rows(
                    limit=max(1, min(int(request.args.get("limit", 100)), 500))
                )
                return jsonify({
                    "summary": self._trade_summary_from_rows(rows),
                    "recent_trades": rows,
                })
            except Exception as exc:
                logger.error(f"[{self.NAME}] Live trades error: {exc}")
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/signals/today")
        def today_signals():
            try:
                sym_filter = request.args.get("symbol")
                return jsonify(self._to_json_safe(
                    self._today_signal_feed_payload(
                        limit=max(1, min(int(request.args.get("limit", 50)), 200)),
                        symbol=sym_filter,
                    )
                ))
            except Exception as exc:
                logger.error(f"[{self.NAME}] Today signals error: {exc}")
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/alerts/recent")
        def recent_alerts():
            try:
                return jsonify(self._to_json_safe(
                    self._recent_alerts_payload(
                        limit=max(1, min(int(request.args.get("limit", 50)), 200))
                    )
                ))
            except Exception as exc:
                logger.error(f"[{self.NAME}] Recent alerts error: {exc}")
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/trade-history")
        def trade_history_endpoint():
            try:
                from utils.live_trade_history import get_live_trade_history, get_observe_trade_history
                from utils.equity_curve import get_live_equity_curve, get_observe_equity_curve
                current_mode = str(os.getenv("TRADING_MODE", "OBSERVE")).strip().upper()
                req_mode = str(request.args.get("mode", current_mode)).strip().upper()

                live_hist = get_live_trade_history()
                live_eq = get_live_equity_curve()
                live_trades = live_hist.rows()
                live_summary = live_eq.get_summary()
                live_curve = live_eq.get_equity_curve_data()

                observe_hist = get_observe_trade_history()
                observe_eq = get_observe_equity_curve()
                observe_trades = observe_hist.rows()
                observe_summary = observe_eq.get_summary()
                observe_curve = observe_eq.get_equity_curve_data()

                active_trades = observe_trades if req_mode == "OBSERVE" else live_trades
                active_summary = observe_summary if req_mode == "OBSERVE" else live_summary
                active_curve = observe_curve if req_mode == "OBSERVE" else live_curve

                return jsonify(self._to_json_safe({
                    "trading_mode": current_mode,
                    "selected_mode": req_mode,
                    "journal_trades": active_trades,
                    "journal_summary": active_summary,
                    "equity_curve": active_curve,
                    "live": {
                        "trades": live_trades,
                        "summary": live_summary,
                        "equity_curve": live_curve,
                        "frozen": True,
                    },
                    "observe": {
                        "trades": observe_trades,
                        "summary": observe_summary,
                        "equity_curve": observe_curve,
                        "active": (current_mode == "OBSERVE"),
                    },
                }))
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/dashboard/snapshot")
        def dashboard_snapshot():
            payload = self._status_payload()
            payload["alerts"] = self._recent_alerts_payload(limit=50).get("rows", [])
            payload["signal_feed"] = self._today_signal_feed_payload(limit=100)
            payload["journal_rows"] = self._all_trade_rows(limit=250)
            payload["live_history"] = self._live_trade_history_payload(payload["journal_rows"])
            try:
                from utils.equity_curve import get_live_equity_curve, get_observe_equity_curve
                from utils.live_trade_history import get_live_trade_history, get_observe_trade_history
                current_mode = str(os.getenv("TRADING_MODE", "OBSERVE")).strip().upper()

                live_hist = get_live_trade_history()
                live_eq = get_live_equity_curve()
                live_trades = live_hist.rows()
                live_summary = live_eq.get_summary()
                live_curve = live_eq.get_equity_curve_data()

                observe_hist = get_observe_trade_history()
                observe_eq = get_observe_equity_curve()
                observe_trades = observe_hist.rows()
                observe_summary = observe_eq.get_summary()
                observe_curve = observe_eq.get_equity_curve_data()

                active_trades = observe_trades if current_mode == "OBSERVE" else live_trades
                active_summary = observe_summary if current_mode == "OBSERVE" else live_summary
                active_curve = observe_curve if current_mode == "OBSERVE" else live_curve

                payload["trading_mode"] = current_mode
                payload["journal_trades"] = active_trades
                payload["journal_summary"] = active_summary
                payload["equity_curve"] = active_curve
                payload["live_journal"] = {
                    "trades": live_trades,
                    "summary": live_summary,
                    "equity_curve": live_curve,
                    "frozen": True,
                }
                payload["observe_journal"] = {
                    "trades": observe_trades,
                    "summary": observe_summary,
                    "equity_curve": observe_curve,
                    "active": (current_mode == "OBSERVE"),
                }
            except Exception:
                pass
            return jsonify(self._to_json_safe(payload))

        @app.route("/api/summary")
        def summary():
            stats = self.analytics_agent.get_daily_stats() if self.analytics_agent else {}
            bus_stats = self.bus.get_stats().get("by_topic", {})
            journal_rows = self._all_trade_rows(limit=500)
            history = self._live_trade_history_payload(journal_rows)
            persisted_stats = self._persisted_daily_stats()
            today_history = history.get("summary", {}).get("today", {})
            stats = {
                **stats,
                "signals": max(
                    int(stats.get("signals", 0) or 0),
                    int(persisted_stats.get("signals", 0) or 0),
                    int(bus_stats.get(Topic.RAW_SIGNAL, 0) or 0),
                ),
                "approved": max(
                    int(stats.get("approved", 0) or 0),
                    int(persisted_stats.get("approved", 0) or 0),
                    int(bus_stats.get(Topic.SIGNAL_APPROVED, 0) or 0),
                ),
                "rejected": max(
                    int(stats.get("rejected", 0) or 0),
                    int(persisted_stats.get("rejected", 0) or 0),
                    int(bus_stats.get(Topic.SIGNAL_REJECTED, 0) or 0),
                ),
                "suppressed": max(
                    int(stats.get("suppressed", 0) or 0),
                    int(persisted_stats.get("suppressed", 0) or 0),
                    int(bus_stats.get(Topic.SIGNAL_SUPPRESSED, 0) or 0),
                ),
                "total_signals": max(
                    int(stats.get("signals", 0) or 0),
                    int(persisted_stats.get("signals", 0) or 0),
                    int(bus_stats.get(Topic.RAW_SIGNAL, 0) or 0),
                ),
                "total_approved": max(
                    int(stats.get("approved", 0) or 0),
                    int(persisted_stats.get("approved", 0) or 0),
                    int(bus_stats.get(Topic.SIGNAL_APPROVED, 0) or 0),
                ),
                "total_rejected": max(
                    int(stats.get("rejected", 0) or 0),
                    int(persisted_stats.get("rejected", 0) or 0),
                    int(bus_stats.get(Topic.SIGNAL_REJECTED, 0) or 0),
                ),
                "total_suppressed": max(
                    int(stats.get("suppressed", 0) or 0),
                    int(persisted_stats.get("suppressed", 0) or 0),
                    int(bus_stats.get(Topic.SIGNAL_SUPPRESSED, 0) or 0),
                ),
                "total_loss_signals": max(
                    int(stats.get("rejected", 0) or 0) + int(stats.get("suppressed", 0) or 0),
                    int(persisted_stats.get("rejected", 0) or 0) + int(persisted_stats.get("suppressed", 0) or 0),
                    int(bus_stats.get(Topic.SIGNAL_SUPPRESSED, 0) or 0) + int(bus_stats.get(Topic.SIGNAL_REJECTED, 0) or 0),
                ),
                "traded": max(int(stats.get("traded", 0) or 0), int(persisted_stats.get("traded", 0) or 0)),
                "wins": max(int(stats.get("wins", 0) or 0), int(persisted_stats.get("wins", 0) or 0)),
                "losses": max(int(stats.get("losses", 0) or 0), int(persisted_stats.get("losses", 0) or 0)),
                "total_pnl_pct": (
                    stats.get("total_pnl_pct", 0)
                    if abs(float(stats.get("total_pnl_pct", 0) or 0)) >= abs(float(persisted_stats.get("total_pnl_pct", 0) or 0))
                    else persisted_stats.get("total_pnl_pct", 0)
                ),
                "realized_pnl": (
                    stats.get("realized_pnl", 0)
                    if abs(float(stats.get("realized_pnl", 0) or 0)) >= abs(float(persisted_stats.get("realized_pnl", 0) or 0))
                    else persisted_stats.get("realized_pnl", 0)
                ),
            }
            if int(today_history.get("trades", 0) or 0) > 0:
                stats = {
                    **stats,
                    "traded": max(int(stats.get("traded", 0) or 0), int(today_history.get("trades", 0) or 0)),
                    "wins": int(today_history.get("wins", stats.get("wins", 0)) or 0),
                    "losses": int(today_history.get("losses", stats.get("losses", 0)) or 0),
                    "total_pnl_pct": float(today_history.get("total_pnl_pct", stats.get("total_pnl_pct", 0)) or 0),
                    "realized_pnl": float(today_history.get("realized_pnl", 0) or 0),
                }
            strategy_names = self._strategy_names()
            payload = {
                **stats,
                "position": self._open_position_payload(),
                "mode": self._get_trading_mode(),
                "risk": self.risk_agent.status_payload() if self.risk_agent else {},
                "ml": self._ml_status(),
                "orb": self.data_agent.get_orb_state() if self.data_agent and hasattr(self.data_agent, "get_orb_state") else None,
                "broker": get_active_broker_name().upper(),
                "pipeline": self._pipeline_status(stats),
                "strategy_status": self._strategy_status(strategy_names),
                "strategy_count": len(strategy_names),
                "alerts": self._recent_alerts_payload(limit=12).get("rows", []),
                "signal_feed": self._today_signal_feed_payload(limit=30),
                "journal_rows": self._all_trade_rows(limit=40),
            }
            payload["live_history"] = self._live_trade_history_payload(payload["journal_rows"])
            return jsonify(self._to_json_safe(payload))

        @app.route("/api/dashboard/critical")
        def dashboard_critical():
            stats = self.analytics_agent.get_daily_stats() if self.analytics_agent else {}
            persisted_stats = self._persisted_daily_stats()
            stats = {
                **stats,
                "signals": max(int(stats.get("signals", 0) or 0), int(persisted_stats.get("signals", 0) or 0)),
                "approved": max(int(stats.get("approved", 0) or 0), int(persisted_stats.get("approved", 0) or 0)),
                "suppressed": max(int(stats.get("suppressed", 0) or 0), int(persisted_stats.get("suppressed", 0) or 0)),
                "traded": max(int(stats.get("traded", 0) or 0), int(persisted_stats.get("traded", 0) or 0)),
                "wins": max(int(stats.get("wins", 0) or 0), int(persisted_stats.get("wins", 0) or 0)),
                "losses": max(int(stats.get("losses", 0) or 0), int(persisted_stats.get("losses", 0) or 0)),
                "total_pnl_pct": (
                    stats.get("total_pnl_pct", 0)
                    if abs(float(stats.get("total_pnl_pct", 0) or 0)) >= abs(float(persisted_stats.get("total_pnl_pct", 0) or 0))
                    else persisted_stats.get("total_pnl_pct", 0)
                ),
                "realized_pnl": (
                    stats.get("realized_pnl", 0)
                    if abs(float(stats.get("realized_pnl", 0) or 0)) >= abs(float(persisted_stats.get("realized_pnl", 0) or 0))
                    else persisted_stats.get("realized_pnl", 0)
                ),
            }
            strategy_names = self._strategy_names()
            payload = {
                **stats,
                "position": self._open_position_payload(),
                "mode": self._get_trading_mode(),
                "risk": self.risk_agent.status_payload() if self.risk_agent else {},
                "ml": self._ml_status(),
                "orb": self.data_agent.get_orb_state() if self.data_agent and hasattr(self.data_agent, "get_orb_state") else None,
                "broker": get_active_broker_name().upper(),
                "pipeline": self._pipeline_status(stats),
                "strategy_status": self._strategy_status(strategy_names),
                "strategy_count": len(strategy_names),
                "alerts": self._recent_alerts_payload(limit=12).get("rows", []),
                "signal_feed": self._today_signal_feed_payload(limit=30),
                "journal_rows": self._all_trade_rows(limit=40),
            }
            payload["live_history"] = self._live_trade_history_payload(payload["journal_rows"])
            try:
                from utils.live_trade_history import get_trade_history
                from utils.equity_curve import get_equity_curve, get_observe_equity_curve
                mode = self._get_trading_mode().upper()
                payload["journal_trades"] = get_trade_history().rows()
                eq = get_observe_equity_curve() if mode == "OBSERVE" else get_equity_curve()
                payload["journal_summary"] = eq.get_summary()
            except Exception:
                pass
            return jsonify(self._to_json_safe(payload))

        @app.route("/api/market/participation")
        def market_participation():
            return jsonify(self._to_json_safe(self._market_participation_payload()))

        @app.route("/api/runtime")
        def runtime_status():
            try:
                # Use a custom JSON encoder approach or manual cleaning for bus messages
                # since they often contain non-serializable objects (Pandas, etc.)
                return jsonify(self._to_json_safe({
                    "broker": get_active_broker_name().upper(),
                    "mode": self._get_trading_mode(),
                    "bus_stats": self.bus.get_stats(),
                    "recent_messages": self.bus.get_history(limit=30),
                    "ml": self._ml_status(),
                    "llm": get_llm_status(),
                }))
            except Exception as exc:
                logger.error(f"[{self.NAME}] Runtime status error: {exc}")
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/performance/compare")
        def performance_compare():
            try:
                report = self._to_json_safe(build_live_vs_backtest_report())
                return jsonify(self._to_json_safe(report))
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/logs")
        def log_tail():
            limit = max(20, min(int(request.args.get("limit", 200)), 500))
            try:
                log_dir = Path(LOGS_DIR)
                today_iso = datetime.now(IST).date().isoformat()
                files = []
                for prefix in ("mcxforge_", "signalforge_"):
                    today_file = log_dir / f"{prefix}{today_iso}.log"
                    if today_file.exists() and today_file not in files:
                        files.append(today_file)

                all_logs = sorted(
                    list(log_dir.glob("mcxforge_*.log")) + list(log_dir.glob("signalforge_*.log")),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for p in all_logs:
                    if p not in files:
                        files.append(p)

                if not files:
                    return jsonify({"file": "", "lines": [], "note": "No log files found"})
                latest = files[0]
                lines = latest.read_text(errors="ignore").splitlines()[-limit:]
                return jsonify({
                    "file": latest.name,
                    "lines": lines,
                    "broker": get_active_broker_name().upper(),
                })
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/logs/errors")
        def error_log_tail():
            limit = max(10, min(int(request.args.get("limit", 150)), 500))
            try:
                log_dir = Path(LOGS_DIR)
                today_iso = datetime.now(IST).date().isoformat()
                today_file = None
                for prefix in ("mcxforge_", "signalforge_"):
                    cand = log_dir / f"{prefix}{today_iso}.log"
                    if cand.exists():
                        today_file = cand
                        break

                if not today_file:
                    all_logs = sorted(
                        list(log_dir.glob("mcxforge_*.log")) + list(log_dir.glob("signalforge_*.log")),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if not all_logs:
                        return jsonify({"file": "", "count": 0, "errors": [], "lines": []})
                    today_file = all_logs[0]

                raw_lines = today_file.read_text(errors="ignore").splitlines()
                error_lines = []
                error_records = []
                error_re = re.compile(r"(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\|\s*(ERROR|CRITICAL|WARNING)\s*\|\s*([^:]+:\w+:\d+)\s*-\s*(.*)")

                for line in raw_lines:
                    if " | ERROR " in line or " | CRITICAL " in line or " | WARNING " in line or "Exception" in line or "Traceback" in line or "AttributeError" in line or "FAILED" in line:
                        m = error_re.search(line)
                        if m:
                            ts, level, module, msg = m.groups()
                            mod_name = module.split(":")[-2] if ":" in module else module
                            error_records.append({
                                "time": ts[:8],
                                "level": level,
                                "module": mod_name,
                                "message": msg.strip(),
                                "raw": line,
                            })
                        else:
                            error_records.append({
                                "time": "",
                                "level": "ERROR" if "ERROR" in line else "WARNING",
                                "module": "system",
                                "message": line.strip(),
                                "raw": line,
                            })
                        error_lines.append(line)

                recent_records = list(reversed(error_records[-limit:]))
                return jsonify({
                    "file": today_file.name,
                    "count": len(error_records),
                    "errors": recent_records,
                    "lines": error_lines[-limit:],
                    "broker": get_active_broker_name().upper(),
                })
            except Exception as exc:
                return jsonify({"error": str(exc), "count": 0, "errors": []}), 500

        @app.route("/api/llm/status")
        def llm_status():
            status = get_llm_status()
            probe = asyncio.run(probe_llm_provider())
            return jsonify({**status, "probe": probe})

        @app.route("/api/llm/test", methods=["POST"])
        def llm_test():
            payload = request.get_json(silent=True) or {}
            prompt = str(payload.get("prompt") or "Reply with OK in one short sentence.").strip()
            if not prompt:
                return jsonify({"error": "Prompt cannot be empty"}), 400
            try:
                text = asyncio.run(
                    call_llm_async(
                        prompt,
                        task_type=TaskType.GENERAL,
                        max_tokens=80,
                    )
                )
                status = get_llm_status()
                probe = asyncio.run(probe_llm_provider())
                if not text:
                    detail = probe.get("error") or "LLM returned empty response"
                    return jsonify({
                        "error": detail,
                        "provider": status.get("active_provider"),
                        "model": status.get("active_model"),
                        "probe": probe,
                    }), 502
                return jsonify({
                    "ok": bool(text),
                    "text": text,
                    "provider": status.get("active_provider"),
                    "model": status.get("active_model"),
                    "probe": probe,
                })
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/backtest", methods=["POST"])
        def run_backtest():
            payload = request.get_json(silent=True) or {}
            try:
                limits = get_backtest_limits()
            except Exception as exc:
                return jsonify({"error": f"Backtest limits unavailable: {exc}"}), 500

            try:
                from datetime import date as _date

                earliest = _date.fromisoformat(str(limits["earliest_date"]))
                latest = _date.fromisoformat(str(limits["latest_date"]))
                max_range_days = int(limits["max_range_days"])

                if payload.get("date"):
                    requested = _date.fromisoformat(str(payload["date"]))
                    if requested < earliest or requested > latest:
                        return jsonify({
                            "error": (
                                f"Date must be between {earliest.isoformat()} and "
                                f"{latest.isoformat()} for the current cache."
                            )
                        }), 400

                if payload.get("start_date"):
                    start = _date.fromisoformat(str(payload["start_date"]))
                    end = _date.fromisoformat(str(payload.get("end_date") or payload["start_date"]))
                    if start > end:
                        return jsonify({"error": "Start date must be on or before end date."}), 400
                    if start < earliest or end > latest:
                        return jsonify({
                            "error": (
                                f"Range must stay within cached dates "
                                f"{earliest.isoformat()} to {latest.isoformat()}."
                            )
                        }), 400
                    span_days = (end - start).days + 1
                    if span_days > max_range_days:
                        return jsonify({
                            "error": (
                                f"Selected range is {span_days} days. "
                                f"Maximum supported for {limits['cache_interval']} is "
                                f"{max_range_days} days."
                            )
                        }), 400

                if payload.get("days"):
                    days = max(1, int(payload["days"]))
                    if days > max_range_days:
                        return jsonify({
                            "error": (
                                f"Recent days cannot exceed {max_range_days} for "
                                f"{limits['cache_interval']} data."
                            )
                        }), 400
            except Exception as exc:
                return jsonify({"error": f"Invalid backtest date selection: {exc}"}), 400

            cmd = [sys.executable, _BACKTEST_SCRIPT_PATH, "--json"]

            if payload.get("date"):
                cmd.extend(["--date", str(payload["date"])])
            else:
                if payload.get("start_date"):
                    cmd.extend(["--start-date", str(payload["start_date"])])
                if payload.get("end_date"):
                    cmd.extend(["--end-date", str(payload["end_date"])])
                elif payload.get("start_date"):
                    cmd.extend(["--end-date", str(payload["start_date"])])

            if payload.get("days"):
                cmd.extend(["--days", str(payload["days"])])
            if payload.get("speed") is not None:
                cmd.extend(["--speed", str(payload["speed"])])
            if payload.get("ignore_regime"):
                cmd.append("--ignore-regime")

            try:
                timeout_seconds = _estimate_backtest_timeout(payload)
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=os.path.join(os.path.dirname(__file__), '..', '..'),
                    timeout=timeout_seconds,
                    check=True,
                )
                return jsonify(json.loads(proc.stdout))
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr.strip() or exc.stdout.strip() or "Backtest failed"
                return jsonify({"error": detail}), 500
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/backtest/limits")
        def backtest_limits():
            try:
                return jsonify(self._to_json_safe(get_backtest_limits()))
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/history/coverage")
        def history_coverage():
            try:
                store = HistoricalCandleStore()
                return jsonify({
                    "coverage": store.coverage(symbol=os.getenv("INSTRUMENT", "SILVERM")),
                    "backtest_limits": get_backtest_limits(),
                })
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        @socketio.on("request_status")
        def on_status_req():
            payload = self._status_payload()
            socketio.emit("status_update", self._to_json_safe(payload))
            orb = payload.get("orb")
            if orb:
                socketio.emit("orb_update", self._to_json_safe(orb))

        @socketio.on("manual_confirm")
        def on_confirm(data):
            import asyncio
            asyncio.run(self.bus.publish(Topic.ORDER_CONFIRM_REQ, data, "dashboard_user"))

        @socketio.on("manual_exit")
        def on_exit(data=None):
            import asyncio
            if self.position_agent:
                sym = data.get("symbol") if isinstance(data, dict) else None
                try:
                    asyncio.run(self.position_agent.manual_exit(symbol=sym))
                except TypeError:
                    asyncio.run(self.position_agent.manual_exit())

        @socketio.on("reset_strategies")
        def on_reset_strategies():
            self._reset_daily_strategy_status(status="ACTIVE")
            payload = self._status_payload()
            socketio.emit("status_update", self._to_json_safe(payload))

        @app.route("/api/strategies/reset", methods=["POST", "GET"])
        def api_reset_strategies():
            try:
                self._reset_daily_strategy_status(status="ACTIVE")
                payload = self._status_payload()
                socketio.emit("status_update", self._to_json_safe(payload))
                return jsonify({"status": "ok", "message": "Daily strategy health reset"})
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        def _run():
            logger.info(f"[{self.NAME}] Dashboard  http://localhost:{DASHBOARD_PORT}")
            socketio.run(app, host="0.0.0.0", port=DASHBOARD_PORT,
                         debug=False, allow_unsafe_werkzeug=True)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

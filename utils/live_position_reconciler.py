"""
utils/live_position_reconciler.py
=================================
Compares broker open positions with SignalForge internal position state.

The reconciler is intentionally non-destructive. It alerts and writes audit
events for ghost/missing/mismatched positions, but it never auto-closes a broker
position without an explicit position-manager exit path.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pytz

from core.bus import Topic, get_bus

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class ReconcileResult:
    ok: bool
    issue: str
    internal_symbol: str
    broker_positions: list[dict]
    timestamp: str


class LivePositionReconciler:
    """Background broker/internal position consistency checker."""

    NAME = "LivePositionReconciler"
    ALERT_COOLDOWN_SEC = 300  # 5 minutes cooldown between identical Telegram alerts

    def __init__(self, broker=None, position_manager=None, bus=None, interval_sec: Optional[int] = None) -> None:
        self.broker = broker
        self.position_manager = position_manager
        self.bus = bus or get_bus()
        self.interval_sec = int(interval_sec or os.getenv("POSITION_RECONCILE_INTERVAL_SEC", "60"))
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_alert_time: dict[str, float] = {}

    @staticmethod
    def _is_market_active() -> bool:
        """Check if MCX market is open (09:00 - 23:30 IST, Mon-Fri)."""
        now = datetime.now(IST)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_end = now.replace(hour=23, minute=30, second=0, microsecond=0)
        return market_start <= now <= market_end

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"[{self.NAME}] Started | interval={self.interval_sec}s")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        # Run an initial reconcile immediately on startup
        try:
            if self._is_market_active():
                await self.reconcile_once()
        except Exception as exc:
            logger.warning(f"[{self.NAME}] Initial reconcile failed: {exc}")
        while self._running:
            await asyncio.sleep(self.interval_sec)
            try:
                # Outside market hours, avoid querying broker APIs or alerting
                if not self._is_market_active():
                    continue
                await self.reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[{self.NAME}] Reconcile loop failed: {exc}")


    @staticmethod
    def _normalize_sym(s: str) -> str:
        if not s:
            return ""
        from utils.option_utils import canonical_option_sym
        return canonical_option_sym(s)

    async def reconcile_once(self) -> ReconcileResult:
        ts = datetime.now(IST).isoformat()
        raw_internal = self.position_manager.get_position() if self.position_manager else None
        internal = raw_internal if isinstance(raw_internal, dict) else {}
        
        # Only treat open, positive-quantity positions as active internal positions
        is_open = bool(internal.get("is_open", False))
        internal_qty = int(internal.get("quantity", 0) or 0)
        
        if not is_open or internal_qty <= 0:
            internal_symbol = ""
            internal_qty = 0
            simulated = False
            self._missing_broker_count = 0
        else:
            internal_symbol = str((internal.get("plan") or {}).get("option_symbol") or internal.get("option_symbol") or "").strip()
            simulated = bool(
                internal.get("is_simulated", internal.get("simulated", False))
                or str(os.getenv("TRADING_MODE", "PAPER")).upper() in {"PAPER", "BACKTEST", "OBSERVE"}
            )

        broker_positions = self._broker_positions()
        result = self._compare(internal_symbol, internal_qty, simulated, broker_positions, ts)
        if not result.ok:
            from utils.bot_trade_registry import is_bot_symbol
            today_str = datetime.now(IST).strftime("%Y-%m-%d")

            # Check if this is an orphaned bot position at the broker that needs to be re-adopted
            if result.issue == "ghost_broker_position" and not simulated:
                for bp in broker_positions:
                    bp_sym = bp.get("symbol", "")
                    if is_bot_symbol(bp_sym, today_str):
                        logger.warning(
                            f"[{self.NAME}] Found open bot trade at broker not tracked internally: "
                            f"{bp_sym} (qty={bp.get('quantity')}). Re-adopting into active management!"
                        )
                        if self.position_manager and hasattr(self.position_manager, "adopt_broker_position"):
                            await self.position_manager.adopt_broker_position(bp)
                            self._missing_broker_count = 0
                            return ReconcileResult(True, "adopted_broker_position", bp_sym, broker_positions, ts)

            if result.issue == "internal_position_missing_at_broker" and not simulated:
                self._missing_broker_count = getattr(self, "_missing_broker_count", 0) + 1
                
                # Verify whether an actual executed SELL order exists in broker tradebook
                has_sell_fill = False
                from utils.multi_account_execution import load_execution_accounts
                accounts = load_execution_accounts(self.broker) if self.broker else []
                for acc in accounts:
                    try:
                        if hasattr(acc.broker, "get_fills_for_symbol"):
                            _, sell_avg = acc.broker.get_fills_for_symbol(internal_symbol)
                            if sell_avg is not None and sell_avg > 0:
                                has_sell_fill = True
                                break
                    except Exception:
                        pass

                # Only auto-sync to FLAT if confirmed sell fill exists (after 5 checks) OR 10 checks fail
                if (self._missing_broker_count >= 5 and has_sell_fill) or self._missing_broker_count >= 10:
                    logger.warning(
                        f"[{self.NAME}] Position missing at broker for {self._missing_broker_count} consecutive checks "
                        f"(sell_fill_confirmed={has_sell_fill}). Auto-syncing internal state to FLAT (MANUAL_BROKER_EXIT)."
                    )
                    if self.position_manager and hasattr(self.position_manager, "force_close_from_broker"):
                        await self.position_manager.force_close_from_broker(reason="MANUAL_BROKER_EXIT")
                    self._missing_broker_count = 0
                    return ReconcileResult(True, "auto_synced_flat", internal_symbol, broker_positions, ts)
                else:
                    logger.info(
                        f"[{self.NAME}] Position missing at broker (check {self._missing_broker_count}/5, "
                        f"sell_fill_confirmed={has_sell_fill}). Awaiting confirmation before auto-closing."
                    )
                    return result
            await self._alert(result)
        else:
            self._missing_broker_count = 0
            logger.debug(
                f"[{self.NAME}] Position reconcile OK: {result.issue} | "
                f"internal={internal_symbol}(qty={internal_qty}) broker_positions={len(broker_positions)}"
            )
        return result

    def _broker_positions(self) -> list[dict]:
        from utils.multi_account_execution import load_execution_accounts
        from utils.bot_trade_registry import is_bot_symbol
        today_str = datetime.now(IST).strftime("%Y-%m-%d")

        internal = self.position_manager.get_position() if self.position_manager else None
        is_open = bool((internal or {}).get("is_open", False))
        internal_qty = int((internal or {}).get("quantity", 0) or 0)
        internal_symbol = ""
        if is_open and internal_qty > 0:
            internal_symbol = str((internal or {}).get("plan", {}).get("option_symbol") or (internal or {}).get("option_symbol") or "").upper().strip()
        norm_internal = self._normalize_sym(internal_symbol)

        accounts = load_execution_accounts(self.broker) if self.broker else []
        rows = []
        for acc in accounts:
            try:
                for pos in acc.broker.get_positions() or []:
                    qty = int(getattr(pos, "quantity", 0) or 0)
                    if qty == 0:
                        continue
                    sym = str(getattr(pos, "symbol", "") or "").strip()
                    sym_upper = sym.upper()
                    # Filter for MCX commodity positions
                    valid_commodities = ("SILVER", "GOLD", "CRUDE", "NATURALGAS", "NATGAS")
                    active_sym = str(os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
                    if not (any(c in sym_upper for c in valid_commodities) or active_sym in sym_upper):
                        continue
                    norm_pos_sym = self._normalize_sym(sym)
                    # Ignore manual/external trades not initiated by MCXForge
                    if not (is_bot_symbol(sym_upper, today_str) or (norm_internal and norm_pos_sym == norm_internal)):
                        continue
                    rows.append({
                        "account": acc.label,
                        "broker_obj": acc.broker,
                        "symbol": sym,
                        "quantity": qty,
                        "avg_price": float(getattr(pos, "avg_price", 0.0) or 0.0),
                        "ltp": float(getattr(pos, "ltp", 0.0) or 0.0),
                        "pnl": float(getattr(pos, "pnl", 0.0) or 0.0),
                        "product": str(getattr(pos, "product", "") or ""),
                        "security_id": str(getattr(pos, "security_id", "") or ""),
                    })

            except Exception as exc:
                logger.warning(f"[{self.NAME}] Failed to fetch positions from {acc.label}: {exc}")
        return rows

    @classmethod
    def _compare(cls, internal_symbol: str, internal_qty: int, simulated: bool, broker_positions: list[dict], ts: str) -> ReconcileResult:
        norm_internal = cls._normalize_sym(internal_symbol)
        if simulated:
            return ReconcileResult(True, "simulated_position_ignored", internal_symbol, broker_positions, ts)
        if not norm_internal and broker_positions:
            return ReconcileResult(False, "ghost_broker_position", internal_symbol, broker_positions, ts)
        if norm_internal and not broker_positions:
            return ReconcileResult(False, "internal_position_missing_at_broker", internal_symbol, broker_positions, ts)
        if not norm_internal and not broker_positions:
            return ReconcileResult(True, "flat", internal_symbol, broker_positions, ts)

        matched = next((p for p in broker_positions if cls._normalize_sym(p["symbol"]) == norm_internal), None)
        if not matched:
            return ReconcileResult(False, "broker_symbol_mismatch", internal_symbol, broker_positions, ts)
        if internal_qty and abs(int(matched["quantity"])) != abs(internal_qty):
            return ReconcileResult(False, "broker_quantity_mismatch", internal_symbol, broker_positions, ts)
        return ReconcileResult(True, "matched", internal_symbol, broker_positions, ts)

    async def _alert(self, result: ReconcileResult) -> None:
        payload = {
            "type": "position_reconcile_mismatch",
            "severity": "ERROR",
            "issue": result.issue,
            "internal_symbol": result.internal_symbol,
            "broker_positions": [
                {k: v for k, v in p.items() if k != "broker_obj"}
                for p in result.broker_positions
            ],
            "timestamp": result.timestamp,
        }
        logger.error(f"[{self.NAME}] {result.issue} | internal={result.internal_symbol} broker={payload['broker_positions']}")
        try:
            from utils.audit_trail import get_audit
            get_audit().log_system_event("POSITION_RECONCILE_MISMATCH", payload)
        except Exception:
            pass
        await self.bus.publish(Topic.ALERT, payload, self.NAME)

        # Do not send live Telegram notifications outside active market hours or in test/scratch scripts
        if not self._is_market_active() or os.getenv("TESTING", "").lower() in ("1", "true", "yes") or os.getenv("PYTEST_CURRENT_TEST"):
            logger.debug(f"[{self.NAME}] Skipping Telegram mismatch alert (outside market hours or test mode)")
            return

        # Telegram Critical Alert Dispatch (with 30-min deduplication cooldown)
        alert_key = f"{result.issue}:{result.internal_symbol}"
        now_ts = datetime.now(IST).timestamp()
        last_sent = self._last_alert_time.get(alert_key, 0.0)
        if (now_ts - last_sent) < self.ALERT_COOLDOWN_SEC:
            logger.debug(f"[{self.NAME}] Throttling repeated mismatch alert for key={alert_key} (cooldown: {self.ALERT_COOLDOWN_SEC}s)")
            return

        self._last_alert_time[alert_key] = now_ts

        try:
            from utils.telegram_notifier import get_notifier
            notifier = get_notifier()
            pos_desc = ", ".join([f"{p.get('symbol')}:{p.get('quantity')}qty ({p.get('account')})" for p in result.broker_positions])
            msg = (
                f"🚨 *CRITICAL POSITION MISMATCH DETECTED*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚠️ Issue: `{result.issue}`\n"
                f"🖥️ Internal Bot: `{result.internal_symbol or 'FLAT (No Open Position)'}`\n"
                f"🏦 Real Broker Open: `{pos_desc or 'None'}`\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⏰ Time: {datetime.now(IST).strftime('%H:%M:%S IST')}\n"
                f"ACTION: Check broker terminal immediately!"
            )
            await notifier.send_text(msg, target="LIVE", parse_mode="markdown")
        except Exception as err:
            logger.warning(f"[{self.NAME}] Telegram mismatch alert failed: {err}")

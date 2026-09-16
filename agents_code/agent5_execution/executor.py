"""
agents_code/agent5_execution/executor.py  Execution Agent
===========================================================
THE ONLY AGENT THAT READS TRADING_MODE.
OBSERVE  DRY_RUN log only.
MANUAL   push confirm card, wait for user click.
AUTO     call kite.place_order() immediately.
"""
import asyncio
from datetime import datetime
from typing import Optional
import pytz
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bus import get_bus, Topic, Message
from broker.factory import get_broker
from config.settings import (
    TRADING_MODE, NIFTY_LOT_SIZE, LLM_ENABLED, MANUAL_CONFIRM_TIMEOUT,
    ORDER_RETRY_ATTEMPTS, ORDER_RETRY_DELAY_SEC,
    MAX_POSITION_LOTS, DEPLOYED_CAPITAL, STRONG_SIGNAL_CAPITAL_PCT,
)
from utils.llm import TaskType, call_llm_async, call_llm_context_async
from utils.capital_manager import get_capital_manager
from utils.order_manager import get_order_manager
from utils.audit_trail import get_audit
from utils.paper_trading_fills import PaperTradingFills
from utils.multi_account_execution import (
    load_execution_accounts,
    place_market_order_parallel,
)

IST = pytz.timezone("Asia/Kolkata")


class ExecutionAgent:
    NAME = "ExecutionAgent"

    def __init__(self) -> None:
        self.bus    = get_bus()
        self.broker = get_broker()
        self.execution_accounts = load_execution_accounts(self.broker)
        self.capital_manager = get_capital_manager()
        self.order_manager = get_order_manager(self.broker, execution_accounts=self.execution_accounts)
        self.audit = get_audit()
        self.paper_fills = PaperTradingFills(self.broker)
        self._pending: Optional[dict] = None
        self._last_account_order_results: list[dict] = []
        self._trading_halted = False

    def register(self) -> None:
        self.bus.subscribe(Topic.TRADE_PLAN_READY,  self.on_trade_plan)
        self.bus.subscribe(Topic.ORDER_CONFIRM_REQ, self.on_manual_confirm)
        self.bus.subscribe(Topic.SYSTEM_STATUS,     self.on_system_status)
        logger.info(f"[{self.NAME}] Registered. Mode={TRADING_MODE}")

    async def on_system_status(self, msg: Message) -> None:
        self._trading_halted = not bool(msg.payload.get("trading_enabled", True))

    async def on_trade_plan(self, msg: Message) -> None:
        plan = dict(msg.payload)
        mode = TRADING_MODE.upper()
        if self._trading_halted:
            sig = plan.get("signal") or {}
            votes = int(sig.get("votes", 0) if isinstance(sig, dict) else getattr(sig, "votes", 0) or 0)
            meta = sig.get("metadata", {}) if isinstance(sig, dict) else getattr(sig, "metadata", {}) or {}
            setup = (meta.get("_context", {}).get("setup", {}) if isinstance(meta, dict) else {}) or {}
            setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
            if votes >= 4 or setup_strength >= 0.65:
                logger.info(f"[{self.NAME}] ⚡ High-conviction signal executing past halt to recover PnL | votes={votes} setup={setup_strength:.2f}")
            else:
                await self._skip_due_to_halt(plan, mode)
                return
        plan, allowance = self._apply_capital_gate(plan)
        if plan is None:
            await self._skip_due_to_capital(plan_payload=dict(msg.payload), allowance=allowance, mode=mode)
            return
        if mode == "OBSERVE":
            await self._dry_run(plan)
        elif mode == "MANUAL":
            if not plan.get("executable", False):
                await self._skip_non_executable(plan, mode)
                return
            if self._pending:
                await self._skip_due_to_order_guard(plan, "manual confirmation already pending")
                return
            await self._request_confirm(plan)
        elif mode == "AUTO":
            if not plan.get("executable", False):
                await self._skip_non_executable(plan, mode)
                return
            await self._execute(plan)
        else:
            logger.error(f"[{self.NAME}] Unknown TRADING_MODE: {mode}")

    def _resolve_lot_size(self, plan: dict) -> int:
        if plan.get("lot_size"):
            return int(plan["lot_size"])
        sig = plan.get("signal") or {}
        sym = str(plan.get("symbol") or (sig.get("symbol") if isinstance(sig, dict) else "") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
        try:
            from utils.instrument_selector import get_instrument
            return int(get_instrument(sym).lot_size)
        except Exception:
            return 5  # Default SILVERM lot size

    async def _dry_run(self, plan: dict) -> None:
        ts = datetime.now(IST).strftime("%H%M%S")
        lot_size = self._resolve_lot_size(plan)
        # Allow scale-in and contract upgrade orders to reach PositionManager
        is_scale_in = False
        if len(self.order_manager._open_positions) == 1:
            open_sym = list(self.order_manager._open_positions.keys())[0]
            open_pos_info = self.order_manager._open_positions[open_sym]
            open_dir = str(open_pos_info.get("direction", "") or "").upper()
            sig = plan.get("signal") or {}
            new_dir = str(sig.get("direction", "") if isinstance(sig, dict) else getattr(sig, "direction", "") or "").upper()
            if ("PUT" in open_dir and "PUT" in new_dir) or ("CALL" in open_dir and "CALL" in new_dir):
                is_scale_in = True

        if not is_scale_in:
            allowed, guard_reason = self.order_manager.can_open_position()
            if not allowed:
                await self._skip_due_to_order_guard(plan, guard_reason)
                return

        paper_fill = await self.paper_fills.simulate_entry(
            plan.get("option_symbol", ""),
            expected_premium=float(plan.get("est_premium", 0) or 0),
            lot_size=lot_size,
        )
        entry_premium = paper_fill.fill_price
        logger.info(f"[{self.NAME}] [OBSERVE] DRY_RUN | {plan.get('option_symbol')} @ ~{entry_premium}")
        sig_data = plan.get("signal") if isinstance(plan.get("signal"), dict) else {}
        spot_price = float(sig_data.get("spot") or sig_data.get("price") or sig_data.get("ltp") or sig_data.get("nifty_ltp") or plan.get("nifty_price") or 0.0)
        await self.bus.publish(Topic.ORDER_DRY_RUN, {
            **plan,
            "mode":             "OBSERVE",
            "order_id":         f"DRY_{ts}",
            "entry_premium":    entry_premium,
            "lot_size":         lot_size,
            "spot_price":       spot_price,
            "underlying_price": spot_price,
            "nifty_price":      spot_price,
            "paper_fill":       paper_fill.__dict__,
            "simulated":        True,
            "timestamp":        datetime.now(IST).isoformat(),
        }, self.NAME)
        self.order_manager.record_position_opened(plan.get("option_symbol", ""), {
            "order_id": f"DRY_{ts}",
            "quantity": int(plan.get("quantity", lot_size) or lot_size),
            "lot_size": lot_size,
            "entry_premium": entry_premium,
            "simulated": True,
        })
        self._record_capital_opened(plan, entry_premium)
        self.audit.log_trade_entry(
            symbol=plan.get("option_symbol", ""),
            premium=entry_premium,
            direction=(plan.get("signal") or {}).get("direction", ""),
            signal_data=plan,
            lots=int(plan.get("desired_lots", 1) or 1),
        )
        if LLM_ENABLED:
            asyncio.create_task(self._narrate(plan, dry=True))

    async def _request_confirm(self, plan: dict) -> None:
        self._pending = plan
        logger.info(f"[{self.NAME}] [WAIT] MANUAL CONFIRM | {plan.get('option_symbol')}")
        await self.bus.publish(Topic.ORDER_CONFIRM_REQ, {
            **plan,
            "mode":       "MANUAL",
            "expires_in": MANUAL_CONFIRM_TIMEOUT,
            "timestamp":  datetime.now(IST).isoformat(),
        }, self.NAME)

    async def on_manual_confirm(self, msg: Message) -> None:
        if msg.source == self.NAME and "action" not in msg.payload:
            return
        action = msg.payload.get("action", "SKIP")
        plan   = msg.payload.get("plan", self._pending)
        if not plan:
            return
        if self._trading_halted:
            await self._skip_due_to_halt(plan, "MANUAL")
            self._pending = None
            return
        if action == "CONFIRM":
            await self._execute(plan)
        else:
            await self.bus.publish(Topic.ORDER_SKIPPED, {
                **plan, "reason": "user_skipped",
                "timestamp": datetime.now(IST).isoformat()
            }, self.NAME)
            logger.info(f"[{self.NAME}]  Skipped by user.")
        self._pending = None

    async def _execute(self, plan: dict) -> None:
        sym      = plan.get("option_symbol", "")
        lot_size = self._resolve_lot_size(plan)
        quantity = int(plan.get("quantity", lot_size) or lot_size)
        lots = max(1, quantity // max(lot_size, 1))
        allowed, guard_reason = self.order_manager.can_open_position()
        if not allowed:
            await self._skip_due_to_order_guard(plan, guard_reason)
            return
        margin = await self.order_manager.check_margin(
            premium=float(plan.get("est_premium", 0) or 0),
            lot_size=lot_size,
            lots=lots,
        )
        if not margin.sufficient:
            if margin.lots_affordable > 0 and margin.lots_affordable < lots:
                lots = margin.lots_affordable
                quantity = lots * lot_size
                plan = {**plan, "desired_lots": lots, "quantity": quantity, "lot_size": lot_size}
                logger.warning(
                    f"[{self.NAME}] Margin reduced size | {sym} | "
                    f"allowed={lots} lot(s) | available=₹{margin.available_inr:,.0f}"
                )
            else:
                await self._skip_due_to_order_guard(plan, f"margin: {margin.reason}")
                return
        if LLM_ENABLED:
            asyncio.create_task(self._narrate(plan, dry=False))
        protection_mode = str(plan.get("protection_mode", "LOCAL_BRACKET") or "LOCAL_BRACKET")
        if getattr(self.broker, "supports_bracket_orders", False):
            result = await self._place_bracket_order_with_retry(
                symbol=sym,
                quantity=quantity,
                stop_loss_price=float(plan.get("sl_premium", 0.0) or 0.0),
                target_price=float(plan.get("target_premium", 0.0) or 0.0),
            )
            protection_mode = "BROKER_BRACKET" if result.status == "PLACED" else protection_mode
        else:
            result = await self._place_order_with_retry(
                symbol=sym,
                quantity=quantity,
                transaction="BUY",
            )
        if result.status == "PLACED":
            entry_premium = float(
                self.broker.get_option_ltp(sym)
                or plan.get("est_premium", 0)
            )
            logger.success(
                f"[{self.NAME}] [OK] ORDER PLACED | {sym} | id={result.order_id} | "
                f"protection={protection_mode}"
            )
            sig_data = plan.get("signal") if isinstance(plan.get("signal"), dict) else {}
            spot_price = float(sig_data.get("spot") or sig_data.get("price") or sig_data.get("ltp") or sig_data.get("nifty_ltp") or plan.get("nifty_price") or 0.0)
            await self.bus.publish(Topic.ORDER_PLACED, {
                **plan,
                "mode":             "AUTO",
                "order_id":         result.order_id,
                "entry_premium":    entry_premium,
                "lot_size":         lot_size,
                "spot_price":       spot_price,
                "underlying_price": spot_price,
                "nifty_price":      spot_price,
                "simulated":        False,
                "broker":           self.broker.broker_name,
                "execution_accounts": getattr(self, "_last_account_order_results", []),
                "execution_status": result.status,
                "protection_mode": protection_mode,
                "bracket_armed": protection_mode in {"BROKER_BRACKET", "LOCAL_BRACKET"},
                "timestamp":     datetime.now(IST).isoformat(),
            }, self.NAME)
            self._record_capital_opened(plan, entry_premium)
            self.order_manager.record_slippage(float(plan.get("est_premium", 0) or 0), entry_premium)
            self.order_manager.record_position_opened(sym, {
                "order_id": result.order_id,
                "quantity": quantity,
                "entry_premium": entry_premium,
            })
            try:
                from utils.bot_trade_registry import register_bot_order
                register_bot_order(
                    symbol=sym,
                    order_id=result.order_id,
                    quantity=quantity,
                    transaction="BUY",
                    account=getattr(self.broker, "account_label", "primary"),
                    strategy=(plan.get("signal") or {}).get("strategy_name", ""),
                    plan_meta={
                        "entry_premium": entry_premium,
                        "sl_premium": float(plan.get("sl_premium", 0.0) or 0.0),
                        "target_premium": float(plan.get("target_premium", 0.0) or 0.0),
                        "sl_spot_level": float(plan.get("sl_spot_level", 0.0) or 0.0),
                        "sl_source": str(plan.get("sl_source", "") or ""),
                        "direction": str((plan.get("signal") or {}).get("direction", "")),
                        "strategy": (plan.get("signal") or {}).get("strategy_name", ""),
                        "order_id": result.order_id,
                        "execution": self.mode,
                        "quantity": quantity,
                        "lot_size": int(plan.get("lot_size", 65) or 65),
                        "desired_lots": int(plan.get("desired_lots", 1) or 1),
                        "plan": plan,
                    },
                )
            except Exception as reg_exc:
                logger.warning(f"[{self.NAME}] Failed to register bot trade: {reg_exc}")
            self.audit.log_trade_entry(
                symbol=sym,
                premium=entry_premium,
                direction=(plan.get("signal") or {}).get("direction", ""),
                signal_data=plan,
                lots=int(plan.get("desired_lots", 1) or 1),
            )
            return

        logger.error("[{}] Order failed: {}", self.NAME, result.message)
        await self.bus.publish(Topic.ORDER_SKIPPED, {
            **plan,
            "reason": result.message or "broker_order_failed",
            "broker": self.broker.broker_name,
            "execution_accounts": getattr(self, "_last_account_order_results", []),
            "timestamp": datetime.now(IST).isoformat(),
        }, self.NAME)
        await self.bus.publish(Topic.ALERT,
            {"type": "order_error",
             "text": f"Order failed for {sym}: {result.message}",
             "severity": "ERROR"},
            self.NAME)
        await self._halt_trading(
            reason=f"Entry order failed after retries for {sym}: {result.message}",
            severity="ERROR",
        )

    async def _skip_non_executable(self, plan: dict, mode: str) -> None:
        reason = plan.get(
            "execution_block_reason",
            "Trade is not executable with current broker data.",
        )
        logger.warning(f"[{self.NAME}] {mode} blocked | {plan.get('option_symbol')} | {reason}")
        await self.bus.publish(Topic.ORDER_SKIPPED, {
            **plan,
            "reason": reason,
            "broker": self.broker.broker_name,
            "timestamp": datetime.now(IST).isoformat(),
        }, self.NAME)
        await self.bus.publish(Topic.ALERT,
            {"type": "order_error",
             "text": f"{mode} blocked for {plan.get('option_symbol')}: {reason}",
             "severity": "WARNING"},
            self.NAME)

    def _apply_capital_gate(self, plan: dict):
        premium = float(plan.get("est_premium", plan.get("entry_premium", 0)) or 0)
        lot_size = self._resolve_lot_size(plan)
        quantity = int(plan.get("quantity", lot_size) or lot_size)
        requested_lots = int(plan.get("desired_lots") or max(1, quantity // max(lot_size, 1)) or 1)
        requested_lots = self._cap_lots_by_policy(
            requested_lots,
            premium=premium,
            lot_size=lot_size,
        )
        lots_to_use = max(1, requested_lots)
        capital_status = self.capital_manager.get_status()

        gated = dict(plan)
        gated["lot_size"] = lot_size
        gated["desired_lots"] = lots_to_use
        gated["quantity"] = lots_to_use * lot_size
        gated["total_invested"] = round(premium * gated["quantity"], 2)
        gated["capital_daily_budget"] = float(capital_status.daily_budget)
        gated["capital_used_before"] = float(capital_status.capital_used)
        gated["capital_required"] = round(premium * lot_size * lots_to_use, 2)
        gated["capital_remaining_after"] = round(
            float(capital_status.capital_remaining) - gated["capital_required"],
            2,
        )
        gated.setdefault("selection_notes", [])
        if isinstance(gated["selection_notes"], list):
            gated["selection_notes"].append(
                f"Capital gate disabled: {lots_to_use} lot(s), "
                f"₹{gated['capital_required']:,.0f} notionally tracked"
            )
        return gated, None

    @staticmethod
    def _cap_lots_by_policy(lots: int, *, premium: float, lot_size: int) -> int:
        max_lots = max(1, min(int(MAX_POSITION_LOTS), 3))
        premium = max(float(premium or 0.0), 0.0)
        lot_size = max(int(lot_size or 5), 1)
        lots = max(1, min(int(lots or 1), max_lots))
        cap_pct = 10.0
        if lots > 1:
            cap_pct = max(10.0, min(float(STRONG_SIGNAL_CAPITAL_PCT or 10.0), 25.0))
        max_trade_value = max(float(DEPLOYED_CAPITAL or 0.0) * (cap_pct / 100.0), premium * lot_size * lots)
        max_lots_by_capital = max(1, int(max_trade_value // max(premium * lot_size, 1.0)))
        return max(1, min(lots, max_lots_by_capital, max_lots))

    def _record_capital_opened(self, plan: dict, entry_premium: float) -> None:
        premium = float(entry_premium or plan.get("est_premium", 0) or 0)
        lot_size = self._resolve_lot_size(plan)
        lots = int(plan.get("desired_lots") or max(1, int(plan.get("quantity", lot_size) or lot_size) // max(lot_size, 1)))
        trade_id = str(plan.get("option_symbol") or plan.get("order_id") or datetime.now(IST).isoformat())
        self.capital_manager.record_trade_opened(
            trade_id=trade_id,
            premium=premium,
            lot_size=lot_size,
            lots=lots,
        )

    async def _skip_due_to_capital(self, plan_payload: dict, allowance, mode: str) -> None:
        reason = getattr(allowance, "reason", "daily capital budget exceeded")
        logger.warning(
            f"[{self.NAME}] {mode} blocked by capital cap | "
            f"{plan_payload.get('option_symbol')} | {reason}"
        )
        await self.bus.publish(Topic.ORDER_SKIPPED, {
            **plan_payload,
            "reason": f"capital_cap: {reason}",
            "broker": self.broker.broker_name,
            "timestamp": datetime.now(IST).isoformat(),
        }, self.NAME)
        self.audit.log_signal_blocked(reason=reason, signal_data=plan_payload)

    async def _skip_due_to_order_guard(self, plan: dict, reason: str) -> None:
        logger.warning(f"[{self.NAME}] Order blocked | {plan.get('option_symbol')} | {reason}")
        await self.bus.publish(Topic.ORDER_SKIPPED, {
            **plan,
            "reason": f"order_guard: {reason}",
            "broker": self.broker.broker_name,
            "timestamp": datetime.now(IST).isoformat(),
        }, self.NAME)
        self.audit.log_signal_blocked(reason=f"order_guard: {reason}", signal_data=plan)

    async def _skip_due_to_halt(self, plan: dict, mode: str) -> None:
        reason = "Trading is halted by system safety controls."
        await self.bus.publish(Topic.ORDER_SKIPPED, {
            **plan,
            "reason": reason,
            "broker": self.broker.broker_name,
            "timestamp": datetime.now(IST).isoformat(),
        }, self.NAME)
        logger.warning(f"[{self.NAME}] {mode} skipped | {reason}")

    async def _place_order_with_retry(
        self,
        symbol: str,
        quantity: int,
        transaction: str,
    ):
        result, account_results = await place_market_order_parallel(
            self.execution_accounts,
            symbol=symbol,
            quantity=quantity,
            transaction=transaction,
            product="MIS",
            exchange="NFO",
            attempts=ORDER_RETRY_ATTEMPTS,
            retry_delay_sec=ORDER_RETRY_DELAY_SEC,
        )
        self._last_account_order_results = account_results
        if len(self.execution_accounts) > 1:
            placed = [row for row in account_results if row.get("status") == "PLACED"]
            summary = ", ".join(f"{r.get('account')}:{r.get('status')}({r.get('order_id') or r.get('message')})" for r in account_results)
            logger.info(
                f"[{self.NAME}] Parallel order result | {transaction} {symbol} "
                f"| placed={len(placed)}/{len(account_results)} | {summary}"
            )
        return result

    async def _place_bracket_order_with_retry(
        self,
        symbol: str,
        quantity: int,
        stop_loss_price: float,
        target_price: float,
    ):
        last_result = None
        for attempt in range(1, ORDER_RETRY_ATTEMPTS + 1):
            result = self.broker.place_bracket_order(
                symbol=symbol,
                quantity=quantity,
                transaction="BUY",
                stop_loss_price=stop_loss_price,
                target_price=target_price,
                product="MIS",
                exchange="NFO",
            )
            if result.status == "PLACED":
                return result
            last_result = result
            logger.warning(
                f"[{self.NAME}] Bracket attempt {attempt}/{ORDER_RETRY_ATTEMPTS} failed "
                f"| {symbol} | {result.message}"
            )
            if attempt < ORDER_RETRY_ATTEMPTS:
                await asyncio.sleep(ORDER_RETRY_DELAY_SEC)
        return last_result

    async def _halt_trading(self, reason: str, severity: str = "WARNING") -> None:
        self._trading_halted = True
        payload = {
            "trading_enabled": False,
            "status": "HALTED",
            "reason": reason,
            "source": self.NAME,
            "timestamp": datetime.now(IST).isoformat(),
        }
        await self.bus.publish(Topic.SYSTEM_STATUS, payload, self.NAME)
        await self.bus.publish(Topic.ALERT,
            {"type": "system_halt", "text": reason, "severity": severity},
            self.NAME)

    async def _narrate(self, plan: dict, dry: bool) -> None:
        action = "observing (dry-run, no real order)" if dry else "placing order now"
        sig    = plan.get("signal", {})
        sym    = str(plan.get("symbol") or sig.get("symbol") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
        price  = sig.get("spot") or sig.get("price") or sig.get("ltp") or sig.get("nifty_ltp") or 0
        prompt = (
            f"Narrate this {sym} trade {action} in 2 sentences.\n"
            f"Contract: {plan.get('option_symbol')} | {sym} Price: {price}\n"
            f"Entry ~{plan.get('est_premium')} | SL {plan.get('sl_premium')} | Target {plan.get('target_premium')}\n"
            f"Confidence: {plan.get('ml_confidence', 0):.0%}"
        )
        text = await call_llm_context_async(
            prompt,
            cache_key=f"trade_narrative|{sig.get('timestamp', plan.get('timestamp', ''))}|{plan.get('option_symbol', '')}|{action}",
            task_type=TaskType.TRADE_NARRATIVE,
            max_tokens=100,
            rank_score=float(plan.get("ml_rank_score", plan.get("ml_confidence", 0.0)) or 0.0),
            success_prob=float(plan.get("ml_confidence", 0.0) or 0.0),
        )
        if text:
            await self.bus.publish(Topic.ALERT, {"type": "trade_narrative", "text": text}, self.NAME)

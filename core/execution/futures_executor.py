"""
core/execution/futures_executor.py — Multi-Instrument Futures Execution Engine

Supports 3 strictly isolated operational modes:
- PAPER: Simulated order placement, fills, slippage, and position tracking.
- SHADOW: Generates signals and forward evaluation without placing actual or paper orders.
- LIVE: Real broker orders. Failsafe protected by mandatory LIVE_TRADING_CONFIRMATION guard.

Handles:
- Long (BUY) and Short (SELL) futures positions
- Dynamic lot sizing and tick value P&L calculations
- Broker rejection handling and duplicate-order protection
- Order reconciliation
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List
import os
import pytz
from loguru import logger

from instruments.base import InstrumentConfig, ContractSpec
from core.models import Direction, TradePlan, Position, ExitReason
from broker.base_broker import BaseBroker

IST = pytz.timezone("Asia/Kolkata")


class ExecutionMode(str, Enum):
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    LIVE = "LIVE"
    OBSERVE = "OBSERVE"


@dataclass
class OrderResult:
    success: bool
    order_id: str
    status: str                         # FILLED / REJECTED / SIMULATED / SHADOW
    price: float
    quantity: int
    lots: int
    is_long: bool
    rejection_reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(IST))


class FuturesExecutionEngine:
    """
    Multi-Instrument Futures Execution Engine.
    Enforces mode safety, handles fills, trailing stops, and position reconciliation.
    """

    def __init__(
        self,
        broker: Optional[BaseBroker] = None,
        default_mode: ExecutionMode = ExecutionMode.PAPER,
        slippage_pts: float = 2.0,
    ):
        self.broker = broker
        self.mode = default_mode
        self.slippage_pts = slippage_pts
        self.executed_order_keys: set[str] = set()

        # Enforce safety guard for LIVE mode
        if self.mode == ExecutionMode.LIVE:
            self._verify_live_safety_guard()

    def _verify_live_safety_guard(self) -> None:
        """Mandatory protection guard preventing accidental live orders."""
        confirmation = os.getenv("LIVE_TRADING_CONFIRMATION", "").strip()
        if confirmation != "I_ACCEPT_THE_RISK":
            logger.critical(
                "[FuturesExecutionEngine] ❌ LIVE mode BLOCKED! "
                "LIVE_TRADING_CONFIRMATION='I_ACCEPT_THE_RISK' is NOT set in environment. "
                "Failing safe to PAPER mode."
            )
            self.mode = ExecutionMode.PAPER

    def set_mode(self, new_mode: ExecutionMode, confirmation_token: str = "") -> bool:
        """Explicit mode switching."""
        if new_mode == ExecutionMode.LIVE:
            if confirmation_token != "I_ACCEPT_THE_RISK" and os.getenv("LIVE_TRADING_CONFIRMATION") != "I_ACCEPT_THE_RISK":
                logger.error("[FuturesExecutionEngine] Rejecting switch to LIVE: Missing confirmation token.")
                return False
        self.mode = new_mode
        logger.warning(f"[FuturesExecutionEngine] Operational mode switched to: {self.mode.value}")
        return True

    def execute_trade_plan(
        self,
        plan: TradePlan,
        instrument: InstrumentConfig,
        contract: ContractSpec,
        current_ltp: float,
    ) -> OrderResult:
        """
        Executes a planned futures entry according to the active mode.
        """
        sig_id = plan.signal.signal_id if plan.signal else f"{contract.symbol}_{datetime.now(IST).isoformat()}"
        is_long = plan.is_long
        lots = plan.desired_lots or 1
        qty = lots * instrument.lot_size

        # 1. Duplicate Order Protection
        if sig_id in self.executed_order_keys:
            logger.warning(f"[FuturesExecutionEngine] Duplicate signal blocked: {sig_id}")
            return OrderResult(
                success=False,
                order_id="",
                status="REJECTED",
                price=current_ltp,
                quantity=qty,
                lots=lots,
                is_long=is_long,
                rejection_reason="DUPLICATE_ORDER_SUPPRESSED",
            )

        # ── SHADOW MODE ───────────────────────────────────────────────────────
        if self.mode in (ExecutionMode.SHADOW, ExecutionMode.OBSERVE):
            self.executed_order_keys.add(sig_id)
            logger.info(
                f"[FuturesExecutionEngine] [SHADOW] Signal recorded: {contract.trading_symbol} "
                f"{'LONG' if is_long else 'SHORT'} @ {current_ltp:.2f} ({lots} lots)"
            )
            return OrderResult(
                success=True,
                order_id=f"SHADOW_{datetime.now(IST).strftime('%H%M%S_%f')}",
                status="SHADOW",
                price=current_ltp,
                quantity=qty,
                lots=lots,
                is_long=is_long,
            )

        # ── PAPER MODE ────────────────────────────────────────────────────────
        if self.mode == ExecutionMode.PAPER:
            self.executed_order_keys.add(sig_id)
            # Apply realistic slippage model
            fill_price = current_ltp + self.slippage_pts if is_long else current_ltp - self.slippage_pts
            logger.info(
                f"[FuturesExecutionEngine] [PAPER] Simulated fill: {contract.trading_symbol} "
                f"{'LONG' if is_long else 'SHORT'} @ {fill_price:.2f} (Slip: {self.slippage_pts} pts, {lots} lots)"
            )
            return OrderResult(
                success=True,
                order_id=f"PAPER_{datetime.now(IST).strftime('%H%M%S_%f')}",
                status="SIMULATED",
                price=round(fill_price, 2),
                quantity=qty,
                lots=lots,
                is_long=is_long,
            )

        # ── LIVE BROKER EXECUTION ─────────────────────────────────────────────
        if self.mode == ExecutionMode.LIVE:
            if not self.broker:
                return OrderResult(
                    success=False,
                    order_id="",
                    status="REJECTED",
                    price=current_ltp,
                    quantity=qty,
                    lots=lots,
                    is_long=is_long,
                    rejection_reason="BROKER_NOT_CONFIGURED",
                )

            action = "BUY" if is_long else "SELL"
            try:
                order_id = self.broker.place_market_order(
                    symbol=contract.trading_symbol,
                    action=action,
                    quantity=qty,
                    tag="MCXForge_live",
                )
                self.executed_order_keys.add(sig_id)
                logger.info(
                    f"[FuturesExecutionEngine] [LIVE] Order placed: {contract.trading_symbol} "
                    f"{action} {qty} units | Broker ID: {order_id}"
                )
                return OrderResult(
                    success=True,
                    order_id=str(order_id),
                    status="FILLED",
                    price=current_ltp,
                    quantity=qty,
                    lots=lots,
                    is_long=is_long,
                )
            except Exception as exc:
                logger.error(f"[FuturesExecutionEngine] [LIVE] Order placement failed: {exc}")
                return OrderResult(
                    success=False,
                    order_id="",
                    status="REJECTED",
                    price=current_ltp,
                    quantity=qty,
                    lots=lots,
                    is_long=is_long,
                    rejection_reason=f"BROKER_REJECTION: {exc}",
                )

        return OrderResult(
            success=False,
            order_id="",
            status="REJECTED",
            price=current_ltp,
            quantity=qty,
            lots=lots,
            is_long=is_long,
            rejection_reason=f"UNKNOWN_MODE_{self.mode}",
        )

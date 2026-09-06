"""
utils/partial_exit_executor.py
==============================
Broker-safe partial square-off helper for profit-ladder exits.

This module deliberately only places market SELL exits for already-open long
option positions. It does not create new exposure, reverse a position, or assume
broker-specific bracket/cover order support.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pytz
from utils.multi_account_execution import load_execution_accounts, place_market_order_parallel

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class PartialExitResult:
    success: bool
    order_id: str
    symbol: str
    quantity: int
    lots: int
    fill_price: float
    reason: str
    simulated: bool
    timestamp: str
    account_results: list[dict] | None = None


class PartialExitExecutor:
    """Executes partial exits without changing the rest of the position state."""

    def __init__(self, broker=None, *, product: str = "MIS") -> None:
        self.broker = broker
        self.product = product
        self.execution_accounts = load_execution_accounts(broker) if broker is not None else []

    @staticmethod
    def _exchange_for_symbol(symbol: str) -> str:
        upper = str(symbol or "").upper()
        if "SENSEX" in upper or upper.startswith("BSE:"):
            return "BFO"
        return "NFO"

    async def partial_exit(
        self,
        *,
        symbol: str,
        lots_to_close: int,
        lot_size: int,
        current_premium: float,
        simulated: bool,
        reason: str,
        product: Optional[str] = None,
    ) -> PartialExitResult:
        lots = max(int(lots_to_close or 0), 0)
        lot = max(int(lot_size or 0), 0)
        quantity = lots * lot
        ts = datetime.now(IST).isoformat()

        if not symbol or lots <= 0 or quantity <= 0:
            return PartialExitResult(False, "", symbol, quantity, lots, current_premium, "invalid_partial_exit_size", simulated, ts)

        if simulated or self.broker is None:
            order_id = f"SIM_PARTIAL_{datetime.now(IST).strftime('%Y%m%d%H%M%S')}"
            logger.info(
                f"[PartialExitExecutor] SIM partial exit | {symbol} | "
                f"lots={lots} qty={quantity} @ {current_premium:.2f} | {reason}"
            )
            return PartialExitResult(True, order_id, symbol, quantity, lots, current_premium, reason, True, ts, [])

        try:
            result, account_results = await place_market_order_parallel(
                self.execution_accounts or load_execution_accounts(self.broker),
                symbol=symbol,
                quantity=quantity,
                transaction="SELL",
                product=product or self.product,
                exchange=self._exchange_for_symbol(symbol),
                attempts=2,
                retry_delay_sec=1.0,
            )
            ok = str(getattr(result, "status", "")).upper() in {"PLACED", "COMPLETE", "COMPLETED", "SUCCESS"}
            message = str(getattr(result, "message", "") or reason)
            order_id = str(getattr(result, "order_id", "") or "")
            if ok:
                logger.info(
                    f"[PartialExitExecutor] Partial exit placed | {symbol} | "
                    f"order={order_id} lots={lots} qty={quantity}"
                )
            else:
                logger.error(f"[PartialExitExecutor] Partial exit rejected | {symbol} | {message}")
            return PartialExitResult(ok, order_id, symbol, quantity, lots, current_premium, message, False, ts, account_results)
        except Exception as exc:
            logger.error(f"[PartialExitExecutor] Partial exit failed | {symbol} | {exc}")
            return PartialExitResult(False, "", symbol, quantity, lots, current_premium, str(exc), False, ts, [])

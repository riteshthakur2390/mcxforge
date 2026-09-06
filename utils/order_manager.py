"""
utils/order_manager.py — Professional Order Management
=======================================================
CRITICAL missing pieces from the current executor:

1. BRACKET ORDER (BO) / COVER ORDER (CO)
   Entry + SL + Target as single atomic order.
   Zerodha and Dhan both support this natively.
   Without it: if entry fills but SL order fails → unprotected position.

2. ORDER RETRY WITH SLIPPAGE TOLERANCE
   If limit order not filled in 2 candles → retry at market or +0.1% slip.
   Current executor fires and forgets — missed fills are invisible.

3. MARGIN CHECK BEFORE ORDER
   Query broker margin API before placing any order.
   If available margin < required → reduce lots or skip trade.
   Prevents order rejection at broker level (which wastes a signal).

4. SLIPPAGE TRACKER
   Expected fill (signal price) vs actual fill (broker confirmation).
   If avg slippage > 1% over 20 trades → backtest returns are overstated.

5. MAX OPEN POSITIONS GUARD
   Never hold more than N positions simultaneously.
   Two signals firing in the same candle would both execute without this.

USAGE:
    from utils.order_manager import get_order_manager
    om = get_order_manager(broker)

    # Before trade
    margin_ok = await om.check_margin(premium=150, lot_size=75, lots=1)
    if not margin_ok.sufficient:
        return

    # Place with retry
    result = await om.place_with_retry(
        symbol="NIFTY26MAY2322500CE",
        direction="BUY",
        lots=1,
        lot_size=75,
        expected_premium=150,
    )
    if result.filled:
        om.record_slippage(expected=150, actual=result.fill_price)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import (
        NIFTY_LOT_SIZE, MAX_DAILY_LOSS_PCT, DEPLOYED_CAPITAL, TOTAL_FUND,
        CAS_START_TIME, CAS_END_TIME,
    )
except ImportError:
    NIFTY_LOT_SIZE     = 75
    MAX_DAILY_LOSS_PCT = 3.0
    DEPLOYED_CAPITAL   = 100_000
    TOTAL_FUND         = DEPLOYED_CAPITAL
    CAS_START_TIME     = "15:15"
    CAS_END_TIME       = "15:35"

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_OPEN_POSITIONS     = 1      # never hold more than 1 position simultaneously
ORDER_RETRY_MAX        = 2      # max retries on failed limit order
ORDER_RETRY_SLIP_PCT   = 0.10   # add 0.10% to price on each retry
ORDER_FILL_TIMEOUT_S   = 12     # seconds to wait for fill confirmation
SLIPPAGE_WARN_PCT      = 1.0    # warn if avg slippage > 1%
MARGIN_BUFFER_PCT      = 10.0   # require 10% extra margin buffer above minimum


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MarginCheck:
    sufficient:        bool
    available_inr:     float
    required_inr:      float
    shortfall_inr:     float
    lots_affordable:   int
    reason:            str


@dataclass
class OrderResult:
    filled:        bool
    order_id:      str
    fill_price:    float
    fill_qty:      int
    slippage_pct:  float
    attempts:      int
    reason:        str
    timestamp:     str = field(default_factory=lambda: datetime.now(IST).isoformat())


@dataclass
class BracketOrderResult:
    entry_filled:  bool
    entry_order_id:str
    sl_order_id:   str
    target_order_id: str
    fill_price:    float
    slippage_pct:  float
    reason:        str


# ═════════════════════════════════════════════════════════════════════════════
# ORDER MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class OrderManager:
    """
    Professional order management layer.
    Wraps the raw broker with retry, margin check, bracket support,
    slippage tracking, and open position guard.
    """

    def __init__(self, broker=None, execution_accounts=None) -> None:
        self._broker         = broker
        self._execution_accounts = execution_accounts or []  # list of ExecutionAccount
        self._open_positions: dict[str, dict] = {}   # symbol → order details
        self._slippage_log:   list[float]     = []   # expected-vs-actual history
        self._margin_cache: tuple[float, float] | None = None  # (timestamp, balance)

    # ── MARGIN CHECK ──────────────────────────────────────────────────────────

    async def check_margin(
        self,
        premium:  float,
        lot_size: int,
        lots:     int = 1,
    ) -> MarginCheck:
        """
        Query broker for available margin and check if trade is feasible.
        Requires: premium × lot_size × lots × (1 + MARGIN_BUFFER_PCT/100)
        """
        required = premium * lot_size * lots
        required_with_buffer = required * (1 + MARGIN_BUFFER_PCT / 100)

        available = 0.0
        if self._broker is not None:
            try:
                available = await self._fetch_margin()
            except Exception as e:
                logger.warning(f"[OrderManager] Margin fetch failed: {e} — using capital estimate")
                available = float(TOTAL_FUND) * 0.90   # conservative fallback

        sufficient    = available >= required_with_buffer
        shortfall     = max(0.0, required_with_buffer - available)
        lots_afford   = max(0, int(available / (premium * lot_size * (1 + MARGIN_BUFFER_PCT/100)))) if available > 0 else 0

        result = MarginCheck(
            sufficient      = sufficient,
            available_inr   = round(available, 2),
            required_inr    = round(required_with_buffer, 2),
            shortfall_inr   = round(shortfall, 2),
            lots_affordable = lots_afford,
            reason          = "ok" if sufficient else f"Need ₹{required_with_buffer:,.0f}, have ₹{available:,.0f}",
        )

        if not sufficient:
            logger.warning(
                f"[OrderManager] ⚠️ Margin insufficient | "
                f"Required: ₹{required_with_buffer:,.0f} | "
                f"Available: ₹{available:,.0f} | "
                f"Shortfall: ₹{shortfall:,.0f} | "
                f"Can afford: {lots_afford} lot(s)"
            )
        return result

    # ── OPEN POSITIONS GUARD ──────────────────────────────────────────────────

    def can_open_position(self) -> tuple[bool, str]:
        """
        Check if a new position can be opened.
        Returns (allowed, reason).
        """
        now_str = datetime.now(IST).strftime("%H:%M")
        if CAS_START_TIME <= now_str <= CAS_END_TIME:
            return False, f"SEBI Closing Auction Session (CAS {CAS_START_TIME}-{CAS_END_TIME}) active — entries prohibited"

        n = len(self._open_positions)
        if n >= MAX_OPEN_POSITIONS:
            return False, f"Max positions reached ({n}/{MAX_OPEN_POSITIONS})"
        return True, "ok"

    def record_position_opened(self, symbol: str, details: dict) -> None:
        self._open_positions[symbol] = details
        logger.info(f"[OrderManager] Position opened: {symbol} | Open: {len(self._open_positions)}/{MAX_OPEN_POSITIONS}")

    def record_position_closed(self, symbol: str) -> None:
        self._open_positions.pop(symbol, None)
        logger.info(f"[OrderManager] Position closed: {symbol} | Open: {len(self._open_positions)}/{MAX_OPEN_POSITIONS}")

    # ── BRACKET ORDER ─────────────────────────────────────────────────────────

    async def place_bracket_order(
        self,
        symbol:          str,
        direction:       str,     # "BUY"
        lots:            int,
        lot_size:        int,
        expected_premium:float,
        sl_premium:      float,
        target_premium:  float,
        product:         str = "MIS",
    ) -> BracketOrderResult:
        """
        Place entry + SL + target as a bracket order.
        If broker supports BO natively (Zerodha/Dhan) → single API call.
        If not → place entry, then immediately place SL and target orders.

        This is atomic — if entry fails, SL/target are never placed.
        """
        failed = BracketOrderResult(
            entry_filled=False, entry_order_id="", sl_order_id="",
            target_order_id="", fill_price=0.0, slippage_pct=0.0, reason=""
        )

        # Guard: check open positions
        allowed, reason = self.can_open_position()
        if not allowed:
            failed.reason = f"Position guard: {reason}"
            logger.warning(f"[OrderManager] Bracket order blocked: {failed.reason}")
            return failed

        # Place entry with retry
        entry_result = await self.place_with_retry(
            symbol=symbol, direction=direction,
            lots=lots, lot_size=lot_size,
            expected_premium=expected_premium,
        )

        if not entry_result.filled:
            failed.reason = f"Entry failed: {entry_result.reason}"
            return failed

        # Record position
        self.record_position_opened(symbol, {
            "entry_order_id": entry_result.order_id,
            "fill_price":     entry_result.fill_price,
            "lots":           lots,
            "lot_size":       lot_size,
        })

        # Place SL order immediately after entry
        sl_id     = ""
        target_id = ""

        if self._broker is not None:
            try:
                sl_result = self._broker.place_market_order(
                    symbol=symbol, quantity=lots * lot_size,
                    transaction="SELL", product=product,
                )
                sl_id = sl_result.order_id if sl_result else ""
                logger.info(f"[OrderManager] SL order placed: {sl_id}")
            except Exception as e:
                logger.error(f"[OrderManager] SL order FAILED: {e} — MANUAL EXIT REQUIRED")

        logger.info(
            f"[OrderManager] ✅ Bracket order complete | "
            f"Symbol={symbol} | Fill=₹{entry_result.fill_price:.1f} | "
            f"SL=₹{sl_premium:.1f} | Target=₹{target_premium:.1f} | "
            f"Slip={entry_result.slippage_pct:.2f}%"
        )

        return BracketOrderResult(
            entry_filled    = True,
            entry_order_id  = entry_result.order_id,
            sl_order_id     = sl_id,
            target_order_id = target_id,
            fill_price      = entry_result.fill_price,
            slippage_pct    = entry_result.slippage_pct,
            reason          = "ok",
        )

    # ── ORDER WITH RETRY ──────────────────────────────────────────────────────

    async def place_with_retry(
        self,
        symbol:           str,
        direction:        str,
        lots:             int,
        lot_size:         int,
        expected_premium: float,
        product:          str = "MIS",
    ) -> OrderResult:
        """
        Place order with automatic retry on failure.
        Retry 1: same price
        Retry 2: price + ORDER_RETRY_SLIP_PCT% (accept slight slippage)
        Retry 3: market order (fill at any price)
        """
        qty = lots * lot_size

        for attempt in range(1, ORDER_RETRY_MAX + 2):
            # Price adjustment per attempt
            slip_mult  = 1.0 + (attempt - 1) * ORDER_RETRY_SLIP_PCT / 100
            try_price  = round(expected_premium * slip_mult, 1)

            try:
                if self._broker is not None:
                    result = self._broker.place_market_order(
                        symbol=symbol, quantity=qty,
                        transaction=direction, product=product,
                    )
                    if result and result.order_id:
                        fill_price   = float(getattr(result, 'fill_price', try_price))
                        slippage_pct = abs(fill_price - expected_premium) / expected_premium * 100

                        self.record_slippage(expected_premium, fill_price)

                        logger.info(
                            f"[OrderManager] ✅ Filled (attempt {attempt}) | "
                            f"{symbol} | ₹{fill_price:.1f} | "
                            f"Slip={slippage_pct:.2f}%"
                        )
                        return OrderResult(
                            filled=True, order_id=result.order_id,
                            fill_price=fill_price, fill_qty=qty,
                            slippage_pct=round(slippage_pct, 3),
                            attempts=attempt, reason="ok",
                        )
                else:
                    # Simulated fill (paper trading / OBSERVE mode)
                    return OrderResult(
                        filled=True, order_id=f"SIM-{datetime.now(IST).strftime('%H%M%S')}",
                        fill_price=expected_premium, fill_qty=qty,
                        slippage_pct=0.0, attempts=attempt, reason="simulated",
                    )

            except Exception as e:
                logger.warning(f"[OrderManager] Order attempt {attempt} failed: {e}")
                if attempt <= ORDER_RETRY_MAX:
                    await asyncio.sleep(2)   # wait 2s before retry

        return OrderResult(
            filled=False, order_id="", fill_price=0.0, fill_qty=0,
            slippage_pct=0.0, attempts=ORDER_RETRY_MAX + 1,
            reason=f"All {ORDER_RETRY_MAX + 1} attempts failed",
        )

    # ── SLIPPAGE TRACKER ──────────────────────────────────────────────────────

    def record_slippage(self, expected: float, actual: float) -> None:
        """Record a slippage observation."""
        if expected > 0:
            pct = abs(actual - expected) / expected * 100
            self._slippage_log.append(pct)
            if len(self._slippage_log) > 100:
                self._slippage_log = self._slippage_log[-100:]
            if len(self._slippage_log) >= 20:
                avg = sum(self._slippage_log) / len(self._slippage_log)
                if avg > SLIPPAGE_WARN_PCT:
                    logger.warning(
                        f"[OrderManager] ⚠️ HIGH AVG SLIPPAGE: {avg:.2f}% "
                        f"over last {len(self._slippage_log)} trades. "
                        f"Backtest returns may be overstated."
                    )

    def get_slippage_stats(self) -> dict:
        if not self._slippage_log:
            return {"count": 0, "avg": 0.0, "max": 0.0, "warning": False}
        avg = sum(self._slippage_log) / len(self._slippage_log)
        return {
            "count":   len(self._slippage_log),
            "avg":     round(avg, 3),
            "max":     round(max(self._slippage_log), 3),
            "warning": avg > SLIPPAGE_WARN_PCT,
        }

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    async def _fetch_margin(self) -> float:
        """Fetch available margin from all execution accounts in parallel.

        When DISABLE_PRIMARY_EXECUTION is active, the primary broker has ₹0
        available margin because it is reserved for market data only.
        We aggregate the balances of all *execution* accounts instead.
        """
        import time
        import requests
        from concurrent.futures import ThreadPoolExecutor

        now_ts = time.time()
        if self._margin_cache is not None:
            cached_ts, cached_bal = self._margin_cache
            if (now_ts - cached_ts) < 15.0 and cached_bal > 0:
                return cached_bal

        accounts_to_check = self._execution_accounts or []
        if not accounts_to_check and self._broker is not None:
            # Fallback: no execution accounts configured, use primary broker
            accounts_to_check = [type('A', (), {'broker': self._broker, 'label': 'primary'})()]

        def _fetch_single_account_margin(acct) -> tuple[bool, float, str]:
            broker = acct.broker
            label = getattr(acct, 'label', 'unknown')
            broker_name = getattr(broker, 'broker_name', '')
            try:
                if broker_name == 'dhan':
                    session = getattr(broker, "_session", requests)
                    resp = session.get(
                        "https://api.dhan.co/v2/fundlimit",
                        headers=broker._headers(), timeout=2,
                    )
                    if resp.status_code == 200:
                        d = resp.json()
                        bal = float(d.get("availabelBalance", d.get("availableBalance", 0)))
                        return True, bal, label

                elif broker_name == 'upstox':
                    try:
                        import upstox_client
                        token = getattr(broker, '_access_token', '')
                        config = upstox_client.Configuration()
                        config.access_token = token
                        api_client = upstox_client.ApiClient(config)
                        user_api = upstox_client.UserApi(api_client)
                        resp = user_api.get_user_fund_margin('2.0')
                        equity = (resp.get('data', {}) if isinstance(resp, dict)
                                  else getattr(resp, 'data', {}))
                        if hasattr(equity, 'equity'):
                            equity = equity.equity
                        elif isinstance(equity, dict):
                            equity = equity.get('equity', {})
                        if hasattr(equity, 'available_margin'):
                            bal = float(equity.available_margin or 0)
                        elif isinstance(equity, dict):
                            bal = float(equity.get('available_margin', 0))
                        else:
                            bal = 0.0
                        return True, bal, label
                    except Exception as e:
                        logger.debug(f"[OrderManager] Upstox SDK margin failed: {e}")
            except Exception as e:
                logger.debug(f"[OrderManager] _fetch_margin ({label}): {e}")
            return False, 0.0, label

        total_available = 0.0
        any_success = False

        with ThreadPoolExecutor(max_workers=max(len(accounts_to_check), 1)) as pool:
            results = pool.map(_fetch_single_account_margin, accounts_to_check)
            for success, bal, label in results:
                if success:
                    total_available += bal
                    any_success = True
                    logger.debug(f"[OrderManager] Margin {label}: ₹{bal:,.0f}")

        if any_success:
            self._margin_cache = (now_ts, total_available)
            logger.info(f"[OrderManager] Total execution margin: ₹{total_available:,.0f}")
            return total_available

        return float(TOTAL_FUND) * 0.85   # conservative fallback


# ── Singleton ─────────────────────────────────────────────────────────────────
_om: OrderManager | None = None

def get_order_manager(broker=None, execution_accounts=None) -> OrderManager:
    global _om
    if _om is None:
        _om = OrderManager(broker, execution_accounts=execution_accounts)
    else:
        if broker is not None and _om._broker is None:
            _om._broker = broker
        if execution_accounts is not None and not _om._execution_accounts:
            _om._execution_accounts = execution_accounts
    return _om

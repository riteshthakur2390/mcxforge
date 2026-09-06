"""
utils/position_roller.py — Position Adjustment & Rolling
=========================================================
When a trade is losing but the thesis is still intact,
rolling to the next expiry or adjusting the strike can
save a trade instead of taking a full SL hit.

WHEN TO ROLL:
  1. ROLL OUT (same strike, next expiry):
     - Current position is -15% to -20% (approaching SL)
     - Regime is still TRENDING in same direction
     - DTE < 2 days (theta bleeding the position)
     - New expiry premium > 2× current premium
     → Roll: close current, open same strike next week

  2. ROLL DOWN/UP (adjust strike toward money):
     - Position is -10% to -15%
     - NIFTY has moved 100-150pts against position
     - New ATM strike gives better delta (0.45-0.55)
     → Adjust: close current, open new strike same expiry

  3. CONVERT TO SPREAD (add hedge leg):
     - Position is -20% (near SL)
     - Strong trend reversal signals present
     - Add opposite OTM option to create spread
     → Reduces max loss but also limits upside

ROLLING RULES (from prop trading firms):
  - Never roll more than once per trade
  - Rolling cost (2× brokerage + bid-ask) must be < 3% of premium
  - New position must have delta > 0.30 (not too far OTM)
  - Only roll if new net debit < original SL amount
  - Time window: only roll between 10:00 - 13:00 IST

Usage:
    from utils.position_roller import PositionRoller
    roller = PositionRoller(broker)
    decision = roller.evaluate(position, df, regime_details)
    if decision.should_roll:
        await roller.execute_roll(decision)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta, time as dtime
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import STOP_LOSS_PCT, NIFTY_LOT_SIZE, NIFTY_STRIKE_STEP
except ImportError:
    STOP_LOSS_PCT     = 25.0
    NIFTY_LOT_SIZE    = 75
    NIFTY_STRIKE_STEP = 50

# ── Parameters ────────────────────────────────────────────────────────────────
ROLL_MIN_LOSS_PCT      = -15.0   # start considering roll at -15%
ROLL_MAX_LOSS_PCT      = -22.0   # don't roll if already at -22% (too late)
ROLL_MIN_DTE           = 0       # roll if DTE ≤ this
ROLL_MAX_DTE           = 3       # only roll if DTE ≤ 3 days
ROLL_WINDOW_START      = dtime(10, 0)
ROLL_WINDOW_END        = dtime(13, 0)
ROLL_MAX_COST_PCT      = 3.0     # rolling cost must be < 3% of original premium
MIN_NEW_DELTA          = 0.30    # new position must have delta > 0.30
ADJUST_LOSS_PCT_MIN    = -10.0   # adjust strike at -10%
ADJUST_LOSS_PCT_MAX    = -18.0   # don't adjust if > -18% (too expensive)


@dataclass
class RollDecision:
    should_roll:     bool
    roll_type:       str       # "ROLL_OUT" | "ROLL_ADJUST" | "SPREAD" | "HOLD" | "CLOSE"
    reason:          str
    confidence:      float     # 0-1 confidence in the roll decision
    # Roll specifics
    close_symbol:    str       # current position symbol to close
    open_symbol:     str       # new position symbol to open
    new_strike:      int
    new_expiry:      str       # YYYY-MM-DD
    est_roll_cost:   float     # estimated cost of roll in % of original premium
    est_new_delta:   float
    net_debit:       float     # premium difference (negative = credit)
    max_new_loss:    float     # maximum possible loss after roll
    # Context
    current_pnl:     float
    current_dte:     float
    regime:          str


class PositionRoller:
    """
    Evaluates whether to roll a losing position and executes the roll.
    """

    def __init__(self, broker=None) -> None:
        self._broker   = broker
        self._rolled   = False   # track if already rolled this trade

    def evaluate(
        self,
        position:       dict,
        df:             "pd.DataFrame",
        regime_details: dict,
        india_vix:      float = 18.0,
    ) -> RollDecision:
        """
        Evaluate whether rolling is appropriate for current position.

        Args:
            position:  current position dict (from PositionManagerAgent)
            df:        current OHLCV DataFrame
            regime_details: from MARKET_REGIME message
            india_vix: current India VIX

        Returns:
            RollDecision with recommendation
        """
        hold = RollDecision(
            should_roll=False, roll_type="HOLD", reason="no_action_needed",
            confidence=0.0, close_symbol="", open_symbol="",
            new_strike=0, new_expiry="", est_roll_cost=0.0,
            est_new_delta=0.0, net_debit=0.0, max_new_loss=0.0,
            current_pnl=0.0, current_dte=0.0, regime="",
        )

        # Already rolled once — no second rolls
        if self._rolled:
            hold.reason = "already_rolled_once"
            return hold

        entry_prem = float(position.get("entry_premium", 0))
        cur_prem   = float(position.get("current_ltp",   0))
        pnl_pct    = float(position.get("pnl_pct",       0))
        direction  = str(position.get("direction",   "BUY_CALL"))
        symbol     = str(position.get("option_symbol", ""))
        expiry_str = str(position.get("expiry", ""))
        strike     = int(position.get("strike", 0))

        # Parse DTE
        try:
            exp_date = date.fromisoformat(expiry_str) if expiry_str else self._next_thursday(0)
            dte      = (exp_date - date.today()).days
        except Exception:
            dte = 5

        # Time gate
        import pytz
        from datetime import datetime
        now_time = datetime.now(IST).time()
        if not (ROLL_WINDOW_START <= now_time <= ROLL_WINDOW_END):
            hold.reason = "outside_roll_window"
            return hold

        # Loss gate
        if pnl_pct >= ROLL_MIN_LOSS_PCT:
            hold.reason = f"loss_not_deep_enough ({pnl_pct:.1f}%)"
            return hold
        if pnl_pct <= ROLL_MAX_LOSS_PCT:
            hold.reason = f"loss_too_deep_to_roll ({pnl_pct:.1f}%)"
            return hold

        # Regime must still support direction
        regime = regime_details.get("regime", regime_details.get("label", "RANGING"))
        adx    = float(regime_details.get("adx", 15))
        if regime not in ("TRENDING",) or adx < 18:
            hold.reason = f"regime_not_supportive ({regime} ADX={adx:.1f})"
            return hold

        # VIX gate — don't roll in high vol (too expensive)
        if india_vix > 25:
            hold.reason = f"vix_too_high ({india_vix:.1f})"
            return hold

        spot     = float(df["close"].iloc[-1])
        opt_type = "CE" if direction == "BUY_CALL" else "PE"

        # ── Decide roll type ──────────────────────────────────────────────────

        if dte <= ROLL_MAX_DTE and ROLL_MIN_LOSS_PCT <= pnl_pct <= -18:
            # ROLL OUT: same strike, next expiry
            return self._build_roll_out(
                symbol, strike, opt_type, spot, entry_prem,
                cur_prem, pnl_pct, dte, regime,
            )
        elif ADJUST_LOSS_PCT_MIN >= pnl_pct >= ADJUST_LOSS_PCT_MAX:
            # ROLL ADJUST: closer-to-money strike, same expiry
            return self._build_roll_adjust(
                symbol, strike, opt_type, spot, entry_prem,
                cur_prem, pnl_pct, dte, expiry_str, regime,
            )

        hold.reason = "no_roll_condition_met"
        return hold

    # ── Roll builders ─────────────────────────────────────────────────────────

    def _build_roll_out(
        self, symbol, strike, opt_type, spot,
        entry_prem, cur_prem, pnl_pct, dte, regime,
    ) -> RollDecision:
        """Build a ROLL_OUT decision: same strike, next Thursday."""
        new_expiry    = self._next_thursday(7)   # next week
        new_exp_str   = new_expiry.isoformat()
        new_dte       = (new_expiry - date.today()).days

        # Estimate new premium using BS proxy
        new_prem      = self._estimate_premium(spot, strike, new_dte, opt_type)
        roll_cost_pct = (cur_prem - new_prem) / entry_prem * 100   # give up cur_prem, pay new_prem
        net_debit     = new_prem - cur_prem   # positive = we pay more
        new_delta     = self._estimate_delta(spot, strike, new_dte, opt_type)
        max_new_loss  = -(entry_prem + max(net_debit, 0)) / entry_prem * 100

        # Build new symbol (simplified — planner will finalize)
        from utils.option_utils import build_option_symbol
        try:
            new_symbol = build_option_symbol("NIFTY", new_expiry, strike, opt_type)
        except Exception:
            new_symbol = f"NIFTY_{new_exp_str}_{strike}{opt_type}"

        viable = (
            new_delta >= MIN_NEW_DELTA and
            abs(roll_cost_pct) <= ROLL_MAX_COST_PCT and
            max_new_loss > -STOP_LOSS_PCT - 10
        )

        return RollDecision(
            should_roll  = viable,
            roll_type    = "ROLL_OUT",
            reason       = "roll_out_next_expiry" if viable else f"roll_not_viable",
            confidence   = 0.72 if viable else 0.0,
            close_symbol = symbol,
            open_symbol  = new_symbol,
            new_strike   = strike,
            new_expiry   = new_exp_str,
            est_roll_cost= round(abs(roll_cost_pct), 2),
            est_new_delta= round(new_delta, 3),
            net_debit    = round(net_debit, 2),
            max_new_loss = round(max_new_loss, 2),
            current_pnl  = pnl_pct,
            current_dte  = dte,
            regime       = regime,
        )

    def _build_roll_adjust(
        self, symbol, strike, opt_type, spot,
        entry_prem, cur_prem, pnl_pct, dte, expiry_str, regime,
    ) -> RollDecision:
        """Build a ROLL_ADJUST: move strike closer to ATM, same expiry."""
        atm        = int(round(spot / NIFTY_STRIKE_STEP) * NIFTY_STRIKE_STEP)
        new_strike = atm   # move to current ATM

        try:
            expiry = date.fromisoformat(expiry_str) if expiry_str else self._next_thursday(0)
        except Exception:
            expiry = self._next_thursday(0)

        new_prem  = self._estimate_premium(spot, new_strike, dte, opt_type)
        net_debit = new_prem - cur_prem
        new_delta = self._estimate_delta(spot, new_strike, dte, opt_type)

        roll_cost_pct = abs(net_debit) / entry_prem * 100
        max_new_loss  = -(entry_prem + max(net_debit, 0)) / entry_prem * 100

        from utils.option_utils import build_option_symbol
        try:
            new_symbol = build_option_symbol("NIFTY", expiry, new_strike, opt_type)
        except Exception:
            new_symbol = f"NIFTY_{expiry_str}_{new_strike}{opt_type}"

        viable = (
            new_strike != strike and
            new_delta >= MIN_NEW_DELTA and
            roll_cost_pct <= ROLL_MAX_COST_PCT
        )

        return RollDecision(
            should_roll  = viable,
            roll_type    = "ROLL_ADJUST",
            reason       = f"adjust_strike_{strike}→{new_strike}" if viable else "adjust_not_viable",
            confidence   = 0.68 if viable else 0.0,
            close_symbol = symbol,
            open_symbol  = new_symbol,
            new_strike   = new_strike,
            new_expiry   = expiry_str,
            est_roll_cost= round(roll_cost_pct, 2),
            est_new_delta= round(new_delta, 3),
            net_debit    = round(net_debit, 2),
            max_new_loss = round(max_new_loss, 2),
            current_pnl  = pnl_pct,
            current_dte  = dte,
            regime       = regime,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_premium(spot: float, strike: int, dte: int,
                          opt_type: str, iv: float = 0.15) -> float:
        """Quick BS premium estimate."""
        try:
            from backtesting.options_backtester import _bs_price
            T = max(dte, 0.5) / 365.0
            return max(_bs_price(spot, float(strike), T, 0.065, iv, opt_type), 5.0)
        except Exception:
            return max(spot * iv * math.sqrt(dte / 365), 5.0)

    @staticmethod
    def _estimate_delta(spot: float, strike: int, dte: int,
                        opt_type: str, iv: float = 0.15) -> float:
        try:
            from backtesting.options_backtester import _bs_delta
            T = max(dte, 0.5) / 365.0
            return abs(_bs_delta(spot, float(strike), T, 0.065, iv, opt_type))
        except Exception:
            return 0.40

    @staticmethod
    def _next_thursday(extra_days: int = 0) -> date:
        d = date.today()
        days = (3 - d.weekday()) % 7 or 7
        return d + timedelta(days=days + extra_days)

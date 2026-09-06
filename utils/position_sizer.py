"""
utils/position_sizer.py — Position Sizing
==========================================
WHAT THIS ADDS (completely new — not in any existing code):

Professional position sizing used by all serious algo traders.
Current system uses the same size for every trade regardless of:
- How confident the signal is
- How volatile the market is
- How much capital is at risk
- Historical win rate

THREE MODES:

1. FIXED (current behaviour, safe baseline):
   Always trade CAPITAL_PER_TRADE / premium = N lots
   Simple, predictable, no overfitting to win rate estimates

2. VOLATILITY-ADJUSTED (recommended for Phase 2):
   When ATR is high → market is more volatile → smaller position
   When ATR is low  → market is calmer → larger position
   Formula: lots = base_lots × (avg_atr / current_atr)

3. KELLY (Phase 3, requires 50+ trade history):
   Uses historical win rate and average win/loss ratio
   Kelly fraction = (win_rate × avg_win - loss_rate × avg_loss) / avg_win
   Use fractional Kelly (25%) to be conservative

TRANSACTION COST FILTER:
  Before sizing, checks if expected profit covers transaction costs.
  If expected_profit < MIN_EXPECTED_PROFIT after broker fees, returns 0 lots.
  This eliminates the small-return trades that look profitable in backtest
  but lose money in reality due to STT, brokerage, and stamp duty.

USAGE:
  from utils.position_sizer import PositionSizer
  sizer = PositionSizer()
  lots  = sizer.compute(
      premium=115.0, confidence=0.78, atr_pct=0.4,
      win_rate=0.63, avg_win_pct=35.0, avg_loss_pct=-8.0
  )
"""

import math
from loguru import logger


class PositionSizer:

    def __init__(self):
        from config.settings import (
            POSITION_SIZING_MODE, CAPITAL_PER_TRADE, MAX_LOTS_PER_TRADE,
            KELLY_FRACTION, NIFTY_LOT_SIZE,
            BROKERAGE_PER_LOT, STT_SELL_PCT, SEBI_CHARGES_PCT,
            STAMP_DUTY_PCT, MIN_EXPECTED_PROFIT,
            TARGET_PCT, STOP_LOSS_PCT,
        )
        self.mode             = POSITION_SIZING_MODE
        self.capital          = CAPITAL_PER_TRADE
        self.max_lots         = MAX_LOTS_PER_TRADE
        self.kelly_fraction   = KELLY_FRACTION
        self.lot_size         = NIFTY_LOT_SIZE
        self.brokerage        = BROKERAGE_PER_LOT
        self.stt_pct          = STT_SELL_PCT / 100
        self.sebi_pct         = SEBI_CHARGES_PCT / 100
        self.stamp_pct        = STAMP_DUTY_PCT / 100
        self.min_profit       = MIN_EXPECTED_PROFIT
        self.target_pct       = TARGET_PCT / 100
        self.sl_pct           = STOP_LOSS_PCT / 100

    def compute(
        self,
        premium:      float,
        confidence:   float = 0.70,
        atr_pct:      float = 0.3,
        win_rate:     float = 0.60,
        avg_win_pct:  float = 30.0,
        avg_loss_pct: float = -8.0,
    ) -> int:
        """
        Compute number of lots to trade.

        Args:
            premium:      current option premium in ₹
            confidence:   signal confidence 0-1
            atr_pct:      ATR as % of price (volatility measure)
            win_rate:     historical win rate (0-1)
            avg_win_pct:  average winning trade P&L %
            avg_loss_pct: average losing trade P&L % (negative)

        Returns:
            Number of lots (0 if trade should be skipped due to costs)
        """
        # Step 1: Check transaction cost viability
        if not self._is_worth_trading(premium):
            return 0

        # Step 2: Compute lots based on mode
        if self.mode == "kelly":
            lots = self._kelly_lots(premium, win_rate, avg_win_pct, avg_loss_pct)
        elif self.mode == "volatility":
            lots = self._volatility_lots(premium, atr_pct)
        else:
            lots = self._fixed_lots(premium)

        # Step 3: Scale slightly by confidence (confidence >0.80 → +1 lot max)
        if lots > 0 and confidence >= 0.80 and lots < self.max_lots:
            lots = min(lots + 1, self.max_lots)

        return max(0, min(lots, self.max_lots))

    # ── TRANSACTION COST CHECK ────────────────────────────────────────────────

    def _is_worth_trading(self, premium: float) -> bool:
        """
        Reject trades where expected profit after costs < MIN_EXPECTED_PROFIT.
        This is the primary filter for low-quality small trades.
        """
        if premium <= 0:
            return False

        # Estimated costs for 1 lot entry + exit
        entry_value  = premium * self.lot_size
        exit_value   = premium * (1 + self.target_pct) * self.lot_size

        cost_brokerage = self.brokerage * 2          # entry + exit
        cost_stt       = exit_value * self.stt_pct   # STT on sell
        cost_sebi      = (entry_value + exit_value) * self.sebi_pct
        cost_stamp     = entry_value * self.stamp_pct
        total_cost     = cost_brokerage + cost_stt + cost_sebi + cost_stamp

        expected_profit = (premium * self.target_pct * self.lot_size) - total_cost

        if expected_profit < self.min_profit:
            logger.debug(
                f"[Sizer] Skip: premium=₹{premium:.0f} → "
                f"expected profit=₹{expected_profit:.0f} < "
                f"min=₹{self.min_profit:.0f} | cost=₹{total_cost:.0f}"
            )
            return False

        return True

    # ── SIZING METHODS ────────────────────────────────────────────────────────

    def _fixed_lots(self, premium: float) -> int:
        lots = max(1, int(self.capital / (premium * self.lot_size)))
        return min(lots, self.max_lots)

    def _volatility_lots(self, premium: float, atr_pct: float) -> int:
        """
        Reduce position size when market is more volatile.
        Base volatility anchor = 0.3% ATR.
        """
        base_lots   = self._fixed_lots(premium)
        anchor_atr  = 0.30   # normal NIFTY 5-min ATR %
        vol_factor  = anchor_atr / max(atr_pct, 0.05)
        adjusted    = base_lots * vol_factor
        return max(1, min(int(adjusted), self.max_lots))

    def _kelly_lots(
        self,
        premium:      float,
        win_rate:     float,
        avg_win_pct:  float,
        avg_loss_pct: float,
    ) -> int:
        """
        Fractional Kelly criterion.
        Only use when win_rate is based on 50+ real trades.
        """
        if win_rate <= 0 or win_rate >= 1:
            return self._fixed_lots(premium)

        avg_win  = abs(avg_win_pct) / 100
        avg_loss = abs(avg_loss_pct) / 100
        loss_rate = 1 - win_rate

        if avg_loss == 0:
            return self._fixed_lots(premium)

        # Kelly formula: f = (p × b - q) / b where b = avg_win/avg_loss
        b = avg_win / avg_loss
        kelly = (win_rate * b - loss_rate) / b
        kelly = max(0.0, kelly)   # never bet negative
        kelly *= self.kelly_fraction   # use fractional Kelly

        # Convert fraction of capital to lots
        bet_size = kelly * self.capital
        lots     = max(1, int(bet_size / (premium * self.lot_size)))
        return min(lots, self.max_lots)

    def cost_breakdown(self, premium: float, lots: int = 1) -> dict:
        """Return full cost breakdown for a given premium and lot count."""
        entry_value = premium * self.lot_size * lots
        exit_value  = premium * (1 + self.target_pct) * self.lot_size * lots
        return {
            "entry_value":  round(entry_value, 2),
            "brokerage":    round(self.brokerage * 2 * lots, 2),
            "stt":          round(exit_value * self.stt_pct, 2),
            "sebi":         round((entry_value + exit_value) * self.sebi_pct, 2),
            "stamp_duty":   round(entry_value * self.stamp_pct, 2),
            "total_cost":   round(
                self.brokerage * 2 * lots +
                exit_value * self.stt_pct +
                (entry_value + exit_value) * self.sebi_pct +
                entry_value * self.stamp_pct, 2
            ),
            "expected_profit_if_target": round(
                premium * self.target_pct * self.lot_size * lots -
                (self.brokerage * 2 * lots + exit_value * self.stt_pct), 2
            ),
        }
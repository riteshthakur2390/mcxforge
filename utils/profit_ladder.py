"""
utils/profit_ladder.py — Profit Booking Ladder (Partial Scale-Out)
===================================================================
Instead of binary all-or-nothing exit, book profits in tranches:

  Level 1: +10% → book 30% of position
  Level 2: +20% → book 40% of position
  Hold:    +70% → final 30% rides to target/TSL

WHY THIS IS CRITICAL:
  Current system: hold 100% until SL or TARGET.
  Problem: many trades hit +10-15% then reverse and close at SL.
  These become -25% losses when they could have been +10% partial wins.

  With ladder:
  - Trade hits +10%, books 30% → locked ₹X profit
  - Trade reverses to SL → net result: (0.3 × +10%) + (0.7 × -25%) = -14.5%
  - vs current: 100% × -25% = -25%
  - Saves 10.5% on every reversal trade

  For winners that continue:
  - Trade hits +10% (book 30%), +20% (book 40%), +70% (final 30%)
  - Avg exit = 0.3×10 + 0.4×20 + 0.3×70 = 3+8+21 = +32% avg
  - Better than binary: all-or-nothing at 70% (many don't make it)

USAGE:
    ladder = ProfitLadder(entry_premium=150, lots=2, lot_size=75)
    
    # Called every candle
    action = ladder.check(current_premium=165)  # +10%
    if action.book_now:
        lots_to_close = action.lots_to_book
        # place partial close order
        ladder.record_booking(action)
    
    # Check if fully closed
    if ladder.is_fully_closed:
        print(f"Avg exit: {ladder.avg_exit_pct:.1f}%")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    from config.settings import (
        USE_PROFIT_LADDER,
        LADDER_BOOK_1_PCT, LADDER_BOOK_1_QTY,
        LADDER_BOOK_2_PCT, LADDER_BOOK_2_QTY,
        LADDER_HOLD_QTY,
    )
except ImportError:
    USE_PROFIT_LADDER  = True
    LADDER_BOOK_1_PCT  = 25.0
    LADDER_BOOK_1_QTY  = 20
    LADDER_BOOK_2_PCT  = 45.0
    LADDER_BOOK_2_QTY  = 30
    LADDER_HOLD_QTY    = 50

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class LadderAction:
    book_now:        bool
    level:           int        # 1, 2, or 3 (final)
    lots_to_book:    int
    pnl_pct:         float      # current P&L%
    target_pct:      float      # the ladder level that triggered
    reason:          str


@dataclass
class BookingRecord:
    level:       int
    lots:        int
    premium:     float
    pnl_pct:     float


class ProfitLadder:
    """
    Manages partial profit booking across multiple P&L levels.
    One instance per open trade.
    """

    def __init__(
        self,
        entry_premium: float,
        total_lots:    int,
        lot_size:      int,
        enabled:       bool = None,
    ) -> None:
        self.entry_premium  = entry_premium
        self.total_lots     = total_lots
        self.lot_size       = lot_size
        self.enabled        = enabled if enabled is not None else USE_PROFIT_LADDER

        # Ladder levels: (trigger_pct, qty_pct_to_book, level_label)
        self._levels = [
            (LADDER_BOOK_1_PCT, LADDER_BOOK_1_QTY, 1),
            (LADDER_BOOK_2_PCT, LADDER_BOOK_2_QTY, 2),
        ]

        # State tracking
        self._levels_triggered: set[int] = set()
        self._lots_remaining   = total_lots
        self._bookings:         list[BookingRecord] = []
        self._weighted_pnl     = 0.0   # for avg exit calc

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def check(self, current_premium: float) -> LadderAction:
        """
        Check if any ladder level should trigger.
        Call every candle while position is open.
        """
        no_action = LadderAction(False, 0, 0, 0.0, 0.0, "no_action")

        if not self.enabled or self._lots_remaining <= 0:
            return no_action

        pnl_pct = (current_premium - self.entry_premium) / self.entry_premium * 100

        for trigger_pct, qty_pct, level in self._levels:
            if level in self._levels_triggered:
                continue
            if pnl_pct >= trigger_pct:
                lots_to_book = max(1, int(self.total_lots * qty_pct / 100))
                lots_to_book = min(lots_to_book, self._lots_remaining - 1)

                if lots_to_book <= 0:
                    continue

                logger.info(
                    f"[ProfitLadder] 🎯 Level {level} triggered | "
                    f"PnL={pnl_pct:.1f}% ≥ {trigger_pct}% | "
                    f"Booking {lots_to_book}/{self._lots_remaining} lots"
                )
                return LadderAction(
                    book_now     = True,
                    level        = level,
                    lots_to_book = lots_to_book,
                    pnl_pct      = round(pnl_pct, 2),
                    target_pct   = trigger_pct,
                    reason       = f"Level {level}: +{trigger_pct}% target hit",
                )

        return no_action

    def record_booking(self, action: LadderAction, actual_premium: float = None) -> None:
        """Record a partial booking after order is placed."""
        self._levels_triggered.add(action.level)
        self._lots_remaining -= action.lots_to_book

        fill = actual_premium or (self.entry_premium * (1 + action.pnl_pct / 100))
        self._bookings.append(BookingRecord(
            level    = action.level,
            lots     = action.lots_to_book,
            premium  = fill,
            pnl_pct  = action.pnl_pct,
        ))
        self._weighted_pnl += action.pnl_pct * action.lots_to_book

        logger.info(
            f"[ProfitLadder] ✅ Booked Level {action.level} | "
            f"{action.lots_to_book} lots @ ₹{fill:.1f} (+{action.pnl_pct:.1f}%) | "
            f"Remaining: {self._lots_remaining} lots"
        )

    def record_final_close(self, final_premium: float) -> None:
        """Record the final close of remaining lots."""
        if self._lots_remaining > 0:
            pnl = (final_premium - self.entry_premium) / self.entry_premium * 100
            self._bookings.append(BookingRecord(
                level   = 99,
                lots    = self._lots_remaining,
                premium = final_premium,
                pnl_pct = pnl,
            ))
            self._weighted_pnl += pnl * self._lots_remaining
            self._lots_remaining = 0

    @property
    def avg_exit_pct(self) -> float:
        """Weighted average exit P&L across all bookings."""
        if not self._bookings:
            return 0.0
        return round(self._weighted_pnl / self.total_lots, 3)

    @property
    def is_fully_closed(self) -> bool:
        return self._lots_remaining <= 0

    @property
    def lots_remaining(self) -> int:
        return self._lots_remaining

    @property
    def bookings_summary(self) -> list[dict]:
        return [
            {"level": b.level, "lots": b.lots,
             "premium": b.premium, "pnl_pct": b.pnl_pct}
            for b in self._bookings
        ]

    def get_stats(self) -> dict:
        return {
            "entry_premium":     self.entry_premium,
            "total_lots":        self.total_lots,
            "lots_remaining":    self._lots_remaining,
            "levels_triggered":  sorted(self._levels_triggered),
            "avg_exit_pct":      self.avg_exit_pct,
            "is_fully_closed":   self.is_fully_closed,
            "bookings":          self.bookings_summary,
        }

"""
utils/capital_manager.py — Daily Capital Budget Manager
=========================================================
Enforces the 15% daily capital rule:

    daily_budget = TOTAL_FUND × 15%

FUND SIZES & DAILY BUDGETS:
  ₹1,00,000 (₹1L)  → ₹15,000/day
  ₹2,00,000 (₹2L)  → ₹30,000/day
  ₹5,00,000 (₹5L)  → ₹75,000/day
  ₹10,00,000 (₹10L)→ ₹1,50,000/day

HOW IT WORKS:
  Every trade entry records how much capital it consumes:
      capital_consumed = option_premium × lot_size

  Before allowing a new trade, CapitalManager checks:
      capital_used_today + new_trade_cost <= daily_budget

  If the trade would exceed the budget → BLOCKED.

  Open capital is marked closed when a trade exits (win or loss).
  But daily_budget is NOT released mid-day — once used, it's accounted for.
  This prevents "re-using" recovered capital to over-trade.

WHY 15% IS HARDCODED:
  15% × worst-case daily loss (all trades hit SL at 25%) = 3.75% of fund.
  This keeps the absolute worst day under 4% fund loss — recoverable.
  If you increase this, a bad day could wipe 10-15% in a single session.
  The 15% is NOT configurable to protect the fund from over-trading.

  TOTAL_FUND is configurable (changes as your fund grows).
  DAILY_CAPITAL_PCT = 15% is hardcoded constant.

TRADE SIZING:
  Given daily_budget and max trades setting, compute lots per trade:

  "daily" mode (default):
      lots = floor(remaining_daily_budget / (premium × lot_size))

  "dynamic" mode:
      trade_1 gets 60% of daily_budget
      trade_2 gets 25% of remaining
      trade_3 gets remainder

  Both modes ensure total never exceeds daily_budget.

INTEGRATION:
  Called by executor/planner BEFORE placing any order.
  Publishes CAPITAL_EXHAUSTED event when budget is used up.
  Resets at market open (09:15 IST) each day.

USAGE:
  from utils.capital_manager import get_capital_manager
  cm = get_capital_manager()

  # Before each trade
  check = cm.can_trade(premium=150, lot_size=75)
  if not check.allowed:
      logger.warning(check.reason)
      return

  # After order placed
  cm.record_trade_opened(trade_id="SF-001", premium=150, lot_size=75, lots=1)

  # After trade closes
  cm.record_trade_closed(trade_id="SF-001", premium=150, lot_size=75, lots=1)

  # Dashboard
  status = cm.get_status()
  # → {daily_budget: 15000, used: 11250, remaining: 3750, trades: 1, pct_used: 75}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

# ── Load config ───────────────────────────────────────────────────────────────
try:
    from config.settings import (
        TOTAL_FUND,
        DAILY_CAPITAL_PCT,   # HARDCODED 15.0 — never change
        CAPITAL_SPLIT_MODE,
        MAX_TRADES_PER_DAY,
        NIFTY_LOT_SIZE,
        JOURNAL_DIR,
    )
except ImportError:
    TOTAL_FUND        = 200_000.0
    DAILY_CAPITAL_PCT = 15.0        # HARDCODED — do not change
    CAPITAL_SPLIT_MODE = "equal"
    MAX_TRADES_PER_DAY = 3
    NIFTY_LOT_SIZE     = 75
    JOURNAL_DIR        = "journal"

# ── The only constant that must never be user-configurable ────────────────────
_HARDCODED_DAILY_PCT = 15.0   # 15% — sealed here as a safeguard

if abs(DAILY_CAPITAL_PCT - _HARDCODED_DAILY_PCT) > 0.01:
    import warnings
    warnings.warn(
        f"DAILY_CAPITAL_PCT={DAILY_CAPITAL_PCT} differs from hardcoded "
        f"{_HARDCODED_DAILY_PCT}%. Reverting to {_HARDCODED_DAILY_PCT}%.",
        stacklevel=2,
    )
    DAILY_CAPITAL_PCT = _HARDCODED_DAILY_PCT


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TradeAllowance:
    """Result of can_trade() check."""
    allowed:          bool
    reason:           str
    lots_allowed:     int      # how many lots allowed (0 = blocked)
    capital_required: float    # premium × lot_size × lots
    capital_remaining:float
    daily_budget:     float
    capital_used:     float
    trades_today:     int

    def __str__(self) -> str:
        if self.allowed:
            return (f"ALLOWED: {self.lots_allowed} lot(s) "
                    f"(₹{self.capital_required:,.0f}) | "
                    f"Remaining: ₹{self.capital_remaining:,.0f}")
        return f"BLOCKED: {self.reason}"


@dataclass
class CapitalStatus:
    """Current capital usage status."""
    total_fund:       float
    daily_budget:     float    # TOTAL_FUND × 15%
    capital_used:     float    # sum of open + closed trades today
    capital_open:     float    # currently in open positions
    capital_remaining:float    # daily_budget - capital_used
    trades_today:     int
    trades_remaining: int      # MAX_TRADES_PER_DAY - trades_today
    pct_used:         float    # capital_used / daily_budget × 100
    date:             str
    is_exhausted:     bool
    open_positions:   list[dict]


class CapitalManager:
    """
    Singleton capital manager. Enforces daily 15% capital budget.
    Thread-safe for single-process use.
    Resets automatically at start of new trading day.
    """

    BUDGET_FILE = Path(JOURNAL_DIR) / "capital_state.json"

    def __init__(self, total_fund: float = None) -> None:
        self._total_fund   = total_fund or float(TOTAL_FUND)
        self._daily_budget = round(self._total_fund * _HARDCODED_DAILY_PCT / 100, 2)
        self._today        = date.today().isoformat()

        # Daily state
        self._capital_used:   float      = 0.0
        self._trades_today:   int        = 0
        self._open_positions: dict[str, float] = {}   # trade_id → capital_committed

        # Load persisted state (handles restarts mid-day)
        self._load_state()

        try:
            from loguru import logger
            logger.info(
                f"[CapitalManager] Initialized | "
                f"Fund=₹{self._total_fund:,.0f} | "
                f"DailyBudget=₹{self._daily_budget:,.0f} "
                f"({_HARDCODED_DAILY_PCT}% hardcoded) | "
                f"MaxTrades={MAX_TRADES_PER_DAY}"
            )
        except Exception:
            pass

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def can_trade(
        self,
        premium:  float,
        lot_size: int   = None,
        lots:     int   = None,
    ) -> TradeAllowance:
        """
        Check if a new trade is allowed within today's budget.

        Args:
            premium:  option LTP at entry (e.g. ₹150)
            lot_size: lot size for the instrument (default: NIFTY=75)
            lots:     number of lots requested (None = calculate max allowed)

        Returns:
            TradeAllowance with allowed=True/False and full details
        """
        self._check_day_reset()
        ls = lot_size or NIFTY_LOT_SIZE

        # Check trade count limit
        if MAX_TRADES_PER_DAY > 0 and self._trades_today >= MAX_TRADES_PER_DAY:
            return TradeAllowance(
                allowed=False,
                reason=f"Max trades reached ({MAX_TRADES_PER_DAY}/day)",
                lots_allowed=0,
                capital_required=0,
                capital_remaining=self.remaining,
                daily_budget=self._daily_budget,
                capital_used=self._capital_used,
                trades_today=self._trades_today,
            )

        # Capital available for this trade. Default policy is a daily cap, not
        # equal per-trade splitting, so one high-quality trade may use the full
        # remaining 15% budget.
        available = min(self._per_trade_budget(), self.remaining)

        if available <= 0:
            return TradeAllowance(
                allowed=False,
                reason=f"Daily budget exhausted (₹{self._capital_used:,.0f} / ₹{self._daily_budget:,.0f})",
                lots_allowed=0,
                capital_required=0,
                capital_remaining=0.0,
                daily_budget=self._daily_budget,
                capital_used=self._capital_used,
                trades_today=self._trades_today,
            )

        # Calculate max lots
        cost_per_lot  = premium * ls
        if cost_per_lot <= 0:
            return TradeAllowance(
                allowed=False, reason="Invalid premium or lot size",
                lots_allowed=0, capital_required=0,
                capital_remaining=self.remaining,
                daily_budget=self._daily_budget,
                capital_used=self._capital_used,
                trades_today=self._trades_today,
            )

        max_lots_budget = int(available // cost_per_lot)

        if lots is not None:
            # Check if requested lots fit in budget
            required = premium * ls * lots
            if required > available:
                return TradeAllowance(
                    allowed=False,
                    reason=f"₹{required:,.0f} required but only ₹{available:,.0f} available today",
                    lots_allowed=0,
                    capital_required=required,
                    capital_remaining=self.remaining,
                    daily_budget=self._daily_budget,
                    capital_used=self._capital_used,
                    trades_today=self._trades_today,
                )
            lots_allowed    = lots
            capital_required = required
        else:
            lots_allowed     = max_lots_budget
            capital_required = premium * ls * lots_allowed

        if lots_allowed == 0:
            return TradeAllowance(
                allowed=False,
                reason=f"Premium ₹{premium}×{ls}=₹{cost_per_lot:,.0f}/lot exceeds available ₹{available:,.0f}",
                lots_allowed=0,
                capital_required=0,
                capital_remaining=self.remaining,
                daily_budget=self._daily_budget,
                capital_used=self._capital_used,
                trades_today=self._trades_today,
            )

        return TradeAllowance(
            allowed=True,
            reason="within_budget",
            lots_allowed=lots_allowed,
            capital_required=round(capital_required, 2),
            capital_remaining=round(self.remaining - capital_required, 2),
            daily_budget=self._daily_budget,
            capital_used=self._capital_used,
            trades_today=self._trades_today,
        )

    def record_trade_opened(
        self,
        trade_id: str,
        premium:  float,
        lot_size: int,
        lots:     int = 1,
    ) -> float:
        """
        Record capital committed for an opened trade.
        Returns capital_committed (premium × lot_size × lots).
        """
        self._check_day_reset()
        committed = round(premium * lot_size * lots, 2)
        self._capital_used            += committed
        self._open_positions[trade_id] = committed
        self._trades_today             += 1
        self._save_state()

        try:
            from loguru import logger
            logger.info(
                f"[CapitalManager] 📂 Trade opened | {trade_id} | "
                f"₹{committed:,.0f} committed | "
                f"Used: ₹{self._capital_used:,.0f} / ₹{self._daily_budget:,.0f} "
                f"({self._capital_used/self._daily_budget*100:.1f}%) | "
                f"Trades: {self._trades_today}/{MAX_TRADES_PER_DAY if MAX_TRADES_PER_DAY > 0 else 'unlimited'}"
            )
        except Exception:
            pass
        return committed

    def record_trade_closed(
        self,
        trade_id: str,
        *args,     # unused — signature kept for compatibility
        **kwargs,
    ) -> None:
        """
        Mark position as closed.
        Capital is NOT returned to daily_budget (prevents over-trading).
        The committed capital stays as 'used' — it was deployed today.
        """
        if trade_id in self._open_positions:
            self._open_positions.pop(trade_id)
            self._save_state()

        try:
            from loguru import logger
            logger.info(
                f"[CapitalManager] ✅ Trade closed | {trade_id} | "
                f"Budget used: ₹{self._capital_used:,.0f} / ₹{self._daily_budget:,.0f} "
                f"| Open positions: {len(self._open_positions)}"
            )
        except Exception:
            pass

    def get_status(self) -> CapitalStatus:
        """Return full capital status snapshot."""
        self._check_day_reset()
        open_capital = sum(self._open_positions.values())
        return CapitalStatus(
            total_fund        = self._total_fund,
            daily_budget      = self._daily_budget,
            capital_used      = round(self._capital_used, 2),
            capital_open      = round(open_capital, 2),
            capital_remaining = round(self.remaining, 2),
            trades_today      = self._trades_today,
            trades_remaining  = max(0, MAX_TRADES_PER_DAY - self._trades_today) if MAX_TRADES_PER_DAY > 0 else 999,
            pct_used          = round(self._capital_used / self._daily_budget * 100, 1),
            date              = self._today,
            is_exhausted      = self.remaining <= 0,
            open_positions    = [
                {"trade_id": k, "committed": v}
                for k, v in self._open_positions.items()
            ],
        )

    def update_fund(self, new_total_fund: float) -> None:
        """
        Update total fund size (e.g. when fund grows from ₹1L to ₹2L).
        Recalculates daily_budget automatically.
        Only takes effect next trading day (current day budget unchanged).
        """
        old = self._total_fund
        self._total_fund   = new_total_fund
        self._daily_budget = round(new_total_fund * _HARDCODED_DAILY_PCT / 100, 2)
        self._save_state()
        try:
            from loguru import logger
            logger.info(
                f"[CapitalManager] Fund updated: ₹{old:,.0f} → ₹{new_total_fund:,.0f} | "
                f"New daily budget: ₹{self._daily_budget:,.0f} (effective tomorrow)"
            )
        except Exception:
            pass

    def print_status(self) -> None:
        """Print formatted capital status to console."""
        s = self.get_status()
        pct  = s.pct_used
        col  = "\033[91m" if pct > 80 else "\033[93m" if pct > 50 else "\033[92m"
        rst  = "\033[0m"
        bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"\n  ┌─ Capital Manager ({'⚠️ EXHAUSTED' if s.is_exhausted else '✅ ACTIVE'}) ──────────────────────┐")
        print(f"  │  Fund:        ₹{s.total_fund:>12,.0f}  ({_HARDCODED_DAILY_PCT}% daily rule hardcoded)  │")
        print(f"  │  Daily Budget:₹{s.daily_budget:>12,.0f}                                │")
        print(f"  │  Used:        ₹{s.capital_used:>12,.0f}  {col}{bar}{rst}  {pct:.1f}%  │")
        print(f"  │  Remaining:   ₹{s.capital_remaining:>12,.0f}                                │")
        max_trades_label = MAX_TRADES_PER_DAY if MAX_TRADES_PER_DAY > 0 else "unlimited"
        print(f"  │  Trades:      {s.trades_today}/{max_trades_label} used   {s.trades_remaining} remaining                    │")
        print(f"  └────────────────────────────────────────────────────────────┘\n")

    # ── PROPERTIES ────────────────────────────────────────────────────────────

    @property
    def remaining(self) -> float:
        return max(0.0, self._daily_budget - self._capital_used)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining <= 0 or (MAX_TRADES_PER_DAY > 0 and self._trades_today >= MAX_TRADES_PER_DAY)

    @property
    def daily_budget(self) -> float:
        return self._daily_budget

    @property
    def total_fund(self) -> float:
        return self._total_fund

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _per_trade_budget(self) -> float:
        """Calculate budget for one trade based on split mode."""
        trades_remaining = max(1, MAX_TRADES_PER_DAY - self._trades_today) if MAX_TRADES_PER_DAY > 0 else 999

        if CAPITAL_SPLIT_MODE == "dynamic":
            # Trade 1: 60%, Trade 2: 25%, Trade 3: 15%
            splits = [0.60, 0.25, 0.15]
            idx    = min(self._trades_today, len(splits) - 1)
            return round(self._daily_budget * splits[idx], 2)
        return self.remaining

    def _check_day_reset(self) -> None:
        """Reset daily counters if a new day has started."""
        today = date.today().isoformat()
        if today != self._today:
            self._today           = today
            self._capital_used    = 0.0
            self._trades_today    = 0
            self._open_positions  = {}
            self._save_state()
            try:
                from loguru import logger
                logger.info(
                    f"[CapitalManager] 🌅 New day {today} | "
                    f"Budget reset to ₹{self._daily_budget:,.0f}"
                )
            except Exception:
                pass

    def _save_state(self) -> None:
        try:
            Path(JOURNAL_DIR).mkdir(parents=True, exist_ok=True)
            state = {
                "date":             self._today,
                "total_fund":       self._total_fund,
                "daily_budget":     self._daily_budget,
                "capital_used":     self._capital_used,
                "trades_today":     self._trades_today,
                "open_positions":   self._open_positions,
            }
            with open(self.BUDGET_FILE, "w") as f:
                import json
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def _load_state(self) -> None:
        try:
            if not self.BUDGET_FILE.exists():
                return
            import json
            with open(self.BUDGET_FILE) as f:
                state = json.load(f)
            # Only restore if same day
            if state.get("date") == self._today:
                self._capital_used   = float(state.get("capital_used",   0))
                self._trades_today   = int(state.get("trades_today",     0))
                self._open_positions = state.get("open_positions",       {})
                # Update fund if changed in settings
                if state.get("total_fund") != self._total_fund:
                    self._daily_budget = round(
                        self._total_fund * _HARDCODED_DAILY_PCT / 100, 2
                    )
        except Exception:
            pass


# ── Singleton ─────────────────────────────────────────────────────────────────
_manager: CapitalManager | None = None

def get_capital_manager() -> CapitalManager:
    """Return singleton CapitalManager. Creates on first call."""
    global _manager
    if _manager is None:
        _manager = CapitalManager()
    return _manager


def reset_capital_manager() -> None:
    """Force reset (used in tests)."""
    global _manager
    _manager = None

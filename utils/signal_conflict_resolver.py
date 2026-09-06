"""
utils/signal_conflict_resolver.py — Real-Time Signal Suppression
=================================================================
Prevents cascading losses by suppressing signals when recent
performance is bad. Two mechanisms:

1. DIRECTIONAL CONFLICT RESOLVER
   If last 3 trades in PUT direction all lost → pause PUT signals 30 min.
   If last 3 trades in CALL direction all lost → pause CALL signals 30 min.
   Market may have shifted against that direction today.

2. TIME-OF-DAY PERFORMANCE FILTER
   Tracks win rate per 30-min time bucket per strategy.
   If bucket historically wins < 35% → reduce position size 50%.
   Example: BBSqueeze fires at 13:30 and historically wins only 28%
   at that time → half size or skip.

3. STRATEGY SUPPRESSION
   If strategy X has < 30% win rate on last 5 live signals →
   suppress that strategy for rest of day.
   Resets at market open next morning.

USAGE:
    resolver = get_resolver()

    # Before processing signal
    check = resolver.check_signal(direction="BUY_PUT", strategies=["SMC","ADX+PSAR"])
    if check.blocked:
        return  # suppress this signal
    if check.size_multiplier < 1.0:
        lots = max(1, int(lots * check.size_multiplier))

    # After trade closes
    resolver.record_outcome(direction="BUY_PUT", pnl_pct=-8.5,
                             strategies=["SMC","ADX+PSAR"], time="13:30")
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import (
        CONFLICT_RESOLVER_ENABLED, CONFLICT_LOOKBACK_TRADES,
        CONFLICT_MAX_LOSS_STREAK, CONFLICT_PAUSE_MINUTES,
        CONFLICT_STRATEGY_LOOKBACK, CONFLICT_STRATEGY_MIN_WR,
        ENABLE_TOD_FILTER, TOD_MIN_TRADES, TOD_WEAK_WIN_RATE,
        JOURNAL_DIR,
    )
except ImportError:
    CONFLICT_RESOLVER_ENABLED  = True
    CONFLICT_LOOKBACK_TRADES   = 5
    CONFLICT_MAX_LOSS_STREAK   = 3
    CONFLICT_PAUSE_MINUTES     = 30
    CONFLICT_STRATEGY_LOOKBACK = 5
    CONFLICT_STRATEGY_MIN_WR   = 0.30
    ENABLE_TOD_FILTER          = True
    TOD_MIN_TRADES             = 10
    TOD_WEAK_WIN_RATE          = 0.35
    JOURNAL_DIR                = "journal"

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ConflictCheck:
    allowed:         bool
    blocked:         bool
    size_multiplier: float     # 1.0 = normal, 0.5 = half size, 0.0 = blocked
    block_reason:    str
    warnings:        list[str]


class SignalConflictResolver:
    """
    Real-time signal quality gatekeeper.
    Tracks recent performance and suppresses underperforming signals.
    State persists intraday, resets at market open.
    """

    STATE_FILE = Path(JOURNAL_DIR) / "conflict_state.json"

    def __init__(self, *, persist_state: bool | None = None) -> None:
        if persist_state is None:
            persist_state = not _is_backtest_mode()
        self._persist_state = bool(persist_state)
        self._backtest_mode = not self._persist_state
        self._today        = date.today().isoformat()
        # Directional loss streaks: {"BUY_CALL": 0, "BUY_PUT": 0}
        self._loss_streak: dict[str, int]      = defaultdict(int)
        self._paused_until: dict[str, Optional[datetime]] = {}
        # Per-strategy recent outcomes: {strategy: deque of (win/loss, time)}
        self._strat_outcomes: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        # Time-of-day stats: {(strategy, bucket): [outcomes]}
        self._tod_stats: dict[tuple, list]     = defaultdict(list)
        # All recent trades for streak tracking
        self._recent_trades: deque             = deque(maxlen=20)
        if self._persist_state:
            self._load_state()

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def check_signal(
        self,
        direction:  str,
        strategies: list[str],
        candle_time: Optional[str] = None,
        current_time: Optional[datetime] = None,
    ) -> ConflictCheck:
        """
        Check if this signal should be allowed, reduced, or blocked.

        Returns ConflictCheck with:
          allowed=True  → trade normally
          allowed=True, size_multiplier=0.5 → trade at half size
          allowed=False → block entirely
        """
        now_time = current_time if current_time else datetime.now(IST)
        self._check_day_reset(now_time)

        if not CONFLICT_RESOLVER_ENABLED:
            return ConflictCheck(True, False, 1.0, "", [])

        warnings   = []
        multiplier = 1.0

        # ── Check 1: Directional pause ────────────────────────────────────────
        pause_until = self._paused_until.get(direction)
        if pause_until and now_time < pause_until:
            remaining = int((pause_until - now_time).total_seconds() / 60)
            return ConflictCheck(
                allowed=False, blocked=True, size_multiplier=0.0,
                block_reason=f"{direction} paused for {remaining}min "
                             f"(consecutive loss streak={CONFLICT_MAX_LOSS_STREAK})",
                warnings=[],
            )

        # ── Check 2: Loss streak ──────────────────────────────────────────────
        streak = self._loss_streak.get(direction, 0)
        if streak >= CONFLICT_MAX_LOSS_STREAK:
            # Trigger pause
            from datetime import timedelta
            pause_until = now_time + timedelta(minutes=CONFLICT_PAUSE_MINUTES)
            self._paused_until[direction] = pause_until
            return ConflictCheck(
                allowed=False, blocked=True, size_multiplier=0.0,
                block_reason=f"{direction} loss streak={streak} → "
                             f"paused {CONFLICT_PAUSE_MINUTES}min",
                warnings=[],
            )

        # ── Check 3: Strategy suppression ────────────────────────────────────
        for strat in strategies:
            outcomes = list(self._strat_outcomes.get(strat, []))
            if len(outcomes) >= CONFLICT_STRATEGY_LOOKBACK:
                recent = outcomes[-CONFLICT_STRATEGY_LOOKBACK:]
                wr     = sum(1 for o in recent if o > 0) / len(recent)
                if wr < CONFLICT_STRATEGY_MIN_WR:
                    warnings.append(
                        f"{strat} suppressed (WR={wr:.0%} last {len(recent)} signals)"
                    )
                    multiplier = min(multiplier, 0.0)   # block if lead strategy is bad
                elif wr < 0.45:
                    warnings.append(f"{strat} weak (WR={wr:.0%})")
                    multiplier = min(multiplier, 0.5)

        # ── Check 4: Time-of-day performance ─────────────────────────────────
        if ENABLE_TOD_FILTER and candle_time:
            bucket = self._get_tod_bucket(candle_time)
            for strat in strategies:
                key     = (strat, bucket)
                history = self._tod_stats.get(key, [])
                if len(history) >= TOD_MIN_TRADES:
                    tod_wr = sum(1 for o in history if o > 0) / len(history)
                    if tod_wr < TOD_WEAK_WIN_RATE:
                        warnings.append(
                            f"{strat} at {bucket} historically weak "
                            f"(WR={tod_wr:.0%} over {len(history)} trades)"
                        )
                        multiplier = min(multiplier, 0.5)

        if multiplier == 0.0:
            return ConflictCheck(
                allowed=False, blocked=True, size_multiplier=0.0,
                block_reason="; ".join(warnings),
                warnings=warnings,
            )

        return ConflictCheck(
            allowed=True, blocked=False,
            size_multiplier=multiplier,
            block_reason="",
            warnings=warnings,
        )

    def record_outcome(
        self,
        direction:   str,
        pnl_pct:     float,
        strategies:  list[str],
        candle_time: str = "",
        current_time: Optional[datetime] = None,
    ) -> None:
        """Record trade outcome. Call after every trade close."""
        now_time = current_time if current_time else datetime.now(IST)
        self._check_day_reset(now_time)
        win = pnl_pct > 0

        # Update directional streak
        if win:
            self._loss_streak[direction] = 0
        else:
            self._loss_streak[direction] = self._loss_streak.get(direction, 0) + 1

        # Update per-strategy outcomes
        for strat in strategies:
            self._strat_outcomes[strat].append(pnl_pct)

        # Update time-of-day stats
        if candle_time:
            bucket = self._get_tod_bucket(candle_time)
            for strat in strategies:
                self._tod_stats[(strat, bucket)].append(pnl_pct)

        self._recent_trades.append({
            "direction": direction, "pnl": pnl_pct,
            "strategies": strategies, "time": candle_time,
        })

        streak = self._loss_streak.get(direction, 0)
        logger.info(
            f"[ConflictResolver] {'✅' if win else '❌'} {direction} {pnl_pct:+.1f}% | "
            f"Loss streak: {streak}/{CONFLICT_MAX_LOSS_STREAK}"
        )
        self._save_state()

    def get_status(self) -> dict:
        """Return current resolver state for dashboard."""
        return {
            "loss_streaks":    dict(self._loss_streak),
            "paused_directions": {
                k: v.isoformat() if v else None
                for k, v in self._paused_until.items()
            },
            "strategy_recent_wr": {
                s: round(sum(1 for o in list(q) if o > 0) / max(len(q), 1), 3)
                for s, q in self._strat_outcomes.items() if q
            },
        }

    def reset_day(self, today_str: Optional[str] = None) -> None:
        """Manual reset — called at market open."""
        self._loss_streak    = defaultdict(int)
        self._paused_until   = {}
        self._recent_trades  = deque(maxlen=20)
        self._today          = today_str if today_str else date.today().isoformat()
        self._save_state()
        logger.info(f"[ConflictResolver] Day reset ✅ ({self._today})")

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _check_day_reset(self, current_time: datetime) -> None:
        today_str = current_time.date().isoformat()
        if today_str != self._today:
            self.reset_day(today_str)

    @staticmethod
    def _get_tod_bucket(time_str: str) -> str:
        """Map time string to 30-min bucket label."""
        buckets = [
            "09:15", "09:45", "10:15", "10:45",
            "11:15", "11:45", "12:15", "12:45",
            "13:15", "13:45", "14:00",
        ]
        try:
            h, m = int(time_str[:2]), int(time_str[3:5])
            mins = h * 60 + m
            for b in reversed(buckets):
                bh, bm = int(b[:2]), int(b[3:])
                if mins >= bh * 60 + bm:
                    return b
        except Exception:
            pass
        return "09:15"

    def _save_state(self) -> None:
        if not self._persist_state:
            return
        try:
            Path(JOURNAL_DIR).mkdir(exist_ok=True)
            state = {
                "date":          self._today,
                "loss_streak":   dict(self._loss_streak),
                "paused_until":  {k: v.isoformat() if v else None
                                  for k, v in self._paused_until.items()},
            }
            with open(self.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def _load_state(self) -> None:
        if not self._persist_state:
            return
        try:
            if not self.STATE_FILE.exists():
                return
            with open(self.STATE_FILE) as f:
                state = json.load(f)
            if state.get("date") == self._today:
                self._loss_streak = defaultdict(int, state.get("loss_streak", {}))
                for k, v in state.get("paused_until", {}).items():
                    if v:
                        from datetime import datetime
                        try:
                            self._paused_until[k] = datetime.fromisoformat(v).replace(tzinfo=IST)
                        except Exception:
                            pass
        except Exception:
            pass


# ── Singleton ─────────────────────────────────────────────────────────────────
_resolver: SignalConflictResolver | None = None


def _is_backtest_mode() -> bool:
    return os.getenv("TRADING_MODE", "").strip().upper() == "BACKTEST"


def get_resolver() -> SignalConflictResolver:
    global _resolver
    backtest_mode = _is_backtest_mode()
    if _resolver is None or bool(getattr(_resolver, "_backtest_mode", False)) != backtest_mode:
        _resolver = SignalConflictResolver(persist_state=not backtest_mode)
    return _resolver


def reset_resolver(*, backtest_mode: bool | None = None) -> SignalConflictResolver:
    """Replace singleton state. Backtests use clean in-memory state."""
    global _resolver
    if backtest_mode is None:
        backtest_mode = _is_backtest_mode()
    _resolver = SignalConflictResolver(persist_state=not bool(backtest_mode))
    return _resolver

"""
agents_code/agent12_lifecycle/lifecycle_auditor.py — Agent 12: Trade Lifecycle Auditor
========================================================================================
PURPOSE:
  For every CLOSED trade, continue watching NIFTY/premium for N candles AFTER exit.
  Answer: "Did we exit too early? Did we exit too late? Was the exit reason correct?"

  Example questions this agent answers:
    "We exited at SL_HIT for -25%. Did premium recover to profit within 6 candles?
     If yes → SL was too tight, or structural SL level was wrong."

    "We exited at STALE_EXIT (flat for 8 candles). Did the trade explode to +40%
     in the next 4 candles? If yes → stale exit was premature, theta wasn't the issue."

    "We exited at TARGET_HIT for +70%. Did premium continue to +120%?
     If yes → target was too conservative, leaving money on the table."

    "TSL exited at +2% (floor breach). Was peak +54% and exit +62%? 
     → GOOD exit, TSL worked as designed (this is the reference case from user)."

THIS AGENT NEVER:
  - Changes settings
  - Re-enters trades
  - Publishes ORDER_* or TRADE_PLAN_* topics
  - Affects the live position manager in any way

WHAT IT PRODUCES:
  Per-trade "exit quality score":
    PREMATURE_LOSS_EXIT:  exited at loss, but would have recovered to profit
    PREMATURE_PROFIT_EXIT: exited at small profit, but much bigger profit was available
    GOOD_EXIT:            exit was well-timed (within 80% of subsequent peak/trough)
    LATE_EXIT:            held too long after peak, gave back significant profit
    CORRECT_SL:           SL was hit and price continued adversely (SL was right)

  Aggregated daily report:
    "Out of 8 SL_HIT exits today: 5 were CORRECT_SL, 3 were PREMATURE_LOSS_EXIT
     (would have recovered to avg +8% within 6 candles).
     
     Out of 3 TSL exits: all 3 were GOOD_EXIT (within 15% of peak).
     
     Recommendation: 3/8 SL hits recovering suggests SL may be too tight on
     [STRATEGY NAMES]. Consider reviewing structural SL source for these trades."

HOW IT WORKS:
  1. Subscribes to POSITION_CLOSED (every trade exit)
  2. Subscribes to CANDLES_READY (to track forward price action)
  3. For each closed trade, tracks NIFTY spot for LIFECYCLE_TRACK_CANDLES candles
  4. Estimates what premium WOULD have been at each forward candle (delta proxy)
  5. Classifies the exit quality
  6. Aggregates into daily report with strategy-level breakdown

INTEGRATION WITH AGENT 11:
  Agent 11 (Shadow) answers: "should we change the THRESHOLD?"
  Agent 12 (Lifecycle) answers: "was THIS SPECIFIC EXIT correct?"
  Together they give a complete picture: entry-side tuning (11) +
  exit-side tuning (12).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.bus import Topic
except ImportError:
    class Topic:
        POSITION_CLOSED = "POSITION_CLOSED"
        CANDLES_READY   = "CANDLES_READY"

try:
    from config.settings import JOURNAL_DIR
except ImportError:
    JOURNAL_DIR = "journal"

# ── Parameters ────────────────────────────────────────────────────────────────
LIFECYCLE_TRACK_CANDLES = 8     # candles to watch after exit
PREMATURE_LOSS_RECOVERY = 5.0   # if recovered to >= +5% it was premature
PREMATURE_PROFIT_GAP    = 15.0  # if subsequent peak > exit + 15%, profit exit was premature
GOOD_EXIT_TOLERANCE     = 0.80  # exit within 80% of subsequent peak/trough = good
LATE_EXIT_GIVEBACK      = 10.0  # if gave back > 10% from peak before exit = late


@dataclass
class TrackedExit:
    """A closed trade being tracked for post-exit price action."""
    timestamp:        str
    option_symbol:    str
    direction:        str
    exit_reason:      str
    entry_premium:    float
    exit_premium:     float
    pnl_pct:          float
    peak_pnl_pct:     float       # peak P&L reached DURING the trade
    nifty_at_exit:    float
    strategies_fired: list[str]
    delta_est:        float = 0.45
    candles_tracked:  int   = 0
    forward_pnls:     list[float] = field(default_factory=list)
    resolved:         bool  = False
    classification:   str   = ""


@dataclass
class ExitQualityStats:
    correct_sl:           int = 0
    premature_loss_exit:  int = 0
    good_exit:            int = 0
    premature_profit_exit:int = 0
    late_exit:            int = 0
    total:                int = 0
    avg_recovery_missed:  float = 0.0   # avg % missed on premature exits


class TradeLifecycleAuditor:
    """
    Agent 12: Trade Lifecycle Auditor.
    Tracks post-exit price action to evaluate exit quality.
    Pure observer — produces recommendations only.
    """

    NAME = "TradeLifecycleAuditor"

    def __init__(self, bus=None) -> None:
        self.bus = bus
        self._tracked:  list[TrackedExit] = []
        self._resolved: list[TrackedExit] = []
        self._today     = date.today().isoformat()
        self._log_path  = Path(JOURNAL_DIR) / f"lifecycle_audit_{self._today}.json"

    async def register(self) -> None:
        """Subscribe to relevant topics. Call once at startup."""
        if self.bus is None:
            return
        self.bus.subscribe(Topic.POSITION_CLOSED, self.on_position_closed)
        self.bus.subscribe(Topic.CANDLES_READY,   self.on_candle)
        logger.info(f"[{self.NAME}] Registered — tracking exit quality "
                    f"for {LIFECYCLE_TRACK_CANDLES} candles post-exit")

    # ── EVENT HANDLERS ────────────────────────────────────────────────────────

    async def on_position_closed(self, msg) -> None:
        """A trade just closed — start tracking post-exit price action."""
        try:
            p = msg.payload
            exit_ = TrackedExit(
                timestamp        = datetime.now(IST).isoformat(),
                option_symbol    = p.get("option_symbol", ""),
                direction        = p.get("direction", "BUY_CALL"),
                exit_reason      = p.get("exit_reason", "UNKNOWN"),
                entry_premium    = float(p.get("entry_premium", 0)),
                exit_premium     = float(p.get("exit_premium", 0)),
                pnl_pct          = float(p.get("pnl_pct", 0)),
                peak_pnl_pct     = float(p.get("peak_pnl_pct", p.get("pnl_pct", 0))),
                nifty_at_exit    = float(p.get("nifty_at_exit", 0)),
                strategies_fired = p.get("strategies_fired", []),
                delta_est        = float(p.get("delta_est", 0.45)),
            )
            if exit_.nifty_at_exit > 0 and exit_.entry_premium > 0:
                self._tracked.append(exit_)
                logger.debug(
                    f"[{self.NAME}] Tracking exit: {exit_.option_symbol} "
                    f"{exit_.exit_reason} pnl={exit_.pnl_pct:+.1f}%"
                )
        except Exception as e:
            logger.debug(f"[{self.NAME}] on_position_closed error: {e}")

    async def on_candle(self, msg) -> None:
        """Track forward price action for all pending exits."""
        try:
            spot = float(msg.payload.get("ltp", 0))
            if spot <= 0:
                return

            still_tracking = []
            for ex in self._tracked:
                # NIFTY move since exit
                spot_move_pct = (spot - ex.nifty_at_exit) / ex.nifty_at_exit * 100
                if ex.direction == "BUY_PUT":
                    spot_move_pct = -spot_move_pct

                # Estimate option premium % change using delta proxy
                leverage = ex.nifty_at_exit / max(ex.exit_premium, 1) * ex.delta_est
                option_pnl_from_exit = spot_move_pct * leverage

                ex.forward_pnls.append(round(option_pnl_from_exit, 2))
                ex.candles_tracked += 1

                if ex.candles_tracked >= LIFECYCLE_TRACK_CANDLES:
                    self._classify_exit(ex)
                    ex.resolved = True
                    self._resolved.append(ex)
                else:
                    still_tracking.append(ex)

            self._tracked = still_tracking

            if len(self._resolved) > 0 and len(self._resolved) % 3 == 0:
                self._save_log()

        except Exception as e:
            logger.debug(f"[{self.NAME}] on_candle error: {e}")

    # ── CLASSIFICATION ────────────────────────────────────────────────────────

    def _classify_exit(self, ex: TrackedExit) -> None:
        """Classify exit quality based on post-exit price action."""
        if not ex.forward_pnls:
            ex.classification = "UNKNOWN"
            return

        post_peak  = max(ex.forward_pnls)
        post_trough= min(ex.forward_pnls)
        final_pct  = ex.forward_pnls[-1]

        if ex.exit_reason in ("SL_HIT", "STRUCTURAL_SL"):
            # Was the exit correct? Did price continue adversely or recover?
            if post_peak >= PREMATURE_LOSS_RECOVERY:
                ex.classification = "PREMATURE_LOSS_EXIT"
            else:
                ex.classification = "CORRECT_SL"

        elif ex.exit_reason in ("TARGET_HIT", "TSL_EXIT", "PROFIT_LADDER"):
            # Did we leave significant profit on the table?
            if post_peak >= ex.pnl_pct + PREMATURE_PROFIT_GAP:
                ex.classification = "PREMATURE_PROFIT_EXIT"
            elif post_trough <= -(GOOD_EXIT_TOLERANCE * abs(ex.pnl_pct)) and ex.pnl_pct > 0:
                ex.classification = "GOOD_EXIT"   # exited before giveback
            else:
                ex.classification = "GOOD_EXIT"

        elif ex.exit_reason in ("STALE_EXIT", "TIME_PROFIT_EXIT"):
            if post_peak >= PREMATURE_LOSS_RECOVERY + 10:
                ex.classification = "PREMATURE_PROFIT_EXIT"  # the move came after all
            else:
                ex.classification = "GOOD_EXIT"   # correctly identified dead trade

        elif ex.exit_reason in ("SIGNAL_EXIT", "BREAKEVEN_STOP", "EARLY_LOCK"):
            # Check if peak was already captured (these are profit-protection exits)
            giveback = ex.peak_pnl_pct - ex.pnl_pct
            if giveback > LATE_EXIT_GIVEBACK:
                ex.classification = "LATE_EXIT"
            else:
                ex.classification = "GOOD_EXIT"
        else:
            ex.classification = "GOOD_EXIT"

        logger.info(
            f"[{self.NAME}] {ex.option_symbol} {ex.exit_reason} "
            f"exit_pnl={ex.pnl_pct:+.1f}% → {ex.classification} "
            f"(post-exit peak={post_peak:+.1f}% trough={post_trough:+.1f}%)"
        )

    # ── REPORTING ─────────────────────────────────────────────────────────────

    def get_report(self) -> dict:
        """Generate daily exit-quality report with strategy-level breakdown."""
        stats = ExitQualityStats()
        by_strategy: dict[str, ExitQualityStats] = defaultdict(ExitQualityStats)
        missed_pcts = []

        for ex in self._resolved:
            stats.total += 1
            cls = ex.classification

            if cls == "CORRECT_SL":
                stats.correct_sl += 1
            elif cls == "PREMATURE_LOSS_EXIT":
                stats.premature_loss_exit += 1
                missed_pcts.append(max(ex.forward_pnls) - ex.pnl_pct)
            elif cls == "GOOD_EXIT":
                stats.good_exit += 1
            elif cls == "PREMATURE_PROFIT_EXIT":
                stats.premature_profit_exit += 1
                missed_pcts.append(max(ex.forward_pnls) - ex.pnl_pct)
            elif cls == "LATE_EXIT":
                stats.late_exit += 1

            for strat in ex.strategies_fired:
                sstats = by_strategy[strat]
                sstats.total += 1
                if cls == "CORRECT_SL":            sstats.correct_sl += 1
                elif cls == "PREMATURE_LOSS_EXIT":  sstats.premature_loss_exit += 1
                elif cls == "GOOD_EXIT":            sstats.good_exit += 1
                elif cls == "PREMATURE_PROFIT_EXIT":sstats.premature_profit_exit += 1
                elif cls == "LATE_EXIT":            sstats.late_exit += 1

        stats.avg_recovery_missed = (
            round(sum(missed_pcts) / len(missed_pcts), 2) if missed_pcts else 0.0
        )

        # Build recommendations
        recommendations = []

        if stats.total >= 5:
            premature_loss_rate = stats.premature_loss_exit / max(stats.correct_sl + stats.premature_loss_exit, 1)
            if premature_loss_rate > 0.35:
                recommendations.append(
                    f"{stats.premature_loss_exit}/{stats.correct_sl + stats.premature_loss_exit} "
                    f"SL exits recovered to profit (avg missed +{stats.avg_recovery_missed:.1f}%). "
                    f"SL may be too tight — review structural SL distances or ATR multiplier. "
                    f"REQUIRES YOUR REVIEW."
                )

            if stats.late_exit >= 3:
                recommendations.append(
                    f"{stats.late_exit} exits gave back >{LATE_EXIT_GIVEBACK}% from peak "
                    f"before exiting. Consider tightening EARLY_LOCK_TRAIL_PCT. "
                    f"REQUIRES YOUR REVIEW."
                )

            if stats.premature_profit_exit >= 3:
                recommendations.append(
                    f"{stats.premature_profit_exit} profit exits left significant gains "
                    f"on the table (avg missed +{stats.avg_recovery_missed:.1f}%). "
                    f"Consider raising TARGET_PCT or widening profit ladder levels. "
                    f"REQUIRES YOUR REVIEW."
                )

        # Per-strategy worst offenders
        strategy_issues = {}
        for strat, sstats in by_strategy.items():
            if sstats.total >= 3:
                bad_rate = (sstats.premature_loss_exit + sstats.late_exit) / sstats.total
                if bad_rate > 0.4:
                    strategy_issues[strat] = {
                        "total": sstats.total,
                        "premature_loss": sstats.premature_loss_exit,
                        "late_exit": sstats.late_exit,
                        "bad_rate": round(bad_rate, 2),
                    }

        return {
            "date":            self._today,
            "total_resolved":  stats.total,
            "pending":         len(self._tracked),
            "summary": {
                "correct_sl":            stats.correct_sl,
                "premature_loss_exit":   stats.premature_loss_exit,
                "good_exit":             stats.good_exit,
                "premature_profit_exit": stats.premature_profit_exit,
                "late_exit":             stats.late_exit,
                "avg_recovery_missed_pct": stats.avg_recovery_missed,
            },
            "strategy_issues":  strategy_issues,
            "recommendations":  recommendations or ["No significant issues detected yet."],
        }

    def get_trade_detail(self, option_symbol: str) -> Optional[dict]:
        """Get detailed lifecycle for a specific trade (for debugging)."""
        for ex in self._resolved:
            if ex.option_symbol == option_symbol:
                return asdict(ex)
        return None

    def _save_log(self) -> None:
        try:
            Path(JOURNAL_DIR).mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "w") as f:
                json.dump(self.get_report(), f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[{self.NAME}] save_log error: {e}")


from typing import Optional

# ── Singleton ─────────────────────────────────────────────────────────────────
_lifecycle_agent: TradeLifecycleAuditor | None = None

def get_lifecycle_agent(bus=None) -> TradeLifecycleAuditor:
    global _lifecycle_agent
    if _lifecycle_agent is None:
        _lifecycle_agent = TradeLifecycleAuditor(bus)
    return _lifecycle_agent

"""
agents_code/agent11_shadow/shadow_agent.py — Agent 11: Shadow Parameter Agent
==============================================================================
PURPOSE:
  Run "what if?" parameter variants alongside the LIVE system, in real-time,
  WITHOUT affecting any actual trading decisions.

  Example question this agent answers continuously:
    "If ML_MIN_CONFIDENCE was 0.40 instead of 0.45, would we get more trades?
     Would those EXTRA trades have been profitable?"

  This agent NEVER changes settings, NEVER places orders, NEVER blocks signals.
  It is a pure OBSERVER that logs counterfactual outcomes for human review.

WHY THIS MATTERS:
  You can't know if loosening ML_MIN_CONFIDENCE from 0.45→0.40 is good
  without either (a) waiting weeks for live data at the new threshold,
  or (b) running this shadow agent which evaluates BOTH thresholds
  on the SAME live candles simultaneously.

  After 2-4 weeks, you get a report:
    "At threshold 0.40: +12 extra trades, 7 wins / 5 losses, net +₹4,200
     At threshold 0.50: -8 fewer trades, would have avoided 6 losses, net +₹1,100"

  YOU decide whether to change the setting. The agent only informs.

HOW IT WORKS:
  1. Subscribes to SIGNAL_REJECTED (signals the ML filter blocked)
  2. Subscribes to SIGNAL_APPROVED (signals that were approved and traded)
  3. Subscribes to POSITION_CLOSED (actual trade outcomes)
  4. For each REJECTED signal, checks: would it pass under variant thresholds?
  5. For signals that WOULD pass under a variant, tracks forward NIFTY/premium
     movement for N candles to estimate counterfactual P&L
  6. Aggregates into a daily/weekly report

PARAMETER VARIANTS TRACKED (configurable):
  ML_MIN_CONFIDENCE:  current ± [0.04, 0.08, -0.04, -0.08]
  MIN_STRATEGY_VOTES: current ± [1]
  ENTRY_MIN_DET_CONF: current ± [0.05, 0.10, -0.05, -0.10]

OUTPUT:
  journal/shadow_variants_YYYY-MM-DD.json — daily counterfactual log
  Weekly summary via scripts/shadow_report.py

SAFETY:
  - Read-only subscriptions (never publishes ORDER_* or TRADE_PLAN_* topics)
  - No settings.py modification capability
  - All recommendations require human approval via memory_user_edits or
    manual settings.py edit
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from core.bus import Topic, Message
except ImportError:
    class Topic:
        SIGNAL_REJECTED  = "SIGNAL_REJECTED"
        SIGNAL_APPROVED  = "SIGNAL_APPROVED"
        POSITION_CLOSED  = "POSITION_CLOSED"
        CANDLES_READY    = "CANDLES_READY"
    class Message:
        payload: dict

try:
    from config.settings import (
        ML_MIN_CONFIDENCE, MIN_STRATEGY_VOTES,
        ENTRY_MIN_DET_CONF, JOURNAL_DIR,
    )
except ImportError:
    ML_MIN_CONFIDENCE  = 0.36
    MIN_STRATEGY_VOTES = 2
    ENTRY_MIN_DET_CONF = 0.62
    JOURNAL_DIR        = "journal"

# ── Parameter variants to shadow-test ──────────────────────────────────────────
ML_CONF_DELTAS    = [-0.08, -0.04, 0.0, +0.04, +0.08]
VOTES_DELTAS      = [-1, 0, +1]
DET_CONF_DELTAS   = [-0.10, -0.05, 0.0, +0.05, +0.10]

FORWARD_TRACK_CANDLES = 6   # track outcome for N candles after rejection


@dataclass
class RejectedSignal:
    """A signal that was rejected, tracked for counterfactual analysis."""
    timestamp:        str
    direction:        str
    strategies_fired: list[str]
    n_votes:          int
    ml_confidence:    float
    det_conf:         float
    rejection_reason: str
    entry_premium_est:float
    entry_spot:       float
    candles_tracked:  int = 0
    forward_pnls:     list[float] = field(default_factory=list)
    resolved:         bool = False


@dataclass
class VariantOutcome:
    """Aggregated outcome for one parameter variant."""
    param_name:       str
    variant_value:    float
    extra_signals:    int = 0   # signals that would pass under this variant but not current
    extra_wins:       int = 0
    extra_losses:     int = 0
    extra_pnl_sum:    float = 0.0
    extra_pnl_avg:    float = 0.0


class ShadowParameterAgent:
    """
    Agent 11: Shadow / Counterfactual Parameter Agent.
    Pure observer — logs "what if" outcomes for human review.
    """

    NAME = "ShadowParameterAgent"

    def __init__(self, bus=None) -> None:
        self.bus = bus
        self._pending: list[RejectedSignal] = []
        self._variant_outcomes: dict[str, VariantOutcome] = {}
        self._today = date.today().isoformat()
        self._log_path = Path(JOURNAL_DIR) / f"shadow_variants_{self._today}.json"
        self._init_variants()

    def _init_variants(self) -> None:
        for d in ML_CONF_DELTAS:
            if d == 0:
                continue
            key = f"ML_MIN_CONFIDENCE_{d:+.2f}"
            self._variant_outcomes[key] = VariantOutcome("ML_MIN_CONFIDENCE",
                                                          round(ML_MIN_CONFIDENCE + d, 4))
        for d in VOTES_DELTAS:
            if d == 0:
                continue
            key = f"MIN_STRATEGY_VOTES_{d:+d}"
            self._variant_outcomes[key] = VariantOutcome("MIN_STRATEGY_VOTES",
                                                          max(1, MIN_STRATEGY_VOTES + d))
        for d in DET_CONF_DELTAS:
            if d == 0:
                continue
            key = f"ENTRY_MIN_DET_CONF_{d:+.2f}"
            self._variant_outcomes[key] = VariantOutcome("ENTRY_MIN_DET_CONF",
                                                          round(ENTRY_MIN_DET_CONF + d, 4))

    async def register(self) -> None:
        """Subscribe to relevant topics. Call once at startup."""
        if self.bus is None:
            return
        self.bus.subscribe(Topic.SIGNAL_REJECTED, self.on_signal_rejected)
        self.bus.subscribe(Topic.CANDLES_READY,   self.on_candle)
        logger.info(f"[{self.NAME}] Registered — shadow-testing "
                    f"{len(self._variant_outcomes)} parameter variants")

    # ── EVENT HANDLERS ────────────────────────────────────────────────────────

    async def on_signal_rejected(self, msg) -> None:
        """A signal was rejected by ML filter or other gates."""
        try:
            payload = msg.payload
            sig = RejectedSignal(
                timestamp        = datetime.now(IST).isoformat(),
                direction        = payload.get("direction", "NONE"),
                strategies_fired = payload.get("strategies_fired", []),
                n_votes          = payload.get("n_votes", 0),
                ml_confidence    = float(payload.get("ml_confidence", 0)),
                det_conf         = float(payload.get("det_conf", 0)),
                rejection_reason = payload.get("reason", "unknown"),
                entry_premium_est= float(payload.get("est_premium", 0)),
                entry_spot       = float(payload.get("spot", 0)),
            )

            # Check which variants WOULD have approved this signal
            for key, variant in self._variant_outcomes.items():
                would_pass = self._would_pass_variant(sig, key, variant)
                if would_pass:
                    variant.extra_signals += 1

            # Track this signal forward for outcome estimation
            if sig.entry_spot > 0:
                self._pending.append(sig)

        except Exception as e:
            logger.debug(f"[{self.NAME}] on_signal_rejected error: {e}")

    async def on_candle(self, msg) -> None:
        """Track forward outcomes for pending rejected signals."""
        try:
            spot = float(msg.payload.get("ltp", 0))
            if spot <= 0:
                return

            still_pending = []
            for sig in self._pending:
                if sig.entry_spot <= 0:
                    continue
                # Estimate forward P&L using simple delta proxy (0.45)
                pnl_pct = (spot - sig.entry_spot) / sig.entry_spot * 100
                if sig.direction == "BUY_PUT":
                    pnl_pct = -pnl_pct
                # Convert to option premium % move (delta ~0.45, leverage ~5-8x)
                option_pnl_pct = pnl_pct * 0.45 / max(sig.entry_premium_est, 1) * sig.entry_spot

                sig.forward_pnls.append(round(option_pnl_pct, 2))
                sig.candles_tracked += 1

                if sig.candles_tracked >= FORWARD_TRACK_CANDLES:
                    self._resolve_signal(sig)
                    sig.resolved = True
                else:
                    still_pending.append(sig)

            self._pending = still_pending

            # Periodically save
            if len(self._pending) == 0 or len(self._pending) % 5 == 0:
                self._save_log()

        except Exception as e:
            logger.debug(f"[{self.NAME}] on_candle error: {e}")

    # ── VARIANT EVALUATION ────────────────────────────────────────────────────

    def _would_pass_variant(
        self, sig: RejectedSignal, key: str, variant: VariantOutcome
    ) -> bool:
        """Check if this rejected signal would PASS under the variant threshold."""
        if variant.param_name == "ML_MIN_CONFIDENCE":
            # Would pass if ml_confidence >= variant value (lower threshold)
            return sig.ml_confidence >= variant.variant_value and \
                   sig.rejection_reason in ("ml_confidence", "low_ml_conf")

        elif variant.param_name == "MIN_STRATEGY_VOTES":
            return sig.n_votes >= variant.variant_value and \
                   sig.rejection_reason in ("min_votes", "insufficient_votes")

        elif variant.param_name == "ENTRY_MIN_DET_CONF":
            return sig.det_conf >= variant.variant_value and \
                   sig.rejection_reason in ("det_conf", "regime_confidence")

        return False

    def _resolve_signal(self, sig: RejectedSignal) -> None:
        """Compute final outcome and attribute to matching variants."""
        if not sig.forward_pnls:
            return
        final_pnl = sig.forward_pnls[-1]
        peak_pnl  = max(sig.forward_pnls)
        # Use peak with some pullback assumption (mimics TSL behaviour)
        realized_pnl = peak_pnl * 0.7 if peak_pnl > 5 else final_pnl

        for key, variant in self._variant_outcomes.items():
            if self._would_pass_variant(sig, key, variant):
                variant.extra_pnl_sum += realized_pnl
                if realized_pnl > 0:
                    variant.extra_wins += 1
                else:
                    variant.extra_losses += 1
                n = variant.extra_wins + variant.extra_losses
                variant.extra_pnl_avg = variant.extra_pnl_sum / max(n, 1)

    # ── REPORTING ─────────────────────────────────────────────────────────────

    def get_report(self) -> dict:
        """Return current shadow-test report."""
        report = {
            "date":             self._today,
            "current_settings": {
                "ML_MIN_CONFIDENCE":  ML_MIN_CONFIDENCE,
                "MIN_STRATEGY_VOTES": MIN_STRATEGY_VOTES,
                "ENTRY_MIN_DET_CONF": ENTRY_MIN_DET_CONF,
            },
            "variants": {},
            "pending_tracked": len(self._pending),
            "recommendation":  "",
        }

        best_variant = None
        best_pnl     = 0.0

        for key, v in self._variant_outcomes.items():
            report["variants"][key] = asdict(v)
            if v.extra_pnl_sum > best_pnl and (v.extra_wins + v.extra_losses) >= 3:
                best_pnl     = v.extra_pnl_sum
                best_variant = key

        if best_variant:
            v = self._variant_outcomes[best_variant]
            report["recommendation"] = (
                f"Variant {best_variant} would have added {v.extra_signals} signals "
                f"({v.extra_wins}W/{v.extra_losses}L, net {v.extra_pnl_sum:+.1f}% cumulative). "
                f"Consider changing {v.param_name} to {v.variant_value} — "
                f"REQUIRES YOUR REVIEW before applying."
            )
        else:
            report["recommendation"] = (
                "Insufficient data yet (need >= 3 resolved signals per variant). "
                "Continue monitoring."
            )

        return report

    def _save_log(self) -> None:
        try:
            Path(JOURNAL_DIR).mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "w") as f:
                json.dump(self.get_report(), f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[{self.NAME}] save_log error: {e}")


# ── Singleton ─────────────────────────────────────────────────────────────────
_shadow_agent: ShadowParameterAgent | None = None

def get_shadow_agent(bus=None) -> ShadowParameterAgent:
    global _shadow_agent
    if _shadow_agent is None:
        _shadow_agent = ShadowParameterAgent(bus)
    return _shadow_agent

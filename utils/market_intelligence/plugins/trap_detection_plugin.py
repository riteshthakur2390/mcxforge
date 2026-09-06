"""
utils/market_intelligence/plugins/trap_detection_plugin.py — Trap Detection Engine
========================================================================================
Detects situations where retail traders are likely getting trapped —
bull traps (breakout that reverses) and bear traps (breakdown that
reverses). Never rejects a trade outright; reduces Trade Quality Score
proportional to trap risk, per spec.

INPUTS COMBINED (each contributes to Bull Trap Score OR Bear Trap Score):
  - PCR Extreme (from PCRPlugin's raw_data)
  - Price exhaustion (extended move without fresh momentum)
  - OI divergence (price making new high/low but OI not confirming)
  - Call/Put writing concentration
  - VWAP failure (price crossed VWAP then failed to hold)
  - VIX expansion (rising fear undermines a breakout)
  - Momentum weakening (RSI/momentum divergence)
  - Large candle rejection (long wick against the move)
  - Support/Resistance failure (broke a level then reclaimed it)
"""

from __future__ import annotations
from dataclasses import dataclass

from utils.market_intelligence.base import MarketContextPlugin, ContextScore

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

TRAP_RISK_BANDS = [
    (0, 25,  "LOW"),
    (25, 50, "MEDIUM"),
    (50, 75, "HIGH"),
    (75, 101, "VERY_HIGH"),
]

# Score contribution reduction applied to Trade Quality Score per trap band
TRAP_SCORE_PENALTY = {"LOW": 0.0, "MEDIUM": -5.0, "HIGH": -12.0, "VERY_HIGH": -20.0}


@dataclass
class TrapInputs:
    pcr_extreme:            bool = False
    price_exhaustion:         bool = False
    oi_divergence:              bool = False
    call_writing_heavy:           bool = False
    put_writing_heavy:             bool = False
    vwap_failure:                    bool = False
    vix_expanding:                    bool = False
    momentum_weakening:                 bool = False
    large_rejection_candle:               bool = False
    support_resistance_failure:             bool = False


class TrapDetectionEngine(MarketContextPlugin):
    """Computes Bull Trap / Bear Trap risk scores from confluence of signals."""

    name = "TrapDetection"
    max_positive_contribution = 0.0     # trap detection only ever penalizes, never boosts
    max_negative_contribution = -20.0

    # Weight of each input toward a trap score (out of 100 total possible)
    WEIGHTS = {
        "pcr_extreme": 15, "price_exhaustion": 15, "oi_divergence": 15,
        "call_writing_heavy": 10, "put_writing_heavy": 10, "vwap_failure": 12,
        "vix_expanding": 10, "momentum_weakening": 8,
        "large_rejection_candle": 8, "support_resistance_failure": 7,
    }

    def evaluate(self, ctx: dict) -> ContextScore:
        direction = ctx.get("direction", "BUY_CALL")
        inputs = self._gather_inputs(ctx)

        bull_trap_score = self._compute_bull_trap(inputs)
        bear_trap_score = self._compute_bear_trap(inputs)

        # The trap risk RELEVANT to this trade is the one matching its direction:
        # buying calls into a bull trap is the danger; buying puts into a bear trap is the danger.
        relevant_score = bull_trap_score if direction == "BUY_CALL" else bear_trap_score
        band = self._band(relevant_score)
        penalty = TRAP_SCORE_PENALTY[band]

        explanation = (
            f"BullTrap={bull_trap_score:.0f} BearTrap={bear_trap_score:.0f} → "
            f"relevant({direction})={relevant_score:.0f} ({band}) | "
            f"active_flags={self._active_flags(inputs)}"
        )
        logger.debug(f"[TrapDetection] {explanation} → penalty={penalty:+.1f}")

        return ContextScore(
            plugin_name=self.name,
            score_contribution=penalty,
            label=f"{band} Trap Risk",
            explanation=explanation,
            raw_data={
                "bull_trap_score": round(bull_trap_score, 1),
                "bear_trap_score": round(bear_trap_score, 1),
                "relevant_score": round(relevant_score, 1),
                "band": band,
            },
        )

    def _gather_inputs(self, ctx: dict) -> TrapInputs:
        pcr_state = ctx.get("_pcr_state", "")   # engine wires PCRPlugin's output here
        return TrapInputs(
            pcr_extreme=pcr_state == "EXTREME",
            price_exhaustion=ctx.get("price_exhaustion", False),
            oi_divergence=ctx.get("oi_divergence", False),
            call_writing_heavy=ctx.get("call_writing_heavy", False),
            put_writing_heavy=ctx.get("put_writing_heavy", False),
            vwap_failure=ctx.get("vwap_failure", False),
            vix_expanding=ctx.get("vix_change_pct", 0) is not None and ctx.get("vix_change_pct", 0) > 3,
            momentum_weakening=ctx.get("momentum_weakening", False),
            large_rejection_candle=ctx.get("large_rejection_candle", False),
            support_resistance_failure=ctx.get("support_resistance_failure", False),
        )

    def _compute_bull_trap(self, i: TrapInputs) -> float:
        """Bull trap: breakout looks strong but is likely to fail/reverse down."""
        score = 0.0
        if i.pcr_extreme and i.call_writing_heavy: score += self.WEIGHTS["pcr_extreme"]
        if i.price_exhaustion: score += self.WEIGHTS["price_exhaustion"]
        if i.oi_divergence: score += self.WEIGHTS["oi_divergence"]
        if i.call_writing_heavy: score += self.WEIGHTS["call_writing_heavy"]
        if i.vwap_failure: score += self.WEIGHTS["vwap_failure"]
        if i.vix_expanding: score += self.WEIGHTS["vix_expanding"]
        if i.momentum_weakening: score += self.WEIGHTS["momentum_weakening"]
        if i.large_rejection_candle: score += self.WEIGHTS["large_rejection_candle"]
        if i.support_resistance_failure: score += self.WEIGHTS["support_resistance_failure"]
        return min(100.0, score)

    def _compute_bear_trap(self, i: TrapInputs) -> float:
        """Bear trap: breakdown looks strong but is likely to fail/reverse up."""
        score = 0.0
        if i.pcr_extreme and i.put_writing_heavy: score += self.WEIGHTS["pcr_extreme"]
        if i.price_exhaustion: score += self.WEIGHTS["price_exhaustion"]
        if i.oi_divergence: score += self.WEIGHTS["oi_divergence"]
        if i.put_writing_heavy: score += self.WEIGHTS["put_writing_heavy"]
        if i.vwap_failure: score += self.WEIGHTS["vwap_failure"]
        if i.vix_expanding: score += self.WEIGHTS["vix_expanding"]
        if i.momentum_weakening: score += self.WEIGHTS["momentum_weakening"]
        if i.large_rejection_candle: score += self.WEIGHTS["large_rejection_candle"]
        if i.support_resistance_failure: score += self.WEIGHTS["support_resistance_failure"]
        return min(100.0, score)

    @staticmethod
    def _band(score: float) -> str:
        for lo, hi, label in TRAP_RISK_BANDS:
            if lo <= score < hi:
                return label
        return "VERY_HIGH"

    @staticmethod
    def _active_flags(i: TrapInputs) -> list[str]:
        return [k for k, v in vars(i).items() if v]


def get_trap_detection_plugin() -> TrapDetectionEngine:
    return TrapDetectionEngine()

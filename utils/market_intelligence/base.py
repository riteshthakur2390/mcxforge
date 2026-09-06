"""
utils/market_intelligence/base.py — Market Context Plugin Interface
========================================================================
Every market-context module (PCR, OI, FII, VIX, Trap Detection, Max Pain,
future modules) implements this ONE interface. The engine doesn't know
or care what's inside a plugin — it just calls evaluate(ctx) and
aggregates the ContextScore objects returned.

CRITICAL DESIGN RULE (per spec): a plugin NEVER returns "REJECT" or a
hard veto. It only returns a signed score contribution (-15 to +15
typical range) plus a human-readable label and explanation. The engine
sums these into a single 0-100 Trade Quality Score — PCR (or any other
plugin) never single-handedly blocks a trade.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContextScore:
    plugin_name:        str
    score_contribution:  float   # signed, added into the 0-100 composite
    label:                 str    # short human-readable state, e.g. "Slightly Elevated"
    explanation:            str    # one-line reasoning for logs/UI
    raw_data:                 dict = field(default_factory=dict)   # plugin-specific detail for debugging/UI


class MarketContextPlugin:
    """Base class every Market Intelligence plugin must implement."""

    name = "BasePlugin"
    max_positive_contribution = 10.0
    max_negative_contribution = -10.0

    def evaluate(self, ctx: dict) -> ContextScore:
        """
        ctx: the shared market-context dict built each candle by the
        engine (price, EMA, VWAP, momentum, volume, PCR, OI, VIX,
        regime, time-of-day, etc.) — same pattern as the strategy
        runner's ctx dict elsewhere in SignalForge.
        """
        raise NotImplementedError

    def _neutral(self, reason: str = "insufficient data") -> ContextScore:
        return ContextScore(self.name, 0.0, "Neutral", reason)

    def _clip(self, score: float) -> float:
        return max(self.max_negative_contribution, min(self.max_positive_contribution, score))

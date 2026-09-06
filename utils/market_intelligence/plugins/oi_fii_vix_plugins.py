"""
utils/market_intelligence/plugins/oi_fii_vix_plugins.py — Adapter Plugins
================================================================================
Wraps ALREADY-BUILT SignalForge intelligence (OI velocity/walls in
advanced_filters.py, FII/DII + Max Pain + BankNifty divergence in
market_microstructure.py) into the new plugin interface — this is the
extensibility the spec asked for: future modules (IV Rank, Gamma
Exposure, Dealer Position) plug in the exact same way, no engine
changes required.
"""

from __future__ import annotations
from utils.market_intelligence.base import MarketContextPlugin, ContextScore

try:
    from utils.advanced_filters import get_oi_velocity, get_wall_tracker
except ImportError:
    get_oi_velocity = get_wall_tracker = None

try:
    from utils.market_microstructure import get_microstructure
except ImportError:
    get_microstructure = None

try:
    from agents_code.agent9_regime.vix_slope import VIXSlopeTracker
except ImportError:
    VIXSlopeTracker = None

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# OI PLUGIN — wraps OIVelocityTracker + PutCallWallTracker
# ══════════════════════════════════════════════════════════════════════════════

class OIPlugin(MarketContextPlugin):
    name = "OI"
    max_positive_contribution = 6.0
    max_negative_contribution = -6.0

    def evaluate(self, ctx: dict) -> ContextScore:
        ce_oi = ctx.get("ce_oi_by_strike")
        pe_oi = ctx.get("pe_oi_by_strike")
        spot = ctx.get("spot")
        direction = ctx.get("direction", "BUY_CALL")

        if not ce_oi or not pe_oi or spot is None or get_wall_tracker is None:
            return self._neutral("no option chain OI data supplied")

        wall = get_wall_tracker().update(spot, ce_oi, pe_oi)

        aligned = (wall.wall_signal == direction)
        contribution = 6.0 if aligned else (-4.0 if wall.wall_signal != "NEUTRAL" else 0.0)

        explanation = f"Wall signal={wall.wall_signal} pos={wall.spot_position} | {wall.note[:100]}"
        return ContextScore(self.name, self._clip(contribution), wall.spot_position, explanation,
                            {"call_wall": wall.call_wall, "put_wall": wall.put_wall})


# ══════════════════════════════════════════════════════════════════════════════
# FII/DII PLUGIN — wraps FIIDIIFetcher
# ══════════════════════════════════════════════════════════════════════════════

class FIIPlugin(MarketContextPlugin):
    name = "FII_DII"
    max_positive_contribution = 5.0
    max_negative_contribution = -5.0

    def evaluate(self, ctx: dict) -> ContextScore:
        fii_signal = ctx.get("fii_signal")   # engine expects this pre-fetched (daily, not per-candle)
        direction = ctx.get("direction", "BUY_CALL")

        if not fii_signal:
            return self._neutral("no FII/DII data supplied today")

        bullish = fii_signal in ("STRONG_BULL", "BULL")
        bearish = fii_signal in ("STRONG_BEAR", "BEAR")
        strong = fii_signal in ("STRONG_BULL", "STRONG_BEAR")

        if direction == "BUY_CALL" and bullish:
            contribution = 5.0 if strong else 3.0
        elif direction == "BUY_PUT" and bearish:
            contribution = 5.0 if strong else 3.0
        elif direction == "BUY_CALL" and bearish:
            contribution = -5.0 if strong else -3.0
        elif direction == "BUY_PUT" and bullish:
            contribution = -5.0 if strong else -3.0
        else:
            contribution = 0.0

        return ContextScore(self.name, self._clip(contribution), fii_signal,
                            f"FII flow={fii_signal}, trade direction={direction}",
                            {"fii_signal": fii_signal})


# ══════════════════════════════════════════════════════════════════════════════
# VIX PLUGIN — wraps VIXSlopeTracker
# ══════════════════════════════════════════════════════════════════════════════

class VIXPlugin(MarketContextPlugin):
    name = "VIX"
    max_positive_contribution = 5.0
    max_negative_contribution = -8.0

    def __init__(self) -> None:
        self._tracker = VIXSlopeTracker() if VIXSlopeTracker else None

    def evaluate(self, ctx: dict) -> ContextScore:
        vix = ctx.get("india_vix")
        if vix is None or self._tracker is None:
            return self._neutral("no VIX data supplied")

        slope = self._tracker.update(vix)

        if slope.trend == "RISING" and slope.advice == "AVOID_BUYING":
            contribution = -6.0
        elif slope.trend == "FALLING" and slope.advice == "GOOD_FOR_BUYERS":
            contribution = 4.0
        else:
            contribution = 0.0

        explanation = f"VIX={slope.current} trend={slope.trend} slope3d={slope.slope_3d} advice={slope.advice}"
        return ContextScore(self.name, self._clip(contribution), slope.trend, explanation,
                            {"vix": slope.current, "iv_rank_proxy": slope.iv_rank})


# ══════════════════════════════════════════════════════════════════════════════
# MAX PAIN PLUGIN — wraps MaxPainCalculator
# ══════════════════════════════════════════════════════════════════════════════

class MaxPainPlugin(MarketContextPlugin):
    name = "MaxPain"
    max_positive_contribution = 4.0
    max_negative_contribution = -4.0

    def evaluate(self, ctx: dict) -> ContextScore:
        ce_oi = ctx.get("ce_oi_by_strike")
        pe_oi = ctx.get("pe_oi_by_strike")
        spot = ctx.get("spot")
        direction = ctx.get("direction", "BUY_CALL")
        is_expiry = ctx.get("is_expiry_day", False)

        if not ce_oi or not pe_oi or spot is None or get_microstructure is None:
            return self._neutral("no option chain OI data supplied")

        mp = get_microstructure().max_pain.compute(spot, ce_oi, pe_oi)

        # Max pain gravity matters most near expiry — scale contribution accordingly
        weight = 1.0 if is_expiry else 0.4
        if mp.bias == "ABOVE_MAX_PAIN" and direction == "BUY_PUT":
            contribution = 4.0 * weight
        elif mp.bias == "BELOW_MAX_PAIN" and direction == "BUY_CALL":
            contribution = 4.0 * weight
        elif mp.bias == "ABOVE_MAX_PAIN" and direction == "BUY_CALL":
            contribution = -3.0 * weight
        elif mp.bias == "BELOW_MAX_PAIN" and direction == "BUY_PUT":
            contribution = -3.0 * weight
        else:
            contribution = 0.0

        return ContextScore(self.name, self._clip(contribution), mp.bias, mp.expiry_signal,
                            {"max_pain_strike": mp.max_pain_strike, "distance_pts": mp.distance_pts})


def get_oi_plugin() -> OIPlugin: return OIPlugin()
def get_fii_plugin() -> FIIPlugin: return FIIPlugin()
def get_vix_plugin() -> VIXPlugin: return VIXPlugin()
def get_max_pain_plugin() -> MaxPainPlugin: return MaxPainPlugin()

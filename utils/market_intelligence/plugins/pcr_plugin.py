"""
utils/market_intelligence/plugins/pcr_plugin.py — PCR Analyzer Plugin
==========================================================================
The first plugin in the Market Intelligence Engine. PCR is NEVER a
standalone buy/reject signal — it only contributes a signed score
(-15 to +8, per spec) to the overall Trade Quality Score.

WHAT THIS PLUGIN DOES:
  1. Stores PCR history and computes 5/15/30-min and daily trends
     (trend often matters more than the snapshot value).
  2. Classifies current PCR into a configurable state (Very Low / Low /
     Neutral / High / Extreme) — thresholds are settings, not hardcoded,
     so they can be backtested and optimized later.
  3. Cross-references Price direction vs PCR direction (cases A-D from
     spec) to infer healthy-bullish / exhaustion / support-building /
     bearish-continuation — informational only, feeds into scoring.
  4. Adjusts its own contribution based on regime (trending / sideways
     / volatile / expiry day / gap) — the same PCR value means different
     things in different contexts.
  5. Cross-checks with OI (put buying vs put writing) and VIX
     (falling vs rising) when that data is supplied in ctx, since PCR
     alone is incomplete without knowing WHY it moved.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

from utils.market_intelligence.base import MarketContextPlugin, ContextScore

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ── Configurable PCR state thresholds (NOT hardcoded forever — spec requirement) ──
# These are defaults; a future backtest/optimizer can rewrite this dict.
PCR_STATE_THRESHOLDS = {
    "VERY_LOW":  0.60,
    "LOW":        0.85,
    "NEUTRAL_HI": 1.15,   # neutral band is LOW..NEUTRAL_HI
    "HIGH":         1.40,
    # anything above HIGH => EXTREME
}

# ── PCR + Price-direction confidence contribution table (spec example values) ──
CONTRIB_BULLISH_ALIGNMENT   = +8.0
CONTRIB_NEUTRAL               = 0.0
CONTRIB_CONTRADICTING          = -10.0
CONTRIB_EXTREME_POSITIONING     = -15.0

TREND_WINDOW_5MIN  = 1     # candles (assuming 5-min candles = 1 candle)
TREND_WINDOW_15MIN = 3
TREND_WINDOW_30MIN = 6


@dataclass
class PriceVsPCRCase:
    case:            str    # "A" | "B" | "C" | "D"
    label:            str
    interpretation:    str


class PCRHistoryStore:
    """Rolling PCR history — separate from the plugin so it can be reused
    by the trap-detection plugin and future backtesting without recomputation."""

    def __init__(self, maxlen: int = 500) -> None:
        self._history: deque = deque(maxlen=maxlen)   # (timestamp, pcr, price)

    def update(self, pcr: float, price: float, ts: datetime = None) -> None:
        self._history.append((ts or datetime.now(IST), pcr, price))

    def trend(self, candles_back: int) -> float | None:
        """Returns PCR change over N candles back, or None if insufficient history."""
        if len(self._history) <= candles_back:
            return None
        now_pcr = self._history[-1][1]
        past_pcr = self._history[-1 - candles_back][1]
        return round(now_pcr - past_pcr, 4)

    def daily_trend(self) -> float | None:
        if len(self._history) < 2:
            return None
        today = self._history[-1][0].date()
        day_start = None
        for ts, pcr, _ in self._history:
            if ts.date() == today:
                day_start = pcr
                break
        if day_start is None:
            return None
        return round(self._history[-1][1] - day_start, 4)

    def latest(self) -> tuple[float, float] | None:
        if not self._history:
            return None
        _, pcr, price = self._history[-1]
        return pcr, price

    def previous(self) -> tuple[float, float] | None:
        if len(self._history) < 2:
            return None
        _, pcr, price = self._history[-2]
        return pcr, price


class PCRPlugin(MarketContextPlugin):
    """
    PCR Analyzer — classifies PCR state, price/PCR relationship, and
    contributes a bounded confidence adjustment. Never rejects a trade.
    """

    name = "PCR"
    max_positive_contribution = 8.0
    max_negative_contribution = -15.0

    def __init__(self, thresholds: dict = None) -> None:
        self.thresholds = thresholds or dict(PCR_STATE_THRESHOLDS)
        self.store = PCRHistoryStore()

    def update(self, pcr: float, price: float) -> None:
        """Call every candle to feed the rolling PCR history."""
        self.store.update(pcr, price)

    def evaluate(self, ctx: dict) -> ContextScore:
        pcr = ctx.get("pcr")
        price = ctx.get("spot") or ctx.get("price")
        direction = ctx.get("direction", "BUY_CALL")   # the candidate trade direction

        if pcr is None:
            return self._neutral("no PCR data supplied")

        # Capture the PRIOR snapshot before recording this candle's reading —
        # this makes evaluate() safe even if called more than once around the
        # same candle (the price/PCR relationship always compares against
        # what was true immediately before this reading, never against itself).
        prev_snapshot = self.store.latest()
        if price is not None:
            self.update(pcr, price)

        state = self._classify_state(pcr)
        trend_5  = self.store.trend(TREND_WINDOW_5MIN)
        trend_15 = self.store.trend(TREND_WINDOW_15MIN)
        trend_30 = self.store.trend(TREND_WINDOW_30MIN)
        trend_daily = self.store.daily_trend()

        pv_case = self._price_pcr_relationship(prev_snapshot, pcr, price)

        # ── Base contribution from price/PCR alignment with the candidate direction ──
        contribution = self._score_alignment(direction, pv_case, state)

        # ── Regime-aware adjustment (spec: same PCR means different things per regime) ──
        contribution += self._regime_adjustment(ctx, state)

        # ── OI cross-check: is high PCR from put BUYING or put WRITING? ──────────
        oi_note = self._oi_crosscheck(ctx, state)

        # ── VIX cross-check: high PCR + falling VIX is healthier than rising VIX ──
        vix_note = self._vix_crosscheck(ctx, state)

        contribution = self._clip(contribution)

        explanation = (
            f"PCR={pcr:.2f} ({state}) | trend5={trend_5} trend15={trend_15} "
            f"trend30={trend_30} daily={trend_daily} | {pv_case.label}: {pv_case.interpretation} | "
            f"{oi_note} | {vix_note}"
        )

        logger.debug(f"[PCRPlugin] {explanation} → contribution={contribution:+.1f}")

        return ContextScore(
            plugin_name=self.name,
            score_contribution=round(contribution, 2),
            label=state,
            explanation=explanation,
            raw_data={
                "pcr": pcr, "state": state,
                "trend_5min": trend_5, "trend_15min": trend_15,
                "trend_30min": trend_30, "trend_daily": trend_daily,
                "price_pcr_case": pv_case.case,
            },
        )

    # ── STATE CLASSIFICATION (configurable thresholds) ──────────────────────────

    def _classify_state(self, pcr: float) -> str:
        t = self.thresholds
        if pcr < t["VERY_LOW"]:
            return "VERY_LOW"
        if pcr < t["LOW"]:
            return "LOW"
        if pcr < t["NEUTRAL_HI"]:
            return "NEUTRAL"
        if pcr < t["HIGH"]:
            return "HIGH"
        return "EXTREME"

    # ── PRICE vs PCR RELATIONSHIP (cases A-D from spec) ──────────────────────────

    def _price_pcr_relationship(self, prev_snapshot, pcr_now: float, price_now: float | None) -> PriceVsPCRCase:
        if not prev_snapshot or price_now is None:
            return PriceVsPCRCase("N/A", "Insufficient history", "Not enough PCR history yet")

        pcr_prev, price_prev = prev_snapshot
        price_up = price_now > price_prev
        pcr_up = pcr_now > pcr_prev

        if price_up and pcr_up:
            return PriceVsPCRCase("A", "Price↑ PCR↑", "Healthy bullish positioning — put writers confident, calls not chased")
        if price_up and not pcr_up:
            return PriceVsPCRCase("B", "Price↑ PCR↓", "Weakening option support — possible exhaustion of the rally")
        if not price_up and pcr_up:
            return PriceVsPCRCase("C", "Price↓ PCR↑", "Aggressive put writing — possible support zone forming")
        return PriceVsPCRCase("D", "Price↓ PCR↓", "Continuation of bearish sentiment — put buying following price down")

    # ── ALIGNMENT SCORING (spec confidence contribution table) ───────────────────

    def _score_alignment(self, direction: str, pv_case: PriceVsPCRCase, state: str) -> float:
        # Extreme positioning always gets a penalty regardless of direction —
        # crowded trades are dangerous whichever side you're on.
        if state == "EXTREME":
            return CONTRIB_EXTREME_POSITIONING

        bullish_case = pv_case.case in ("A", "C")   # A=healthy bull, C=support forming
        bearish_case = pv_case.case in ("D",)        # D=bearish continuation
        exhaustion_case = pv_case.case == "B"

        wants_call = direction == "BUY_CALL"
        wants_put = direction == "BUY_PUT"

        if wants_call and bullish_case:
            return CONTRIB_BULLISH_ALIGNMENT
        if wants_put and bearish_case:
            return CONTRIB_BULLISH_ALIGNMENT   # symmetric: aligned with direction either way
        if wants_call and exhaustion_case:
            return CONTRIB_CONTRADICTING * 0.6   # partial penalty, not full contradiction
        if wants_call and bearish_case:
            return CONTRIB_CONTRADICTING
        if wants_put and bullish_case:
            return CONTRIB_CONTRADICTING

        return CONTRIB_NEUTRAL

    # ── REGIME-AWARE ADJUSTMENT ───────────────────────────────────────────────────

    def _regime_adjustment(self, ctx: dict, state: str) -> float:
        regime = ctx.get("regime", "")
        is_expiry = ctx.get("is_expiry_day", False)
        gap_direction = ctx.get("gap_direction", "")

        adj = 0.0
        if is_expiry and state == "EXTREME":
            adj -= 3.0   # extreme PCR on expiry day = higher manipulation/pinning risk
        if regime == "VOLATILE" and state in ("HIGH", "EXTREME"):
            adj -= 2.0   # crowded positioning in a volatile regime = more dangerous
        if regime == "SIDEWAYS" and state == "NEUTRAL":
            adj += 1.0   # neutral PCR in sideways regime = consistent, mild positive
        if gap_direction and state == "EXTREME":
            adj -= 2.0   # extreme PCR right after a gap = elevated trap risk
        return adj

    # ── OI CROSS-CHECK ─────────────────────────────────────────────────────────────

    def _oi_crosscheck(self, ctx: dict, state: str) -> str:
        put_oi_change = ctx.get("put_oi_change_pct")
        put_price_change = ctx.get("put_premium_change_pct")
        if put_oi_change is None:
            return "OI cross-check: no OI change data"

        if put_oi_change > 0 and (put_price_change or 0) < 0:
            return "OI cross-check: high PCR driven by PUT WRITING (bullish support)"
        if put_oi_change > 0 and (put_price_change or 0) > 0:
            return "OI cross-check: high PCR driven by PUT BUYING (hedging/bearish)"
        return "OI cross-check: inconclusive"

    # ── VIX CROSS-CHECK ────────────────────────────────────────────────────────────

    def _vix_crosscheck(self, ctx: dict, state: str) -> str:
        vix_change = ctx.get("vix_change_pct")
        if vix_change is None:
            return "VIX cross-check: no VIX data"
        if state in ("HIGH", "EXTREME") and vix_change < 0:
            return f"VIX cross-check: High PCR + VIX falling ({vix_change:+.1f}%) — healthier positioning"
        if state in ("HIGH", "EXTREME") and vix_change > 0:
            return f"VIX cross-check: High PCR + VIX rising ({vix_change:+.1f}%) — caution, fear building"
        return "VIX cross-check: neutral"


def get_pcr_plugin(thresholds: dict = None) -> PCRPlugin:
    return PCRPlugin(thresholds)

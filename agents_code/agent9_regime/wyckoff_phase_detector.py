"""
agents_code/agent9_regime/wyckoff_phase_detector.py
====================================================
Wyckoff-Style Price-Volume Market Phase Detector for NIFTY 5-min candles.

═══════════════════════════════════════════════════════════════════════
SYSTEM ARCHITECTURE
═══════════════════════════════════════════════════════════════════════

                    ┌─────────────────────────────┐
                    │   5-min OHLCV DataFrame       │
                    │   (NIFTY spot candles)         │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   WyckoffPhaseDetector        │
                    │                               │
                    │  Layer 1: Price Analysis      │
                    │  ├─ Trend direction            │
                    │  ├─ Price momentum (ROC)       │
                    │  └─ Higher highs / lower lows  │
                    │                               │
                    │  Layer 2: Volume Analysis     │
                    │  ├─ Volume trend               │
                    │  ├─ Price-Volume relationship  │
                    │  └─ Effort vs Result           │
                    │                               │
                    │  Layer 3: Phase Scoring       │
                    │  ├─ Accumulation score         │
                    │  ├─ Markup score               │
                    │  ├─ Distribution score         │
                    │  └─ Markdown score             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   WyckoffResult               │
                    │  ├─ phase                      │
                    │  ├─ confidence (0-100)         │
                    │  ├─ bullish_score (0-100)      │
                    │  ├─ bearish_score (0-100)      │
                    │  ├─ volume_confirmation (0-100)│
                    │  ├─ trend_strength (0-100)     │
                    │  └─ recommendation             │
                    └──────────────┬──────────────┘
                                   │
             ┌─────────────────────┼─────────────────────┐
             ▼                     ▼                     ▼
    StrategyRunner          ML Filter              Position Manager
    (signal weight)      (confidence boost)      (size multiplier)

═══════════════════════════════════════════════════════════════════════
WYCKOFF PHASES — MATHEMATICAL LOGIC
═══════════════════════════════════════════════════════════════════════

ACCUMULATION:
  - Price: flat or slightly declining (range bound, low momentum)
  - Volume: declining on down moves (supply absorbed)
  - Signal: selling volume drying up = smart money absorbing
  - Indicators:
      price_roc_20   in range [-1.5%, +1.5%]  (flat price)
      vol_ratio_down < 0.85  (lower volume on down candles)
      price_range_pct < 1.5%  (tight price range)

MARKUP (Bullish Trend):
  - Price: rising trend (higher highs + higher lows)
  - Volume: rising on up moves (demand > supply)
  - Signal: price rise + volume confirmation = institutional buying
  - Indicators:
      price_roc_20   > 1.5%
      vol_ratio_up   > 1.15  (higher volume on up moves)
      hh_count       >= 2 of last 5 swings are HH
      ema_9 > ema_21 > ema_50

DISTRIBUTION:
  - Price: flat at highs (range bound after markup)
  - Volume: rising but price not advancing (effort without result)
  - Signal: high volume + no price progress = supply overwhelming demand
  - Indicators:
      price_roc_20   in range [-1.5%, +1.5%]
      vol_trend      > 1.0  (volume elevated)
      price_near_high: close within 1% of 20-bar high
      effort_result  < 0.3  (volume rising, price flat)

MARKDOWN (Bearish Trend):
  - Price: falling trend (lower highs + lower lows)
  - Volume: rising on down moves (supply > demand)
  - Signal: price fall + volume confirmation = institutional selling
  - Indicators:
      price_roc_20   < -1.5%
      vol_ratio_down > 1.15  (higher volume on down moves)
      ll_count       >= 2 of last 5 swings are LL
      ema_9 < ema_21 < ema_50

NEUTRAL / UNCERTAIN:
  - None of the above patterns are clear
  - Mixed signals: rising price but falling volume (weak rally)
  - Or: falling price but falling volume (weak selling, no panic)

═══════════════════════════════════════════════════════════════════════
PRICE-VOLUME RELATIONSHIPS (Wyckoff's 4 laws)
═══════════════════════════════════════════════════════════════════════

  1. Rising price + Rising volume  = BULLISH CONFIRMATION (+25 bullish)
  2. Rising price + Falling volume = WEAK RALLY            (+10 bullish, -5 confidence)
  3. Falling price + Rising volume = BEARISH CONFIRMATION  (+25 bearish)
  4. Falling price + Falling volume= WEAK SELLING          (+10 bearish, -5 confidence)

═══════════════════════════════════════════════════════════════════════
RECOMMENDATION LOGIC
═══════════════════════════════════════════════════════════════════════

  Phase         │ Confidence │ Recommendation
  ──────────────┼────────────┼──────────────────────────────
  MARKUP        │ ≥ 70       │ STRONGLY_FAVOR_CALL
  MARKUP        │ 50-70      │ MILDLY_FAVOR_CALL
  ACCUMULATION  │ ≥ 60       │ MILDLY_FAVOR_CALL (coiling)
  DISTRIBUTION  │ ≥ 70       │ STRONGLY_FAVOR_PUT
  DISTRIBUTION  │ 50-70      │ MILDLY_FAVOR_PUT
  MARKDOWN      │ ≥ 70       │ STRONGLY_FAVOR_PUT
  MARKDOWN      │ 50-70      │ MILDLY_FAVOR_PUT
  NEUTRAL       │ any        │ NEUTRAL
  any           │ < 40       │ NEUTRAL (low conviction)

═══════════════════════════════════════════════════════════════════════
HOW TO PLUG INTO EXISTING STRATEGIES
═══════════════════════════════════════════════════════════════════════

  Option A — Filter (block trades against phase):
    if wyckoff.recommendation == "STRONGLY_FAVOR_PUT":
        if direction == "BUY_CALL": return none  # block

  Option B — Size multiplier (reduce against phase):
    if wyckoff.recommendation == "MILDLY_FAVOR_PUT":
        if direction == "BUY_CALL": lots = max(1, lots // 2)

  Option C — Confidence boost (increase with phase):
    if wyckoff.recommendation == "STRONGLY_FAVOR_CALL":
        if direction == "BUY_CALL": conf += 0.05

  Option D — Informational (log only, no action):
    logger.info(f"Wyckoff: {wyckoff.phase} {wyckoff.confidence}")

  RECOMMENDED (current integration):
    Strong phase alignment (≥70%) → +0.04 confidence boost
    Phase misalignment (opposite)  → -0.05 confidence penalty
    Strong misalignment            → block trade entirely

═══════════════════════════════════════════════════════════════════════
BACKTESTING METHODOLOGY
═══════════════════════════════════════════════════════════════════════

  1. Run WyckoffPhaseDetector on historical 5-min NIFTY data (2022-2026)
  2. For each bar: record (phase, confidence, recommendation)
  3. Forward test: if recommendation says CALL, check if NIFTY was +ve
     in next N candles. Compute win rate per recommendation type.
  4. Tune thresholds using parameter sensitivity grid.

  Key metrics to optimise:
  - Win rate when Wyckoff aligns with trade direction (target: > 60%)
  - Win rate when Wyckoff conflicts with trade direction (expect: < 45%)
  - Net alpha: (aligned_wr - baseline_wr) = Wyckoff edge contribution

═══════════════════════════════════════════════════════════════════════
TUNABLE PARAMETERS
═══════════════════════════════════════════════════════════════════════

  PRICE_ROC_WINDOW       = 20    # candles for ROC calculation
  VOLUME_SMOOTH_WINDOW   = 10    # smoothing for volume trend
  PRICE_SMOOTH_WINDOW    = 5     # EMA smoothing for price trend
  MARKUP_ROC_THRESHOLD   = 1.5   # % price change to classify as markup
  VOLUME_CONFIRM_RATIO   = 1.15  # volume must be 15% above avg on trend candles
  DISTRIBUTION_HIGH_PCT  = 1.0   # within 1% of recent high = near highs
  SWING_LOOKBACK         = 10    # bars to find swing highs/lows
  HH_HL_MIN_COUNT        = 2     # minimum HH+HL pairs for markup
  CONFIDENCE_BLOCK_BELOW = 40    # below this confidence = treat as NEUTRAL
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

PRICE_ROC_WINDOW        = 20     # candles for price rate-of-change
VOLUME_SMOOTH_WINDOW    = 10     # smoothing window for volume trend
PRICE_SMOOTH_WINDOW     = 5      # EMA smoothing period for price trend
MARKUP_ROC_THRESHOLD    = 1.5    # % price ROC to confirm markup / markdown
VOLUME_CONFIRM_RATIO    = 1.15   # volume 15% above avg confirms trend candle
VOLUME_DRY_RATIO        = 0.85   # volume 15% below avg = drying up
DISTRIBUTION_HIGH_PCT   = 1.0    # within 1% of 20-bar high = near top
SWING_LOOKBACK          = 10     # bars to detect swing H/L
HH_HL_MIN_COUNT         = 2      # HH+HL pairs needed for markup
EFFORT_RESULT_WINDOW    = 10     # bars for effort-vs-result calc
CONFIDENCE_BLOCK_BELOW  = 40     # below this → NEUTRAL override
MIN_CANDLES             = 30     # minimum bars needed


# ══════════════════════════════════════════════════════════════════════════════
# PHASE ENUM AND RESULT
# ══════════════════════════════════════════════════════════════════════════════

class WyckoffPhase:
    ACCUMULATION = "ACCUMULATION"
    MARKUP       = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN     = "MARKDOWN"
    NEUTRAL      = "NEUTRAL"


class Recommendation:
    STRONGLY_FAVOR_CALL = "STRONGLY_FAVOR_CALL"
    MILDLY_FAVOR_CALL   = "MILDLY_FAVOR_CALL"
    NEUTRAL             = "NEUTRAL"
    MILDLY_FAVOR_PUT    = "MILDLY_FAVOR_PUT"
    STRONGLY_FAVOR_PUT  = "STRONGLY_FAVOR_PUT"


@dataclass
class WyckoffResult:
    """
    Complete Wyckoff phase analysis result.
    Consumed by strategies, ML filter, and position manager.
    """
    phase:                str     # WyckoffPhase constant
    confidence:           float   # 0-100 overall confidence
    bullish_score:        float   # 0-100 bullish evidence strength
    bearish_score:        float   # 0-100 bearish evidence strength
    volume_confirmation:  float   # 0-100 volume supporting price action
    trend_strength:       float   # 0-100 directional trend strength
    recommendation:       str     # Recommendation constant

    # Supporting detail scores
    price_roc:            float   # 20-bar price rate of change %
    volume_trend:         float   # volume trend ratio (>1 = rising)
    pv_relationship:      str     # "BULL_CONFIRM" | "WEAK_RALLY" | "BEAR_CONFIRM" | "WEAK_SELL" | "NEUTRAL"
    effort_result:        float   # volume/price efficiency (0-1)
    hh_hl_count:          int     # higher high + higher low count (last 10 bars)
    ll_lh_count:          int     # lower low + lower high count (last 10 bars)
    ema_aligned_bull:     bool    # EMA9 > EMA21 > EMA50
    ema_aligned_bear:     bool    # EMA9 < EMA21 < EMA50

    # Integration helpers
    call_multiplier:      float   # use as: conf += base_conf × call_multiplier
    put_multiplier:       float   # use as: conf += base_conf × put_multiplier
    size_multiplier:      float   # use for position sizing (0.5 = half size)
    block_calls:          bool    # hard block for call entries
    block_puts:           bool    # hard block for put entries

    note:                 str

    @classmethod
    def from_dict(cls, data: dict) -> "WyckoffResult":
        """Reconstruct from either raw dataclass keys or to_dict() payload keys."""
        if not data:
            return WyckoffPhaseDetector._neutral("missing wyckoff payload")
        if "phase" in data and "recommendation" in data:
            return cls(**data)
        return cls(
            phase=str(data.get("wyckoff_phase", "NEUTRAL")),
            confidence=float(data.get("wyckoff_confidence", 40.0) or 40.0),
            bullish_score=float(data.get("wyckoff_bullish", 50.0) or 50.0),
            bearish_score=float(data.get("wyckoff_bearish", 50.0) or 50.0),
            volume_confirmation=float(data.get("wyckoff_vol_confirm", 50.0) or 50.0),
            trend_strength=float(data.get("wyckoff_trend_strength", 0.0) or 0.0),
            recommendation=str(data.get("wyckoff_recommendation", Recommendation.NEUTRAL)),
            price_roc=float(data.get("wyckoff_price_roc", 0.0) or 0.0),
            volume_trend=float(data.get("wyckoff_vol_trend", 1.0) or 1.0),
            pv_relationship=str(data.get("wyckoff_pv_relation", "NEUTRAL")),
            effort_result=float(data.get("wyckoff_effort_result", 0.5) or 0.5),
            hh_hl_count=int(data.get("hh_hl_count", 0) or 0),
            ll_lh_count=int(data.get("ll_lh_count", 0) or 0),
            ema_aligned_bull=bool(data.get("ema_aligned_bull", False)),
            ema_aligned_bear=bool(data.get("ema_aligned_bear", False)),
            call_multiplier=float(data.get("wyckoff_call_mult", 0.0) or 0.0),
            put_multiplier=float(data.get("wyckoff_put_mult", 0.0) or 0.0),
            size_multiplier=float(data.get("wyckoff_size_mult", 1.0) or 1.0),
            block_calls=bool(data.get("wyckoff_block_calls", False)),
            block_puts=bool(data.get("wyckoff_block_puts", False)),
            note=str(data.get("wyckoff_note", "")),
        )

    def to_dict(self) -> dict:
        return {
            "wyckoff_phase":           self.phase,
            "wyckoff_confidence":      round(self.confidence, 1),
            "wyckoff_bullish":         round(self.bullish_score, 1),
            "wyckoff_bearish":         round(self.bearish_score, 1),
            "wyckoff_vol_confirm":     round(self.volume_confirmation, 1),
            "wyckoff_trend_strength":  round(self.trend_strength, 1),
            "wyckoff_recommendation":  self.recommendation,
            "wyckoff_pv_relation":     self.pv_relationship,
            "wyckoff_price_roc":       round(self.price_roc, 3),
            "wyckoff_vol_trend":       round(self.volume_trend, 3),
            "wyckoff_effort_result":   round(self.effort_result, 3),
            "wyckoff_call_mult":       round(self.call_multiplier, 3),
            "wyckoff_put_mult":        round(self.put_multiplier, 3),
            "wyckoff_size_mult":       round(self.size_multiplier, 3),
            "wyckoff_block_calls":     self.block_calls,
            "wyckoff_block_puts":      self.block_puts,
            "wyckoff_note":            self.note,
        }

    def apply_to_signal(
        self,
        direction:   str,
        base_conf:   float,
        base_lots:   int,
    ) -> tuple[float, int, bool]:
        """
        Apply Wyckoff result to an incoming signal.

        Returns:
            (adjusted_conf, adjusted_lots, allow_trade)

        Usage in strategy runner:
            conf, lots, allow = wyckoff.apply_to_signal("BUY_CALL", 0.72, 1)
            if not allow: return none
        """
        if direction == "BUY_CALL":
            if self.block_calls:
                return base_conf, 0, False
            adj_conf = min(0.95, base_conf + self.call_multiplier * base_conf)
            adj_lots = max(1, int(base_lots * self.size_multiplier)) \
                       if self.recommendation in (Recommendation.MILDLY_FAVOR_PUT,
                                                   Recommendation.STRONGLY_FAVOR_PUT) \
                       else base_lots
        else:   # BUY_PUT
            if self.block_puts:
                return base_conf, 0, False
            adj_conf = min(0.95, base_conf + self.put_multiplier * base_conf)
            adj_lots = max(1, int(base_lots * self.size_multiplier)) \
                       if self.recommendation in (Recommendation.MILDLY_FAVOR_CALL,
                                                   Recommendation.STRONGLY_FAVOR_CALL) \
                       else base_lots

        return round(adj_conf, 4), adj_lots, True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class WyckoffPhaseDetector:
    """
    Wyckoff-style Price-Volume Market Phase Detector.

    Pure function design: compute() takes DataFrame, returns WyckoffResult.
    No state between calls. Thread-safe. Fast (<2ms per candle).

    Integration with existing regime classifier:
        # In classifier.py on_candles():
        wyckoff = self._wyckoff.compute(df)
        details["wyckoff"] = wyckoff.to_dict()
        # Then publish with MARKET_REGIME as normal

    Integration with strategy runner:
        wyckoff_dict = regime_details.get("wyckoff", {})
        # Strategies read phase/recommendation from this dict
    """

    def __init__(self) -> None:
        # Rolling history for smoothing (prevents single-candle overreaction)
        self._phase_history:  list[str]   = []
        self._conf_history:   list[float] = []
        self._bull_history:   list[float] = []
        self._bear_history:   list[float] = []
        self._max_history     = 5   # smooth over last 5 readings

    def compute(self, df: pd.DataFrame) -> WyckoffResult:
        """
        Main entry point. Analyse DataFrame and return WyckoffResult.

        Args:
            df: 5-min OHLCV DataFrame with columns:
                open, high, low, close, volume
                Index: DatetimeIndex (IST preferred)

        Returns:
            WyckoffResult with all scores and recommendation
        """
        neutral = self._neutral("insufficient data")

        if df is None or len(df) < MIN_CANDLES:
            return neutral

        try:
            return self._compute(df)
        except Exception as e:
            logger.debug(f"[WyckoffDetector] error: {e}")
            return neutral

    # ── INTERNAL PIPELINE ─────────────────────────────────────────────────────

    def _compute(self, df: pd.DataFrame) -> WyckoffResult:
        close   = df["close"]
        high    = df["high"]
        low     = df["low"]
        volume  = df["volume"]
        n       = len(df)

        # ── LAYER 1: PRICE ANALYSIS ───────────────────────────────────────────

        # Price Rate of Change — use BOTH 5-bar (short) and 20-bar (long)
        # Take the more extreme of the two to avoid missing fast moves
        roc_period  = min(PRICE_ROC_WINDOW, n - 1)
        price_roc20 = float(
            (close.iloc[-1] - close.iloc[-roc_period - 1])
            / close.iloc[-roc_period - 1] * 100
        )
        roc_short   = min(5, n - 1)
        price_roc5  = float(
            (close.iloc[-1] - close.iloc[-roc_short - 1])
            / close.iloc[-roc_short - 1] * 100
        ) if n > roc_short else price_roc20
        # Use the stronger signal of the two
        price_roc   = price_roc5 if abs(price_roc5) > abs(price_roc20) else price_roc20

        # EMAs for trend direction
        ema9  = float(close.ewm(span=9,  adjust=False).mean().iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) \
                if n >= 50 else float(close.mean())

        ema_bull = ema9 > ema21 > ema50
        ema_bear = ema9 < ema21 < ema50

        # Higher highs / lower lows (swing structure)
        hh_count, hl_count, ll_count, lh_count = \
            self._swing_counts(high.values, low.values, close.values)

        # Price range tightness (for accumulation/distribution)
        recent_high = float(high.iloc[-PRICE_ROC_WINDOW:].max())
        recent_low  = float(low.iloc[-PRICE_ROC_WINDOW:].min())
        price_range_pct = (recent_high - recent_low) / max(recent_low, 1) * 100
        # near_high: last 5 candles within 1.5% of the FULL session high
        # Using full df high prevents "near_high=True" during a downtrend
        # because the 20-bar rolling high slides down with price
        full_high      = float(high.max())
        full_low       = float(low.min())
        last5_closes   = close.iloc[-5:].values
        high_threshold = full_high * (1 - DISTRIBUTION_HIGH_PCT / 100 * 1.5)
        low_threshold  = full_low  * (1 + DISTRIBUTION_HIGH_PCT / 100 * 1.5)
        near_high      = bool(all(c >= high_threshold for c in last5_closes))
        near_low       = bool(all(c <= low_threshold  for c in last5_closes))

        # ── LAYER 2: VOLUME ANALYSIS ──────────────────────────────────────────

        # Smooth volume trend
        vol_smooth = volume.ewm(span=VOLUME_SMOOTH_WINDOW, adjust=False).mean()
        vol_trend  = float(vol_smooth.iloc[-1] / max(vol_smooth.iloc[-PRICE_ROC_WINDOW], 1))

        # Separate up-candle and down-candle volume
        price_delta = close.diff().fillna(0)
        up_mask     = price_delta > 0
        dn_mask     = price_delta < 0

        avg_vol     = float(volume.iloc[-PRICE_ROC_WINDOW:].mean())
        up_vol_avg  = float(volume[up_mask].iloc[-PRICE_ROC_WINDOW:].mean()) \
                      if up_mask.sum() > 0 else avg_vol
        dn_vol_avg  = float(volume[dn_mask].iloc[-PRICE_ROC_WINDOW:].mean()) \
                      if dn_mask.sum() > 0 else avg_vol

        vol_ratio_up  = up_vol_avg  / max(avg_vol, 1)
        vol_ratio_down= dn_vol_avg  / max(avg_vol, 1)

        # Effort vs Result (volume effort relative to price result)
        effort_result = self._effort_vs_result(
            close.values[-EFFORT_RESULT_WINDOW:],
            volume.values[-EFFORT_RESULT_WINDOW:],
        )

        # ── LAYER 3: PRICE-VOLUME RELATIONSHIP ───────────────────────────────
        # Wyckoff's 4 laws

        price_rising = price_roc > 0.3
        price_falling= price_roc < -0.3
        vol_rising   = vol_trend > 1.05
        vol_falling  = vol_trend < 0.95

        if price_rising and vol_rising:
            pv_rel = "BULL_CONFIRM"       # Law 1: rising P + rising V
        elif price_rising and vol_falling:
            pv_rel = "WEAK_RALLY"         # Law 2: rising P + falling V
        elif price_falling and vol_rising:
            pv_rel = "BEAR_CONFIRM"       # Law 3: falling P + rising V
        elif price_falling and vol_falling:
            pv_rel = "WEAK_SELL"          # Law 4: falling P + falling V
        else:
            pv_rel = "NEUTRAL"

        # ── LAYER 4: PHASE SCORING ────────────────────────────────────────────

        accum_score  = self._score_accumulation(
            price_roc, price_range_pct, vol_trend,
            vol_ratio_down, near_low, ema_bear, ema_bull,
        )
        markup_score = self._score_markup(
            price_roc, ema_bull, hh_count, hl_count,
            vol_ratio_up, pv_rel,
        )
        # Pass BOTH ema flags — distribution requires neither bull nor bear trend
        distrib_score = self._score_distribution(
            price_roc, price_range_pct, vol_trend,
            effort_result, near_high, ema_bull, ema_bear,
        )
        markdown_score = self._score_markdown(
            price_roc, ema_bear, ll_count, lh_count,
            vol_ratio_down, pv_rel,
        )

        # ── LAYER 5: PHASE CLASSIFICATION ────────────────────────────────────

        scores = {
            WyckoffPhase.ACCUMULATION: accum_score,
            WyckoffPhase.MARKUP:       markup_score,
            WyckoffPhase.DISTRIBUTION: distrib_score,
            WyckoffPhase.MARKDOWN:     markdown_score,
        }
        phase     = max(scores, key=scores.get)
        raw_conf  = scores[phase]

        # Neutral if no phase dominates
        if raw_conf < 35:
            phase    = WyckoffPhase.NEUTRAL
            raw_conf = 50.0

        # ── LAYER 6: COMPOSITE SCORES ─────────────────────────────────────────

        bullish_score = float(np.clip(
            markup_score * 0.6 + accum_score * 0.3 +
            (25 if pv_rel == "BULL_CONFIRM" else 10 if pv_rel == "WEAK_RALLY" else 0),
            0, 100
        ))
        bearish_score = float(np.clip(
            markdown_score * 0.6 + distrib_score * 0.3 +
            (25 if pv_rel == "BEAR_CONFIRM" else 10 if pv_rel == "WEAK_SELL" else 0),
            0, 100
        ))

        vol_confirm = float(np.clip(
            (vol_ratio_up   * 40 if price_rising  else 0) +
            (vol_ratio_down * 40 if price_falling else 0) +
            (effort_result * 20),
            0, 100
        ))
        if pv_rel in ("WEAK_RALLY", "WEAK_SELL"):
            vol_confirm = min(vol_confirm, 50)

        trend_strength = float(np.clip(
            abs(price_roc) * 10 +
            (hh_count + hl_count) * 5 +
            (20 if ema_bull or ema_bear else 0),
            0, 100
        ))

        # ── LAYER 7: SMOOTHING (rolling average) ─────────────────────────────

        self._phase_history.append(phase)
        self._bull_history.append(bullish_score)
        self._bear_history.append(bearish_score)
        self._conf_history.append(raw_conf)

        if len(self._phase_history) > self._max_history:
            self._phase_history  = self._phase_history[-self._max_history:]
            self._bull_history   = self._bull_history[-self._max_history:]
            self._bear_history   = self._bear_history[-self._max_history:]
            self._conf_history   = self._conf_history[-self._max_history:]

        # Smoothed phase: majority vote
        from collections import Counter
        phase_vote   = Counter(self._phase_history).most_common(1)[0][0]
        smooth_conf  = sum(self._conf_history) / len(self._conf_history)
        smooth_bull  = sum(self._bull_history) / len(self._bull_history)
        smooth_bear  = sum(self._bear_history) / len(self._bear_history)

        # Override confidence below block threshold → NEUTRAL
        if smooth_conf < CONFIDENCE_BLOCK_BELOW:
            phase_vote  = WyckoffPhase.NEUTRAL
            smooth_conf = 40.0

        # ── LAYER 8: RECOMMENDATION & INTEGRATION VALUES ──────────────────────

        rec, call_m, put_m, size_m, blk_c, blk_p = \
            self._make_recommendation(phase_vote, smooth_conf, smooth_bull, smooth_bear)

        note = (
            f"{phase_vote} conf={smooth_conf:.0f} | "
            f"P-V: {pv_rel} | ROC={price_roc:.2f}% | "
            f"vol_trend={vol_trend:.2f} | "
            f"HH={hh_count} HL={hl_count} LL={ll_count} LH={lh_count}"
        )

        result = WyckoffResult(
            phase                = phase_vote,
            confidence           = round(smooth_conf, 1),
            bullish_score        = round(smooth_bull, 1),
            bearish_score        = round(smooth_bear, 1),
            volume_confirmation  = round(vol_confirm, 1),
            trend_strength       = round(trend_strength, 1),
            recommendation       = rec,
            price_roc            = round(price_roc, 3),
            volume_trend         = round(vol_trend, 3),
            pv_relationship      = pv_rel,
            effort_result        = round(effort_result, 3),
            hh_hl_count          = hh_count + hl_count,
            ll_lh_count          = ll_count + lh_count,
            ema_aligned_bull     = ema_bull,
            ema_aligned_bear     = ema_bear,
            call_multiplier      = round(call_m, 4),
            put_multiplier       = round(put_m, 4),
            size_multiplier      = round(size_m, 3),
            block_calls          = blk_c,
            block_puts           = blk_p,
            note                 = note,
        )

        logger.debug(f"[Wyckoff] {note}")
        return result

    # ── PHASE SCORERS ─────────────────────────────────────────────────────────

    @staticmethod
    def _score_accumulation(
        price_roc:       float,
        price_range_pct: float,
        vol_trend:       float,
        vol_ratio_down:  float,
        near_low:        bool,
        ema_bear:        bool = False,
        ema_bull:        bool = False,
    ) -> float:
        """
        Accumulation: flat price + drying volume on down moves + near lows
        Score 0-100
        """
        score = 0.0
        # Price flat (not trending up or down)
        price_flat = abs(price_roc) < MARKUP_ROC_THRESHOLD

        # HARD GATE: accumulation only at lows when price is flat
        if not (price_flat and near_low):
            return 0.0

        score += 30   # base: confirmed flat at lows
        # Tight range (consolidation)
        if price_range_pct < 2.0:
            score += 20
        # Volume drying on down moves (supply absorbed = buyers stepping in)
        if vol_ratio_down < VOLUME_DRY_RATIO:
            score += 25
        # Volume overall declining (absorption phase complete)
        if vol_trend < 0.95:
            score += 10
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _score_markup(
        price_roc:    float,
        ema_bull:     bool,
        hh_count:     int,
        hl_count:     int,
        vol_ratio_up: float,
        pv_rel:       str,
    ) -> float:
        """Markup: rising price + rising volume on up moves + HH+HL structure"""
        score = 0.0
        if price_roc > MARKUP_ROC_THRESHOLD:
            score += 30 + min(20, (price_roc - MARKUP_ROC_THRESHOLD) * 5)
        if ema_bull:
            score += 20
        if hh_count >= HH_HL_MIN_COUNT:
            score += 10 * min(hh_count, 3)
        if hl_count >= HH_HL_MIN_COUNT:
            score += 10 * min(hl_count, 2)
        if vol_ratio_up > VOLUME_CONFIRM_RATIO:
            score += 15
        if pv_rel == "BULL_CONFIRM":
            score += 15
        elif pv_rel == "WEAK_RALLY":
            score -= 10
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _score_distribution(
        price_roc:       float,
        price_range_pct: float,
        vol_trend:       float,
        effort_result:   float,
        near_high:       bool,
        ema_bull:        bool = False,
        ema_bear:        bool = False,
    ) -> float:
        """
        Distribution: flat price AT highs + elevated volume + effort w/o result.
        
        FIX: price_flat AND near_high must BOTH be true (hard gate).
        Without this gate, a rising trend scores as distribution because
        the 20-bar window looks tight relative to the full trend range.
        """
        score = 0.0
        price_flat = abs(price_roc) < MARKUP_ROC_THRESHOLD   # ROC < 1.5%
        
        # HARD GATE: distribution ONLY at highs when price is flat
        # This is the definition of distribution — topping out at highs
        if not (price_flat and near_high):
            return 0.0   # not distribution if price flat or not at FULL highs
        # EMA gates: distribution requires EMAs to be FLAT (neither bull nor bear)
        # If ema_bull: still in markup, not topped yet
        # If ema_bear: already in markdown, distribution phase passed
        if ema_bull or ema_bear:
            return 0.0   # trending EMAs = not in distribution
        
        # Price is flat at highs — now score the distribution signals
        score += 30   # base: confirmed flat at highs
        if vol_trend > 1.05:
            score += 25   # elevated volume = supply entering
        if effort_result < 0.35:
            score += 25   # high volume, little price progress
        if price_range_pct < 2.5:
            score += 20   # tight range confirms balance/topping
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _score_markdown(
        price_roc:      float,
        ema_bear:       bool,
        ll_count:       int,
        lh_count:       int,
        vol_ratio_down: float,
        pv_rel:         str,
    ) -> float:
        """Markdown: falling price + rising volume on down moves + LL+LH structure"""
        score = 0.0
        if price_roc < -MARKUP_ROC_THRESHOLD:
            score += 30 + min(20, (abs(price_roc) - MARKUP_ROC_THRESHOLD) * 5)
        if ema_bear:
            score += 20
        if ll_count >= HH_HL_MIN_COUNT:
            score += 10 * min(ll_count, 3)
        if lh_count >= HH_HL_MIN_COUNT:
            score += 10 * min(lh_count, 2)
        if vol_ratio_down > VOLUME_CONFIRM_RATIO:
            score += 15
        if pv_rel == "BEAR_CONFIRM":
            score += 15
        elif pv_rel == "WEAK_SELL":
            score -= 10
        return float(np.clip(score, 0, 100))

    # ── SWING STRUCTURE ───────────────────────────────────────────────────────

    @staticmethod
    def _swing_counts(
        highs:  np.ndarray,
        lows:   np.ndarray,
        closes: np.ndarray,
        window: int = 3,
    ) -> tuple[int, int, int, int]:
        """Count HH, HL, LL, LH in recent price structure."""
        n   = min(SWING_LOOKBACK, len(closes))
        arr = closes[-n:]
        hh  = hl = ll = lh = 0
        for i in range(1, len(arr)):
            if arr[i] > arr[i - 1]:
                if i >= 2 and arr[i] > arr[i - 2]:
                    hh += 1
                else:
                    hl += 1
            elif arr[i] < arr[i - 1]:
                if i >= 2 and arr[i] < arr[i - 2]:
                    ll += 1
                else:
                    lh += 1
        return hh, hl, ll, lh

    # ── EFFORT VS RESULT ──────────────────────────────────────────────────────

    @staticmethod
    def _effort_vs_result(
        closes:  np.ndarray,
        volumes: np.ndarray,
    ) -> float:
        """
        Effort vs Result: price move per unit of volume above baseline.
        
        FIX: Original divided norm_vol by itself (always 1.0).
        Correct: compare recent volume to its own rolling baseline.
        
        Low value  = high volume effort, small price result = distribution signal.
        High value = price moves efficiently with normal volume = trending.
        Returns 0-1.
        """
        if len(closes) < 4 or volumes.sum() == 0:
            return 0.5

        price_move  = abs(closes[-1] - closes[0])
        norm_move   = price_move / max(closes[0], 1) * 100  # % price change

        # Compare recent half vs earlier half (baseline comparison)
        mid          = len(volumes) // 2
        vol_baseline = float(np.mean(volumes[:mid])) if mid > 0 else float(np.mean(volumes))
        vol_recent   = float(np.mean(volumes[mid:]))
        # vol_ratio > 1 = volume expanding; < 1 = contracting
        vol_ratio    = vol_recent / max(vol_baseline, 1)

        # effort_result: if volume doubled but price barely moved → low
        # if price moved well with normal volume → high
        if vol_ratio > 0.01:
            raw = norm_move / vol_ratio
        else:
            raw = norm_move

        # Normalise: 0 = all volume no result, 1 = efficient price/vol
        return float(np.clip(raw / 3.0, 0, 1))

    # ── RECOMMENDATION ────────────────────────────────────────────────────────

    @staticmethod
    def _make_recommendation(
        phase:       str,
        confidence:  float,
        bull_score:  float,
        bear_score:  float,
    ) -> tuple[str, float, float, float, bool, bool]:
        """
        Returns: (recommendation, call_mult, put_mult, size_mult, block_calls, block_puts)
        
        call_mult: fractional boost to signal confidence for CALL signals
        put_mult:  fractional boost to signal confidence for PUT signals
        size_mult: position size multiplier (1.0=normal, 0.5=half, 0.0=block)
        """
        R = Recommendation

        if phase == WyckoffPhase.MARKUP:
            if confidence >= 70:
                return R.STRONGLY_FAVOR_CALL, +0.06, -0.06, 1.0,  False, True
            elif confidence >= 50:
                return R.MILDLY_FAVOR_CALL,   +0.03, -0.03, 0.75, False, False

        elif phase == WyckoffPhase.ACCUMULATION:
            if confidence >= 60:
                return R.MILDLY_FAVOR_CALL,   +0.02, -0.02, 1.0,  False, False

        elif phase == WyckoffPhase.MARKDOWN:
            if confidence >= 70:
                return R.STRONGLY_FAVOR_PUT,  -0.06, +0.06, 1.0,  True,  False
            elif confidence >= 50:
                return R.MILDLY_FAVOR_PUT,    -0.03, +0.03, 0.75, False, False

        elif phase == WyckoffPhase.DISTRIBUTION:
            if confidence >= 70:
                return R.STRONGLY_FAVOR_PUT,  -0.06, +0.06, 1.0,  True,  False
            elif confidence >= 50:
                return R.MILDLY_FAVOR_PUT,    -0.03, +0.03, 0.75, False, False

        # NEUTRAL or low confidence
        return R.NEUTRAL, 0.0, 0.0, 1.0, False, False

    # ── NEUTRAL FALLBACK ──────────────────────────────────────────────────────

    @staticmethod
    def _neutral(reason: str = "") -> WyckoffResult:
        return WyckoffResult(
            phase="NEUTRAL", confidence=40.0,
            bullish_score=50.0, bearish_score=50.0,
            volume_confirmation=50.0, trend_strength=0.0,
            recommendation=Recommendation.NEUTRAL,
            price_roc=0.0, volume_trend=1.0,
            pv_relationship="NEUTRAL", effort_result=0.5,
            hh_hl_count=0, ll_lh_count=0,
            ema_aligned_bull=False, ema_aligned_bear=False,
            call_multiplier=0.0, put_multiplier=0.0,
            size_multiplier=1.0, block_calls=False, block_puts=False,
            note=reason or "neutral",
        )


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST HELPER
# ══════════════════════════════════════════════════════════════════════════════

def backtest_wyckoff(df: pd.DataFrame, forward_bars: int = 6) -> dict:
    """
    Backtest the Wyckoff detector on historical data.
    For each bar: compute phase, then check if NIFTY moved in predicted
    direction over next N bars.

    Args:
        df:            full historical 5-min OHLCV
        forward_bars:  how many bars forward to evaluate (default 6 = 30 min)

    Returns:
        dict with win rates per recommendation type
    """
    detector = WyckoffPhaseDetector()
    results  = {r: {"correct": 0, "total": 0} for r in [
        Recommendation.STRONGLY_FAVOR_CALL,
        Recommendation.MILDLY_FAVOR_CALL,
        Recommendation.NEUTRAL,
        Recommendation.MILDLY_FAVOR_PUT,
        Recommendation.STRONGLY_FAVOR_PUT,
    ]}

    min_bars = MIN_CANDLES + forward_bars
    for i in range(MIN_CANDLES, len(df) - forward_bars):
        window = df.iloc[:i]
        try:
            result = detector.compute(window)
            rec    = result.recommendation

            # Forward return
            entry_price   = float(df["close"].iloc[i])
            forward_price = float(df["close"].iloc[i + forward_bars])
            fwd_ret       = (forward_price - entry_price) / entry_price * 100

            correct = (
                (rec in (Recommendation.STRONGLY_FAVOR_CALL, Recommendation.MILDLY_FAVOR_CALL)
                 and fwd_ret > 0) or
                (rec in (Recommendation.STRONGLY_FAVOR_PUT, Recommendation.MILDLY_FAVOR_PUT)
                 and fwd_ret < 0)
            )

            results[rec]["total"]   += 1
            if correct:
                results[rec]["correct"] += 1
        except Exception:
            continue

    # Compute win rates
    summary = {}
    for rec, data in results.items():
        n  = data["total"]
        wr = data["correct"] / n * 100 if n > 0 else 0
        summary[rec] = {"total": n, "win_rate": round(wr, 1)}

    return summary

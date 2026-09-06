"""
S9: Ichimoku Cloud  Full Confluence Strategy
===============================================
Ichimoku Kinko Hyo ("equilibrium chart at a glance") uses 5 lines
to show support/resistance, trend direction, and momentum simultaneously.
It is one of the most comprehensive single-indicator systems.

The 5 Lines:
    Tenkan-sen  (Conversion):  (9-period H+L) / 2
    Kijun-sen   (Base):        (26-period H+L) / 2
    Senkou A    (Leading A):   (Tenkan + Kijun) / 2  [plotted 26 ahead]
    Senkou B    (Leading B):   (52-period H+L) / 2   [plotted 26 ahead]
    Chikou     (Lagging):      Close plotted 26 periods behind

The "Cloud" (Kumo) = area between Senkou A and Senkou B
     Bullish cloud: Senkou A > Senkou B (green cloud)
     Bearish cloud: Senkou A < Senkou B (red cloud)

NIFTY-specific adaptation for 5-min intraday:
    Standard Ichimoku uses 9/26/52 (designed for daily).
    For 5-min NIFTY intraday, we use scaled parameters:
        Tenkan:  9   (45 min  short-term momentum)
        Kijun:   26  (130 min  medium-term baseline)
        Senkou B: 52 (260 min  longer cloud boundary)
    These are the standard settings  they work well on 5-min NIFTY.

Signal Logic  5 confluence checks (strongest when all agree):

    STRONG BUY CALL (all 5 conditions):
        1. Price ABOVE the cloud (bullish structure)
        2. Cloud ahead is BULLISH (Senkou A > Senkou B)
        3. Tenkan-sen ABOVE Kijun-sen (TK cross bullish)
        4. Tenkan crossed ABOVE Kijun in last 3 candles (fresh signal)
        5. Chikou (lagging close) ABOVE price 26 periods ago

    STRONG BUY PUT (all 5 conditions inverted):
        1. Price BELOW the cloud
        2. Cloud ahead is BEARISH (Senkou A < Senkou B)
        3. Tenkan BELOW Kijun
        4. Tenkan crossed BELOW Kijun recently
        5. Chikou BELOW price 26 periods ago

    MINIMUM REQUIREMENT (3 of 5 conditions):
        Signal fires with lower confidence if 3+ conditions align
        but not all 5. This catches early entries.

Confidence scoring:
    Number of conditions met:
        5/5  base 0.76 (strong confluence)
        4/5  base 0.70
        3/5  base 0.64
    + 0.04 if price is far above/below cloud (strong breakout)
    + 0.03 if cloud is thick (strong support/resistance)
    + 0.02 if volume > 1.3 average
    Cap: 0.85

Why Ichimoku is orthogonal to S1-S8:
    - CPR (S8) uses inter-session price levels
    - Ichimoku uses CURRENT session's price action across 5 timeframes
      simultaneously  it sees support, resistance, trend, AND momentum
      in one calculation. No other strategy does this.
    - The cloud's forward projection is unique  no other indicator
      in S1-S7 uses future price levels.

Parameters:
    ICHI_TENKAN   = 9    Conversion line period
    ICHI_KIJUN    = 26   Base line period
    ICHI_SENKOU_B = 52   Slower cloud boundary period
    ICHI_CHIKOU   = 26   Lagging span shift
    ICHI_VOL_MULT = 1.3  Volume confirmation threshold
    ICHI_MIN_COND = 3    Minimum conditions to fire (out of 5)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from core.models import Direction
from config.settings.strategy import (
    S9_CHIKOU,
    S9_CONF_BASE_3,
    S9_CONF_BASE_4,
    S9_CONF_BASE_5,
    S9_CONF_BASE_DEFAULT,
    S9_CONF_DIST_BONUS_1,
    S9_CONF_DIST_BONUS_1_THRESHOLD,
    S9_CONF_DIST_BONUS_2,
    S9_CONF_DIST_BONUS_2_THRESHOLD,
    S9_CONF_FRESH_CROSS_BONUS,
    S9_CONF_MAX,
    S9_CONF_THICK_CLOUD_BONUS_1,
    S9_CONF_THICK_CLOUD_BONUS_1_THRESHOLD,
    S9_CONF_THICK_CLOUD_BONUS_2,
    S9_CONF_THICK_CLOUD_BONUS_2_THRESHOLD,
    S9_CONF_VOL_BONUS,
    S9_CONF_VOL_BONUS_THRESHOLD,
    S9_FRESH_TK_CROSS,
    S9_FRESH_TK_CROSS_BONUS_BARS,
    S9_KIJUN,
    S9_MIN_CONDITIONS,
    S9_SENKOU_B,
    S9_TENKAN,
    S9_TK_CROSS_LOOKBACK,
    S9_VOL_MA_PERIOD,
    S9_VOL_MULT,
)


@dataclass
class IchimokuLevels:
    """All 5 Ichimoku components at current candle."""
    tenkan:        float   # Conversion line
    kijun:         float   # Base line
    senkou_a:      float   # Leading span A (current)
    senkou_b:      float   # Leading span B (current)
    chikou:        float   # Lagging span (current close)
    chikou_ref:    float   # Price 26 bars ago (chikou comparison)
    future_a:      float   # Senkou A projected 26 bars ahead
    future_b:      float   # Senkou B projected 26 bars ahead
    close:         float   # Current close
    cloud_top:     float   # max(senkou_a, senkou_b)
    cloud_bottom:  float   # min(senkou_a, senkou_b)
    cloud_thick_pct: float # cloud thickness as % of price
    tk_cross_bars: int     # bars since last TK cross (0 = no recent cross)


class IchimokuStrategy:
    name = "Ichimoku"

    #  PUBLIC ENTRY POINT 

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None) -> dict:
        """Standard interface  returns direction, confidence, name, meta."""
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        # Need at least 52 + 26 candles for full Ichimoku
        if len(df) < S9_SENKOU_B + S9_KIJUN + 10:
            return none

        ichi = self._compute(df)
        if ichi is None:
            return none

        # Volume check
        vol    = float(df["volume"].iloc[-1])
        vol_ma = float(df["volume"].rolling(S9_VOL_MA_PERIOD).mean().iloc[-1])
        if vol_ma <= 0:
            return none
        vol_ratio = vol / vol_ma

        # Evaluate bullish and bearish conditions
        bull_score, bull_conditions = self._score_bullish(ichi, vol_ratio)
        bear_score, bear_conditions = self._score_bearish(ichi, vol_ratio)

        # Need minimum conditions met
        if bull_score >= S9_MIN_CONDITIONS and bull_score > bear_score:
            conf = self._confidence(bull_score, ichi, vol_ratio, "bull")
            return {
                "direction":  Direction.BUY_CALL,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "conditions_met": bull_score,
                    "tenkan":         round(ichi.tenkan, 2),
                    "kijun":          round(ichi.kijun, 2),
                    "cloud_top":      round(ichi.cloud_top, 2),
                    "cloud_bottom":   round(ichi.cloud_bottom, 2),
                    "cloud_pct":      ichi.cloud_thick_pct,
                    "tk_cross_bars":  ichi.tk_cross_bars,
                    "vol_ratio":      round(vol_ratio, 2),
                    "met":            bull_conditions,
                },
            }

        if bear_score >= S9_MIN_CONDITIONS and bear_score > bull_score:
            conf = self._confidence(bear_score, ichi, vol_ratio, "bear")
            return {
                "direction":  Direction.BUY_PUT,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "conditions_met": bear_score,
                    "tenkan":         round(ichi.tenkan, 2),
                    "kijun":          round(ichi.kijun, 2),
                    "cloud_top":      round(ichi.cloud_top, 2),
                    "cloud_bottom":   round(ichi.cloud_bottom, 2),
                    "cloud_pct":      ichi.cloud_thick_pct,
                    "tk_cross_bars":  ichi.tk_cross_bars,
                    "vol_ratio":      round(vol_ratio, 2),
                    "met":            bear_conditions,
                },
            }

        return none

    #  ICHIMOKU COMPUTATION 

    def _compute(self, df: pd.DataFrame) -> Optional[IchimokuLevels]:
        """
        Compute all 5 Ichimoku components.
        Uses midpoint of H+L for each period (standard Ichimoku math).
        """
        try:
            h = df["high"]
            l = df["low"]
            c = df["close"]
            n = len(df)

            def midpoint(period: int, offset: int = 0) -> pd.Series:
                h_ = h.shift(offset)
                l_ = l.shift(offset)
                return (h_.rolling(period).max() + l_.rolling(period).min()) / 2

            tenkan  = midpoint(S9_TENKAN)
            kijun   = midpoint(S9_KIJUN)

            # Senkou A = (Tenkan + Kijun) / 2  use 26-bar-ago values
            senkou_a_series = (tenkan + kijun) / 2
            senkou_b_series = midpoint(S9_SENKOU_B)

            # Current cloud = senkou shifted BACK by 26 (what was plotted 26 ago)
            # i.e., senkou_a[now] was computed exactly 26 bars ago (index -27 from current bar -1)
            if n < S9_SENKOU_B + S9_KIJUN:
                return None

            current_senkou_a = float(senkou_a_series.iloc[-S9_KIJUN - 1])
            current_senkou_b = float(senkou_b_series.iloc[-S9_KIJUN - 1])

            # Future cloud = current Senkou A/B (they are plotted 26 ahead into future)
            future_a = float(senkou_a_series.iloc[-1])
            future_b = float(senkou_b_series.iloc[-1])

            # Chikou = current close compared to price 26 bars ago
            chikou_ref = float(c.iloc[-S9_CHIKOU - 1]) if n > S9_CHIKOU else float(c.iloc[0])

            t_now = float(tenkan.iloc[-1])
            k_now = float(kijun.iloc[-1])

            # Find last TK cross (how many bars ago)
            tk_cross_bars = 0
            for i in range(1, min(S9_TK_CROSS_LOOKBACK, n)):
                t_i   = float(tenkan.iloc[-i])
                k_i   = float(kijun.iloc[-i])
                t_i1  = float(tenkan.iloc[-i-1])
                k_i1  = float(kijun.iloc[-i-1])
                crossed = (t_i > k_i and t_i1 <= k_i1) or (t_i < k_i and t_i1 >= k_i1)
                if crossed:
                    tk_cross_bars = i
                    break

            close_now = float(c.iloc[-1])
            cloud_top    = max(current_senkou_a, current_senkou_b)
            cloud_bottom = min(current_senkou_a, current_senkou_b)
            cloud_thick  = (cloud_top - cloud_bottom) / max(close_now, 1) * 100

            # Validate  skip if any NaN
            vals = [t_now, k_now, current_senkou_a, current_senkou_b,
                    future_a, future_b, chikou_ref]
            if any(np.isnan(v) for v in vals):
                return None

            return IchimokuLevels(
                tenkan        = t_now,
                kijun         = k_now,
                senkou_a      = current_senkou_a,
                senkou_b      = current_senkou_b,
                chikou        = close_now,
                chikou_ref    = chikou_ref,
                future_a      = future_a,
                future_b      = future_b,
                close         = close_now,
                cloud_top     = cloud_top,
                cloud_bottom  = cloud_bottom,
                cloud_thick_pct = round(cloud_thick, 3),
                tk_cross_bars = tk_cross_bars,
            )
        except Exception:
            return None

    #  CONDITION SCORING 

    @staticmethod
    def _score_bullish(ichi: IchimokuLevels, vol_ratio: float) -> tuple[int, list]:
        """Count how many of the 5 bullish conditions are met."""
        conditions = []
        score = 0

        # C1: Price above cloud
        if ichi.close > ichi.cloud_top:
            score += 1
            conditions.append("above_cloud")

        # C2: Future cloud is bullish
        if ichi.future_a > ichi.future_b:
            score += 1
            conditions.append("bullish_cloud_ahead")

        # C3: Tenkan above Kijun (TK bullish)
        if ichi.tenkan > ichi.kijun:
            score += 1
            conditions.append("tk_bullish")

        # C4: Fresh TK cross (within last 5 candles)
        if 0 < ichi.tk_cross_bars <= S9_FRESH_TK_CROSS and ichi.tenkan > ichi.kijun:
            score += 1
            conditions.append(f"fresh_tk_cross_{ichi.tk_cross_bars}bars")

        # C5: Chikou above price 26 bars ago
        if ichi.chikou > ichi.chikou_ref:
            score += 1
            conditions.append("chikou_bullish")

        return score, conditions

    @staticmethod
    def _score_bearish(ichi: IchimokuLevels, vol_ratio: float) -> tuple[int, list]:
        """Count how many of the 5 bearish conditions are met."""
        conditions = []
        score = 0

        # C1: Price below cloud
        if ichi.close < ichi.cloud_bottom:
            score += 1
            conditions.append("below_cloud")

        # C2: Future cloud is bearish
        if ichi.future_a < ichi.future_b:
            score += 1
            conditions.append("bearish_cloud_ahead")

        # C3: Tenkan below Kijun
        if ichi.tenkan < ichi.kijun:
            score += 1
            conditions.append("tk_bearish")

        # C4: Fresh TK cross (within last 5 candles)
        if 0 < ichi.tk_cross_bars <= S9_FRESH_TK_CROSS and ichi.tenkan < ichi.kijun:
            score += 1
            conditions.append(f"fresh_tk_cross_{ichi.tk_cross_bars}bars")

        # C5: Chikou below price 26 bars ago
        if ichi.chikou < ichi.chikou_ref:
            score += 1
            conditions.append("chikou_bearish")

        return score, conditions

    #  CONFIDENCE SCORING 

    @staticmethod
    def _confidence(
        conditions: int,
        ichi:       IchimokuLevels,
        vol_ratio:  float,
        side:       str,
    ) -> float:
        """Confidence scales with number of conditions met."""
        base = {5: S9_CONF_BASE_5, 4: S9_CONF_BASE_4, 3: S9_CONF_BASE_3}.get(conditions, S9_CONF_BASE_DEFAULT)
        conf = base

        # Bonus: price well outside cloud (strong breakout)
        if side == "bull":
            dist_pct = (ichi.close - ichi.cloud_top) / max(ichi.cloud_top, 1) * 100
        else:
            dist_pct = (ichi.cloud_bottom - ichi.close) / max(ichi.cloud_bottom, 1) * 100

        if dist_pct > S9_CONF_DIST_BONUS_1_THRESHOLD:
            conf += S9_CONF_DIST_BONUS_1
        elif dist_pct > S9_CONF_DIST_BONUS_2_THRESHOLD:
            conf += S9_CONF_DIST_BONUS_2

        # Bonus: thick cloud = stronger support/resistance
        if ichi.cloud_thick_pct > S9_CONF_THICK_CLOUD_BONUS_1_THRESHOLD:
            conf += S9_CONF_THICK_CLOUD_BONUS_1
        elif ichi.cloud_thick_pct > S9_CONF_THICK_CLOUD_BONUS_2_THRESHOLD:
            conf += S9_CONF_THICK_CLOUD_BONUS_2

        # Volume bonus
        if vol_ratio >= S9_CONF_VOL_BONUS_THRESHOLD:
            conf += S9_CONF_VOL_BONUS

        # Fresh TK cross bonus
        if ichi.tk_cross_bars in S9_FRESH_TK_CROSS_BONUS_BARS:
            conf += S9_CONF_FRESH_CROSS_BONUS

        return round(min(S9_CONF_MAX, conf), 4)

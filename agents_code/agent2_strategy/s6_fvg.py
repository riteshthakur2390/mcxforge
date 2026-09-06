"""
S6: Fair Value Gap (FVG)  Smart Money Concept
================================================
A Fair Value Gap is a 3-candle imbalance pattern used in Smart Money
Concepts (SMC) / ICT trading methodology.

What is an FVG?
    Bullish FVG: Candle[i-2].high < Candle[i].low
         Price gapped up so fast that an unfilled gap exists between
          the high of 2 candles ago and the low of the current candle.
         When price retraces INTO this gap and holds, it signals
          continuation of the upward move.

    Bearish FVG: Candle[i-2].low > Candle[i].high
         Price gapped down so fast that an unfilled gap exists.
         When price retraces INTO this gap and holds, it signals
          continuation of the downward move.

Signal logic for NIFTY:
    BULLISH FVG + entry signal:
        1. Identify most recent bullish FVG in last N candles
        2. Current price has retraced INTO the FVG zone (touching it)
        3. Current candle closes ABOVE the FVG midpoint (gap held)
        4. Volume on current candle > 1.2 20-bar average (confirmation)
        5. Trending context: price above EMA20 (macro bias is up)
         BUY CALL

    BEARISH FVG + entry signal:
        1. Identify most recent bearish FVG in last N candles
        2. Current price has retraced INTO the FVG zone (touching it)
        3. Current candle closes BELOW the FVG midpoint (gap held)
        4. Volume confirmation
        5. Price below EMA20
         BUY PUT

Confidence scoring:
    Base: 0.64
    +0.04 if FVG size > 0.15% of price (meaningful gap, not noise)
    +0.03 if price bounced cleanly off FVG lower edge (not mid-zone)
    +0.02 if volume > 1.5 average (strong confirmation)
    Cap: 0.82

Parameters (all tunable in config/settings/):
    FVG_LOOKBACK        = 20   candles to scan for FVGs
    FVG_MIN_GAP_PCT     = 0.05 minimum gap size as % of price
    FVG_VOL_MULTIPLIER  = 1.2  volume confirmation threshold
    FVG_EMA_PERIOD      = 20   EMA for macro trend filter

Valid window: 09:3015:00 IST (same as all strategies)
No time restriction like S3  FVGs are valid all session.
"""

import pandas as pd
import pandas_ta as ta
from dataclasses import dataclass
from core.models import Direction
from config.settings.strategy import (
    S6_FVG_LOOKBACK, S6_MIN_GAP_PCT, S6_VOL_MULT, S6_VOL_MA_PERIOD, S6_EMA_PERIOD,
    S6_RETRACE_BULL_FACTOR, S6_RETRACE_BEAR_FACTOR, S6_ZONE_BULL_FACTOR,
    S6_ZONE_BEAR_FACTOR, S6_CONF_BASE, S6_CONF_GAP_PCT_BONUS_1_THRESHOLD,
    S6_CONF_GAP_PCT_BONUS_1, S6_CONF_GAP_PCT_BONUS_2_THRESHOLD, S6_CONF_GAP_PCT_BONUS_2,
    S6_CONF_PROXIMITY_BONUS_THRESHOLD, S6_CONF_PROXIMITY_BONUS,
    S6_CONF_VOL_BONUS_1_THRESHOLD, S6_CONF_VOL_BONUS_1, S6_CONF_VOL_BONUS_2_THRESHOLD,
    S6_CONF_VOL_BONUS_2, S6_CONF_MAX, S6_MIN_DF_OFFSET, S6_WINDOW_OFFSET,
    S6_PATTERN_SIZE, S6_EPSILON
)


@dataclass
class FVGZone:
    """Represents a single identified Fair Value Gap."""
    kind:      str     # "bullish" or "bearish"
    top:       float   # upper boundary of the gap
    bottom:    float   # lower boundary of the gap
    midpoint:  float   # (top + bottom) / 2
    gap_size:  float   # top - bottom in points
    gap_pct:   float   # gap_size / price  100
    candle_idx: int    # index in df where FVG was formed


class FVGStrategy:
    name = "FVG"

    #  PUBLIC ENTRY POINT 

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None) -> dict:
        """
        Standard interface matching all other strategies.
        Returns dict with direction, confidence, name, meta.
        """
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        if len(df) < S6_FVG_LOOKBACK + S6_MIN_DF_OFFSET:
            return none

        close   = float(df["close"].iloc[-1])
        vol     = float(df["volume"].iloc[-1])
        vol_ma  = float(df["volume"].rolling(S6_VOL_MA_PERIOD).mean().iloc[-1])

        if vol_ma <= 0:
            return none

        vol_ratio = vol / vol_ma

        # Volume gate  minimum confirmation required
        if vol_ratio < S6_VOL_MULT:
            return none

        # EMA trend filter
        ema     = ta.ema(df["close"], S6_EMA_PERIOD)
        ema_val = float(ema.iloc[-1]) if ema is not None and not ema.dropna().empty else close

        # Scan recent candles for FVG zones
        window = df.tail(S6_FVG_LOOKBACK + S6_WINDOW_OFFSET)
        fvg_zones = self._scan_fvgs(window, close)

        if not fvg_zones:
            return none

        # Check each FVG zone for valid entry
        for zone in fvg_zones:
            result = self._check_entry(zone, close, ema_val, vol_ratio, df)
            if result["direction"] != Direction.NONE:
                return result

        return none

    #  FVG SCANNING 

    def _scan_fvgs(self, df: pd.DataFrame, current_price: float) -> list[FVGZone]:
        """
        Scan the window for valid FVG zones where price has recently
        retraced into the gap. Returns list ordered most-recent first.
        """
        zones = []
        candles = df.reset_index(drop=True)
        n = len(candles)

        # Need at least 3 candles for FVG pattern
        for i in range(S6_PATTERN_SIZE - 1, n):
            c0 = candles.iloc[i - (S6_PATTERN_SIZE - 1)]   # two candles ago
            c1 = candles.iloc[i - 1]   # middle candle (impulse)
            c2 = candles.iloc[i]       # current pattern candle

            #  Bullish FVG 
            # Gap between high of c0 and low of c2
            if float(c2["low"]) > float(c0["high"]):
                gap_bottom = float(c0["high"])
                gap_top    = float(c2["low"])
                gap_size   = gap_top - gap_bottom
                gap_pct    = gap_size / max(float(c1["close"]), 1.0) * 100.0

                if gap_pct >= S6_MIN_GAP_PCT:
                    # Only valid if current price is NEAR or IN the gap
                    # (retrace happened or is happening)
                    if current_price <= gap_top * S6_RETRACE_BULL_FACTOR:
                        zones.append(FVGZone(
                            kind       = "bullish",
                            top        = gap_top,
                            bottom     = gap_bottom,
                            midpoint   = (gap_top + gap_bottom) / 2.0,
                            gap_size   = gap_size,
                            gap_pct    = round(gap_pct, 3),
                            candle_idx = i,
                        ))

            #  Bearish FVG 
            # Gap between low of c0 and high of c2
            elif float(c2["high"]) < float(c0["low"]):
                gap_top    = float(c0["low"])
                gap_bottom = float(c2["high"])
                gap_size   = gap_top - gap_bottom
                gap_pct    = gap_size / max(float(c1["close"]), 1.0) * 100.0

                if gap_pct >= S6_MIN_GAP_PCT:
                    # Current price near or in the gap
                    if current_price >= gap_bottom * S6_RETRACE_BEAR_FACTOR:
                        zones.append(FVGZone(
                            kind       = "bearish",
                            top        = gap_top,
                            bottom     = gap_bottom,
                            midpoint   = (gap_top + gap_bottom) / 2.0,
                            gap_size   = gap_size,
                            gap_pct    = round(gap_pct, 3),
                            candle_idx = i,
                        ))

        # Most recent FVG first
        return sorted(zones, key=lambda z: z.candle_idx, reverse=True)

    #  ENTRY CHECK 

    def _check_entry(
        self,
        zone:      FVGZone,
        close:     float,
        ema_val:   float,
        vol_ratio: float,
        df:        pd.DataFrame,
    ) -> dict:
        """
        Check if current price action gives a valid entry signal
        for this FVG zone.
        """
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        #  BULLISH FVG ENTRY 
        if zone.kind == "bullish":
            # Price must have touched or entered the FVG zone
            price_in_zone = zone.bottom <= close <= zone.top * S6_ZONE_BULL_FACTOR
            # Price must be closing above midpoint (gap holding as support)
            holding_above_mid = close >= zone.midpoint
            # Macro trend must be up
            bullish_trend = close >= ema_val * S6_RETRACE_BEAR_FACTOR

            if price_in_zone and holding_above_mid and bullish_trend:
                conf = self._score(zone, vol_ratio, "bullish", close)
                return {
                    "direction":  Direction.BUY_CALL,
                    "confidence": conf,
                    "name":       self.name,
                    "meta": {
                        "fvg_type":    "bullish",
                        "fvg_top":     round(zone.top, 2),
                        "fvg_bottom":  round(zone.bottom, 2),
                        "fvg_gap_pct": zone.gap_pct,
                        "vol_ratio":   round(vol_ratio, 2),
                    },
                }

        #  BEARISH FVG ENTRY 
        elif zone.kind == "bearish":
            price_in_zone     = zone.bottom * S6_ZONE_BEAR_FACTOR <= close <= zone.top
            holding_below_mid = close <= zone.midpoint
            bearish_trend     = close <= ema_val * S6_RETRACE_BULL_FACTOR

            if price_in_zone and holding_below_mid and bearish_trend:
                conf = self._score(zone, vol_ratio, "bearish", close)
                return {
                    "direction":  Direction.BUY_PUT,
                    "confidence": conf,
                    "name":       self.name,
                    "meta": {
                        "fvg_type":    "bearish",
                        "fvg_top":     round(zone.top, 2),
                        "fvg_bottom":  round(zone.bottom, 2),
                        "fvg_gap_pct": zone.gap_pct,
                        "vol_ratio":   round(vol_ratio, 2),
                    },
                }

        return none

    #  CONFIDENCE SCORING 

    @staticmethod
    def _score(
        zone:      FVGZone,
        vol_ratio: float,
        kind:      str,
        close:     float,
    ) -> float:
        """
        Score this FVG signal.
        Base 0.64 + bonuses for quality signals.
        """
        conf = S6_CONF_BASE

        # Bonus: meaningful gap size (not noise)
        if zone.gap_pct >= S6_CONF_GAP_PCT_BONUS_1_THRESHOLD:
            conf += S6_CONF_GAP_PCT_BONUS_1
        elif zone.gap_pct >= S6_CONF_GAP_PCT_BONUS_2_THRESHOLD:
            conf += S6_CONF_GAP_PCT_BONUS_2

        # Bonus: price bounced cleanly off FVG edge (not deep into midzone)
        if kind == "bullish":
            proximity = (close - zone.bottom) / max(zone.gap_size, S6_EPSILON)
            if proximity <= S6_CONF_PROXIMITY_BONUS_THRESHOLD:
                conf += S6_CONF_PROXIMITY_BONUS
        else:
            proximity = (zone.top - close) / max(zone.gap_size, S6_EPSILON)
            if proximity <= S6_CONF_PROXIMITY_BONUS_THRESHOLD:
                conf += S6_CONF_PROXIMITY_BONUS

        # Bonus: strong volume confirmation
        if vol_ratio >= S6_CONF_VOL_BONUS_2_THRESHOLD:
            conf += S6_CONF_VOL_BONUS_2
        elif vol_ratio >= S6_CONF_VOL_BONUS_1_THRESHOLD:
            conf += S6_CONF_VOL_BONUS_1

        return round(min(S6_CONF_MAX, conf), 4)
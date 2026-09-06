"""
S11: Liquidity Sweep  Stop Hunt & Reversal Strategy
======================================================
A Liquidity Sweep (also called a Stop Hunt, Liquidity Grab, or
Judas Swing in ICT/SMC methodology) occurs when price spikes beyond
a key level  specifically to hit retail stop losses clustered
just beyond swing highs/lows  then IMMEDIATELY reverses.

Why this works on NIFTY:
    Large institutions (FIIs, DIIs, prop desks) know exactly where
    retail stop losses are clustered  just below swing lows and
    just above swing highs. They briefly push price beyond these
    levels to fill their own large orders against those stops,
    then reverse hard. NIFTY F&O is especially prone to this due
    to heavy option writing activity that pins certain price levels.

What a Liquidity Sweep looks like:
    BULLISH SWEEP (setup for BUY CALL):
        
          Previous swing low: 22400             
          Stop losses cluster: just below 22400 
                                                
          Sweep candle: wicks DOWN to 22380,    
          hits all the stops, then CLOSES above 
          22400 with a long lower wick           
                                                
          Signal: price is now going UP         
        
         Retail shorts just got stopped out
         Institutions absorbed their sell stops as buy orders
         BUY CALL

    BEARISH SWEEP (setup for BUY PUT):
        Price wicks ABOVE a swing high (hits buy stops),
        then closes BELOW the swing high.
         Retail longs stopped out
         BUY PUT

Detection algorithm:
    1. Find swing lows/highs in last SWEEP_LOOKBACK candles
       (a swing low = low lower than N candles each side)
    2. Check if current candle's wick spiked BEYOND the swing level
    3. Check if current candle CLOSED back on the other side
    4. Measure sweep size (wick beyond level vs body)
    5. Volume must spike on sweep candle (institutional activity)
    6. Follow-through: next candle (current) should confirm direction

Quality filters:
    - Sweep wick must be at least 1.5 the candle body
      (wick-dominant candle = classic liquidity grab signature)
    - Volume on sweep candle  1.8 average
      (institutions create volume when they sweep)
    - Close must be inside the range, not just at the level
    - The swept level must be a CLEAR swing high/low
      (must have at least 3 candles on each side confirming it)

Confidence scoring:
    Base: 0.65
    +0.06 if sweep volume > 2.5 average (major institutional activity)
    +0.04 if wick-to-body ratio > 3.0 (very clean sweep)
    +0.03 if multiple swing levels swept simultaneously (stop cluster)
    +0.02 if follow-through candle confirms (close direction)
    Cap: 0.84

Parameters:
    SWEEP_LOOKBACK      = 30   candles to scan for swing levels
    SWEEP_SWING_BARS    = 3    bars each side for swing confirmation
    SWEEP_MIN_WICK_BODY = 1.5  min ratio of wick beyond level to body
    SWEEP_VOL_MULT      = 1.8  volume spike needed on sweep candle
    SWEEP_CLOSE_INSIDE  = True sweep candle must close back inside level
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Union
from core.models import Direction
from config.settings.strategy import (
    S11_SWEEP_LOOKBACK,
    S11_SWEEP_SWING_BARS,
    S11_SWEEP_MIN_WICK_BODY,
    S11_SWEEP_VOL_MULT,
    S11_SWEEP_CLOSE_INSIDE,
    S11_VOL_MA_PERIOD,
    S11_CONF_BASE,
    S11_CONF_VOL_STRONG_THRESHOLD,
    S11_CONF_VOL_STRONG_BONUS,
    S11_CONF_VOL_MULT_BONUS,
    S11_CONF_WICK_STRONG_THRESHOLD,
    S11_CONF_WICK_STRONG_BONUS,
    S11_CONF_WICK_MIN_BONUS,
    S11_CONF_SWEEP_PCT_STRONG_THRESHOLD,
    S11_CONF_SWEEP_PCT_STRONG_BONUS,
    S11_CONF_SWEEP_PCT_MIN_THRESHOLD,
    S11_CONF_SWEEP_PCT_MIN_BONUS,
    S11_CONF_MAX,
)


@dataclass
class SwingLevel:
    """A confirmed swing high or swing low."""
    kind:       str    # "high" or "low"
    price:      float  # level price
    bar_index:  int    # position in df
    strength:   int    # bars on each side confirming it


@dataclass
class SweepSignal:
    """A detected liquidity sweep event."""
    kind:          str    # "bullish" or "bearish"
    swept_level:   float  # the swing level that was swept
    sweep_low:     float  # lowest point of sweep wick
    sweep_high:    float  # highest point of sweep wick
    wick_size:     float  # how far beyond the level the wick went
    body_size:     float  # body size of sweep candle
    wick_body_ratio: float
    vol_ratio:     float  # volume vs average on sweep candle


class LiquiditySweepStrategy:
    name = "LiqSweep"

    #  PUBLIC ENTRY POINT 

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        # Need lookback + swing detection bars
        if len(df) < S11_SWEEP_LOOKBACK + S11_SWEEP_SWING_BARS * 2 + 5:
            return none

        # Volume average
        vol_avg = float(df["volume"].rolling(S11_VOL_MA_PERIOD).mean().iloc[-1])
        if vol_avg <= 0:
            return none

        # Find swing levels in the lookback window
        # Exclude last 2 candles  we check THOSE for the sweep
        scan_window = df.iloc[-(S11_SWEEP_LOOKBACK + S11_SWEEP_SWING_BARS + 2):
                               -(S11_SWEEP_SWING_BARS)]
        swings = self._find_swings(scan_window)

        if not swings:
            return none

        # The "sweep candle" = second-to-last candle (already closed)
        # The "current candle" = last candle (confirms follow-through)
        sweep_candle = df.iloc[-2]
        curr_candle  = df.iloc[-1]

        sweep_vol_ratio = float(sweep_candle["volume"]) / vol_avg

        # Volume gate on sweep candle
        if sweep_vol_ratio < S11_SWEEP_VOL_MULT:
            return none

        # Check each swing level for a sweep
        for swing in swings:
            result = self._check_sweep(
                sweep_candle, curr_candle,
                swing, sweep_vol_ratio, df
            )
            if result:
                return result

        return none

    #  SWING DETECTION 

    def _find_swings(self, df: pd.DataFrame) -> list[SwingLevel]:
        """
        Find confirmed swing highs and lows.
        A swing low = low[i] < low[i-n..i+n] for all n in range.
        """
        swings = []
        n      = S11_SWEEP_SWING_BARS
        highs  = df["high"].values
        lows   = df["low"].values

        for i in range(n, len(df) - n):
            # Swing low
            if all(lows[i] <= lows[i-j] for j in range(1, n+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, n+1)):
                swings.append(SwingLevel(
                    kind="low",
                    price=float(lows[i]),
                    bar_index=i,
                    strength=n,
                ))

            # Swing high
            if all(highs[i] >= highs[i-j] for j in range(1, n+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, n+1)):
                swings.append(SwingLevel(
                    kind="high",
                    price=float(highs[i]),
                    bar_index=i,
                    strength=n,
                ))

        # Return most recent and most significant swings only
        # Sort by recency (most recent first)
        return sorted(swings, key=lambda s: s.bar_index, reverse=True)[:8]

    #  SWEEP DETECTION 

    def _check_sweep(
        self,
        sweep_c:       pd.Series,
        curr_c:        pd.Series,
        swing:         SwingLevel,
        vol_ratio:     float,
        df:            pd.DataFrame,
    ) -> Union[dict, None]:
        """
        Check if sweep_candle swept the swing level and then reversed.
        """
        s_open  = float(sweep_c["open"])
        s_close = float(sweep_c["close"])
        s_high  = float(sweep_c["high"])
        s_low   = float(sweep_c["low"])
        c_close = float(curr_c["close"])

        body_size = abs(s_close - s_open)
        if body_size < 0.01:
            body_size = 0.01  # prevent division by zero

        #  BULLISH SWEEP (swept a swing LOW, reversed up) 
        if swing.kind == "low":
            # Wick went below the swing low
            wick_below = swing.price - s_low
            if wick_below <= 0:
                return None

            # Candle closed ABOVE the swing low (reversal complete)
            if S11_SWEEP_CLOSE_INSIDE and s_close <= swing.price:
                return None

            # Wick dominates the candle (not just noise)
            wick_body_ratio = wick_below / body_size
            if wick_body_ratio < S11_SWEEP_MIN_WICK_BODY:
                return None

            # Follow-through: current candle also closed above swing low
            if c_close <= swing.price:
                return None

            sweep = SweepSignal(
                kind           = "bullish",
                swept_level    = swing.price,
                sweep_low      = s_low,
                sweep_high     = s_high,
                wick_size      = wick_below,
                body_size      = body_size,
                wick_body_ratio= round(wick_body_ratio, 2),
                vol_ratio      = round(vol_ratio, 2),
            )
            conf = self._score(sweep)
            return {
                "direction":  Direction.BUY_CALL,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "signal":          "bullish_liq_sweep",
                    "swept_level":     round(swing.price, 2),
                    "wick_low":        round(s_low, 2),
                    "wick_size":       round(wick_below, 2),
                    "wick_body_ratio": round(wick_body_ratio, 2),
                    "vol_ratio":       round(vol_ratio, 2),
                },
            }

        #  BEARISH SWEEP (swept a swing HIGH, reversed down) 
        if swing.kind == "high":
            wick_above = s_high - swing.price
            if wick_above <= 0:
                return None

            if S11_SWEEP_CLOSE_INSIDE and s_close >= swing.price:
                return None

            wick_body_ratio = wick_above / body_size
            if wick_body_ratio < S11_SWEEP_MIN_WICK_BODY:
                return None

            if c_close >= swing.price:
                return None

            sweep = SweepSignal(
                kind           = "bearish",
                swept_level    = swing.price,
                sweep_low      = s_low,
                sweep_high     = s_high,
                wick_size      = wick_above,
                body_size      = body_size,
                wick_body_ratio= round(wick_body_ratio, 2),
                vol_ratio      = round(vol_ratio, 2),
            )
            conf = self._score(sweep)
            return {
                "direction":  Direction.BUY_PUT,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "signal":          "bearish_liq_sweep",
                    "swept_level":     round(swing.price, 2),
                    "wick_high":       round(s_high, 2),
                    "wick_size":       round(wick_above, 2),
                    "wick_body_ratio": round(wick_body_ratio, 2),
                    "vol_ratio":       round(vol_ratio, 2),
                },
            }

        return None

    #  CONFIDENCE SCORING 

    @staticmethod
    def _score(sweep: SweepSignal) -> float:
        conf = S11_CONF_BASE

        # Major institutional activity
        if sweep.vol_ratio >= S11_CONF_VOL_STRONG_THRESHOLD:
            conf += S11_CONF_VOL_STRONG_BONUS
        elif sweep.vol_ratio >= S11_SWEEP_VOL_MULT:
            conf += S11_CONF_VOL_MULT_BONUS

        # Very clean sweep (wick >> body)
        if sweep.wick_body_ratio >= S11_CONF_WICK_STRONG_THRESHOLD:
            conf += S11_CONF_WICK_STRONG_BONUS
        elif sweep.wick_body_ratio >= S11_SWEEP_MIN_WICK_BODY:
            conf += S11_CONF_WICK_MIN_BONUS

        # Size of sweep wick as % of level
        sweep_pct = sweep.wick_size / max(sweep.swept_level, 1) * 100
        if sweep_pct >= S11_CONF_SWEEP_PCT_STRONG_THRESHOLD:
            conf += S11_CONF_SWEEP_PCT_STRONG_BONUS
        elif sweep_pct >= S11_CONF_SWEEP_PCT_MIN_THRESHOLD:
            conf += S11_CONF_SWEEP_PCT_MIN_BONUS

        return round(min(S11_CONF_MAX, conf), 4)
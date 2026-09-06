"""
S10: Volume Profile  VPOC & Value Area Strategy
==================================================
Volume Profile distributes trading volume across price levels,
revealing WHERE the most business was conducted  not just WHEN.

Key concepts:
    VPOC  (Volume Point of Control):
        The single price level with the HIGHEST volume traded.
        This is the strongest magnet and support/resistance level.
        Institutional algorithms defend and attack VPOC aggressively.

    VAH   (Value Area High):
        Upper boundary of the Value Area (top 70% of volume).
        Price above VAH = premium zone  likely to return to value.

    VAL   (Value Area Low):
        Lower boundary of the Value Area (bottom 70% of volume).
        Price below VAL = discount zone  likely to return to value.

    Value Area:
        Price range containing 70% of the day's total volume.
        When price is inside Value Area = balanced/accepted.
        When price breaks outside = directional move or trap.

Why this is DIFFERENT from all S1-S9:
    Every other strategy uses TIME-based indicators (RSI, EMA, ADX).
    Volume Profile is PRICE-based  it shows where money actually
    traded, not just price movement. Institutions leave footprints
    in volume that time-based indicators completely miss.

Signal Logic for NIFTY 5-min:

    VPOC BREAKOUT BUY CALL:
        1. Price was below VPOC for  3 candles (base building)
        2. Current candle CLOSES above VPOC
        3. VPOC is below current session's midpoint (trend is up)
        4. Volume on breakout candle > 1.5 session average
        5. Close > VAH (breaking into premium zone)
         BUY CALL  breakout above highest volume node

    VPOC BREAKOUT BUY PUT:
        1. Price was above VPOC for  3 candles
        2. Current candle CLOSES below VPOC
        3. Volume confirmation
        4. Close < VAL (breaking into discount zone)
         BUY PUT

    VALUE AREA REJECTION BUY CALL:
        1. Price tested VAL from above (touched discount zone)
        2. Current candle closes BACK inside Value Area
        3. Volume shows buying at VAL (volume spike)
         BUY CALL  institutions defending value area low

    VALUE AREA REJECTION BUY PUT:
        1. Price tested VAH from below (touched premium zone)
        2. Current candle closes BACK inside Value Area
         BUY PUT  distribution at value area high

Confidence scoring:
    Base: 0.64
    +0.05 if VPOC is a high-volume node (volume > 1.8 avg level vol)
    +0.04 if price broke cleanly (no wick back into previous zone)
    +0.03 if VAH/VAL rejection is clean (closed well inside)
    +0.02 if volume on signal candle > 2 session average
    Cap: 0.83

Parameters:
    VP_LOOKBACK      = 75   candles to build profile (one session)
    VP_NUM_LEVELS    = 50   price buckets to distribute volume into
    VP_VALUE_AREA    = 0.70 fraction of volume defining value area
    VP_VOL_MULT      = 1.5  volume confirmation threshold
    VP_MIN_BASE      = 3    candles price must hold near VPOC before breakout
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from core.models import Direction
from config.settings.strategy import (
    S10_CONF_BASE,
    S10_CONF_CLEAN_BREAKOUT_BONUS,
    S10_CONF_DIST_BONUS,
    S10_CONF_DIST_BONUS_THRESHOLD,
    S10_CONF_MAX,
    S10_CONF_VOL_BONUS_1,
    S10_CONF_VOL_BONUS_1_THRESHOLD,
    S10_CONF_VOL_BONUS_2,
    S10_CONF_VPOC_VOL_BONUS,
    S10_CONF_VPOC_VOL_FACTOR,
    S10_REJECTION_PENALTY,
    S10_REJECTION_TOUCH_FACTOR_BEAR,
    S10_REJECTION_TOUCH_FACTOR_BULL,
    S10_VP_LOOKBACK,
    S10_VP_MIN_BASE,
    S10_VP_NUM_LEVELS,
    S10_VP_VALUE_AREA,
    S10_VP_VOL_MULT,
    S10_MIN_DF_OFFSET,
)


@dataclass
class VolumeProfile:
    vpoc:     float   # Point of Control price
    vah:      float   # Value Area High
    val:      float   # Value Area Low
    va_range: float   # VAH - VAL
    vpoc_vol: float   # Volume at VPOC level
    total_vol: float  # Total volume in profile


class VolumeProfileStrategy:
    name = "VolumeProfile"

    #  PUBLIC ENTRY POINT 

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        if len(df) < S10_VP_LOOKBACK + S10_MIN_DF_OFFSET:
            return none

        # Build volume profile from recent session
        window = df.tail(S10_VP_LOOKBACK)
        vp = self._build_profile(window)
        if vp is None:
            return none

        # Volume stats for signal candle
        close    = float(df["close"].iloc[-1])
        vol      = float(df["volume"].iloc[-1])
        vol_avg  = float(window["volume"].mean())
        if vol_avg <= 0:
            return none
        vol_ratio = vol / vol_avg

        # Volume gate
        if vol_ratio < S10_VP_VOL_MULT:
            return none

        # Check all signal types
        result = (
            self._vpoc_breakout_call(df, close, vp, vol_ratio)
            or self._vpoc_breakout_put(df, close, vp, vol_ratio)
            or self._va_rejection_call(df, close, vp, vol_ratio)
            or self._va_rejection_put(df, close, vp, vol_ratio)
        )

        return result if result else none

    #  VOLUME PROFILE BUILDER 

    def _build_profile(self, df: pd.DataFrame) -> Optional[VolumeProfile]:
        """
        Distribute volume across price levels using TPO-style bucketing.
        Each candle's volume is distributed across its high-low range.
        """
        try:
            price_min = float(df["low"].min())
            price_max = float(df["high"].max())
            if price_max <= price_min:
                return None

            levels = max(10, min(int(S10_VP_NUM_LEVELS or 50), 200))

            # Create price buckets
            bucket_size = (price_max - price_min) / levels
            if bucket_size <= 0:
                return None

            vol_by_level = np.zeros(levels, dtype=float)

            for _, row in df.iterrows():
                h    = float(row["high"])
                l    = float(row["low"])
                v    = float(row["volume"])
                if not np.isfinite(h) or not np.isfinite(l) or not np.isfinite(v) or v <= 0:
                    continue
                rng  = h - l
                if rng <= 0:
                    # Single-price candle  put all volume at close
                    idx = min(
                        int((float(row["close"]) - price_min) / bucket_size),
                        levels - 1
                    )
                    vol_by_level[max(0, idx)] += v
                else:
                    # Distribute volume proportionally across price range
                    lo_idx = int((l - price_min) / bucket_size)
                    hi_idx = int((h - price_min) / bucket_size)
                    lo_idx = max(0, min(lo_idx, levels - 1))
                    hi_idx = max(0, min(hi_idx, levels - 1))
                    if hi_idx < lo_idx:
                        lo_idx, hi_idx = hi_idx, lo_idx
                    span   = hi_idx - lo_idx + 1
                    vol_by_level[lo_idx:hi_idx + 1] += v / span

            if not np.isfinite(vol_by_level).all() or float(vol_by_level.sum()) <= 0:
                return None

            # VPOC = bucket with highest volume
            vpoc_idx  = int(np.argmax(vol_by_level))
            vpoc_price = price_min + (vpoc_idx + 0.5) * bucket_size
            vpoc_vol  = float(vol_by_level[vpoc_idx])
            total_vol = float(vol_by_level.sum())

            # Value Area: 70% of total volume, expanding from VPOC
            target_vol = total_vol * S10_VP_VALUE_AREA
            accumulated = vpoc_vol
            va_lo_idx   = vpoc_idx
            va_hi_idx   = vpoc_idx

            for _ in range(levels):
                if accumulated >= target_vol:
                    break
                expand_hi = va_hi_idx < levels - 1
                expand_lo = va_lo_idx > 0
                if not expand_hi and not expand_lo:
                    break
                vol_above = vol_by_level[va_hi_idx + 1] if expand_hi else 0
                vol_below = vol_by_level[va_lo_idx - 1] if expand_lo else 0
                if vol_above >= vol_below:
                    va_hi_idx += 1
                    accumulated += vol_above
                else:
                    va_lo_idx -= 1
                    accumulated += vol_below

            vah = price_min + (va_hi_idx + 1) * bucket_size
            val = price_min + va_lo_idx * bucket_size

            return VolumeProfile(
                vpoc      = round(vpoc_price, 2),
                vah       = round(vah, 2),
                val       = round(val, 2),
                va_range  = round(vah - val, 2),
                vpoc_vol  = vpoc_vol,
                total_vol = total_vol,
            )
        except Exception:
            return None

    #  VPOC BREAKOUT CALLS 

    def _vpoc_breakout_call(
        self,
        df:        pd.DataFrame,
        close:     float,
        vp:        VolumeProfile,
        vol_ratio: float,
    ) -> Optional[dict]:
        """Price closes above VPOC after holding below for min bars."""
        prev_close = float(df["close"].iloc[-2])

        # Previous close was below VPOC (was under)
        if prev_close >= vp.vpoc:
            return None

        # Current close broke above VPOC
        if close <= vp.vpoc:
            return None

        # Check that price was holding below VPOC for min_base candles
        recent = df["close"].iloc[-(S10_VP_MIN_BASE + 1):-1]
        if not all(float(c) < vp.vpoc for c in recent):
            return None

        # Ideally also broke into or above VAH
        in_premium = close >= vp.vah

        conf = self._score(vp, vol_ratio, close, vp.vpoc, "vpoc_call", in_premium)
        return {
            "direction":  Direction.BUY_CALL,
            "confidence": conf,
            "name":       self.name,
            "meta": {
                "signal":     "vpoc_breakout_call",
                "vpoc":       vp.vpoc,
                "vah":        vp.vah,
                "val":        vp.val,
                "vol_ratio":  round(vol_ratio, 2),
                "in_premium": in_premium,
            },
        }

    def _vpoc_breakout_put(
        self,
        df:        pd.DataFrame,
        close:     float,
        vp:        VolumeProfile,
        vol_ratio: float,
    ) -> Optional[dict]:
        """Price closes below VPOC after holding above for min bars."""
        prev_close = float(df["close"].iloc[-2])

        if prev_close <= vp.vpoc:
            return None
        if close >= vp.vpoc:
            return None

        recent = df["close"].iloc[-(S10_VP_MIN_BASE + 1):-1]
        if not all(float(c) > vp.vpoc for c in recent):
            return None

        in_discount = close <= vp.val

        conf = self._score(vp, vol_ratio, close, vp.vpoc, "vpoc_put", in_discount)
        return {
            "direction":  Direction.BUY_PUT,
            "confidence": conf,
            "name":       self.name,
            "meta": {
                "signal":      "vpoc_breakout_put",
                "vpoc":        vp.vpoc,
                "vah":         vp.vah,
                "val":         vp.val,
                "vol_ratio":   round(vol_ratio, 2),
                "in_discount": in_discount,
            },
        }

    #  VALUE AREA REJECTION 

    def _va_rejection_call(
        self,
        df:        pd.DataFrame,
        close:     float,
        vp:        VolumeProfile,
        vol_ratio: float,
    ) -> Optional[dict]:
        """Price tested VAL and closed back inside Value Area (bullish rejection)."""
        low_prev  = float(df["low"].iloc[-2])
        prev_close = float(df["close"].iloc[-2])

        # Previous candle touched or pierced VAL
        touched_val = low_prev <= vp.val * S10_REJECTION_TOUCH_FACTOR_BULL
        # Current close is back inside value area
        back_inside = vp.val < close < vp.vah

        if not touched_val or not back_inside:
            return None

        conf = self._score(vp, vol_ratio, close, vp.val, "val_rejection", False)
        conf -= S10_REJECTION_PENALTY
        return {
            "direction":  Direction.BUY_CALL,
            "confidence": conf,
            "name":       self.name,
            "meta": {
                "signal":    "va_rejection_call",
                "vpoc":      vp.vpoc,
                "val":       vp.val,
                "vol_ratio": round(vol_ratio, 2),
            },
        }

    def _va_rejection_put(
        self,
        df:        pd.DataFrame,
        close:     float,
        vp:        VolumeProfile,
        vol_ratio: float,
    ) -> Optional[dict]:
        """Price tested VAH and closed back inside Value Area (bearish rejection)."""
        high_prev  = float(df["high"].iloc[-2])
        prev_close = float(df["close"].iloc[-2])

        touched_vah = high_prev >= vp.vah * S10_REJECTION_TOUCH_FACTOR_BEAR
        back_inside = vp.val < close < vp.vah

        if not touched_vah or not back_inside:
            return None

        conf = self._score(vp, vol_ratio, close, vp.vah, "vah_rejection", False)
        conf -= S10_REJECTION_PENALTY
        return {
            "direction":  Direction.BUY_PUT,
            "confidence": conf,
            "name":       self.name,
            "meta": {
                "signal":    "va_rejection_put",
                "vpoc":      vp.vpoc,
                "vah":       vp.vah,
                "vol_ratio": round(vol_ratio, 2),
            },
        }

    #  CONFIDENCE SCORING 

    @staticmethod
    def _score(
        vp:        VolumeProfile,
        vol_ratio: float,
        close:     float,
        ref:       float,
        signal:    str,
        clean:     bool,
    ) -> float:
        conf = S10_CONF_BASE

        # High-volume node (VPOC concentration)
        avg_level_vol = vp.total_vol / S10_VP_NUM_LEVELS
        if vp.vpoc_vol > avg_level_vol * S10_CONF_VPOC_VOL_FACTOR:
            conf += S10_CONF_VPOC_VOL_BONUS

        # Clean breakout (no wick back)
        if clean:
            conf += S10_CONF_CLEAN_BREAKOUT_BONUS

        # Strong volume
        if vol_ratio >= S10_CONF_VOL_BONUS_1_THRESHOLD:
            conf += S10_CONF_VOL_BONUS_1
        elif vol_ratio >= S10_VP_VOL_MULT:
            conf += S10_CONF_VOL_BONUS_2

        # Distance from level
        dist_pct = abs(close - ref) / max(ref, 1) * 100
        if dist_pct > S10_CONF_DIST_BONUS_THRESHOLD:
            conf += S10_CONF_DIST_BONUS

        return round(min(S10_CONF_MAX, conf), 4)

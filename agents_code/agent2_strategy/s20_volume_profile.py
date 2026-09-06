"""
agents_code/agent2_strategy/s20_volume_profile.py — Volume Profile / Value Area
================================================================================
WHAT IS VOLUME PROFILE?
  Instead of plotting volume over TIME (traditional), plot volume over PRICE.
  Shows exactly which price levels had the most trading activity.
  
  Key levels:
  - POC  (Point of Control): price with MOST volume = strongest magnet
  - VAH  (Value Area High):  top of the 70% volume zone
  - VAL  (Value Area Low):   bottom of the 70% volume zone
  - HVN  (High Volume Node): price cluster = support/resistance
  - LVN  (Low Volume Node):  price gap = price moves fast through here

WHY THIS GENERATES MORE SIGNALS:
  Every professional institutional desk uses Volume Profile.
  Breakout of VAH = strong bullish signal (price leaving value = momentum)
  Rejection at VAH = strong bearish signal (sellers defending value)
  Return to POC after breakout = common pattern = tradeable

SIGNAL LOGIC:
  BUY_CALL:
    1. Price breaks above VAH with volume confirmation → breakout play
    2. Price bounces off VAL (support) → mean reversion to POC
    3. Price at LVN above current level → fast move expected upward
  
  BUY_PUT:
    1. Price breaks below VAL with volume → breakdown play
    2. Price rejected at VAH (resistance) → mean reversion to POC
    3. Price at LVN below current level → fast move expected downward

CONFIDENCE SCORING:
  Base: 0.66
  +0.06 if volume on breakout candle > 1.5x avg (volume confirms)
  +0.04 if POC is on the right side (POC below for calls, above for puts)
  +0.04 if ADX > 20 (trend present to sustain breakout)
  +0.03 if breakout gap > 0.2% (not a marginal break)
  -0.05 if in RANGING regime (breakouts fail in ranges)
"""

from __future__ import annotations

from config.settings.strategy import (
    S20_VP_BINS,
    S20_VALUE_AREA_PCT,
    S20_BREAKOUT_MIN_PCT,
    S20_VOLUME_CONFIRM_RATIO,
    S20_MIN_CANDLES,
)


import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from config.settings import ADX_TREND_THRESHOLD
except ImportError:
    ADX_TREND_THRESHOLD = 18.0
try:
    from core.models import Direction
except ImportError:
    class Direction:
        NONE     = "NONE"
        BUY_CALL = "BUY_CALL"
        BUY_PUT  = "BUY_PUT"

# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class VolumeProfileLevels:
    poc:   float   # Point of Control
    vah:   float   # Value Area High
    val:   float   # Value Area Low
    total_volume: float
    above_value:  bool   # current price above value area
    below_value:  bool   # current price below value area
    in_value:     bool   # current price inside value area
    near_poc:     bool   # within 0.1% of POC

class VolumeProfileStrategy:
    """S20: Volume Profile / Value Area breakout and rejection strategy."""
    name = "ValueArea"

    def evaluate(
        self,
        df:       pd.DataFrame,
        orb_high: Optional[float] = None,
        orb_low:  Optional[float] = None,
        **kwargs,
    ) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if df is None or len(df) < S20_MIN_CANDLES:
            return none
        try:
            return self._evaluate(df, kwargs)
        except Exception:
            return none

    def _evaluate(self, df: pd.DataFrame, kwargs: dict) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        close   = float(df["close"].iloc[-1])
        volume  = df["volume"].values
        vol_avg = float(np.mean(volume[-20:]))
        vol_now = float(volume[-1])

        # Compute volume profile
        vp = self._compute_vp(df)
        if vp is None:
            return none

        # ADX for trend filter
        adx = self._adx(df)
        regime = kwargs.get("regime_details", {})
        regime_label = regime.get("regime", regime.get("label", "NEUTRAL"))

        # ── Signal 1: VAH Breakout (BUY_CALL) ───────────────────────────────
        vah_breakout = (close > vp.vah and
                        (close - vp.vah) / vp.vah * 100 > S20_BREAKOUT_MIN_PCT)

        # ── Signal 2: VAL Breakdown (BUY_PUT) ───────────────────────────────
        val_breakdown = (close < vp.val and
                         (vp.val - close) / vp.val * 100 > S20_BREAKOUT_MIN_PCT)

        # ── Signal 3: VAH Rejection (BUY_PUT) ───────────────────────────────
        # Price tested VAH but closed back inside value area
        prev_close = float(df["close"].iloc[-2])
        vah_rejection = (prev_close > vp.vah and close < vp.vah and
                         close > vp.val)

        # ── Signal 4: VAL Bounce (BUY_CALL) ─────────────────────────────────
        val_bounce = (prev_close < vp.val and close > vp.val and
                      close < vp.vah)

        # Determine direction
        if vah_breakout or val_bounce:
            direction   = Direction.BUY_CALL
            signal_type = "VAH_BREAKOUT" if vah_breakout else "VAL_BOUNCE"
            key_level   = vp.vah if vah_breakout else vp.val
        elif val_breakdown or vah_rejection:
            direction   = Direction.BUY_PUT
            signal_type = "VAL_BREAKDOWN" if val_breakdown else "VAH_REJECTION"
            key_level   = vp.val if val_breakdown else vp.vah
        else:
            return none

        # ── Confidence scoring ────────────────────────────────────────────────
        conf = 0.66

        # Volume confirms breakout
        if vol_now > vol_avg * S20_VOLUME_CONFIRM_RATIO:
            conf += 0.06

        # POC position supports direction
        if direction == Direction.BUY_CALL and vp.poc < close:
            conf += 0.04   # POC below = upward support
        elif direction == Direction.BUY_PUT and vp.poc > close:
            conf += 0.04   # POC above = downward resistance

        # ADX confirms
        if adx > 20:
            conf += 0.04
        if adx > 30:
            conf += 0.02

        # Margin of breakout
        breakout_gap = abs(close - key_level) / key_level * 100
        if breakout_gap > 0.2:
            conf += 0.03

        # Regime penalty
        if regime_label in ("RANGING", "CHOPPY"):
            conf -= 0.05

        conf = round(min(0.86, max(0.55, conf)), 4)

        return {
            "direction":  direction,
            "confidence": conf,
            "name":       self.name,
            "meta": {
                "signal_type":  signal_type,
                "poc":          round(vp.poc, 1),
                "vah":          round(vp.vah, 1),
                "val":          round(vp.val, 1),
                "breakout_gap": round(breakout_gap, 3),
                "vol_ratio":    round(vol_now / max(vol_avg, 1), 2),
                "adx":          round(adx, 1),
                "regime":       regime_label,
            },
        }

    # ── Volume Profile Computation ────────────────────────────────────────────

    def _compute_vp(self, df: pd.DataFrame) -> Optional[VolumeProfileLevels]:
        """Compute POC, VAH, VAL from OHLCV data."""
        try:
            highs  = df["high"].values
            lows   = df["low"].values
            closes = df["close"].values
            vols   = df["volume"].values

            price_min = float(np.min(lows))
            price_max = float(np.max(highs))
            if price_max <= price_min:
                return None

            # Create price bins
            bins      = np.linspace(price_min, price_max, S20_VP_BINS + 1)
            bin_vol   = np.zeros(S20_VP_BINS)
            bin_mid   = (bins[:-1] + bins[1:]) / 2

            # Distribute volume across price range for each candle
            for i in range(len(df)):
                lo, hi, vol = lows[i], highs[i], vols[i]
                # Find which bins this candle spans
                for b in range(S20_VP_BINS):
                    # Overlap between candle range and bin range
                    overlap_lo = max(lo, bins[b])
                    overlap_hi = min(hi, bins[b + 1])
                    if overlap_hi > overlap_lo:
                        # Fraction of candle range that falls in this bin
                        candle_range = max(hi - lo, 0.01)
                        frac = (overlap_hi - overlap_lo) / candle_range
                        bin_vol[b] += vol * frac

            # POC = bin with most volume
            poc_idx = int(np.argmax(bin_vol))
            poc     = float(bin_mid[poc_idx])

            # Value Area: 70% of total volume centred around POC
            total_vol    = float(np.sum(bin_vol))
            target_vol   = total_vol * S20_VALUE_AREA_PCT
            va_vol        = bin_vol[poc_idx]
            va_low_idx    = poc_idx
            va_high_idx   = poc_idx

            while va_vol < target_vol:
                # Expand to whichever adjacent bin has more volume
                can_expand_up  = va_high_idx < S20_VP_BINS - 1
                can_expand_dn  = va_low_idx  > 0
                if not can_expand_up and not can_expand_dn:
                    break
                vol_up = bin_vol[va_high_idx + 1] if can_expand_up else -1
                vol_dn = bin_vol[va_low_idx  - 1] if can_expand_dn  else -1
                if vol_up >= vol_dn:
                    va_high_idx += 1
                    va_vol += bin_vol[va_high_idx]
                else:
                    va_low_idx  -= 1
                    va_vol += bin_vol[va_low_idx]

            vah   = float(bins[va_high_idx + 1])
            val   = float(bins[va_low_idx])
            close = float(closes[-1])

            return VolumeProfileLevels(
                poc          = poc,
                vah          = vah,
                val          = val,
                total_volume = total_vol,
                above_value  = close > vah,
                below_value  = close < val,
                in_value     = val <= close <= vah,
                near_poc     = abs(close - poc) / poc < 0.001,
            )
        except Exception:
            return None

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> float:
        try:
            h, l, c = df["high"], df["low"], df["close"]
            pc = c.shift(1)
            tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
            return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
        except Exception:
            return 20.0

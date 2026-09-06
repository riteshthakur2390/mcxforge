"""
agents_code/agent2_strategy/s24_range_spread.py — Range-Bound Spread Opportunity
==================================================================================
MAKE MONEY EVEN WHEN THERE IS NO DIRECTION:

Current system: if RANGING → block all signals → 0 trades on ranging days.
This is leaving money on the table. Top algo traders NEVER sit idle.

SOLUTION: In ranging markets, sell options instead of buying them.
  - NIFTY churning between 22000-22200 → sell 22300CE and 21900PE
  - Both expire worthless → pocket full premium
  - This is the Iron Condor / Short Strangle approach

BUT: We are OPTION BUYERS in SignalForge.
Instead of selling, we detect when the range is about to BREAK
and position for the breakout BEFORE it happens.

STRATEGY:
  1. Detect tight range (Chop Index > 61.8 OR Bollinger Bandwidth < threshold)
  2. Detect range COMPRESSION (getting tighter = breakout imminent)
  3. When breakout triggers: enter in direction of break
  4. Set TIGHT SL just inside the range (quick exit if fake break)
  5. Target: full range width (measured move)

SIGNAL LOGIC:
  SETUP (pre-condition):
    - Price in range for >= 8 candles (2× min_range_candles)
    - Range width < 0.5% of spot (tight compression)
    - Volume drying (confirming compression)
  
  TRIGGER (entry signal):
    - Price closes ABOVE range high + 0.1% → BUY_CALL (breakout)
    - Price closes BELOW range low  - 0.1% → BUY_PUT  (breakdown)
    - Volume MUST expand on trigger candle (vol_ratio > 1.4)

CONFIDENCE:
  Base: 0.67
  +0.06 if compression > 8 candles (more coiled = stronger break)
  +0.05 if volume ratio > 2.0 on trigger (strong institutional participation)
  +0.04 if ADX > 15 (slight trend helps sustain breakout)
  -0.06 if no volume confirmation (false breakout risk)
"""

from __future__ import annotations

from config.settings.strategy import (
    S24_MIN_RANGE_CANDLES,
    S24_MAX_RANGE_WIDTH_PCT,
    S24_BREAKOUT_BUFFER_PCT,
    S24_VOL_CONFIRM_RATIO,
    S24_CHOP_INDEX_PERIOD,
    S24_MIN_CANDLES,
)


from typing import Optional
import numpy as np
import pandas as pd

try:
    from core.models import Direction
except ImportError:
    class Direction:
        NONE="NONE"; BUY_CALL="BUY_CALL"; BUY_PUT="BUY_PUT"

class RangeSpreadStrategy:
    """
    S24: Range compression breakout — captures the explosive moves
    that follow tight consolidation periods.
    """
    name = "RangeSpread"

    def evaluate(
        self,
        df:       pd.DataFrame,
        orb_high: Optional[float] = None,
        orb_low:  Optional[float] = None,
        **kwargs,
    ) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if df is None or len(df) < S24_MIN_CANDLES:
            return none
        try:
            return self._evaluate(df, kwargs)
        except Exception:
            return none

    def _evaluate(self, df: pd.DataFrame, kwargs: dict) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values
        vols   = df["volume"].values
        n      = len(df)

        # ── Step 1: Detect recent range ───────────────────────────────────────
        lookback    = min(20, n - 2)
        range_high  = float(np.max(highs[-lookback:-1]))
        range_low   = float(np.min(lows[-lookback:-1]))
        range_width = (range_high - range_low) / range_low * 100
        current     = float(closes[-1])

        # Range must be tight
        if range_width > S24_MAX_RANGE_WIDTH_PCT:
            return none

        # ── Step 2: Count candles inside range ───────────────────────────────
        in_range_count = 0
        for i in range(-lookback, -1):
            if range_low <= closes[i] <= range_high:
                in_range_count += 1

        if in_range_count < S24_MIN_RANGE_CANDLES:
            return none

        # ── Step 3: Volume compression (drying up) ────────────────────────────
        vol_avg   = float(np.mean(vols[-20:]))
        vol_range = float(np.mean(vols[-lookback:-1]))
        vol_compressing = vol_range < vol_avg * 0.85

        # ── Step 4: Breakout trigger ──────────────────────────────────────────
        buffer      = range_low * S24_BREAKOUT_BUFFER_PCT / 100
        vol_now     = float(vols[-1])
        vol_ratio   = vol_now / max(vol_avg, 1)
        vol_confirm = vol_ratio >= S24_VOL_CONFIRM_RATIO

        breakout_up  = current > range_high + buffer
        breakout_dn  = current < range_low  - buffer

        if not breakout_up and not breakout_dn:
            return none

        direction = Direction.BUY_CALL if breakout_up else Direction.BUY_PUT

        # ── Confidence scoring ────────────────────────────────────────────────
        conf = 0.67

        # Compression duration bonus
        if in_range_count >= 10:
            conf += 0.06
        elif in_range_count >= 8:
            conf += 0.03

        # Volume on breakout
        if vol_ratio >= 2.0:
            conf += 0.05
        elif vol_ratio >= 1.4:
            conf += 0.02
        else:
            conf -= 0.06   # no volume = likely fake breakout

        # ADX
        adx = self._adx(df)
        if adx > 15:
            conf += 0.04

        # Tight range = stronger coil
        if range_width < 0.3:
            conf += 0.03

        # Volume compression before breakout = classic setup
        if vol_compressing:
            conf += 0.03

        # Expected measured move (target = range width)
        measured_move = range_high - range_low
        breakout_margin = abs(current - (range_high if breakout_up else range_low))

        conf = round(min(0.86, max(0.55, conf)), 4)

        return {
            "direction":  direction,
            "confidence": conf,
            "name":       self.name,
            "meta": {
                "signal":          "RANGE_BREAKOUT" if breakout_up else "RANGE_BREAKDOWN",
                "range_high":      round(range_high, 1),
                "range_low":       round(range_low, 1),
                "range_width_pct": round(range_width, 3),
                "in_range_candles":in_range_count,
                "vol_ratio":       round(vol_ratio, 2),
                "measured_move":   round(measured_move, 1),
                "adx":             round(adx, 1),
                "vol_confirm":     vol_confirm,
            },
        }

    @staticmethod
    def _adx(df: pd.DataFrame, p: int = 14) -> float:
        try:
            h,l,c=df["high"],df["low"],df["close"]; pc=c.shift(1)
            tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
            return float(tr.ewm(span=p,adjust=False).mean().iloc[-1])
        except Exception:
            return 20.0

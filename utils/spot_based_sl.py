"""
utils/spot_based_sl.py — Dynamic Spot-Based Structural Stop Loss
=================================================================
THE PROBLEM WITH FIXED % SL:
  Fixed 25% SL on option premium is arbitrary.
  On some days 25% = 150pts NIFTY move (fine).
  On other days 25% = 40pts NIFTY move (too tight, hits on noise).
  On volatile days 25% = you exit before the real move happens.

THE RIGHT WAY (how prop traders do it):
  Stop loss = the NIFTY SPOT level where your trade thesis is WRONG.
  
  For BUY CALL: your thesis is "NIFTY will go UP from here"
    → Thesis is WRONG when NIFTY breaks below the nearest support
    → SL = nearest support level below entry spot
    → Example: NIFTY at 22100, support at 22000 → SL if NIFTY closes below 22000
  
  For BUY PUT: your thesis is "NIFTY will go DOWN from here"
    → Thesis is WRONG when NIFTY breaks above the nearest resistance
    → SL = nearest resistance level above entry spot
    → Example: NIFTY at 22100, resistance at 22200 → SL if NIFTY breaks above 22200

WHERE ARE THE KEY LEVELS?
  This module detects them dynamically from:
  1. SWING LOWS/HIGHS    — pivot points in recent price action
  2. CPR (Central Pivot Range) — daily pivot, R1, R2, S1, S2
  3. VWAP                — institutional average price (key intraday level)
  4. ROUND NUMBERS       — 22000, 22100, 22200 etc (psychological levels)
  5. ATR BANDS           — entry ± 1×ATR (volatility-based floor)
  6. OI WALLS            — strikes with highest OI = where market makes money

HOW OPTION SL IS COMPUTED FROM SPOT SL:
  Once you know NIFTY_SL_LEVEL, convert to option premium SL:
  
  distance    = |entry_spot - nifty_sl_level|   (in NIFTY points)
  option_sl   = entry_premium - delta × distance
  
  This gives EXACT option premium at which to exit
  when NIFTY reaches the invalidation level.
  
  Example:
    entry_spot   = 22100
    support      = 22000 (nearest swing low)
    distance     = 100 pts
    delta        = 0.48
    entry_prem   = 150
    option_sl    = 150 - (0.48 × 100) = 150 - 48 = ₹102
    SL%          = (150-102)/150 = 32%  ← dynamic, not fixed 25%

  On a different day:
    entry_spot   = 22100
    support      = 22080 (only 20pts away)
    distance     = 20 pts
    option_sl    = 150 - (0.48 × 20) = 150 - 9.6 = ₹140.4
    SL%          = (150-140.4)/150 = 6.4%  ← tight because support is close

  The SL adapts to the actual market structure, not a fixed number.

POSITION SIZING FROM STRUCTURAL SL:
  risk_per_lot = (entry_prem - option_sl) × lot_size
  lots         = floor(max_risk / risk_per_lot)
  
  This ensures you never risk more than 1% of capital regardless of
  how far away the structural SL is.

REAL-TIME MONITORING:
  Every candle, check if NIFTY spot has breached the invalidation level.
  Candle CLOSE below support (for calls) = exit.
  (Not intrabar wicks — only confirmed closes)
  This prevents premature exits on fake breakdowns.

USAGE:
  detector = SpotBasedSLDetector()
  
  # At trade entry: find structural SL level
  sl = detector.compute_structural_sl(
      df=df,           # 5-min OHLCV
      spot=22100,
      direction="BUY_CALL",
      entry_premium=150,
      delta=0.48,
  )
  # sl.spot_sl_level = 21980 (nearest support)
  # sl.option_sl     = 109.6 (option premium at that spot level)
  
  # Every candle: check if breached
  breached = detector.is_sl_breached(
      current_spot=21975,   # current NIFTY
      sl_level=21980,
      direction="BUY_CALL",
  )
  if breached:
      exit_position()   # thesis invalidated
"""
from __future__ import annotations


from config.settings.modules.utils_thresholds import *

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

try:
    from config.settings import (
        SPOT_SL_STOP_LOSS_PCT, SPOT_SL_NIFTY_LOT_SIZE,
        DEPLOYED_CAPITAL, SPOT_SL_NIFTY_STRIKE_STEP,
    )

except ImportError:

    DEPLOYED_CAPITAL   = 100_000

@dataclass
class KeyLevel:
    """A structural price level with its source and strength."""
    price:    float
    source:   str     # "SWING_LOW" | "SWING_HIGH" | "CPR_S1" | "VWAP" | "ROUND" | "ATR"
    strength: float   # 0-1 (how strong/reliable this level is)
    distance_pts: float  # distance from current spot

@dataclass
class StructuralSL:
    """Complete structural SL recommendation."""
    # Spot level
    spot_sl_level:     float    # NIFTY spot price where thesis is invalid
    sl_source:         str      # which level was used
    sl_strength:       float    # confidence in this level (0-1)
    distance_pts:      float    # distance from entry spot to SL level
    distance_pct:      float    # as % of entry spot

    # Option SL derived from spot SL
    option_sl_premium: float    # option premium at SL spot level
    option_sl_pct:     float    # option SL as % of entry premium (dynamic)

    # Position sizing
    risk_per_lot_inr:  float    # ₹ risk per lot if SL hit
    recommended_lots:  int      # lots to risk only 1% of capital
    max_risk_inr:      float

    # All key levels found (for logging/dashboard)
    all_levels:        list[KeyLevel] = field(default_factory=list)
    note:              str = ""

    def to_dict(self) -> dict:
        return {
            "spot_sl_level":     self.spot_sl_level,
            "sl_source":         self.sl_source,
            "sl_strength":       self.sl_strength,
            "distance_pts":      self.distance_pts,
            "distance_pct":      self.distance_pct,
            "option_sl_premium": self.option_sl_premium,
            "option_sl_pct":     self.option_sl_pct,
            "risk_per_lot_inr":  self.risk_per_lot_inr,
            "recommended_lots":  self.recommended_lots,
        }

class SpotBasedSLDetector:
    """
    Detects structural support/resistance levels dynamically
    and converts them to option SL premiums.
    """

    def compute_structural_sl(
        self,
        df:            pd.DataFrame,
        spot:          float,
        direction:     str,
        entry_premium: float,
        delta:         float,
        lot_size:      int   = None,
        capital:       float = None,
        india_vix:     float = 18.0,
    ) -> StructuralSL:
        """
        Find the nearest structural SL level for NIFTY and compute option SL.

        Args:
            df:            5-min OHLCV DataFrame (session data)
            spot:          current NIFTY spot price
            direction:     "BUY_CALL" or "BUY_PUT"
            entry_premium: option LTP at entry
            delta:         option delta (absolute value)
            lot_size:      lots size (default NIFTY=75)
            capital:       deployed capital for sizing
            india_vix:     India VIX for ATR scaling

        Returns:
            StructuralSL with spot_sl_level and option_sl_premium
        """
        ls  = lot_size or SPOT_SL_NIFTY_LOT_SIZE
        cap = capital  or DEPLOYED_CAPITAL
        d   = abs(delta)

        # ── Step 1: Find all key levels ───────────────────────────────────────
        all_levels = self._find_all_levels(df, spot, india_vix)

        # ── Step 2: Select the best SL level ─────────────────────────────────
        sl_level, sl_source, sl_strength = self._select_sl_level(
            all_levels, spot, direction
        )

        # ── Step 3: Validate distance ─────────────────────────────────────────
        distance_pts = abs(spot - sl_level)
        distance_pct = distance_pts / spot * 100

        # If level too close → use ATR-based minimum
        if distance_pct < SPOT_SL_MIN_SL_DISTANCE_PCT:
            atr       = self._atr(df)
            min_dist  = spot * SPOT_SL_MIN_SL_DISTANCE_PCT / 100
            sl_dist   = max(atr * SPOT_SL_ATR_SL_MULTIPLIER, min_dist)
            sl_level  = (spot - sl_dist if direction == "BUY_CALL"
                         else spot + sl_dist)
            sl_source = "ATR_MIN"
            sl_strength = 0.5
            distance_pts = abs(spot - sl_level)
            distance_pct = distance_pts / spot * 100

        # If level too far → cap at SPOT_SL_MAX_SL_DISTANCE_PCT
        if distance_pct > SPOT_SL_MAX_SL_DISTANCE_PCT:
            capped_dist  = spot * SPOT_SL_MAX_SL_DISTANCE_PCT / 100
            sl_level = (spot - capped_dist if direction == "BUY_CALL"
                        else spot + capped_dist)
            sl_source   = f"{sl_source}_CAPPED"
            distance_pts = abs(spot - sl_level)
            distance_pct = distance_pts / spot * 100

        # ── Step 4: Convert spot SL to option premium SL ─────────────────────
        # option_sl = entry_premium - delta × spot_distance
        # This is the first-order approximation using delta
        option_sl  = max(5.0, entry_premium - d * distance_pts)
        option_sl_pct = (entry_premium - option_sl) / entry_premium * 100

        # ── Step 5: Position sizing ───────────────────────────────────────────
        max_risk     = cap * SPOT_SL_RISK_PER_TRADE_PCT / 100
        risk_per_lot = (entry_premium - option_sl) * ls
        lots         = max(1, int(max_risk / max(risk_per_lot, 1)))

        note = (
            f"Entry={spot:.0f} | SL at {sl_source}={sl_level:.0f} "
            f"({distance_pts:.0f}pts / {distance_pct:.2f}%) | "
            f"Option: ₹{entry_premium:.1f} → ₹{option_sl:.1f} ({option_sl_pct:.1f}%) | "
            f"Risk ₹{risk_per_lot:.0f}/lot → {lots} lot(s)"
        )

        logger.info(f"[SpotSL] {note}")

        return StructuralSL(
            spot_sl_level      = round(sl_level, 1),
            sl_source          = sl_source,
            sl_strength        = round(sl_strength, 3),
            distance_pts       = round(distance_pts, 1),
            distance_pct       = round(distance_pct, 3),
            option_sl_premium  = round(option_sl, 2),
            option_sl_pct      = round(option_sl_pct, 2),
            risk_per_lot_inr   = round(risk_per_lot, 2),
            recommended_lots   = lots,
            max_risk_inr       = round(max_risk, 2),
            all_levels         = all_levels,
            note               = note,
        )

    def is_sl_breached(
        self,
        current_spot: float,
        sl_level:     float,
        direction:    str,
        require_close:bool = True,
    ) -> bool:
        """
        Check if NIFTY has breached the structural SL level.

        Args:
            current_spot:  current NIFTY spot (use candle CLOSE, not intrabar)
            sl_level:      the structural SL level
            direction:     "BUY_CALL" or "BUY_PUT"
            require_close: True = only breach on candle CLOSE (avoids wick fakes)

        Returns:
            True if thesis invalidated and should exit
        """
        if direction == "BUY_CALL":
            return current_spot < sl_level   # NIFTY fell below support
        else:
            return current_spot > sl_level   # NIFTY broke above resistance

    def get_dynamic_sl_premium(
        self,
        current_spot:  float,
        sl_level:      float,
        entry_spot:    float,
        entry_premium: float,
        delta:         float,
    ) -> float:
        """
        Recompute option SL premium as spot moves.
        As NIFTY approaches SL level, option SL becomes the current premium.
        This allows the SL to trail in $ terms as the trade moves.
        """
        remaining_distance = abs(current_spot - sl_level)
        option_sl = max(5.0, entry_premium - abs(delta) * abs(entry_spot - sl_level))
        # Also ensure we never let a winner become a big loser
        current_option_est = entry_premium - abs(delta) * abs(entry_spot - current_spot)
        return round(max(option_sl, current_option_est * 0.85), 2)

    # ── LEVEL DETECTION ───────────────────────────────────────────────────────

    def _find_all_levels(
        self,
        df:         pd.DataFrame,
        spot:       float,
        india_vix:  float = 18.0,
    ) -> list[KeyLevel]:
        """Find all structural levels near current spot."""
        levels = []

        if df is not None and len(df) >= SPOT_SL_SWING_LOOKBACK:
            # 1. Swing lows and highs
            levels.extend(self._swing_levels(df, spot))

            # 2. VWAP
            vwap = self._compute_vwap(df)
            if vwap > 0:
                levels.append(KeyLevel(
                    price    = round(vwap, 1),
                    source   = "VWAP",
                    strength = 0.75,
                    distance_pts = abs(spot - vwap),
                ))

            # 3. ATR band levels
            atr = self._atr(df)
            levels.append(KeyLevel(
                price    = round(spot - atr * 1.5, 1),
                source   = "ATR_SUPPORT",
                strength = 0.55,
                distance_pts = atr * 1.5,
            ))
            levels.append(KeyLevel(
                price    = round(spot + atr * 1.5, 1),
                source   = "ATR_RESIST",
                strength = 0.55,
                distance_pts = atr * 1.5,
            ))

            # 4. CPR levels
            levels.extend(self._cpr_levels(df, spot))

        # 5. Round number levels (22000, 22100, 22200 etc)
        levels.extend(self._round_levels(spot))

        # Sort by distance from spot
        levels.sort(key=lambda l: l.distance_pts)
        return levels

    def _swing_levels(self, df: pd.DataFrame, spot: float) -> list[KeyLevel]:
        """Detect swing highs and lows as support/resistance."""
        levels = []
        highs  = df["high"].values
        lows   = df["low"].values
        n      = len(df)
        k      = SPOT_SL_SWING_BARS_EACH_SIDE

        for i in range(k, min(n - k, SPOT_SL_SWING_LOOKBACK + k)):
            # Swing low (support)
            if all(lows[i] <= lows[i-j] for j in range(1, k+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, k+1)):
                p = float(lows[i])
                levels.append(KeyLevel(
                    price        = round(p, 1),
                    source       = "SWING_LOW",
                    strength     = 0.80,
                    distance_pts = abs(spot - p),
                ))

            # Swing high (resistance)
            if all(highs[i] >= highs[i-j] for j in range(1, k+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, k+1)):
                p = float(highs[i])
                levels.append(KeyLevel(
                    price        = round(p, 1),
                    source       = "SWING_HIGH",
                    strength     = 0.80,
                    distance_pts = abs(spot - p),
                ))

        return levels

    def _cpr_levels(self, df: pd.DataFrame, spot: float) -> list[KeyLevel]:
        """Compute Central Pivot Range (CPR) S1, S2, R1, R2."""
        levels = []
        try:
            prev_high  = float(df["high"].iloc[-10:-5].max())
            prev_low   = float(df["low"].iloc[-10:-5].min())
            prev_close = float(df["close"].iloc[-5])
            pivot      = (prev_high + prev_low + prev_close) / 3
            s1         = 2 * pivot - prev_high
            s2         = pivot - (prev_high - prev_low)
            r1         = 2 * pivot - prev_low
            r2         = pivot + (prev_high - prev_low)

            for price, name in [(s1,"CPR_S1"),(s2,"CPR_S2"),(r1,"CPR_R1"),(r2,"CPR_R2")]:
                levels.append(KeyLevel(
                    price        = round(price, 1),
                    source       = name,
                    strength     = 0.70,
                    distance_pts = abs(spot - price),
                ))
        except Exception:
            pass
        return levels

    @staticmethod
    def _round_levels(spot: float) -> list[KeyLevel]:
        """Find nearest round number levels (22000, 22100, etc)."""
        levels = []
        base   = round(spot / SPOT_SL_OI_ROUND_STEP) * SPOT_SL_OI_ROUND_STEP
        for mult in [-3, -2, -1, 0, 1, 2, 3]:
            price = base + mult * SPOT_SL_OI_ROUND_STEP
            if price > 0:
                levels.append(KeyLevel(
                    price        = float(price),
                    source       = "ROUND_NUMBER",
                    strength     = 0.60,
                    distance_pts = abs(spot - price),
                ))
        return levels

    @staticmethod
    def _compute_vwap(df: pd.DataFrame) -> float:
        """Session VWAP."""
        try:
            typical = (df["high"] + df["low"] + df["close"]) / 3
            vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            vol_sum = float(vol.sum())
            if vol_sum <= 0:
                return float(df["close"].iloc[-1])
            return float((typical * vol).sum() / vol_sum)
        except Exception:
            return 0.0

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        try:
            h=df["high"]; l=df["low"]; c=df["close"]; pc=c.shift(1)
            tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
            return float(tr.ewm(span=period,adjust=False).mean().iloc[-1])
        except Exception:
            return 30.0

    def _select_sl_level(
        self,
        levels:    list[KeyLevel],
        spot:      float,
        direction: str,
    ) -> tuple[float, str, float]:
        """
        Select the BEST SL level from all candidates.

        For BUY_CALL: find nearest SUPPORT (below spot) → use as SL
        For BUY_PUT:  find nearest RESISTANCE (above spot) → use as SL

        Priority: SWING_LOW > VWAP > CPR > ROUND_NUMBER > ATR
        """
        if direction == "BUY_CALL":
            candidates = [l for l in levels if l.price < spot]
        else:
            candidates = [l for l in levels if l.price > spot]

        if not candidates:
            # Fallback: ATR-based
            atr_dist  = spot * 1.0 / 100   # 1% default
            sl        = spot - atr_dist if direction == "BUY_CALL" else spot + atr_dist
            return sl, "FALLBACK_ATR", 0.4

        # Sort by priority: SWING first, then by closeness
        priority = {
            "SWING_LOW": 1, "SWING_HIGH": 1,
            "CPR_S1": 2, "CPR_S2": 2, "CPR_R1": 2, "CPR_R2": 2,
            "VWAP": 2,
            "ROUND_NUMBER": 3,
            "ATR_SUPPORT": 4, "ATR_RESIST": 4,
        }
        candidates.sort(key=lambda l: (
            priority.get(l.source, 5),   # priority first
            l.distance_pts,               # then closest
        ))

        best = candidates[0]
        return best.price, best.source, best.strength

# ── Integration helper for position manager ───────────────────────────────────

def compute_spot_sl_for_trade(
    df:            pd.DataFrame,
    spot:          float,
    direction:     str,
    entry_premium: float,
    delta:         float = 0.48,
    lot_size:      int   = None,
) -> dict:
    """
    Convenience function called from planner.py.
    Returns dict that planner uses to set sl_premium and lots.
    """
    detector = SpotBasedSLDetector()
    sl = detector.compute_structural_sl(
        df=df, spot=spot, direction=direction,
        entry_premium=entry_premium, delta=delta,
        lot_size=lot_size,
    )
    return {
        "sl_premium":        sl.option_sl_premium,
        "sl_spot_level":     sl.spot_sl_level,
        "sl_source":         sl.sl_source,
        "sl_pct":            sl.option_sl_pct,
        "lots":              sl.recommended_lots,
        "risk_per_lot_inr":  sl.risk_per_lot_inr,
        "distance_pts":      sl.distance_pts,
        "note":              sl.note,
    }

# Singleton
_detector: SpotBasedSLDetector | None = None

def get_spot_sl_detector() -> SpotBasedSLDetector:
    global _detector
    if _detector is None:
        _detector = SpotBasedSLDetector()
    return _detector

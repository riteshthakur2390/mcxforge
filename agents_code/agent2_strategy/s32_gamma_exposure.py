"""
agents_code/agent2_strategy/s32_gamma_exposure.py — Gamma Exposure (GEX)
=========================================================================
Option market makers have massive gamma at key strikes.
They MUST hedge when price approaches → creates predictable support/resistance.
GEX predicts WHERE price will find support/resistance BEFORE it gets there.

GEX = sum(gamma × OI × lot_size × spot²/100) per strike
Positive GEX = market makers long gamma = they BUY dips, SELL rallies = dampening
Negative GEX = market makers short gamma = they SELL dips, BUY rallies = amplifying

SIGNAL:
  Price approaching +GEX wall from below = resistance (likely reversal)
  Price breaking through -GEX zone = explosive move (amplified by MMs)
  Price at zero-GEX level = transition zone = high momentum expected
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
try:
    from core.models import Direction
except ImportError:
    class Direction:
        NONE="NONE"; BUY_CALL="BUY_CALL"; BUY_PUT="BUY_PUT"
try:
    from instruments.registry import get_instrument_config
    _cfg = get_instrument_config("SILVERM")
    COMMODITY_STRIKE_STEP = getattr(_cfg, "strike_step", 1000) or 1000
except Exception:
    COMMODITY_STRIKE_STEP = 1000

MIN_CANDLES = 15
GEX_WALL_DIST_PCT = 0.3   # within 0.3% of GEX wall


class GammaExposureStrategy:
    """S32: GEX-based support/resistance with proxy calculation."""
    name = "GammaExposure"

    def __init__(self):
        # GEX levels updated externally from option chain
        self._gex_walls: dict[int, float] = {}   # {strike: gex_value}
        self._net_gex = 0.0

    def update_gex(self, gex_by_strike: dict[int, float]) -> None:
        """Call with GEX values from option chain. {strike: gex_inr_cr}"""
        self._gex_walls = gex_by_strike
        self._net_gex   = sum(gex_by_strike.values())

    def reset(self) -> None:
        """Clear state between sessions to ensure clean isolation."""
        self._gex_walls.clear()
        self._net_gex = 0.0

    def evaluate(self, df, orb_high=None, orb_low=None, **kwargs):
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if df is None or len(df) < MIN_CANDLES:
            return none
        try:
            close = float(df["close"].iloc[-1])
            prev  = float(df["close"].iloc[-2])

            # If live GEX data is absent, evaluate round-strike gamma explosion proxy
            if not self._gex_walls:
                return self._proxy_gex(df, close, prev)

            # Find nearest GEX wall
            walls = sorted(self._gex_walls.items(), key=lambda x: abs(x[0]-close))
            if not walls:
                return none

            nearest_strike, nearest_gex = walls[0]
            dist_pct = abs(close - nearest_strike) / close * 100

            if dist_pct > GEX_WALL_DIST_PCT:
                return none

            # Positive GEX wall above = resistance (sell call / buy put)
            # Positive GEX wall below = support  (buy call)
            if nearest_gex > 0:
                if nearest_strike > close:  # resistance above
                    direction = Direction.BUY_PUT
                    signal    = "GEX_WALL_RESISTANCE"
                else:                        # support below
                    direction = Direction.BUY_CALL
                    signal    = "GEX_WALL_SUPPORT"
            else:
                # Negative GEX = amplifying move in current direction
                direction = Direction.BUY_CALL if close > prev else Direction.BUY_PUT
                signal    = "NEGATIVE_GEX_AMPLIFY"

            gex_strength = min(1.0, abs(nearest_gex) / 1000)
            conf = 0.64 + gex_strength * 0.12
            conf = round(min(0.82, conf), 4)

            return {"direction": direction, "confidence": conf, "name": self.name,
                    "meta": {"signal": signal, "nearest_strike": nearest_strike,
                             "nearest_gex": round(nearest_gex,1),
                             "dist_pct": round(dist_pct,3), "net_gex": round(self._net_gex,1)}}
        except Exception:
            return none

    def _proxy_gex(self, df, close: float, prev: float) -> dict:
        """Proxy GEX from price/volume when no live option chain is connected.
        Identifies Gamma Explosion / Strike Walls around key commodity round numbers.
        """
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        try:
            step = COMMODITY_STRIKE_STEP  # 500 pt strikes in SILVERM
            atm = int(round(close / step) * step)
            dist = abs(close - atm)

            # Must be near the strike barrier (within 100 points of the 500-point round level)
            if dist > 100:
                return none

            # Significant volume surge required (institutional gamma positioning)
            vols = df["volume"].values
            vol_avg = float(np.mean(vols[-20:]))
            vol_ratio = float(vols[-1]) / max(vol_avg, 1)
            if vol_ratio < 1.75:
                return none

            o = float(df["open"].iloc[-1])
            h = float(df["high"].iloc[-1])
            l = float(df["low"].iloc[-1])

            # 1. Breakout Expansion through Strike Barrier
            if prev < atm <= close and close > o:
                direction = Direction.BUY_CALL
                signal = "GEX_STRIKE_BREAKOUT_UP"
            elif prev > atm >= close and close < o:
                direction = Direction.BUY_PUT
                signal = "GEX_STRIKE_BREAKOUT_DOWN"
            # 2. Strike Barrier Bounce / Defense
            elif l <= atm and close > atm and close > o:
                direction = Direction.BUY_CALL
                signal = "GEX_STRIKE_SUPPORT_BOUNCE"
            elif h >= atm and close < atm and close < o:
                direction = Direction.BUY_PUT
                signal = "GEX_STRIKE_RESISTANCE_REJECT"
            else:
                return none

            conf = 0.68 + min(0.12, (vol_ratio - 1.75) * 0.08)
            conf = round(min(0.84, conf), 4)

            return {
                "direction": direction,
                "confidence": conf,
                "name": self.name,
                "meta": {
                    "signal": signal,
                    "atm": atm,
                    "dist_pts": round(dist, 1),
                    "vol_ratio": round(vol_ratio, 2),
                },
            }
        except Exception:
            return none

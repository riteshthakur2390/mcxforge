"""
agents_code/agent2_strategy/s26_stoch_rsi.py — Stochastic RSI Leading Signal
==============================================================================
RSI of RSI — fires 3-4 candles BEFORE standard RSI.

StochRSI = (RSI - lowest_RSI_14) / (highest_RSI_14 - lowest_RSI_14)
K = 3-bar SMA of StochRSI
D = 3-bar SMA of K

SIGNAL:
  K crosses above 20 from below → oversold reversal → BUY_CALL
  K crosses below 80 from above → overbought reversal → BUY_PUT
  K crosses D while in zone → confirmation

LEADING ADVANTAGE:
  StochRSI reaches 0/100 much faster than RSI reaches 30/70.
  Entry signal arrives 3-4 candles earlier than RSI-based systems.
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

MIN_CANDLES = 30
RSI_PERIOD  = 14
STOCH_PERIOD= 14
K_PERIOD    = 3
D_PERIOD    = 3
OVERSOLD    = 20.0
OVERBOUGHT  = 80.0


class StochRSIStrategy:
    """S26: Stochastic RSI — leading momentum reversal signal."""
    name = "StochRSI"

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None, **kwargs) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if df is None or len(df) < MIN_CANDLES:
            return none
        try:
            return self._evaluate(df, kwargs)
        except Exception:
            return none

    def _evaluate(self, df: pd.DataFrame, kwargs: dict) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        close = df["close"]
        rsi   = self._rsi(close, RSI_PERIOD)

        # Stochastic of RSI
        rsi_min = rsi.rolling(STOCH_PERIOD).min()
        rsi_max = rsi.rolling(STOCH_PERIOD).max()
        rsi_range = (rsi_max - rsi_min).replace(0, 0.001)
        stoch_rsi = (rsi - rsi_min) / rsi_range * 100

        K = stoch_rsi.rolling(K_PERIOD).mean()
        D = K.rolling(D_PERIOD).mean()

        k_now  = float(K.iloc[-1])
        k_prev = float(K.iloc[-2])
        d_now  = float(D.iloc[-1])

        # Crossings
        cross_up   = k_prev <= OVERSOLD   and k_now > OVERSOLD    # oversold → recovery
        cross_down = k_prev >= OVERBOUGHT and k_now < OVERBOUGHT  # overbought → reversal
        k_cross_d_up   = float(K.iloc[-2]) < float(D.iloc[-2]) and k_now > d_now
        k_cross_d_down = float(K.iloc[-2]) > float(D.iloc[-2]) and k_now < d_now

        if cross_up or (k_cross_d_up and k_now < 50):
            direction = Direction.BUY_CALL
            strength  = (OVERSOLD - min(k_prev, OVERSOLD)) / OVERSOLD
        elif cross_down or (k_cross_d_down and k_now > 50):
            direction = Direction.BUY_PUT
            strength  = (max(k_prev, OVERBOUGHT) - OVERBOUGHT) / (100 - OVERBOUGHT)
        else:
            return none

        rsi_now = float(rsi.iloc[-1])
        close_now = float(close.iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        if direction == Direction.BUY_CALL and not (rsi_now <= 45 or close_now > ema21):
            return none
        if direction == Direction.BUY_PUT and not (rsi_now >= 55 or close_now < ema21):
            return none

        # Regime filter
        regime = kwargs.get("regime_details", {}).get("regime", "NEUTRAL")
        if regime == "TRENDING" and direction == Direction.BUY_CALL and k_now < 30:
            pass   # oversold in uptrend = good
        elif regime == "TRENDING" and direction == Direction.BUY_PUT and k_now > 70:
            pass   # overbought in downtrend = good
        elif regime == "RANGING":
            pass   # StochRSI works well in ranges
        # else allow through

        conf = 0.62 + min(0.12, strength * 0.15)
        if cross_up and cross_down:
            conf = 0.64
        if abs(k_now - d_now) > 5:
            conf += 0.03   # K and D diverging = stronger signal

        conf = round(min(0.82, max(0.56, conf)), 4)

        return {
            "direction":  direction,
            "confidence": conf,
            "name":       self.name,
            "meta": {
                "signal":      "STOCHRSI_OVERSOLD_CROSS" if cross_up else "STOCHRSI_OVERBOUGHT_CROSS",
                "k_now":       round(k_now, 1),
                "d_now":       round(d_now, 1),
                "k_prev":      round(k_prev, 1),
                "rsi_now":     round(rsi_now, 1),
                "ema21":       round(ema21, 1),
            },
        }

    @staticmethod
    def _rsi(close: pd.Series, p: int = 14) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(span=p, adjust=False).mean()
        loss  = (-delta).clip(lower=0).ewm(span=p, adjust=False).mean()
        rs    = gain / loss.replace(0, 1e-10)
        return 100 - 100 / (1 + rs)

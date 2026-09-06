"""
agents_code/agent2_strategy/s27_ema_slope.py — EMA Slope Acceleration
======================================================================
Rate of change of EMA slope — fires 2-3 candles before ADX confirms trend.
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

MIN_CANDLES = 25

class EMASlopeStrategy:
    name = "EMASlope"

    def evaluate(self, df, orb_high=None, orb_low=None, **kwargs):
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if df is None or len(df) < MIN_CANDLES:
            return none
        try:
            close  = df["close"]
            ema9   = close.ewm(span=9,  adjust=False).mean()
            ema21  = close.ewm(span=21, adjust=False).mean()

            # Slope of EMA9 over last 3 bars
            slope_now  = float(ema9.iloc[-1] - ema9.iloc[-4]) / 3
            slope_prev = float(ema9.iloc[-4] - ema9.iloc[-7]) / 3 if len(df) > 7 else slope_now
            accel      = slope_now - slope_prev   # second derivative of price

            # Slope acceleration threshold (normalized for commodity price scale)
            last_close = float(close.iloc[-1])
            min_accel = max(0.05, last_close * 0.000045)
            if abs(accel) < min_accel:  # need meaningful acceleration
                return none

            # Direction
            if accel > 0 and float(ema9.iloc[-1]) > float(ema21.iloc[-1]):
                direction = Direction.BUY_CALL
            elif accel < 0 and float(ema9.iloc[-1]) < float(ema21.iloc[-1]):
                direction = Direction.BUY_PUT
            else:
                return none

            # RSI confirmation
            delta = close.diff().dropna()
            g = delta.clip(lower=0).ewm(span=14,adjust=False).mean()
            l = (-delta).clip(lower=0).ewm(span=14,adjust=False).mean()
            rsi = float(100 - 100/(1 + g.iloc[-1]/max(l.iloc[-1],1e-10)))

            if direction == Direction.BUY_CALL and rsi < 45:
                return none
            if direction == Direction.BUY_PUT  and rsi > 55:
                return none

            norm_pct = (abs(accel) / max(last_close, 1.0)) * 100.0
            conf = 0.63 + min(0.12, (norm_pct / 0.10) * 0.12)
            conf = round(min(0.80, conf), 4)

            return {"direction": direction, "confidence": conf, "name": self.name,
                    "meta": {"slope_now": round(slope_now,3),
                             "accel": round(accel,3), "rsi": round(rsi,1)}}
        except Exception:
            return none

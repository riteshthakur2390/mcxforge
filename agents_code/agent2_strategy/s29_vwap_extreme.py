"""
agents_code/agent2_strategy/s29_vwap_extreme.py — VWAP Standard Deviation Extreme
====================================================================================
Price at 2nd/3rd SD of VWAP = mean reversion BEFORE it happens.
More predictive than Bollinger Bands (volume-weighted).
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

MIN_CANDLES = 20


class VWAPExtremeStrategy:
    name = "VWAPExtreme"

    def evaluate(self, df, orb_high=None, orb_low=None, **kwargs):
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if df is None or len(df) < MIN_CANDLES:
            return none
        try:
            tp   = (df["high"] + df["low"] + df["close"]) / 3
            vol  = df["volume"]
            vwap = (tp * vol).cumsum() / vol.cumsum()
            vwap_now = float(vwap.iloc[-1])

            # VWAP standard deviation
            deviations = (df["close"] - vwap) ** 2
            sd = float(np.sqrt(deviations.iloc[-20:].mean()))

            close = float(df["close"].iloc[-1])
            prev  = float(df["close"].iloc[-2])
            sd2_upper = vwap_now + 2 * sd
            sd2_lower = vwap_now - 2 * sd
            sd3_upper = vwap_now + 3 * sd
            sd3_lower = vwap_now - 3 * sd

            # At extreme + starting to revert
            at_upper2 = close >= sd2_upper and prev >= sd2_upper and float(df["close"].iloc[-3]) < sd2_upper
            at_lower2 = close <= sd2_lower and prev <= sd2_lower and float(df["close"].iloc[-3]) > sd2_lower
            at_upper3 = close >= sd3_upper
            at_lower3 = close <= sd3_lower

            # Reverting back toward VWAP
            o = float(df["open"].iloc[-1])
            h = float(df["high"].iloc[-1])
            l = float(df["low"].iloc[-1])
            candle_range = max(h - l, 0.01)
            upper_reject = (h - max(o, close)) / candle_range >= 0.35 and close < o
            lower_reject = (min(o, close) - l) / candle_range >= 0.35 and close > o
            rsi = self._rsi(df["close"])

            reverting_bear = at_upper2 and close < prev and upper_reject and rsi >= 65
            reverting_bull = at_lower2 and close > prev and lower_reject and rsi <= 35

            if (at_upper3 and upper_reject and rsi >= 70) or reverting_bear:
                direction = Direction.BUY_PUT
                sigma = (close - vwap_now) / max(sd, 1)
            elif (at_lower3 and lower_reject and rsi <= 30) or reverting_bull:
                direction = Direction.BUY_CALL
                sigma = (vwap_now - close) / max(sd, 1)
            else:
                return none

            conf = 0.65 + min(0.12, (sigma - 2) * 0.04)
            if at_upper3 or at_lower3:
                conf += 0.05  # 3rd SD = stronger reversion signal

            conf = round(min(0.84, max(0.58, conf)), 4)
            return {"direction": direction, "confidence": conf, "name": self.name,
                    "meta": {"vwap": round(vwap_now,1), "close": round(close,1),
                             "sigma": round(sigma,2), "sd": round(sd,1),
                             "rsi": round(rsi, 1),
                             "sd2_upper": round(sd2_upper,1), "sd2_lower": round(sd2_lower,1)}}
        except Exception:
            return none

    @staticmethod
    def _rsi(close: pd.Series, p: int = 14) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(span=p, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(span=p, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-10)
        return float((100 - 100 / (1 + rs)).iloc[-1])

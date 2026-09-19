"""
agents_code/agent2_strategy/s28_heikin_ashi.py — Heikin Ashi Trend Change
==========================================================================
HA candles smooth noise. Switch from red to green with NO lower wick = strong leading signal.
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
CONSECUTIVE_SAME = 3   # require a real prior HA leg before reversal


class HeikinAshiStrategy:
    name = "HeikinAshi"

    def evaluate(self, df, orb_high=None, orb_low=None, **kwargs):
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if df is None or len(df) < MIN_CANDLES:
            return none
        try:
            o = df["open"].values; h = df["high"].values
            l = df["low"].values;  c = df["close"].values

            # Compute Heikin Ashi
            ha_close = (o + h + l + c) / 4
            ha_open  = np.zeros(len(c))
            ha_open[0] = (o[0] + c[0]) / 2
            for i in range(1, len(c)):
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            ha_high  = np.maximum(h, np.maximum(ha_open, ha_close))
            ha_low   = np.minimum(l, np.minimum(ha_open, ha_close))

            # Candle color: green if ha_close > ha_open
            colors = ["G" if ha_close[i] > ha_open[i] else "R" for i in range(len(c))]

            # Last candle color change
            cur_color  = colors[-1]
            prev_color = colors[-2]
            ema21 = pd.Series(c).ewm(span=21, adjust=False).mean()
            close_now = float(c[-1])
            ema_now = float(ema21.iloc[-1])
            vol = df["volume"].values
            vol_avg   = float(np.mean(vol[-20:]))
            vol_ratio = float(vol[-1]) / max(vol_avg, 1)

            # Mode 1: Trend Continuation (same color streak in strong trend)
            if cur_color == prev_color:
                same_count = 0
                for i in range(len(colors)-1, max(-1, len(colors)-8), -1):
                    if colors[i] == cur_color:
                        same_count += 1
                    else:
                        break
                
                if same_count >= 3:
                    rng = max(ha_high[-1] - ha_low[-1], 1e-4)
                    if cur_color == "G" and close_now > ema_now:
                        lower_shadow = (min(ha_open[-1], ha_close[-1]) - ha_low[-1]) / rng
                        if lower_shadow <= 0.20:
                            conf = min(0.78, 0.64 + min(0.10, same_count * 0.02))
                            return {
                                "direction": Direction.BUY_CALL,
                                "confidence": round(conf, 4),
                                "name": self.name,
                                "meta": {"mode": "continuation", "streak": same_count, "ema21": round(ema_now, 1)}
                            }
                    elif cur_color == "R" and close_now < ema_now:
                        upper_shadow = (ha_high[-1] - max(ha_open[-1], ha_close[-1])) / rng
                        if upper_shadow <= 0.20:
                            conf = min(0.78, 0.64 + min(0.10, same_count * 0.02))
                            return {
                                "direction": Direction.BUY_PUT,
                                "confidence": round(conf, 4),
                                "name": self.name,
                                "meta": {"mode": "continuation", "streak": same_count, "ema21": round(ema_now, 1)}
                            }
                return none

            # Mode 2: Reversal (color switch after previous run)
            if cur_color == "G":
                direction = Direction.BUY_CALL
                no_lower_wick = ha_low[-1] == min(ha_open[-1], ha_close[-1])
            else:
                direction = Direction.BUY_PUT
                no_lower_wick = ha_high[-1] == max(ha_open[-1], ha_close[-1])

            # Count consecutive previous candles of opposite color
            consec = 0
            for i in range(len(colors)-2, max(0, len(colors)-8), -1):
                if colors[i] != cur_color:
                    consec += 1
                else:
                    break

            if consec < 2:
                return none

            if direction == Direction.BUY_CALL and close_now < ema_now * 0.998:
                return none
            if direction == Direction.BUY_PUT and close_now > ema_now * 1.002:
                return none

            if vol_ratio < 1.00:
                return none

            conf = 0.65
            if no_lower_wick:
                conf += 0.06
            if consec >= 3:
                conf += 0.04
            if vol_ratio > 1.3:
                conf += 0.03

            conf = round(min(0.82, conf), 4)

            return {"direction": direction, "confidence": conf, "name": self.name,
                    "meta": {"color_change": f"{prev_color}→{cur_color}",
                             "no_lower_wick": no_lower_wick,
                             "prev_consec": consec,
                             "vol_ratio": round(vol_ratio,2),
                             "ema21": round(ema_now, 1)}}
        except Exception:
            return none

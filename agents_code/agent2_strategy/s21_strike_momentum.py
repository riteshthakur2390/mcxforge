"""
agents_code/agent2_strategy/s21_strike_momentum.py — Multi-Strike Momentum
===========================================================================
Tracks premium movement across 5 strikes simultaneously.
If OTM calls rising faster than ATM = institutional directional positioning.
Far OTM CE buying = large directional bet by smart money → follow them.

SIGNAL LOGIC:
  BUY_CALL:
    OTM CE (ATM+100) premium rising > 20% faster than ATM CE
    OR 2+ OTM CE strikes simultaneously rising with volume
  BUY_PUT:
    OTM PE (ATM-100) premium rising > 20% faster than ATM PE
    OR 2+ OTM PE strikes simultaneously rising with volume
"""

from __future__ import annotations

from config.settings.strategy import (
    S21_MIN_CANDLES,
    S21_MOMENTUM_WINDOW,
    S21_OTM_ACCEL_THRESH,
)


from collections import deque
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
    COMMODITY_STRIKE_STEP = getattr(_cfg, "strike_step", 500) or 500
except Exception:
    COMMODITY_STRIKE_STEP = 500

class StrikeMomentumStrategy:
    """S21: Multi-strike premium momentum — follow where money is flowing."""
    name = "StrikeMomentum"

    def __init__(self):
        # Track premium history per strike: {strike: deque of premiums}
        self._ce_hist: dict[int, deque] = {}
        self._pe_hist: dict[int, deque] = {}

    def update_premiums(
        self,
        spot:          float,
        ce_premiums:   dict[int, float],   # {strike: premium}
        pe_premiums:   dict[int, float],
    ) -> None:
        """Call every candle with current option premiums."""
        for k, prem in ce_premiums.items():
            if k not in self._ce_hist:
                self._ce_hist[k] = deque(maxlen=S21_MOMENTUM_WINDOW + 2)
            self._ce_hist[k].append(prem)
        for k, prem in pe_premiums.items():
            if k not in self._pe_hist:
                self._pe_hist[k] = deque(maxlen=S21_MOMENTUM_WINDOW + 2)
            self._pe_hist[k].append(prem)

    def evaluate(
        self,
        df:       pd.DataFrame,
        orb_high: Optional[float] = None,
        orb_low:  Optional[float] = None,
        **kwargs,
    ) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if df is None or len(df) < S21_MIN_CANDLES:
            return none
        try:
            return self._evaluate(df, kwargs)
        except Exception:
            return none

    def _evaluate(self, df: pd.DataFrame, kwargs: dict) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        spot    = float(df["close"].iloc[-1])
        atm     = int(round(spot / COMMODITY_STRIKE_STEP) * COMMODITY_STRIKE_STEP)
        otm_ce  = atm + 2 * COMMODITY_STRIKE_STEP   # 2 strikes OTM call
        otm_pe  = atm - 2 * COMMODITY_STRIKE_STEP   # 2 strikes OTM put

        # If we have real premium history, use it
        call_momentum = self._compute_momentum(self._ce_hist, atm, otm_ce)
        put_momentum  = self._compute_momentum(self._pe_hist, atm, otm_pe)

        # Proxy: use price action as momentum proxy when no broker data
        if call_momentum == 0.0 and put_momentum == 0.0:
            closes = df["close"].values
            vol    = df["volume"].values
            vol_avg= float(np.mean(vol[-20:]))
            roc3   = (closes[-1] - closes[-4]) / closes[-4] * 100
            roc_accel = roc3 - (closes[-4] - closes[-7]) / closes[-7] * 100

            if roc3 > 0.3 and roc_accel > 0 and float(vol[-1]) > vol_avg:
                call_momentum = min(0.8, roc3 * 0.5)
            elif roc3 < -0.3 and roc_accel < 0 and float(vol[-1]) > vol_avg:
                put_momentum  = min(0.8, abs(roc3) * 0.5)

        if call_momentum > S21_OTM_ACCEL_THRESH:
            direction = Direction.BUY_CALL
            strength  = call_momentum
        elif put_momentum > S21_OTM_ACCEL_THRESH:
            direction = Direction.BUY_PUT
            strength  = put_momentum
        else:
            return none

        # ADX gate
        adx = self._adx(df)
        if adx < 16:
            return none

        conf = 0.64 + min(0.20, strength * 0.3)
        if adx > 25:
            conf += 0.03

        return {
            "direction":  direction,
            "confidence": round(min(0.84, conf), 4),
            "name":       self.name,
            "meta": {
                "call_momentum": round(call_momentum, 3),
                "put_momentum":  round(put_momentum, 3),
                "atm":           atm,
                "adx":           round(adx, 1),
            },
        }

    def _compute_momentum(
        self,
        hist:    dict[int, deque],
        atm:     int,
        otm:     int,
    ) -> float:
        """Compare OTM premium velocity vs ATM premium velocity."""
        if atm not in hist or otm not in hist:
            return 0.0
        atm_h = list(hist[atm])
        otm_h = list(hist[otm])
        if len(atm_h) < 3 or len(otm_h) < 3:
            return 0.0
        atm_chg = (atm_h[-1] - atm_h[0]) / max(atm_h[0], 1)
        otm_chg = (otm_h[-1] - otm_h[0]) / max(otm_h[0], 1)
        # OTM accelerating faster than ATM = directional bet
        return max(0.0, otm_chg - atm_chg)

    @staticmethod
    def _adx(df: pd.DataFrame, p: int = 14) -> float:
        try:
            h,l,c=df["high"],df["low"],df["close"]; pc=c.shift(1)
            tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
            return float(tr.ewm(span=p,adjust=False).mean().iloc[-1])
        except Exception:
            return 20.0

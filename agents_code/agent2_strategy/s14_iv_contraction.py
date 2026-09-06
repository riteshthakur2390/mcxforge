import pandas as pd
import numpy as np
import pandas_ta as ta
from dataclasses import dataclass
from typing import Union
from core.models import Direction
from config.settings import strategy as s
from loguru import logger

@dataclass
class CompressionZone:
    high: float; low: float; width: float; width_pct: float; coil_candles: int
    atr_ratio: float; bb_ratio: float; hv_rank: float; iv_contraction: float = 1.0

class IVContractionStrategy:
    name = "IVContraction"
    def __init__(self) -> None: self._recorder = None
    def set_recorder(self, recorder) -> None: self._recorder = recorder

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None, cache=None) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if len(df) < s.S14_BB_PERIOD + s.S14_HV_LOOKBACK: return none
        zone = self._detect_compression(df, cache=cache)
        if zone is None: return none
        close = float(df["close"].iloc[-1])
        vol = float(df["volume"].iloc[-1])
        vol_ma = float(cache.vol_ma_20.iloc[-1]) if cache and cache.vol_ma_20 is not None else float(df["volume"].rolling(s.S14_VOL_MA_PERIOD).mean().iloc[-1])
        vol_ratio = vol / max(vol_ma, 1)
        if vol_ratio < s.S14_VOL_BREAKOUT_MULT: return none
        ema50 = float(cache.ema_50.iloc[-1]) if cache and cache.ema_50 is not None else float(ta.ema(df["close"], s.S14_EMA_TREND).iloc[-1])
        o, h, l = float(df["open"].iloc[-1]), float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
        body_pct = abs(close - o) / max(h - l, 0.01)
        
        # Actual IV check if recorder available
        iv_now = 0.14; iv_prev = 0.14; iv_ratio = 1.0
        if self._recorder:
            atm = int(round(close / 50) * 50)
            iv_now = self._recorder.get_iv_at_candle(0, strike=atm) or 0.14
            iv_prev = self._recorder.get_iv_at_candle(5, strike=atm) or 0.14
            iv_ratio = iv_now / iv_prev if iv_prev > 0 else 1.0

        if close > zone.high and close > ema50:
            return {"direction": Direction.BUY_CALL, "confidence": self._score(zone, vol_ratio, body_pct, True, "bullish", iv_ratio), "name": self.name, "meta": {"iv_ratio": round(iv_ratio, 2), "hv_rank": round(zone.hv_rank, 2)}}
        if close < zone.low and close < ema50:
            return {"direction": Direction.BUY_PUT, "confidence": self._score(zone, vol_ratio, body_pct, True, "bearish", iv_ratio), "name": self.name, "meta": {"iv_ratio": round(iv_ratio, 2), "hv_rank": round(zone.hv_rank, 2)}}
        return none

    def _detect_compression(self, df: pd.DataFrame, cache=None) -> Union[CompressionZone, None]:
        try:
            # RELAXED: Check compression state as of the PREVIOUS candle
            # because the current candle might be the breakout candle (expanding ATR/BB)
            atr = cache.atr_14 if cache and cache.atr_14 is not None else ta.atr(df["high"], df["low"], df["close"], 14)
            if len(atr) < 22: return None
            atr_now, atr_past = float(atr.iloc[-2]), float(atr.iloc[-21])
            atr_ratio = atr_now / atr_past
            
            # RELAXED: was 0.85, now 0.95
            if atr_ratio >= 0.95: return None
            
            bb_w = cache.bb_width if cache and cache.bb_width is not None else None
            if bb_w is None:
                bb = ta.bbands(df["close"], 20, 2.0); bb_w = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / df["close"] * 100
            
            bb_now, bb_past = float(bb_w.iloc[-2]), float(bb_w.iloc[-21])
            bb_ratio = bb_now / bb_past
            
            # RELAXED: was 0.8, now 0.9
            if atr_ratio >= 0.9 and bb_ratio >= 0.9: return None
            
            coil_count = 0
            for i in range(1, 15):
                c = df.iloc[-i-1]; body = abs(c["close"] - c["open"]) / c["close"] * 100
                if body <= 0.3: coil_count += 1
                else: break
            if coil_count < 3: return None
            window = df.iloc[-(coil_count+2):-1]
            return CompressionZone(high=float(window["high"].max()), low=float(window["low"].min()), width=0, width_pct=0, coil_candles=coil_count, atr_ratio=atr_ratio, bb_ratio=bb_ratio, hv_rank=0.5)
        except: return None

    @staticmethod
    def _score(zone, vol_ratio, body_pct, ema200, side, iv_ratio):
        conf = 0.65
        if zone.coil_candles > 5: conf += 0.05
        if iv_ratio < 0.95: conf += 0.05  # Bonus for actual IV contraction
        if vol_ratio > 1.8: conf += 0.03
        return round(min(0.88, conf), 2)

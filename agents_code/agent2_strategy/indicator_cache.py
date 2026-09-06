"""
agents_code/agent2_strategy/indicator_cache.py
================================================
Lazy, on-demand cached technical indicators per candle batch.
Strategies read from cache instead of recomputing independently.
Indicators are computed ONLY when first accessed by an active strategy,
eliminating unneeded heavy calculations (like PSAR, Keltner, etc.).
"""

import pandas as pd
import numpy as np
from loguru import logger

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


class IndicatorCache:
    """
    Computes technical indicators lazily on first access and caches them.
    Passed to each strategy's evaluate() call.
    Strategies check `if cache:` before using — fully backward compatible.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._data: dict = {}

    def __bool__(self) -> bool:
        return TA_AVAILABLE and self._df is not None and not self._df.empty

    # --- MOMENTUM ---
    @property
    def rsi_14(self):
        if "rsi_14" not in self._data:
            try:
                self._data["rsi_14"] = ta.rsi(self._df["close"], 14) if TA_AVAILABLE else None
            except Exception:
                self._data["rsi_14"] = None
        return self._data["rsi_14"]

    @property
    def rsi_9(self):
        if "rsi_9" not in self._data:
            try:
                self._data["rsi_9"] = ta.rsi(self._df["close"], 9) if TA_AVAILABLE else None
            except Exception:
                self._data["rsi_9"] = None
        return self._data["rsi_9"]

    def _ensure_macd(self):
        if "macd_hist" not in self._data:
            try:
                macd = ta.macd(self._df["close"]) if TA_AVAILABLE else None
                if macd is not None:
                    self._data["macd_hist"] = macd.get("MACDh_12_26_9")
                    self._data["macd_signal"] = macd.get("MACDs_12_26_9")
                    self._data["macd_line"] = macd.get("MACD_12_26_9")
                else:
                    self._data["macd_hist"] = self._data["macd_signal"] = self._data["macd_line"] = None
            except Exception:
                self._data["macd_hist"] = self._data["macd_signal"] = self._data["macd_line"] = None

    @property
    def macd_hist(self):
        self._ensure_macd()
        return self._data.get("macd_hist")

    @property
    def macd_signal(self):
        self._ensure_macd()
        return self._data.get("macd_signal")

    @property
    def macd_line(self):
        self._ensure_macd()
        return self._data.get("macd_line")

    def _ensure_stoch(self):
        if "stoch_k" not in self._data:
            try:
                stoch = ta.stoch(self._df["high"], self._df["low"], self._df["close"]) if TA_AVAILABLE else None
                if stoch is not None:
                    self._data["stoch_k"] = stoch.get("STOCHk_14_3_3")
                    self._data["stoch_d"] = stoch.get("STOCHd_14_3_3")
                else:
                    self._data["stoch_k"] = self._data["stoch_d"] = None
            except Exception:
                self._data["stoch_k"] = self._data["stoch_d"] = None

    @property
    def stoch_k(self):
        self._ensure_stoch()
        return self._data.get("stoch_k")

    @property
    def stoch_d(self):
        self._ensure_stoch()
        return self._data.get("stoch_d")

    # --- TREND ---
    def _ensure_adx(self):
        if "adx" not in self._data:
            try:
                adx = ta.adx(self._df["high"], self._df["low"], self._df["close"], 14) if TA_AVAILABLE else None
                if adx is not None:
                    self._data["adx"] = adx.get("ADX_14")
                    self._data["plus_di"] = adx.get("DMP_14")
                    self._data["minus_di"] = adx.get("DMN_14")
                else:
                    self._data["adx"] = self._data["plus_di"] = self._data["minus_di"] = None
            except Exception:
                self._data["adx"] = self._data["plus_di"] = self._data["minus_di"] = None

    @property
    def adx(self):
        self._ensure_adx()
        return self._data.get("adx")

    @property
    def plus_di(self):
        self._ensure_adx()
        return self._data.get("plus_di")

    @property
    def minus_di(self):
        self._ensure_adx()
        return self._data.get("minus_di")

    def _get_ema(self, period: int):
        key = f"ema_{period}"
        if key not in self._data:
            try:
                self._data[key] = ta.ema(self._df["close"], period) if TA_AVAILABLE else None
            except Exception:
                self._data[key] = None
        return self._data[key]

    @property
    def ema_9(self): return self._get_ema(9)
    @property
    def ema_20(self): return self._get_ema(20)
    @property
    def ema_21(self): return self._get_ema(21)
    @property
    def ema_50(self): return self._get_ema(50)
    @property
    def ema_200(self): return self._get_ema(200)

    @property
    def psar(self):
        if "psar" not in self._data:
            try:
                self._data["psar"] = ta.psar(self._df["high"], self._df["low"], self._df["close"]) if TA_AVAILABLE else None
            except Exception:
                self._data["psar"] = None
        return self._data["psar"]

    # --- VOLATILITY ---
    @property
    def atr_14(self):
        if "atr_14" not in self._data:
            try:
                self._data["atr_14"] = ta.atr(self._df["high"], self._df["low"], self._df["close"], 14) if TA_AVAILABLE else None
            except Exception:
                self._data["atr_14"] = None
        return self._data["atr_14"]

    @property
    def atr_10(self):
        if "atr_10" not in self._data:
            try:
                self._data["atr_10"] = ta.atr(self._df["high"], self._df["low"], self._df["close"], 10) if TA_AVAILABLE else None
            except Exception:
                self._data["atr_10"] = None
        return self._data["atr_10"]

    def _ensure_bb(self):
        if "bb_upper" not in self._data:
            try:
                bb = ta.bbands(self._df["close"], 20, 2.0) if TA_AVAILABLE else None
                if bb is not None:
                    c = self._df["close"]
                    self._data["bb_upper"] = bb.get("BBU_20_2.0")
                    self._data["bb_lower"] = bb.get("BBL_20_2.0")
                    self._data["bb_mid"] = bb.get("BBM_20_2.0")
                    self._data["bb_range"] = self._data["bb_upper"] - self._data["bb_lower"]
                    self._data["bb_width"] = (self._data["bb_range"] / c) * 100
                else:
                    self._data["bb_upper"] = self._data["bb_lower"] = self._data["bb_mid"] = None
                    self._data["bb_range"] = self._data["bb_width"] = None
            except Exception:
                self._data["bb_upper"] = self._data["bb_lower"] = self._data["bb_mid"] = None
                self._data["bb_range"] = self._data["bb_width"] = None

    @property
    def bb_upper(self): self._ensure_bb(); return self._data.get("bb_upper")
    @property
    def bb_lower(self): self._ensure_bb(); return self._data.get("bb_lower")
    @property
    def bb_mid(self): self._ensure_bb(); return self._data.get("bb_mid")
    @property
    def bb_range(self): self._ensure_bb(); return self._data.get("bb_range")
    @property
    def bb_width(self): self._ensure_bb(); return self._data.get("bb_width")

    # --- VOLUME & VWAP ---
    @property
    def vol_ma_20(self):
        if "vol_ma_20" not in self._data:
            try:
                self._data["vol_ma_20"] = self._df["volume"].rolling(20).mean()
            except Exception:
                self._data["vol_ma_20"] = None
        return self._data["vol_ma_20"]

    @property
    def vol_ratio(self):
        if "vol_ratio" not in self._data:
            try:
                v = self._df["volume"]
                ma = self.vol_ma_20
                self._data["vol_ratio"] = v / ma.replace(0, 1) if ma is not None else None
            except Exception:
                self._data["vol_ratio"] = None
        return self._data["vol_ratio"]

    @property
    def vwap(self):
        if "vwap" not in self._data:
            try:
                h = self._df["high"]
                l = self._df["low"]
                c = self._df["close"]
                v = self._df["volume"]
                tp = (h + l + c) / 3
                session = pd.Index(self._df.index).date
                cum_tpv = (tp * v).groupby(session).cumsum()
                cum_vol = v.groupby(session).cumsum()
                self._data["vwap"] = cum_tpv / cum_vol.replace(0, 1)
            except Exception:
                self._data["vwap"] = None
        return self._data["vwap"]

    def get(self, key: str, default=None):
        """Generic accessor for properties."""
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return self._data.get(key, default)

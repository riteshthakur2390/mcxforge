"""S1: SuperTrend + RSI + Volume Confirmation"""
import pandas as pd
import pandas_ta as ta
from core.models import Direction
from config.settings.strategy import (
    S1_LOOKBACK_CANDLES,
    S1_RSI_BEAR_ENTRY,
    S1_RSI_BEAR_OVERSOLD,
    S1_RSI_BULL_ENTRY,
    S1_RSI_BULL_OVERBOUGHT,
    S1_RSI_PERIOD,
    S1_SUPERTREND_LENGTH,
    S1_SUPERTREND_MULTIPLIER,
    S1_VOLUME_MA_PERIOD,
    S1_MIN_DF_LEN,
    S1_CONF_BASE_NEW,
    S1_VOL_RATIO_THRESH,
    S1_CONF_BONUS,
    S1_CONF_MAX_NEW,
)


class SuperTrendRSI:
    name = "SuperTrend+RSI"

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None, cache=None) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if len(df) < S1_MIN_DF_LEN:
            return none
            
        # Supertrend computation is slow, truncate to recent history
        df = df.tail(S1_LOOKBACK_CANDLES).copy()
        
        st = ta.supertrend(df["high"], df["low"], df["close"],
                           length=S1_SUPERTREND_LENGTH, multiplier=S1_SUPERTREND_MULTIPLIER)
        
        if st is None or st.empty:
            return none
            
        dir_col = [c for c in st.columns if "SUPERTd" in c][0]
        line_col = [c for c in st.columns if "SUPERT_" in c and "SUPERTd" not in c][0]
        df  = df.copy()
        df["st_dir"] = st[dir_col]
        df["st_line"] = st[line_col]
        df["rsi"] = cache.rsi_14.reindex(df.index) if cache and cache.rsi_14 is not None and S1_RSI_PERIOD == 14 else ta.rsi(df["close"], length=S1_RSI_PERIOD)
        df["vol_ma"] = cache.vol_ma_20.reindex(df.index) if cache and cache.vol_ma_20 is not None and S1_VOLUME_MA_PERIOD == 20 else df["volume"].rolling(S1_VOLUME_MA_PERIOD).mean()

        curr_dir = df["st_dir"].iloc[-1]
        prev_dir = df["st_dir"].iloc[-2]
        rsi      = df["rsi"].iloc[-1]
        vol_ma = float(df["vol_ma"].iloc[-1]) if not pd.isna(df["vol_ma"].iloc[-1]) else 0.0
        vol_ratio = (float(df["volume"].iloc[-1]) / vol_ma) if vol_ma > 0 else 1.0

        bull_flip = prev_dir == -1 and curr_dir == 1
        bear_flip = prev_dir == 1 and curr_dir == -1

        confidence = S1_CONF_BASE_NEW
        if vol_ratio >= S1_VOL_RATIO_THRESH:
            confidence += S1_CONF_BONUS
        if bull_flip and rsi > (S1_RSI_BULL_ENTRY + S1_RSI_BULL_OVERBOUGHT) / 2:
            confidence += S1_CONF_BONUS
        if bear_flip and rsi < (S1_RSI_BEAR_ENTRY + S1_RSI_BEAR_OVERSOLD) / 2:
            confidence += S1_CONF_BONUS
        confidence = round(min(S1_CONF_MAX_NEW, confidence), 2)

        if curr_dir == 1 and rsi > S1_RSI_BULL_ENTRY and rsi < S1_RSI_BULL_OVERBOUGHT:
            conf = confidence
            return {"direction": Direction.BUY_CALL, "confidence": conf,
                    "name": self.name, "meta": {"rsi": round(rsi, 1), "vol_ratio": round(vol_ratio, 2)}}
        if curr_dir == -1 and rsi < S1_RSI_BEAR_ENTRY and rsi > S1_RSI_BEAR_OVERSOLD:
            conf = confidence
            return {"direction": Direction.BUY_PUT, "confidence": conf,
                    "name": self.name, "meta": {"rsi": round(rsi, 1), "vol_ratio": round(vol_ratio, 2)}}
        return {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

"""S4: Bollinger Band Squeeze + Stochastic
FIX: Stoch threshold was 70/30 (overbought/oversold) which almost never
coincides with a squeeze breakout. Relaxed to 55/45  momentum direction."""
import pandas as pd
import pandas_ta as ta
from core.models import Direction
from config.settings.strategy import (
    S4_BB_LENGTH, S4_BB_STD, S4_STOCH_K, S4_STOCH_D, S4_STOCH_SMOOTH, S4_BBW_MA_PERIOD,
    S4_BBW_SQUEEZE_FACTOR, S4_BULL_STOCH_K, S4_BEAR_STOCH_K, S4_CONFIDENCE, S4_MIN_DF_LEN
)


class BBSqueeze:
    name = "BBSqueeze"

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None, cache=None) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        bb_upper = cache.bb_upper.reindex(df.index) if cache and cache.bb_upper is not None else None
        bb_lower = cache.bb_lower.reindex(df.index) if cache and cache.bb_lower is not None else None
        bb_mid = cache.bb_mid.reindex(df.index) if cache and cache.bb_mid is not None else None
        bb_width = cache.bb_width.reindex(df.index) if cache and cache.bb_width is not None else None
        stoch_k = cache.stoch_k.reindex(df.index) if cache and cache.stoch_k is not None else None
        if bb_upper is None or bb_lower is None or bb_mid is None or bb_width is None or stoch_k is None:
            bb = ta.bbands(df["close"], length=S4_BB_LENGTH, std=S4_BB_STD)
            stoch = ta.stoch(df["high"], df["low"], df["close"], k=S4_STOCH_K, d=S4_STOCH_D)
            if bb is None or stoch is None:
                return none
            bb_upper = bb[f"BBU_{S4_BB_LENGTH}_{S4_BB_STD}"]
            bb_lower = bb[f"BBL_{S4_BB_LENGTH}_{S4_BB_STD}"]
            bb_mid = bb[f"BBM_{S4_BB_LENGTH}_{S4_BB_STD}"]
            bb_width = (bb_upper - bb_lower) / df["close"] * 100
            stoch_k = stoch[f"STOCHk_{S4_STOCH_K}_{S4_STOCH_D}_{S4_STOCH_SMOOTH}"]
        if len(df) < S4_MIN_DF_LEN:
            return none

        df = df.copy()
        df["bb_w"] = bb_width
        df["sk"]   = stoch_k
        df["sk_prev"] = df["sk"].shift(1)

        width_ref = df["bb_w"].rolling(S4_BBW_MA_PERIOD).mean().iloc[-1]
        squeeze   = df["bb_w"].iloc[-1] <= width_ref * S4_BBW_SQUEEZE_FACTOR if not pd.isna(width_ref) else False
        close     = df["close"].iloc[-1]
        close_prev = df["close"].iloc[-2]
        bb_upper  = float(bb_upper.iloc[-1])
        bb_lower  = float(bb_lower.iloc[-1])
        bb_mid    = float(bb_mid.iloc[-1])
        sk        = df["sk"].iloc[-1]
        sk_prev   = df["sk_prev"].iloc[-1]

        if not squeeze:
            return none

        bull_break = close > bb_upper or (close > bb_mid and close > close_prev and sk > S4_BULL_STOCH_K and sk >= sk_prev)
        bear_break = close < bb_lower or (close < bb_mid and close < close_prev and sk < S4_BEAR_STOCH_K and sk <= sk_prev)

        if bull_break:
            return {"direction": Direction.BUY_CALL, "confidence": S4_CONFIDENCE,
                    "name": self.name,
                    "meta": {"bb_width": round(df["bb_w"].iloc[-1], 2), "stoch_k": round(sk, 1)}}
        if bear_break:
            return {"direction": Direction.BUY_PUT, "confidence": S4_CONFIDENCE,
                    "name": self.name,
                    "meta": {"bb_width": round(df["bb_w"].iloc[-1], 2), "stoch_k": round(sk, 1)}}
        return none

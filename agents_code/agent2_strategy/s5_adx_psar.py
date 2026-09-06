"""S5: ADX + Parabolic SAR + EMA200 Trend Filter"""
import pandas as pd
import pandas_ta as ta
from core.models import Direction
from config.settings.strategy import (
    S5_MIN_DF_LEN, S5_ADX_LEN, S5_ADX_THRESHOLD, S5_PSAR_TAIL, S5_EMA_LEN,
    S5_PSAR_AF, S5_PSAR_MAX_AF, S5_CONF_BASE, S5_CONF_ADX_BONUS,
    S5_CONF_ADX_DIVISOR
)


class ADXParabolicSAR:
    name = "ADX+PSAR"

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None, cache=None) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if len(df) < S5_MIN_DF_LEN:
            return none

        if cache and cache.adx is not None and cache.plus_di is not None and cache.minus_di is not None:
            adx = float(cache.adx.iloc[-1])
            plus_di = float(cache.plus_di.iloc[-1])
            minus_di = float(cache.minus_di.iloc[-1])
        else:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=S5_ADX_LEN)
            if adx_df is None:
                return none
            adx = float(adx_df[f"ADX_{S5_ADX_LEN}"].iloc[-1])
            plus_di = float(adx_df[f"DMP_{S5_ADX_LEN}"].iloc[-1])
            minus_di = float(adx_df[f"DMN_{S5_ADX_LEN}"].iloc[-1])

        if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
            return none
        
        # Massive optimization: PSAR is very slow to calculate. 
        # Only calculate PSAR and EMA if ADX is >= threshold since it's a prerequisite.
        if adx < S5_ADX_THRESHOLD:
            return none
            
        psar   = ta.psar(df["high"].tail(S5_PSAR_TAIL), df["low"].tail(S5_PSAR_TAIL), df["close"].tail(S5_PSAR_TAIL),
                          af=S5_PSAR_AF, max_af=S5_PSAR_MAX_AF)
        ema50 = cache.ema_50 if cache and cache.ema_50 is not None else ta.ema(df["close"], length=S5_EMA_LEN)

        if psar is None:
            return none

        # SAR below price = bullish (long signal active)
        psar_long_now  = not pd.isna(psar[f"PSARl_{S5_PSAR_AF}_{S5_PSAR_MAX_AF}"].iloc[-1])
        psar_long_prev = not pd.isna(psar[f"PSARl_{S5_PSAR_AF}_{S5_PSAR_MAX_AF}"].iloc[-2])
        sar_flip_up    = not psar_long_prev and psar_long_now
        sar_flip_down  = psar_long_prev and not psar_long_now

        close = float(df["close"].iloc[-1])
        close_prev = float(df["close"].iloc[-2])
        e50  = float(ema50.iloc[-1]) if ema50 is not None and not pd.isna(ema50.iloc[-1]) else 0

        bull_trend = adx >= S5_ADX_THRESHOLD and psar_long_now and plus_di >= minus_di and close > e50 and close >= close_prev
        bear_trend = adx >= S5_ADX_THRESHOLD and (not psar_long_now) and minus_di >= plus_di and close < e50 and close <= close_prev

        if bull_trend and (sar_flip_up or close > e50):
            conf = round(S5_CONF_BASE + min(S5_CONF_ADX_BONUS, max(0.0, adx - S5_ADX_THRESHOLD) / S5_CONF_ADX_DIVISOR), 4)
            return {"direction": Direction.BUY_CALL, "confidence": conf,
                    "name": self.name, "meta": {"adx": round(adx, 1)}}

        if bear_trend and (sar_flip_down or close < e50):
            conf = round(S5_CONF_BASE + min(S5_CONF_ADX_BONUS, max(0.0, adx - S5_ADX_THRESHOLD) / S5_CONF_ADX_DIVISOR), 4)
            return {"direction": Direction.BUY_PUT, "confidence": conf,
                    "name": self.name, "meta": {"adx": round(adx, 1)}}

        return none

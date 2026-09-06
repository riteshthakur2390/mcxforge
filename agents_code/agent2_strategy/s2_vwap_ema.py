from loguru import logger
"""S2: VWAP + EMA Crossover + MACD
FIX: allow cross within last 3 candles (not just current candle).
EMA9/21 cross is a single-candle event  requiring it on exactly
the current candle made this strategy nearly silent."""
import pandas as pd
import pandas_ta as ta
from core.models import Direction
from config.settings.strategy import (
    S2_EMA_SHORT, S2_EMA_LONG, S2_MACD_FAST, S2_MACD_SLOW, S2_MACD_SIGNAL,
    S2_RECENT_CROSS_LOOKBACK, S2_CONF_CROSS_BASE, S2_CONF_CROSS_GAP_BONUS,
    S2_CONF_CROSS_GAP_DIVISOR, S2_CONF_TREND_BASE, S2_CONF_TREND_GAP_BONUS,
    S2_CONF_TREND_GAP_DIVISOR
)


class VWAPEMACross:
    name = "VWAP+EMA"

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None, cache=None) -> dict:
        df = df.copy()
        # Intraday VWAP
        if cache and cache.vwap is not None:
            df["vwap"] = cache.vwap.reindex(df.index)
        else:
            df["tp"]   = (df["high"] + df["low"] + df["close"]) / 3
            df["date"] = [i.date() for i in df.index]
            tpv        = df["tp"] * df["volume"]
            cumtpv     = tpv.groupby(df["date"]).cumsum()
            cumv       = df["volume"].groupby(df["date"]).cumsum()
            df["vwap"] = cumtpv / cumv

        df["ema9"] = cache.ema_9.reindex(df.index) if cache and cache.ema_9 is not None else ta.ema(df["close"], length=S2_EMA_SHORT)
        df["ema21"] = cache.ema_21.reindex(df.index) if cache and cache.ema_21 is not None else ta.ema(df["close"], length=S2_EMA_LONG)
        if cache and cache.macd_hist is not None:
            df["macd_hist"] = cache.macd_hist.reindex(df.index)
        else:
            macd_df = ta.macd(df["close"], fast=S2_MACD_FAST, slow=S2_MACD_SLOW, signal=S2_MACD_SIGNAL)
            df["macd_hist"] = macd_df[f"MACDh_{S2_MACD_FAST}_{S2_MACD_SLOW}_{S2_MACD_SIGNAL}"] if macd_df is not None else 0.0

        try:
            close   = float(df["close"].iloc[-1])
            close_prev = float(df["close"].iloc[-2])
            vwap    = float(df["vwap"].iloc[-1]) if pd.notna(df["vwap"].iloc[-1]) else close
            e9_now  = float(df["ema9"].iloc[-1]) if pd.notna(df["ema9"].iloc[-1]) else close
            e9_prev  = float(df["ema9"].iloc[-2]) if pd.notna(df["ema9"].iloc[-2]) else close
            e21_now = float(df["ema21"].iloc[-1]) if pd.notna(df["ema21"].iloc[-1]) else close
            e21_prev = float(df["ema21"].iloc[-2]) if pd.notna(df["ema21"].iloc[-2]) else close
            mhist   = float(df["macd_hist"].iloc[-1]) if pd.notna(df["macd_hist"].iloc[-1]) else 0.0
            mhist_prev = float(df["macd_hist"].iloc[-2]) if pd.notna(df["macd_hist"].iloc[-2]) else 0.0
        except Exception as e:
            logger.error(f"[S2] Error extracting values: {e}")
            return {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        cross_up   = e9_prev <= e21_prev and e9_now > e21_now
        cross_down = e9_prev >= e21_prev and e9_now < e21_now
        bull_trend = e9_now > e21_now and close > vwap and mhist > 0 and mhist >= mhist_prev and close >= close_prev
        bear_trend = e9_now < e21_now and close < vwap and mhist < 0 and mhist <= mhist_prev and close <= close_prev

        # FIX: check if cross happened within last 3 candles (not just current)
        # This makes S2 fire on the candles AFTER the cross while trend confirms
        cross_up_recent   = False
        cross_down_recent = False
        for i in range(1, min(S2_RECENT_CROSS_LOOKBACK, len(df))):
            try:
                e9_c_val = df["ema9"].iloc[-i]
                e21_c_val = df["ema21"].iloc[-i]
                e9_p_val = df["ema9"].iloc[-i-1]
                e21_p_val = df["ema21"].iloc[-i-1]
                
                if pd.isna(e9_c_val) or pd.isna(e21_c_val) or pd.isna(e9_p_val) or pd.isna(e21_p_val):
                    continue
                    
                e9_c  = float(e9_c_val)
                e21_c = float(e21_c_val)
                e9_p  = float(e9_p_val)
                e21_p = float(e21_p_val)
                
                if e9_p <= e21_p and e9_c > e21_c:
                    cross_up_recent = True
                if e9_p >= e21_p and e9_c < e21_c:
                    cross_down_recent = True
            except Exception:
                continue

        if cross_up_recent and close > vwap and mhist > 0:
            gap = (close - vwap) / max(vwap, 1) * 100
            conf = round(S2_CONF_CROSS_BASE + min(S2_CONF_CROSS_GAP_BONUS, gap / S2_CONF_CROSS_GAP_DIVISOR), 4)
            return {"direction": Direction.BUY_CALL, "confidence": conf,
                    "name": self.name, "meta": {"vwap": round(vwap, 2)}}
        if bull_trend:
            gap = (close - vwap) / max(vwap, 1) * 100
            conf = round(S2_CONF_TREND_BASE + min(S2_CONF_TREND_GAP_BONUS, gap / S2_CONF_TREND_GAP_DIVISOR), 4)
            return {"direction": Direction.BUY_CALL, "confidence": conf,
                    "name": self.name, "meta": {"vwap": round(vwap, 2), "mode": "trend_cont"}}
        if cross_down_recent and close < vwap and mhist < 0:
            gap = (vwap - close) / max(vwap, 1) * 100
            conf = round(S2_CONF_CROSS_BASE + min(S2_CONF_CROSS_GAP_BONUS, gap / S2_CONF_CROSS_GAP_DIVISOR), 4)
            return {"direction": Direction.BUY_PUT, "confidence": conf,
                    "name": self.name, "meta": {"vwap": round(vwap, 2)}}
        if bear_trend:
            gap = (vwap - close) / max(vwap, 1) * 100
            conf = round(S2_CONF_TREND_BASE + min(S2_CONF_TREND_GAP_BONUS, gap / S2_CONF_TREND_GAP_DIVISOR), 4)
            return {"direction": Direction.BUY_PUT, "confidence": conf,
                    "name": self.name, "meta": {"vwap": round(vwap, 2), "mode": "trend_cont"}}
        return {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

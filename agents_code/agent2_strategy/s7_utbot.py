"""
S7: UT Bot  ATR Trailing Stop Crossover
==========================================
Based on the UT Bot Alerts Pine Script (strategies_v1.rtf).

What makes this DIFFERENT from S1 (SuperTrend+RSI):
    - SuperTrend uses ATR to build a band that flips direction
    - UT Bot builds a RATCHETING trailing stop:
         In an uptrend: stop can only move UP (locks in gains)
         In a downtrend: stop can only move DOWN (locks in shorts)
    - Signal fires when EMA(close,1) CROSSES the trailing stop
       this is a momentum exhaustion / reversal detection
    - No RSI filter  pure price action vs dynamic stop

Why it has an edge on NIFTY:
    - NIFTY options benefit from catching the EXACT moment trend
      resumes after a shallow pullback (the crossover moment)
    - The ratcheting stop is more adaptive than a fixed band during
      volatile intraday sessions (high VIX environments)

Signal Logic:
    BULLISH entry (BUY CALL):
        1. Compute ATR trailing stop (ratchets up in uptrend)
        2. EMA(close,1) crosses ABOVE the trailing stop
        3. Close > trailing stop (price above stop  bullish context)
        4. Volume > 1.3 20-bar average (momentum confirmation)
        5. Close > EMA(close, 50) (macro trend filter)
         BUY CALL

    BEARISH entry (BUY PUT):
        1. EMA(close,1) crosses BELOW the trailing stop
        2. Close < trailing stop (price below stop  bearish context)
        3. Volume confirmation
        4. Close < EMA(close, 50)
         BUY PUT

Confidence scoring:
    Base: 0.64
    +0.04 if crossover candle has strong close (body > 60% of range)
    +0.03 if volume > 1.6 average (strong momentum)
    +0.02 if ATR is expanding (rising volatility = stronger move)
    Cap: 0.81

Parameters:
    UT_ATR_PERIOD   = 10   ATR period (from original UT Bot)
    UT_KEY_VALUE    = 1.0  ATR multiplier sensitivity
    UT_EMA_TREND    = 50   EMA period for macro trend filter
    UT_VOL_MULT     = 1.3  Volume confirmation threshold

Key difference from S1 in the ensemble:
    S1 fires on SuperTrend BAND FLIP (regime change)
    S7 fires on TRAILING STOP CROSSOVER (momentum resumption)
    Together they catch complementary entry points  S1 at trend
    starts, S7 at pullback recoveries within existing trends.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from core.models import Direction
from typing import Union
from config.settings.strategy import (
    S7_ATR_PERIOD, S7_SENSITIVITY, S7_EMA_TREND_PERIOD, S7_VOL_MULT, S7_VOL_MA_PERIOD,
    S7_CONF_BASE, S7_CONF_BODY_PCT_BONUS_1_THRESHOLD, S7_CONF_BODY_PCT_BONUS_1,
    S7_CONF_BODY_PCT_BONUS_2_THRESHOLD, S7_CONF_BODY_PCT_BONUS_2,
    S7_CONF_VOL_BONUS_1_THRESHOLD, S7_CONF_VOL_BONUS_1, S7_CONF_VOL_BONUS_2_THRESHOLD,
    S7_CONF_VOL_BONUS_2, S7_CONF_ATR_BONUS_WINDOW, S7_CONF_ATR_BONUS_FACTOR,
    S7_CONF_ATR_BONUS, S7_CONF_MAX, S7_MIN_DF_OFFSET, S7_MIN_DF_LEN, S7_EPSILON
)


class UTBotStrategy:
    name = "UTBot"

    #  PUBLIC ENTRY POINT 

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None, cache=None) -> dict:
        """Standard interface  returns direction, confidence, name, meta."""
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        if len(df) < S7_ATR_PERIOD + S7_MIN_DF_OFFSET:
            return none

        close  = df["close"].values.astype(float)
        high   = df["high"].values.astype(float)
        low    = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        #  Compute ATR 
        atr_series = cache.atr_10.reindex(df.index) if cache and cache.atr_10 is not None else ta.atr(df["high"], df["low"], df["close"], S7_ATR_PERIOD)
        if atr_series is None or atr_series.dropna().empty:
            return none
        atr = atr_series.values.astype(float)

        #  Build ratcheting ATR trailing stop 
        trailing_stop = self._compute_trailing_stop(close, atr, S7_SENSITIVITY)
        if trailing_stop is None:
            return none

        #  EMA(close, 1) = close itself  use as src for crossover detection
        src = close   # EMA(close, 1) simplifies to close

        #  Detect crossover on last 2 candles 
        ts_now  = trailing_stop[-1]
        ts_prev = trailing_stop[-2]
        src_now  = src[-1]
        src_prev = src[-2]

        cross_above = src_prev <= ts_prev and src_now > ts_now   # bullish
        cross_below = src_prev >= ts_prev and src_now < ts_now   # bearish

        if not cross_above and not cross_below:
            return none

        #  Volume filter 
        if cache and cache.vol_ma_20 is not None:
            vol_ma = float(cache.vol_ma_20.reindex(df.index).iloc[-1])
        else:
            vol_ma = float(np.nanmean(volume[-S7_VOL_MA_PERIOD:]))
        if vol_ma <= 0:
            return none
        vol_ratio = float(volume[-1]) / vol_ma
        if vol_ratio < S7_VOL_MULT:
            return none

        #  Macro trend filter (EMA50) 
        ema50 = cache.ema_50.reindex(df.index) if cache and cache.ema_50 is not None else ta.ema(df["close"], S7_EMA_TREND_PERIOD)
        ema50_val = float(ema50.iloc[-1]) if ema50 is not None and not ema50.dropna().empty else src_now

        #  Build result 
        c_now = float(close[-1])
        o_now = float(df["open"].iloc[-1])
        h_now = float(high[-1])
        l_now = float(low[-1])
        candle_range = max(h_now - l_now, S7_EPSILON)
        body_pct     = abs(c_now - o_now) / candle_range

        if cross_above and c_now > ema50_val:
            conf = self._score(body_pct, vol_ratio, atr, "bull")
            return {
                "direction":  Direction.BUY_CALL,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "trailing_stop": round(ts_now, 2),
                    "vol_ratio":     round(vol_ratio, 2),
                    "body_pct":      round(body_pct, 2),
                    "atr":           round(float(atr[-1]), 2),
                },
            }

        if cross_below and c_now < ema50_val:
            conf = self._score(body_pct, vol_ratio, atr, "bear")
            return {
                "direction":  Direction.BUY_PUT,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "trailing_stop": round(ts_now, 2),
                    "vol_ratio":     round(vol_ratio, 2),
                    "body_pct":      round(body_pct, 2),
                    "atr":           round(float(atr[-1]), 2),
                },
            }

        return none

    #  TRAILING STOP COMPUTATION 

    @staticmethod
    def _compute_trailing_stop(
        close: np.ndarray,
        atr:   np.ndarray,
        key_value: float,
    ) -> Union[np.ndarray, None]:
        """
        Python translation of the UT Bot xATRTrailingStop logic.

        Pine Script logic:
            nLoss = key_value  ATR
            if close > prev_stop and prev_close > prev_stop:
                stop = max(prev_stop, close - nLoss)    # ratchets UP
            elif close < prev_stop and prev_close < prev_stop:
                stop = min(prev_stop, close + nLoss)    # ratchets DOWN
            else:
                stop = close - nLoss  if bullish  else  close + nLoss
        """
        n = len(close)
        if n < S7_MIN_DF_LEN:
            return None

        stop = np.zeros(n)
        stop[0] = close[0] - key_value * atr[0] if not np.isnan(atr[0]) else close[0]

        for i in range(1, n):
            if np.isnan(atr[i]):
                stop[i] = stop[i - 1]
                continue

            n_loss    = key_value * atr[i]
            prev_stop = stop[i - 1]
            c_now     = close[i]
            c_prev    = close[i - 1]

            if c_now > prev_stop and c_prev > prev_stop:
                # Uptrend  ratchet stop upward only
                stop[i] = max(prev_stop, c_now - n_loss)
            elif c_now < prev_stop and c_prev < prev_stop:
                # Downtrend  ratchet stop downward only
                stop[i] = min(prev_stop, c_now + n_loss)
            else:
                # Crossover candle  reset stop
                stop[i] = c_now - n_loss if c_now > prev_stop else c_now + n_loss

        return stop

    #  CONFIDENCE SCORING 

    @staticmethod
    def _score(
        body_pct:  float,
        vol_ratio: float,
        atr:       np.ndarray,
        side:      str,
    ) -> float:
        conf = S7_CONF_BASE

        # Bonus: strong conviction candle (body dominant)
        if body_pct >= S7_CONF_BODY_PCT_BONUS_1_THRESHOLD:
            conf += S7_CONF_BODY_PCT_BONUS_1
        elif body_pct >= S7_CONF_BODY_PCT_BONUS_2_THRESHOLD:
            conf += S7_CONF_BODY_PCT_BONUS_2

        # Bonus: strong volume
        if vol_ratio >= S7_CONF_VOL_BONUS_1_THRESHOLD:
            conf += S7_CONF_VOL_BONUS_1
        elif vol_ratio >= S7_CONF_VOL_BONUS_2_THRESHOLD:
            conf += S7_CONF_VOL_BONUS_2

        # Bonus: ATR expanding (rising momentum)
        if len(atr) >= S7_CONF_ATR_BONUS_WINDOW:
            atr_clean = atr[~np.isnan(atr)]
            if len(atr_clean) >= S7_CONF_ATR_BONUS_WINDOW:
                atr_now  = float(atr_clean[-1])
                atr_prev = float(np.mean(atr_clean[-S7_CONF_ATR_BONUS_WINDOW:-1]))
                if atr_now > atr_prev * S7_CONF_ATR_BONUS_FACTOR:
                    conf += S7_CONF_ATR_BONUS

        return round(min(S7_CONF_MAX, conf), 4)

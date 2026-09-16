"""
S8: CPR  Central Pivot Range
===============================
CPR (Central Pivot Range) is a widely used intraday tool in Indian
markets, especially by NIFTY options traders. It defines three key
levels calculated from the PREVIOUS day's OHLC data:

    TC  (Top Central)   = (Pivot - BC) + Pivot
    Pivot               = (Prev High + Prev Low + Prev Close) / 3
    BC  (Bottom Central)= (Prev High + Prev Low) / 2

These three levels form a "range" on the current day's chart.

NIFTY-specific CPR concepts used here:
    1. Narrow CPR  (gap < 0.1% of price) = trending day expected
        price will break decisively in one direction
    2. Wide CPR    (gap > 0.3% of price) = sideways/choppy expected
        avoid trading (lower confidence)
    3. Virgin CPR  = CPR that was NOT tested yesterday
        extremely powerful magnet for next day price action
    4. CPR Breakout= price opens ABOVE TC or BELOW BC and holds
        strong directional bias
    5. CPR Rejection= price tests TC/BC and reverses
        fade the test

Signal Logic:
    BUY CALL:
        1. Price is ABOVE TC (broke above the full CPR range)
        2. CPR is narrow (trending day likely)
        3. Price > previous day's pivot (above macro support)
        4. First test of TC from below already bounced up OR
           current candle is continuing above TC after breakout
        5. Volume on breakout candle > 1.4 20-bar avg
         BUY CALL

    BUY PUT:
        1. Price is BELOW BC (broke below the full CPR range)
        2. CPR is narrow
        3. Price < previous day's pivot
        4. Volume confirmation
         BUY PUT

    CPR Rejection BUY CALL (fade):
        1. Price tested TC from above, dipped into CPR zone
        2. Current candle closes BACK above TC
        3. Volume surge confirms rejection
         BUY CALL (mean-reversion back up)

Confidence scoring:
    Base: 0.63
    +0.05 if CPR is narrow (< 0.08% of price)  trending day
    +0.04 if virgin CPR (not tested previous session)
    +0.03 if price is well above/below CPR (not just touching)
    +0.02 if volume > 1.6 average
    Cap: 0.83

Why CPR is orthogonal to all S1-S7:
    All existing strategies use single-session indicators (RSI, MACD,
    BB, VWAP etc). CPR is the ONLY strategy here using inter-session
    (previous day) data. It captures institutional memory of key
    levels that no intraday indicator can replicate.

Parameters:
    CPR_NARROW_PCT   = 0.10  % of price  below this = narrow CPR
    CPR_WIDE_PCT     = 0.30  % of price  above this = wide (avoid)
    CPR_VOL_MULT     = 1.4   volume confirmation threshold
    CPR_BREAKOUT_BUF = 0.05  % buffer above/below CPR for breakout confirm
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Union, Optional, Dict, Tuple, List
from core.models import Direction
from config.settings.strategy import (
    S8_CPR_NARROW_PCT, S8_CPR_WIDE_PCT, S8_CPR_VOL_MULT, S8_CPR_BREAKOUT_BUF,
    S8_MIN_DF_LEN, S8_VOL_MA_PERIOD, S8_MIN_PREV_DAY_CANDLES, S8_CONF_BASE,
    S8_CONF_NARROW_CPR_BONUS_1_THRESHOLD, S8_CONF_NARROW_CPR_BONUS_1,
    S8_CONF_NARROW_CPR_BONUS_2, S8_CONF_DISTANCE_BONUS_1_THRESHOLD,
    S8_CONF_DISTANCE_BONUS_1, S8_CONF_DISTANCE_BONUS_2_THRESHOLD,
    S8_CONF_DISTANCE_BONUS_2, S8_CONF_VOL_BONUS_1_THRESHOLD, S8_CONF_VOL_BONUS_1,
    S8_CONF_VOL_BONUS_2, S8_CONF_REJECTION_PENALTY, S8_CONF_MAX
)


import os
import sqlite3
from datetime import datetime, date
from loguru import logger


@dataclass
class CPRLevels:
    """CPR levels for the current trading day."""
    pivot:  float    # (H + L + C) / 3
    tc:     float    # Top Central
    bc:     float    # Bottom Central
    width:  float    # TC - BC in points
    width_pct: float # width as % of pivot
    is_narrow: bool  # width_pct < CPR_NARROW_PCT
    is_wide:   bool  # width_pct > CPR_WIDE_PCT


class CPRStrategy:
    name = "CPR"

    def __init__(self, db_path: str = "data/historical/market_history.sqlite3"):
        self._daily_cache: dict[date, tuple[float, float, float]] = {}
        self._db_path = db_path
        self._initialized_db = False

    def _load_daily_cache(self) -> None:
        if self._initialized_db:
            return
        self._initialized_db = True
        try:
            if os.path.exists(self._db_path):
                with sqlite3.connect(self._db_path) as conn:
                    cur = conn.cursor()
                    comm = os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM")).upper()
                    cur.execute(
                        "SELECT ts, high, low, close FROM candles WHERE symbol IN (?, 'SILVERM', 'NIFTY') AND interval='day' ORDER BY ts ASC",
                        (comm,),
                    )
                    for ts_str, h, l, c in cur.fetchall():
                        try:
                            d = datetime.fromisoformat(str(ts_str)).date()
                            self._daily_cache[d] = (float(h), float(l), float(c))
                        except Exception:
                            continue
        except Exception as e:
            logger.debug(f"[CPR] Daily cache initialization skipped: {e}")

    # ── PUBLIC ENTRY POINT ──────────────────────────────────────

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None, prev_day_ohlc=None, **kwargs) -> dict:
        """Standard interface — returns direction, confidence, name, meta."""
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        if len(df) < S8_MIN_DF_LEN:
            return none

        # ── Get CPR levels from previous day OHLC ──
        cpr = self._compute_cpr(df, prev_day_ohlc=prev_day_ohlc)
        if cpr is None:
            return none

        # Wide CPR → choppy day likely → skip
        if cpr.is_wide:
            return none

        close    = float(df["close"].iloc[-1])
        vol      = float(df["volume"].iloc[-1])
        vol_ma   = float(df["volume"].rolling(S8_VOL_MA_PERIOD).mean().iloc[-1])

        if vol_ma <= 0:
            return none
        vol_ratio = vol / vol_ma

        # Volume gate
        if vol_ratio < S8_CPR_VOL_MULT:
            return none

        # Buffer in points for breakout confirmation
        buf = cpr.pivot * S8_CPR_BREAKOUT_BUF / 100

        result = (
            self._check_bullish(df, close, cpr, vol_ratio, buf)
            or self._check_bearish(df, close, cpr, vol_ratio, buf)
            or self._check_cpr_rejection(df, close, cpr, vol_ratio)
        )

        return result if result else none

    # ── CPR COMPUTATION ─────────────────────────────────────────

    def _compute_cpr(self, df: pd.DataFrame, prev_day_ohlc: Optional[dict] = None) -> Union[CPRLevels, None]:
        """
        Compute CPR levels strictly from the correct COMPLETED previous trading session.
        Never calculates CPR from a partial or truncated rolling intraday slice.
        """
        try:
            prev_high: Optional[float] = None
            prev_low: Optional[float] = None
            prev_close: Optional[float] = None

            # 1. Explicitly provided full-day OHLC
            if prev_day_ohlc and isinstance(prev_day_ohlc, dict):
                prev_high = float(prev_day_ohlc["high"])
                prev_low = float(prev_day_ohlc["low"])
                prev_close = float(prev_day_ohlc["close"])

            # 2. Independent complete daily database cache
            if prev_high is None:
                self._load_daily_cache()
                if self._daily_cache and len(df) > 0:
                    last_idx = df.index[-1]
                    current_date = last_idx.date() if hasattr(last_idx, "date") else last_idx
                    if isinstance(current_date, datetime):
                        current_date = current_date.date()
                    
                    prior_dates = [d for d in self._daily_cache if d < current_date]
                    if prior_dates:
                        prev_date = max(prior_dates)
                        prev_high, prev_low, prev_close = self._daily_cache[prev_date]

            # 3. Fallback to dataframe candles ONLY if it contains a verified COMPLETE prior session
            if prev_high is None:
                df = df.copy()
                df["_date"] = [
                    idx.date() if hasattr(idx, "date") else idx
                    for idx in df.index
                ]
                dates = sorted(df["_date"].unique())

                if len(dates) < 2:
                    return None

                yesterday = dates[-2]
                prev_day = df[df["_date"] == yesterday]

                # VALIDATION: Must be a complete trading session (>=40 5m bars),
                # starting in the morning (<=09:35) and ending at market close (>=15:00 for NSE / >=23:00 for MCX).
                # Never compute CPR on a truncated rolling window fragment!
                if len(prev_day) < 40:
                    return None
                
                first_ts = prev_day.index[0]
                last_ts = prev_day.index[-1]
                if hasattr(first_ts, "strftime") and first_ts.strftime("%H:%M") > "09:35":
                    return None
                if hasattr(last_ts, "strftime") and last_ts.strftime("%H:%M") < "15:00":
                    return None

                prev_high = float(prev_day["high"].max())
                prev_low = float(prev_day["low"].min())
                prev_close = float(prev_day["close"].iloc[-1])

            if prev_high is None or prev_low is None or prev_close is None:
                return None

            pivot = (prev_high + prev_low + prev_close) / 3
            bc    = (prev_high + prev_low) / 2
            tc    = (pivot - bc) + pivot

            # Ensure TC > BC
            if tc < bc:
                tc, bc = bc, tc

            width     = tc - bc
            width_pct = width / max(pivot, 1) * 100

            return CPRLevels(
                pivot      = round(pivot, 2),
                tc         = round(tc, 2),
                bc         = round(bc, 2),
                width      = round(width, 2),
                width_pct  = round(width_pct, 4),
                is_narrow  = width_pct < S8_CPR_NARROW_PCT,
                is_wide    = width_pct > S8_CPR_WIDE_PCT,
            )
        except Exception:
            return None

    #  BULLISH BREAKOUT 

    def _check_bullish(
        self,
        df:        pd.DataFrame,
        close:     float,
        cpr:       CPRLevels,
        vol_ratio: float,
        buf:       float,
    ) -> Union[dict, None]:
        """
        Price broke above TC with buffer and is holding.
        Additional filter: previous candle also above TC (sustained).
        """
        prev_close = float(df["close"].iloc[-2])

        # Both current and previous close above TC + buffer
        if close > (cpr.tc + buf) and prev_close > cpr.tc and close > cpr.pivot:
            conf = self._score(cpr, vol_ratio, close, cpr.tc, "above")
            return {
                "direction":  Direction.BUY_CALL,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "signal":    "above_tc",
                    "pivot":     cpr.pivot,
                    "tc":        cpr.tc,
                    "bc":        cpr.bc,
                    "cpr_width": cpr.width_pct,
                    "vol_ratio": round(vol_ratio, 2),
                },
            }
        return None

    #  BEARISH BREAKOUT 

    def _check_bearish(
        self,
        df:        pd.DataFrame,
        close:     float,
        cpr:       CPRLevels,
        vol_ratio: float,
        buf:       float,
    ) -> Union[dict, None]:
        """Price broke below BC with buffer and is holding."""
        prev_close = float(df["close"].iloc[-2])

        if close < (cpr.bc - buf) and prev_close < cpr.bc and close < cpr.pivot:
            conf = self._score(cpr, vol_ratio, close, cpr.bc, "below")
            return {
                "direction":  Direction.BUY_PUT,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "signal":    "below_bc",
                    "pivot":     cpr.pivot,
                    "tc":        cpr.tc,
                    "bc":        cpr.bc,
                    "cpr_width": cpr.width_pct,
                    "vol_ratio": round(vol_ratio, 2),
                },
            }
        return None

    #  CPR REJECTION (FADE) 

    def _check_cpr_rejection(
        self,
        df:        pd.DataFrame,
        close:     float,
        cpr:       CPRLevels,
        vol_ratio: float,
    ) -> Union[dict, None]:
        """
        Price tested TC/BC and was rejected  close re-establishes position.

        Bullish rejection: price dipped into CPR zone from above,
        current candle closes back above TC (CPR acted as support).

        Bearish rejection: price spiked into CPR zone from below,
        current candle closes back below BC (CPR acted as resistance).
        """
        if len(df) < 3:
            return None

        low_now   = float(df["low"].iloc[-1])
        high_now  = float(df["high"].iloc[-1])
        low_prev  = float(df["low"].iloc[-2])
        high_prev = float(df["high"].iloc[-2])

        # Bullish rejection: previous candle dipped into/below CPR,
        # current candle closes above TC
        prev_dipped_into_cpr = low_prev <= cpr.tc and high_prev >= cpr.bc
        if prev_dipped_into_cpr and close > cpr.tc and close > cpr.pivot:
            conf = self._score(cpr, vol_ratio, close, cpr.tc, "rejection_bull")
            return {
                "direction":  Direction.BUY_CALL,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "signal":    "cpr_rejection_bullish",
                    "pivot":     cpr.pivot,
                    "tc":        cpr.tc,
                    "vol_ratio": round(vol_ratio, 2),
                },
            }

        # Bearish rejection: previous candle spiked into/above CPR,
        # current candle closes below BC
        prev_spiked_into_cpr = high_prev >= cpr.bc and low_prev <= cpr.tc
        if prev_spiked_into_cpr and close < cpr.bc and close < cpr.pivot:
            conf = self._score(cpr, vol_ratio, close, cpr.bc, "rejection_bear")
            return {
                "direction":  Direction.BUY_PUT,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "signal":    "cpr_rejection_bearish",
                    "pivot":     cpr.pivot,
                    "bc":        cpr.bc,
                    "vol_ratio": round(vol_ratio, 2),
                },
            }

        return None

    #  CONFIDENCE SCORING 

    @staticmethod
    def _score(
        cpr:       CPRLevels,
        vol_ratio: float,
        close:     float,
        ref_level: float,
        signal:    str,
    ) -> float:
        conf = S8_CONF_BASE

        # Narrow CPR = trending day  higher probability
        if cpr.width_pct < S8_CONF_NARROW_CPR_BONUS_1_THRESHOLD:
            conf += S8_CONF_NARROW_CPR_BONUS_1
        elif cpr.width_pct < S8_CPR_NARROW_PCT:
            conf += S8_CONF_NARROW_CPR_BONUS_2

        # Price well outside CPR (not just touching edge)
        distance_pct = abs(close - ref_level) / max(ref_level, 1) * 100
        if distance_pct > S8_CONF_DISTANCE_BONUS_1_THRESHOLD:
            conf += S8_CONF_DISTANCE_BONUS_1
        elif distance_pct > S8_CONF_DISTANCE_BONUS_2_THRESHOLD:
            conf += S8_CONF_DISTANCE_BONUS_2

        # Volume confirmation
        if vol_ratio >= S8_CONF_VOL_BONUS_1_THRESHOLD:
            conf += S8_CONF_VOL_BONUS_1
        elif vol_ratio >= S8_CPR_VOL_MULT:
            conf += S8_CONF_VOL_BONUS_2

        # Rejection signals have slightly lower base (less reliable)
        if "rejection" in signal:
            conf -= S8_CONF_REJECTION_PENALTY

        return round(min(S8_CONF_MAX, conf), 4)
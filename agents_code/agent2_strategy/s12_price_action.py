"""
S12: Price Action Candle Patterns  Key Level Reactions
========================================================
Pure price action strategy using high-probability candlestick
patterns AT meaningful levels, with volume confirmation.

Trading candle patterns in isolation has poor win rate (~40%).
Trading them AT key levels with volume confirmation is different.
This strategy ONLY fires when a strong candle pattern occurs at:
    - A previous support/resistance zone
    - Near VWAP (intraday fair value)
    - Near EMA (dynamic support/resistance)

Patterns implemented:
    1. BULLISH ENGULFING at support
       Current candle body completely engulfs previous candle body
       + closes higher + occurs at support zone
        Strong reversal signal

    2. BEARISH ENGULFING at resistance
       Current body engulfs previous body
       + closes lower + occurs at resistance zone

    3. PIN BAR / HAMMER at support (bullish)
       Long lower wick  2 body + small upper wick
       + closes in upper half of range
       + occurs at support

    4. SHOOTING STAR / INVERTED HAMMER at resistance (bearish)
       Long upper wick  2 body + small lower wick
       + closes in lower half of range
       + occurs at resistance

    5. INSIDE BAR BREAKOUT
       Current candle breaks above/below a preceding inside bar
       (inside bar = candle completely within previous candle's range)
       + volume surge confirms direction

    6. MORNING STAR VARIANT (3-candle bullish)
       Bearish candle  small doji/indecision  bullish engulf
       At support level

    7. EVENING STAR VARIANT (3-candle bearish)
       Bullish candle  doji  bearish engulf
       At resistance

Level detection (key zones for pattern confirmation):
    - EMA20  0.15%: dynamic support/resistance band
    - Swing high/low from last 20 candles
    - VWAP  0.10%: intraday fair value band
    - Round numbers: levels divisible by 50 within 0.2% of price

Confidence scoring:
    Base by pattern strength:
        Engulfing:      0.65
        Pin Bar:        0.64
        Inside Bar:     0.63
        3-candle:       0.67
    + 0.05 if AT key level (not just near it)
    + 0.04 if volume > 2 average (institutional confirmation)
    + 0.03 if pattern aligns with EMA trend direction
    + 0.02 if round number confluence
    Cap: 0.82

Parameters:
    PA_EMA_PERIOD    = 20    EMA for trend and level detection
    PA_VWAP_BAND     = 0.10  % band around VWAP for "at VWAP" test
    PA_LEVEL_BAND    = 0.20  % band around any level for "at level" test
    PA_VOL_MULT      = 1.4   volume needed for pattern confirmation
    PA_SWING_LOOKBACK= 20    bars for swing high/low detection
    PA_ENGULF_FACTOR = 1.02  engulfing candle must be this much bigger
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from dataclasses import dataclass
from typing import Union
from core.models import Direction
from config.settings.strategy import (
    S12_EMA_PERIOD, S12_VWAP_BAND, S12_LEVEL_BAND, S12_VOL_MULT,
    S12_SWING_LOOKBACK, S12_ENGULF_FACTOR, S12_VOL_MA_PERIOD,
    S12_CONF_LEVEL_BONUS, S12_CONF_VOL_STRONG_THRESHOLD,
    S12_CONF_VOL_STRONG_BONUS, S12_CONF_VOL_MIN_BONUS,
    S12_CONF_EMA_ALIGN_BONUS, S12_CONF_ROUND_NUM_BONUS,
    S12_CONF_MAX, S12_ROUND_NUM_THRESHOLD, S12_PATTERN_ENGULF_CONF,
    S12_PATTERN_PINBAR_CONF, S12_PATTERN_INSIDEBAR_CONF, S12_PATTERN_3CANDLE_CONF,
    S12_SWING_LEVEL_DIFF_PCT, S12_RANGE_MIN_DIFF, S12_INSIDE_BAR_LOOKBACK,
    S12_PINBAR_BODY_RATIO, S12_PINBAR_WICK_RATIO
)





@dataclass
class PatternResult:
    pattern:   str
    direction: Direction
    base_conf: float
    at_level:  bool
    level_name: str


class PriceActionStrategy:
    name = "PriceAction"

    #  PUBLIC ENTRY POINT 

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        if len(df) < S12_SWING_LOOKBACK + 10:
            return none

        #  Compute key levels 
        close    = float(df["close"].iloc[-1])
        vol      = float(df["volume"].iloc[-1])
        vol_ma   = float(df["volume"].rolling(S12_VOL_MA_PERIOD).mean().iloc[-1])
        if vol_ma <= 0:
            return none
        vol_ratio = vol / vol_ma

        if vol_ratio < S12_VOL_MULT:
            return none

        ema20  = ta.ema(df["close"], S12_EMA_PERIOD)
        ema_val = float(ema20.iloc[-1]) if ema20 is not None and not ema20.dropna().empty else close
        vwap   = self._compute_vwap(df)
        swing_hi, swing_lo = self._find_swings(df)
        levels = self._collect_levels(ema_val, vwap, swing_hi, swing_lo, close)

        #  Detect patterns 
        patterns = []

        r = self._engulfing(df)
        if r: patterns.append(r)

        r = self._pin_bar(df)
        if r: patterns.append(r)

        r = self._inside_bar_breakout(df)
        if r: patterns.append(r)

        r = self._three_candle(df)
        if r: patterns.append(r)

        if not patterns:
            return none

        # Pick strongest pattern (highest base confidence)
        best = max(patterns, key=lambda p: p.base_conf)

        # Check level confluence
        at_level, level_name = self._near_directional_level(close, levels, best.direction)

        # EMA trend alignment
        ema_aligned = (
            (best.direction == Direction.BUY_CALL and close > ema_val) or
            (best.direction == Direction.BUY_PUT  and close < ema_val)
        )

        # Round number confluence (dynamically scaled for commodities: e.g. 500 for Silver/Gold, 50 for Crude, 5 for NatGas)
        step = 500 if close > 20000 else (50 if close > 1000 else 5)
        base_r = round(close / step) * step
        near_round = abs(close - base_r) / max(base_r, 1) * 100 < S12_ROUND_NUM_THRESHOLD

        if not at_level:
            return none
        if best.pattern in {"hammer", "shooting_star", "bullish_engulf", "bearish_engulf"} and not ema_aligned:
            return none

        conf = self._score(
            best, at_level, vol_ratio, ema_aligned, near_round
        )

        return {
            "direction":  best.direction,
            "confidence": conf,
            "name":       self.name,
            "meta": {
                "pattern":     best.pattern,
                "at_level":    level_name if at_level else "none",
                "ema":         round(ema_val, 2),
                "vwap":        round(vwap, 2) if vwap else None,
                "vol_ratio":   round(vol_ratio, 2),
                "ema_aligned": ema_aligned,
                "near_round":  near_round,
            },
        }

    #  PATTERN DETECTORS 

    def _engulfing(self, df: pd.DataFrame) -> Union[PatternResult, None]:
        """Bullish or bearish engulfing candle."""
        c0 = df.iloc[-2]   # previous candle
        c1 = df.iloc[-1]   # current candle

        o0, c0c = float(c0["open"]), float(c0["close"])
        o1, c1c = float(c1["open"]), float(c1["close"])

        prev_body = abs(c0c - o0)
        curr_body = abs(c1c - o1)

        if prev_body < 0.01:
            return None

        # Bullish engulfing
        if (c0c < o0                         # prev was bearish
                and c1c > o1                 # curr is bullish
                and o1 <= c0c               # curr opens at or below prev close
                and c1c >= o0 * S12_ENGULF_FACTOR  # curr close above prev open
                and curr_body > prev_body):
            return PatternResult(
                pattern="bullish_engulf",
                direction=Direction.BUY_CALL,
                base_conf=S12_PATTERN_ENGULF_CONF,
                at_level=False, level_name="",
            )

        # Bearish engulfing
        if (c0c > o0                          # prev was bullish
                and c1c < o1                  # curr is bearish
                and o1 >= c0c               # curr opens at or above prev close
                and c1c <= o0 / S12_ENGULF_FACTOR  # curr close below prev open
                and curr_body > prev_body):
            return PatternResult(
                pattern="bearish_engulf",
                direction=Direction.BUY_PUT,
                base_conf=S12_PATTERN_ENGULF_CONF,
                at_level=False, level_name="",
            )

        return None

    def _pin_bar(self, df: pd.DataFrame) -> Union[PatternResult, None]:
        """Hammer (bullish) or shooting star (bearish) pin bar."""
        c = df.iloc[-1]
        o = float(c["open"])
        h = float(c["high"])
        l = float(c["low"])
        cl = float(c["close"])

        total_range = max(h - l, S12_RANGE_MIN_DIFF)
        body        = abs(cl - o)
        upper_wick  = h - max(o, cl)
        lower_wick  = min(o, cl) - l

        # Hammer (bullish pin bar)
        if (lower_wick >= 2 * body           # long lower wick
                and upper_wick <= body * S12_PINBAR_WICK_RATIO  # small upper wick
                and cl > (h + l) / 2          # closes in upper half
                and body / total_range < S12_PINBAR_BODY_RATIO): # small body relative to range
            return PatternResult(
                pattern="hammer",
                direction=Direction.BUY_CALL,
                base_conf=S12_PATTERN_PINBAR_CONF,
                at_level=False, level_name="",
            )

        # Shooting star (bearish pin bar)
        if (upper_wick >= 2 * body
                and lower_wick <= body * S12_PINBAR_WICK_RATIO
                and cl < (h + l) / 2
                and body / total_range < S12_PINBAR_BODY_RATIO):
            return PatternResult(
                pattern="shooting_star",
                direction=Direction.BUY_PUT,
                base_conf=S12_PATTERN_PINBAR_CONF,
                at_level=False, level_name="",
            )

        return None

    def _inside_bar_breakout(self, df: pd.DataFrame) -> Union[PatternResult, None]:
        """Inside bar followed by breakout with volume."""
        if len(df) < S12_INSIDE_BAR_LOOKBACK:
            return None

        mother = df.iloc[-3]   # mother candle
        inside = df.iloc[-2]   # inside bar
        curr   = df.iloc[-1]   # breakout candle

        m_h = float(mother["high"])
        m_l = float(mother["low"])
        i_h = float(inside["high"])
        i_l = float(inside["low"])
        c_h = float(curr["high"])
        c_l = float(curr["low"])
        c_cl = float(curr["close"])

        # Confirm inside bar
        if not (i_h <= m_h and i_l >= m_l):
            return None

        # Bullish breakout above mother high
        if c_h > m_h and c_cl > m_h:
            return PatternResult(
                pattern="inside_bar_bull",
                direction=Direction.BUY_CALL,
                base_conf=S12_PATTERN_INSIDEBAR_CONF,
                at_level=False, level_name="",
            )

        # Bearish breakout below mother low
        if c_l < m_l and c_cl < m_l:
            return PatternResult(
                pattern="inside_bar_bear",
                direction=Direction.BUY_PUT,
                base_conf=S12_PATTERN_INSIDEBAR_CONF,
                at_level=False, level_name="",
            )

        return None

    def _three_candle(self, df: pd.DataFrame) -> Union[PatternResult, None]:
        """Morning star (bullish) or evening star (bearish) 3-candle pattern."""
        if len(df) < 3:
            return None

        c0 = df.iloc[-3]
        c1 = df.iloc[-2]   # middle (doji/indecision)
        c2 = df.iloc[-1]

        o0, cl0 = float(c0["open"]), float(c0["close"])
        o1, cl1 = float(c1["open"]), float(c1["close"])
        o2, cl2 = float(c2["open"]), float(c2["close"])

        body0 = abs(cl0 - o0)
        body1 = abs(cl1 - o1)
        body2 = abs(cl2 - o2)

        # Middle candle is small (indecision)
        if body1 > body0 * 0.4:
            return None

        # Morning star (bullish)
        if (cl0 < o0                    # first candle bearish
                and cl2 > o2            # third candle bullish
                and body2 > body0 * 0.5  # third body meaningful
                and cl2 > (o0 + cl0) / 2):  # third closes above midpoint of first
            return PatternResult(
                pattern="morning_star",
                direction=Direction.BUY_CALL,
                base_conf=S12_PATTERN_3CANDLE_CONF,
                at_level=False, level_name="",
            )

        # Evening star (bearish)
        if (cl0 > o0
                and cl2 < o2
                and body2 > body0 * 0.5
                and cl2 < (o0 + cl0) / 2):
            return PatternResult(
                pattern="evening_star",
                direction=Direction.BUY_PUT,
                base_conf=S12_PATTERN_3CANDLE_CONF,
                at_level=False, level_name="",
            )

        return None

    #  LEVEL DETECTION 

    def _compute_vwap(self, df: pd.DataFrame) -> float:
        """Intraday VWAP from today's candles."""
        try:
            today = [idx.date() for idx in df.index][-1]
            today_df = df[[idx.date() == today for idx in df.index]]
            if len(today_df) < 2:
                return 0.0
            tp  = (today_df["high"] + today_df["low"] + today_df["close"]) / 3
            cum_tpv = (tp * today_df["volume"]).cumsum()
            cum_vol = today_df["volume"].cumsum()
            return float((cum_tpv / cum_vol).iloc[-1])
        except Exception:
            return 0.0

    def _find_swings(self, df: pd.DataFrame) -> tuple[float, float]:
        """Recent swing high and swing low."""
        window = df.tail(S12_SWING_LOOKBACK)
        return float(window["high"].max()), float(window["low"].min())

    def _collect_levels(
        self,
        ema: float, vwap: float,
        swing_hi: float, swing_lo: float,
        close: float,
    ) -> list[tuple[float, str]]:
        """All meaningful levels with names."""
        levels = [(ema, f"EMA{S12_EMA_PERIOD}")]
        if vwap > 0:
            levels.append((vwap, "VWAP"))
        if abs(swing_hi - close) / max(close, 1) * 100 < S12_SWING_LEVEL_DIFF_PCT:
            levels.append((swing_hi, "SwingHigh"))
        if abs(swing_lo - close) / max(close, 1) * 100 < S12_SWING_LEVEL_DIFF_PCT:
            levels.append((swing_lo, "SwingLow"))
        return levels

    def _near_any_level(
        self,
        price:  float,
        levels: list[tuple[float, str]],
    ) -> tuple[bool, str]:
        """Check if price is within PA_LEVEL_BAND% of any key level."""
        for level, name in levels:
            if level <= 0:
                continue
            pct = abs(price - level) / level * 100
            if name == "VWAP":
                if pct <= S12_VWAP_BAND:
                    return True, name
            elif pct <= S12_LEVEL_BAND:
                return True, name
        return False, ""

    def _near_directional_level(
        self,
        price: float,
        levels: list[tuple[float, str]],
        direction: Direction,
    ) -> tuple[bool, str]:
        support_names = {"VWAP", f"EMA{S12_EMA_PERIOD}", "SwingLow"}
        resistance_names = {"VWAP", f"EMA{S12_EMA_PERIOD}", "SwingHigh"}
        allowed = support_names if direction == Direction.BUY_CALL else resistance_names
        filtered = [(level, name) for level, name in levels if name in allowed]
        return self._near_any_level(price, filtered)

    #  CONFIDENCE SCORING 

    @staticmethod
    def _score(
        pattern:    PatternResult,
        at_level:   bool,
        vol_ratio:  float,
        ema_align:  bool,
        near_round: bool,
    ) -> float:
        conf = pattern.base_conf

        # Level confluence is the most important bonus
        if at_level:
            conf += S12_CONF_LEVEL_BONUS

        # Volume confirmation
        if vol_ratio >= S12_CONF_VOL_STRONG_THRESHOLD:
            conf += S12_CONF_VOL_STRONG_BONUS
        elif vol_ratio >= S12_VOL_MULT:
            conf += S12_CONF_VOL_MIN_BONUS

        # EMA trend aligned with pattern
        if ema_align:
            conf += S12_CONF_EMA_ALIGN_BONUS

        # Round number confluence
        if near_round:
            conf += S12_CONF_ROUND_NUM_BONUS

        return round(min(S12_CONF_MAX, conf), 4)

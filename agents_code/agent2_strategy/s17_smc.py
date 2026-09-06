"""
S17: Smart Money Concepts (SMC)  Full Suite
============================================
Implements the four missing SMC pillars as ONE strategy module.
All four concepts share the same swing detection and structure
tracking infrastructure, so they are most efficient together.

Already built in SignalForge:
  [OK] S6   Fair Value Gap (FVG)
  [OK] S11  Liquidity Sweep
  [OK] S15  AMD (Accumulation / Manipulation / Distribution)

Built here (S17):
  [OK] Order Block (OB)         primary SMC entry concept
  [OK] BOS + ChoCH              structure shift confirmation
  [OK] Breaker Block            failed OB acting as opposite zone
  [OK] Premium / Discount       filter: only buy in discount, sell in premium


CONCEPT 1: ORDER BLOCK (OB)

An Order Block is the LAST candle in the OPPOSITE direction
before a strong impulsive move.

  Bullish OB:
    - Last BEARISH candle before a strong bullish impulse move
    - Institutions placed BUY orders on that candle (absorbing sells)
    - When price returns to the OB zone  institutions defend it  BUY CALL

  Bearish OB:
    - Last BULLISH candle before a strong bearish impulse move
    - When price returns to OB zone  BUY PUT

  Impulse validation:
    - Move after OB must be  OB_IMPULSE_ATR_MULT  ATR(14)
    - This filters weak OBs that just happen to be before any move

  Entry condition:
    - Price must RETRACE into the OB zone (not just near it)
    - A reaction candle (close) must come from within the OB
    - Volume on reaction  1.2 average (institutions defending)

  Example (Bullish OB):
    Candle[i]:  open=22100, high=22150, low=22080, close=22090   BEARISH OB
    Candle[i+1..i+3]: strong up move +180 pts                    impulse
    Later: price returns to 2208022150 zone
    Entry: close above OB midpoint on retrace  BUY CALL


CONCEPT 2: BREAK OF STRUCTURE (BOS) + CHANGE OF CHARACTER (ChoCH)

Market structure is defined by swing highs and lows.

  In a DOWNTREND: price makes Lower Highs (LH) and Lower Lows (LL)
  In an UPTREND:  price makes Higher Highs (HH) and Higher Lows (HL)

  BOS (Break of Structure):
    - Price breaks ABOVE the most recent LH in a downtrend  Bullish BOS
    - Price breaks BELOW the most recent HL in an uptrend  Bearish BOS
    - Confirms the trend change has STARTED

  ChoCH (Change of Character):
    - The FIRST BOS after a trend  the earliest sign of reversal
    - Less confirmation than BOS but gives EARLIER entry
    - Higher risk, higher reward than BOS

  Signal logic:
    - Track last 3 swing highs and lows
    - Identify current structure (HH/HL = bullish, LH/LL = bearish)
    - On close above last LH  bullish BOS  BUY CALL
    - On close below last HL  bearish BOS  BUY PUT
    - ChoCH: fire when FIRST structure break occurs after opposite trend


CONCEPT 3: BREAKER BLOCK

A Breaker Block is an Order Block that FAILED  price broke through it.

  When price breaks through an OB, the OB becomes a Breaker Block:
    - Previous Bullish OB (support)  price breaks BELOW  Bearish Breaker
      When price returns to the broken OB from below  SELL (BUY PUT)
    - Previous Bearish OB (resistance)  price breaks ABOVE  Bullish Breaker
      When price returns to the broken OB from above  BUY (BUY CALL)

  Why Breakers have higher win rate than regular OBs:
    - The break through the OB confirms institutional commitment
    - The return to the Breaker is a re-test of the break level
    - Retail expects support/resistance; institutions use the re-test
      to fill in the direction of the break


CONCEPT 4: PREMIUM / DISCOUNT ZONES

ICT Premium/Discount framework divides any price range into:

  PREMIUM zone: top 50% of the range (above equilibrium)
     Institutions SELL here (expensive)
     Only take SHORT entries (BUY PUT) in premium

  DISCOUNT zone: bottom 50% of the range (below equilibrium)
     Institutions BUY here (cheap)
     Only take LONG entries (BUY CALL) in discount

  Range definition: swing high to swing low in last OB_LOOKBACK candles.

  This is used as a FILTER for OB and Breaker entries:
    - Bullish OB in discount zone  HIGH confidence
    - Bullish OB in premium zone  filtered OUT (counterproductive)
    - Bearish OB in premium zone  HIGH confidence
    - Bearish OB in discount zone  filtered OUT


SIGNAL HIERARCHY (how the four concepts combine)


  Most signals require 2 of these 4 concepts to align:

  HIGHEST confidence (conf 0.80-0.87):
    OB + in Discount/Premium zone + BOS confirms

  HIGH confidence (conf 0.74-0.80):
    OB + BOS confirms
    Breaker + BOS confirms
    OB + in Discount/Premium zone

  MEDIUM confidence (conf 0.68-0.74):
    OB alone (in correct zone)
    Breaker alone
    ChoCH alone (early signal)

  This module fires the HIGHEST confidence scenario it can identify.
  The runner voting system then combines this with other strategies.


INTEGRATION

  Registered as S17 in STRATEGY_REGISTRY (runner.py)
  min_candles = 50 (needs enough history for swing detection + OB)
  Category: "structure" in signal_scorer (high independence weight)
  Returns standard dict: {direction, confidence, name, meta}
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from config.settings.strategy import (
    BOS_CONFIRM_CLOSE,
    BOS_LOOKBACK,
    BREAKER_MAX_AGE,
    OB_IMPULSE_ATR_MULT,
    OB_LOOKBACK,
    OB_MIN_BODY_PCT,
    OB_RETRACE_TOLERANCE,
    OB_VOL_MULT,
    SMC_ATR_FALLBACK,
    SMC_ATR_PERIOD,
    SMC_BOS_CONFIDENCE,
    SMC_CHOCH_CONFIDENCE,
    SMC_CHOCH_SCORE_BONUS,
    SMC_CONFIDENCE_ROUND_DIGITS,
    SMC_DEFAULT_SCORE,
    SMC_DEFAULT_VOLUME_RATIO,
    SMC_FRESH_OB_BARS_TIER_1,
    SMC_FRESH_OB_BARS_TIER_2,
    SMC_FRESH_OB_BONUS_TIER_1,
    SMC_FRESH_OB_BONUS_TIER_2,
    SMC_LOCATION_ROUND_DIGITS,
    SMC_MAX_CONFIDENCE,
    SMC_MAX_ORDER_BLOCKS,
    SMC_MAX_SWINGS_TRACKED,
    SMC_META_ROUND_DIGITS,
    SMC_MIN_CANDLE_RANGE,
    SMC_MIN_STRUCTURE_SWINGS,
    SMC_MIN_SWINGS,
    SMC_ORDER_BLOCK_IMPULSE_LOOKAHEAD_END,
    SMC_ORDER_BLOCK_SCAN_END_OFFSET,
    SMC_ORDER_BLOCK_SCAN_START,
    SMC_PREMIUM_DISCOUNT_NEUTRAL,
    SMC_PREMIUM_DISCOUNT_SWING_WINDOW,
    SMC_SCORE_BREAKER,
    SMC_SCORE_BREAKER_BOS,
    SMC_SCORE_OB_BOS,
    SMC_SCORE_OB_BOS_PD,
    SMC_SCORE_OB_PD,
    SMC_VOLUME_AVG_WINDOW,
    SMC_VOLUME_BONUS_HIGH,
    SMC_VOLUME_BONUS_LOW,
    SMC_VOLUME_BONUS_THRESHOLD_HIGH,
    SMC_VOLUME_BONUS_THRESHOLD_LOW,
    SWING_BARS,
)
from core.models import Direction


# 
# DATA CLASSES
# 

@dataclass
class OrderBlock:
    kind:       str    # "bullish" or "bearish"
    top:        float  # high of OB candle
    bottom:     float  # low of OB candle
    open_:      float  # open of OB candle
    close_:     float  # close of OB candle
    midpoint:   float  # (top + bottom) / 2
    bar_idx:    int    # position in df (absolute index from end, 0=current)
    impulse_sz: float  # size of the impulse move that validated the OB
    broken:     bool   # True if price later broke THROUGH the OB ( Breaker)


@dataclass
class SwingPoint:
    kind:    str    # "high" or "low"
    price:   float
    bar_idx: int    # from end of df (0 = most recent)


@dataclass
class MarketStructure:
    trend:        str          # "bullish", "bearish", "ranging"
    last_hh:      float | None # most recent Higher High
    last_hl:      float | None # most recent Higher Low
    last_lh:      float | None # most recent Lower High
    last_ll:      float | None # most recent Lower Low
    swings:       list[SwingPoint]
    bos_level:    float | None # level a BOS would confirm at
    choch_level:  float | None # level a ChoCH would fire at


@dataclass
class PremiumDiscount:
    range_high:   float
    range_low:    float
    equilibrium:  float        # midpoint
    zone:         str          # "premium", "discount", "equilibrium"
    location_pct: float        # current price position 0=bottom 1=top


# 
# MAIN STRATEGY
# 

class SMCStrategy:
    """
    S17: Full SMC Suite  Order Block, BOS/ChoCH, Breaker Block,
    Premium/Discount zones.
    Single entry point: evaluate()  standard runner interface.
    """
    name = "SMC"

    def evaluate(
        self,
        df:       pd.DataFrame,
        orb_high: float | None = None,
        orb_low:  float | None = None,
        **kwargs,
    ) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        if df is None or len(df) < OB_LOOKBACK:
            return none

        try:
            return self._evaluate(df)
        except Exception:
            return none

    #  INTERNAL PIPELINE 

    def _evaluate(self, df: pd.DataFrame) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        close = float(df["close"].iloc[-1])
        atr   = self._atr(df)

        #  Step 1: Swing detection 
        swings = self._find_swings(df)
        if len(swings) < SMC_MIN_SWINGS:
            return none

        #  Step 2: Market structure (BOS / ChoCH) 
        structure = self._classify_structure(df, swings)

        #  Step 3: Premium / Discount zone 
        pd_zone = self._premium_discount(df, swings)

        #  Step 4: Order Block scan 
        obs = self._find_order_blocks(df, atr)

        #  Step 5: Breaker Block scan 
        breakers = self._find_breakers(obs, df)

        #  Step 6: Build best signal (highest confidence wins) 
        return self._build_signal(close, structure, pd_zone, obs, breakers, df)

    # 
    # SWING DETECTION
    # 

    def _find_swings(self, df: pd.DataFrame) -> list[SwingPoint]:
        """
        Detect swing highs and lows using SWING_BARS pivot logic.
        Returns list ordered most-recent first (bar_idx=0 is current).
        """
        n      = SWING_BARS
        highs  = df["high"].values
        lows   = df["low"].values
        length = len(df)
        swings = []

        # Scan from recent to old (skip last n bars  not yet confirmed)
        for i in range(length - n - 1, n - 1, -1):
            bar_from_end = length - 1 - i

            # Swing high: highest among n bars each side
            is_sh = (
                all(highs[i] >= highs[i - j] for j in range(1, n + 1)) and
                all(highs[i] >= highs[i + j] for j in range(1, n + 1))
            )
            if is_sh:
                swings.append(SwingPoint("high", float(highs[i]), bar_from_end))

            # Swing low
            is_sl = (
                all(lows[i] <= lows[i - j] for j in range(1, n + 1)) and
                all(lows[i] <= lows[i + j] for j in range(1, n + 1))
            )
            if is_sl:
                swings.append(SwingPoint("low", float(lows[i]), bar_from_end))

        # Sort by recency
        return sorted(swings, key=lambda s: s.bar_idx)[:SMC_MAX_SWINGS_TRACKED]

    # 
    # MARKET STRUCTURE  BOS / ChoCH
    # 

    def _classify_structure(
        self, df: pd.DataFrame, swings: list[SwingPoint]
    ) -> MarketStructure:
        """
        Classify market structure as bullish / bearish / ranging.
        Identify the BOS and ChoCH trigger levels.
        """
        highs = [s for s in swings if s.kind == "high"]
        lows  = [s for s in swings if s.kind == "low"]

        if len(highs) < SMC_MIN_STRUCTURE_SWINGS or len(lows) < SMC_MIN_STRUCTURE_SWINGS:
            return MarketStructure("ranging", None, None, None, None, swings, None, None)

        # Swings are sorted most-recent first, so index 0 is newer than index 1.
        hh = highs[0].price > highs[1].price
        hl = lows[0].price  > lows[1].price
        lh = highs[0].price < highs[1].price
        ll = lows[0].price  < lows[1].price

        if hh and hl:
            # Bullish structure: Higher Highs + Higher Lows
            bos_level   = highs[0].price   # break above  bullish BOS continuation
            choch_level = lows[0].price    # break below  ChoCH (bearish reversal)
            return MarketStructure(
                trend       = "bullish",
                last_hh     = highs[0].price,
                last_hl     = lows[0].price,
                last_lh     = None,
                last_ll     = None,
                swings      = swings,
                bos_level   = bos_level,
                choch_level = choch_level,
            )

        if lh and ll:
            # Bearish structure: Lower Highs + Lower Lows
            bos_level   = lows[0].price    # break below  bearish BOS continuation
            choch_level = highs[0].price   # break above  ChoCH (bullish reversal)
            return MarketStructure(
                trend       = "bearish",
                last_hh     = None,
                last_hl     = None,
                last_lh     = highs[0].price,
                last_ll     = lows[0].price,
                swings      = swings,
                bos_level   = bos_level,
                choch_level = choch_level,
            )

        return MarketStructure("ranging", None, None, None, None, swings, None, None)

    def _check_bos(
        self, df: pd.DataFrame, structure: MarketStructure
    ) -> tuple[str, str] | None:
        """
        Returns (signal_direction, bos_type) if a BOS or ChoCH is active.
        signal_direction: "BUY_CALL" or "BUY_PUT"
        bos_type: "BOS" or "ChoCH"
        """
        close = float(df["close"].iloc[-1])
        prev  = float(df["close"].iloc[-2])

        if structure.trend == "bearish" and structure.choch_level:
            # ChoCH: bearish structure  price breaks above last LH
            if close > structure.choch_level and prev <= structure.choch_level:
                return ("BUY_CALL", "ChoCH")

        if structure.trend == "bullish" and structure.choch_level:
            # ChoCH: bullish structure  price breaks below last HL
            if close < structure.choch_level and prev >= structure.choch_level:
                return ("BUY_PUT", "ChoCH")

        if structure.trend == "bearish" and structure.bos_level:
            # BOS continuation: bearish  price breaks below last LL
            if close < structure.bos_level and prev >= structure.bos_level:
                return ("BUY_PUT", "BOS")

        if structure.trend == "bullish" and structure.bos_level:
            # BOS continuation: bullish  price breaks above last HH
            if close > structure.bos_level and prev <= structure.bos_level:
                return ("BUY_CALL", "BOS")

        return None

    # 
    # PREMIUM / DISCOUNT
    # 

    def _premium_discount(
        self, df: pd.DataFrame, swings: list[SwingPoint]
    ) -> PremiumDiscount:
        """
        Compute Premium/Discount zone from the most recent
        swing high to swing low range.
        """
        highs = [s for s in swings if s.kind == "high"]
        lows  = [s for s in swings if s.kind == "low"]

        if not highs or not lows:
            price = float(df["close"].iloc[-1])
            return PremiumDiscount(price, price, price, "equilibrium", SMC_PREMIUM_DISCOUNT_NEUTRAL)

        range_high = max(s.price for s in highs[:SMC_PREMIUM_DISCOUNT_SWING_WINDOW])
        range_low  = min(s.price for s in lows[:SMC_PREMIUM_DISCOUNT_SWING_WINDOW])
        equilib    = (range_high + range_low) / 2
        close      = float(df["close"].iloc[-1])

        rng = range_high - range_low
        loc = (close - range_low) / rng if rng > 0 else SMC_PREMIUM_DISCOUNT_NEUTRAL

        if loc > SMC_PREMIUM_DISCOUNT_NEUTRAL:
            zone = "premium"
        elif loc < SMC_PREMIUM_DISCOUNT_NEUTRAL:
            zone = "discount"
        else:
            zone = "equilibrium"

        return PremiumDiscount(
            range_high,
            range_low,
            equilib,
            zone,
            round(loc, SMC_LOCATION_ROUND_DIGITS),
        )

    # 
    # ORDER BLOCK DETECTION
    # 

    def _find_order_blocks(self, df: pd.DataFrame, atr: float) -> list[OrderBlock]:
        """
        Scan for valid Order Blocks:
          - Bullish OB: last bearish candle before bullish impulse  ATR  mult
          - Bearish OB: last bullish candle before bearish impulse  ATR  mult

        An OB is "broken" if price later traded through the entire OB zone.
        """
        obs    = []
        n      = len(df)
        opens  = df["open"].values
        closes = df["close"].values
        highs  = df["high"].values
        lows   = df["low"].values

        impulse_min = atr * OB_IMPULSE_ATR_MULT

        # Scan from oldest to most recent (skip last 2 bars  need impulse room)
        for i in range(SMC_ORDER_BLOCK_SCAN_START, n - SMC_ORDER_BLOCK_SCAN_END_OFFSET):
            o, c, h, l = opens[i], closes[i], highs[i], lows[i]
            body = abs(c - o)
            rng  = h - l
            if rng < SMC_MIN_CANDLE_RANGE:
                continue
            body_pct = body / rng

            # OB candle must have a meaningful body
            if body_pct < OB_MIN_BODY_PCT:
                continue

            #  Bullish OB: bearish candle (c < o) before bullish impulse 
            if c < o:
                # Check impulse: next 1-3 candles must move up by impulse_min
                lookahead_highs = highs[i + 1 : min(i + 4, n)]
                if len(lookahead_highs) == 0:
                    continue
                future_high = max(lookahead_highs)
                impulse     = future_high - h
                if impulse >= impulse_min:
                    # Check if OB was later broken (price fell below OB low)
                    future_lows  = lows[i+1:]
                    broken       = any(fl < l for fl in future_lows)
                    obs.append(OrderBlock(
                        kind       = "bullish",
                        top        = float(h),
                        bottom     = float(l),
                        open_      = float(o),
                        close_     = float(c),
                        midpoint   = float((h + l) / 2),
                        bar_idx    = n - 1 - i,
                        impulse_sz = float(impulse),
                        broken     = broken,
                    ))

            #  Bearish OB: bullish candle (c > o) before bearish impulse 
            elif c > o:
                lookahead_lows = lows[i + 1 : min(i + 4, n)]
                if len(lookahead_lows) == 0:
                    continue
                future_low  = min(lookahead_lows)
                impulse     = l - future_low
                if impulse >= impulse_min:
                    future_highs = highs[i+1:]
                    broken       = any(fh > h for fh in future_highs)
                    obs.append(OrderBlock(
                        kind       = "bearish",
                        top        = float(h),
                        bottom     = float(l),
                        open_      = float(o),
                        close_     = float(c),
                        midpoint   = float((h + l) / 2),
                        bar_idx    = n - 1 - i,
                        impulse_sz = float(impulse),
                        broken     = broken,
                    ))

        # Return most recent first, unbroken preferred
        return sorted(obs, key=lambda ob: (ob.broken, ob.bar_idx))[:SMC_MAX_ORDER_BLOCKS]

    # 
    # BREAKER BLOCK DETECTION
    # 

    def _find_breakers(
        self, obs: list[OrderBlock], df: pd.DataFrame
    ) -> list[OrderBlock]:
        """
        A Breaker Block is an OB that price broke THROUGH.
        It now acts as the OPPOSITE zone:
          - Broken Bullish OB (was support)  now resistance  Bearish Breaker
          - Broken Bearish OB (was resistance)  now support  Bullish Breaker
        """
        breakers = []
        close    = float(df["close"].iloc[-1])

        for ob in obs:
            if not ob.broken:
                continue
            if ob.bar_idx > BREAKER_MAX_AGE:
                continue

            if ob.kind == "bullish":
                # Broken bullish OB  bearish breaker
                # Price returning to OB zone from below  sell pressure expected
                price_at_breaker = ob.bottom <= close <= ob.top * (1 + OB_RETRACE_TOLERANCE)
                if price_at_breaker:
                    breaker = OrderBlock(
                        kind       = "bearish_breaker",
                        top        = ob.top,
                        bottom     = ob.bottom,
                        open_      = ob.open_,
                        close_     = ob.close_,
                        midpoint   = ob.midpoint,
                        bar_idx    = ob.bar_idx,
                        impulse_sz = ob.impulse_sz,
                        broken     = True,
                    )
                    breakers.append(breaker)

            elif ob.kind == "bearish":
                # Broken bearish OB  bullish breaker
                price_at_breaker = ob.bottom * (1 - OB_RETRACE_TOLERANCE) <= close <= ob.top
                if price_at_breaker:
                    breaker = OrderBlock(
                        kind       = "bullish_breaker",
                        top        = ob.top,
                        bottom     = ob.bottom,
                        open_      = ob.open_,
                        close_     = ob.close_,
                        midpoint   = ob.midpoint,
                        bar_idx    = ob.bar_idx,
                        impulse_sz = ob.impulse_sz,
                        broken     = True,
                    )
                    breakers.append(breaker)

        return sorted(breakers, key=lambda b: b.bar_idx)

    # 
    # SIGNAL BUILDER  combines all four concepts
    # 

    def _build_signal(
        self,
        close:     float,
        structure: MarketStructure,
        pd_zone:   PremiumDiscount,
        obs:       list[OrderBlock],
        breakers:  list[OrderBlock],
        df:        pd.DataFrame,
    ) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        vol_avg = float(df["volume"].rolling(SMC_VOLUME_AVG_WINDOW).mean().iloc[-1])
        vol_now = float(df["volume"].iloc[-1])
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else SMC_DEFAULT_VOLUME_RATIO

        # Volume must show some activity
        if vol_ratio < OB_VOL_MULT:
            return none

        bos_result = self._check_bos(df, structure)

        #  Priority 1: OB + BOS + Premium/Discount alignment 
        for ob in obs:
            if ob.broken:
                continue
            in_zone, direction = self._ob_retrace_check(ob, close)
            if not in_zone:
                continue

            pd_aligned = (
                (direction == "BUY_CALL" and pd_zone.zone == "discount") or
                (direction == "BUY_PUT"  and pd_zone.zone == "premium")
            )
            bos_aligned = (
                bos_result is not None and bos_result[0] == direction
            )

            if pd_aligned and bos_aligned:
                conf = self._score("OB+BOS+PD", ob, vol_ratio, bos_result)
                return self._make_signal(direction, conf, "OB+BOS+PD", ob, pd_zone, bos_result, vol_ratio)

        #  Priority 2: OB + BOS (no PD requirement) 
        for ob in obs:
            if ob.broken:
                continue
            in_zone, direction = self._ob_retrace_check(ob, close)
            if not in_zone:
                continue
            if bos_result and bos_result[0] == direction:
                conf = self._score("OB+BOS", ob, vol_ratio, bos_result)
                return self._make_signal(direction, conf, "OB+BOS", ob, pd_zone, bos_result, vol_ratio)

        #  Priority 3: OB + Premium/Discount (no BOS yet) 
        for ob in obs:
            if ob.broken:
                continue
            in_zone, direction = self._ob_retrace_check(ob, close)
            if not in_zone:
                continue
            pd_aligned = (
                (direction == "BUY_CALL" and pd_zone.zone == "discount") or
                (direction == "BUY_PUT"  and pd_zone.zone == "premium")
            )
            if pd_aligned:
                conf = self._score("OB+PD", ob, vol_ratio, None)
                return self._make_signal(direction, conf, "OB+PD", ob, pd_zone, None, vol_ratio)

        #  Priority 4: Breaker Block + BOS 
        for brk in breakers:
            direction = "BUY_CALL" if brk.kind == "bullish_breaker" else "BUY_PUT"
            if bos_result and bos_result[0] == direction:
                conf = self._score("Breaker+BOS", brk, vol_ratio, bos_result)
                return self._make_signal(direction, conf, "Breaker+BOS", brk, pd_zone, bos_result, vol_ratio)

        #  Priority 5: Breaker Block alone 
        for brk in breakers:
            direction = "BUY_CALL" if brk.kind == "bullish_breaker" else "BUY_PUT"
            conf = self._score("Breaker", brk, vol_ratio, None)
            return self._make_signal(direction, conf, "Breaker", brk, pd_zone, None, vol_ratio)

        #  Priority 6: BOS / ChoCH alone 
        if bos_result:
            direction, bos_type = bos_result
            conf = SMC_BOS_CONFIDENCE if bos_type == "BOS" else SMC_CHOCH_CONFIDENCE
            return {
                "direction":  Direction.BUY_CALL if direction == "BUY_CALL" else Direction.BUY_PUT,
                "confidence": conf,
                "name":       self.name,
                "meta": {
                    "setup":         bos_type,
                    "structure":     structure.trend,
                    "pd_zone":       pd_zone.zone,
                    "pd_location":   pd_zone.location_pct,
                    "bos_type":      bos_type,
                    "vol_ratio":     round(vol_ratio, SMC_META_ROUND_DIGITS),
                },
            }

        return none

    #  OB retrace check 

    @staticmethod
    def _ob_retrace_check(ob: OrderBlock, close: float) -> tuple[bool, str]:
        """Check if current price is within the OB zone."""
        tol  = OB_RETRACE_TOLERANCE
        if ob.kind == "bullish":
            in_zone = ob.bottom * (1 - tol) <= close <= ob.top * (1 + tol)
            return in_zone, "BUY_CALL"
        elif ob.kind == "bearish":
            in_zone = ob.bottom * (1 - tol) <= close <= ob.top * (1 + tol)
            return in_zone, "BUY_PUT"
        return False, ""

    #  Signal factory 

    def _make_signal(
        self,
        direction:  str,
        conf:       float,
        setup:      str,
        ob:         OrderBlock,
        pd_zone:    PremiumDiscount,
        bos_result: tuple | None,
        vol_ratio:  float,
    ) -> dict:
        return {
            "direction":  Direction.BUY_CALL if direction == "BUY_CALL" else Direction.BUY_PUT,
            "confidence": conf,
            "name":       self.name,
            "meta": {
                "setup":         setup,
                "ob_kind":       ob.kind,
                "ob_top":        round(ob.top, SMC_META_ROUND_DIGITS),
                "ob_bottom":     round(ob.bottom, SMC_META_ROUND_DIGITS),
                "ob_midpoint":   round(ob.midpoint, SMC_META_ROUND_DIGITS),
                "ob_age_bars":   ob.bar_idx,
                "ob_impulse":    round(ob.impulse_sz, SMC_META_ROUND_DIGITS),
                "pd_zone":       pd_zone.zone,
                "pd_location":   pd_zone.location_pct,
                "bos_type":      bos_result[1] if bos_result else "none",
                "vol_ratio":     round(vol_ratio, SMC_META_ROUND_DIGITS),
            },
        }

    # 
    # CONFIDENCE SCORING
    # 

    @staticmethod
    def _score(
        setup:      str,
        ob:         OrderBlock,
        vol_ratio:  float,
        bos_result: tuple | None,
    ) -> float:
        """
        Score based on setup quality:
          OB+BOS+PD   0.800.87  (all 3 concepts aligned)
          OB+BOS      0.740.80
          OB+PD       0.700.75
          Breaker+BOS 0.740.80
          Breaker     0.680.73
        """
        base_map = {
            "OB+BOS+PD":   SMC_SCORE_OB_BOS_PD,
            "OB+BOS":      SMC_SCORE_OB_BOS,
            "OB+PD":       SMC_SCORE_OB_PD,
            "Breaker+BOS": SMC_SCORE_BREAKER_BOS,
            "Breaker":     SMC_SCORE_BREAKER,
        }
        conf = base_map.get(setup, SMC_DEFAULT_SCORE)

        # Bonus: strong impulse validated the OB
        if ob.impulse_sz > 0:
            from config.settings import ADX_TREND_THRESHOLD
            pass  # impulse already validated at detection stage
        # Bonus: fresh OB (recent is more reliable)
        if ob.bar_idx <= SMC_FRESH_OB_BARS_TIER_1:
            conf += SMC_FRESH_OB_BONUS_TIER_1
        elif ob.bar_idx <= SMC_FRESH_OB_BARS_TIER_2:
            conf += SMC_FRESH_OB_BONUS_TIER_2

        # Bonus: ChoCH (early reversal  riskier but precise)
        if bos_result and bos_result[1] == "ChoCH":
            conf += SMC_CHOCH_SCORE_BONUS

        # Bonus: strong volume on reaction
        if vol_ratio >= SMC_VOLUME_BONUS_THRESHOLD_HIGH:
            conf += SMC_VOLUME_BONUS_HIGH
        elif vol_ratio >= SMC_VOLUME_BONUS_THRESHOLD_LOW:
            conf += SMC_VOLUME_BONUS_LOW

        return round(min(SMC_MAX_CONFIDENCE, conf), SMC_CONFIDENCE_ROUND_DIGITS)

    #  ATR helper 

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = SMC_ATR_PERIOD) -> float:
        try:
            h  = df["high"]
            l  = df["low"]
            pc = df["close"].shift(1)
            tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
            return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
        except Exception:
            return SMC_ATR_FALLBACK

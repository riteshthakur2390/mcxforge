"""
S15: AMD  Accumulation, Manipulation, Distribution
=====================================================
Based on the ICT (Inner Circle Trader) / Smart Money Concept framework.
Designed specifically for NIFTY 5-minute options trading.


THE THREE PHASES


  PHASE 1  ACCUMULATION
  
  Large participants (institutions) quietly accumulate positions
  inside a tight consolidation zone. This appears as:
    - Low ATR (volatility contracting)
    - Price oscillating within a tight range
    - Equal highs / equal lows (engineered resting liquidity)
    - Choppy, directionless candles

  During accumulation: NO TRADE. We only identify the range.

  PHASE 2  MANIPULATION (False Breakout / Liquidity Sweep)
  
  Institutions sweep the liquidity resting above/below the range
  to fill their positions against retail stops. This appears as:
    - Price briefly breaks ABOVE range_high (triggers long stops)
       then snaps back inside  bearish distribution follows
    - Price briefly breaks BELOW range_low (triggers short stops)
       then snaps back inside  bullish distribution follows
    - The manipulation candle has a prominent wick
    - Volume spikes during the sweep

  PHASE 3  DISTRIBUTION (The Real Move)
  
  After manipulation, price moves aggressively in the OPPOSITE direction
  of the sweep. A break of structure (BOS) confirms the move:
    - Sweep ABOVE highs  distribution is DOWN  BUY PUT
    - Sweep BELOW lows   distribution is UP    BUY CALL


SIGNAL LOGIC FLOWCHART


  Every candle:
    
     STEP 1: Detect or extend ACCUMULATION range
        Requirements:
        - Last AMD_ACC_CANDLES candles have range < AMD_ACC_RANGE_PCT of price
        - ATR(14) < AMD_ATR_CONTRACTION  20-bar ATR average
        - At least AMD_ACC_CANDLES_MIN bars of tight range
        Output: range_high, range_low, accumulation_flag
    
     STEP 2: Check for MANIPULATION (only if accumulation is active)
        Requirements:
        - A recent candle (last AMD_MANIP_LOOKBACK bars) spiked beyond range
        - The SAME candle's close returned INSIDE the range
        - The wick outside range  AMD_WICK_MIN_PCT of the range width
        - Volume on manipulation candle  AMD_MANIP_VOL_MULT  average
        Output: sweep_direction (UP/DOWN), manipulation_flag, sweep_extreme
    
     STEP 3: Check for DISTRIBUTION ENTRY (only after confirmed manipulation)
        Requirements:
        - Manipulation was confirmed in last AMD_DIST_MAX_CANDLES candles
        - Break of structure:
            If sweep was UP: current close < range_low   BUY PUT
            If sweep was DOWN: current close > range_high  BUY CALL
        - At least 1 confirmation candle closed in distribution direction
        Output: direction, confidence, sl, target
    
     NO SIGNAL if any phase is incomplete or invalidated


RISK MANAGEMENT


  Stop loss placement:
    - BUY CALL: SL = sweep_low - AMD_SL_BUFFER_PCT  range_width
      (below the manipulation wick extreme  beyond retail stops)
    - BUY PUT:  SL = sweep_high + AMD_SL_BUFFER_PCT  range_width
      (above the manipulation wick extreme)

  Target placement (minimum 2R):
    - Uses opposite side liquidity level as target
    - BUY CALL target = range_high + (range_high - sweep_low)  AMD_TARGET_R
    - BUY PUT  target = range_low  - (sweep_high - range_low)  AMD_TARGET_R
    - Always at least 2.0R from entry


CONFIDENCE SCORING

  Base: 0.66
  +0.06 if accumulation lasted > AMD_ACC_CANDLES_MIN  1.5 (strong range)
  +0.04 if manipulation wick > AMD_WICK_STRONG_PCT  range width (clean sweep)
  +0.04 if manipulation volume > AMD_MANIP_VOL_STRONG (institutional conviction)
  +0.03 if BOS candle is strong (body > 60% of candle range)
  +0.02 if ATR was deeply contracted (< 0.7  average) before manipulation
  0.05 if manipulation happened > AMD_DIST_MAX_CANDLES/2 bars ago (signal aging)
  Cap: 0.85


INTEGRATION

  Plugs into existing STRATEGY_REGISTRY in runner.py as S15.
  min_candles = 60 (needs enough data to identify accumulation + manipulation).
  Returns standard dict: {direction, confidence, name, meta}.
  meta includes: phase, range_high, range_low, sweep_direction,
                 sl_level, target_level, risk_reward, acc_candles,
                 manip_vol_ratio, wick_pct.

Parameters (tunable without code changes):
  AMD_ACC_CANDLES      = 15    # minimum candles for valid accumulation zone
  AMD_ACC_CANDLES_MIN  = 10    # hard floor  fewer than this = no signal
  AMD_ACC_RANGE_PCT    = 0.30  # max range as % of price during accumulation
  AMD_ATR_CONTRACTION  = 0.80  # ATR must be < this  20-bar avg
  AMD_MANIP_LOOKBACK   = 8     # candles to look back for manipulation signal
  AMD_WICK_MIN_PCT     = 0.15  # min wick outside range as % of range width
  AMD_WICK_STRONG_PCT  = 0.40  # strong sweep: wick > this  range width
  AMD_MANIP_VOL_MULT   = 1.5   # volume on manip candle vs average
  AMD_MANIP_VOL_STRONG = 2.5   # strong institutional activity
  AMD_DIST_MAX_CANDLES = 6     # max candles after manipulation to enter
  AMD_SL_BUFFER_PCT    = 0.10  # SL buffer beyond sweep extreme (% of range)
  AMD_TARGET_R         = 2.0   # minimum risk multiple for target
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from core.models import Direction


from config.settings.strategy import (
    S15_ACC_CANDLES, S15_ACC_CANDLES_MIN, S15_ACC_RANGE_PCT, S15_ATR_CONTRACTION,
    S15_MANIP_LOOKBACK, S15_WICK_MIN_PCT, S15_WICK_STRONG_PCT, S15_MANIP_VOL_MULT,
    S15_MANIP_VOL_STRONG, S15_DIST_MAX_CANDLES, S15_SL_BUFFER_PCT, S15_TARGET_R,
    S15_VOL_MA_PERIOD, S15_ATR_PERIOD, S15_ATR_AVG_PERIOD,
    S15_CONF_BASE, S15_CONF_ACC_LONG_BONUS, S15_CONF_ACC_MIN_BONUS,
    S15_CONF_ATR_DEEP_THRESHOLD, S15_CONF_ATR_DEEP_BONUS,
    S15_CONF_EQ_HL_BOTH_BONUS, S15_CONF_EQ_HL_SINGLE_BONUS,
    S15_CONF_WICK_STRONG_BONUS, S15_CONF_WICK_MIN_BONUS,
    S15_CONF_VOL_STRONG_BONUS, S15_CONF_VOL_MIN_BONUS,
    S15_CONF_FRESH_MANIP_BONUS, S15_CONF_BOS_STRONG_THRESHOLD,
    S15_CONF_BOS_STRONG_BONUS, S15_CONF_RR_BONUS_THRESHOLD, S15_CONF_RR_BONUS,
    S15_CONF_AGING_PENALTY, S15_CONF_MAX, S15_CONF_MIN,
    S15_MIN_DF_OFFSET, S15_EQ_TOLERANCE_PCT
)

# Compatibility aliases
AMD_DIST_MAX_CANDLES = S15_DIST_MAX_CANDLES
AMD_SL_BUFFER_PCT = S15_SL_BUFFER_PCT
AMD_TARGET_R = S15_TARGET_R
AMD_ACC_CANDLES_MIN = S15_ACC_CANDLES_MIN
AMD_WICK_STRONG_PCT = S15_WICK_STRONG_PCT

# 
# DATA CLASSES  internal representation of each phase
# 

@dataclass
class AccumulationZone:
    """
    A detected consolidation range.
    Only valid if tight_enough=True AND candle_count >= AMD_ACC_CANDLES_MIN.
    """
    range_high:    float    # highest high in accumulation window
    range_low:     float    # lowest low in accumulation window
    range_width:   float    # range_high - range_low
    range_pct:     float    # range_width / midpoint  100
    midpoint:      float    # (range_high + range_low) / 2
    candle_count:  int      # how many bars were inside this range
    atr_ratio:     float    # current ATR / 20-bar ATR avg (< 1 = contracted)
    tight_enough:  bool     # True if range qualifies as accumulation
    equal_highs:   bool     # multiple touches of range_high (resting liq)
    equal_lows:    bool     # multiple touches of range_low (resting liq)


@dataclass
class ManipulationEvent:
    """
    A detected false breakout / liquidity sweep.
    sweep_direction: "UP" = swept above range_high, "DOWN" = swept below range_low
    """
    sweep_direction: str    # "UP" or "DOWN"
    sweep_extreme:   float  # furthest point of the wick beyond the range
    wick_pct:        float  # (sweep_extreme - range boundary) / range_width  100
    vol_ratio:       float  # volume on manipulation candle vs average
    bar_index:       int    # position from end of df (0 = current, 1 = prev, etc.)
    confirmed:       bool   # True if candle closed back inside range
    strong:          bool   # True if wick_pct > AMD_WICK_STRONG_PCT


@dataclass
class DistributionSetup:
    """
    Entry signal: break of structure following manipulation.
    Only valid after a confirmed ManipulationEvent.
    """
    direction:     str      # "BUY_CALL" or "BUY_PUT"
    entry_price:   float    # approximate entry (current close)
    sl_level:      float    # stop loss price level on underlying
    target_level:  float    # target price level on underlying
    risk_points:   float    # distance from entry to SL in NIFTY points
    reward_points: float    # distance from entry to target in NIFTY points
    risk_reward:   float    # reward / risk ratio
    bos_strength:  float    # body / candle_range ratio of BOS candle (0-1)


# 
# MAIN STRATEGY CLASS
# 

class AMDStrategy:
    """
    AMD  Accumulation, Manipulation, Distribution.

    Standard interface: evaluate(df, orb_high, orb_low, cache=None)  dict
    Plugs directly into existing STRATEGY_REGISTRY.
    """
    name = "AMD"

    def evaluate(
        self,
        df:       pd.DataFrame,
        orb_high: Optional[float] = None,
        orb_low:  Optional[float] = None,
        cache=    None,
    ) -> dict:
        """
        Main entry point called every candle.
        Runs all three AMD phase detectors in sequence.
        Returns standard strategy result dict.
        """
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        # Need enough candles for all three phases
        if len(df) < S15_ACC_CANDLES + S15_MANIP_LOOKBACK + S15_MIN_DF_OFFSET:
            return none

        try:
            return self._evaluate_internal(df, none)
        except Exception:
            # Never crash the pipeline
            return none

    #  INTERNAL PIPELINE 

    def _evaluate_internal(self, df: pd.DataFrame, none: dict) -> dict:

        #  Volume average (used in all phases) 
        vol_series = df["volume"]
        vol_avg    = float(vol_series.rolling(S15_VOL_MA_PERIOD).mean().iloc[-1])
        if vol_avg <= 0:
            return none

        #  ATR for volatility measurement 
        atr_series = self._compute_atr(df, S15_ATR_PERIOD)
        atr_now    = float(atr_series.iloc[-1])
        atr_avg_period = float(atr_series.tail(S15_ATR_AVG_PERIOD).mean()) if len(atr_series) >= S15_ATR_AVG_PERIOD else atr_now
        atr_ratio  = atr_now / atr_avg_period if atr_avg_period > 0 else 1.0

        # 
        # PHASE 1  ACCUMULATION DETECTION
        # Look at candles BEFORE the most recent S15_MANIP_LOOKBACK bars.
        # The manipulation and distribution happen in the more recent bars.
        # 
        # Accumulation window: bars [-(S15_ACC_CANDLES + S15_MANIP_LOOKBACK) : -S15_MANIP_LOOKBACK]
        acc_end   = len(df) - S15_MANIP_LOOKBACK          # end of accumulation window
        acc_start = acc_end - S15_ACC_CANDLES              # start of accumulation window
        if acc_start < 0:
            return none

        acc_window = df.iloc[acc_start:acc_end]
        acc_zone   = self._detect_accumulation(acc_window, atr_ratio)

        if not acc_zone.tight_enough:
            # No valid accumulation  no AMD setup possible
            return none

        # 
        # PHASE 2  MANIPULATION DETECTION
        # Scan the S15_MANIP_LOOKBACK candles after accumulation for a sweep.
        # 
        manip_window = df.iloc[acc_end:]    # the recent candles after accumulation
        manip_event  = self._detect_manipulation(manip_window, acc_zone, vol_avg)

        if manip_event is None or not manip_event.confirmed:
            # No confirmed sweep yet  still in accumulation or manipulation forming
            return none

        # 
        # PHASE 3  DISTRIBUTION ENTRY
        # Check if current price has broken structure in distribution direction.
        # Entry is OPPOSITE to the manipulation sweep.
        # 
        dist_setup = self._detect_distribution(df, acc_zone, manip_event)

        if dist_setup is None:
            return none

        #  Build signal 
        direction = (
            Direction.BUY_CALL if dist_setup.direction == "BUY_CALL"
            else Direction.BUY_PUT
        )
        confidence = self._score(acc_zone, manip_event, dist_setup)

        return {
            "direction":  direction,
            "confidence": confidence,
            "name":       self.name,
            "meta": {
                # Phase summary
                "phase":            "DISTRIBUTION",
                "amd_pattern":      f"sweep_{manip_event.sweep_direction.lower()}_then_{direction.value.lower()}",
                # Accumulation details
                "range_high":       round(acc_zone.range_high, 2),
                "range_low":        round(acc_zone.range_low, 2),
                "range_pct":        round(acc_zone.range_pct, 3),
                "acc_candles":      acc_zone.candle_count,
                "equal_highs":      acc_zone.equal_highs,
                "equal_lows":       acc_zone.equal_lows,
                "atr_ratio":        round(atr_ratio, 3),
                # Manipulation details
                "sweep_direction":  manip_event.sweep_direction,
                "sweep_extreme":    round(manip_event.sweep_extreme, 2),
                "wick_pct":         round(manip_event.wick_pct, 2),
                "manip_vol_ratio":  round(manip_event.vol_ratio, 2),
                "manip_strong":     manip_event.strong,
                # Risk management
                "sl_level":         round(dist_setup.sl_level, 2),
                "target_level":     round(dist_setup.target_level, 2),
                "risk_points":      round(dist_setup.risk_points, 2),
                "reward_points":    round(dist_setup.reward_points, 2),
                "risk_reward":      round(dist_setup.risk_reward, 2),
                "bos_strength":     round(dist_setup.bos_strength, 3),
            },
        }

    # 
    # PHASE 1: ACCUMULATION DETECTION
    # 

    def _detect_accumulation(
        self,
        window:    pd.DataFrame,
        atr_ratio: float,
    ) -> AccumulationZone:
        """
        Identify a valid accumulation (consolidation) zone.

        Valid accumulation requires:
        1. Tight price range (range_pct < AMD_ACC_RANGE_PCT)
        2. ATR contracted below AMD_ATR_CONTRACTION  20-bar avg
        3. At least AMD_ACC_CANDLES_MIN candles within the range
        4. Optional: equal highs/lows (resting liquidity)
        """
        if len(window) < 3:
            # Not enough data  return a dummy non-qualifying zone
            return AccumulationZone(
                range_high=0, range_low=0, range_width=0, range_pct=999,
                midpoint=0, candle_count=0, atr_ratio=1.0,
                tight_enough=False, equal_highs=False, equal_lows=False,
            )

        highs  = window["high"].values
        lows   = window["low"].values
        closes = window["close"].values

        range_high = float(np.max(highs))
        range_low  = float(np.min(lows))
        range_width = range_high - range_low
        midpoint    = (range_high + range_low) / 2
        range_pct   = range_width / midpoint * 100 if midpoint > 0 else 999

        # Count candles that stayed WITHIN the range (close-to-close)
        candles_inside = int(np.sum(
            (closes >= range_low) & (closes <= range_high)
        ))

        # ATR must be contracted (low volatility)
        atr_ok = atr_ratio <= S15_ATR_CONTRACTION

        # Range must be tight
        range_ok = range_pct <= S15_ACC_RANGE_PCT

        # Minimum qualifying candles
        enough_candles = candles_inside >= S15_ACC_CANDLES_MIN

        # Equal highs: at least 2 candle highs within S15_EQ_TOLERANCE_PCT of range_high
        # (retail stops cluster at these equal levels)
        eq_tol     = range_width * S15_EQ_TOLERANCE_PCT
        equal_highs = int(np.sum(np.abs(highs - range_high) <= eq_tol)) >= 2
        equal_lows  = int(np.sum(np.abs(lows  - range_low)  <= eq_tol)) >= 2

        tight_enough = range_ok and enough_candles

        return AccumulationZone(
            range_high   = range_high,
            range_low    = range_low,
            range_width  = range_width,
            range_pct    = round(range_pct, 4),
            midpoint     = round(midpoint, 2),
            candle_count = candles_inside,
            atr_ratio    = round(atr_ratio, 3),
            tight_enough = tight_enough,
            equal_highs  = equal_highs,
            equal_lows   = equal_lows,
        )

    # 
    # PHASE 2: MANIPULATION DETECTION
    # 

    def _detect_manipulation(
        self,
        window:   pd.DataFrame,
        acc:      AccumulationZone,
        vol_avg:  float,
    ) -> Optional[ManipulationEvent]:
        """
        Scan for a false breakout beyond the accumulation range.

        A valid manipulation candle must:
        1. Have its wick spike BEYOND range_high or range_low
        2. CLOSE back INSIDE the accumulation range
        3. The wick beyond the level  S15_WICK_MIN_PCT  range_width
        4. Volume  S15_MANIP_VOL_MULT  vol_avg
        5. Must have occurred within S15_MANIP_LOOKBACK bars

        We scan from oldest to newest so the most recent valid sweep wins.
        """
        if len(window) == 0:
            return None

        best: Optional[ManipulationEvent] = None
        n = min(len(window), S15_MANIP_LOOKBACK)

        for i in range(n):
            # bar_index: 0 = current candle, higher = older
            bar_index = n - 1 - i
            candle    = window.iloc[i]

            c_high  = float(candle["high"])
            c_low   = float(candle["low"])
            c_close = float(candle["close"])
            c_vol   = float(candle["volume"])
            vol_r   = c_vol / vol_avg if vol_avg > 0 else 0.0

            # Volume gate  must have institutional activity
            if vol_r < S15_MANIP_VOL_MULT:
                continue

            #  UPWARD SWEEP: wick above range_high, close back inside 
            if c_high > acc.range_high and c_close <= acc.range_high:
                wick_above = c_high - acc.range_high
                wick_pct   = wick_above / acc.range_width * 100 if acc.range_width > 0 else 0

                if wick_pct >= S15_WICK_MIN_PCT:
                    event = ManipulationEvent(
                        sweep_direction = "UP",
                        sweep_extreme   = c_high,
                        wick_pct        = round(wick_pct, 3),
                        vol_ratio       = round(vol_r, 3),
                        bar_index       = bar_index,
                        confirmed       = True,
                        strong          = wick_pct >= S15_WICK_STRONG_PCT,
                    )
                    # Prefer the most RECENT confirmed manipulation
                    if best is None or bar_index < best.bar_index:
                        best = event

            #  DOWNWARD SWEEP: wick below range_low, close back inside 
            elif c_low < acc.range_low and c_close >= acc.range_low:
                wick_below = acc.range_low - c_low
                wick_pct   = wick_below / acc.range_width * 100 if acc.range_width > 0 else 0

                if wick_pct >= S15_WICK_MIN_PCT:
                    event = ManipulationEvent(
                        sweep_direction = "DOWN",
                        sweep_extreme   = c_low,
                        wick_pct        = round(wick_pct, 3),
                        vol_ratio       = round(vol_r, 3),
                        bar_index       = bar_index,
                        confirmed       = True,
                        strong          = wick_pct >= S15_WICK_STRONG_PCT,
                    )
                    if best is None or bar_index < best.bar_index:
                        best = event

        return best

    # 
    # PHASE 3: DISTRIBUTION ENTRY
    # 

    def _detect_distribution(
        self,
        df:    pd.DataFrame,
        acc:   AccumulationZone,
        manip: ManipulationEvent,
    ) -> Optional[DistributionSetup]:
        """
        Detect a break of structure (BOS) confirming distribution direction.

        Rules:
        - If manipulation swept UP (above highs):
             Institutions sold into the liquidity grab
             Now price should BREAK BELOW range_low (BOS confirms DOWN)
             Entry = BUY PUT
             SL = above sweep_extreme + buffer

        - If manipulation swept DOWN (below lows):
             Institutions bought the liquidity grab
             Now price should BREAK ABOVE range_high (BOS confirms UP)
             Entry = BUY CALL
             SL = below sweep_extreme - buffer

        Timing gate: manipulation must be recent (within AMD_DIST_MAX_CANDLES)
        """
        curr      = df.iloc[-1]
        bos_c     = df.iloc[-2]   # the BOS candle (prior closed candle confirms)
        c_close   = float(curr["close"])
        bos_close = float(bos_c["close"])
        bos_open  = float(bos_c["open"])
        bos_high  = float(bos_c["high"])
        bos_low   = float(bos_c["low"])

        bos_body  = abs(bos_close - bos_open)
        bos_range = max(bos_high - bos_low, 0.01)
        bos_strength = bos_body / bos_range   # 0.0 = doji, 1.0 = marubozu

        # Timing gate: manipulation must be fresh
        if manip.bar_index > AMD_DIST_MAX_CANDLES:
            return None

        buffer = acc.range_width * AMD_SL_BUFFER_PCT

        #  DISTRIBUTION AFTER UP-SWEEP  BUY PUT (sell signal) 
        if manip.sweep_direction == "UP":
            # BOS: current and prior candle must close BELOW range_low
            bos_confirmed = (bos_close < acc.range_low and
                             c_close   < acc.range_low)
            if not bos_confirmed:
                return None
            # BOS candle must be bearish
            if bos_close >= bos_open:
                return None

            sl_level = manip.sweep_extreme + buffer    # above the wick high
            entry    = c_close

            # Risk = entry (current price) to SL
            # For a PUT option, we profit when price falls.
            # Risk in underlying points = SL - entry (how far back to the trap)
            risk_pts   = max(sl_level - entry, 1.0)
            target_lvl = entry - risk_pts * AMD_TARGET_R
            reward_pts = entry - target_lvl

            return DistributionSetup(
                direction     = "BUY_PUT",
                entry_price   = round(entry, 2),
                sl_level      = round(sl_level, 2),
                target_level  = round(target_lvl, 2),
                risk_points   = round(risk_pts, 2),
                reward_points = round(reward_pts, 2),
                risk_reward   = round(reward_pts / max(risk_pts, 0.01), 2),
                bos_strength  = round(bos_strength, 3),
            )

        #  DISTRIBUTION AFTER DOWN-SWEEP  BUY CALL (buy signal) 
        if manip.sweep_direction == "DOWN":
            # BOS: current and prior candle must close ABOVE range_high
            bos_confirmed = (bos_close > acc.range_high and
                             c_close   > acc.range_high)
            if not bos_confirmed:
                return None
            # BOS candle must be bullish
            if bos_close <= bos_open:
                return None

            sl_level = manip.sweep_extreme - buffer    # below the wick low
            entry    = c_close

            risk_pts   = max(entry - sl_level, 1.0)
            target_lvl = entry + risk_pts * AMD_TARGET_R
            reward_pts = target_lvl - entry

            return DistributionSetup(
                direction     = "BUY_CALL",
                entry_price   = round(entry, 2),
                sl_level      = round(sl_level, 2),
                target_level  = round(target_lvl, 2),
                risk_points   = round(risk_pts, 2),
                reward_points = round(reward_pts, 2),
                risk_reward   = round(reward_pts / max(risk_pts, 0.01), 2),
                bos_strength  = round(bos_strength, 3),
            )

        return None

    # 
    # CONFIDENCE SCORING
    # 

    @staticmethod
    def _score(
        acc:   AccumulationZone,
        manip: ManipulationEvent,
        dist:  DistributionSetup,
    ) -> float:
        """
        Score this AMD setup on a scale of 0.0 to 0.85.

        Quality factors (additive bonuses, no single factor dominates):
          Accumulation quality:
            +0.06 if long accumulation (many candles in range = strong range)
            +0.02 if ATR deeply contracted (clear low-vol phase)
            +0.02 if equal highs/lows present (resting liquidity confirmed)

          Manipulation quality:
            +0.04 if strong wick (deep sweep = more stop orders hit)
            +0.04 if high manipulation volume (strong institutional activity)
            +0.03 if manipulation was very recent (0-2 bars ago = fresh)

          Distribution quality:
            +0.03 if BOS candle has strong body (conviction candle)
            +0.02 if risk-reward > 2.5

        Penalties:
            0.05 if manipulation is aging (> AMD_DIST_MAX_CANDLES/2 bars old)
        """
        conf = S15_CONF_BASE   # base confidence

        #  Accumulation quality 
        if acc.candle_count >= int(S15_ACC_CANDLES_MIN * 1.5):
            conf += S15_CONF_ACC_LONG_BONUS
        elif acc.candle_count >= S15_ACC_CANDLES_MIN:
            conf += S15_CONF_ACC_MIN_BONUS

        if acc.atr_ratio <= S15_CONF_ATR_DEEP_THRESHOLD:    # deeply contracted volatility
            conf += S15_CONF_ATR_DEEP_BONUS

        if acc.equal_highs and acc.equal_lows:
            conf += S15_CONF_EQ_HL_BOTH_BONUS
        elif acc.equal_highs or acc.equal_lows:
            conf += S15_CONF_EQ_HL_SINGLE_BONUS

        #  Manipulation quality 
        if manip.strong:                        # wick > S15_WICK_STRONG_PCT
            conf += S15_CONF_WICK_STRONG_BONUS
        elif manip.wick_pct >= S15_WICK_MIN_PCT:
            conf += S15_CONF_WICK_MIN_BONUS

        if manip.vol_ratio >= S15_MANIP_VOL_STRONG:
            conf += S15_CONF_VOL_STRONG_BONUS
        elif manip.vol_ratio >= S15_MANIP_VOL_MULT:
            conf += S15_CONF_VOL_MIN_BONUS

        if manip.bar_index <= 2:               # very fresh manipulation
            conf += S15_CONF_FRESH_MANIP_BONUS

        #  Distribution quality 
        if dist.bos_strength >= S15_CONF_BOS_STRONG_THRESHOLD:          # strong BOS candle body
            conf += S15_CONF_BOS_STRONG_BONUS

        if dist.risk_reward >= S15_CONF_RR_BONUS_THRESHOLD:
            conf += S15_CONF_RR_BONUS

        #  Penalty: aging manipulation 
        if manip.bar_index > S15_DIST_MAX_CANDLES // 2:
            conf -= S15_CONF_AGING_PENALTY

        return round(min(S15_CONF_MAX, max(S15_CONF_MIN, conf)), 4)

    # 
    # HELPERS
    # 

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = S15_ATR_PERIOD) -> pd.Series:
        """
        Manual ATR computation  does not require pandas_ta.
        Keeps the strategy independent of optional TA library.
        """
        h   = df["high"]
        l   = df["low"]
        c   = df["close"].shift(1)
        tr  = pd.concat(
            [h - l, (h - c).abs(), (l - c).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

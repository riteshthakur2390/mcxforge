"""
utils/smart_entry_filter.py — Smart Money Entry Filter (3 Tasks)
=================================================================

TASK 1: OI + DELTA + GAMMA SMART MONEY FILTER
══════════════════════════════════════════════
Enter where option sellers sit. Option sellers concentrate at:
  - Strikes with HIGH OI (> 100,000 contracts written there)
  - Delta zone 0.45-0.55 (ATM — where sellers collect premium)
  - Low gamma (< 0.0007 — sellers love low gamma, stable premium)

Logic:
  If OI > 100,000 at the strike → sellers are ACTIVE here
    → Buyer SHOULD enter here (price likely to defend/bounce off this)
    → BUT only if delta > 0.45 (enough directional sensitivity)
    → AND gamma > 0.0007 (options are explosively responsive)

  If delta is between 0.45-0.50 → you are in the seller zone
    → Only buy if gamma > 0.0007 (acceleration available)

  Decision table:
    OI > 100k AND delta < 0.45          → SKIP  (too far OTM, sellers not here)
    OI > 100k AND delta 0.45-0.55       → BUY   (inside seller zone, explosive)
    OI > 100k AND delta > 0.55          → BUY   (ITM, strong move already)
    OI < 100k AND gamma < 0.0007        → SKIP  (no seller concentration, low gamma)
    OI < 100k AND gamma ≥ 0.0007        → OK    (normal entry, watch delta)

TASK 2: INVALIDATION-BASED STOP LOSS
══════════════════════════════════════
Stop loss at the price level where the trade THESIS becomes invalid,
not at a fixed percentage loss.

  Trade thesis example:
    "NIFTY will rally from support at 22000 (CPR/VWAP)"
    Thesis is invalid if: NIFTY closes below 22000 on any 5-min candle
    Invalidation price: 22000 (the support level)

  SL calculation:
    option_sl = entry_premium × (1 - delta × (entry_spot - invalidation_spot) / entry_spot)

  Position sizing from invalidation:
    max_risk_inr   = capital × risk_pct (e.g. 1% of fund = ₹1000)
    risk_per_lot   = abs(entry_premium - option_sl_at_invalidation) × lot_size
    lots           = floor(max_risk_inr / risk_per_lot)

  This is how prop traders size positions — risk-first, not capital-first.

TASK 3: SETUP TYPE CLASSIFIER + SL RULES
══════════════════════════════════════════
Three setup types, each with specific SL logic:

  MOMENTUM:
    Definition: strong directional move with volume, ADX > 25, RSI > 60/< 40
    SL rule:    below the last momentum candle low (for calls)
                above the last momentum candle high (for puts)
    Why:        once momentum structure breaks (the last strong candle violated),
                the move is over. Exit immediately.
    SL buffer:  0.5× ATR beyond the structure level (tight)

  REVERSAL:
    Definition: price at support/resistance, RSI divergence, volume climax
    SL rule:    beyond the reversal candle's extreme + 0.25× ATR
    Why:        reversals can accelerate hard if wrong — get out faster
    SL buffer:  0.25× ATR (tightest — adverse moves are dangerous here)

  ACCUMULATION:
    Definition: Wyckoff accumulation, price coiling, OI building, vol drying
    SL rule:    beyond key structure level + 1.5× ATR (wide breathing room)
    Why:        accumulation involves manipulation — fake breakdowns and
                shakeouts. Need wide stop to avoid being stopped out by noise
                before the real move develops.
    SL buffer:  1.5× ATR (widest — patience is the edge here)

═══════════════════════════════════════════════════════════════
INTEGRATION
═══════════════════════════════════════════════════════════════
  Called by planner.py before finalising sl_premium and lots.

  from utils.smart_entry_filter from config.settings.modules.utils_thresholds import *
import SmartEntryFilter
  sef = SmartEntryFilter()

  # Task 1: OI/Delta/Gamma check
  ok, reason = sef.check_oi_delta_gamma(oi=150000, delta=0.48, gamma=0.0009)

  # Task 2: Invalidation SL
  sl = sef.compute_invalidation_sl(
      entry_spot=22100, invalidation_spot=22000,
      entry_premium=150, delta=0.48, lot_size=75,
  )
  lots = sef.size_from_invalidation(sl, entry_premium=150, lot_size=75)

  # Task 3: Setup classifier + SL
  setup = sef.classify_setup(df, direction="BUY_CALL")
  sl_premium = sef.compute_setup_sl(
      setup_type=setup.setup_type, entry_spot=22100,
      entry_premium=150, df=df, direction="BUY_CALL",
  )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from config.settings import (
        DEPLOYED_CAPITAL, SMART_ENTRY_DAILY_CAPITAL_PCT,
        SMART_ENTRY_NIFTY_LOT_SIZE, SMART_ENTRY_STOP_LOSS_PCT, SMART_ENTRY_DELTA_SELLER_ZONE_LOW, SMART_ENTRY_DELTA_SELLER_ZONE_HIGH, SMART_ENTRY_DELTA_MIN_BUY, SMART_ENTRY_GAMMA_MIN_BUY, SMART_ENTRY_GAMMA_SELLER_ACTIVE, SMART_ENTRY_RISK_PER_TRADE_PCT, SMART_ENTRY_MOMENTUM_ADX_MIN, SMART_ENTRY_MOMENTUM_RSI_BULL, SMART_ENTRY_MOMENTUM_RSI_BEAR, SMART_ENTRY_SETUP_CONFIDENCE_MIN, SMART_ENTRY_MOMENTUM_ATR_BUFFER, SMART_ENTRY_REVERSAL_ATR_BUFFER, SMART_ENTRY_ACCUMULATION_ATR_BUFFER,
    )

except ImportError:
    DEPLOYED_CAPITAL   = 100_000

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

OI_SELLER_THRESHOLD    = 100_000    # OI > 100k = active option sellers here

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OIGreeksCheck:
    """Result of Task 1: OI + Delta + Gamma filter."""
    allow_buy:         bool
    reason:            str
    oi:                float
    delta:             float
    gamma:             float
    in_seller_zone:    bool      # delta 0.45-0.55
    oi_pressure:       str       # "SELLER_HEAVY" | "NORMAL" | "THIN"
    edge:              str       # "BUYER_EDGE" | "NEUTRAL" | "SELLER_EDGE"
    confidence_boost:  float     # +0.04 if buyer edge, 0 neutral, -0.04 seller edge

@dataclass
class InvalidationSL:
    """Result of Task 2: Invalidation-based stop loss."""
    invalidation_spot:   float    # NIFTY price that kills the thesis
    option_sl_premium:   float    # option premium at invalidation
    sl_pct_of_entry:     float    # SL as % of entry premium
    risk_per_lot_inr:    float    # ₹ risk per lot if SL hit
    recommended_lots:    int      # lots sized to max_risk_inr
    max_risk_inr:        float    # capital × risk_pct
    method:              str      # "invalidation" | "fallback_fixed_pct"
    note:                str

@dataclass
class SetupClassification:
    """Result of Task 3: Setup type classification."""
    setup_type:          str      # "MOMENTUM" | "REVERSAL" | "ACCUMULATION" | "UNKNOWN"
    confidence:          float    # 0-1
    atr:                 float
    adx:                 float
    rsi:                 float
    momentum_signals:    int      # count of momentum indicators firing
    reversal_signals:    int
    accumulation_signals:int
    sl_buffer_atr_mult:  float    # ATR multiplier for this setup's SL
    note:                str

@dataclass
class SetupSL:
    """Complete SL recommendation for a given setup."""
    setup_type:          str
    sl_spot_level:       float    # NIFTY spot level for SL
    sl_premium:          float    # option premium at SL spot level
    sl_pct:              float    # as % of entry premium
    atr_buffer_pts:      float    # ATR buffer added to structure level
    structure_level:     float    # the key level (momentum low/high, etc.)
    note:                str

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SmartEntryFilter:
    """
    Three-task smart entry and SL framework.
    Plug into planner.py before finalising trade parameters.
    """

    # ── TASK 1: OI + DELTA + GAMMA ────────────────────────────────────────────

    def check_oi_delta_gamma(
        self,
        oi:    float,
        delta: float,
        gamma: float,
    ) -> OIGreeksCheck:
        """
        Check if entry conditions match the smart money setup.

        Args:
            oi:    Open Interest at the target strike (contracts)
            delta: option delta (0-1 for calls, 0 to -1 for puts, pass absolute)
            gamma: option gamma

        Returns:
            OIGreeksCheck with allow_buy and full reasoning
        """
        delta   = abs(delta)
        in_zone = SMART_ENTRY_DELTA_SELLER_ZONE_LOW <= delta <= SMART_ENTRY_DELTA_SELLER_ZONE_HIGH
        oi_heavy= oi >= OI_SELLER_THRESHOLD
        hi_gamma= gamma >= SMART_ENTRY_GAMMA_MIN_BUY

        # OI pressure classification
        if oi >= OI_SELLER_THRESHOLD * 2:
            oi_pressure = "SELLER_HEAVY"
        elif oi >= OI_SELLER_THRESHOLD:
            oi_pressure = "SELLER_ACTIVE"
        elif oi >= OI_SELLER_THRESHOLD * 0.5:
            oi_pressure = "NORMAL"
        else:
            oi_pressure = "THIN"

        # ── Decision logic ────────────────────────────────────────────────
        # Rule 1: OI > 100k AND delta < 0.45 → skip (too far OTM)
        if oi_heavy and delta < SMART_ENTRY_DELTA_MIN_BUY:
            return OIGreeksCheck(
                allow_buy        = False,
                reason           = (f"OI={oi:,.0f} > {OI_SELLER_THRESHOLD:,} BUT "
                                   f"delta={delta:.3f} < {SMART_ENTRY_DELTA_MIN_BUY} — too far OTM"),
                oi=oi, delta=delta, gamma=gamma,
                in_seller_zone   = in_zone,
                oi_pressure      = oi_pressure,
                edge             = "SELLER_EDGE",
                confidence_boost = -0.04,
            )

        # Rule 2: gamma < threshold → sellers dominating, low responsiveness
        if gamma < SMART_ENTRY_GAMMA_SELLER_ACTIVE:
            return OIGreeksCheck(
                allow_buy        = False,
                reason           = (f"Gamma={gamma:.5f} < {SMART_ENTRY_GAMMA_SELLER_ACTIVE} — "
                                   f"option sellers active, low buyer responsiveness"),
                oi=oi, delta=delta, gamma=gamma,
                in_seller_zone   = in_zone,
                oi_pressure      = oi_pressure,
                edge             = "SELLER_EDGE",
                confidence_boost = -0.04,
            )

        # Rule 3: BEST setup — OI heavy + delta in seller zone + high gamma
        if oi_heavy and in_zone and hi_gamma:
            edge   = "BUYER_EDGE"
            boost  = +0.06
            reason = (f"✅ OPTIMAL: OI={oi:,.0f} (sellers here) | "
                     f"delta={delta:.3f} (seller zone) | "
                     f"gamma={gamma:.5f} (explosive) — enter where sellers sit")

        # Rule 4: OI heavy + high gamma but delta out of zone
        elif oi_heavy and hi_gamma:
            edge   = "BUYER_EDGE"
            boost  = +0.04
            reason = (f"✅ GOOD: OI={oi:,.0f} (seller concentration) | "
                     f"delta={delta:.3f} | gamma={gamma:.5f}")

        # Rule 5: Normal OI but gamma sufficient
        elif hi_gamma:
            edge   = "NEUTRAL"
            boost  = 0.0
            reason = f"OK: gamma={gamma:.5f} sufficient | OI={oi:,.0f} (normal)"

        else:
            edge   = "NEUTRAL"
            boost  = 0.0
            reason = f"MARGINAL: gamma={gamma:.5f} borderline | OI={oi:,.0f}"

        logger.debug(
            f"[SmartEntry T1] OI={oi:,.0f} δ={delta:.3f} γ={gamma:.5f} "
            f"→ {edge} boost={boost:+.2f}"
        )

        return OIGreeksCheck(
            allow_buy        = True,
            reason           = reason,
            oi               = oi,
            delta            = delta,
            gamma            = gamma,
            in_seller_zone   = in_zone,
            oi_pressure      = oi_pressure,
            edge             = edge,
            confidence_boost = boost,
        )

    # ── TASK 2: INVALIDATION-BASED SL ────────────────────────────────────────

    def compute_invalidation_sl(
        self,
        entry_spot:          float,
        invalidation_spot:   float,
        entry_premium:       float,
        delta:               float,
        lot_size:            int   = None,
        direction:           str   = "BUY_CALL",
    ) -> InvalidationSL:
        """
        Compute SL at trade thesis invalidation level.

        For BUY_CALL: thesis invalid when NIFTY breaks BELOW support
        For BUY_PUT:  thesis invalid when NIFTY breaks ABOVE resistance

        The option premium at invalidation is estimated using delta:
            option_sl = entry_premium - delta × |entry_spot - invalidation_spot|

        Args:
            entry_spot:        NIFTY price at entry
            invalidation_spot: NIFTY price that kills the trade thesis
            entry_premium:     option LTP at entry
            delta:             option delta at entry (absolute value)
            lot_size:          lot size (default NIFTY=75)
            direction:         "BUY_CALL" or "BUY_PUT"
        """
        ls      = lot_size or SMART_ENTRY_NIFTY_LOT_SIZE
        delta_a = abs(delta)

        # Distance from entry to invalidation in NIFTY points
        spot_distance = abs(entry_spot - invalidation_spot)

        # Validate: invalidation must be in right direction
        if direction == "BUY_CALL" and invalidation_spot >= entry_spot:
            logger.warning(
                f"[SmartEntry T2] CALL invalidation {invalidation_spot} "
                f">= entry {entry_spot} — falling back to fixed SL"
            )
            return self._fallback_sl(entry_premium, ls)

        if direction == "BUY_PUT" and invalidation_spot <= entry_spot:
            logger.warning(
                f"[SmartEntry T2] PUT invalidation {invalidation_spot} "
                f"<= entry {entry_spot} — falling back to fixed SL"
            )
            return self._fallback_sl(entry_premium, ls)

        # Option premium change when NIFTY reaches invalidation
        # Estimated using delta (first-order approximation)
        premium_change = delta_a * spot_distance
        option_sl      = max(5.0, entry_premium - premium_change)
        sl_pct         = (entry_premium - option_sl) / entry_premium * 100

        # Clamp: SL never worse than global SMART_ENTRY_STOP_LOSS_PCT
        if sl_pct > SMART_ENTRY_STOP_LOSS_PCT:
            option_sl = entry_premium * (1 - SMART_ENTRY_STOP_LOSS_PCT / 100)
            sl_pct    = SMART_ENTRY_STOP_LOSS_PCT

        # Position sizing from this SL
        max_risk     = DEPLOYED_CAPITAL * SMART_ENTRY_RISK_PER_TRADE_PCT / 100
        risk_per_lot = (entry_premium - option_sl) * ls
        lots         = max(1, int(max_risk / max(risk_per_lot, 1)))

        note = (
            f"Entry={entry_spot:.0f} → Invalidation={invalidation_spot:.0f} "
            f"({spot_distance:.0f}pts) | "
            f"Option SL: ₹{entry_premium:.1f} → ₹{option_sl:.1f} ({sl_pct:.1f}%) | "
            f"Risk=₹{risk_per_lot:.0f}/lot → {lots} lots"
        )

        logger.info(f"[SmartEntry T2] {note}")

        return InvalidationSL(
            invalidation_spot  = invalidation_spot,
            option_sl_premium  = round(option_sl, 2),
            sl_pct_of_entry    = round(sl_pct, 2),
            risk_per_lot_inr   = round(risk_per_lot, 2),
            recommended_lots   = lots,
            max_risk_inr       = round(max_risk, 2),
            method             = "invalidation",
            note               = note,
        )

    def size_from_invalidation(
        self,
        sl:            InvalidationSL,
        entry_premium: float,
        lot_size:      int = None,
        capital:       float = None,
    ) -> int:
        """
        Compute lot count so that max loss = 1% of capital.
        Returns minimum 1 lot.
        """
        ls      = lot_size or SMART_ENTRY_NIFTY_LOT_SIZE
        cap     = capital  or DEPLOYED_CAPITAL
        risk    = cap * SMART_ENTRY_RISK_PER_TRADE_PCT / 100
        loss_pl = (entry_premium - sl.option_sl_premium) * ls
        if loss_pl <= 0:
            return 1
        return max(1, int(risk / loss_pl))

    # ── TASK 3: SETUP CLASSIFIER ──────────────────────────────────────────────

    def classify_setup(
        self,
        df:        pd.DataFrame,
        direction: str = "BUY_CALL",
    ) -> SetupClassification:
        """
        Classify the current market setup into Momentum, Reversal, or Accumulation.

        Uses:
          - ADX + RSI + volume for momentum
          - RSI divergence + price at S/R for reversal
          - OI buildup + tight range + Wyckoff signals for accumulation

        Returns SetupClassification with setup_type and SL buffer multiplier.
        """
        if df is None or len(df) < 20:
            return self._unknown_setup()

        try:
            atr  = self._atr(df)
            adx  = self._adx(df)
            rsi  = self._rsi(df["close"])
            close= df["close"].values
            high = df["high"].values
            low  = df["low"].values
            vol  = df["volume"].values
            n    = len(df)

            # ── MOMENTUM signals ──────────────────────────────────────────
            momentum_sigs = 0
            # ADX above threshold → trending
            if adx > SMART_ENTRY_MOMENTUM_ADX_MIN:
                momentum_sigs += 1
            # RSI confirms direction
            if direction == "BUY_CALL" and rsi > SMART_ENTRY_MOMENTUM_RSI_BULL:
                momentum_sigs += 1
            elif direction == "BUY_PUT" and rsi < SMART_ENTRY_MOMENTUM_RSI_BEAR:
                momentum_sigs += 1
            # Volume expanding (last 3 candles vol > 20-bar avg)
            vol_avg = float(np.mean(vol[-20:]))
            if all(vol[i] > vol_avg for i in [-3, -2, -1]):
                momentum_sigs += 1
            # Price making consecutive HH (for calls) or LL (for puts)
            if direction == "BUY_CALL":
                if close[-1] > close[-2] > close[-3]:
                    momentum_sigs += 1
            else:
                if close[-1] < close[-2] < close[-3]:
                    momentum_sigs += 1

            # ── REVERSAL signals ──────────────────────────────────────────
            reversal_sigs = 0
            # RSI divergence: price new high/low but RSI not confirming
            price_5   = (close[-1] - close[-5]) / close[-5] * 100
            rsi_5ago  = self._rsi_at(df["close"], -5)
            if direction == "BUY_CALL":
                if close[-1] < close[-5] and rsi > rsi_5ago:
                    reversal_sigs += 1   # bullish divergence
            else:
                if close[-1] > close[-5] and rsi < rsi_5ago:
                    reversal_sigs += 1   # bearish divergence
            # High volume on the reversal candle (climax)
            if vol[-1] > vol_avg * 1.8:
                reversal_sigs += 1
            # Price at recent support/resistance (within 0.3% of 20-bar extreme)
            if direction == "BUY_CALL":
                recent_low_20 = float(np.min(low[-20:]))
                if abs(close[-1] - recent_low_20) / recent_low_20 < 0.003:
                    reversal_sigs += 1
            else:
                recent_high_20 = float(np.max(high[-20:]))
                if abs(close[-1] - recent_high_20) / recent_high_20 < 0.003:
                    reversal_sigs += 1
            # ADX low (no trend = better reversal environment)
            if adx < 20:
                reversal_sigs += 1

            # ── ACCUMULATION signals ──────────────────────────────────────
            accum_sigs = 0
            # Tight price range (last 10 candles)
            range_10   = float(np.max(high[-10:]) - np.min(low[-10:]))
            range_20   = float(np.max(high[-20:]) - np.min(low[-20:]))
            if range_10 < range_20 * 0.5:
                accum_sigs += 1   # range contracting = coiling
            # Volume drying up (declining trend)
            vol_5  = float(np.mean(vol[-5:]))
            vol_20 = float(np.mean(vol[-20:]))
            if vol_5 < vol_20 * 0.75:
                accum_sigs += 1   # volume drying = absorption
            # ADX low (flat, no trend = accumulation possible)
            if adx < 18:
                accum_sigs += 1
            # Price hovering near mean (low absolute ROC)
            roc_10 = abs((close[-1] - close[-10]) / close[-10] * 100)
            if roc_10 < 0.5:
                accum_sigs += 1   # price barely moving

            # ── Classify ──────────────────────────────────────────────────
            scores = {
                "MOMENTUM":    momentum_sigs,
                "REVERSAL":    reversal_sigs,
                "ACCUMULATION":accum_sigs,
            }
            best_type  = max(scores, key=scores.get)
            best_count = scores[best_type]
            confidence = best_count / 4.0   # max 4 signals each

            if confidence < SMART_ENTRY_SETUP_CONFIDENCE_MIN:
                return self._unknown_setup(atr, adx, rsi)

            sl_buffers = {
                "MOMENTUM":    SMART_ENTRY_MOMENTUM_ATR_BUFFER,
                "REVERSAL":    SMART_ENTRY_REVERSAL_ATR_BUFFER,
                "ACCUMULATION":SMART_ENTRY_ACCUMULATION_ATR_BUFFER,
            }

            note = (
                f"{best_type}: {best_count}/4 signals | "
                f"ADX={adx:.1f} RSI={rsi:.1f} ATR={atr:.1f} | "
                f"SL buffer={sl_buffers[best_type]}×ATR"
            )
            logger.info(f"[SmartEntry T3] {note}")

            return SetupClassification(
                setup_type           = best_type,
                confidence           = round(min(confidence, 1.0), 3),
                atr                  = round(atr, 2),
                adx                  = round(adx, 1),
                rsi                  = round(rsi, 1),
                momentum_signals     = momentum_sigs,
                reversal_signals     = reversal_sigs,
                accumulation_signals = accum_sigs,
                sl_buffer_atr_mult   = sl_buffers[best_type],
                note                 = note,
            )

        except Exception as e:
            logger.debug(f"[SmartEntry T3] classify_setup error: {e}")
            return self._unknown_setup()

    def compute_setup_sl(
        self,
        setup_type:    str,
        direction:     str,
        entry_spot:    float,
        entry_premium: float,
        df:            pd.DataFrame = None,
        delta:         float = 0.48,
        lot_size:      int   = None,
    ) -> SetupSL:
        """
        Compute the SL level for a given setup type.

        MOMENTUM:    SL below last momentum candle's low (for calls)
        REVERSAL:    SL beyond reversal candle's extreme + 0.25×ATR
        ACCUMULATION:SL beyond key structure + 1.5×ATR (wide breathing room)
        """
        ls   = lot_size or SMART_ENTRY_NIFTY_LOT_SIZE
        atr  = self._atr(df) if df is not None and len(df) >= 14 else 30.0

        buffers = {
            "MOMENTUM":    SMART_ENTRY_MOMENTUM_ATR_BUFFER,
            "REVERSAL":    SMART_ENTRY_REVERSAL_ATR_BUFFER,
            "ACCUMULATION":SMART_ENTRY_ACCUMULATION_ATR_BUFFER,
        }
        buffer = buffers.get(setup_type, SMART_ENTRY_MOMENTUM_ATR_BUFFER)
        atr_pts = atr * buffer

        high = df["high"].values if df is not None and len(df) >= 3 else None
        low  = df["low"].values  if df is not None and len(df) >= 3 else None

        if setup_type == "MOMENTUM":
            # SL below last strong momentum candle (last 3 bars)
            if direction == "BUY_CALL" and low is not None:
                structure = float(np.min(low[-3:]))   # low of last 3 momentum bars
                sl_spot   = structure - atr_pts
                note      = f"Below last 3-bar low {structure:.0f} - {atr_pts:.0f}pts ATR buffer"
            else:
                structure = float(np.max(high[-3:])) if high is not None else entry_spot
                sl_spot   = structure + atr_pts
                note      = f"Above last 3-bar high {structure:.0f} + {atr_pts:.0f}pts ATR buffer"

        elif setup_type == "REVERSAL":
            # SL just beyond the reversal candle's extreme
            if direction == "BUY_CALL" and low is not None:
                structure = float(low[-1])   # reversal candle's low
                sl_spot   = structure - atr_pts
                note      = f"Below reversal candle low {structure:.0f} - {atr_pts:.0f}pts (tight)"
            else:
                structure = float(high[-1]) if high is not None else entry_spot
                sl_spot   = structure + atr_pts
                note      = f"Above reversal candle high {structure:.0f} + {atr_pts:.0f}pts (tight)"

        elif setup_type == "ACCUMULATION":
            # SL wide — beyond key structure level to withstand manipulation
            if direction == "BUY_CALL" and low is not None:
                structure = float(np.min(low[-10:]))  # 10-bar structure low
                sl_spot   = structure - atr_pts
                note      = f"Below 10-bar structure low {structure:.0f} - {atr_pts:.0f}pts (wide)"
            else:
                structure = float(np.max(high[-10:])) if high is not None else entry_spot
                sl_spot   = structure + atr_pts
                note      = f"Above 10-bar structure high {structure:.0f} + {atr_pts:.0f}pts (wide)"

        else:
            # UNKNOWN: fallback to fixed %
            sl_spot   = entry_spot * 0.975 if direction == "BUY_CALL" else entry_spot * 1.025
            structure = sl_spot
            note      = "Unknown setup: fixed 2.5% SL on spot"

        # Convert spot SL to option premium SL
        spot_distance  = abs(entry_spot - sl_spot)
        option_sl      = max(5.0, entry_premium - abs(delta) * spot_distance)
        sl_pct         = (entry_premium - option_sl) / entry_premium * 100

        # Never exceed global SMART_ENTRY_STOP_LOSS_PCT
        if sl_pct > SMART_ENTRY_STOP_LOSS_PCT:
            option_sl = entry_premium * (1 - SMART_ENTRY_STOP_LOSS_PCT / 100)
            sl_pct    = SMART_ENTRY_STOP_LOSS_PCT

        logger.info(
            f"[SmartEntry T3 SL] {setup_type} | {direction} | "
            f"structure={structure:.0f} | sl_spot={sl_spot:.0f} | "
            f"option_sl=₹{option_sl:.1f} ({sl_pct:.1f}%) | {note}"
        )

        return SetupSL(
            setup_type      = setup_type,
            sl_spot_level   = round(sl_spot, 1),
            sl_premium      = round(option_sl, 2),
            sl_pct          = round(sl_pct, 2),
            atr_buffer_pts  = round(atr_pts, 1),
            structure_level = round(structure, 1),
            note            = note,
        )

    # ── CONVENIENCE: ALL-IN-ONE ───────────────────────────────────────────────

    def evaluate(
        self,
        df:                 pd.DataFrame,
        direction:          str,
        entry_spot:         float,
        entry_premium:      float,
        delta:              float,
        gamma:              float,
        oi:                 float,
        invalidation_spot:  float,
        lot_size:           int   = None,
    ) -> dict:
        """
        Run all 3 tasks and return combined result dict.
        Used by planner.py as a single call.
        """
        # Task 1
        t1 = self.check_oi_delta_gamma(oi, delta, gamma)

        # Task 2
        t2 = self.compute_invalidation_sl(
            entry_spot, invalidation_spot, entry_premium,
            delta, lot_size, direction,
        )

        # Task 3
        setup = self.classify_setup(df, direction)
        t3_sl = self.compute_setup_sl(
            setup.setup_type, direction, entry_spot, entry_premium, df,
            delta, lot_size,
        )

        # Final SL: use the tighter of invalidation SL and setup SL
        final_sl = min(t2.option_sl_premium, t3_sl.sl_premium)
        # Final lots: use invalidation-based sizing
        final_lots = t2.recommended_lots

        return {
            # Task 1
            "allow_buy":          t1.allow_buy,
            "oi_edge":            t1.edge,
            "oi_confidence_boost":t1.confidence_boost,
            "oi_reason":          t1.reason,
            "in_seller_zone":     t1.in_seller_zone,
            # Task 2
            "invalidation_spot":  t2.invalidation_spot,
            "invalidation_sl":    t2.option_sl_premium,
            "invalidation_sl_pct":t2.sl_pct_of_entry,
            "risk_per_lot_inr":   t2.risk_per_lot_inr,
            # Task 3
            "setup_type":         setup.setup_type,
            "setup_confidence":   setup.confidence,
            "setup_sl":           t3_sl.sl_premium,
            "setup_sl_pct":       t3_sl.sl_pct,
            "sl_buffer_atr_mult": setup.sl_buffer_atr_mult,
            "structure_level":    t3_sl.structure_level,
            # Combined
            "final_sl_premium":   round(final_sl, 2),
            "final_lots":         final_lots,
            "setup_note":         setup.note,
            "sl_note":            t3_sl.note,
        }

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _fallback_sl(self, entry_premium: float, lot_size: int) -> InvalidationSL:
        sl = entry_premium * (1 - SMART_ENTRY_STOP_LOSS_PCT / 100)
        rl = (entry_premium - sl) * lot_size
        max_risk = DEPLOYED_CAPITAL * SMART_ENTRY_RISK_PER_TRADE_PCT / 100
        return InvalidationSL(
            invalidation_spot  = 0,
            option_sl_premium  = round(sl, 2),
            sl_pct_of_entry    = SMART_ENTRY_STOP_LOSS_PCT,
            risk_per_lot_inr   = round(rl, 2),
            recommended_lots   = max(1, int(max_risk / max(rl, 1))),
            max_risk_inr       = round(max_risk, 2),
            method             = "fallback_fixed_pct",
            note               = f"Fixed {SMART_ENTRY_STOP_LOSS_PCT}% SL (invalidation fallback)",
        )

    def _unknown_setup(
        self, atr: float = 30.0, adx: float = 20.0, rsi: float = 50.0
    ) -> SetupClassification:
        return SetupClassification(
            setup_type="UNKNOWN", confidence=0.0,
            atr=atr, adx=adx, rsi=rsi,
            momentum_signals=0, reversal_signals=0, accumulation_signals=0,
            sl_buffer_atr_mult=SMART_ENTRY_MOMENTUM_ATR_BUFFER,
            note="Insufficient signals to classify",
        )

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        try:
            h=df["high"]; l=df["low"]; c=df["close"]; pc=c.shift(1)
            tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
            return float(tr.ewm(span=period,adjust=False).mean().iloc[-1])
        except Exception:
            return 30.0

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> float:
        try:
            h=df["high"]; l=df["low"]; c=df["close"]; pc=c.shift(1)
            tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
            return float(tr.ewm(span=period,adjust=False).mean().iloc[-1])
        except Exception:
            return 20.0

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float:
        try:
            d=close.diff().dropna()
            g=d.clip(lower=0).ewm(span=period,adjust=False).mean()
            l=(-d).clip(lower=0).ewm(span=period,adjust=False).mean()
            rs=g.iloc[-1]/max(l.iloc[-1],1e-10)
            return round(100-100/(1+rs),2)
        except Exception:
            return 50.0

    @staticmethod
    def _rsi_at(close: pd.Series, offset: int, period: int = 14) -> float:
        try:
            sub=close.iloc[:offset] if offset < 0 else close.iloc[offset:]
            if len(sub)<period+1:
                return 50.0
            d=sub.diff().dropna()
            g=d.clip(lower=0).ewm(span=period,adjust=False).mean()
            l=(-d).clip(lower=0).ewm(span=period,adjust=False).mean()
            rs=g.iloc[-1]/max(l.iloc[-1],1e-10)
            return round(100-100/(1+rs),2)
        except Exception:
            return 50.0

# ── Singleton ─────────────────────────────────────────────────────────────────
_sef: SmartEntryFilter | None = None

def get_smart_entry_filter() -> SmartEntryFilter:
    global _sef
    if _sef is None:
        _sef = SmartEntryFilter()
    return _sef
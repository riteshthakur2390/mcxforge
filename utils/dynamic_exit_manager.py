"""
utils/dynamic_exit_manager.py — ATR Dynamic SL, Regime-Adaptive Exits, EM Strike Selection
============================================================================================
Three of the five biggest alpha gaps vs top 1% algo traders — in one module.

1. ATR-BASED DYNAMIC SL
   SL = Entry_premium × (1 - atr_sl_pct)
   Where atr_sl_pct is derived from current ATR relative to entry premium.
   
   Why fixed % fails:
   - On a volatile day (ATR=80pts), 25% SL on ₹150 premium = ₹37.5 room
   - On a calm day (ATR=30pts), same ₹37.5 room is 4× too wide
   - ATR-based SL gives proportional room relative to current volatility

2. REGIME-ADAPTIVE SL/TARGET
   TRENDING strong (ADX>30): SL=15%, Target≈42%  → executable expansion target
   TRENDING weak  (ADX18-30): SL=22%, Target≈36%  → first major expansion band
   RANGING:                    SL=30%, Target≈32%  → fast in/out
   CHOPPY:                     SL=20%, Target≈30%  → very fast exit

3. EXPECTED MOVE STRIKE SELECTION
   EM = Spot × IV × sqrt(DTE/365)
   Strike = ATM + 0.5 × EM  (for calls) or ATM - 0.5 × EM (for puts)
   
   High-IV day (VIX=22): EM=400pts → strike 22200CE (200pts OTM)
   Low-IV day  (VIX=12): EM=200pts → strike 22100CE (100pts OTM)
   
   This is how institutional option desks select strikes.
   Fixed OTM_OFFSET is a retail approximation.
"""
from __future__ import annotations


from config.settings.modules.utils_thresholds import *

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

try:
    from config.settings import (
        USE_ATR_SL, DYN_EXIT_ATR_SL_MULTIPLIER, DYN_EXIT_ATR_SL_MIN_PCT, DYN_EXIT_ATR_SL_MAX_PCT,
        DYN_EXIT_ATR_PERIOD, REGIME_ADAPTIVE_EXITS,
        DYN_EXIT_SL_TRENDING_STRONG, DYN_EXIT_TARGET_TRENDING_STRONG,
        DYN_EXIT_SL_TRENDING_WEAK,   DYN_EXIT_TARGET_TRENDING_WEAK,
        DYN_EXIT_SL_RANGING,         DYN_EXIT_TARGET_RANGING,
        DYN_EXIT_SL_CHOPPY,          DYN_EXIT_TARGET_CHOPPY,
        DYN_EXIT_STOP_LOSS_PCT,      DYN_EXIT_TARGET_PCT,
        USE_EM_STRIKE,      DYN_EXIT_EM_FRACTION,
        DYN_EXIT_EM_MIN_OTM_PCT,     DYN_EXIT_EM_MAX_OTM_PCT,
        DYN_EXIT_NIFTY_STRIKE_STEP,
    )
except ImportError:
    USE_ATR_SL             = True

    REGIME_ADAPTIVE_EXITS  = True

    USE_EM_STRIKE          = True

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def _target_capture_band(regime: str, adx: float) -> tuple[float, float]:
    """Return executable target floor/cap for option premium expansion."""
    regime = str(regime or "").upper()
    if regime == "TRENDING" and float(adx or 0.0) >= 30.0:
        return 38.0, 46.0
    if regime == "TRENDING":
        return 34.0, 40.0
    if regime == "RANGING":
        return 30.0, 34.0
    if regime == "CHOPPY":
        return 30.0, 32.0
    return 32.0, 40.0

@dataclass
class ExitParams:
    sl_pct:       float     # stop loss as % of entry premium
    target_pct:   float     # target as % of entry premium
    sl_premium:   float     # absolute SL premium level
    target_prem:  float     # absolute target premium level
    method:       str       # "ATR" | "REGIME" | "FIXED"
    regime:       str       # current regime used
    adx:          float
    note:         str

@dataclass
class StrikeSelection:
    strike:          int
    method:          str      # "EM" | "FIXED"
    em_pts:          float    # expected move in NIFTY points
    otm_pts:         int      # OTM offset applied
    otm_pct:         float    # OTM as % of spot
    entry_delta_est: float    # estimated delta of selected strike
    note:            str

# ═════════════════════════════════════════════════════════════════════════════
# 1. ATR-BASED DYNAMIC SL
# ═════════════════════════════════════════════════════════════════════════════

def compute_atr_sl(
    df:            pd.DataFrame,
    entry_premium: float,
    regime:        str   = "TRENDING",
    adx:           float = 20.0,
    use_atr:       bool  = None,
) -> ExitParams:
    """
    Compute SL and Target using ATR-based or regime-adaptive approach.

    Priority:
    1. If USE_ATR_SL and df has enough data → ATR-based SL
    2. If REGIME_ADAPTIVE_EXITS → regime-based fixed %
    3. Fallback → global DYN_EXIT_STOP_LOSS_PCT / DYN_EXIT_TARGET_PCT

    Args:
        df:            5-min OHLCV DataFrame (needs >= DYN_EXIT_ATR_PERIOD+1 bars)
        entry_premium: option LTP at entry
        regime:        current regime label
        adx:           current ADX value
        use_atr:       override USE_ATR_SL setting
    """
    _use_atr = use_atr if use_atr is not None else USE_ATR_SL

    # ── Method 1: ATR-based SL ────────────────────────────────────────────────
    if _use_atr and df is not None and len(df) >= DYN_EXIT_ATR_PERIOD + 1:
        atr_val = _compute_atr(df, DYN_EXIT_ATR_PERIOD)
        if atr_val > 0 and entry_premium > 0:
            # Convert ATR (in NIFTY points) to option premium SL
            # Option moves ~delta per NIFTY point. Use 0.40 as avg delta.
            avg_delta      = 0.40
            atr_premium_eq = atr_val * avg_delta * DYN_EXIT_ATR_SL_MULTIPLIER
            atr_sl_pct     = (atr_premium_eq / entry_premium) * 100

            # Clamp to min/max
            sl_pct = float(np.clip(atr_sl_pct, DYN_EXIT_ATR_SL_MIN_PCT, DYN_EXIT_ATR_SL_MAX_PCT))

            # Target the first tradable FnO expansion band. The previous
            # 3R/80% target was rarely executable and converted most winners
            # into PROFIT_PROTECT exits before TARGET_HIT.
            target_floor, target_cap = _target_capture_band(regime, adx)
            tgt_pct = float(np.clip(max(sl_pct * 2.0, target_floor), target_floor, target_cap))
            if regime == "RANGING":
                tgt_pct = sl_pct * 1.5   # tighter target in ranging
            elif regime == "CHOPPY":
                tgt_pct = sl_pct * 1.2

            sl_prem  = round(entry_premium * (1 - sl_pct / 100), 2)
            tgt_prem = round(entry_premium * (1 + tgt_pct / 100), 2)

            logger.debug(
                f"[DynamicExit] ATR={atr_val:.1f} | "
                f"ATR_SL={sl_pct:.1f}% | Target={tgt_pct:.1f}% | "
                f"Regime={regime} ADX={adx:.1f}"
            )

            return ExitParams(
                sl_pct      = round(sl_pct, 2),
                target_pct  = round(tgt_pct, 2),
                sl_premium  = sl_prem,
                target_prem = tgt_prem,
                method      = "ATR",
                regime      = regime,
                adx         = adx,
                note        = f"ATR={atr_val:.1f}pts × {DYN_EXIT_ATR_SL_MULTIPLIER} × delta=0.40",
            )

    # ── Method 2: Regime-adaptive fixed % ────────────────────────────────────
    if REGIME_ADAPTIVE_EXITS:
        if regime == "TRENDING" and adx >= 30:
            sl, tgt = DYN_EXIT_SL_TRENDING_STRONG, DYN_EXIT_TARGET_TRENDING_STRONG
            note    = f"TRENDING_STRONG (ADX={adx:.1f}≥30)"
        elif regime == "TRENDING":
            sl, tgt = DYN_EXIT_SL_TRENDING_WEAK, DYN_EXIT_TARGET_TRENDING_WEAK
            note    = f"TRENDING_WEAK (ADX={adx:.1f}<30)"
        elif regime == "RANGING":
            sl, tgt = DYN_EXIT_SL_RANGING, DYN_EXIT_TARGET_RANGING
            note    = "RANGING: tight target, wider SL"
        elif regime == "CHOPPY":
            sl, tgt = DYN_EXIT_SL_CHOPPY, DYN_EXIT_TARGET_CHOPPY
            note    = "CHOPPY: very fast in/out"
        else:
            sl, tgt = DYN_EXIT_STOP_LOSS_PCT, DYN_EXIT_TARGET_PCT
            note    = f"REGIME={regime}: using defaults"

        return ExitParams(
            sl_pct      = sl,
            target_pct  = tgt,
            sl_premium  = round(entry_premium * (1 - sl / 100), 2),
            target_prem = round(entry_premium * (1 + tgt / 100), 2),
            method      = "REGIME",
            regime      = regime,
            adx         = adx,
            note        = note,
        )

    # ── Fallback ──────────────────────────────────────────────────────────────
    return ExitParams(
        sl_pct      = DYN_EXIT_STOP_LOSS_PCT,
        target_pct  = DYN_EXIT_TARGET_PCT,
        sl_premium  = round(entry_premium * (1 - DYN_EXIT_STOP_LOSS_PCT / 100), 2),
        target_prem = round(entry_premium * (1 + DYN_EXIT_TARGET_PCT / 100), 2),
        method      = "FIXED",
        regime      = regime,
        adx         = adx,
        note        = "fallback to global settings",
    )

# ═════════════════════════════════════════════════════════════════════════════
# 2. EXPECTED MOVE STRIKE SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def select_strike_em(
    spot:        float,
    direction:   str,
    dte:         int,
    iv:          float,
    step:        int   = None,
    use_em:      bool  = None,
) -> StrikeSelection:
    """
    Select option strike using Expected Move formula.

    EM = Spot × IV × sqrt(DTE/365)
    Strike = ATM ± (DYN_EXIT_EM_FRACTION × EM)  rounded to strike step

    This adapts strike selection to current IV and DTE automatically.
    High-IV: goes deeper OTM (more leverage, options already expensive)
    Low-IV:  stays near ATM (options cheap, want more delta)
    Near expiry: strikes closer to ATM (EM shrinks with DTE)

    Args:
        spot:      underlying price (NIFTY or SENSEX)
        direction: "BUY_CALL" or "BUY_PUT"
        dte:       days to expiry
        iv:        implied volatility (e.g. 0.15 = 15%)
        step:      strike step (50 for NIFTY, 100 for SENSEX)
        use_em:    override USE_EM_STRIKE

    Returns:
        StrikeSelection with recommended strike and metadata
    """
    _step   = step   or DYN_EXIT_NIFTY_STRIKE_STEP
    _use_em = use_em if use_em is not None else USE_EM_STRIKE

    atm = int(round(spot / _step) * _step)

    if not _use_em or dte <= 0 or iv <= 0:
        # Fallback: fixed offset
        fixed_offset = int(spot * 0.005 / _step) * _step   # ~0.5% OTM
        strike = atm + fixed_offset if direction == "BUY_CALL" else atm - fixed_offset
        return StrikeSelection(
            strike          = strike,
            method          = "FIXED",
            em_pts          = 0.0,
            otm_pts         = fixed_offset,
            otm_pct         = round(fixed_offset / spot * 100, 3),
            entry_delta_est = 0.45,
            note            = "fixed offset fallback",
        )

    # Expected Move in underlying points
    em_pts = spot * iv * math.sqrt(dte / 365.0)

    # OTM offset = DYN_EXIT_EM_FRACTION × EM
    raw_otm = em_pts * DYN_EXIT_EM_FRACTION

    # Clamp to min/max % of spot
    min_otm = spot * DYN_EXIT_EM_MIN_OTM_PCT / 100
    max_otm = spot * DYN_EXIT_EM_MAX_OTM_PCT / 100
    otm_pts = float(np.clip(raw_otm, min_otm, max_otm))

    # Round to nearest strike step
    otm_rounded = int(round(otm_pts / _step) * _step)

    if direction == "BUY_CALL":
        strike = atm + otm_rounded
    else:
        strike = atm - otm_rounded

    # Estimate delta at selected strike
    moneyness  = abs(strike - atm) / spot
    delta_est  = max(0.10, 0.50 - moneyness * 8)   # rough linear approximation

    logger.debug(
        f"[EM Strike] Spot={spot:.0f} IV={iv:.0%} DTE={dte} | "
        f"EM={em_pts:.0f}pts | OTM={otm_rounded}pts | "
        f"Strike={strike} | δ≈{delta_est:.2f}"
    )

    return StrikeSelection(
        strike          = strike,
        method          = "EM",
        em_pts          = round(em_pts, 1),
        otm_pts         = otm_rounded,
        otm_pct         = round(otm_rounded / spot * 100, 3),
        entry_delta_est = round(delta_est, 3),
        note            = f"EM={em_pts:.0f}pts × {DYN_EXIT_EM_FRACTION} = {otm_pts:.0f}pts OTM",
    )

# ── ATR helper ────────────────────────────────────────────────────────────────

def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        h  = df["high"]
        l  = df["low"]
        pc = df["close"].shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
    except Exception:
        return 0.0

def get_current_iv_proxy(df: pd.DataFrame, vix: float = 18.0) -> float:
    """
    Estimate current IV from India VIX or historical volatility.
    VIX is the most reliable proxy for ATM IV.
    """
    if vix > 0:
        return vix / 100.0   # VIX 18 = 18% annualised IV

    # Fallback: compute from price returns
    try:
        close    = df["close"]
        log_ret  = np.log(close / close.shift(1)).dropna()
        daily_vol= float(log_ret.tail(20).std())
        return daily_vol * math.sqrt(252 * 75)   # annualise (75 5-min bars/day)
    except Exception:
        return 0.15   # safe default

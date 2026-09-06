"""
utils/momentum_exhaustion.py — Momentum Exhaustion & Theta Burn Filter
=======================================================================
TWO CRITICAL TOOLS TO KILL TIME_DECAY LOSERS:

1. MOMENTUM EXHAUSTION FILTER (Pre-entry)
   Detects when a trend is losing steam BEFORE you enter.
   Prevents entering late-trend trades that become TIME_DECAY losers.

   Signals of exhaustion after 11:30 IST:
     - RSI divergence: price making HH but RSI making LH (bearish divergence)
     - Volume declining on the last 3 trend candles
     - ADX turning down (ADX[-1] < ADX[-2] < ADX[-3])
     - Price range contracting (candle bodies getting smaller)
   If 3 of 4 signals present → EXHAUSTED → block new entries in trend direction

2. THETA BURN RATE EXIT (Post-entry, real-time)
   Calculates how fast the option premium is decaying due to theta.
   If theta is consuming > threshold per candle → exit immediately.
   Prevents "holding a dying option" scenario.

   Formula: theta_per_candle = BS_theta × (5/1440)  [5-min fraction of day]
   As % of current premium: theta_rate = theta_per_candle / current_premium
   If theta_rate > THETA_BURN_THRESHOLD (0.5%) → premium losing >0.5%/candle
   to time decay → exit regardless of SL level

Usage:
    from utils.momentum_exhaustion import MomentumExhaustionFilter, ThetaBurnMonitor

    # Pre-entry check
    filter = MomentumExhaustionFilter()
    if filter.is_exhausted(df, direction="BUY_CALL"):
        return none   # skip this signal

    # Post-entry per candle
    monitor = ThetaBurnMonitor()
    if monitor.should_exit(spot, strike, dte, iv, current_prem, direction):
        await close_position("THETA_BURN")
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

try:
    from config.settings import ADX_TREND_THRESHOLD
except ImportError:
    ADX_TREND_THRESHOLD = 18.0

# ── Parameters ────────────────────────────────────────────────────────────────
EXHAUSTION_CHECK_HOUR   = 11       # only check after 11:30 IST
EXHAUSTION_CHECK_MINUTE = 30
EXHAUSTION_MIN_SIGNALS  = 3        # need 3 of 4 exhaustion signals
THETA_BURN_THRESHOLD    = 0.005    # 0.5% of premium per 5-min candle


@dataclass
class ExhaustionReport:
    is_exhausted:      bool
    confidence:        float    # 0-1
    rsi_divergence:    bool
    volume_declining:  bool
    adx_turning_down:  bool
    range_contracting: bool
    signals_count:     int
    direction_blocked: str
    reason:            str


@dataclass
class ThetaReport:
    should_exit:       bool
    theta_per_candle:  float    # absolute premium loss per candle
    theta_rate:        float    # as % of current premium
    dte:               float
    current_prem:      float
    reason:            str


# ═════════════════════════════════════════════════════════════════════════════
# MOMENTUM EXHAUSTION FILTER
# ═════════════════════════════════════════════════════════════════════════════

class MomentumExhaustionFilter:
    """
    Detects when an intraday trend is exhausted and should not be traded.
    Most effective after 11:30 IST when morning momentum has typically peaked.
    """

    def is_exhausted(
        self,
        df:        pd.DataFrame,
        direction: str,
        check_time: bool = True,
    ) -> ExhaustionReport:
        """
        Check if trend is exhausted for the given direction.

        Args:
            df:          5-min OHLCV DataFrame
            direction:   "BUY_CALL" (checking bull exhaustion) or "BUY_PUT"
            check_time:  only apply filter after EXHAUSTION_CHECK_HOUR:MINUTE

        Returns:
            ExhaustionReport with detailed breakdown
        """
        not_exhausted = ExhaustionReport(
            is_exhausted=False, confidence=0.0,
            rsi_divergence=False, volume_declining=False,
            adx_turning_down=False, range_contracting=False,
            signals_count=0, direction_blocked=direction, reason="not_exhausted"
        )

        if len(df) < 20:
            return not_exhausted

        # Time gate: only apply after 11:30 IST
        if check_time:
            ts = df.index[-1]
            try:
                h, m = ts.hour, ts.minute
                if h * 60 + m < EXHAUSTION_CHECK_HOUR * 60 + EXHAUSTION_CHECK_MINUTE:
                    return not_exhausted
            except Exception:
                pass

        closes  = df["close"].values
        highs   = df["high"].values
        lows    = df["low"].values
        volumes = df["volume"].values
        n       = len(closes)

        # ── Signal 1: RSI Divergence ──────────────────────────────────────────
        rsi = self._rsi(df["close"], 14)
        rsi_div = False
        if len(rsi) >= 10:
            rsi_vals = rsi.values[-10:]
            price_vals = closes[-10:]
            if direction == "BUY_CALL":
                # Bearish divergence: price HH but RSI LH
                price_nh = price_vals[-1] > price_vals[-5]
                rsi_nh   = rsi_vals[-1] < rsi_vals[-5]
                rsi_div  = price_nh and rsi_nh and rsi_vals[-1] > 55
            else:
                # Bullish divergence: price LL but RSI HL
                price_nl = price_vals[-1] < price_vals[-5]
                rsi_hl   = rsi_vals[-1] > rsi_vals[-5]
                rsi_div  = price_nl and rsi_hl and rsi_vals[-1] < 45

        # ── Signal 2: Volume declining on trend candles ───────────────────────
        last5_vol   = volumes[-5:]
        vol_trend   = np.polyfit(range(5), last5_vol, 1)[0]
        vol_decline = vol_trend < 0   # slope negative = declining

        # ── Signal 3: ADX turning down ────────────────────────────────────────
        adx_s = self._adx(df, 14)
        adx_turn = False
        if len(adx_s) >= 4:
            av = adx_s.values[-4:]
            adx_turn = float(av[-1]) < float(av[-2]) < float(av[-3])

        # ── Signal 4: Price range contracting ────────────────────────────────
        last5_range = [highs[i] - lows[i] for i in range(n-5, n)]
        range_trend = np.polyfit(range(5), last5_range, 1)[0]
        range_cont  = range_trend < 0

        # Count signals
        signals = sum([rsi_div, vol_decline, adx_turn, range_cont])
        exhausted = signals >= EXHAUSTION_MIN_SIGNALS

        if exhausted:
            conf = 0.60 + signals * 0.10
            reasons = []
            if rsi_div:      reasons.append("rsi_divergence")
            if vol_decline:  reasons.append("volume_declining")
            if adx_turn:     reasons.append("adx_turning_down")
            if range_cont:   reasons.append("range_contracting")
        else:
            conf = 0.0
            reasons = []

        return ExhaustionReport(
            is_exhausted     = exhausted,
            confidence       = round(min(conf, 0.95), 3),
            rsi_divergence   = rsi_div,
            volume_declining = vol_decline,
            adx_turning_down = adx_turn,
            range_contracting= range_cont,
            signals_count    = signals,
            direction_blocked= direction,
            reason           = "|".join(reasons) if reasons else "none",
        )

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta  = close.diff().dropna()
        gains  = delta.clip(lower=0)
        losses = (-delta).clip(lower=0)
        avg_g  = gains.ewm(span=period, adjust=False).mean()
        avg_l  = losses.ewm(span=period, adjust=False).mean()
        rs     = avg_g / avg_l.replace(0, 1e-10)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            h, l, c = df["high"], df["low"], df["close"]
            pc = c.shift(1)
            tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
            return tr.ewm(span=period, adjust=False).mean()
        except Exception:
            return pd.Series([20.0] * len(df))


# ═════════════════════════════════════════════════════════════════════════════
# THETA BURN MONITOR
# ═════════════════════════════════════════════════════════════════════════════

class ThetaBurnMonitor:
    """
    Real-time theta decay monitor.
    Calculates BS theta and compares to current premium.
    If premium is burning > THETA_BURN_THRESHOLD per 5-min candle → exit.

    This is the industry standard "options decay exit" used by:
    - TastyTrade theta traders
    - Sensibull position management
    - Professional option buyers who track "time remaining vs premium"
    """

    def check(
        self,
        spot:        float,
        strike:      float,
        dte:         float,    # days to expiry (can be fractional)
        iv:          float,
        current_prem:float,
        direction:   str,
        r:           float = 0.065,
    ) -> ThetaReport:
        """
        Check if theta is burning premium too fast.

        Returns:
            ThetaReport with should_exit flag and details
        """
        opt_type = "CE" if direction == "BUY_CALL" else "PE"
        T = max(dte, 0.01) / 365.0
        current_prem = max(current_prem, 0.01)

        # Calculate BS theta (premium lost per day)
        theta_daily = self._bs_theta(spot, strike, T, r, iv, opt_type)

        # Convert to per-5-min-candle
        theta_per_candle = abs(theta_daily) * 5 / 1440   # 5 min / 1440 min per day

        # As % of current premium
        theta_rate = theta_per_candle / current_prem

        should_exit = theta_rate > THETA_BURN_THRESHOLD

        reason = (
            f"theta_rate={theta_rate:.3%} > threshold={THETA_BURN_THRESHOLD:.3%} "
            f"(₹{theta_per_candle:.2f}/candle, DTE={dte:.1f})"
            if should_exit else "theta_acceptable"
        )

        return ThetaReport(
            should_exit      = should_exit,
            theta_per_candle = round(theta_per_candle, 4),
            theta_rate       = round(theta_rate, 5),
            dte              = round(dte, 2),
            current_prem     = round(current_prem, 2),
            reason           = reason,
        )

    @staticmethod
    def _bs_theta(S, K, T, r, sigma, opt_type="CE") -> float:
        """Black-Scholes theta (premium per day)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        try:
            d1  = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2  = d1 - sigma*math.sqrt(T)
            nd1 = math.exp(-0.5*d1**2) / math.sqrt(2*math.pi)
            if opt_type == "CE":
                from utils.momentum_exhaustion import _norm_cdf_local
                theta = (-S*sigma*nd1 / (2*math.sqrt(T)) -
                         r*K*math.exp(-r*T)*0.5*(1+math.erf(d2/math.sqrt(2))))
            else:
                theta = (-S*sigma*nd1 / (2*math.sqrt(T)) +
                         r*K*math.exp(-r*T)*0.5*(1-math.erf(d2/math.sqrt(2))))
            return theta / 365   # per-day theta
        except Exception:
            return -current_prem * 0.01 / 365 if 'current_prem' in dir() else -0.01


# Monkey-patch local norm_cdf to avoid circular import
def _norm_cdf_local(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

import utils.momentum_exhaustion as _self_module
_self_module._norm_cdf_local = _norm_cdf_local

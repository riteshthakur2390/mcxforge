"""
utils/morning_bias.py — Morning Bias System + Adaptive Gate Loosening
======================================================================
THE #1 REASON FOR LOW SIGNAL COUNT:
  System waits for signals to emerge from the registered strategy set + gates.
  Top traders START the day with a directional bias BEFORE the first candle.
  Then only take signals in that direction → win rate doubles immediately.

THREE COMPONENTS:

1. MORNING BIAS SCORE (pre-market, computed at 09:00 IST)
   Combines:
   - SGX Nifty / Dow Futures direction (global bias)
   - India VIX level and trend
   - Previous day's close vs CPR (above = bullish, below = bearish)
   - Gap at open (gap up = bull bias, gap down = bear bias)
   - FII/DII flow from previous day
   Output: bias = BULLISH/BEARISH/NEUTRAL + strength 0-100

2. ADAPTIVE GATE LOOSENING (real-time)
   When conditions are STRONG, relax the filters to capture more signals.
   Strong uptrend (ADX>30 + price above VWAP + bull bias) →
     MIN_STRATEGY_VOTES: 2 → 1 (allow single strong strategy signal)
     ML_MIN_CONFIDENCE:  0.36 → 0.32
     ENTRY_MIN_DET_CONF: 0.62 → 0.55
   
   This is how professional systems work — not one-size-fits-all thresholds.
   Tight gates in uncertainty, loose gates in strong conviction.

3. SESSION TIME WEIGHTING
   09:15-10:00: Opening range — DO NOT TRADE (institutions placing orders)
   10:00-11:30: PRIME WINDOW — best signal quality, full size
   11:30-13:00: SECONDARY WINDOW — good signals, normal size
   13:00-14:00: WEAK WINDOW — reduce size 50%, higher bar
   14:00-14:30: LAST WINDOW — only very high conf (>0.85) signals
   14:30+:      NO NEW SIGNALS

   Most retail traders miss this entirely — they treat all candles equally.
   Institutional activity peaks 10:00-11:30. That's where the real edge is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from config.settings import (
        MIN_STRATEGY_VOTES, ML_MIN_CONFIDENCE,
        ENTRY_MIN_DET_CONF, ADX_TREND_THRESHOLD,
    )
except ImportError:
    MIN_STRATEGY_VOTES = 2
    ML_MIN_CONFIDENCE  = 0.36
    ENTRY_MIN_DET_CONF = 0.62
    ADX_TREND_THRESHOLD= 18.0

# ── Session time windows (MCX Commodity: 09:00 - 23:30 IST) ───────────────────
SESSION_WINDOWS = [
    # (start, end, label, size_mult, min_conf_override, notes)
    (dtime(9,  0),  dtime(9,  30), "OPENING_RANGE", 0.0,  0.99, "Opening range build (ORB window)"),
    (dtime(9, 30),  dtime(13,  0), "MORNING_ACTIVE",1.0,  0.00, "Morning active session — full size, normal gates"),
    (dtime(13, 0),  dtime(17,  0), "MIDDAY_ACTIVE", 1.0,  0.00, "Midday European session — full size, normal gates"),
    (dtime(17, 0),  dtime(22, 30), "EVENING_PRIME", 1.0,  0.00, "Evening US session — peak volume, full size"),
    (dtime(22, 30), dtime(23, 15), "LATE_SESSION",  0.75, 0.70, "Late session — 75% size, higher confidence"),
    (dtime(23, 15), dtime(23, 30), "EOD_CUTOFF",    0.0,  0.99, "EOD square-off / market close"),
]


@dataclass
class MorningBias:
    bias:          str     # "BULLISH" | "BEARISH" | "NEUTRAL"
    strength:      float   # 0-100
    direction:     str     # "BUY_CALL" | "BUY_PUT" | "BOTH"
    gap_pct:       float
    vix_signal:    str     # "FAVOURABLE" | "ELEVATED" | "EXTREME"
    prev_close_vs_cpr: str # "ABOVE" | "BELOW" | "INSIDE"
    fii_flow:      str     # "BUYING" | "SELLING" | "NEUTRAL"
    global_bias:   str     # "BULLISH" | "BEARISH" | "NEUTRAL"
    note:          str


@dataclass
class SessionWindow:
    label:         str
    size_mult:     float   # position size multiplier
    min_conf:      float   # minimum confidence override (0 = use system default)
    is_tradeable:  bool
    note:          str


@dataclass
class AdaptiveGates:
    """Dynamically loosened or tightened gate thresholds."""
    min_votes:         int
    ml_min_confidence: float
    entry_min_det_conf:float
    adx_threshold:     float
    reason:            str
    loosened:          bool   # True if gates were loosened from defaults


class MorningBiasSystem:
    """
    Establishes pre-market bias and adjusts gates dynamically.
    Call compute_bias() once at 09:00 IST.
    Call get_session_window() every candle.
    Call get_adaptive_gates() before each signal check.
    """

    def __init__(self) -> None:
        self._bias: Optional[MorningBias] = None
        self._bias_date = ""

    # ── MORNING BIAS ─────────────────────────────────────────────────────────

    def compute_bias(
        self,
        gap_pct:        float = 0.0,   # Commodity open gap vs prev close %
        india_vix:      float = 18.0,
        prev_close:     float = 0.0,
        cpr_top:        float = 0.0,   # CPR top (BC)
        cpr_bottom:     float = 0.0,   # CPR bottom (TC)
        fii_net_cr:     float = 0.0,   # FII net in ₹Cr (+ = buying)
        dow_futures_pct:float = 0.0,   # Dow futures change %
        sgx_nifty_pct:  float = 0.0,   # Global futures change %
    ) -> MorningBias:
        """
        Compute morning directional bias from pre-market data.
        Call at 09:00 IST before market opens.
        """
        bull_score = 0.0
        bear_score = 0.0

        # Gap direction (40% weight — strongest predictor)
        if gap_pct > 0.5:
            bull_score += 40
        elif gap_pct > 0.2:
            bull_score += 20
        elif gap_pct < -0.5:
            bear_score += 40
        elif gap_pct < -0.2:
            bear_score += 20

        # VIX (20% weight)
        if india_vix < 14:
            bull_score += 20   # low fear = complacency = possible bull
        elif india_vix < 18:
            bull_score += 10
        elif india_vix > 25:
            bear_score += 20   # high fear = selling likely
        elif india_vix > 20:
            bear_score += 10

        vix_signal = ("EXTREME" if india_vix > 25
                      else "ELEVATED" if india_vix > 20
                      else "FAVOURABLE")

        # CPR position (20% weight)
        cpr_bias = "INSIDE"
        if cpr_top > 0 and cpr_bottom > 0 and prev_close > 0:
            if prev_close > cpr_top:
                bull_score += 20
                cpr_bias    = "ABOVE"
            elif prev_close < cpr_bottom:
                bear_score += 20
                cpr_bias    = "BELOW"
            else:
                cpr_bias    = "INSIDE"

        # FII flow (10% weight)
        fii_signal = "NEUTRAL"
        if fii_net_cr > 1500:
            bull_score += 10
            fii_signal  = "BUYING"
        elif fii_net_cr < -1500:
            bear_score += 10
            fii_signal  = "SELLING"

        # Global markets (10% weight)
        global_score = (dow_futures_pct + sgx_nifty_pct) / 2
        global_bias  = "NEUTRAL"
        if global_score > 0.3:
            bull_score += 10
            global_bias = "BULLISH"
        elif global_score < -0.3:
            bear_score += 10
            global_bias = "BEARISH"

        # Final bias
        net   = bull_score - bear_score
        total = bull_score + bear_score

        if net > 25:
            bias      = "BULLISH"
            strength  = min(100, bull_score)
            direction = "BUY_CALL"
        elif net < -25:
            bias      = "BEARISH"
            strength  = min(100, bear_score)
            direction = "BUY_PUT"
        else:
            bias      = "NEUTRAL"
            strength  = 50.0
            direction = "BOTH"

        note = (
            f"Gap={gap_pct:+.2f}% | VIX={india_vix:.1f}({vix_signal}) | "
            f"CPR={cpr_bias} | FII={fii_signal} | Global={global_bias} | "
            f"Bull={bull_score:.0f} Bear={bear_score:.0f} → {bias}"
        )

        self._bias      = MorningBias(
            bias=bias, strength=strength, direction=direction,
            gap_pct=gap_pct, vix_signal=vix_signal,
            prev_close_vs_cpr=cpr_bias, fii_flow=fii_signal,
            global_bias=global_bias, note=note,
        )
        self._bias_date = date.today().isoformat()

        logger.info(f"[MorningBias] {note}")
        return self._bias

    def get_bias(self) -> Optional[MorningBias]:
        """Return today's morning bias (None if not computed yet)."""
        if self._bias_date != date.today().isoformat():
            return None
        return self._bias

    def is_signal_aligned_with_bias(self, direction: str) -> tuple[bool, float]:
        """
        Check if signal direction aligns with morning bias.
        Returns (aligned, confidence_modifier).
        """
        bias = self.get_bias()
        if not bias or bias.bias == "NEUTRAL":
            return True, 0.0

        if bias.direction == direction:
            mod = bias.strength / 100 * 0.06   # up to +0.06 conf boost
            return True, round(mod, 4)
        elif bias.direction != "BOTH" and bias.direction != direction:
            mod = -(bias.strength / 100 * 0.04)  # up to -0.04 penalty
            return False, round(mod, 4)
        return True, 0.0

    # ── SESSION TIME WINDOW ───────────────────────────────────────────────────

    def get_session_window(self, t: Optional[dtime] = None) -> SessionWindow:
        """Get current trading session window with size and conf adjustments."""
        now = t or datetime.now(IST).time()
        for start, end, label, size_mult, min_conf, note in SESSION_WINDOWS:
            if start <= now < end:
                return SessionWindow(
                    label        = label,
                    size_mult    = size_mult,
                    min_conf     = min_conf,
                    is_tradeable = size_mult > 0,
                    note         = note,
                )
        return SessionWindow("CLOSED", 0.0, 0.99, False, "Market closed")

    # ── ADAPTIVE GATE LOOSENING ───────────────────────────────────────────────

    def get_adaptive_gates(
        self,
        adx:         float,
        regime:      str,
        session:     Optional[SessionWindow] = None,
        bias:        Optional[MorningBias]   = None,
    ) -> AdaptiveGates:
        """
        Dynamically compute gate thresholds based on current conditions.

        LOOSEN gates when:
          - Strong trend (ADX > 30)
          - Morning bias aligns strongly
          - Prime session window (10:00-11:30)
          - Regime = TRENDING

        TIGHTEN gates when:
          - Weak trend (ADX < 18)
          - Weak session window (after 13:00)
          - Bias is NEUTRAL
          - Regime = RANGING/CHOPPY
        """
        sess   = session or self.get_session_window()
        b      = bias    or self.get_bias()
        bias_strong = b and b.strength > 60 and b.bias != "NEUTRAL"

        # Start with defaults
        votes  = MIN_STRATEGY_VOTES
        ml_min = ML_MIN_CONFIDENCE
        det_min= ENTRY_MIN_DET_CONF
        adx_thr= ADX_TREND_THRESHOLD
        loosened = False
        reasons  = []

        # ── LOOSEN conditions ─────────────────────────────────────────────────
        if adx > 30 and regime == "TRENDING":
            votes   = max(1, votes - 1)      # only 1 vote needed in strong trend
            ml_min  = max(0.32, ml_min - 0.08)
            det_min = max(0.50, det_min - 0.12)
            loosened = True
            reasons.append(f"ADX={adx:.0f}>30+TRENDING")

        elif adx > 22 and regime == "TRENDING":
            votes   = max(2, votes)
            ml_min  = max(0.32, ml_min - 0.06)
            det_min = max(0.55, det_min - 0.07)
            loosened = True
            reasons.append(f"ADX={adx:.0f}>22")

        if bias_strong:
            ml_min   = max(0.32, ml_min - 0.04)
            loosened = True
            reasons.append(f"strong_bias={b.bias}")

        if sess.label in ("PRIME", "EVENING_PRIME", "MORNING_ACTIVE", "MIDDAY_ACTIVE"):
            ml_min   = max(0.30, ml_min - 0.02)
            loosened = True
            reasons.append(f"{sess.label}")

        # ── TIGHTEN conditions ────────────────────────────────────────────────
        if adx < 15 or regime in ("RANGING", "CHOPPY"):
            votes   = min(3, votes + 1)
            ml_min  = min(0.50, ml_min + 0.08)
            det_min = min(0.75, det_min + 0.10)
            reasons.append(f"ADX={adx:.0f}<15/RANGING")

        if sess.label in ("WEAK", "LAST_CHANCE", "LATE_SESSION"):
            ml_min  = min(0.55, ml_min + 0.06)
            reasons.append(f"{sess.label}")

        if sess.min_conf > 0:
            ml_min = max(ml_min, sess.min_conf)

        reason = " | ".join(reasons) if reasons else "defaults"

        return AdaptiveGates(
            min_votes          = votes,
            ml_min_confidence  = round(ml_min, 4),
            entry_min_det_conf = round(det_min, 4),
            adx_threshold      = adx_thr,
            reason             = reason,
            loosened           = loosened,
        )

    def get_daily_signal_budget(self) -> dict:
        """
        Expected signal budget for today based on session time windows.
        Shows how many quality signals to expect.
        """
        windows = {
            "OPENING_RANGE": {"duration_min": 45,  "signal_rate": 0.0},
            "PRIME":         {"duration_min": 90,  "signal_rate": 0.8},
            "SECONDARY":     {"duration_min": 90,  "signal_rate": 0.5},
            "WEAK":          {"duration_min": 60,  "signal_rate": 0.3},
            "LAST_CHANCE":   {"duration_min": 30,  "signal_rate": 0.15},
        }
        total_expected = sum(
            w["duration_min"] / 30 * w["signal_rate"]   # per 30-min bucket
            for w in windows.values()
        )
        return {
            "expected_signals_per_day": round(total_expected, 1),
            "best_window":              "PRIME (10:00-11:30 IST)",
            "windows":                  windows,
            "note": "Signal rate assumes strong trend day. Flat days = fewer signals.",
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_bias_system: MorningBiasSystem | None = None

def get_morning_bias_system() -> MorningBiasSystem:
    global _bias_system
    if _bias_system is None:
        _bias_system = MorningBiasSystem()
    return _bias_system

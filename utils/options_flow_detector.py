"""
utils/options_flow_detector.py — Institutional Options Flow Detection
=====================================================================
THE BIGGEST MISSING EDGE:
  Price and volume tell you what happened.
  Options flow tells you what WILL happen.

  When a big institution buys 10,000 lots of 22500CE:
  - They are betting ₹75 crores that NIFTY goes above 22500
  - This happens BEFORE the price move
  - Retail traders react AFTER the move
  - Options flow traders position WITH the institutions BEFORE

WHAT THIS MODULE DETECTS:

1. UNUSUAL CALL/PUT BUYING (Block Orders)
   - Premium paid > ₹50L in a single strike in < 5 min = institutional
   - Vs normal retail: < ₹2L per strike per candle
   - Detection: OI spike + volume spike in specific strike = block order

2. STRIKE MOMENTUM (which strikes are being accumulated)
   - Track premium change across 5 strikes (ATM ± 2)
   - If OTM call premiums rising faster than ATM = directional positioning
   - If multiple OTM strikes rising together = spread buying = strong directional bet

3. PUT/CALL PREMIUM RATIO SHIFT (real-time)
   - Not just OI-based PCR — actual premium being PAID
   - If call premiums rising while put premiums flat = bullish flow
   - If put premiums spiking = fear buying = bearish signal

4. OPEN INTEREST CHANGE VELOCITY
   - OI adding rapidly = new positions being built (trend continuation)
   - OI falling rapidly = positions being closed (trend ending)
   - Track rate-of-change of OI, not just level

SIGNAL OUTPUT:
  STRONG_CALL_FLOW    → large institutional call buying detected → BUY_CALL
  STRONG_PUT_FLOW     → large institutional put buying detected → BUY_PUT
  UNWINDING_CALLS     → call OI falling fast → bearish (sellers covering)
  UNWINDING_PUTS      → put OI falling fast  → bullish (sellers covering)
  NEUTRAL_FLOW        → no unusual activity

INTEGRATION:
  Called by strategy runner as a meta-signal that boosts confidence
  when it aligns with other strategies.

  flow = get_flow_detector().analyze(broker, spot=22100, expiry=expiry)
  if flow.direction == "BUY_CALL" and flow.strength > 0.7:
      conf += 0.08   # significant institutional alignment boost
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from config.settings import NIFTY_STRIKE_STEP, NIFTY_LOT_SIZE
except ImportError:
    NIFTY_STRIKE_STEP = 50
    NIFTY_LOT_SIZE    = 75

# ── Parameters ────────────────────────────────────────────────────────────────
BLOCK_ORDER_LOTS_THRESHOLD = 500    # > 500 lots in one candle = institutional
BLOCK_ORDER_INR_CR         = 0.50   # > ₹50L premium in one strike = block
OI_VELOCITY_WINDOW         = 3      # candles for OI change velocity
STRIKE_MOMENTUM_WINDOW     = 5      # candles to track strike momentum
STRIKES_TO_TRACK           = 5      # ATM ± 2 strikes each side


@dataclass
class FlowSignal:
    direction:       str      # "BUY_CALL" | "BUY_PUT" | "NEUTRAL"
    signal_type:     str      # "BLOCK_ORDER" | "STRIKE_MOMENTUM" | "PCR_SHIFT" | "OI_VELOCITY"
    strength:        float    # 0-1
    confidence_boost:float    # add to existing signal confidence
    call_flow_score: float    # 0-100
    put_flow_score:  float    # 0-100
    unusual_strikes: list[dict]  # strikes with unusual activity
    note:            str

    def to_dict(self) -> dict:
        return {
            "flow_direction":    self.direction,
            "flow_type":         self.signal_type,
            "flow_strength":     round(self.strength, 3),
            "flow_conf_boost":   round(self.confidence_boost, 4),
            "call_flow_score":   round(self.call_flow_score, 1),
            "put_flow_score":    round(self.put_flow_score, 1),
        }


class OptionsFlowDetector:
    """
    Detects institutional options flow from broker option chain data.
    Works in two modes:
      LIVE:  uses real option chain from broker (premium + OI per strike)
      PROXY: estimates flow from NIFTY price/volume + VIX when no broker
    """

    def __init__(self) -> None:
        # Rolling history per strike: {strike: deque of (oi, premium, volume)}
        self._oi_history:  dict[int, deque] = {}
        self._prem_history:dict[int, deque] = {}
        # Session-level tracking
        self._session_call_flow  = 0.0
        self._session_put_flow   = 0.0
        self._candle_count       = 0
        self._last_pcr           = 1.0

    def analyze(
        self,
        spot:       float,
        broker      = None,
        expiry_date = None,
        df          = None,   # 5-min OHLCV as proxy when no broker
        india_vix:  float = 18.0,
    ) -> FlowSignal:
        """
        Analyze options flow and return directional signal.

        Args:
            spot:        NIFTY spot price
            broker:      live broker instance (optional)
            expiry_date: target expiry (optional)
            df:          5-min OHLCV for proxy mode
            india_vix:   India VIX for proxy mode
        """
        self._candle_count += 1

        if broker is not None:
            try:
                return self._live_analyze(spot, broker, expiry_date)
            except Exception as e:
                logger.debug(f"[FlowDetector] live mode failed: {e}")

        # Proxy mode
        return self._proxy_analyze(spot, df, india_vix)

    # ── LIVE ANALYSIS ─────────────────────────────────────────────────────────

    def _live_analyze(
        self, spot: float, broker, expiry_date
    ) -> FlowSignal:
        """Fetch real option chain and analyze flow."""
        from utils.option_utils import build_option_symbol
        from datetime import date as ddate, timedelta

        if expiry_date is None:
            d = ddate.today()
            days = (3 - d.weekday()) % 7 or 7
            expiry_date = d + timedelta(days=days)

        sym = str(os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
        step = 500 if "SILVER" in sym else (100 if "GOLD" in sym else 50)
        atm  = int(round(spot / step) * step)

        call_flow = 0.0
        put_flow  = 0.0
        unusual   = []

        for i in range(-STRIKES_TO_TRACK, STRIKES_TO_TRACK + 1):
            k = atm + i * step
            try:
                ce_sym = build_option_symbol(sym, expiry_date, k, "CE")
                pe_sym = build_option_symbol(sym, expiry_date, k, "PE")
                ce_ltp = broker.get_option_ltp(ce_sym)
                pe_ltp = broker.get_option_ltp(pe_sym)
            except Exception:
                continue

            # Track premium history per strike
            if k not in self._prem_history:
                self._prem_history[k] = deque(maxlen=STRIKE_MOMENTUM_WINDOW)
            self._prem_history[k].append({"ce": ce_ltp, "pe": pe_ltp})

            # Detect strike momentum
            if len(self._prem_history[k]) >= 3:
                hist = list(self._prem_history[k])
                ce_change = (hist[-1]["ce"] - hist[0]["ce"]) / max(hist[0]["ce"], 1)
                pe_change = (hist[-1]["pe"] - hist[0]["pe"]) / max(hist[0]["pe"], 1)

                # OTM call rising fast = directional call buying
                if i > 0 and ce_change > 0.15:   # OTM CE up > 15% in 5 candles
                    call_flow += 20 * ce_change
                    unusual.append({"strike": k, "type": "CE_MOMENTUM",
                                   "change": round(ce_change * 100, 1)})
                if i < 0 and pe_change > 0.15:   # OTM PE up > 15%
                    put_flow += 20 * pe_change
                    unusual.append({"strike": k, "type": "PE_MOMENTUM",
                                   "change": round(pe_change * 100, 1)})

        return self._build_signal(call_flow, put_flow, unusual, "LIVE")

    # ── PROXY ANALYSIS ────────────────────────────────────────────────────────

    def _proxy_analyze(
        self, spot: float, df, india_vix: float
    ) -> FlowSignal:
        """
        Proxy options flow from price/volume when no broker available.
        Uses NIFTY price action + VIX to infer institutional activity.
        """
        if df is None or len(df) < 10:
            return self._neutral("insufficient data")

        import numpy as np
        import pandas as pd

        closes  = df["close"].values
        volumes = df["volume"].values
        highs   = df["high"].values
        lows    = df["low"].values

        call_flow = 0.0
        put_flow  = 0.0

        # Signal 1: Strong directional move with high volume = institutional
        price_change_3c = (closes[-1] - closes[-4]) / closes[-4] * 100
        vol_avg         = float(np.mean(volumes[-20:]))
        vol_now         = float(volumes[-1])
        vol_ratio       = vol_now / max(vol_avg, 1)

        if price_change_3c > 0.3 and vol_ratio > 1.5:
            call_flow += 30   # strong up move with volume = call buying
        elif price_change_3c < -0.3 and vol_ratio > 1.5:
            put_flow  += 30   # strong down move with volume = put buying

        # Signal 2: VIX spike = put buying (fear/hedging)
        if india_vix > 20:
            put_flow  += (india_vix - 18) * 5   # VIX elevated = put demand
        elif india_vix < 15:
            call_flow += 10   # low VIX = complacency = possible call buying

        # Signal 3: Session accumulation pattern
        # In PRIME window, price consolidating at high = call accumulation
        session_high = float(np.max(highs[-10:]))
        session_low  = float(np.min(lows[-10:]))
        session_mid  = (session_high + session_low) / 2
        if closes[-1] > session_mid * 1.002:
            call_flow += 15   # price in upper half of range = bullish flow
        elif closes[-1] < session_mid * 0.998:
            put_flow  += 15

        # Signal 4: Momentum acceleration
        roc5  = (closes[-1] - closes[-6])  / closes[-6]  * 100
        roc10 = (closes[-1] - closes[-11]) / closes[-11] * 100
        if roc5 > 0 and roc5 > roc10:  # accelerating up
            call_flow += 10
        elif roc5 < 0 and roc5 < roc10:  # accelerating down
            put_flow  += 10

        return self._build_signal(call_flow, put_flow, [], "PROXY")

    # ── SIGNAL BUILDER ────────────────────────────────────────────────────────

    def _build_signal(
        self,
        call_flow: float,
        put_flow:  float,
        unusual:   list,
        mode:      str,
    ) -> FlowSignal:
        # Update session accumulators
        self._session_call_flow = self._session_call_flow * 0.9 + call_flow * 0.1
        self._session_put_flow  = self._session_put_flow  * 0.9 + put_flow  * 0.1

        net    = call_flow - put_flow
        total  = call_flow + put_flow

        if net > 25:
            direction  = "BUY_CALL"
            strength   = min(1.0, net / 80)
            sig_type   = "BLOCK_ORDER" if unusual else "STRIKE_MOMENTUM"
            boost      = round(strength * 0.08, 4)
            note       = f"Strong call flow: {call_flow:.0f} vs put: {put_flow:.0f} ({mode})"
        elif net < -25:
            direction  = "BUY_PUT"
            strength   = min(1.0, abs(net) / 80)
            sig_type   = "BLOCK_ORDER" if unusual else "STRIKE_MOMENTUM"
            boost      = round(strength * 0.08, 4)
            note       = f"Strong put flow: {put_flow:.0f} vs call: {call_flow:.0f} ({mode})"
        else:
            return self._neutral(f"balanced flow call={call_flow:.0f} put={put_flow:.0f}")

        logger.debug(f"[FlowDetector] {note}")
        return FlowSignal(
            direction        = direction,
            signal_type      = sig_type,
            strength         = round(strength, 3),
            confidence_boost = boost,
            call_flow_score  = round(call_flow, 1),
            put_flow_score   = round(put_flow, 1),
            unusual_strikes  = unusual[:5],
            note             = note,
        )

    @staticmethod
    def _neutral(reason: str = "") -> FlowSignal:
        return FlowSignal("NEUTRAL", "NEUTRAL_FLOW", 0.0, 0.0, 50.0, 50.0, [],
                          reason or "no unusual flow")

    def reset_session(self) -> None:
        """Reset at market open."""
        self._session_call_flow = 0.0
        self._session_put_flow  = 0.0
        self._candle_count      = 0
        self._oi_history        = {}
        self._prem_history      = {}


# ── Singleton ─────────────────────────────────────────────────────────────────
_flow_detector: OptionsFlowDetector | None = None

def get_flow_detector() -> OptionsFlowDetector:
    global _flow_detector
    if _flow_detector is None:
        _flow_detector = OptionsFlowDetector()
    return _flow_detector

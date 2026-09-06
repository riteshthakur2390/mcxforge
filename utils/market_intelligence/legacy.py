"""
utils/market_intelligence/legacy.py — Market Breadth, PCR, VWAP Deviation, VIX Trend
===============================================================================
Legacy MarketIntelligence class and indicators preserved for TradePlannerAgent compatibility.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
import pandas as pd
import numpy as np
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ── Parameters ────────────────────────────────────────────────────────────────
BREADTH_BULL_THRESHOLD   = 0.65   # > 65% advancing = bullish breadth
BREADTH_BEAR_THRESHOLD   = 0.35   # < 35% advancing = bearish breadth
PCR_BULL_THRESHOLD       = 1.3    # PCR > 1.3 = contrarian bullish
PCR_BEAR_THRESHOLD       = 0.7    # PCR < 0.7 = contrarian bearish
VWAP_SIGMA_EXIT          = 2.0    # exit if price > 2σ from VWAP
VIX_TREND_DAYS           = 3      # VIX trend over last N days


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BreadthSignal:
    advance_pct:   float    # % of stocks advancing
    decline_pct:   float    # % of stocks declining
    adv_dec_ratio: float    # advances / declines
    bias:          str      # "BULLISH" | "BEARISH" | "NEUTRAL"
    strength:      float    # 0-1 signal strength
    note:          str
    allow_calls:   bool
    allow_puts:    bool


@dataclass
class PCRSignal:
    pcr:           float    # current PCR
    pcr_5d_avg:    float    # 5-day average PCR
    bias:          str      # "BULLISH" | "BEARISH" | "NEUTRAL"
    strength:      float
    note:          str


@dataclass
class VWAPCheck:
    current_price: float
    vwap:          float
    deviation_pct: float    # % deviation from VWAP
    std_dev:       float    # VWAP std dev
    sigma:         float    # how many sigma from VWAP
    is_extended:   bool     # True if > VWAP_SIGMA_EXIT
    direction:     str      # "ABOVE" | "BELOW"
    reverting:     bool     # True if price moving back toward VWAP
    should_exit:   bool     # True if extended + reverting


@dataclass
class VIXTrend:
    current_vix:   float
    vix_3d_ago:    float
    trend:         str      # "RISING" | "FALLING" | "FLAT"
    slope:         float    # daily change
    advice:        str      # "AVOID_BUYING" | "GOOD_FOR_BUYERS" | "NEUTRAL"
    note:          str


# ═════════════════════════════════════════════════════════════════════════════
# 1. MARKET BREADTH
# ═════════════════════════════════════════════════════════════════════════════

class MarketBreadthMonitor:
    """Tracks NSE advance-decline ratio as a macro filter."""

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=20)

    def update(self, advances: int, declines: int, unchanged: int = 0) -> BreadthSignal:
        total    = advances + declines + unchanged
        if total == 0:
            return self._neutral()

        adv_pct  = advances / total
        dec_pct  = declines / total
        adr      = advances / max(declines, 1)

        self._history.append(adv_pct)

        if adv_pct > BREADTH_BULL_THRESHOLD:
            bias       = "BULLISH"
            strength   = min(1.0, (adv_pct - 0.5) * 2)
            note       = f"{adv_pct:.0%} stocks advancing — broad strength"
            allow_calls= True
            allow_puts = False
        elif adv_pct < BREADTH_BEAR_THRESHOLD:
            bias       = "BEARISH"
            strength   = min(1.0, (0.5 - adv_pct) * 2)
            note       = f"{dec_pct:.0%} stocks declining — broad weakness"
            allow_calls= False
            allow_puts = True
        else:
            bias       = "NEUTRAL"
            strength   = 0.0
            note       = f"Mixed breadth: {adv_pct:.0%} adv / {dec_pct:.0%} dec"
            allow_calls= True
            allow_puts = True

        return BreadthSignal(
            advance_pct   = round(adv_pct, 3),
            decline_pct   = round(dec_pct, 3),
            adv_dec_ratio = round(adr, 2),
            bias          = bias,
            strength      = round(strength, 3),
            note          = note,
            allow_calls   = allow_calls,
            allow_puts    = allow_puts,
        )

    def from_nse_api(self) -> BreadthSignal:
        try:
            import requests
            resp = requests.get(
                "https://www.nseindia.com/api/market-status",
                headers={"User-Agent": "Mozilla/5.0",
                         "Accept": "application/json"},
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                mkt  = data.get("marketState", [])
                for m in mkt:
                    if "advance" in str(m).lower():
                        adv = int(m.get("advance", m.get("advances", 0)))
                        dec = int(m.get("decline", m.get("declines", 0)))
                        if adv + dec > 0:
                            return self.update(adv, dec)
        except Exception as e:
            logger.debug(f"[MarketBreadth] NSE API: {e}")
        return self._neutral()

    @staticmethod
    def _neutral() -> BreadthSignal:
        return BreadthSignal(0.5, 0.5, 1.0, "NEUTRAL", 0.0,
                             "Data unavailable", True, True)


# ═════════════════════════════════════════════════════════════════════════════
# 2. PUT-CALL RATIO
# ═════════════════════════════════════════════════════════════════════════════

class PCRMonitor:
    """Tracks NIFTY Put-Call Ratio as a contrarian signal."""

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=10)

    def compute(self, total_ce_oi: float, total_pe_oi: float) -> PCRSignal:
        if total_ce_oi <= 0:
            return self._neutral()

        pcr = round(total_pe_oi / total_ce_oi, 3)
        self._history.append(pcr)
        avg_5d = round(sum(list(self._history)[-5:]) / min(len(self._history), 5), 3)

        if pcr > PCR_BULL_THRESHOLD:
            bias   = "BULLISH"
            strength = min(1.0, (pcr - 1.0) / 0.5)
            note   = f"PCR={pcr:.2f} — excessive put buying, contrarian bullish"
        elif pcr < PCR_BEAR_THRESHOLD:
            bias   = "BEARISH"
            strength = min(1.0, (1.0 - pcr) / 0.3)
            note   = f"PCR={pcr:.2f} — call heavy, complacency = bearish"
        else:
            bias   = "NEUTRAL"
            strength = 0.0
            note   = f"PCR={pcr:.2f} — balanced options market"

        return PCRSignal(
            pcr       = pcr,
            pcr_5d_avg= avg_5d,
            bias      = bias,
            strength  = round(strength, 3),
            note      = note,
        )

    def from_nse_api(self) -> PCRSignal:
        try:
            import requests
            resp = requests.get(
                "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY",
                headers={"User-Agent": "Mozilla/5.0",
                         "Accept": "application/json",
                         "Referer": "https://www.nseindia.com/"},
                timeout=10,
            )
            if resp.status_code == 200:
                data     = resp.json()
                filtered = data.get("filtered", {})
                ce_oi    = float(filtered.get("CE", {}).get("totOI", 0))
                pe_oi    = float(filtered.get("PE", {}).get("totOI", 0))
                if ce_oi + pe_oi > 0:
                    return self.compute(ce_oi, pe_oi)
        except Exception as e:
            logger.debug(f"[PCRMonitor] NSE API: {e}")
        return self._neutral()

    @staticmethod
    def _neutral() -> PCRSignal:
        return PCRSignal(1.0, 1.0, "NEUTRAL", 0.0, "PCR data unavailable")


# ═════════════════════════════════════════════════════════════════════════════
# 3. VWAP DEVIATION
# ═════════════════════════════════════════════════════════════════════════════

class VWAPMonitor:
    """Tracks session VWAP and flags when price is too extended."""

    def check(
        self,
        df:        "pd.DataFrame",
        direction: str = "BUY_CALL",
    ) -> VWAPCheck:
        try:
            if len(df) < 5:
                return self._no_action(float(df["close"].iloc[-1]))

            typical  = (df["high"] + df["low"] + df["close"]) / 3
            vol      = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            cum_vol  = vol.cumsum()
            if float(cum_vol.iloc[-1]) <= 0:
                close_fallback = float(df["close"].iloc[-1])
                vwap = pd.Series(close_fallback, index=df.index)
            else:
                vwap = ((typical * vol).cumsum() / cum_vol).ffill().fillna(df["close"])
            vwap_val = float(vwap.iloc[-1])

            price_arr = df["close"].values
            deviations = (price_arr - vwap.values) ** 2
            std_dev   = float(np.sqrt(np.mean(deviations[-20:])))

            close     = float(df["close"].iloc[-1])
            prev_close= float(df["close"].iloc[-2])
            dev_pct   = (close - vwap_val) / vwap_val * 100
            sigma     = abs(close - vwap_val) / max(std_dev, 0.01)

            above     = close > vwap_val
            direction_str = "ABOVE" if above else "BELOW"

            if above:
                reverting = prev_close > close
            else:
                reverting = prev_close < close

            extended   = sigma > VWAP_SIGMA_EXIT
            should_exit = (
                extended and reverting and
                (
                    (direction == "BUY_CALL" and above) or
                    (direction == "BUY_PUT"  and not above)
                )
            )

            return VWAPCheck(
                current_price = round(close, 2),
                vwap          = round(vwap_val, 2),
                deviation_pct = round(dev_pct, 3),
                std_dev       = round(std_dev, 2),
                sigma         = round(sigma, 2),
                is_extended   = extended,
                direction     = direction_str,
                reverting     = reverting,
                should_exit   = should_exit,
            )
        except Exception as e:
            logger.debug(f"[VWAPMonitor] error: {e}")
            return self._no_action(0.0)

    @staticmethod
    def _no_action(price: float) -> VWAPCheck:
        return VWAPCheck(price, price, 0.0, 0.0, 0.0, False, "ABOVE", False, False)


# ═════════════════════════════════════════════════════════════════════════════
# 4. VIX TREND MONITOR
# ═════════════════════════════════════════════════════════════════════════════

class VIXTrendMonitor:
    """Tracks VIX trend direction."""

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=10)

    def update(self, current_vix: float) -> VIXTrend:
        self._history.append(current_vix)

        if len(self._history) < VIX_TREND_DAYS:
            return VIXTrend(
                current_vix = current_vix,
                vix_3d_ago  = current_vix,
                trend       = "FLAT",
                slope       = 0.0,
                advice      = "NEUTRAL",
                note        = "Insufficient VIX history",
            )

        hist     = list(self._history)
        vix_ago  = hist[-VIX_TREND_DAYS]
        slope    = (current_vix - vix_ago) / VIX_TREND_DAYS

        if slope > 0.5:
            trend  = "RISING"
            advice = "AVOID_BUYING"
            note   = f"VIX rising +{slope:.1f}/day — volatility expanding, options expensive"
        elif slope < -0.5:
            trend  = "FALLING"
            advice = "GOOD_FOR_BUYERS"
            note   = f"VIX falling {slope:.1f}/day — volatility contracting, good entry timing"
        else:
            trend  = "FLAT"
            advice = "NEUTRAL"
            note   = f"VIX stable ({current_vix:.1f}) — no directional vol bias"

        return VIXTrend(
            current_vix = round(current_vix, 2),
            vix_3d_ago  = round(vix_ago, 2),
            trend       = trend,
            slope       = round(slope, 3),
            advice      = advice,
            note        = note,
        )


# ── Composite market intelligence ─────────────────────────────────────────────

class MarketIntelligence:
    """Aggregates legacy market intelligence signals for planner compatibility."""

    def __init__(self) -> None:
        self.breadth = MarketBreadthMonitor()
        self.pcr     = PCRMonitor()
        self.vwap    = VWAPMonitor()
        self.vix     = VIXTrendMonitor()

    def should_trade(
        self,
        direction:   str,
        df:          "pd.DataFrame",
        india_vix:   float,
        ce_oi:       float = 0,
        pe_oi:       float = 0,
        advances:    int   = 0,
        declines:    int   = 0,
    ) -> dict:
        breadth_sig = self.breadth.update(advances, declines) if advances + declines > 0 \
                      else BreadthSignal(0.5, 0.5, 1.0, "NEUTRAL", 0.0, "No breadth data", True, True)
        pcr_sig     = self.pcr.compute(ce_oi, pe_oi) if ce_oi + pe_oi > 0 \
                      else PCRSignal(1.0, 1.0, "NEUTRAL", 0.0, "No PCR data")
        vwap_chk    = self.vwap.check(df, direction)
        vix_trend   = self.vix.update(india_vix)

        blocks = []
        if direction == "BUY_CALL" and not breadth_sig.allow_calls:
            blocks.append(f"breadth bearish ({breadth_sig.advance_pct:.0%} adv)")
        if direction == "BUY_PUT" and not breadth_sig.allow_puts:
            blocks.append(f"breadth bullish ({breadth_sig.advance_pct:.0%} adv)")
        if vwap_chk.should_exit:
            blocks.append(f"VWAP extended {vwap_chk.sigma:.1f}σ + reverting")
        if vix_trend.advice == "AVOID_BUYING":
            blocks.append(f"VIX rising ({vix_trend.slope:+.1f}/day)")

        allowed = len(blocks) == 0

        return {
            "allowed":   allowed,
            "blocks":    blocks,
            "breadth":   breadth_sig.__dict__,
            "pcr":       pcr_sig.__dict__,
            "vwap":      vwap_chk.__dict__,
            "vix_trend": vix_trend.__dict__,
            "summary":   "GO" if allowed else f"NO-GO: {', '.join(blocks)}",
        }

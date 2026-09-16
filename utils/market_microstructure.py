"""
utils/market_microstructure.py — FII/DII Flow, OI Buildup, IV Term Structure
=============================================================================
Three institutional-level market intelligence signals in one module.

1. FII/DII FLOW SIGNAL
   NSE publishes FII and DII net buy/sell data daily at ~17:00 IST.
   This is forward-looking institutional intent signal used by every prop desk.

   Logic:
     FII net sellers > ₹2000 Cr AND DII net buyers < ₹1500 Cr → BEAR bias
     FII net buyers  > ₹1500 Cr AND DII net sellers             → BULL bias
     FII net sellers AND DII net buyers (both covering)          → NEUTRAL

2. OI BUILDUP PATTERN
   Tracks Open Interest changes across strikes to find:
     - Support levels (heavy PE writing = market expects this level to hold)
     - Resistance levels (heavy CE writing = market expects cap here)
     - Unwinding (OI falling = positions being closed)
     - Buildup (OI rising = new positions being added)

3. IV TERM STRUCTURE
   Measures the spread between near-week and next-week IV.
   When near IV >> far IV: event expected → option buying expensive
   When near IV ≈ far IV: calm → buying is cheaper → better R:R

Usage:
    from utils.market_microstructure import get_fii_signal, get_oi_buildup, get_iv_term

    # Morning (after 09:00)
    fii = get_fii_signal()   # → {"bias": "BEARISH", "fii_net": -2340}

    # At signal time
    oi  = get_oi_buildup(broker, spot=22150)  # → {"support": 22000, "resistance": 22500}

    # IV structure check before buying
    iv  = get_iv_term(broker, spot=22150)     # → {"spread": 0.03, "bias": "avoid_buying"}
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import JOURNAL_DIR
except ImportError:
    JOURNAL_DIR = "journal"

CACHE_DIR = Path(JOURNAL_DIR) / "microstructure_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1. FII / DII FLOW SIGNAL
# ═════════════════════════════════════════════════════════════════════════════

def get_fii_signal(
    fii_net_cr:  Optional[float] = None,
    dii_net_cr:  Optional[float] = None,
    use_cache:   bool = True,
) -> dict:
    """
    Compute institutional flow bias from FII/DII data.

    Args:
        fii_net_cr: FII net (positive = buying, negative = selling) in ₹ Cr
                    If None, tries to fetch from NSE or use yesterday's cache.
        dii_net_cr: DII net in ₹ Cr (DII = domestic institutions, counter FII)
        use_cache:  use yesterday's data if today's not available

    Returns:
        {bias, fii_net, dii_net, combined_flow, signal_strength, note}
    """
    # Try to get today's data
    if fii_net_cr is None:
        cached = _load_fii_cache()
        if cached:
            fii_net_cr = cached.get("fii_net", 0)
            dii_net_cr = cached.get("dii_net", 0)
        else:
            # Try NSE API (bhav copy)
            fii_net_cr, dii_net_cr = _fetch_fii_nse()

    fii_net_cr = fii_net_cr or 0
    dii_net_cr = dii_net_cr or 0
    combined   = fii_net_cr + dii_net_cr

    # Thresholds
    BEAR_FII_SELL = -2000   # FII selling > ₹2000 Cr = strong institutional exit
    BULL_FII_BUY  = 1500    # FII buying  > ₹1500 Cr = institutional accumulation
    DII_COVER_MIN = 500     # DII buying < ₹500 Cr = not covering FII selling

    if fii_net_cr < BEAR_FII_SELL and dii_net_cr < abs(fii_net_cr) * 0.7:
        bias     = "BEARISH"
        strength = min(1.0, abs(fii_net_cr) / 5000)
        note     = f"FII sold ₹{abs(fii_net_cr):.0f}Cr, DII not absorbing fully"
    elif fii_net_cr > BULL_FII_BUY:
        bias     = "BULLISH"
        strength = min(1.0, fii_net_cr / 5000)
        note     = f"FII bought ₹{fii_net_cr:.0f}Cr — institutional accumulation"
    else:
        bias     = "NEUTRAL"
        strength = 0.0
        note     = "No dominant flow signal"

    result = {
        "date":            date.today().isoformat(),
        "bias":            bias,
        "signal_strength": round(strength, 3),
        "fii_net_cr":      fii_net_cr,
        "dii_net_cr":      dii_net_cr,
        "combined_cr":     combined,
        "note":            note,
        "preferred_direction": (
            "BUY_PUT"  if bias == "BEARISH"
            else "BUY_CALL" if bias == "BULLISH"
            else "NONE"
        ),
    }
    _save_fii_cache(result)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 2. OI BUILDUP PATTERN
# ═════════════════════════════════════════════════════════════════════════════

def get_oi_buildup(
    broker,
    spot:          float,
    expiry_date:   str   = None,
    strikes_range: int   = 500,  # ATM ± 500 pts
) -> dict:
    """
    Analyze OI distribution to find support and resistance levels.

    Returns:
        {support, resistance, pcr, max_pain, oi_bias, strong_support, strong_resistance}
    """
    from utils.option_utils import build_option_symbol
    from datetime import date as ddate, timedelta

    if expiry_date is None:
        d = ddate.today()
        days = (3 - d.weekday()) % 7 or 7
        expiry_date = (d + timedelta(days=days)).isoformat()
        expiry = ddate.fromisoformat(expiry_date)

    sym = str(os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
    step = 500 if "SILVER" in sym else (100 if "GOLD" in sym else 50)
    atm  = int(round(spot / step) * step)
    n    = strikes_range // step

    ce_oi: dict[int, float] = {}
    pe_oi: dict[int, float] = {}

    # Fetch OI for each strike
    for i in range(-n, n+1):
        k = atm + i * step
        try:
            ce_sym = build_option_symbol(sym, expiry, k, "CE")
            pe_sym = build_option_symbol(sym, expiry, k, "PE")
            ce_ltp = broker.get_option_ltp(ce_sym)
            pe_ltp = broker.get_option_ltp(pe_sym)
            # Use LTP as OI proxy (real OI needs broker API extension)
            if ce_ltp > 0:
                ce_oi[k] = ce_ltp
            if pe_ltp > 0:
                pe_oi[k] = pe_ltp
        except Exception:
            continue

    if not ce_oi or not pe_oi:
        return _oi_proxy(spot, atm)

    # PCR (Put-Call Ratio)
    total_ce = sum(ce_oi.values())
    total_pe = sum(pe_oi.values())
    pcr = round(total_pe / max(total_ce, 0.01), 3)

    # Max pain: strike where option sellers lose least
    max_pain = _compute_max_pain(ce_oi, pe_oi, spot)

    # Support = strike with highest PE OI (put writers defending this level)
    support = max(pe_oi, key=pe_oi.get) if pe_oi else atm - 200

    # Resistance = strike with highest CE OI (call writers capping this level)
    resistance = max(ce_oi, key=ce_oi.get) if ce_oi else atm + 200

    # OI bias
    if pcr > 1.2:
        oi_bias = "BULLISH"   # more puts = hedging = smart money bullish
    elif pcr < 0.8:
        oi_bias = "BEARISH"   # more calls = speculation = capped upside
    else:
        oi_bias = "NEUTRAL"

    return {
        "spot":             round(spot, 2),
        "atm":              atm,
        "support":          support,
        "resistance":       resistance,
        "pcr":              pcr,
        "max_pain":         max_pain,
        "oi_bias":          oi_bias,
        "total_ce_proxy":   round(total_ce, 2),
        "total_pe_proxy":   round(total_pe, 2),
        "note":             f"Support={support} Resistance={resistance} PCR={pcr}",
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. IV TERM STRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

def get_iv_term_structure(
    broker,
    spot:       float,
    hv_proxy:   float = 0.15,
) -> dict:
    """
    Measure IV spread between near-week and next-week expiry.
    When near IV >> far IV → event expected → avoid buying expensive options.
    When near IV ≈ far IV → calm → better R:R for option buyers.

    Returns:
        {near_iv, far_iv, spread, bias, recommendation}
    """
    from datetime import date as ddate, timedelta
    from utils.option_utils import build_option_symbol
    from backtesting.options_backtester import _iv_from_price

    d     = ddate.today()
    days1 = (3 - d.weekday()) % 7 or 7
    days2 = days1 + 7
    exp1  = d + timedelta(days=days1)
    exp2  = d + timedelta(days=days2)
    sym   = str(os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
    step  = 500 if "SILVER" in sym else (100 if "GOLD" in sym else 50)
    atm   = int(round(spot / step) * step)

    near_iv = hv_proxy
    far_iv  = hv_proxy

    try:
        T1 = max(days1, 1) / 365.0
        T2 = max(days2, 1) / 365.0

        ce1_sym = build_option_symbol(sym, exp1, atm, "CE")
        ce2_sym = build_option_symbol(sym, exp2, atm, "CE")

        ltp1 = broker.get_option_ltp(ce1_sym)
        ltp2 = broker.get_option_ltp(ce2_sym)

        if ltp1 > 5:
            near_iv = _iv_from_price(spot, atm, T1, 0.065, ltp1, "CE")
        if ltp2 > 5:
            far_iv  = _iv_from_price(spot, atm, T2, 0.065, ltp2, "CE")
    except Exception:
        pass

    spread = near_iv - far_iv   # positive = near more expensive

    if spread > 0.05:
        bias = "EVENT_EXPECTED"
        rec  = "AVOID_BUYING — near-week IV elevated, options expensive"
    elif spread < -0.02:
        bias = "INVERTED"
        rec  = "CAUTION — unusual structure"
    elif spread < 0.02:
        bias = "CALM"
        rec  = "GOOD_TO_BUY — IV flat, options fairly priced"
    else:
        bias = "NORMAL"
        rec  = "NORMAL — standard option pricing"

    return {
        "near_iv":        round(near_iv, 4),
        "far_iv":         round(far_iv, 4),
        "spread":         round(spread, 4),
        "near_expiry":    str(exp1),
        "far_expiry":     str(exp2),
        "bias":           bias,
        "recommendation": rec,
        "iv_rank_proxy":  round(min(near_iv / max(hv_proxy * 1.5, 0.01), 1.0), 3),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_max_pain(ce_oi: dict, pe_oi: dict, spot: float) -> int:
    all_strikes = sorted(set(list(ce_oi.keys()) + list(pe_oi.keys())))
    losses = {}
    for exp_k in all_strikes:
        loss = sum(ce_oi.get(k, 0) * max(exp_k - k, 0) for k in all_strikes)
        loss += sum(pe_oi.get(k, 0) * max(k - exp_k, 0) for k in all_strikes)
        losses[exp_k] = loss
    return min(losses, key=losses.get) if losses else int(round(spot/50)*50)


def _oi_proxy(spot: float, atm: int) -> dict:
    """Return neutral proxy when broker unavailable."""
    return {
        "spot": round(spot, 2), "atm": atm,
        "support": atm - 200, "resistance": atm + 200,
        "pcr": 1.0, "max_pain": atm,
        "oi_bias": "NEUTRAL",
        "total_ce_proxy": 0, "total_pe_proxy": 0,
        "note": "proxy mode — broker unavailable",
    }


def _load_fii_cache() -> Optional[dict]:
    path = CACHE_DIR / f"fii_{date.today().isoformat()}.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_fii_cache(data: dict) -> None:
    path = CACHE_DIR / f"fii_{date.today().isoformat()}.json"
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _fetch_fii_nse() -> tuple[float, float]:
    """Attempt to fetch FII/DII from NSE API. Returns (fii_net, dii_net) in Cr."""
    try:
        import requests
        # NSE publishes daily participant data
        url = "https://www.nseindia.com/api/marketTurnover"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            # Parse FII and DII from response
            fii_net = 0.0
            dii_net = 0.0
            for item in data.get("data", []):
                cat = str(item.get("category", "")).upper()
                if "FII" in cat or "FPI" in cat:
                    fii_net = float(item.get("netAmount", 0))
                elif "DII" in cat:
                    dii_net = float(item.get("netAmount", 0))
            return fii_net, dii_net
    except Exception:
        pass
    return 0.0, 0.0

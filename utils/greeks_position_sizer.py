"""
utils/greeks_position_sizer.py — Delta-Adjusted Position Sizing
===============================================================
Target a fixed delta exposure per trade instead of fixed capital %.

WHY DELTA-ADJUSTED SIZING MATTERS:
  ATM option (delta=0.50) + ITM option (delta=0.75) have very different
  risk profiles even if they cost the same premium.

  ITM option: 1 lot = 75 × 0.75 delta = 56.25 net delta
  ATM option: 1 lot = 75 × 0.50 delta = 37.50 net delta

  If you always buy 1 lot regardless of delta, your risk is inconsistent.
  Delta-adjusted sizing normalises risk across all entries.

TARGET DELTA EXPOSURE:
  Target = 30 delta units per trade (configurable)
  ATM CE (delta=0.50): lots = 30/0.50/75 = 0.8 → 1 lot
  OTM CE (delta=0.25): lots = 30/0.25/75 = 1.6 → 2 lots (more exposure to compensate)
  ITM CE (delta=0.70): lots = 30/0.70/75 = 0.57 → 1 lot (already high delta)

INTEGRATION:
  Used by planner.py to determine lot count based on strike delta.
  Combines with capital_manager for dual constraint:
  1. Lots must not exceed capital budget
  2. Lots sized to match target delta exposure
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from config.settings import (
        NIFTY_LOT_SIZE, SENSEX_LOT_SIZE,
        DEPLOYED_CAPITAL, DAILY_CAPITAL_PCT,
    )
except ImportError:
    NIFTY_LOT_SIZE   = 75
    SENSEX_LOT_SIZE  = 10
    DEPLOYED_CAPITAL = 100_000
    DAILY_CAPITAL_PCT = 15.0

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Target delta units per trade (industry standard: 20-40 for retail)
TARGET_DELTA_UNITS  = 30.0   # normalised delta exposure per trade
MIN_DELTA_THRESHOLD = 0.15   # don't trade if delta < 0.15 (dying option)
MAX_LOTS_DELTA      = 5      # cap from delta sizing


@dataclass
class PositionSize:
    lots:              int
    lot_size:          int
    capital_required:  float
    delta_exposure:    float    # actual delta units
    target_delta:      float
    delta_efficiency:  float    # actual/target (1.0 = perfect)
    method:            str      # "delta" | "capital" | "minimum" | "futures_margin"
    note:              str

    @property
    def net_delta(self) -> float:
        return self.delta_exposure

    @property
    def capital_used(self) -> float:
        return self.capital_required

    @property
    def reason(self) -> str:
        return self.note

    def __getitem__(self, item: str):
        if item in ("desired_lots", "lots"):
            return self.lots
        if item == "quantity":
            return self.lots * self.lot_size
        if item in ("allocated_risk", "capital_required", "capital_used"):
            return self.capital_required
        if item in ("method", "sizing_method"):
            return self.method
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


def compute_bs_delta(
    spot:        float,
    strike:      float,
    dte:         int,
    iv:          float,
    option_type: str = "CE",
    r:           float = 0.065,
) -> float:
    """
    Standard Black-Scholes d1 delta approximation.
    CE: N(d1), PE: N(d1) - 1
    """
    if dte <= 0:
        return 0.50
    t = dte / 365.0
    sig = max(iv, 0.05)
    denom = sig * math.sqrt(t)
    if denom == 0:
        return 0.50
    d1 = (math.log(spot / strike) + 0.5 * sig * sig * t) / denom
    # Abramowitz and Stegun approximation for standard normal CDF
    nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    if option_type.upper() == "CE":
        return max(0.01, min(0.99, nd1))
    else:
        return max(0.01, min(0.99, 1.0 - nd1))


def size_by_delta(
    spot:          float = 0.0,
    strike:        int   = 0,
    dte:           int   = 0,
    iv:            float = 0.0,
    option_type:   str   = "FUT",
    entry_premium: float = 0.0,
    lot_size:      int   = None,
    capital_budget:float = None,
    target_delta:  float = None,
    **kwargs,
) -> PositionSize:
    """
    Compute position size targeting a fixed delta exposure or futures margin.
    """
    premium = float(kwargs.get("premium", entry_premium) or entry_premium or spot)
    capital = float(kwargs.get("capital", DEPLOYED_CAPITAL) or DEPLOYED_CAPITAL)
    daily_cap_pct = float(kwargs.get("daily_cap_pct", DAILY_CAPITAL_PCT) or DAILY_CAPITAL_PCT)
    prem_source = kwargs.get("premium_source", "")

    ls      = lot_size or kwargs.get("ls") or 1
    budget  = capital_budget or (capital * daily_cap_pct / 100 / 3)
    t_delta = target_delta or TARGET_DELTA_UNITS

    if option_type in ("FUT", "FUTURE", "") or prem_source == "COMMODITY_FUTURES":
        margin_est = max(premium * 0.15, 1000.0)
        lots = max(1, min(5, int(budget / margin_est)))
        return PositionSize(
            lots=lots,
            lot_size=ls,
            capital_required=round(lots * margin_est, 2),
            delta_exposure=float(lots * ls),
            target_delta=t_delta,
            delta_efficiency=1.0,
            method="futures_margin",
            note="Commodity futures position sizing (delta=1.0)",
        )

    # Compute delta at this strike
    delta = compute_bs_delta(spot, float(strike), dte, iv, option_type)

    # Reject if option is too far OTM (low delta = dying option)
    if delta < MIN_DELTA_THRESHOLD:
        return PositionSize(
            lots=0, lot_size=ls, capital_required=0,
            delta_exposure=0, target_delta=t_delta,
            delta_efficiency=0, method="rejected",
            note=f"Delta {delta:.3f} < min {MIN_DELTA_THRESHOLD} — option too far OTM",
        )

    # Delta-based sizing: lots = target_delta / (delta × lot_size)
    lots_delta = t_delta / (delta * ls)
    lots_delta = max(1, min(round(lots_delta), MAX_LOTS_DELTA))

    # Capital constraint: lots × premium × lot_size <= budget
    cost_per_lot  = entry_premium * ls
    lots_capital  = max(1, int(budget / cost_per_lot)) if cost_per_lot > 0 else 1
    lots_capital  = min(lots_capital, MAX_LOTS_DELTA)

    # Take minimum of delta and capital constraints
    final_lots      = min(lots_delta, lots_capital)
    capital_req     = final_lots * cost_per_lot
    actual_delta    = final_lots * ls * delta
    delta_efficiency= actual_delta / t_delta

    # Determine binding constraint
    if lots_delta <= lots_capital:
        method = "delta"
        note   = f"Delta-sized: δ={delta:.3f} × {ls} × {final_lots}lots = {actual_delta:.1f} units"
    else:
        method = "capital"
        note   = f"Capital-limited: ₹{budget:,.0f} budget = {lots_capital} lots"

    logger.debug(
        f"[DeltaSizer] {option_type} strike={strike} δ={delta:.3f} | "
        f"Target={t_delta} units → {final_lots} lots | "
        f"Actual={actual_delta:.1f} units | Method={method}"
    )

    return PositionSize(
        lots             = final_lots,
        lot_size         = ls,
        capital_required = round(capital_req, 2),
        delta_exposure   = round(actual_delta, 2),
        target_delta     = t_delta,
        delta_efficiency = round(delta_efficiency, 3),
        method           = method,
        note             = note,
    )

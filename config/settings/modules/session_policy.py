"""
config/settings/modules/session_policy.py — MCX Commodity Session-Aware Trading Policy
====================================================================================
Decouples trading parameters between the low-liquidity domestic/European session
(09:00 - 17:00 IST) and the high-liquidity US / COMEX / NYMEX evening session
(17:00 - 23:30 IST).

Also enforces:
1. Macro News Blackout Windows (EIA Crude/Gas, US CPI/NFP/FOMC prints).
2. Expiry Day Late-Evening Cutoff (avoids theta collapse trap after 21:00 IST).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, time, date
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")


@dataclass(frozen=True)
class SessionPolicy:
    session_name: str
    min_strategy_votes: int
    ml_min_confidence: float
    ml_min_rank: float
    max_lots: int
    target_exit_mode: str  # "QUICK_SCALP" vs "RUNNER_LADDER"
    preferred_strategies: list[str] = field(default_factory=list)
    discouraged_strategies: list[str] = field(default_factory=list)
    description: str = ""


# ── Morning / European Session (09:00 - 17:00 IST) ───────────────────────────
# Characteristics: Domestic retail & European open. Thin volume (~20-25% of day).
# High risk of false breakouts, chop, and theta bleed.
# Policy: Defensive, higher consensus (5+ votes), stricter ML gate (0.35+ conf), 1 lot max.
MORNING_POLICY = SessionPolicy(
    session_name="MORNING",
    min_strategy_votes=int(os.getenv("MORNING_MIN_VOTES", "5")),
    ml_min_confidence=float(os.getenv("MORNING_ML_MIN_CONF", "0.35")),
    ml_min_rank=float(os.getenv("MORNING_ML_MIN_RANK", "0.65")),
    max_lots=int(os.getenv("MORNING_MAX_LOTS", "1")),
    target_exit_mode="QUICK_SCALP",
    preferred_strategies=[
        "VWAPMeanReversion",
        "BBMeanReversion",
        "RSI2MeanReversion",
        "CPR",
        "RangeSpread",
        "CalendarSeasonality",
    ],
    discouraged_strategies=[
        "DonchianBreakout",
        "MomentumVolumeBreakout",
    ],
    description="Domestic/European morning: mean-reversion favored, conservative 1-lot sizing, strict ML gate",
)

# Sector-specific Evening Policies
EVENING_ENERGY_POLICY = SessionPolicy(
    session_name="EVENING",
    min_strategy_votes=int(os.getenv("EVENING_MIN_VOTES", "4")),
    ml_min_confidence=float(os.getenv("EVENING_ML_MIN_CONF", "0.22")),
    ml_min_rank=float(os.getenv("EVENING_ML_MIN_RANK", "0.40")),
    max_lots=int(os.getenv("EVENING_MAX_LOTS", "3")),
    target_exit_mode="RUNNER_LADDER",
    preferred_strategies=[
        "TrendFollowing",
        "MomentumVolumeBreakout",
        "SuperTrend+RSI",
        "OrderFlowDelta",
        "SqueezeMomentum",
        "EMASlope",
        "Ichimoku",
        "VWAP+EMA",
        "VolatilityBreakout",
    ],
    discouraged_strategies=[
        "CounterTrendMeanReversion",
    ],
    description="US Energy evening: institutional macro momentum, multi-lot scale-out, progressive profit locks",
)

EVENING_METALS_POLICY = SessionPolicy(
    session_name="EVENING",
    min_strategy_votes=int(os.getenv("EVENING_MIN_VOTES", "4")),
    ml_min_confidence=float(os.getenv("EVENING_ML_MIN_CONF", "0.26")),
    ml_min_rank=float(os.getenv("EVENING_ML_MIN_RANK", "0.40")),
    max_lots=int(os.getenv("EVENING_MAX_LOTS", "2")),
    target_exit_mode="RUNNER_LADDER",
    preferred_strategies=[
        "PriceAction",
        "RangeSpread",
        "VolumeProfile",
        "OIAnalysis",
        "SMC",
        "LiqSweep",
        "FVG",
        "ElliottWave",
        "OpeningRangeBreakout",
    ],
    discouraged_strategies=[
        "Ichimoku",
        "SuperTrend+RSI",
        "ADX+PSAR",
        "TermStructure",
    ],
    description="US COMEX metals evening: institutional smart-money, liquidity sweeps, early 3-4 vote consensus",
)

EVENING_POLICY = EVENING_ENERGY_POLICY

# ── Off-Market Fallback ───────────────────────────────────────────────────────
OFF_MARKET_POLICY = SessionPolicy(
    session_name="OFF_MARKET",
    min_strategy_votes=6,
    ml_min_confidence=0.50,
    ml_min_rank=0.75,
    max_lots=1,
    target_exit_mode="QUICK_SCALP",
    description="Outside active MCX market hours",
)


def get_session_policy(dt: datetime | None = None, symbol: str | None = None) -> SessionPolicy:
    """Returns the active SessionPolicy based on IST time and commodity sector."""
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)
    else:
        dt = dt.astimezone(IST)

    sym_clean = (symbol or "").upper()
    is_metals = any(m in sym_clean for m in ("SILVER", "GOLD"))

    t = dt.time()
    if time(9, 0) <= t < time(17, 0):
        return MORNING_POLICY
    elif time(17, 0) <= t <= time(23, 30):
        return EVENING_METALS_POLICY if is_metals else EVENING_ENERGY_POLICY
    else:
        return OFF_MARKET_POLICY


# ── High-Impact Macro Economic Event Freeze Windows (IST) ─────────────────────
# Freezes new option entries 10 minutes before and 10 minutes after volatile prints.
# Times in IST:
# - Wednesdays 20:00: US EIA Weekly Petroleum Status (Crude Oil)
# - Thursdays 20:00: US EIA Natural Gas Storage Report
# - Daily / Periodic 18:00, 19:00, 19:30, 20:00: US CPI, PPI, NFP, Retail Sales, FOMC
MACRO_FREEZE_ENABLED = os.getenv("MACRO_FREEZE_ENABLED", "1").strip().lower() in ("1", "true", "yes")
MACRO_FREEZE_BUFFER_MINUTES = int(os.getenv("MACRO_FREEZE_BUFFER_MINUTES", "10"))


def is_macro_news_freeze_window(dt: datetime | None = None, symbol: str = "SILVERM") -> tuple[bool, str]:
    """
    Checks whether current IST time is inside a high-impact US macro economic release window.
    Returns: (is_frozen, reason_string)
    """
    if not MACRO_FREEZE_ENABLED:
        return False, ""

    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)
    else:
        dt = dt.astimezone(IST)

    sym_clean = str(symbol or "").upper().strip()
    weekday = dt.weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    hour = dt.hour
    minute = dt.minute
    total_mins = hour * 60 + minute

    # 1. Commodity-Specific EIA Reports (20:00 IST = 1200 minutes)
    eia_target_mins = 20 * 60  # 20:00 IST
    if abs(total_mins - eia_target_mins) <= MACRO_FREEZE_BUFFER_MINUTES:
        # Wednesday: Crude Oil Inventories
        if weekday == 2 and any(k in sym_clean for k in ("CRUDE", "OIL")):
            return True, f"Macro News Freeze: US EIA Crude Oil Inventories (Wed 20:00 IST ±{MACRO_FREEZE_BUFFER_MINUTES}m)"
        # Thursday: Natural Gas Storage
        if weekday == 3 and any(k in sym_clean for k in ("NATGAS", "GAS")):
            return True, f"Macro News Freeze: US EIA Natural Gas Storage (Thu 20:00 IST ±{MACRO_FREEZE_BUFFER_MINUTES}m)"

    # 2. General US Data Releases (18:00, 19:00, 19:30, 20:00 IST) for all commodities
    major_release_windows = [
        (18 * 60, "US Retail Sales / Housing Starts (18:00 IST)"),
        (19 * 60, "US Jobless Claims / PPI / Trade (19:00 IST)"),
        (19 * 60 + 30, "US CPI / Non-Farm Payrolls (19:30 IST)"),
        (20 * 60, "US Major Data / FOMC Rate (20:00 IST)"),
    ]

    for win_mins, label in major_release_windows:
        if abs(total_mins - win_mins) <= MACRO_FREEZE_BUFFER_MINUTES:
            # During US data prints, option spreads widen and slippage blows SLs
            return True, f"Macro News Freeze: {label} (±{MACRO_FREEZE_BUFFER_MINUTES}m)"

    return False, ""


# ── Expiry Day Late-Evening Cutoff (Theta Trap Protection) ─────────────────────
EXPIRY_FREEZE_TIME_STR = os.getenv("EXPIRY_FREEZE_TIME", "21:00")


def is_expiry_option_freeze(dt: datetime | None = None, expiry_date: str | date | None = None) -> tuple[bool, str]:
    """
    Checks if today is option expiry day and current time is past the late-evening cutoff (21:00 IST).
    Buying options after 21:00 on expiry day leads to catastrophic theta evaporation.
    """
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = IST.localize(dt)
    else:
        dt = dt.astimezone(IST)

    if not expiry_date:
        return False, ""

    try:
        if isinstance(expiry_date, str):
            # Parse '24Sep2026', '2026-09-24', etc.
            exp_d = pd.to_datetime(expiry_date).date()
        elif isinstance(expiry_date, datetime):
            exp_d = expiry_date.date()
        else:
            exp_d = expiry_date
    except Exception:
        return False, ""

    if dt.date() == exp_d:
        freeze_h, freeze_m = map(int, EXPIRY_FREEZE_TIME_STR.split(":"))
        freeze_time = time(freeze_h, freeze_m)
        if dt.time() >= freeze_time:
            return True, f"Expiry Day Late Cutoff: No new option buys after {EXPIRY_FREEZE_TIME_STR} IST on expiry ({exp_d})"

    return False, ""

"""
instruments/registry.py — Central Multi-Instrument Registry & Strategy Mapping

Provides dynamic registry for all supported MCX commodity futures,
alias normalization, active contract resolution, and per-instrument strategy configurations.
"""

import os
from typing import Dict, List, Optional
from datetime import date

from instruments.base import InstrumentConfig, ContractSpec
from instruments.silverm import SILVERM_CONFIG, SILVERMIC_CONFIG, get_active_silverm_contract, get_active_silvermic_contract
from instruments.gold import GOLD_CONFIG, GOLDM_CONFIG
from instruments.crudeoil import CRUDEOIL_CONFIG, CRUDEOILM_CONFIG
from instruments.naturalgas import NATURALGAS_CONFIG, NATGASM_CONFIG

# All supported commodity configurations
INSTRUMENT_CATALOG: Dict[str, InstrumentConfig] = {
    "SILVERM": SILVERM_CONFIG,
    "SILVERMIC": SILVERM_CONFIG,  # Aliased to SILVERM
    "GOLDM": GOLDM_CONFIG,
    "GOLD": GOLD_CONFIG,
    "CRUDEOILM": CRUDEOILM_CONFIG,
    "CRUDEOIL": CRUDEOIL_CONFIG,
    "NATGASM": NATGASM_CONFIG,
    "NATURALGAS": NATURALGAS_CONFIG,
}

# Aliases for user convenience and broker mapping
INSTRUMENT_ALIASES: Dict[str, str] = {
    "SILVERM": "SILVERM",
    "SILVER": "SILVERM",
    "SILVERMIC": "SILVERM",
    "SILVER_MIC": "SILVERM",
    "GOLDM": "GOLDM",
    "GOLD_MINI": "GOLDM",
    "GOLD": "GOLDM",                     # Default to mini
    "CRUDE": "CRUDEOILM",
    "CRUDEOIL": "CRUDEOILM",             # Default to mini
    "CRUDEOILM": "CRUDEOILM",
    "NATGAS": "NATGASM",
    "NATGASM": "NATGASM",
    "NATURALGAS": "NATGASM",
}

ALL_COMMODITY_FUTURES_STRATEGIES = [
    "TrendFollowing", "OpeningRangeBreakout", "VWAPMeanReversion", "VolatilityBreakout",
    "DonchianBreakout", "BBMeanReversion", "MACrossover", "RSIDivergence",
    "MomentumVolumeBreakout", "GoldSilverPairs", "OrderFlowDelta", "TimeOfDaySeasonality",
    "RSI2MeanReversion", "CalendarSeasonality", "TermStructure",
    "SuperTrend+RSI", "VWAP+EMA", "ADX+PSAR",
    "FVG", "UTBot", "CPR", "Ichimoku", "VolumeProfile", "LiqSweep",
    "PriceAction", "OIAnalysis", "AMD", "GapDirection", "SMC",
    "GapMomentum", "ADXRising", "RangeSpread", "SqueezeMomentum",
    "StochRSI", "EMASlope", "HeikinAshi", "ElliottWave",
]

# Per-Instrument Strategy & Risk Governance
# Each commodity can independently configure enabled strategies, votes, and risk.
DEFAULT_MIN_VOTES = int(os.getenv("MIN_STRATEGY_VOTES", "5"))

INSTRUMENT_STRATEGY_CONFIG: Dict[str, dict] = {
    "SILVERM": {
        "enabled_strategies": list(ALL_COMMODITY_FUTURES_STRATEGIES),
        "min_votes": DEFAULT_MIN_VOTES,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 1,
    },
    "SILVERMIC": {
        "enabled_strategies": list(ALL_COMMODITY_FUTURES_STRATEGIES),
        "min_votes": DEFAULT_MIN_VOTES,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 2,
    },
    "GOLDM": {
        "enabled_strategies": [
            "TrendFollowing", "OpeningRangeBreakout", "VWAPMeanReversion", "VolatilityBreakout", "DonchianBreakout",
            "SuperTrend+RSI", "VWAP+EMA", "ORB", "ADX+PSAR", "Ichimoku",
            "PriceAction", "CPR", "EMASlope", "HeikinAshi"
        ],
        "min_votes": DEFAULT_MIN_VOTES,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 1,
    },
    "CRUDEOILM": {
        "enabled_strategies": [
            "TrendFollowing", "OpeningRangeBreakout", "VolatilityBreakout", "DonchianBreakout",
            "SuperTrend+RSI", "ORB", "BBSqueeze", "ADX+PSAR", "VolumeProfile",
            "PriceAction", "SqueezeMomentum", "OpeningRangeBias"
        ],
        "min_votes": DEFAULT_MIN_VOTES,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 2,
    },
    "NATGASM": {
        "enabled_strategies": [
            "TrendFollowing", "OpeningRangeBreakout", "VolatilityBreakout", "DonchianBreakout",
            "ORB", "BBSqueeze", "UTBot", "VolumeProfile", "PriceAction"
        ],
        "min_votes": DEFAULT_MIN_VOTES,
        "max_risk_per_trade_pct": 1.5,   # Higher margin/volatility caution
        "max_open_positions": 1,
        "max_lots_per_trade": 1,
    }
}


def normalize_symbol(symbol: str) -> str:
    """Normalizes any broker or user alias to canonical instrument symbol."""
    sym = (symbol or "").strip().upper()
    for prefix in ["MCX:", "MCX_", "FUTCOM:"]:
        if sym.startswith(prefix):
            sym = sym[len(prefix):]
    if sym in INSTRUMENT_ALIASES:
        return INSTRUMENT_ALIASES[sym]
    # Check if contract symbol e.g. SILVERMIC-30Nov2026-FUT
    for root in sorted(INSTRUMENT_CATALOG.keys(), key=len, reverse=True):
        if sym.startswith(root):
            return root
    return sym


def get_instrument_config(symbol: str = "SILVERM") -> InstrumentConfig:
    """Retrieves the InstrumentConfig for a given commodity symbol."""
    norm = normalize_symbol(symbol)
    if norm in INSTRUMENT_CATALOG:
        return INSTRUMENT_CATALOG[norm]
    # Fallback to SILVERM
    return SILVERM_CONFIG


def get_instrument_strategy_config(symbol: str = "SILVERM") -> dict:
    """Retrieves strategy enablement and risk parameters for an instrument."""
    norm = normalize_symbol(symbol)
    return INSTRUMENT_STRATEGY_CONFIG.get(norm, INSTRUMENT_STRATEGY_CONFIG["SILVERM"])


def resolve_active_contract(symbol: str = "SILVERM", as_of: Optional[date] = None) -> ContractSpec:
    """
    Resolves the front-month active contract for the requested commodity.
    For SILVERM, uses the cycle calendar with tender-period protection.
    """
    norm = normalize_symbol(symbol)
    if norm in ("SILVERM", "SILVERMIC"):
        return get_active_silverm_contract(as_of)

    config = get_instrument_config(norm)
    # Generic monthly/cycle resolution for other commodities
    today = as_of or date.today()
    # Find next expiry around 5th or end of month depending on config
    expiry = date(today.year, today.month, 28)
    if expiry < today:
        month = today.month + 1 if today.month < 12 else 1
        year = today.year if today.month < 12 else today.year + 1
        expiry = date(year, month, 28)

    trading_symbol = f"{norm}-{expiry.strftime('%d%b%Y')}-FUT"
    return ContractSpec(
        symbol=norm,
        trading_symbol=trading_symbol,
        expiry_date=expiry,
        lot_size=config.lot_size,
        tick_size=config.tick_size,
        tick_value=config.tick_value,
    )


def resolve_active_option_contract(
    symbol: str = "SILVERM",
    as_of: Optional[date] = None,
    strike: Optional[int] = None,
    option_type: str = "CE",
    rollover_days: int = 5,
) -> tuple[str, date, int]:
    """
    Resolves the front-month active option contract, expiry date, and DTE.
    Rules:
      1. Trades in the current monthly expiry whenever possible.
      2. Rolls to next monthly expiry if within rollover_days (default 5 days) of expiry.
    """
    norm = normalize_symbol(symbol)
    today = as_of or date.today()
    if norm in ("SILVERM", "SILVERMIC"):
        from instruments.silverm import resolve_silverm_option_symbol
        return resolve_silverm_option_symbol(today, strike or 240000, option_type, rollover_days)

    # Generic monthly option resolution
    m = today.month
    y = today.year
    expiry = date(y, m, 24)
    while expiry.weekday() >= 5:
        expiry -= timedelta(days=1)
    if (expiry - today).days <= rollover_days:
        m = m + 1 if m < 12 else 1
        y = y if m > 1 else y + 1
        expiry = date(y, m, 24)
        while expiry.weekday() >= 5:
            expiry -= timedelta(days=1)

    dte = max(1, (expiry - today).days)
    exp_str = expiry.strftime("%d%b%Y")
    trading_symbol = f"{norm}-{exp_str}-{int(strike or 0)}-{option_type.upper()}"
    return trading_symbol, expiry, dte

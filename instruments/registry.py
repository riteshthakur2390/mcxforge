"""
instruments/registry.py — Central Multi-Instrument Registry & Strategy Mapping

Provides dynamic registry for all supported MCX commodity futures,
alias normalization, active contract resolution, and per-instrument strategy configurations.
"""

import os
from typing import Dict, List, Optional
from datetime import date, timedelta

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

# Per-Instrument Strategy, Execution & Risk Governance
# Each commodity independently configures enabled strategies, votes, SL, target, and trailing.
DEFAULT_MIN_VOTES = int(os.getenv("MIN_STRATEGY_VOTES", "5"))

INSTRUMENT_STRATEGY_CONFIG: Dict[str, dict] = {
    "CRUDEOILM": {
        "description": "High-momentum trending commodity (Live Focus)",
        "enabled_strategies": [
            "TrendFollowing", "OpeningRangeBreakout", "ADX+PSAR", "SuperTrend+RSI",
            "VWAP+EMA", "PriceAction", "OIAnalysis", "OpeningRangeBias"
        ],
        "min_votes": 3,
        "min_ml_confidence": 0.22,
        "stop_loss_pct": 10.0,
        "target1_pct": 14.0,
        "target2_pct": 22.0,
        "breakeven_trigger_pct": 10.0,
        "trailing_activation_pct": 12.0,
        "trailing_sl_pct": 8.0,
        "giveback_cap_pct": 5.5,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 2,
    },
    "CRUDEOIL": {
        "description": "High-momentum trending commodity (Live Focus)",
        "enabled_strategies": [
            "TrendFollowing", "OpeningRangeBreakout", "ADX+PSAR", "SuperTrend+RSI",
            "VWAP+EMA", "PriceAction", "OIAnalysis", "OpeningRangeBias"
        ],
        "min_votes": 3,
        "min_ml_confidence": 0.22,
        "stop_loss_pct": 10.0,
        "target1_pct": 14.0,
        "target2_pct": 22.0,
        "breakeven_trigger_pct": 10.0,
        "trailing_activation_pct": 12.0,
        "trailing_sl_pct": 8.0,
        "giveback_cap_pct": 5.5,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 1,
    },
    "GOLDM": {
        "description": "High-capital precision commodity",
        "enabled_strategies": [
            "TrendFollowing", "OpeningRangeBreakout", "RSIDivergence", "RSI2MeanReversion",
            "CalendarSeasonality", "TermStructure", "SuperTrend+RSI", "CPR", "PriceAction"
        ],
        "min_votes": 4,
        "min_ml_confidence": 0.30,
        "stop_loss_pct": 7.5,
        "target1_pct": 8.0,
        "target2_pct": 13.5,
        "breakeven_trigger_pct": 7.0,
        "trailing_activation_pct": 8.0,
        "trailing_sl_pct": 6.0,
        "giveback_cap_pct": 4.5,
        "max_risk_per_trade_pct": 1.5,
        "max_open_positions": 1,
        "max_lots_per_trade": 1,
    },
    "GOLD": {
        "description": "Institutional liquidity sweep & mean-reversion",
        "enabled_strategies": [
            "PriceAction", "VolumeProfile", "LiqSweep", "FVG", "SMC",
            "VWAPMeanReversion", "CPR", "OpeningRangeBreakout", "CalendarSeasonality"
        ],
        "min_votes": 3,
        "min_ml_confidence": 0.30,
        "stop_loss_pct": 7.5,
        "target1_pct": 8.0,
        "target2_pct": 13.5,
        "breakeven_trigger_pct": 7.0,
        "trailing_activation_pct": 8.0,
        "trailing_sl_pct": 6.0,
        "giveback_cap_pct": 4.5,
        "max_risk_per_trade_pct": 1.5,
        "max_open_positions": 1,
        "max_lots_per_trade": 1,
    },
    "SILVERM": {
        "description": "Smart-money expansion & early momentum breakout",
        "enabled_strategies": [
            "SMC", "OrderFlowDelta", "VolumeProfile", "OIAnalysis", "TrendFollowing",
            "OpeningRangeBreakout", "VWAPMeanReversion", "CPR", "FVG", "PriceAction", "ElliottWave"
        ],
        "min_votes": 4,
        "min_ml_confidence": 0.26,
        "stop_loss_pct": 8.5,
        "target1_pct": 12.0,
        "target2_pct": 18.0,
        "breakeven_trigger_pct": 9.0,
        "trailing_activation_pct": 10.0,
        "trailing_sl_pct": 7.5,
        "giveback_cap_pct": 5.5,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 1,
    },
    "SILVERMIC": {
        "description": "Smart-money expansion & early momentum breakout (Micro)",
        "enabled_strategies": [
            "TrendFollowing", "OpeningRangeBreakout", "VWAPMeanReversion",
            "VolatilityBreakout", "DonchianBreakout", "SuperTrend+RSI", "CPR",
            "SMC", "OrderFlowDelta", "VolumeProfile", "OIAnalysis", "FVG", "PriceAction", "ElliottWave"
        ],
        "min_votes": 4,
        "min_ml_confidence": 0.26,
        "stop_loss_pct": 8.5,
        "target1_pct": 12.0,
        "target2_pct": 18.0,
        "breakeven_trigger_pct": 9.0,
        "trailing_activation_pct": 10.0,
        "trailing_sl_pct": 7.5,
        "giveback_cap_pct": 5.5,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 2,
    },
    "NATGASM": {
        "description": "High-beta volatility micro-burst commodity",
        "enabled_strategies": [
            "RSIDivergence", "VolumeProfile", "CPR", "PriceAction",
            "TimeOfDaySeasonality", "MACrossover", "ElliottWave",
            "VWAPMeanReversion", "SqueezeMomentum"
        ],
        "min_votes": 3,
        "min_ml_confidence": 0.28,
        "stop_loss_pct": 14.0,
        "target1_pct": 18.0,
        "target2_pct": 32.0,
        "breakeven_trigger_pct": 12.0,
        "trailing_activation_pct": 15.0,
        "trailing_sl_pct": 10.0,
        "giveback_cap_pct": 6.5,
        "max_risk_per_trade_pct": 1.5,
        "max_open_positions": 1,
        "max_lots_per_trade": 1,
    },
    "NATURALGAS": {
        "description": "High-beta volatility micro-burst commodity",
        "enabled_strategies": [
            "RSIDivergence", "VolumeProfile", "CPR", "PriceAction",
            "TimeOfDaySeasonality", "MACrossover", "ElliottWave",
            "VWAPMeanReversion", "SqueezeMomentum"
        ],
        "min_votes": 3,
        "min_ml_confidence": 0.28,
        "stop_loss_pct": 14.0,
        "target1_pct": 18.0,
        "target2_pct": 32.0,
        "breakeven_trigger_pct": 12.0,
        "trailing_activation_pct": 15.0,
        "trailing_sl_pct": 10.0,
        "giveback_cap_pct": 6.5,
        "max_risk_per_trade_pct": 1.5,
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
    sym_up = (symbol or "").strip().upper()
    if sym_up in INSTRUMENT_STRATEGY_CONFIG:
        return INSTRUMENT_STRATEGY_CONFIG[sym_up]
    norm = normalize_symbol(symbol)
    if norm in INSTRUMENT_STRATEGY_CONFIG:
        return INSTRUMENT_STRATEGY_CONFIG[norm]
    return INSTRUMENT_STRATEGY_CONFIG.get("CRUDEOILM", INSTRUMENT_STRATEGY_CONFIG.get("SILVERM", {}))


def get_instrument_risk_profile(symbol: str) -> dict:
    """Convenience alias for retrieving full risk, SL, TP and trailing profile for an instrument."""
    return get_instrument_strategy_config(symbol)


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
    while expiry.weekday() >= 5:
        expiry -= timedelta(days=1)

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

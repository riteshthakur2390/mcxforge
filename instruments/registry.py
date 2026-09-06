"""
instruments/registry.py — Central Multi-Instrument Registry & Strategy Mapping

Provides dynamic registry for all supported MCX commodity futures,
alias normalization, active contract resolution, and per-instrument strategy configurations.
"""

from typing import Dict, List, Optional
from datetime import date

from instruments.base import InstrumentConfig, ContractSpec
from instruments.silvermic import SILVERMIC_CONFIG, SILVERM_CONFIG, get_active_silvermic_contract
from instruments.gold import GOLD_CONFIG, GOLDM_CONFIG
from instruments.crudeoil import CRUDEOIL_CONFIG, CRUDEOILM_CONFIG
from instruments.naturalgas import NATURALGAS_CONFIG, NATGASM_CONFIG

# All supported commodity configurations
INSTRUMENT_CATALOG: Dict[str, InstrumentConfig] = {
    "SILVERM": SILVERM_CONFIG,
    "SILVERMIC": SILVERMIC_CONFIG,
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
    "SILVERMIC": "SILVERMIC",
    "SILVER_MIC": "SILVERMIC",
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

# Per-Instrument Strategy & Risk Governance
# Each commodity can independently configure enabled strategies, votes, and risk.
INSTRUMENT_STRATEGY_CONFIG: Dict[str, dict] = {
    "SILVERM": {
        "enabled_strategies": [
            "TrendFollowing", "OpeningRangeBreakout", "VWAPMeanReversion", "VolatilityBreakout", "DonchianBreakout",
            "SuperTrend+RSI", "VWAP+EMA", "ORB", "BBSqueeze", "ADX+PSAR",
            "FVG", "UTBot", "CPR", "Ichimoku", "VolumeProfile",
            "LiqSweep", "PriceAction", "AMD", "GapDirection", "SMC",
            "GapMomentum", "SqueezeMomentum", "EMASlope", "HeikinAshi",
            "OpeningRangeBias"
        ],
        "min_votes": 3,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 1,
    },
    "SILVERMIC": {
        "enabled_strategies": [
            # 5 Canonical MCX Baseline Strategies
            "TrendFollowing", "OpeningRangeBreakout", "VWAPMeanReversion", "VolatilityBreakout", "DonchianBreakout",
            # Legacy Technical Models
            "SuperTrend+RSI", "VWAP+EMA", "ORB", "BBSqueeze", "ADX+PSAR",
            "FVG", "UTBot", "CPR", "Ichimoku", "VolumeProfile",
            "LiqSweep", "PriceAction", "AMD", "GapDirection", "SMC",
            "GapMomentum", "SqueezeMomentum", "EMASlope", "HeikinAshi",
            "OpeningRangeBias"
        ],
        "min_votes": 3,
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
        "min_votes": 3,
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
        "min_votes": 3,
        "max_risk_per_trade_pct": 2.0,
        "max_open_positions": 1,
        "max_lots_per_trade": 2,
    },
    "NATGASM": {
        "enabled_strategies": [
            "TrendFollowing", "OpeningRangeBreakout", "VolatilityBreakout", "DonchianBreakout",
            "ORB", "BBSqueeze", "UTBot", "VolumeProfile", "PriceAction"
        ],
        "min_votes": 3,
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


def get_instrument_config(symbol: str = "SILVERMIC") -> InstrumentConfig:
    """Retrieves the InstrumentConfig for a given commodity symbol."""
    norm = normalize_symbol(symbol)
    if norm in INSTRUMENT_CATALOG:
        return INSTRUMENT_CATALOG[norm]
    # Fallback to SILVERMIC
    return SILVERMIC_CONFIG


def get_instrument_strategy_config(symbol: str = "SILVERMIC") -> dict:
    """Retrieves strategy enablement and risk parameters for an instrument."""
    norm = normalize_symbol(symbol)
    return INSTRUMENT_STRATEGY_CONFIG.get(norm, INSTRUMENT_STRATEGY_CONFIG["SILVERMIC"])


def resolve_active_contract(symbol: str = "SILVERMIC", as_of: Optional[date] = None) -> ContractSpec:
    """
    Resolves the front-month active contract for the requested commodity.
    For SILVERMIC, uses the cycle calendar with tender-period protection.
    """
    norm = normalize_symbol(symbol)
    if norm == "SILVERMIC":
        return get_active_silvermic_contract(as_of)

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

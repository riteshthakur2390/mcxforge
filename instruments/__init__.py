"""
instruments — Multi-Instrument Commodity Futures Package for MCXForge
"""

from instruments.base import (
    InstrumentConfig,
    SessionSpec,
    MarginSpec,
    ContractSpec,
    CommoditySector,
)
from instruments.silvermic import SILVERMIC_CONFIG, SILVERM_CONFIG, get_active_silvermic_contract
from instruments.gold import GOLDM_CONFIG, GOLD_CONFIG
from instruments.crudeoil import CRUDEOILM_CONFIG, CRUDEOIL_CONFIG
from instruments.naturalgas import NATGASM_CONFIG, NATURALGAS_CONFIG
from instruments.registry import (
    get_instrument_config,
    get_instrument_strategy_config,
    resolve_active_contract,
    normalize_symbol,
    INSTRUMENT_CATALOG,
    INSTRUMENT_STRATEGY_CONFIG,
)

__all__ = [
    "InstrumentConfig",
    "SessionSpec",
    "MarginSpec",
    "ContractSpec",
    "CommoditySector",
    "SILVERM_CONFIG",
    "SILVERMIC_CONFIG",
    "GOLDM_CONFIG",
    "GOLD_CONFIG",
    "CRUDEOILM_CONFIG",
    "CRUDEOIL_CONFIG",
    "NATGASM_CONFIG",
    "NATURALGAS_CONFIG",
    "get_instrument_config",
    "get_instrument_strategy_config",
    "resolve_active_contract",
    "normalize_symbol",
    "INSTRUMENT_CATALOG",
    "INSTRUMENT_STRATEGY_CONFIG",
]

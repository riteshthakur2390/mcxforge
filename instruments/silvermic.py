"""
instruments/silvermic.py — Deprecated compatibility shim. Use instruments.silverm.
===================================================================================
MCX Silver Micro (SILVERMIC) does not support Option Contracts.
All references have been migrated to SILVERM (Silver Mini, 5 kg lot).
"""

from instruments.silverm import (
    SILVERM_CONFIG,
    SILVERMIC_CONFIG,
    get_silverm_expiry,
    get_silverm_expiry as get_silvermic_expiry,
    get_active_silverm_contract,
    get_active_silverm_contract as get_active_silvermic_contract,
)

__all__ = [
    "SILVERM_CONFIG",
    "SILVERMIC_CONFIG",
    "get_silvermic_expiry",
    "get_active_silvermic_contract",
]

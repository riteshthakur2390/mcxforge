"""
broker/factory.py — Broker Factory
=====================================
100% env-driven. Adding a new broker NEVER requires editing this file.

To add a new broker tomorrow:
    1. Create broker/newbroker_broker.py with class NewBroker(BaseBroker)
    2. Add one line to BROKER_REGISTRY below
    3. Set BROKER=newbroker in .env
    That is ALL. No other file changes needed anywhere.

Supported brokers (BROKER= in .env):
    yfinance → Yahoo Finance     (free, no account — default for testing)
    upstox   → Upstox API        (free account)
    groww    → Groww Trade API   (Rs 499/month)
    kite     → Zerodha Kite      (Rs 500/month, production)
"""

import os
from loguru import logger
from broker.base_broker import BaseBroker
from typing import Union

# ── BROKER REGISTRY ────────────────────────────────────────────────────────
# To add a new broker:
#   "broker_name": ("module.path", "ClassName")
# Nothing else changes anywhere in the codebase.

BROKER_REGISTRY: dict[str, tuple[str, str]] = {
    "yfinance": ("broker.yfinance_broker", "YFinanceBroker"),
    "upstox":   ("broker.upstox_broker",   "UpstoxBroker"),
    "dhan": ("broker.dhan_broker", "DhanBroker"),  # NEW
    "groww":    ("broker.groww_broker",     "GrowwBroker"),
    "kite":     ("broker.kite_broker",      "KiteBroker"),
}

BROKER_INFO: dict[str, str] = {
    "dhan":     "Dhan HQ — 5-year historical data & MCX futures execution (default)",
    "yfinance": "Yahoo Finance — free, no account (testing)",
    "upstox":   "Upstox — free account",
    "groww":    "Groww — Rs 499/month",
    "kite":     "Zerodha KiteConnect — Rs 500/month (production)",
}

# Default broker when BROKER is not set in .env
DEFAULT_BROKER = "dhan"

# ── SINGLETON ──────────────────────────────────────────────────────────────
_broker_instance: Union[BaseBroker, None] = None


def get_broker() -> BaseBroker:
    """
    Returns the active broker singleton.
    Reads BROKER from environment — never hardcoded.
    Lazy-loads the broker class — only imports what is needed.
    """
    global _broker_instance

    if _broker_instance is not None:
        return _broker_instance

    broker_name = os.getenv("BROKER", DEFAULT_BROKER).lower().strip()

    if broker_name not in BROKER_REGISTRY:
        available = ", ".join(BROKER_REGISTRY.keys())
        logger.warning(
            f"[BrokerFactory] Unknown BROKER='{broker_name}'. "
            f"Available: {available}. Defaulting to '{DEFAULT_BROKER}'."
        )
        broker_name = DEFAULT_BROKER

    module_path, class_name = BROKER_REGISTRY[broker_name]

    try:
        import importlib
        module   = importlib.import_module(module_path)
        cls      = getattr(module, class_name)
        _broker_instance = cls()
        logger.info(
            f"[BrokerFactory] Active broker: {broker_name.upper()} "
            f"— {BROKER_INFO.get(broker_name, '')}"
        )
    except ImportError as e:
        logger.error(
            f"[BrokerFactory] Failed to import {class_name} from {module_path}: {e}"
        )
        if broker_name != DEFAULT_BROKER:
            logger.warning(
                f"[BrokerFactory] Falling back to '{DEFAULT_BROKER}'"
            )
            os.environ["BROKER"] = DEFAULT_BROKER
            return get_broker()
        raise

    return _broker_instance


def get_active_broker_name() -> str:
    """Returns the name of currently active broker. Safe to call anytime."""
    return os.getenv("BROKER", DEFAULT_BROKER).lower().strip()


def list_available_brokers() -> dict[str, str]:
    """Returns all registered brokers with descriptions."""
    return dict(BROKER_INFO)


def reset_broker() -> None:
    """Force re-initialization on next get_broker() call. Use in tests only."""
    global _broker_instance
    _broker_instance = None


create_broker = get_broker
"""
utils/bot_trade_registry.py
===========================
Persistent registry tracking orders and positions initiated by SignalForge.

Guarantees that automated square-off routines (such as 15:15 emergency EOD,
profit ladder, position manager, and live reconciler) ONLY act on trades
opened by SignalForge and NEVER touch or square off manual/external trades.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytz

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")
REGISTRY_FILE = Path("data/bot_trades_registry.json")
_LOCK = threading.Lock()


def _today_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


def _load_registry() -> dict:
    with _LOCK:
        if not REGISTRY_FILE.exists():
            return {}
        try:
            with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.debug(f"[BotTradeRegistry] Failed to load registry: {exc}")
            return {}


def _save_registry(data: dict) -> None:
    with _LOCK:
        try:
            REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
            temp_file = REGISTRY_FILE.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            temp_file.replace(REGISTRY_FILE)
        except Exception as exc:
            logger.warning(f"[BotTradeRegistry] Failed to save registry: {exc}")


def register_bot_order(
    symbol: str,
    order_id: str,
    quantity: int,
    transaction: str = "BUY",
    account: str = "primary",
    strategy: str = "",
    date_str: Optional[str] = None,
    plan_meta: Optional[dict] = None,
) -> None:
    """Record an order placed by SignalForge."""
    if not symbol:
        return
    d_str = date_str or _today_str()
    data = _load_registry()
    day_entries = data.setdefault(d_str, {"symbols": {}, "order_ids": []})

    sym_clean = str(symbol).strip().upper()
    sym_record = day_entries["symbols"].setdefault(sym_clean, {
        "symbol": sym_clean,
        "first_seen": datetime.now(IST).isoformat(),
        "last_updated": datetime.now(IST).isoformat(),
        "orders": [],
        "is_active": True,
        "quantity": 0,
        "strategy": strategy,
        "account": account,
    })

    sym_record["last_updated"] = datetime.now(IST).isoformat()
    if plan_meta:
        sym_record["plan_meta"] = plan_meta
    if transaction.upper() == "BUY":
        sym_record["quantity"] = int(sym_record.get("quantity", 0) or 0) + int(quantity)
        sym_record["is_active"] = True
    elif transaction.upper() == "SELL":
        sym_record["quantity"] = max(0, int(sym_record.get("quantity", 0) or 0) - int(quantity))
        if sym_record["quantity"] == 0:
            sym_record["is_active"] = False

    if order_id and str(order_id) not in day_entries["order_ids"]:
        day_entries["order_ids"].append(str(order_id))
        sym_record["orders"].append({
            "order_id": str(order_id),
            "transaction": transaction.upper(),
            "quantity": quantity,
            "account": account,
            "timestamp": datetime.now(IST).isoformat(),
        })

    _save_registry(data)
    logger.info(
        f"[BotTradeRegistry] Registered bot order | {transaction} {sym_clean} "
        f"qty={quantity} id={order_id} active={sym_record['is_active']}"
    )


def mark_bot_trade_closed(symbol: str, date_str: Optional[str] = None) -> None:
    """Mark a bot-traded symbol as fully closed."""
    if not symbol:
        return
    from utils.option_utils import canonical_option_sym
    d_str = date_str or _today_str()
    data = _load_registry()
    sym_clean = str(symbol).strip().upper()
    sym_canonical = canonical_option_sym(sym_clean)
    day_entries = data.get(d_str, {})
    symbols = day_entries.get("symbols", {})

    target_key = None
    if sym_clean in symbols:
        target_key = sym_clean
    elif sym_canonical in symbols:
        target_key = sym_canonical
    else:
        for k in symbols.keys():
            if canonical_option_sym(k) == sym_canonical:
                target_key = k
                break

    if target_key:
        symbols[target_key]["is_active"] = False
        symbols[target_key]["quantity"] = 0
        symbols[target_key]["closed_at"] = datetime.now(IST).isoformat()
        _save_registry(data)
        logger.info(f"[BotTradeRegistry] Marked bot trade closed | {target_key} (input={sym_clean})")


def is_bot_symbol(symbol: str, date_str: Optional[str] = None) -> bool:
    """Check if a given option symbol was traded/managed by SignalForge on a given date."""
    if not symbol:
        return False
    from utils.option_utils import canonical_option_sym
    d_str = date_str or _today_str()
    data = _load_registry()
    day_entries = data.get(d_str, {})
    sym_clean = str(symbol).strip().upper()
    symbols = day_entries.get("symbols", {})
    if sym_clean in symbols:
        return True
    sym_canonical = canonical_option_sym(sym_clean)
    if sym_canonical in symbols:
        return True
    for k in symbols.keys():
        if canonical_option_sym(k) == sym_canonical:
            return True
    return False


def get_bot_trade_info(symbol: str, date_str: Optional[str] = None) -> Optional[dict]:
    """Retrieve the registered trade record for a given symbol today (by exact or canonical match)."""
    if not symbol:
        return None
    from utils.option_utils import canonical_option_sym
    d_str = date_str or _today_str()
    data = _load_registry()
    day_entries = data.get(d_str, {})
    sym_clean = str(symbol).strip().upper()
    symbols = day_entries.get("symbols", {})
    if sym_clean in symbols:
        return symbols[sym_clean]
    sym_canonical = canonical_option_sym(sym_clean)
    if sym_canonical in symbols:
        return symbols[sym_canonical]
    for k, v in symbols.items():
        if canonical_option_sym(k) == sym_canonical:
            return v
    return None


def is_bot_order_id(order_id: str, date_str: Optional[str] = None) -> bool:
    """Check if an order ID was placed by SignalForge."""
    if not order_id:
        return False
    d_str = date_str or _today_str()
    data = _load_registry()
    day_entries = data.get(d_str, {})
    return str(order_id) in day_entries.get("order_ids", [])



def get_active_bot_symbols(date_str: Optional[str] = None) -> set[str]:
    """Get all option symbols currently tracked as active bot positions for today."""
    d_str = date_str or _today_str()
    data = _load_registry()
    day_entries = data.get(d_str, {})
    symbols = day_entries.get("symbols", {})
    return {sym for sym, details in symbols.items() if details.get("is_active", False) or details.get("quantity", 0) > 0}


def get_all_bot_symbols_today(date_str: Optional[str] = None) -> set[str]:
    """Get all option symbols touched by the bot today (open or closed)."""
    d_str = date_str or _today_str()
    data = _load_registry()
    day_entries = data.get(d_str, {})
    return set(day_entries.get("symbols", {}).keys())

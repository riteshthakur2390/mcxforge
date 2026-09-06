"""
utils/multi_account_execution.py
================================
Concurrent order fan-out for multiple execution accounts.

Market data continues to use the primary broker. Extra accounts are used only
for order placement when their credentials are configured.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Iterable

from loguru import logger

from broker.base_broker import BaseBroker, OrderResult


@dataclass
class ExecutionAccount:
    label: str
    broker: BaseBroker
    primary: bool = False


_CACHED_EXECUTION_ACCOUNTS: list[ExecutionAccount] | None = None
_CACHED_PRIMARY_BROKER: BaseBroker | None = None


def load_execution_accounts(primary_broker: BaseBroker, force_reload: bool = False) -> list[ExecutionAccount]:
    global _CACHED_EXECUTION_ACCOUNTS, _CACHED_PRIMARY_BROKER
    if not force_reload and _CACHED_EXECUTION_ACCOUNTS is not None and _CACHED_PRIMARY_BROKER is primary_broker:
        return _CACHED_EXECUTION_ACCOUNTS

    accounts: list[ExecutionAccount] = []

    dhan_execution_enabled = os.getenv("DHAN_ORDER_EXECUTION_ENABLED", os.getenv("DHAN_EXECUTION_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    disable_primary = os.getenv("DISABLE_PRIMARY_EXECUTION", "false").strip().lower() in {"1", "true", "yes", "on"}
    primary_label = str(getattr(primary_broker, "account_label", "") or getattr(primary_broker, "broker_name", "") or "primary")
    is_primary_dhan = "dhan" in primary_label.lower() or getattr(primary_broker, "broker_name", "").lower() == "dhan"

    # If primary is Dhan and Dhan execution is disabled, exclude primary from order execution
    if is_primary_dhan and not dhan_execution_enabled:
        logger.info(f"[MultiAccountExecution] Dhan primary broker ({primary_label}) reserved for market data & candle feeds only (order execution disabled via DHAN_ORDER_EXECUTION_ENABLED=false).")
    elif not disable_primary:
        accounts.append(
            ExecutionAccount(
                label=primary_label,
                broker=primary_broker,
                primary=True,
            )
        )
    else:
        logger.info(f"[MultiAccountExecution] Primary broker ({primary_label}) reserved for market data feeds only (execution disabled).")

    seen_labels = {primary_label} if (not disable_primary and not (is_primary_dhan and not dhan_execution_enabled)) else set()

    # ── 1. Upstox Execution Account ──────────────────────────────────────────
    upstox_token = str(os.getenv("UPSTOX_ACCESS_TOKEN", "")).strip()
    if upstox_token and "upstox" not in seen_labels:
        try:
            from broker.upstox_broker import UpstoxBroker
            upstox_broker = UpstoxBroker()
            accounts.append(
                ExecutionAccount(
                    label="upstox:main",
                    broker=upstox_broker,
                    primary=False,
                )
            )
            seen_labels.add("upstox")
            logger.info("[MultiAccountExecution] Added Upstox account for order execution.")
        except Exception as exc:
            logger.warning(f"[MultiAccountExecution] Unable to load Upstox account: {exc}")

    # ── 2. Dhan Secondary Accounts ───────────────────────────────────────────
    if not dhan_execution_enabled:
        logger.info("[MultiAccountExecution] Dhan secondary accounts excluded from order execution (DHAN_ORDER_EXECUTION_ENABLED=false).")
    else:
        try:
            from broker.dhan_broker import DhanBroker

            primary_dhan_id = str(os.getenv("DHAN_CLIENT_ID", "")).strip()
            seen_dhan = {primary_dhan_id} if (not disable_primary and primary_dhan_id) else set()
            max_accounts = int(os.getenv("DHAN_PARALLEL_ACCOUNT_SCAN_LIMIT", "5") or "5")

            for idx in range(2, max_accounts + 1):
                client_id = str(os.getenv(f"DHAN_CLIENT_ID_{idx}", "")).strip()
                token = str(os.getenv(f"DHAN_ACCESS_TOKEN_{idx}", "")).strip()
                if not client_id and token:
                    try:
                        import base64, json
                        parts = token.split(".")
                        if len(parts) >= 2:
                            padding = "=" * (4 - (len(parts[1]) % 4))
                            claims = json.loads(base64.b64decode(parts[1] + padding))
                            client_id = str(claims.get("dhanClientId") or claims.get("client_id") or "").strip()
                    except Exception:
                        pass

                if not client_id and not token:
                    continue
                if not client_id or not token:
                    logger.warning(
                        f"[MultiAccountExecution] Skipping DHAN account {idx}: "
                        "both client id and access token are required"
                    )
                    continue
                if client_id in seen_dhan:
                    logger.warning(f"[MultiAccountExecution] Skipping duplicate DHAN account {idx}: {client_id}")
                    continue
                seen_dhan.add(client_id)
                accounts.append(
                    ExecutionAccount(
                        label=f"dhan:{client_id}",
                        broker=DhanBroker(client_id=client_id, access_token=token, account_label=f"dhan:{client_id}"),
                        primary=False,
                    )
                )
                logger.info(f"[MultiAccountExecution] Added Dhan account ({client_id}) for parallel order execution.")
        except Exception as exc:
            logger.warning(f"[MultiAccountExecution] Error scanning Dhan accounts: {exc}")

    # Fallback to primary if all secondary accounts failed or were empty
    if not accounts:
        accounts.append(
            ExecutionAccount(
                label=primary_label,
                broker=primary_broker,
                primary=True,
            )
        )

    logger.info(
        "[MultiAccountExecution] Configured execution accounts | "
        f"accounts={[account.label for account in accounts]}"
    )
    _CACHED_EXECUTION_ACCOUNTS = accounts
    _CACHED_PRIMARY_BROKER = primary_broker
    return accounts


async def place_market_order_parallel(
    accounts: Iterable[ExecutionAccount],
    *,
    symbol: str,
    quantity: int,
    transaction: str,
    product: str,
    exchange: str,
    attempts: int,
    retry_delay_sec: float,
) -> tuple[OrderResult, list[dict]]:
    account_list = list(accounts)
    tasks = [
        _place_with_retry(
            account,
            symbol=symbol,
            quantity=quantity,
            transaction=transaction,
            product=product,
            exchange=exchange,
            attempts=attempts,
            retry_delay_sec=retry_delay_sec,
        )
        for account in account_list
    ]
    results = await asyncio.gather(*tasks)
    placed = [row for row in results if row["status"] == "PLACED"]
    failed = [row for row in results if row["status"] != "PLACED"]

    if placed:
        status = "PLACED"
        order_id = ",".join(row["order_id"] for row in placed if row.get("order_id"))
        message = ""
        if failed:
            message = "partial account failure: " + "; ".join(
                f"{row['account']}={row.get('message') or row['status']}" for row in failed
            )
        return (
            OrderResult(order_id, symbol, quantity, "MARKET", status, message),
            results,
        )

    message = "; ".join(f"{row['account']}={row.get('message') or row['status']}" for row in failed)
    return (
        OrderResult("", symbol, quantity, "MARKET", "REJECTED", message or "all account orders failed"),
        results,
    )


async def _place_with_retry(
    account: ExecutionAccount,
    *,
    symbol: str,
    quantity: int,
    transaction: str,
    product: str,
    exchange: str,
    attempts: int,
    retry_delay_sec: float,
) -> dict:
    last_result: OrderResult | None = None
    for attempt in range(1, attempts + 1):
        result = await asyncio.to_thread(
            account.broker.place_market_order,
            symbol=symbol,
            quantity=quantity,
            transaction=transaction,
            product=product,
            exchange=exchange,
        )
        last_result = result
        if result.status == "PLACED":
            return _result_row(account, result, attempt)
        logger.warning(
            "[MultiAccountExecution] Attempt {}/{} failed | account={} {} {} qty={} | {}",
            attempt,
            attempts,
            account.label,
            transaction,
            symbol,
            quantity,
            result.message,
        )
        if attempt < attempts:
            await asyncio.sleep(retry_delay_sec)

    fallback = last_result or OrderResult("", symbol, quantity, "MARKET", "ERROR", "no order result")
    return _result_row(account, fallback, attempts)


def _result_row(account: ExecutionAccount, result: OrderResult, attempts: int) -> dict:
    return {
        "account": account.label,
        "primary": account.primary,
        "order_id": result.order_id,
        "symbol": result.symbol,
        "quantity": result.quantity,
        "status": result.status,
        "message": result.message,
        "attempts": attempts,
    }

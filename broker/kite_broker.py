"""
broker/kite_broker.py — Zerodha KiteConnect Implementation
===========================================================
Production broker. Use for live trading (Phase 2 onwards).
Requires: pip install kiteconnect
Cost: Rs 500/month for API access
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from kiteconnect import KiteConnect

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from broker.base_broker import BaseBroker, OrderResult, PositionInfo
from config.settings import (
    KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN, DATA_CACHE_DIR
)
import pytz

IST = pytz.timezone("Asia/Kolkata")


class KiteBroker(BaseBroker):
    """Zerodha KiteConnect broker implementation."""

    def __init__(self) -> None:
        self._kite = KiteConnect(api_key=KITE_API_KEY)
        if KITE_ACCESS_TOKEN:
            self._kite.set_access_token(KITE_ACCESS_TOKEN)
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        logger.info("[KiteBroker] Initialized.")

    @property
    def broker_name(self) -> str:
        return "kite"

    # ── AUTH ──────────────────────────────────────────────────────────────────

    def get_login_url(self) -> str:
        return self._kite.login_url()

    def generate_session(self, auth_code: str) -> str:
        data  = self._kite.generate_session(auth_code, api_secret=KITE_API_SECRET)
        token = data["access_token"]
        self._kite.set_access_token(token)
        self._save_token_to_env(token)
        logger.info(f"[KiteBroker] Session generated for {data.get('user_name')}")
        return token

    def set_access_token(self, token: str) -> None:
        self._kite.set_access_token(token)

    # ── MARKET DATA ───────────────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> float:
        try:
            q = self._kite.quote([symbol])
            return float(q[symbol]["last_price"])
        except Exception as e:
            logger.warning(f"[KiteBroker] get_ltp({symbol}): {e}")
            return 0.0

    def get_historical_data(
        self, symbol: str, interval: str,
        from_date: str, to_date: str,
    ) -> pd.DataFrame:
        token = self.get_instrument_key(symbol)
        raw   = self._kite.historical_data(
            instrument_token = int(token),
            from_date        = from_date,
            to_date          = to_date,
            interval         = interval,
            continuous       = False,
            oi               = False,
        )
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw).rename(columns={"date": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.set_index("datetime").sort_index()

    def get_option_ltp(self, option_symbol: str) -> float:
        try:
            key = f"NFO:{option_symbol}"
            q   = self._kite.quote([key])
            return float(q[key]["last_price"])
        except Exception as e:
            logger.warning(f"[KiteBroker] option_ltp({option_symbol}): {e}")
            return 0.0

    def get_india_vix(self) -> float:
        try:
            q = self._kite.quote(["NSE:INDIA VIX"])
            return float(q["NSE:INDIA VIX"]["last_price"])
        except Exception:
            return 0.0

    # ── INSTRUMENT LOOKUP ─────────────────────────────────────────────────────

    def get_instrument_key(self, symbol: str, exchange: str = "NSE") -> str:
        cache = Path(DATA_CACHE_DIR) / f"instruments_{exchange}.parquet"
        try:
            if not cache.exists():
                df = pd.DataFrame(self._kite.instruments(exchange))
                df.to_parquet(cache)
            else:
                df = pd.read_parquet(cache)
            row = df[df["tradingsymbol"] == symbol]
            if row.empty:
                raise ValueError(f"Symbol not found: {symbol}")
            return str(int(row.iloc[0]["instrument_token"]))
        except Exception as e:
            logger.error(f"[KiteBroker] get_instrument_key({symbol}): {e}")
            return ""

    def get_option_instrument_key(
        self, symbol: str, expiry: str,
        strike: int, option_type: str,
    ) -> str:
        # Kite uses tradingsymbol directly e.g. NIFTY25MAR2722500CE
        from utils.option_utils import build_option_symbol
        from datetime import date
        expiry_date = date.fromisoformat(expiry)
        return build_option_symbol(symbol, expiry_date, strike, option_type)

    # ── ORDER MANAGEMENT ──────────────────────────────────────────────────────

    def place_market_order(
        self, symbol: str, quantity: int,
        transaction: str, product: str = "MIS",
        exchange: str = os.getenv("EXCHANGE", "MCX"),
    ) -> OrderResult:
        try:
            order_id = self._kite.place_order(
                variety          = "regular",
                exchange         = exchange,
                tradingsymbol    = symbol,
                transaction_type = transaction,
                quantity         = quantity,
                order_type       = "MARKET",
                product          = product,
            )
            logger.success(f"[KiteBroker] Order placed: {symbol} | id={order_id}")
            return OrderResult(
                order_id   = str(order_id),
                symbol     = symbol,
                quantity   = quantity,
                order_type = "MARKET",
                status     = "PLACED",
            )
        except Exception as e:
            logger.error(f"[KiteBroker] place_order failed: {e}")
            return OrderResult(
                order_id="", symbol=symbol, quantity=quantity,
                order_type="MARKET", status="ERROR", message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._kite.cancel_order(variety="regular", order_id=order_id)
            return True
        except Exception as e:
            logger.error(f"[KiteBroker] cancel_order({order_id}): {e}")
            return False

    def get_positions(self) -> list[PositionInfo]:
        try:
            raw = self._kite.positions().get("day", [])
            return [
                PositionInfo(
                    symbol    = p["tradingsymbol"],
                    quantity  = p["quantity"],
                    avg_price = p["average_price"],
                    ltp       = p["last_price"],
                    pnl       = p["pnl"],
                    product   = p["product"],
                )
                for p in raw if p["quantity"] != 0
            ]
        except Exception as e:
            logger.error(f"[KiteBroker] get_positions: {e}")
            return []

    # ── HELPER ────────────────────────────────────────────────────────────────

    @staticmethod
    def _save_token_to_env(token: str) -> None:
        env = Path(".env")
        if not env.exists():
            return
        lines = env.read_text().splitlines()
        updated = []
        found = False
        for line in lines:
            if line.startswith("KITE_ACCESS_TOKEN="):
                updated.append(f"KITE_ACCESS_TOKEN={token}")
                found = True
            else:
                updated.append(line)
        if not found:
            updated.append(f"KITE_ACCESS_TOKEN={token}")
        env.write_text("\n".join(updated) + "\n")

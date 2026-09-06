from config.settings.modules.system_thresholds import *
"""
broker/groww_broker.py — Groww API Implementation
===================================================
Uses official growwapi Python SDK.
Auth: API Key + TOTP (no browser OAuth redirect needed).
Cost: Rs 499 + taxes/month

Key differences from Kite/Upstox:
  - No OAuth redirect — TOTP-based auth is fully programmatic
  - groww_symbol format: "NSE-NIFTY-27Mar25-22500-CE"
  - Historical API returns epoch timestamps for old API,
    ISO format for new backtesting API
  - Segment constants: SEGMENT_CASH, SEGMENT_FNO, SEGMENT_COMMODITY
  - Live feed via GrowwFeed (WebSocket-based)

Docs: https://groww.in/trade-api/docs/python-sdk
"""

import os
import time
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
import pytz

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from broker.base_broker import BaseBroker, OrderResult, PositionInfo
from config.settings import DATA_CACHE_DIR

IST = pytz.timezone("Asia/Kolkata")

# Groww symbol format for common indices
GROWW_SYMBOL_MAP = {
    "NIFTY":     "NSE-NIFTY",
    "NIFTY 50":  "NSE-NIFTY",
    "BANKNIFTY": "NSE-BANKNIFTY",
    "INDIA VIX": "NSE-INDIA VIX",
    "SENSEX":    "BSE-SENSEX",
}

# Groww interval format
INTERVAL_MAP = {
    "1minute":  1,
    "5minute":  5,
    "15minute": 15,
    "30minute": 30,
    "60minute": 60,
    "day":      1440,
}


class GrowwBroker(BaseBroker):
    """
    Groww API broker implementation using official Python SDK.
    Auth is TOTP-based — fully automated, no browser required.
    """

    def __init__(self) -> None:
        self._api_key      = os.getenv("GROWW_API_KEY", "")
        self._api_secret   = os.getenv("GROWW_API_SECRET", "")
        self._totp_secret  = os.getenv("GROWW_TOTP_SECRET", "")
        self._access_token = os.getenv("GROWW_ACCESS_TOKEN", "")
        self._client       = None
        self._feed         = None

        os.makedirs(DATA_CACHE_DIR, exist_ok=True)

        if self._access_token:
            self._init_client()
        logger.info("[GrowwBroker] Initialized.")

    @property
    def broker_name(self) -> str:
        return "groww"

    # ── AUTH ──────────────────────────────────────────────────────────────────

    def get_login_url(self) -> str:
        """
        Groww doesn't use OAuth redirect.
        Returns the Groww Cloud API keys page URL instead.
        """
        return "https://groww.in/trade-api"

    def generate_session(self, auth_code: str = "") -> str:
        """
        Generate access token using TOTP.
        auth_code is ignored — TOTP is auto-generated from TOTP secret.
        """
        try:
            import pyotp
            from growwapi import GrowwAPI as GrowwSDK
        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}\n"
                "Run: pip install growwapi pyotp"
            )

        if not self._totp_secret:
            raise ValueError(
                "GROWW_TOTP_SECRET not set in .env\n"
                "Get from: Groww Cloud → Generate TOTP Token"
            )

        totp     = pyotp.TOTP(self._totp_secret).now()
        token    = GrowwSDK.get_access_token(api_key=self._api_key, totp=totp)

        if not token:
            raise ValueError("Empty access token returned from Groww")

        self._access_token = token
        self._init_client()
        self._save_token_to_env(token)
        logger.info("[GrowwBroker] Session generated via TOTP.")
        return token

    def set_access_token(self, token: str) -> None:
        self._access_token = token
        self._init_client()

    def _init_client(self) -> None:
        try:
            from growwapi import GrowwAPI as GrowwSDK, GrowwFeed
            self._client = GrowwSDK(self._access_token)
            self._feed   = GrowwFeed(self._client)
            logger.info("[GrowwBroker] SDK client initialized.")
        except ImportError:
            logger.warning(
                "[GrowwBroker] growwapi not installed. "
                "Run: pip install growwapi"
            )
            self._client = None

    # ── MARKET DATA ───────────────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> float:
        """Get last traded price via Groww Feed (WebSocket)."""
        try:
            from growwapi import GrowwAPI as GrowwSDK
            exchange_token = "NIFTY" if "NIFTY" in symbol.upper() else symbol
            self._feed.subscribe_index_value(
                [{"exchange": "NSE", "segment": "CASH",
                  "exchange_token": exchange_token}]
            )
            time.sleep(1.5)
            data = self._feed.get_index_value()
            if data and "NSE" in data:
                return float(
                    data["NSE"].get("CASH", {})
                    .get(exchange_token, {})
                    .get("value", 0)
                )
            return 0.0
        except Exception as e:
            logger.warning(f"[GrowwBroker] get_ltp({symbol}): {e}")
            return self._get_ltp_via_rest(symbol)

    def _get_ltp_via_rest(self, symbol: str) -> float:
        """Fallback: get LTP via REST API."""
        try:
            import requests
            exchange_sym = f"NSE_{symbol.replace(' ', '_').upper()}"
            resp = requests.get(
                "https://api.groww.in/v1/live-data/ohlc",
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "X-API-VERSION": "1.0",
                    "Accept":        "application/json",
                },
                params={"segment": "CASH", "exchange_symbols": exchange_sym},
                timeout=BROKER_GROWW_TIMEOUT_8,
            )
            data = resp.json()
            if resp.status_code == 200 and "payload" in data:
                payload = data["payload"]
                key     = list(payload.keys())[0] if payload else None
                if key:
                    return float(payload[key].get("close", 0))
        except Exception as e:
            logger.warning(f"[GrowwBroker] REST LTP fallback failed: {e}")
        return 0.0

    def get_historical_data(
        self,
        symbol:    str,
        interval:  str,
        from_date: str,
        to_date:   str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles using Groww Backtesting API.
        Groww symbol format: "NSE-NIFTY" for index,
        "NSE-WIPRO" for stock.
        """
        try:
            import requests
            groww_symbol   = GROWW_SYMBOL_MAP.get(symbol, f"NSE-{symbol}")
            interval_mins  = INTERVAL_MAP.get(interval, 5)

            resp = requests.get(
                "https://api.groww.in/v1/historical/candle/range",
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "X-API-VERSION": "1.0",
                    "Accept":        "application/json",
                },
                params={
                    "exchange":      "NSE",
                    "segment":       "CASH",
                    "groww_symbol":  groww_symbol,
                    "start_time":    from_date,
                    "end_time":      to_date,
                    "interval":      interval_mins,
                },
                timeout=BROKER_GROWW_TIMEOUT_30,
            )

            data = resp.json()
            if resp.status_code != 200 or data.get("status") != "SUCCESS":
                logger.error(f"[GrowwBroker] Historical API error: {data}")
                return pd.DataFrame()

            candles = data.get("payload", {}).get("candles", [])
            if not candles:
                return pd.DataFrame()

            records = []
            for c in candles:
                # Groww format: [timestamp_epoch, open, high, low, close, volume]
                ts = pd.to_datetime(c[0], unit="s").tz_localize("UTC").tz_convert(IST)
                records.append({
                    "datetime": ts,
                    "open":     float(c[1]),
                    "high":     float(c[2]),
                    "low":      float(c[3]),
                    "close":    float(c[4]),
                    "volume":   int(c[5]) if len(c) > 5 else 0,
                })

            df = pd.DataFrame(records).set_index("datetime").sort_index()
            return df.between_time("09:15", "15:30")

        except Exception as e:
            logger.error(f"[GrowwBroker] get_historical_data: {e}")
            return pd.DataFrame()

    def get_option_ltp(self, option_symbol: str) -> float:
        """Get LTP for an options contract."""
        try:
            import requests
            # Groww option symbol e.g. NIFTY25MAR2722500CE
            resp = requests.get(
                "https://api.groww.in/v1/live-data/ohlc",
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "X-API-VERSION": "1.0",
                    "Accept":        "application/json",
                },
                params={
                    "segment":          "FNO",
                    "exchange_symbols": f"NSE_{option_symbol}",
                },
                timeout=BROKER_GROWW_TIMEOUT_8,
            )
            data = resp.json()
            if resp.status_code == 200 and "payload" in data:
                payload = data["payload"]
                key     = list(payload.keys())[0] if payload else None
                if key:
                    return float(payload[key].get("close", 0))
        except Exception as e:
            logger.warning(f"[GrowwBroker] get_option_ltp({option_symbol}): {e}")
        return 0.0

    def get_india_vix(self) -> float:
        """Get India VIX current value."""
        try:
            return self.get_ltp("INDIA VIX")
        except Exception:
            return 0.0

    # ── INSTRUMENT LOOKUP ─────────────────────────────────────────────────────

    def get_instrument_key(self, symbol: str, exchange: str = "NSE") -> str:
        """Return Groww symbol string."""
        return GROWW_SYMBOL_MAP.get(symbol, f"{exchange}-{symbol}")

    def get_option_instrument_key(
        self, symbol: str, expiry: str,
        strike: int, option_type: str,
    ) -> str:
        """
        Build Groww option symbol.
        Groww format: NSE-NIFTY-27Mar25-22500-CE
        """
        expiry_date = date.fromisoformat(expiry)
        expiry_str  = expiry_date.strftime("%d%b%y")   # e.g. 27Mar25
        return f"NSE-{symbol}-{expiry_str}-{strike}-{option_type}"

    # ── ORDER MANAGEMENT ──────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol:      str,
        quantity:    int,
        transaction: str,
        product:     str = "NRML",
        exchange:    str = "NSE",
    ) -> OrderResult:
        """
        Place market order via Groww SDK.
        product: "NRML" for FNO (Groww doesn't support MIS for options)
        """
        try:
            from growwapi import GrowwAPI as GrowwSDK

            # Map product codes
            groww_product = "NRML" if product in ("MIS", "NRML", "I") else "NRML"
            segment       = self._client.SEGMENT_FNO

            order_id = self._client.place_order(
                trading_symbol   = symbol,
                exchange         = exchange,
                transaction_type = transaction,
                order_type       = "MARKET",
                quantity         = quantity,
                product          = groww_product,
                segment          = segment,
                price            = 0,
                trigger_price    = 0,
                validity         = "DAY",
            )

            logger.success(f"[GrowwBroker] Order placed: {symbol} | id={order_id}")
            return OrderResult(
                order_id   = str(order_id),
                symbol     = symbol,
                quantity   = quantity,
                order_type = "MARKET",
                status     = "PLACED",
            )
        except Exception as e:
            logger.error(f"[GrowwBroker] place_order failed: {e}")
            return OrderResult(
                order_id="", symbol=symbol, quantity=quantity,
                order_type="MARKET", status="ERROR", message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order(order_id=order_id)
            return True
        except Exception as e:
            logger.error(f"[GrowwBroker] cancel_order({order_id}): {e}")
            return False

    def get_positions(self) -> list[PositionInfo]:
        try:
            from growwapi import GrowwAPI as GrowwSDK
            resp = self._client.get_positions_for_user(
                segment=self._client.SEGMENT_FNO, timeout=BROKER_GROWW_TIMEOUT_5
            )
            positions = []
            if resp and hasattr(resp, "positions"):
                for p in resp.positions or []:
                    if getattr(p, "quantity", 0) != 0:
                        positions.append(PositionInfo(
                            symbol    = getattr(p, "trading_symbol", ""),
                            quantity  = getattr(p, "quantity", 0),
                            avg_price = getattr(p, "average_price", 0),
                            ltp       = getattr(p, "ltp", 0),
                            pnl       = getattr(p, "pnl", 0),
                            product   = getattr(p, "product", "NRML"),
                        ))
            return positions
        except Exception as e:
            logger.error(f"[GrowwBroker] get_positions: {e}")
            return []

    # ── HELPER ────────────────────────────────────────────────────────────────

    @staticmethod
    def _save_token_to_env(token: str) -> None:
        env = Path(".env")
        if not env.exists():
            return
        lines = env.read_text().splitlines()
        updated, found = [], False
        for line in lines:
            if line.startswith("GROWW_ACCESS_TOKEN="):
                updated.append(f"GROWW_ACCESS_TOKEN={token}")
                found = True
            else:
                updated.append(line)
        if not found:
            updated.append(f"GROWW_ACCESS_TOKEN={token}")
        env.write_text("\n".join(updated) + "\n")
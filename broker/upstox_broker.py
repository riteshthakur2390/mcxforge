from __future__ import annotations
from config.settings.modules.system_thresholds import *
"""
broker/upstox_broker.py — Upstox API Implementation
======================================================
Free API — use for testing and paper trading (Phase 1).
Requires: pip install upstox-python-sdk
Docs:     https://upstox.com/developer/api-documentation

Key differences from Kite:
  - Auth: OAuth2 → redirect to localhost callback
  - Instrument key format: "NSE_INDEX|Nifty 50" (not integer token)
  - Historical data: limited to 2 years for intraday
  - Paper trading: supported natively
"""

import os
import re
from typing import Any, Optional, Union
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
import pytz

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from broker.base_broker import BaseBroker, OptionContract, OrderResult, PositionInfo
from broker.yfinance_broker import YFinanceBroker
from config.settings import DATA_CACHE_DIR
from data.historical_store import HistoricalCandleStore
from utils.option_utils import build_option_symbol

def _build_option_symbol_compat(symbol: str, expiry_date: date, strike: int, option_type: str) -> str:
    try:
        from utils.option_utils import build_option_symbol
        return build_option_symbol(symbol, expiry_date, strike, option_type)
    except Exception:
        mon = expiry_date.strftime("%b").upper()
        yy = str(expiry_date.year)[-2:]
        return f"{symbol}{yy}{mon}{int(strike)}{option_type.upper()}"

IST = pytz.timezone("Asia/Kolkata")

# Upstox instrument key mappings for common symbols
# Full list: https://assets.upstox.com/market-quote/instruments/exchange/{NSE,BSE,MCX}.json.gz
UPSTOX_SYMBOL_MAP = {
    # MCX Commodities
    "SILVERM":     "MCX_FO|483080",
    "SILVERMIC":   "MCX_FO|562058",
    "SILVER":      "MCX_FO|495214",
    "GOLD":        "MCX_FO|483079",
    "GOLDM":       "MCX_FO|569003",
    "CRUDEOIL":    "MCX_FO|565899",
    "CRUDEOILM":   "MCX_FO|565900",
    "NATURALGAS":  "MCX_FO|568245",
    "NATGASMINI":  "MCX_FO|568246",
    "NATGAS":      "MCX_FO|568245",
    "NATGASM":     "MCX_FO|568246",
    # Legacy / Index mappings
    "BANKNIFTY":   "NSE_INDEX|Nifty Bank",
    "SENSEX":      "BSE_INDEX|SENSEX",
    "BSE SENSEX":  "BSE_INDEX|SENSEX",
}

UPSTOX_FALLBACK_ONLY_SYMBOLS = {"GIFTNIFTY"}
UPSTOX_INSTRUMENT_URLS = {
    "MCX": "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz",
    "NSE": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "BSE": "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz",
}
UPSTOX_INSTRUMENT_CACHES = {
    exchange: Path(DATA_CACHE_DIR) / f"instruments_upstox_{exchange}.parquet"
    for exchange in UPSTOX_INSTRUMENT_URLS
}

# Upstox interval mappings: HistoryApi accepts only (1minute, 30minute, day, week, month)
INTERVAL_MAP = {
    "1minute":  "1minute",
    "3minute":  "1minute",
    "5minute":  "1minute",
    "15minute": "1minute",
    "30minute": "30minute",
    "day":      "day",
}


class UpstoxBroker(BaseBroker):
    """
    Upstox API broker implementation.
    Free tier — suitable for paper trading and testing.
    """

    def __init__(self) -> None:
        self._api_key      = os.getenv("UPSTOX_API_KEY", "")
        self._api_secret   = os.getenv("UPSTOX_API_SECRET", "")
        self._access_token = os.getenv("UPSTOX_ACCESS_TOKEN", "")
        self._client       = None
        self._token_invalid = False
        self._auth_warning_emitted = False
        self._fallback_broker = None
        self._lot_size_cache: dict[str, int] = {}
        self._instrument_key_cache: dict[str, str] = {}
        self._instruments_cache: dict[str, pd.DataFrame] = {}
        self._option_keys_cache: dict[Any, str] = {}
        self._past_days_history_cache: dict[tuple, list] = {}
        self._preload_master_caches()
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)

        if self._access_token:
            self._init_client()
        else:
            # AUTH LOG: no token at startup — warn so operator knows to run upstox_auth.py
            logger.warning(
                "[UpstoxBroker] AUTHENTICATION WARNING — UPSTOX_ACCESS_TOKEN is not set. "
                "All market data calls will return 0 or empty. "
                "Run: python scripts/upstox_auth.py before starting the system."
            )
        logger.info("[UpstoxBroker] Initialized.")

    def _preload_master_caches(self) -> None:
        """Pre-load instrument lot sizes and keys into memory for 0.001ms order execution and option resolution."""
        try:
            for ex in ["MCX", "NSE", "BSE"]:
                cache_file = UPSTOX_INSTRUMENT_CACHES.get(ex)
                if cache_file and cache_file.exists():
                    df = pd.read_parquet(cache_file)
                    self._instruments_cache[ex] = df
                    self._index_instruments_dataframe(ex, df)
        except Exception as e:
            logger.debug(f"[UpstoxBroker] Master cache preload fail-open: {e}")

    def _index_instruments_dataframe(self, exchange: str, df: pd.DataFrame) -> None:
        """Vectorized in-memory indexing of lot sizes, instrument keys, and options."""
        if df.empty:
            return
        try:
            # Index lot sizes & instrument keys via itertuples (100x faster than iterrows)
            for r in df[["instrument_key", "trading_symbol", "lot_size"]].itertuples(index=False):
                ik = str(r[0] or "")
                ts = str(r[1] or "")
                ls = r[2]
                if ls is not None:
                    try:
                        lot_int = int(float(ls))
                        if lot_int > 0:
                            if ik:
                                self._lot_size_cache[ik] = lot_int
                            if ts:
                                self._lot_size_cache[ts] = lot_int
                    except (ValueError, TypeError):
                        pass
                if ts and ik:
                    self._instrument_key_cache[ts] = ik

            # Pre-index option contracts for instantaneous O(1) resolution
            if "instrument_type" in df.columns:
                opt_mask = df["instrument_type"].astype(str).str.upper().isin(["CE", "PE"])
                opt_df = df[opt_mask].copy()
                if not opt_df.empty:
                    raw_expiry = opt_df["expiry"]
                    numeric_expiry = pd.to_numeric(raw_expiry, errors="coerce")
                    if numeric_expiry.notna().any() and float(numeric_expiry.dropna().median() or 0.0) > 10000000000:
                        opt_df["_exp_date"] = pd.to_datetime(numeric_expiry, errors="coerce", unit="ms").dt.date
                    else:
                        opt_df["_exp_date"] = pd.to_datetime(raw_expiry, errors="coerce").dt.date

                    strike_col = "strike_price" if "strike_price" in opt_df.columns else ("strike" if "strike" in opt_df.columns else "")
                    if strike_col:
                        opt_df["_strike_val"] = pd.to_numeric(opt_df[strike_col], errors="coerce").round(2)
                    else:
                        opt_df["_strike_val"] = 0.0

                    underlying_col = "underlying_symbol" if "underlying_symbol" in opt_df.columns else ("name" if "name" in opt_df.columns else "asset_symbol")
                    opt_df["_underlying_val"] = opt_df[underlying_col].astype(str).str.upper()
                    opt_df["_type_val"] = opt_df["instrument_type"].astype(str).str.upper()
                    opt_df["_ts_val"] = opt_df["trading_symbol"].astype(str).str.upper()

                    cols_to_use = ["_underlying_val", "_exp_date", "_strike_val", "_type_val", "instrument_key", "_ts_val"]
                    if "name" in opt_df.columns and underlying_col != "name":
                        opt_df["_name_val"] = opt_df["name"].astype(str).str.upper()
                        cols_to_use.append("_name_val")

                    for row in opt_df[cols_to_use].itertuples(index=False):
                        und = row[0]
                        exp = row[1]
                        stk = float(row[2])
                        itype = row[3]
                        ikey = str(row[4])
                        tsym = row[5]

                        self._option_keys_cache[(und, exp, stk, itype)] = ikey
                        self._option_keys_cache[tsym] = ikey
                        self._option_keys_cache[tsym.replace(" ", "")] = ikey
                        if len(row) > 6:
                            name_val = row[6]
                            self._option_keys_cache[(name_val, exp, stk, itype)] = ikey
        except Exception as exc:
            logger.debug(f"[UpstoxBroker] Error indexing {exchange} options: {exc}")

    @property
    def broker_name(self) -> str:
        return "upstox"

    # ── AUTH ──────────────────────────────────────────────────────────────────

    def get_login_url(self) -> str:
        """
        Returns Upstox OAuth2 login URL.
        After login, Upstox redirects to:
        http://localhost:8080/?code=AUTH_CODE
        """
        redirect_uri = "http://localhost:8080/"
        return (
            f"https://api.upstox.com/v2/login/authorization/dialog"
            f"?client_id={self._api_key}"
            f"&redirect_uri={redirect_uri}"
            f"&response_type=code"
        )

    def generate_session(self, auth_code: str) -> str:
        """Exchange auth code for access token."""
        import requests
        resp = requests.post(
            "https://api.upstox.com/v2/login/authorization/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "code":          auth_code,
                "client_id":     self._api_key,
                "client_secret": self._api_secret,
                "redirect_uri":  "http://localhost:8080/",
                "grant_type":    "authorization_code",
            },
        )
        data  = resp.json()
        token = data.get("access_token", "")
        if not token:
            # AUTH LOG: broker-level token exchange failure — log before raising
            err_code = data.get("error", data.get("status", "unknown"))
            err_msg = data.get("error_description", data.get("message", "no detail"))
            logger.error(
                f"[UpstoxBroker] AUTHENTICATION FAILED — generate_session() | "
                f"HTTP={resp.status_code} | "
                f"error={err_code} | "
                f"description={err_msg} | "
                f"full_response={data}"
            )
            raise ValueError(
                f"Upstox auth failed | HTTP {resp.status_code} | "
                f"error={err_code} | {err_msg}"
            )

        self._access_token = token
        self._init_client()
        self._save_token_to_env(token)
        logger.info("[UpstoxBroker] Session generated successfully.")
        return token

    def set_access_token(self, token: str) -> None:
        self._access_token = token
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Upstox SDK client with access token."""
        try:
            import upstox_client
            # AUTH LOG: log token prefix so operator can verify the right token loaded
            token_preview = (
                self._access_token[:8] + "..." + self._access_token[-4:]
                if len(self._access_token) > 12
                else "(empty or too short)"
            )
            config = upstox_client.Configuration()
            config.access_token = self._access_token
            self._client = upstox_client.ApiClient(config)
            logger.info("[UpstoxBroker] Client initialized with token.")
        except ImportError:
            logger.warning(
                "[UpstoxBroker] upstox-python-sdk not installed. "
                "Run: pip install upstox-python-sdk"
            )
            self._client = None

    # ── MARKET DATA ───────────────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> float:
        """Get last traded price using Upstox Market Quote API."""
        if symbol in UPSTOX_FALLBACK_ONLY_SYMBOLS:
            return self._fallback_ltp(symbol)
        if not self._can_use_upstox():
            return self._fallback_ltp(symbol)
        try:
            import upstox_client
            key = self.get_instrument_key(symbol)
            api = upstox_client.MarketQuoteApi(self._client)
            resp= api.get_full_market_quote(key, "2.0")
            data= resp.data
            
            resp_key = key.replace("|", ":")
            if data and resp_key in data:
                last_price = float(data[resp_key].last_price or 0.0)
                if last_price > 0:
                    return last_price
            if data and key in data:
                last_price = float(data[key].last_price or 0.0)
                if last_price > 0:
                    return last_price
            if data:
                for k, v in data.items():
                    if hasattr(v, "last_price") and v.last_price:
                        last_price = float(v.last_price or 0.0)
                        if last_price > 0:
                            return last_price
            return self._fallback_ltp(symbol)
        except Exception as e:
            if self._handle_auth_failure("get_ltp", e):
                return self._fallback_ltp(symbol)
            logger.warning(f"[UpstoxBroker] get_ltp({symbol}): {e}")
            return self._fallback_ltp(symbol)

    def get_quotes(self, symbols: list[str]) -> dict[str, float]:
        """Fetch multiple LTPs in a single batch request via Upstox Market Quote API."""
        if not self._can_use_upstox():
            return {sym: self._fallback_ltp(sym) for sym in symbols}
        try:
            import upstox_client
            key_to_sym = {}
            unique_keys = []
            for sym in symbols:
                if sym in UPSTOX_FALLBACK_ONLY_SYMBOLS:
                    continue
                k = self.get_instrument_key(sym)
                key_to_sym[k] = sym
                key_to_sym[k.replace("|", ":")] = sym
                if k not in unique_keys:
                    unique_keys.append(k)

            if not unique_keys:
                return {sym: self._fallback_ltp(sym) for sym in symbols}

            api = upstox_client.MarketQuoteApi(self._client)
            resp = api.get_full_market_quote(",".join(unique_keys), "2.0")
            data = resp.data or {}
            results = {}
            for resp_key, item in data.items():
                lp = float(getattr(item, "last_price", 0.0) or 0.0)
                inst_token = getattr(item, "instrument_token", "") or ""
                sym_str = getattr(item, "symbol", "") or ""
                orig_sym = key_to_sym.get(inst_token) or key_to_sym.get(resp_key)
                if not orig_sym:
                    for s in symbols:
                        if f":{s}" in resp_key or f"|{s}" in resp_key or sym_str.startswith(s):
                            orig_sym = s
                            break
                if orig_sym and lp > 0:
                    results[orig_sym] = lp

            for sym in symbols:
                if sym not in results:
                    results[sym] = self._fallback_ltp(sym)
            return results
        except Exception as e:
            logger.debug(f"[UpstoxBroker] get_quotes batch error, falling back: {e}")
            return {sym: self.get_ltp(sym) for sym in symbols}

    def get_historical_data(
        self, symbol: str, interval: str,
        from_date: str, to_date: str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles using Upstox Historical Data API.
        Returns DataFrame with DatetimeIndex (IST).
        """
        if not self._can_use_upstox():
            return self._fallback_historical(symbol, interval, from_date, to_date)
        try:
            import upstox_client
            key = self.get_instrument_key(symbol)
            
            resample_rule = None
            if interval == "3minute":
                upstox_interval = "1minute"
                resample_rule = "3min"
            elif interval == "5minute":
                upstox_interval = "1minute"
                resample_rule = "5min"
            elif interval == "15minute":
                upstox_interval = "1minute"
                resample_rule = "15min"
            else:
                upstox_interval = INTERVAL_MAP.get(interval, "1minute")

            api      = upstox_client.HistoryApi(self._client)

            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            candles = []

            if to_date >= today_str:
                # If range spans before today, fetch/cache past historical days once
                if from_date < today_str:
                    yesterday_dt = datetime.now(IST) - timedelta(days=1)
                    yesterday_str = yesterday_dt.strftime("%Y-%m-%d")
                    cache_key = (key, upstox_interval, from_date, yesterday_str)
                    if cache_key in self._past_days_history_cache:
                        candles.extend(self._past_days_history_cache[cache_key])
                    else:
                        try:
                            resp = api.get_historical_candle_data1(
                                key,
                                upstox_interval,
                                yesterday_str,
                                from_date,
                                "2.0",
                            )
                            past_candles = resp.data.candles or []
                            if past_candles:
                                self._past_days_history_cache[cache_key] = past_candles
                                candles.extend(past_candles)
                        except Exception as hist_err:
                            logger.debug(f"[UpstoxBroker] Past days fetch failed: {hist_err}")

                # Fetch today's live intraday candles (fast, ~50ms)
                if upstox_interval in ["1minute", "30minute"]:
                    try:
                        intra_resp = api.get_intra_day_candle_data(key, upstox_interval, "2.0")
                        intra_candles = intra_resp.data.candles or []
                        if intra_candles:
                            candles.extend(intra_candles)
                    except Exception as intra_err:
                        logger.warning(f"[UpstoxBroker] Intraday fetch failed: {intra_err}")
                elif not candles:
                    try:
                        resp = api.get_historical_candle_data1(key, upstox_interval, to_date, from_date, "2.0")
                        candles.extend(resp.data.candles or [])
                    except Exception:
                        pass
            else:
                resp = api.get_historical_candle_data1(
                    key,
                    upstox_interval,
                    to_date,
                    from_date,
                    "2.0",
                )
                candles = resp.data.candles or []

            if not candles:
                if interval == "day":
                    logger.debug(
                        f"[UpstoxBroker] Daily historical API returned no candles for {symbol}; "
                        "using yfinance fallback."
                    )
                    return self._fallback_historical(symbol, interval, from_date, to_date)
                return pd.DataFrame()

            df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume", "oi"])
            df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601").dt.tz_convert(IST)
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(int)
            df = df.set_index("datetime").sort_index()
            
            # Deduplicate just in case historical and intraday overlap
            df = df[~df.index.duplicated(keep='last')]
            
            if resample_rule:
                max_source_ts = df.index[-1] if not df.empty else None
                df = df.resample(resample_rule).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).dropna()
                # Drop trailing incomplete candle in live session so candle is only published when complete
                if to_date >= today_str and not df.empty and max_source_ts is not None:
                    minutes_map = {"1min": 1, "3min": 3, "5min": 5, "15min": 15, "30min": 30}
                    req_mins = minutes_map.get(resample_rule, 5)
                    last_bucket_ts = df.index[-1]
                    req_complete_ts = last_bucket_ts + timedelta(minutes=req_mins - 1)
                    if max_source_ts < req_complete_ts:
                        df = df.iloc[:-1]
                
            if interval != "day":
                df = df.between_time("09:00", "23:55")
                
            return df

        except Exception as e:
            if self._handle_auth_failure("get_historical_data", e):
                return self._fallback_historical(symbol, interval, from_date, to_date)
            logger.error(f"[UpstoxBroker] get_historical_data: {e}")
            return pd.DataFrame()

    def get_option_ltp(self, option_symbol: str) -> float:
        """Get LTP for an options contract."""
        if not self._can_use_upstox():
            return 0.0
        try:
            import upstox_client
            key = self._resolve_option_symbol_to_key(option_symbol)
            if key:
                api = upstox_client.MarketQuoteApi(self._client)
                resp= api.get_full_market_quote(key, "2.0")
                data= resp.data
                resp_key = key.replace("|", ":")
                if data and resp_key in data:
                    return float(data[resp_key].last_price)
                elif data and key in data:
                    return float(data[key].last_price)
            
            # Fallback A: Try direct symbol format if key resolution failed (robustness)
            try:
                api = upstox_client.MarketQuoteApi(self._client)
                prefixes = ("MCX_FO:", "NSE_FO:") if any(k in str(option_symbol).upper() for k in ("SILVER", "GOLD", "CRUDE", "NAT")) else ("NSE_FO:", "MCX_FO:")
                for prefix in prefixes:
                    key_alt = f"{prefix}{option_symbol}"
                    resp = api.get_full_market_quote(key_alt, "2.0")
                    if resp.data and key_alt in resp.data:
                        lp = float(resp.data[key_alt].last_price or 0.0)
                        if lp > 0:
                            return lp
            except Exception:
                pass

            # Fallback B: Extract info and use option chain (slow but reliable for missing master-list keys)
            m = re.match(r"^([A-Z]+)(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$", str(option_symbol).upper())
            if m:
                sym, exp_part, strike, opt_t = m.groups()
                exp_dt = datetime.strptime(exp_part, "%y%b%d").date()
                chain = self.get_option_contracts(sym, exp_dt, opt_t, [int(strike)])
                if chain and chain[0].last_price > 0:
                    return float(chain[0].last_price)

            return 0.0
        except Exception as e:
            if self._handle_auth_failure("get_option_ltp", e):
                return 0.0
            logger.warning(f"[UpstoxBroker] get_option_ltp({option_symbol}): {e}")
            return 0.0

    def get_india_vix(self) -> float:
        """Get India VIX current value."""
        try:
            return self.get_ltp("INDIA VIX")
        except Exception:
            return 0.0

    # ── INSTRUMENT LOOKUP ─────────────────────────────────────────────────────

    def get_instrument_key(self, symbol: str, exchange: str = "MCX") -> str:
        """Return Upstox instrument key string."""
        if symbol in UPSTOX_SYMBOL_MAP:
            return UPSTOX_SYMBOL_MAP[symbol]
        if symbol in self._instrument_key_cache:
            return self._instrument_key_cache[symbol]
        if exchange in ("MCX", "MCX_FO", "MCX_COMM"):
            return f"MCX_FO|{symbol}"
        return f"{exchange}_EQ|{symbol}"

    def get_option_instrument_key(
        self, symbol: str, expiry: str | date,
        strike: int, option_type: str,
    ) -> str:
        """Return Upstox instrument key for an options contract."""
        expiry_date = date.fromisoformat(expiry) if isinstance(expiry, str) else expiry
        trading_sym = _build_option_symbol_compat(symbol, expiry_date, strike, option_type)
        return self._resolve_option_instrument_key(
            symbol=symbol,
            expiry_date=expiry_date,
            strike=int(strike),
            option_type=option_type,
            trading_symbol=trading_sym,
        ) or f"NSE_FO|{trading_sym}"

    # ── ORDER MANAGEMENT ──────────────────────────────────────────────────────

    def place_market_order(
        self, symbol: str, quantity: int,
        transaction: str, product: str = "I",
        exchange: str = "MCX_FO",
    ) -> OrderResult:
        """
        Place market order via Upstox.
        product: "I" = intraday, "D" = delivery
        """
        if not self._can_use_upstox():
            return OrderResult(
                order_id="",
                symbol=symbol,
                quantity=quantity,
                order_type="MARKET",
                status="ERROR",
                message="Upstox session unavailable. Run `python3 scripts/upstox_auth.py`.",
            )
        try:
            import upstox_client

            # Upstox uses different product codes
            # MIS → I (intraday), NRML → D (delivery)
            upstox_product = "I" if product in ("MIS", "I") else "D"

            # Resolve proper instrument key for commodity futures / options
            instrument_token = ""
            resolved_key = self._resolve_option_symbol_to_key(symbol) if (symbol.endswith("CE") or symbol.endswith("PE")) else None
            if not resolved_key and (symbol.endswith("CE") or symbol.endswith("PE")):
                # Cross-resolution from Dhan/NSE token
                try:
                    from broker.factory import get_broker
                    dhan_b = get_broker()
                    if dhan_b and dhan_b != self:
                        token = dhan_b.get_instrument_key(symbol)
                        if str(token).isdigit():
                            resolved_key = f"NSE_FO|{token}"
                            logger.info(
                                f"[UpstoxBroker] Cross-resolved option token from Dhan master | "
                                f"{symbol} → {resolved_key}"
                            )
                except Exception as exc:
                    logger.debug(f"[UpstoxBroker] Cross-resolution attempt failed: {exc}")

            if resolved_key:
                instrument_token = resolved_key
                logger.debug(
                    f"[UpstoxBroker] Resolved instrument key | "
                    f"{symbol} → {instrument_token}"
                )
            else:
                # Fallback: check if symbol is already a valid instrument key
                if "|" in symbol:
                    instrument_token = symbol
                elif symbol in self._instrument_key_cache:
                    instrument_token = self._instrument_key_cache[symbol]
                elif symbol in UPSTOX_SYMBOL_MAP:
                    instrument_token = UPSTOX_SYMBOL_MAP[symbol]
                else:
                    upstox_exchange = exchange.replace("NFO", "NSE_FO").replace("BFO", "BSE_FO").replace("MCX", "MCX_FO")
                    instrument_token = f"{upstox_exchange}|{symbol}"

            if (symbol.endswith("CE") or symbol.endswith("PE")) and not (
                instrument_token.startswith("NSE_FO|") or instrument_token.startswith("BSE_FO|") or instrument_token.startswith("MCX_FO|")
            ):
                err_msg = f"Cannot place Upstox option order: unresolved instrument token '{instrument_token}' for {symbol}"
                logger.error(f"[UpstoxBroker] Pre-flight validation FAILED | {err_msg}")
                return OrderResult(
                    order_id="",
                    symbol=symbol,
                    quantity=quantity,
                    order_type="MARKET",
                    status="REJECTED",
                    message=err_msg,
                )

            # Pre-flight lot size validation & alignment via RAM cache (< 0.001 ms)
            try:
                lot_val = (
                    self._lot_size_cache.get(instrument_token)
                    or self._lot_size_cache.get(symbol)
                    or (1 if "SILVERMIC" in symbol.upper() else (100 if "CRUDEOIL" in symbol.upper() else (1 if "GOLD" in symbol.upper() else 0)))
                )
                if lot_val > 0 and quantity % lot_val != 0:
                    aligned_qty = max(1, round(quantity / lot_val)) * lot_val
                    logger.warning(
                        f"[UpstoxBroker] Pre-flight aligned quantity from {quantity} to {aligned_qty} "
                        f"(multiple of lot_size={lot_val}) for {symbol}"
                    )
                    quantity = aligned_qty
            except Exception:
                pass

            order_api = upstox_client.OrderApi(self._client)
            order_req = upstox_client.PlaceOrderRequest(
                quantity         = quantity,
                product          = upstox_product,
                validity         = "DAY",
                price            = 0,
                tag              = "mcxforge",
                instrument_token = instrument_token,
                order_type       = "MARKET",
                transaction_type = transaction,
                disclosed_quantity = 0,
                trigger_price    = 0,
                is_amo           = False,
            )

            resp = order_api.place_order(order_req, "2.0")
            order_id = resp.data.order_id
            logger.success(f"[UpstoxBroker] Order placed: {symbol} | id={order_id}")
            return OrderResult(
                order_id   = str(order_id),
                symbol     = symbol,
                quantity   = quantity,
                order_type = "MARKET",
                status     = "PLACED",
            )
        except Exception as e:
            if self._handle_auth_failure("place_market_order", e) and self._can_use_upstox():
                try:
                    order_api = upstox_client.OrderApi(self._client)
                    resp = order_api.place_order(order_req, "2.0")
                    order_id = resp.data.order_id
                    logger.success(f"[UpstoxBroker] Order placed after dynamic token reload: {symbol} | id={order_id}")
                    return OrderResult(
                        order_id   = str(order_id),
                        symbol     = symbol,
                        quantity   = quantity,
                        order_type = "MARKET",
                        status     = "PLACED",
                    )
                except Exception:
                    pass
            err_str = str(e).replace("{", "{{").replace("}", "}}")
            logger.error(f"[UpstoxBroker] place_order failed: {err_str}")
            return OrderResult(
                order_id="", symbol=symbol, quantity=quantity,
                order_type="MARKET", status="ERROR", message=str(e),
            )

    def get_order_details(self, order_id: str) -> dict:
        """Fetch real-time order status and average fill price from Upstox."""
        if not self._can_use_upstox() or not order_id:
            return {}
        try:
            import requests
            headers = {"Accept": "application/json", "Authorization": f"Bearer {self._access_token}"}
            url = f"https://api.upstox.com/v2/order/details?order_id={order_id}"
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                return r.json().get("data", {})
        except Exception as exc:
            logger.debug(f"[UpstoxBroker] get_order_details failed for {order_id}: {exc}")
        return {}

    def get_today_trades(self) -> list[dict]:
        """Fetch all executed trades for today from Upstox tradebook."""
        if not self._can_use_upstox():
            return []
        try:
            import requests
            headers = {"Accept": "application/json", "Authorization": f"Bearer {self._access_token}"}
            url = "https://api.upstox.com/v2/order/trades/get-trades-for-day"
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                return r.json().get("data", [])
        except Exception as exc:
            logger.debug(f"[UpstoxBroker] get_today_trades failed: {exc}")
        return []

    def get_fills_for_symbol(self, symbol: str) -> tuple[Optional[float], Optional[float]]:
        """
        Query Upstox tradebook to return exact (buy_average_price, sell_average_price) for a contract.
        """
        trades = self.get_today_trades()
        if not trades or not symbol:
            return None, None

        import re
        clean_target = re.sub(r"[^A-Z0-9]", "", symbol.upper())
        target_strike_type = re.search(r"(\d{5})(CE|PE)", clean_target)
        key_pattern = target_strike_type.group(0) if target_strike_type else clean_target

        buy_prices, buy_qtys = [], []
        sell_prices, sell_qtys = [], []

        for t in trades:
            tsym = str(t.get("tradingsymbol") or t.get("trading_symbol") or "").upper()
            clean_tsym = re.sub(r"[^A-Z0-9]", "", tsym)
            if key_pattern in clean_tsym or clean_target in clean_tsym or clean_tsym in clean_target:
                px = float(t.get("average_price", 0) or 0)
                qty = int(t.get("quantity", 0) or 0)
                side = str(t.get("transaction_type", "")).upper()
                if side == "BUY" and px > 0 and qty > 0:
                    buy_prices.append(px * qty)
                    buy_qtys.append(qty)
                elif side == "SELL" and px > 0 and qty > 0:
                    sell_prices.append(px * qty)
                    sell_qtys.append(qty)

        buy_avg = round(sum(buy_prices) / sum(buy_qtys), 2) if sum(buy_qtys) > 0 else None
        sell_avg = round(sum(sell_prices) / sum(sell_qtys), 2) if sum(sell_qtys) > 0 else None
        return buy_avg, sell_avg

    def cancel_order(self, order_id: str) -> bool:
        if not self._can_use_upstox():
            return False
        try:
            import upstox_client
            api = upstox_client.OrderApi(self._client)
            api.cancel_order(order_id, "2.0")
            return True
        except Exception as e:
            self._handle_auth_failure("cancel_order", e)
            logger.error(f"[UpstoxBroker] cancel_order({order_id}): {e}")
            return False

    def get_positions(self) -> list[PositionInfo]:
        if not self._can_use_upstox():
            return []
        try:
            import upstox_client
            api  = upstox_client.PortfolioApi(self._client)
            resp = api.get_positions("2.0")
            return [
                PositionInfo(
                    symbol    = p.tradingsymbol,
                    quantity  = p.quantity,
                    avg_price = p.average_price,
                    ltp       = p.last_price,
                    pnl       = p.pnl,
                    product   = p.product,
                )
                for p in (resp.data or []) if p.quantity != 0
            ]
        except Exception as e:
            self._handle_auth_failure("get_positions", e)
            logger.error(f"[UpstoxBroker] get_positions: {e}")
            return []

    def get_option_contracts(
        self,
        symbol: str,
        expiry: date,
        option_type: str,
        strikes: list[int],
    ) -> list[OptionContract]:
        if not self._can_use_upstox():
            return []
        strikes = sorted({int(s) for s in strikes if int(s) > 0})
        if not strikes:
            return []

        trading_symbols = {
            strike: _build_option_symbol_compat(symbol, expiry, strike, option_type)
            for strike in strikes
        }

        # Prefer the option-chain endpoint for OI/IV/volume snapshots. It does
        # not require resolving weekly trading symbols to instrument keys and is
        # the source needed by OI/IV strategies.
        chain_contracts = self._option_contracts_from_chain(symbol, expiry, option_type, strikes, trading_symbols)
        if self._contracts_have_live_market_data(chain_contracts):
            return chain_contracts

        keys = {
            strike: (
                self._resolve_option_instrument_key(
                    symbol=symbol,
                    expiry_date=expiry,
                    strike=strike,
                    option_type=option_type,
                    trading_symbol=trading_symbol,
                )
                or f"{'BFO' if str(symbol).upper() == 'SENSEX' else ('MCX_FO' if any(k in str(symbol).upper() for k in ('SILVER', 'GOLD', 'CRUDE', 'NAT')) else 'NSE_FO')}|{trading_symbol}"
            )
            for strike, trading_symbol in trading_symbols.items()
        }
        try:
            import upstox_client

            api = upstox_client.MarketQuoteApi(self._client)
            resp = api.get_full_market_quote(",".join(keys.values()), "2.0")
            data = resp.data or {}
            contracts: list[OptionContract] = []
            for strike, key in keys.items():
                quote = (
                    data.get(key.replace("|", ":"))
                    or data.get(key)
                    or data.get(trading_symbols[strike])
                )
                contracts.append(
                    self._option_contract_from_quote(
                        trading_symbols[strike],
                        strike,
                        option_type,
                        expiry,
                        quote,
                    )
                )
            if self._contracts_have_live_market_data(contracts):
                return contracts
            chain_contracts = self._option_contracts_from_chain(symbol, expiry, option_type, strikes, trading_symbols)
            if self._contracts_have_live_market_data(chain_contracts):
                logger.debug(
                    f"[UpstoxBroker] Using option-chain market data for {symbol} "
                    f"{expiry.isoformat()} {option_type}; full quote returned empty LTP/volume/OI"
                )
                return chain_contracts
            return contracts
        except Exception as e:
            if self._handle_auth_failure("get_option_contracts", e):
                return []
            logger.debug(f"[UpstoxBroker] Batched option quote failed, falling back to LTP: {e}")

        chain_contracts = self._option_contracts_from_chain(symbol, expiry, option_type, strikes, trading_symbols)
        if self._contracts_have_live_market_data(chain_contracts):
            return chain_contracts

        contracts: list[OptionContract] = []
        for strike, trading_symbol in trading_symbols.items():
            last_price = self.get_option_ltp(trading_symbol)
            contracts.append(
                OptionContract(
                    symbol=trading_symbol,
                    strike=strike,
                    option_type=option_type,
                    expiry_date=expiry.isoformat(),
                    last_price=last_price,
                    source="UPSTOX_LTP" if last_price > 0 else "UPSTOX_NO_LTP",
                )
            )
        return contracts

    @staticmethod
    def _contracts_have_live_market_data(contracts: list[OptionContract]) -> bool:
        if not contracts:
            return False
        return any(
            float(c.last_price or 0.0) > 0
            and (int(c.volume or 0) > 0 or int(c.open_interest or 0) > 0)
            for c in contracts
        )

    def _option_contracts_from_chain(
        self,
        symbol: str,
        expiry: date,
        option_type: str,
        strikes: list[int],
        trading_symbols: dict[int, str],
    ) -> list[OptionContract]:
        try:
            import upstox_client

            api = upstox_client.OptionsApi(self._client)
            resp = api.get_put_call_option_chain(
                self.get_instrument_key(symbol),
                expiry.isoformat(),
            )
            rows = resp.data or []
        except Exception as exc:
            logger.debug(f"[UpstoxBroker] Option chain quote fallback failed: {exc}")
            return []

        by_strike: dict[int, Any] = {}
        leg_attr = "call_options" if option_type.upper() == "CE" else "put_options"
        for row in rows:
            try:
                strike = int(float(getattr(row, "strike_price", 0) or 0))
            except Exception:
                continue
            if strike in strikes:
                by_strike[strike] = getattr(row, leg_attr, None)

        contracts: list[OptionContract] = []
        for strike in strikes:
            quote = by_strike.get(int(strike))
            contracts.append(
                self._option_contract_from_quote(
                    trading_symbols[strike],
                    strike,
                    option_type,
                    expiry,
                    quote,
                )
            )
        return contracts

    def _resolve_option_symbol_to_key(self, option_symbol: str) -> str:
        import re

        match = re.match(r"^([A-Z]+)(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$", str(option_symbol or "").upper())
        if not match:
            return ""
        symbol, expiry_part, strike, option_type = match.groups()
        try:
            expiry_date = datetime.strptime(expiry_part, "%y%b%d").date()
        except Exception:
            return ""
        return self._resolve_option_instrument_key(
            symbol=symbol,
            expiry_date=expiry_date,
            strike=int(strike),
            option_type=option_type,
            trading_symbol=str(option_symbol).upper(),
        )

    def _resolve_option_instrument_key(
        self,
        *,
        symbol: str,
        expiry_date: date,
        strike: int,
        option_type: str,
        trading_symbol: str,
    ) -> str:
        sym_clean = str(symbol or "SILVERM").upper().strip()
        opt_type_clean = str(option_type or "").upper().strip()
        stk_float = float(strike)
        ts_clean = str(trading_symbol or "").upper().strip()
        compact_ts = ts_clean.replace(" ", "")

        # 1. Fast O(1) in-memory cache lookup
        cache_key = (sym_clean, expiry_date, stk_float, opt_type_clean)
        if cache_key in self._option_keys_cache:
            return self._option_keys_cache[cache_key]
        if ts_clean in self._option_keys_cache:
            return self._option_keys_cache[ts_clean]
        if compact_ts in self._option_keys_cache:
            return self._option_keys_cache[compact_ts]

        # Check for commodity underlyings where SILVERM/SILVERMIC might map to SILVER in master
        if sym_clean in ("SILVERM", "SILVERMIC"):
            alt_key = ("SILVER", expiry_date, stk_float, opt_type_clean)
            if alt_key in self._option_keys_cache:
                return self._option_keys_cache[alt_key]

        df = self._load_upstox_instruments(symbol=symbol)
        if df.empty or "instrument_key" not in df.columns:
            self._option_keys_cache[cache_key] = ""
            return ""

        work = df
        mask = pd.Series(True, index=work.index)

        if "segment" in work.columns:
            mask &= work["segment"].astype(str).str.upper().str.contains("FO", na=False)
        if "instrument_type" in work.columns:
            mask &= work["instrument_type"].astype(str).str.upper().eq(opt_type_clean)
        if "expiry" in work.columns:
            raw_expiry = work["expiry"]
            numeric_expiry = pd.to_numeric(raw_expiry, errors="coerce")
            if numeric_expiry.notna().any() and float(numeric_expiry.dropna().median() or 0.0) > 10000000000:
                expiry_values = pd.to_datetime(numeric_expiry, errors="coerce", unit="ms").dt.date
            else:
                expiry_values = pd.to_datetime(raw_expiry, errors="coerce").dt.date
            mask &= expiry_values.eq(expiry_date)
        strike_col = "strike_price" if "strike_price" in work.columns else ("strike" if "strike" in work.columns else "")
        if strike_col:
            strike_values = pd.to_numeric(work[strike_col], errors="coerce")
            mask &= strike_values.round(2).eq(stk_float)

        # Enforce underlying match
        name_matched = False
        match_symbols = [sym_clean]
        if sym_clean in ("SILVERM", "SILVERMIC"):
            match_symbols.append("SILVER")
        elif sym_clean == "CRUDEOILM":
            match_symbols.append("CRUDEOIL")
        elif sym_clean == "NATGASMINI":
            match_symbols.append("NATURALGAS")

        for name_col in ("underlying_symbol", "name", "asset_symbol"):
            if name_col in work.columns:
                exact_name_mask = work[name_col].astype(str).str.strip().str.upper().isin(match_symbols)
                if exact_name_mask.any():
                    mask &= exact_name_mask
                    name_matched = True
                    break

        if not name_matched:
            text_cols = [
                col for col in (
                    "trading_symbol", "tradingsymbol", "name", "short_name",
                    "underlying_symbol", "underlying_key"
                )
                if col in work.columns
            ]
            if text_cols:
                text_mask = pd.Series(False, index=work.index)
                for col in text_cols:
                    text = work[col].astype(str).str.upper()
                    text_mask |= text.str.startswith(sym_clean + " ", na=False) | text.str.eq(sym_clean)
                    text_mask |= text.str.replace(" ", "", regex=False).str.contains(compact_ts, na=False)
                mask &= text_mask

        matches = work[mask]
        if matches.empty:
            logger.debug(
                f"[UpstoxBroker] Option instrument key not found | "
                f"{trading_symbol} expiry={expiry_date} strike={strike} type={option_type}"
            )
            self._option_keys_cache[cache_key] = ""
            return ""

        res_key = str(matches.iloc[0].get("instrument_key", "") or "")
        self._option_keys_cache[cache_key] = res_key
        if ts_clean:
            self._option_keys_cache[ts_clean] = res_key
        if compact_ts:
            self._option_keys_cache[compact_ts] = res_key
        return res_key

    def _load_upstox_instruments(self, symbol: str = "SILVERM", force_reload: bool = False) -> pd.DataFrame:
        sym_upper = str(symbol or "SILVERM").upper().strip()
        if sym_upper == "SENSEX":
            exchange = "BSE"
        elif any(k in sym_upper for k in ("SILVER", "GOLD", "CRUDE", "NAT", "MCX")) or sym_upper in UPSTOX_SYMBOL_MAP:
            exchange = "MCX"
        else:
            exchange = "NSE"
        cache_path = UPSTOX_INSTRUMENT_CACHES[exchange]
        url = UPSTOX_INSTRUMENT_URLS[exchange]

        if not force_reload and exchange in self._instruments_cache:
            return self._instruments_cache[exchange]

        if not force_reload and cache_path.exists():
            try:
                import time as _t
                cache_age_hours = (_t.time() - cache_path.stat().st_mtime) / 3600.0
                if cache_age_hours < 12.0:
                    df = pd.read_parquet(cache_path)
                    self._instruments_cache[exchange] = df
                    self._index_instruments_dataframe(exchange, df)
                    return df
                logger.info(
                    f"[UpstoxBroker] Instrument cache is {cache_age_hours:.1f}h old; refreshing {exchange} master"
                )
            except Exception as exc:
                logger.debug(f"[UpstoxBroker] Failed reading Upstox instrument cache: {exc}")

        try:
            import gzip
            import json
            import requests

            resp = requests.get(url, timeout=BROKER_UPSTOX_TIMEOUT_20)
            resp.raise_for_status()
            payload = gzip.decompress(resp.content)
            df = pd.DataFrame(json.loads(payload.decode("utf-8")))
            if not df.empty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)
                self._instruments_cache[exchange] = df
                self._index_instruments_dataframe(exchange, df)
                logger.info(
                    f"[UpstoxBroker] Cached Upstox {exchange} instruments | rows={len(df)} | "
                    f"path={cache_path}"
                )
                return df
        except Exception as exc:
            logger.warning(f"[UpstoxBroker] Upstox instrument master unavailable: {exc}")
            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    self._instruments_cache[exchange] = df
                    self._index_instruments_dataframe(exchange, df)
                    return df
                except Exception:
                    pass
        return pd.DataFrame()

    def _option_contract_from_quote(
        self,
        trading_symbol: str,
        strike: int,
        option_type: str,
        expiry: date,
        quote: Any,
    ) -> OptionContract:
        def pick(*names: str, default: Any = 0) -> Any:
            for name in names:
                if quote is None:
                    break
                if isinstance(quote, dict) and name in quote:
                    value = quote.get(name)
                else:
                    value = getattr(quote, name, None)
                if value not in (None, ""):
                    return value
                market_data = quote.get("market_data") if isinstance(quote, dict) else getattr(quote, "market_data", None)
                if market_data is not None:
                    if isinstance(market_data, dict) and name in market_data:
                        value = market_data.get(name)
                    else:
                        value = getattr(market_data, name, None)
                    if value not in (None, ""):
                        return value
            return default

        greeks = pick("option_greeks", "greeks", default={})

        def pick_greek(*names: str) -> float:
            for name in names:
                if isinstance(greeks, dict):
                    value = greeks.get(name)
                else:
                    value = getattr(greeks, name, None)
                if value not in (None, ""):
                    try:
                        return float(value or 0.0)
                    except Exception:
                        return 0.0
            return 0.0

        def to_float(value: Any) -> float:
            try:
                return float(value or 0.0)
            except Exception:
                return 0.0

        def to_int(value: Any) -> int:
            try:
                return int(float(value or 0))
            except Exception:
                return 0

        last_price = to_float(pick("last_price", "ltp"))
        volume = to_int(pick("volume", "total_traded_volume", "ttv", "volume_traded"))
        oi = to_int(pick("oi", "open_interest"))
        oi_change = to_int(pick("oi_change", "change_oi", "open_interest_change"))
        return OptionContract(
            symbol=trading_symbol,
            strike=int(strike),
            option_type=option_type,
            expiry_date=expiry.isoformat(),
            last_price=last_price,
            bid_price=to_float(pick("bid_price", "best_bid_price")),
            ask_price=to_float(pick("ask_price", "best_ask_price")),
            volume=volume,
            open_interest=oi,
            oi_change=oi_change,
            implied_volatility=pick_greek("iv", "implied_volatility"),
            delta=pick_greek("delta"),
            theta=pick_greek("theta"),
            gamma=pick_greek("gamma"),
            vega=pick_greek("vega"),
            source="UPSTOX_FULL_QUOTE" if last_price > 0 else "UPSTOX_NO_QUOTE",
        )

    # ── HELPER ────────────────────────────────────────────────────────────────

    def _can_use_upstox(self) -> bool:
        if not bool(self._client) or self._token_invalid:
            try:
                from dotenv import load_dotenv
                load_dotenv(override=True)
                fresh_token = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
                if fresh_token and fresh_token != self._access_token:
                    self._access_token = fresh_token
                    self._token_invalid = False
                    self._auth_warning_emitted = False
                    self._init_client()
                    logger.info("[UpstoxBroker] Reloaded fresh UPSTOX_ACCESS_TOKEN from .env")
            except Exception:
                pass
        return bool(self._client) and not self._token_invalid

    def _handle_auth_failure(self, action: str, err: Exception) -> bool:
        if not self._is_invalid_token_error(err):
            return False

        self._token_invalid = True
        self._client = None

        # Attempt dynamic token reload first
        if self._can_use_upstox():
            logger.info(f"[UpstoxBroker] Dynamic token reload recovered during {action}.")
            return True

        if not self._auth_warning_emitted:
            logger.warning(
                f"[UpstoxBroker] Upstox token rejected during {action}. "
                "Run `python3 scripts/upstox_auth.py` to refresh it. "
                "Falling back to yfinance for index market data; option LTP "
                "and order APIs are disabled until re-auth."
            )
            self._auth_warning_emitted = True
        return True

    def _get_fallback_broker(self) -> Optional[BaseBroker]:
        if self._fallback_broker is None:
            try:
                self._fallback_broker = YFinanceBroker()
                logger.info("[UpstoxBroker] yfinance fallback broker initialized for market data.")
            except Exception as exc:
                logger.error(f"[UpstoxBroker] Failed to initialize fallback broker: {exc}")
                self._fallback_broker = False
        return self._fallback_broker if self._fallback_broker is not False else None

    @staticmethod
    def _is_invalid_token_error(err: Exception) -> bool:
        text = str(err).lower()
        return (
            "udapi100050" in text or
            "invalid token" in text or
            "unauthorized" in text or
            "(401)" in text
        )

    def _fallback_ltp(self, symbol: str) -> float:
        fallback = self._get_fallback_broker()
        if fallback is None:
            logger.warning(f"[UpstoxBroker] fallback get_ltp({symbol}) unavailable. Returning 0.0")
            return 0.0
        try:
            price = float(fallback.get_ltp(symbol) or 0.0)
            if price > 0:
                logger.debug(f"[UpstoxBroker] Fallback LTP via yfinance | {symbol}={price:.2f}")
            return price
        except Exception as exc:
            logger.warning(f"[UpstoxBroker] fallback get_ltp({symbol}) failed: {exc}")
            return 0.0

    def _fallback_historical(
        self,
        symbol: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        fallback = self._get_fallback_broker()
        if fallback is None:
            logger.error(f"[UpstoxBroker] fallback get_historical_data unavailable for {symbol} {interval}.")
            return pd.DataFrame()
        try:
            logger.warning(
                f"[UpstoxBroker] Using yfinance fallback for {symbol} {interval} "
                f"({from_date} → {to_date})."
            )
            return fallback.get_historical_data(symbol, interval, from_date, to_date)
        except Exception as exc:
            logger.error(f"[UpstoxBroker] fallback get_historical_data failed: {exc}")
            return pd.DataFrame()

    @staticmethod
    def _save_token_to_env(token: str) -> None:
        env = Path(".env")
        if not env.exists():
            return
        lines = env.read_text().splitlines()
        updated = []
        found = False
        for line in lines:
            if line.startswith("UPSTOX_ACCESS_TOKEN="):
                updated.append(f"UPSTOX_ACCESS_TOKEN={token}")
                found = True
            else:
                updated.append(line)
        if not found:
            updated.append(f"UPSTOX_ACCESS_TOKEN={token}")
        env.write_text("\n".join(updated) + "\n")

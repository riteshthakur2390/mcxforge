from __future__ import annotations
from config.settings.modules.system_thresholds import *
"""
broker/dhan_broker.py — Dhan HQ Broker Implementation
======================================================
Dhan API integration for SignalForge.
Docs: https://dhanhq.co/docs/latest/

KEY DIFFERENCES FROM UPSTOX / KITE:
─────────────────────────────────────
  Auth    : Dhan tokens must be renewed before expiry. Use
            scripts/dhan_renew_tokens.py from cron for configured accounts.
            If a token has already expired, generate a fresh one from the
            Dhan web dashboard and put it in .env.

  LTP     : POST request (not GET) with a security ID, not symbol name.
            NIFTY 50 index → securityId="13", segment=NSE_INDEX
            India VIX      → securityId="21", segment=NSE_INDEX

  History : POST /v2/charts/historical with securityId + exchangeSegment.
            Candle format: [timestamp_epoch_ms, open, high, low, close, volume]

  Orders  : Uses exchangeSegment, productType (INTRADAY/MARGIN),
            orderType (MARKET/LIMIT), validity (DAY/IOC).

  SDK     : pip install dhanhq  (official Python SDK from Dhan)
            Falls back to raw requests if SDK not installed.

SETUP:
─────
  1. Go to https://web.dhan.co → Apps → Generate Token
  2. Copy Client ID and Access Token
  3. Add to .env:
       DHAN_CLIENT_ID=your_client_id_here
       DHAN_ACCESS_TOKEN=your_access_token_here
       BROKER=dhan

  Then schedule scripts/run_dhan_token_renew.sh before token expiry.

ACTIVATION:
───────────
  Set in .env:   BROKER=dhan
  factory.py already auto-discovers this file — no changes needed there.
"""

import os
import json
import base64
import re
import time
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
import pytz

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from broker.base_broker import BaseBroker, OrderResult, PositionInfo, OptionContract
from utils.option_utils import build_option_symbol
from config.settings import DATA_CACHE_DIR

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

IST = pytz.timezone("Asia/Kolkata")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Dhan security ID mappings ─────────────────────────────────────────────────
# Full master list: https://images.dhan.co/api-data/api-scrip-master.csv
DHAN_SECURITY_IDS = {
    "NIFTY":     "13",    # NIFTY 50 index
    "NIFTY 50":  "13",
    "BANKNIFTY": "25",    # Bank NIFTY index
    "BANK NIFTY": "25",
    "INDIA VIX": "21",    # India VIX
    # BSE indices
    "SENSEX": "51",  # BSE SENSEX 30 (BSE_INDEX)
    "BSE SENSEX": "51",
    "SENSEX 30": "51",
    # MCX Commodities & Futures (Active contracts from Dhan scrip master)
    "SILVERM": "483080",      # SILVERM-30Nov2026-FUT
    "SILVERMIC": "562058",    # SILVERMIC-30Nov2026-FUT
    "SILVERMIC-30NOV2026-FUT": "562058",
    "SILVER": "495214",       # SILVER-04Dec2026-FUT
    "GOLD": "483079",         # GOLD-05Oct2026-FUT
    "GOLDM": "569003",        # GOLDM-05Oct2026-FUT
    "CRUDEOIL": "565899",     # CRUDEOIL-21Sep2026-FUT
    "CRUDEOILM": "565900",    # CRUDEOILM-21Sep2026-FUT
    "NATURALGAS": "568245",   # NATURALGAS-25Sep2026-FUT
    "NATGASMINI": "568246",   # NATGASMINI-25Sep2026-FUT
    "NATGAS":     "568246",   # NATGASMINI alias
    "NATGASM":    "568246",   # NATGASMINI alias
}
# Dhan exchange segment for SENSEX options is BSE_FNO
SENSEX_DHAN_INDEX_SEG  = "BSE_INDEX"
SENSEX_DHAN_OPTION_SEG = "BSE_FNO"   # BSE F&O for SENSEX options

# ── Dhan exchange segment constants ──────────────────────────────────────────
class Seg:
    NSE_INDEX = "NSE_INDEX"   # Indices (NIFTY, VIX)
    NSE_EQ    = "NSE_EQ"      # NSE equity
    NSE_FNO   = "NSE_FNO"     # NSE F&O (options/futures)
    BSE_EQ    = "BSE_EQ"      # BSE equity
    MCX_COMM  = "MCX_COMM"    # MCX Commodities (Silver, Gold, Crude, NatGas)

def is_mcx_symbol(symbol: str) -> bool:
    """Return True if symbol is an MCX commodity (Silver, Gold, Crude, NatGas, etc.)."""
    s = str(symbol).upper().strip()
    return (
        s in (
            "SILVERMIC", "SILVER", "SILVERM",
            "GOLD", "GOLDM", "CRUDEOIL", "CRUDEOILM",
            "NATURALGAS", "NATGAS", "NATGASMINI", "NATGASM",
            "COPPER", "ZINC", "ALUMINIUM", "LEAD", "NICKEL",
        )
        or any(k in s for k in ("SILVER", "GOLD", "CRUDE", "NATGAS", "NATURALGAS", "COPPER", "ZINC", "ALUM", "LEAD", "NICKEL"))
        or s.startswith("MCX")
    )

# ── Dhan candle interval map ──────────────────────────────────────────────────
# SignalForge interval string → Dhan interval string
INTERVAL_MAP = {
    "1minute":  "1",
    "5minute":  "5",
    "15minute": "15",
    "30minute": "25",    # Dhan uses 25 for 30-min
    "60minute": "60",
    "day":      "D",
}

# ── Dhan base URL ─────────────────────────────────────────────────────────────
DHAN_BASE_URL = "https://api.dhan.co"


class DhanBroker(BaseBroker):
    """
    Dhan HQ broker implementation.
    Uses a Dhan access token that should be renewed before expiry.
    """

    def __init__(
        self,
        *,
        client_id: str | None = None,
        access_token: str | None = None,
        account_label: str | None = None,
    ) -> None:
        # Read credentials from environment / settings
        self._client_id    = str(client_id or os.getenv("DHAN_CLIENT_ID", "")).strip("'\" ")
        self._access_token = str(access_token or os.getenv("DHAN_ACCESS_TOKEN", "")).strip("'\" ")
        self.account_label = account_label or f"dhan:{self._client_id or 'unknown'}"
        self._dhan         = None   # SDK client (if installed)
        self._option_feed_blocked_until = 0.0
        self._option_chain_cache: dict[tuple[str, str], tuple[float, dict]] = {}
        self._security_id_cache: dict[str, str] = {}
        self._option_ltp_cache: dict[str, tuple[float, float]] = {}
        self._expired_option_ltp_logged: set[str] = set()
        self._option_ltp_blocked_until = 0.0
        self._index_ltp_cache: dict[str, tuple[float, float]] = {}
        self._ltp_network_blocked_until = 0.0
        self._ltp_network_last_log_at = 0.0
        self._auth_failed_until = 0.0
        self._last_env_token_reload_at = 0.0
        self._data_api_blocked_until = 0.0
        self._data_api_last_log_at = 0.0
        self._vix_fallback_last_log_at = 0.0
        self._option_chain_network_last_log_at = 0.0
        self._secondary_broker = None
        self._last_dhan_marketfeed_at = 0.0
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)

        # Persistent HTTP connection pool for sub-50ms broker API round-trips
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self._session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=30,
            max_retries=Retry(total=2, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]),
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Validate credentials and initialise client
        if not self._client_id or not self._access_token:
            # AUTH LOG: missing credentials at startup — tell operator exactly what to do
            logger.warning(
                "[DhanBroker] AUTHENTICATION WARNING — credentials not set. "
                "DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN is missing in .env. "
                "All market data calls will return 0 or empty. "
                "Fix: go to https://web.dhan.co → Apps → Generate Token, "
                "then add DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN to your .env file."
            )
        else:
            self._warn_if_token_client_mismatch()
            self._init_client()

        logger.info(f"[DhanBroker] Initialized | client_id={self._client_id or 'NOT SET'}")

    @property
    def broker_name(self) -> str:
        return "dhan"

    @property
    def name(self) -> str:
        return "dhan"

    @property
    def is_connected(self) -> bool:
        return bool(self._access_token and self._client_id and self._dhan is not None)

    def _get_secondary_broker(self):
        """Lazily initialize and return the secondary broker (Upstox) for seamless failover."""
        if self._secondary_broker is not None:
            return self._secondary_broker
        try:
            from broker.upstox_broker import UpstoxBroker
            upstox = UpstoxBroker()
            if upstox._can_use_upstox():
                self._secondary_broker = upstox
                return self._secondary_broker
        except Exception as exc:
            logger.debug(f"[DhanBroker] Secondary broker (Upstox) unavailable: {exc}")
        return None

    def _recover_from_parquet_cache(self, option_symbol: str) -> float:
        """Recover last known close from local parquet option candles when both brokers are unreachable."""
        try:
            from utils.option_utils import OPTION_SYMBOL_RE
            match = OPTION_SYMBOL_RE.match(str(option_symbol or "").upper().strip())
            if not match:
                return 0.0
            symbol, yy, mon, dd, strike, option_type = match.groups()
            import glob
            import pandas as pd
            patterns = [
                os.path.join(DATA_CACHE_DIR, f"{symbol}_options_5minute_*.parquet"),
                os.path.join(DATA_CACHE_DIR, f"{symbol}*_options_*.parquet"),
            ]
            for pat in patterns:
                for fpath in glob.glob(pat):
                    df = pd.read_parquet(fpath)
                    if df.empty:
                        continue
                    sub = df[
                        (pd.to_numeric(df["strike"], errors="coerce") == int(strike))
                        & (df["option_type"].astype(str).str.upper() == option_type.upper())
                    ]
                    if not sub.empty:
                        close = float(pd.to_numeric(sub.iloc[-1].get("close"), errors="coerce") or 0.0)
                        if close > 0:
                            return close
        except Exception as exc:
            logger.debug(f"[DhanBroker] Parquet recovery failed for {option_symbol}: {exc}")
        return 0.0

    # ── AUTH ──────────────────────────────────────────────────────────────────
    # Dhan does not use this app's OAuth redirect flow. Tokens are generated
    # from the Dhan dashboard and renewed by scripts/dhan_renew_tokens.py.
    # These methods exist to satisfy the BaseBroker interface.

    def get_login_url(self) -> str:
        """
        Dhan does not use OAuth login URL.
        Credentials are generated once from the Dhan web dashboard.
        Returns dashboard URL as a helpful pointer.
        """
        return "https://web.dhan.co/apps  (generate your token here)"

    def generate_session(self, auth_code: str) -> str:
        """
        Dhan does not use auth code exchange.
        Verify the configured token is valid by calling the fund limit API.
        """
        # Try to verify the token is working
        import requests
        try:
            resp = requests.get(
                f"{DHAN_BASE_URL}/v2/fundlimit",
                headers=self._headers(),
                timeout=BROKER_DHAN_TIMEOUT_10,
            )
            if resp.status_code == 200:
                logger.info("[DhanBroker] Token verified via fund limit API.")
                return self._access_token
            else:
                # AUTH LOG: token check failed — include HTTP status and response body
                data = resp.json() if resp.content else {}
                err_msg = data.get("remarks", data.get("message", "no detail"))
                logger.error(
                    f"[DhanBroker] AUTHENTICATION FAILED — generate_session() | "
                    f"Where: GET {DHAN_BASE_URL}/v2/fundlimit | "
                    f"HTTP={resp.status_code} | "
                    f"error={err_msg} | "
                    f"full_response={data}"
                )
                raise ValueError(
                    f"Dhan token invalid | HTTP {resp.status_code} | {err_msg}"
                )
        except requests.exceptions.Timeout:
            logger.error(
                "[DhanBroker] AUTHENTICATION FAILED — Token verification timed out. "
                f"Where: GET {DHAN_BASE_URL}/v2/fundlimit | "
                "Check internet connection."
            )
            raise

    def set_access_token(self, token: str) -> None:
        """Update access token and reinitialise client."""
        self._access_token = token
        self._auth_failed_until = 0.0
        self._data_api_blocked_until = 0.0
        self._init_client()
        logger.info("[DhanBroker] Access token updated.")

    def _is_token_error(self, resp) -> bool:
        if resp is None:
            return False
        code = getattr(resp, "status_code", None)
        if code == 401:
            return True
        if code == 400:
            try:
                text = str(getattr(resp, "text", "") or "").lower()
                if "invalid token" in text or "dh-906" in text or "token expired" in text or "invalid_token" in text:
                    return True
            except Exception:
                pass
        return False

    def _reload_token_from_env(self, reason: str = "") -> bool:
        """Reload Dhan credentials from .env after an authentication failure."""
        now = time.monotonic()
        if now - self._last_env_token_reload_at < 2.0:
            return False
        self._last_env_token_reload_at = now

        env_path = PROJECT_ROOT / ".env"
        if load_dotenv and env_path.exists():
            load_dotenv(env_path, override=True)

        import base64
        import json

        target_client_id = self._client_id
        matched_token = None
        matched_client_id = None

        for k, v in os.environ.items():
            if k.startswith("DHAN_ACCESS_TOKEN") and v.strip():
                val = v.strip()
                try:
                    parts = val.split(".")
                    if len(parts) >= 2:
                        padding = "=" * (-len(parts[1]) % 4)
                        claims = json.loads(base64.urlsafe_b64decode((parts[1] + padding).encode("utf-8")))
                        cid = str(claims.get("dhanClientId") or "").strip()
                        if cid and (cid == target_client_id or not target_client_id):
                            matched_token = val
                            matched_client_id = cid
                            break
                except Exception:
                    pass

        if not matched_token:
            matched_client_id = os.getenv("DHAN_CLIENT_ID", "").strip()
            matched_token = os.getenv("DHAN_ACCESS_TOKEN", "").strip()

        if not matched_client_id or not matched_token:
            logger.warning(
                f"[DhanBroker] Token reload skipped; missing credentials in {env_path}"
            )
            return False

        if matched_client_id == self._client_id and matched_token == self._access_token:
            logger.warning(
                f"[DhanBroker] Token reload found no credential change for client_id={self._client_id}"
                + (f" | {reason}" if reason else "")
            )
            return False

        self._client_id = str(matched_client_id).strip("'\" ")
        self._access_token = str(matched_token).strip("'\" ")
        self._auth_failed_until = 0.0
        self._data_api_blocked_until = 0.0
        self._warn_if_token_client_mismatch()
        self._init_client()
        logger.warning(
            f"[DhanBroker] Reloaded Dhan token from .env for client_id={self._client_id}"
            + (f" | {reason}" if reason else "")
        )
        return True

    def _is_data_api_access_error(self, resp) -> bool:
        try:
            data = resp.json()
        except Exception:
            data = {"raw": getattr(resp, "text", "")}
        text = json.dumps(data, default=str)
        return (
            getattr(resp, "status_code", None) == 401
            and (
                "DH-902" in text
                or "Data APIs not Subscribed" in text
                or "not subscribed to Data APIs" in text
                or "does not have access to Trading APIs" in text
                or "HTTP Status 451" in text
            )
        )

    def _response_snippet(self, resp, limit: int = 160) -> str:
        text = str(getattr(resp, "text", "") or "")[:limit]
        return text.replace("{", "(").replace("}", ")")

    def _mark_data_api_blocked(self, context: str, resp=None) -> None:
        self._data_api_blocked_until = time.monotonic() + 900.0
        now = time.monotonic()
        if now - self._data_api_last_log_at < 300.0:
            return
        self._data_api_last_log_at = now
        detail = ""
        if resp is not None:
            detail = f" | HTTP {getattr(resp, 'status_code', '')}: {self._response_snippet(resp, 180)}"
        logger.warning(
            "[DhanBroker] Dhan Data API access unavailable; using cache/fallback. "
            "Subscribe to Dhan Data APIs or switch market-data broker for live candles."
            f" | {context}{detail}"
        )

    def _init_client(self) -> None:
        """
        Initialise Dhan SDK client.
        Falls back to raw requests if SDK not installed (still fully functional).
        """
        token_preview = (
            self._access_token[:8] + "..." + self._access_token[-4:]
            if len(self._access_token) > 12
            else "(too short or empty)"
        )
        try:
            from dhanhq import DhanContext, dhanhq
            context = DhanContext(self._client_id, self._access_token)
            self._dhan = dhanhq(context)
            logger.info(
                f"[DhanBroker] SDK client initialised | "
                f"client_id={self._client_id} | token={token_preview}"
            )
        except ImportError as exc:
            self._dhan = None
            logger.info(
                "[DhanBroker] dhanhq SDK not installed; using supported raw HTTP "
                f"requests path. Install dependency `dhanhq>=2.2.0` to enable SDK client. ({exc})"
            )
        except Exception as exc:
            self._dhan = None
            logger.info(
                "[DhanBroker] dhanhq SDK client unavailable; using supported raw HTTP "
                f"requests path. {type(exc).__name__}: {exc}"
            )

    def _warn_if_token_client_mismatch(self) -> None:
        try:
            payload = str(self._access_token or "").split(".")[1]
            payload += "=" * (-len(payload) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload.encode("utf-8")))
            token_client_id = str(claims.get("dhanClientId") or "").strip()
        except Exception:
            return
        if token_client_id and token_client_id != str(self._client_id).strip():
            logger.warning(
                "[DhanBroker] DHAN_CLIENT_ID mismatch detected | "
                f"env_client_id={self._client_id} token_client_id={token_client_id}. "
                f"Auto-syncing client_id to {token_client_id} for seamless API authentication."
            )
            self._client_id = token_client_id

    # ── MARKET DATA ───────────────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> float:
        """
        Get last traded price for MCX commodities (SILVERMIC, GOLD, CRUDEOIL, etc.)
        or indices (NIFTY / BANKNIFTY / SENSEX / INDIA VIX).
        Routes to correct exchange segment (MCX_COMM, NSE_INDEX, or BSE_INDEX).
        Endpoint: POST /v2/marketfeed/ltp
        """
        sym_upper = symbol.upper().strip()
        is_mcx = is_mcx_symbol(sym_upper)
        security_id = self.get_instrument_key(symbol, exchange="MCX" if is_mcx else "NSE") or DHAN_SECURITY_IDS.get(sym_upper, "")
        if not security_id:
            logger.warning(f"[DhanBroker] get_ltp: unknown symbol '{symbol}'")
            return 0.0
        cached_ltp, _ = self._index_ltp_cache.get(sym_upper, (0.0, 0.0))
        if time.monotonic() < self._data_api_blocked_until:
            if is_mcx and cached_ltp <= 0:
                return self._fallback_yfinance_ltp(sym_upper)
            if sym_upper in ("SENSEX", "BSE SENSEX", "SENSEX 30") and cached_ltp <= 0:
                return self._fallback_yfinance_ltp("SENSEX")
            return cached_ltp
        if time.monotonic() < self._ltp_network_blocked_until:
            if is_mcx and cached_ltp <= 0:
                return self._fallback_yfinance_ltp(sym_upper)
            if sym_upper in ("SENSEX", "BSE SENSEX", "SENSEX 30") and cached_ltp <= 0:
                return self._fallback_yfinance_ltp("SENSEX")
            return cached_ltp
        import requests
        max_retries = 2
        quote_timeout = min(float(BROKER_DHAN_TIMEOUT_8), 3.0)
        for attempt in range(max_retries):
            try:
                if is_mcx:
                    seg = Seg.MCX_COMM
                elif sym_upper in ("SENSEX", "BSE SENSEX", "SENSEX 30"):
                    seg = SENSEX_DHAN_INDEX_SEG
                else:
                    seg = Seg.NSE_INDEX

                sec_id_val = int(security_id) if str(security_id).isdigit() else str(security_id)
                self._last_dhan_marketfeed_at = time.monotonic()
                resp = requests.post(
                    f"{DHAN_BASE_URL}/v2/marketfeed/ltp",
                    headers=self._headers(),
                    json={seg: [sec_id_val]},
                    timeout=quote_timeout,
                )

                if resp.status_code == 429:
                    if sym_upper == "INDIA VIX" or is_mcx:
                        chart_price = self._ltp_from_intraday_chart(sym_upper, str(security_id))
                        if chart_price > 0.0:
                            self._index_ltp_cache[sym_upper] = (chart_price, time.monotonic())
                            return chart_price
                    self._ltp_network_blocked_until = time.monotonic() + 60.0
                    time.sleep(1)
                    continue

                if resp.status_code == 401:
                    if self._is_data_api_access_error(resp):
                        self._mark_data_api_blocked(f"get_ltp({symbol})", resp)
                        return cached_ltp
                    if self._reload_token_from_env(reason=f"get_ltp({symbol})"):
                        continue
                    self._auth_failed_until = time.monotonic() + 60.0
                    logger.warning(
                        f"[DhanBroker] get_ltp({symbol}) authentication failed: {self._response_snippet(resp, 120)}"
                    )
                    return cached_ltp

                if resp.status_code != 200:
                    logger.warning(
                        f"[DhanBroker] get_ltp({symbol}) HTTP {resp.status_code}: {self._response_snippet(resp, 120)}"
                    )
                    return 0.0

                data = resp.json()
                seg_data = data.get("data", {}).get(seg, {})
                if not seg_data:
                    resp = requests.post(
                        f"{DHAN_BASE_URL}/v2/marketfeed/quote",
                        headers=self._headers(),
                        json={seg: [sec_id_val]},
                        timeout=quote_timeout,
                    )
                    if resp.status_code == 200:
                        seg_data = resp.json().get("data", {}).get(seg, {})

                sec_entry = seg_data.get(str(security_id)) or seg_data.get(sec_id_val) or {}
                price = float(sec_entry.get("last_price", 0.0) or 0.0)
                if price == 0.0:
                    price = self._ltp_from_intraday_chart(sym_upper, str(security_id))
                if price == 0.0 and is_mcx:
                    price = self._fallback_yfinance_ltp(sym_upper)
                if price > 0.0:
                    self._index_ltp_cache[sym_upper] = (price, time.monotonic())
                elif sym_upper in ("SENSEX", "BSE SENSEX", "SENSEX 30"):
                    price = self._fallback_yfinance_ltp("SENSEX")
                    if price > 0.0:
                        self._index_ltp_cache[sym_upper] = (price, time.monotonic())
                return price
            except requests.exceptions.ConnectionError as e:
                self._ltp_network_blocked_until = time.monotonic() + 30.0
                now = time.monotonic()
                if now - self._ltp_network_last_log_at >= 30.0:
                    self._ltp_network_last_log_at = now
                    logger.warning(
                        f"[DhanBroker] get_ltp({symbol}) network/DNS unavailable; "
                        f"using cached LTP={cached_ltp or 0.0}. {type(e).__name__}: {e}"
                    )
                return cached_ltp
            except Exception as e:
                logger.warning(f"[DhanBroker] get_ltp({symbol}): {type(e).__name__}: {e}")
                time.sleep(0.5)
        if is_mcx and cached_ltp <= 0:
            return self._fallback_yfinance_ltp(sym_upper)
        if sym_upper in ("SENSEX", "BSE SENSEX", "SENSEX 30") and cached_ltp <= 0:
            return self._fallback_yfinance_ltp("SENSEX")
        return cached_ltp

    def _fallback_yfinance_ltp(self, symbol: str) -> float:
        """Delayed fallback for dashboard-only index/commodity quotes when Dhan returns empty data."""
        try:
            import yfinance as yf

            yf_symbol = {
                "SENSEX": "^BSESN",
                "BSE SENSEX": "^BSESN",
                "SENSEX 30": "^BSESN",
                "INDIA VIX": "^INDIAVIX",
                "VIX": "^INDIAVIX",
                "NIFTY": "^NSEI",
                "BANKNIFTY": "^NSEBANK",
                "SILVER": "SI=F",
                "SILVERMIC": "SI=F",
                "SILVERM": "SI=F",
                "GOLD": "GC=F",
                "GOLDM": "GC=F",
                "CRUDEOIL": "CL=F",
                "CRUDEOILM": "CL=F",
                "NATURALGAS": "NG=F",
            }.get(str(symbol).upper(), "")
            if not yf_symbol:
                return 0.0
            hist = yf.Ticker(yf_symbol).history(period="1d", interval="1m")
            if hist is None or hist.empty:
                return 0.0
            return float(hist["Close"].dropna().iloc[-1])
        except Exception as exc:
            logger.debug(f"[DhanBroker] yfinance fallback LTP failed for {symbol}: {exc}")
            return 0.0

    def _ltp_from_intraday_chart(self, sym_upper: str, security_id: str) -> float:
        try:
            import requests

            is_mcx = is_mcx_symbol(sym_upper)
            if is_mcx:
                chart_seg = Seg.MCX_COMM
                instrument = "FUTCOM"
            elif sym_upper in (
                "NIFTY", "NIFTY 50", "BANKNIFTY", "BANK NIFTY",
                "INDIA VIX", "SENSEX", "BSE SENSEX", "SENSEX 30",
            ):
                chart_seg = "IDX_I"
                instrument = "INDEX"
            else:
                chart_seg = Seg.NSE_EQ
                instrument = "EQUITY"

            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            resp_chart = requests.post(
                f"{DHAN_BASE_URL}/v2/charts/intraday",
                headers=self._headers(),
                json={
                    "securityId": str(security_id),
                    "exchangeSegment": chart_seg,
                    "instrument": instrument,
                    "interval": "1",
                    "fromDate": today_str,
                    "toDate": today_str,
                },
                timeout=BROKER_DHAN_TIMEOUT_8,
            )
            if resp_chart.status_code != 200:
                return 0.0
                return 0.0
            chart_data = resp_chart.json()
            closes = chart_data.get("close") or []
            if closes:
                return float(closes[-1] or 0.0)
        except Exception:
            return 0.0
        return 0.0

    def get_sensex_ltp(self) -> float:
        """Get BSE SENSEX 30 spot price."""
        return self.get_ltp("SENSEX")

    def get_historical_data(
        self,
        symbol:    str,
        interval:  str,
        from_date: str,
        to_date:   str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for NIFTY/SILVERM index in chunks to bypass Dhan limits.
        """
        if str(symbol).upper() in ("SILVERMIC", "SILVER"):
            symbol = "SILVERM"
        start_dt = datetime.strptime(from_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(to_date, "%Y-%m-%d")

        chunk_size = 365 if interval == "day" else 30
        current_start = start_dt
        all_frames = []

        while current_start <= end_dt:
            current_end = min(current_start + timedelta(days=chunk_size), end_dt)
            chunk_df = self._fetch_historical_chunk(
                symbol,
                interval,
                current_start.strftime("%Y-%m-%d"),
                current_end.strftime("%Y-%m-%d"),
            )
            if not chunk_df.empty:
                all_frames.append(chunk_df)

            import time
            time.sleep(0.5)
            current_start = current_end + timedelta(days=1)

        if not all_frames:
            return pd.DataFrame()

        merged = pd.concat(all_frames).sort_index()
        return merged[~merged.index.duplicated(keep="last")]

    def _fetch_historical_chunk(
        self,
        symbol:    str,
        interval:  str,
        from_date: str,
        to_date:   str,
    ) -> pd.DataFrame:
        """Internal helper to fetch a single chunk of historical data with retries."""
        if str(symbol).upper() in ("SILVERMIC", "SILVER"):
            symbol = "SILVERM"
        import time
        import requests

        if time.monotonic() < self._data_api_blocked_until:
            return pd.DataFrame()

        sym_upper = symbol.upper().strip()
        is_mcx = is_mcx_symbol(sym_upper)
        security_id = self.get_instrument_key(symbol, exchange="MCX" if is_mcx else "NSE") or DHAN_SECURITY_IDS.get(sym_upper, "13")
        dhan_interval = INTERVAL_MAP.get(interval, "5")

        if is_mcx:
            exchange_seg = Seg.MCX_COMM
            instrument = "FUTCOM"
        elif sym_upper in ("SENSEX", "BSE SENSEX", "SENSEX 30"):
            exchange_seg = "IDX_I"
            instrument = "INDEX"
        elif sym_upper in ("NIFTY", "NIFTY 50", "BANKNIFTY", "BANK NIFTY", "INDIA VIX"):
            exchange_seg = "IDX_I"
            instrument = "INDEX"
        else:
            exchange_seg = Seg.NSE_EQ
            instrument = "EQUITY"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                is_daily = interval == "day"
                endpoint = "historical" if is_daily else "intraday"
                exchange_value = exchange_seg
                payload = {
                    "securityId":      str(security_id),
                    "exchangeSegment": exchange_value,
                    "instrument":      instrument,
                    "fromDate":        from_date,
                    "toDate":          to_date,
                }
                if not is_daily:
                    payload["interval"] = dhan_interval
                resp = requests.post(
                    f"{DHAN_BASE_URL}/v2/charts/{endpoint}",
                    headers=self._headers(),
                    json=payload,
                    timeout=BROKER_DHAN_TIMEOUT_20,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    if "data" in data and isinstance(data["data"], dict):
                        return self._parse_historical_dict(data["data"])
                    if "open" in data and "close" in data:
                        return self._parse_historical_dict(data)
                    if "candles" in data:
                        return self._parse_historical_candles(data["candles"])
                    return pd.DataFrame()

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After", "")
                    try:
                        wait_seconds = max(float(retry_after), 2.0 * (attempt + 1))
                    except (TypeError, ValueError):
                        wait_seconds = 2.0 * (attempt + 1)
                    logger.warning(
                        f"[DhanBroker] Rate limited for {symbol} {interval} "
                        f"chunk {from_date}; retrying in {wait_seconds:.1f}s "
                        f"({attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_seconds)
                    continue

                if self._is_data_api_access_error(resp):
                    self._mark_data_api_blocked(f"historical {symbol} {interval} chunk {from_date}", resp)
                    return pd.DataFrame()

                if self._is_token_error(resp):
                    if self._reload_token_from_env(reason=f"historical {symbol} {interval} chunk {from_date}"):
                        logger.warning(
                            f"[DhanBroker] Chunk {from_date} got token error ({resp.status_code}); reloaded token and retrying..."
                        )
                        continue
                    self._auth_failed_until = time.monotonic() + 60.0
                    logger.warning(
                        f"[DhanBroker] Chunk {from_date} authentication failed "
                        f"({resp.status_code}): {self._response_snippet(resp, 160)}"
                    )
                    return pd.DataFrame()

                if resp.status_code == 400:
                    try:
                        chunk_day = pd.Timestamp(from_date).date()
                    except Exception:
                        chunk_day = None
                    if interval == "day" and chunk_day and chunk_day.weekday() >= 5:
                        logger.debug(
                            f"[DhanBroker] Skipping non-trading daily chunk "
                            f"{from_date} ({resp.status_code})"
                        )
                        return pd.DataFrame()
                    snippet = self._response_snippet(resp, 160)
                    if "DH-905" in snippet or "DH-907" in snippet or "Data_Error" in snippet or "Input_Exception" in snippet:
                        logger.debug(
                            f"[DhanBroker] Chunk {from_date} unavailable (HTTP 400): {snippet}"
                        )
                    else:
                        logger.warning(
                            f"[DhanBroker] Chunk {from_date} rejected "
                            f"(HTTP 400): {snippet}"
                        )
                    return pd.DataFrame()

                logger.warning(
                    f"[DhanBroker] Chunk {from_date} failed "
                    f"(HTTP {resp.status_code}). Retrying {attempt + 1}/{max_retries}..."
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                logger.error(f"[DhanBroker] Connection error on chunk {from_date}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error("[DhanBroker] Max retries reached. Check your internet/DNS.")
            except Exception as e:
                logger.error(f"[DhanBroker] Unexpected error: {e}")
                break

            time.sleep(2)

        return pd.DataFrame()

    _mcx_options_df = None

    @classmethod
    def _get_mcx_scrip_master(cls) -> pd.DataFrame:
        """Loads and caches Dhan MCX options scrip master."""
        if cls._mcx_options_df is None:
            from pathlib import Path
            p = Path("data/cache/dhan_scrip_master.csv")
            if p.exists():
                try:
                    df = pd.read_csv(p, low_memory=False)
                    cls._mcx_options_df = df[(df["SEM_EXM_EXCH_ID"] == "MCX") & (df["SEM_INSTRUMENT_NAME"] == "OPTFUT")]
                except Exception as exc:
                    logger.warning(f"[DhanBroker] Failed to load cached Dhan scrip master: {exc}")
                    cls._mcx_options_df = pd.DataFrame()
            else:
                cls._mcx_options_df = pd.DataFrame()
        return cls._mcx_options_df

    def get_mcx_option_contracts(
        self,
        symbol: str,
        expiry: "date",
        option_type: str,
        strikes: list[int],
        fetch_live_quotes: bool = True,
    ) -> list[OptionContract]:
        """Fetch exact MCX commodity option contracts from Dhan scrip master and live quotes."""
        if not strikes:
            return []
        sym_upper = str(symbol).upper().strip()
        if sym_upper in ("SILVERM", "SILVERMIC", "SILVER") or "SILVER" in sym_upper:
            names = ["SILVERM"]
        elif "CRUDE" in sym_upper:
            names = ["CRUDEOILM", "CRUDEOIL"]
        elif "GOLD" in sym_upper:
            names = ["GOLDM", "GOLD"]
        elif "NAT" in sym_upper:
            names = ["NATGASMINI", "NATURALGAS"]
        else:
            names = [sym_upper]

        df_mcx = self._get_mcx_scrip_master()
        if df_mcx.empty:
            return []

        opt_type_str = str(option_type).upper()
        sub = df_mcx[(df_mcx["SM_SYMBOL_NAME"].isin(names)) & (df_mcx["SEM_OPTION_TYPE"] == opt_type_str)]
        if sub.empty:
            return []

        # Find matching or nearest active expiry
        exp_filter = None
        if expiry:
            exp_str = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)
            exact_match = sub[sub["SEM_EXPIRY_DATE"].astype(str).str.startswith(exp_str[:10])]
            if not exact_match.empty:
                exp_filter = exp_str[:10]

        if not exp_filter:
            expiries = sorted(sub["SEM_EXPIRY_DATE"].astype(str).unique())
            exp_filter = expiries[0][:10] if expiries else None

        if exp_filter:
            sub = sub[sub["SEM_EXPIRY_DATE"].astype(str).str.startswith(exp_filter)]

        contracts = []
        sec_ids_to_fetch = []
        now_mono = time.monotonic()
        COALESCE_TTL_SEC = 10.0

        for strike in sorted({int(value) for value in strikes if int(value) > 0}):
            row = sub[sub["SEM_STRIKE_PRICE"] == float(strike)]
            if row.empty:
                diffs = (sub["SEM_STRIKE_PRICE"] - float(strike)).abs()
                closest_idx = diffs.idxmin() if not diffs.empty else None
                if closest_idx is not None and diffs.loc[closest_idx] < 500:
                    row = sub.loc[[closest_idx]]
            if not row.empty:
                r = row.iloc[0]
                sec_id = str(r["SEM_SMST_SECURITY_ID"])
                trading_symbol = str(r["SEM_TRADING_SYMBOL"])
                self._security_id_cache[trading_symbol] = sec_id
                self._security_id_cache[f"{sym_upper} {strike} {opt_type_str}"] = sec_id

                # TIER 1: Check in-memory coalescing cache first
                c_cached_val, c_cached_at = self._option_ltp_cache.get(trading_symbol.upper().strip(), (0.0, 0.0))
                initial_ltp = 0.0
                initial_src = "DHAN_SCRIP_MASTER"
                if c_cached_val > 0 and (now_mono - c_cached_at) < COALESCE_TTL_SEC:
                    initial_ltp = c_cached_val
                    initial_src = "DHAN_COALESCED_CACHE"
                elif fetch_live_quotes:
                    sec_ids_to_fetch.append(sec_id)

                contracts.append(OptionContract(
                    symbol=trading_symbol,
                    strike=int(float(r["SEM_STRIKE_PRICE"])),
                    option_type=opt_type_str,
                    expiry_date=str(r["SEM_EXPIRY_DATE"])[:10],
                    last_price=initial_ltp,
                    bid_price=0.0,
                    ask_price=0.0,
                    volume=0,
                    open_interest=0,
                    implied_volatility=0.20,
                    delta=0.50,
                    theta=2.0,
                    source=initial_src,
                ))

        # If ALL contracts hit the coalescing cache, return immediately (ZERO network overhead)
        if fetch_live_quotes and not sec_ids_to_fetch and all(c.last_price > 0 for c in contracts):
            return contracts

        # Try batch fetching live LTPs from Dhan
        ltp_map = {}
        dhan_failed_or_rate_limited = False
        time_since_dhan = time.monotonic() - getattr(self, "_last_dhan_marketfeed_at", 0.0)

        if fetch_live_quotes and sec_ids_to_fetch and self._client_id:
            if time.monotonic() < self._option_feed_blocked_until:
                dhan_failed_or_rate_limited = True
            elif time_since_dhan < 1.0:
                # Proactive rate-limit guard: Dhan allows max 1 req/sec on marketfeed.
                # Since Dhan handled a request < 1.0s ago, route directly to Upstox instead of triggering 429!
                dhan_failed_or_rate_limited = True
            else:
                try:
                    import requests
                    int_sec_ids = [int(s) if str(s).isdigit() else str(s) for s in sec_ids_to_fetch]
                    self._last_dhan_marketfeed_at = time.monotonic()
                    resp = requests.post(
                        f"{DHAN_BASE_URL}/v2/marketfeed/ltp",
                        headers=self._headers(),
                        json={Seg.MCX_COMM: int_sec_ids},
                        timeout=3.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", {}) or {}
                        ltp_map = data.get(Seg.MCX_COMM, {}) or {}
                    elif resp.status_code == 429:
                        self._option_feed_blocked_until = time.monotonic() + 30.0
                        dhan_failed_or_rate_limited = True
                        logger.info(
                            f"[DhanBroker] Dhan marketfeed 429 rate limit reached; pausing Dhan for 30s — Upstox handling active feed"
                        )
                    else:
                        dhan_failed_or_rate_limited = True
                except Exception as exc:
                    dhan_failed_or_rate_limited = True
                    logger.debug(f"[DhanBroker] MCX marketfeed/ltp request failed: {exc}")

        # Update contract prices from live LTP map or cache
        for c in contracts:
            if c.last_price > 0:
                continue
            sec = self._security_id_cache.get(c.symbol)
            c_sym_upper = c.symbol.upper().strip()
            c_ltp = 0.0
            if sec and sec in ltp_map:
                c_ltp = float(ltp_map[sec].get("last_price", 0.0) or 0.0)
            if c_ltp <= 0.0 and sec and str(sec) in ltp_map:
                c_ltp = float(ltp_map[str(sec)].get("last_price", 0.0) or 0.0)
            if c_ltp <= 0.0 and sec and str(sec).isdigit() and int(sec) in ltp_map:
                c_ltp = float(ltp_map[int(sec)].get("last_price", 0.0) or 0.0)
            if c_ltp <= 0.0:
                cached_val, cached_at = self._option_ltp_cache.get(c_sym_upper, (0.0, 0.0))
                if cached_val > 0 and (time.monotonic() - cached_at) < 300.0:
                    c_ltp = cached_val
            if c_ltp > 0:
                c.last_price = c_ltp
                c.source = "DHAN_LIVE"
                self._option_ltp_cache[c_sym_upper] = (c_ltp, time.monotonic())

        # If any contracts still have 0 LTP, attempt /v2/marketfeed/ohlc for last_price / previous close
        missing_sec_ids = [
            int(self._security_id_cache[c.symbol])
            for c in contracts
            if c.last_price <= 0 and c.symbol in self._security_id_cache and str(self._security_id_cache[c.symbol]).isdigit()
        ]
        if missing_sec_ids and fetch_live_quotes and self._client_id and not dhan_failed_or_rate_limited and time.monotonic() >= self._option_feed_blocked_until:
            try:
                import requests
                resp_ohlc = requests.post(
                    f"{DHAN_BASE_URL}/v2/marketfeed/ohlc",
                    headers=self._headers(),
                    json={Seg.MCX_COMM: missing_sec_ids},
                    timeout=3.0,
                )
                if resp_ohlc.status_code == 200:
                    ohlc_data = (resp_ohlc.json().get("data", {}) or {}).get(Seg.MCX_COMM, {}) or {}
                    for c in contracts:
                        if c.last_price > 0:
                            continue
                        sec = str(self._security_id_cache.get(c.symbol, ""))
                        item = ohlc_data.get(sec) or ohlc_data.get(int(sec) if sec.isdigit() else sec) or {}
                        px = float(item.get("last_price", 0.0) or 0.0)
                        if px <= 0:
                            px = float((item.get("ohlc", {}) or {}).get("close", 0.0) or 0.0)
                        if px > 0:
                            c.last_price = px
                            c.source = "DHAN_LIVE"
                            self._option_ltp_cache[c.symbol.upper().strip()] = (px, time.monotonic())
                elif resp_ohlc.status_code == 429:
                    self._option_feed_blocked_until = time.monotonic() + 5.0
                    dhan_failed_or_rate_limited = True
            except Exception:
                pass

        # ── TIER 2: DUAL-BROKER FAILOVER TO UPSTOX IF DHAN FAILED OR WAS 429 ──
        if (dhan_failed_or_rate_limited or any(c.last_price <= 0 for c in contracts)) and fetch_live_quotes:
            upstox = self._get_secondary_broker()
            if upstox and getattr(upstox, "_client", None):
                try:
                    import upstox_client
                    up_sec_ids = [
                        str(self._security_id_cache[c.symbol])
                        for c in contracts
                        if c.last_price <= 0 and c.symbol in self._security_id_cache and str(self._security_id_cache[c.symbol]).isdigit()
                    ]
                    if up_sec_ids:
                        api = upstox_client.MarketQuoteApi(upstox._client)
                        up_keys = [f"MCX_FO|{sec}" for sec in up_sec_ids]
                        resp = api.get_full_market_quote(",".join(up_keys), "2.0")
                        up_data = resp.data or {}
                        recovered_count = 0
                        for c in contracts:
                            if c.last_price > 0:
                                continue
                            sec = str(self._security_id_cache.get(c.symbol, ""))
                            for k, q in up_data.items():
                                if sec in k or str(c.strike) in k:
                                    lp = float(getattr(q, "last_price", 0.0) or 0.0)
                                    if lp > 0:
                                        c.last_price = lp
                                        c.bid_price = float(getattr(q, "bid_price", 0.0) or getattr(q, "buy_price", 0.0) or 0.0)
                                        c.ask_price = float(getattr(q, "ask_price", 0.0) or getattr(q, "sell_price", 0.0) or 0.0)
                                        c.volume = int(getattr(q, "volume", 0) or 0)
                                        c.open_interest = int(getattr(q, "oi", 0) or 0)
                                        c.source = "UPSTOX_FAILOVER"
                                        self._option_ltp_cache[c.symbol.upper().strip()] = (lp, time.monotonic())
                                        recovered_count += 1
                                        break
                        if recovered_count > 0:
                            logger.info(f"[DhanBroker] 🔄 Upstox live failover succeeded for {recovered_count} MCX option contracts")
                except Exception as exc:
                    logger.debug(f"[DhanBroker] Upstox market quote failover failed: {exc}")

        # ── TIER 3: LOCAL PARQUET CANDLE CACHE FALLBACK ───────────────────────
        for c in contracts:
            if c.last_price <= 0:
                recovered = self._recover_from_parquet_cache(c.symbol)
                if recovered > 0:
                    c.last_price = recovered
                    c.source = "PARQUET_STORE"
                    self._option_ltp_cache[c.symbol.upper().strip()] = (recovered, time.monotonic())

        return contracts

    def get_option_contracts(
        self,
        symbol:      str,
        expiry:      "date",
        option_type: str,
        strikes:     list[int],
    ) -> list[OptionContract]:
        if not strikes:
            return []
        symbol_key = str(symbol).upper().strip()

        # Route MCX commodity option contracts to Dhan scrip master
        is_mcx = is_mcx_symbol(symbol_key)
        if is_mcx:
            return self.get_mcx_option_contracts(symbol_key, expiry, option_type, strikes)

        expiry_str = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)
        cache_key = (symbol_key, expiry_str)
        cached_at, chain_data = self._option_chain_cache.get(cache_key, (0.0, {}))
        if not chain_data:
            cached_at, chain_data = self._option_chain_cache.get(f"{symbol_key}_{expiry_str}", (0.0, {}))

        cache_age = time.monotonic() - cached_at
        is_cache_fresh = bool(chain_data and cache_age < 15.0)

        if time.monotonic() < self._option_feed_blocked_until:
            if not chain_data:
                return []
        elif not is_cache_fresh:
            underlying_id = DHAN_SECURITY_IDS.get(symbol_key, "")
            if not underlying_id:
                return []
            option_chain_timeout = min(float(BROKER_DHAN_TIMEOUT_8), 3.0)
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    import requests
                    resp = requests.post(
                        f"{DHAN_BASE_URL}/v2/optionchain",
                        headers=self._headers(),
                        json={
                            "UnderlyingScrip": int(underlying_id),
                            "UnderlyingSeg": "IDX_I",
                            "Expiry": expiry_str,
                        },
                        timeout=option_chain_timeout,
                    )
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After", "")
                        try:
                            cooldown = max(float(retry_after), 3.0)
                        except (TypeError, ValueError):
                            cooldown = 3.0
                        self._option_feed_blocked_until = time.monotonic() + cooldown
                        logger.warning(
                            f"[DhanBroker] Option chain rate limited; "
                            f"pausing option snapshots for {cooldown:.0f}s"
                        )
                        break
                    if resp.status_code == 401:
                        if self._reload_token_from_env(reason="get_option_contracts"):
                            continue
                    if resp.status_code != 200:
                        logger.warning(
                            f"[DhanBroker] get_option_contracts HTTP {resp.status_code}: "
                            f"{self._response_snippet(resp, 160)}"
                        )
                        break
                    fetched_data = resp.json().get("data", {}).get("oc", {}) or {}
                    if fetched_data:
                        chain_data = fetched_data
                        now_mono = time.monotonic()
                        self._option_chain_cache[cache_key] = (now_mono, chain_data)
                        self._option_chain_cache[f"{symbol_key}_{expiry_str}"] = (now_mono, chain_data)
                        self._pre_index_security_ids(symbol_key, expiry, fetched_data)
                    break
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                        continue
                    now = time.monotonic()
                    if now - self._option_chain_network_last_log_at >= 30.0:
                        self._option_chain_network_last_log_at = now
                        logger.warning(
                            f"[DhanBroker] Option chain network/DNS unavailable; using cached chain data. "
                            f"{type(exc).__name__}: {exc}"
                        )
                except Exception as exc:
                    logger.warning(f"[DhanBroker] Option chain request failed: {type(exc).__name__}: {exc}")
                    break

            if not chain_data:
                return []

        side_key = "ce" if str(option_type).upper() == "CE" else "pe"
        logger.debug(
            f"[DhanBroker] Reading {len(strikes)} options from chain for "
            f"{symbol_key} {expiry_str} {option_type}"
        )
        contracts = []
        for strike in sorted({int(value) for value in strikes if int(value) > 0}):
            strike_data = (
                chain_data.get(f"{float(strike):.6f}")
                or chain_data.get(str(strike))
                or chain_data.get(str(float(strike)))
                or {}
            )
            cdata = strike_data.get(side_key) or {}
            if not cdata:
                continue
            greeks = cdata.get("greeks") or {}
            last_price = float(cdata.get("last_price", 0.0) or 0.0)
            contracts.append(OptionContract(
                symbol=build_option_symbol(symbol_key, expiry, strike, option_type),
                strike=strike,
                option_type=str(option_type).upper(),
                expiry_date=expiry_str,
                last_price=last_price,
                bid_price=float(cdata.get("top_bid_price", 0.0) or last_price),
                ask_price=float(cdata.get("top_ask_price", 0.0) or last_price),
                volume=int(cdata.get("volume", 0) or 0),
                open_interest=int(cdata.get("oi", 0) or 0),
                oi_change=int(cdata.get("oi", 0) or 0) - int(cdata.get("previous_oi", 0) or 0),
                implied_volatility=float(cdata.get("implied_volatility", 0.0) or 0.0),
                delta=float(greeks.get("delta", 0.0) or 0.0),
                theta=float(greeks.get("theta", 0.0) or 0.0),
                gamma=float(greeks.get("gamma", 0.0) or 0.0),
                vega=float(greeks.get("vega", 0.0) or 0.0),
                source="DHAN_OPTION_CHAIN",
            ))
        return contracts

    def _pre_index_security_ids(self, symbol: str, expiry: "date", chain_data: dict) -> None:
        """Pre-index option symbol -> security_id for instant zero-latency order execution."""
        if not chain_data:
            return
        symbol_key = str(symbol).upper().strip()
        for strike_key, strike_data in chain_data.items():
            try:
                strike_val = int(float(strike_key))
            except (TypeError, ValueError):
                continue
            for opt_type in ["CE", "PE"]:
                side_key = "ce" if opt_type == "CE" else "pe"
                cdata = (strike_data or {}).get(side_key) or {}
                sec_id = cdata.get("security_id")
                if sec_id:
                    opt_sym = build_option_symbol(symbol_key, expiry, strike_val, opt_type)
                    self._security_id_cache[opt_sym.upper().strip()] = str(sec_id)

    @staticmethod
    def _parse_trading_option_symbol(option_symbol: str) -> tuple[str, date, int, str] | None:
        sym = str(option_symbol).upper().strip()
        month_map = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        }
        # Format 1: Dhan MCX style (e.g. CRUDEOIL-17Sep2026-7000-CE or SILVERM-24Sep2026-285000-PE)
        m1 = re.match(r"^([A-Z0-9]+)-(\d{1,2})([A-Z]{3})(\d{4})-(\d+)-(CE|PE)$", sym)
        if m1:
            root, dd, mon, yyyy, strike, opt_t = m1.groups()
            month = month_map.get(mon)
            if month:
                try:
                    return root, date(int(yyyy), month, int(dd)), int(strike), opt_t
                except ValueError:
                    pass

        # Format 2: Zerodha / NFO style (e.g. NIFTY26AUG1824300CE)
        m2 = re.match(r"^([A-Z0-9]+)(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$", sym)
        if m2:
            symbol, yy, mon, dd, strike, option_type = m2.groups()
            month = month_map.get(mon)
            if month:
                try:
                    return symbol, date(2000 + int(yy), month, int(dd)), int(strike), option_type
                except ValueError:
                    pass

        # Format 3: Space-separated style (e.g. SILVERMIC 85000 CE)
        m3 = re.match(r"^([A-Z0-9]+)\s+(\d+)\s+(CE|PE)$", sym)
        if m3:
            root, strike, opt_t = m3.groups()
            return root, date.today(), int(strike), opt_t

        return None

    def _option_ltp_from_chain_cache(self, option_symbol: str) -> float:
        parsed = self._parse_trading_option_symbol(option_symbol)
        if not parsed:
            return 0.0
        symbol, expiry, strike, option_type = parsed
        cached_at, chain_data = self._option_chain_cache.get((symbol, expiry.isoformat()), (0.0, {}))
        if not chain_data or time.monotonic() - cached_at > 60.0:
            return 0.0
        side_key = "ce" if option_type == "CE" else "pe"
        strike_data = (
            chain_data.get(f"{float(strike):.6f}")
            or chain_data.get(str(strike))
            or chain_data.get(str(float(strike)))
            or {}
        )
        cdata = strike_data.get(side_key) or {}
        try:
            ltp = float(cdata.get("last_price", 0.0) or 0.0)
        except (TypeError, ValueError):
            ltp = 0.0
        if ltp > 0:
            self._option_ltp_cache[str(option_symbol).upper().strip()] = (ltp, time.monotonic())
        return ltp

    def get_option_ltp(self, option_symbol: str) -> float:
        """
        Get LTP for a NIFTY options contract.
        option_symbol: standard format e.g. NIFTY26APR2522500CE

        Dhan requires the numeric security_id for the specific strike.
        We query the NSE_FNO segment with the trading symbol directly.
        """
        option_key = str(option_symbol).upper().strip()
        cached_ltp, cached_at = self._option_ltp_cache.get(option_key, (0.0, 0.0))
        if cached_ltp > 0 and time.monotonic() - cached_at <= 300.0:
            return cached_ltp
        parsed = self._parse_trading_option_symbol(option_key)
        if parsed:
            symbol, expiry, strike, option_type = parsed
            if expiry < datetime.now(IST).date():
                if option_key not in self._expired_option_ltp_logged:
                    self._expired_option_ltp_logged.add(option_key)
                    logger.debug(f"[DhanBroker] Skipping expired option LTP lookup: {option_key}")
                return cached_ltp
            chain_ltp = self._option_ltp_from_chain_cache(option_key)
            if chain_ltp > 0:
                return chain_ltp
            contracts = self.get_option_contracts(symbol, expiry, option_type, [strike])
            if contracts:
                chain_ltp = float(contracts[0].last_price or 0.0)
                if chain_ltp > 0:
                    self._option_ltp_cache[option_key] = (chain_ltp, time.monotonic())
                    return chain_ltp
            return cached_ltp
        if time.monotonic() < self._option_ltp_blocked_until:
            chain_ltp = self._option_ltp_from_chain_cache(option_key)
            return chain_ltp or cached_ltp
        try:
            import requests
            if is_mcx_symbol(option_key):
                segment = Seg.MCX_COMM
                sec_id = self._security_id_cache.get(option_key)
                query_items = [int(sec_id)] if (sec_id and str(sec_id).isdigit()) else [option_key]
                url = f"{DHAN_BASE_URL}/v2/marketfeed/ltp"
            elif option_key.startswith("SENSEX"):
                segment = SENSEX_DHAN_OPTION_SEG
                query_items = [option_key]
                url = f"{DHAN_BASE_URL}/v2/marketfeed/quote"
            else:
                segment = Seg.NSE_FNO
                query_items = [option_key]
                url = f"{DHAN_BASE_URL}/v2/marketfeed/quote"

            resp = requests.post(
                url,
                headers=self._headers(),
                json={segment: query_items},
                timeout=BROKER_DHAN_TIMEOUT_8,
            )
            if resp.status_code != 200:
                self._option_ltp_blocked_until = time.monotonic() + 15.0
                chain_ltp = self._option_ltp_from_chain_cache(option_key)
                if chain_ltp > 0:
                    return chain_ltp
                upstox = self._get_secondary_broker()
                if upstox:
                    try:
                        up_ltp = float(upstox.get_option_ltp(option_key) or 0.0)
                        if up_ltp > 0:
                            self._option_ltp_cache[option_key] = (up_ltp, time.monotonic())
                            logger.info(f"[DhanBroker] 🔄 Upstox failover get_option_ltp({option_key}): ₹{up_ltp}")
                            return up_ltp
                    except Exception:
                        pass
                rec = self._recover_from_parquet_cache(option_key)
                if rec > 0:
                    return rec
                logger.warning(
                    f"[DhanBroker] get_option_ltp({option_key}) "
                    f"HTTP {resp.status_code}: {self._response_snippet(resp, 120)}"
                )
                return cached_ltp
            data = resp.json()
            seg_data = data.get("data", {}).get(segment, {})
            c_data = seg_data.get(str(query_items[0]), {}) or seg_data.get(query_items[0], {})
            ltp = float(c_data.get("last_price", 0.0) or 0.0)
            if ltp > 0:
                self._option_ltp_cache[option_key] = (ltp, time.monotonic())
                return ltp
            chain_ltp = self._option_ltp_from_chain_cache(option_key)
            if chain_ltp > 0:
                return chain_ltp
            upstox = self._get_secondary_broker()
            if upstox:
                try:
                    up_ltp = float(upstox.get_option_ltp(option_key) or 0.0)
                    if up_ltp > 0:
                        self._option_ltp_cache[option_key] = (up_ltp, time.monotonic())
                        return up_ltp
                except Exception:
                    pass
            rec = self._recover_from_parquet_cache(option_key)
            return rec or cached_ltp
        except Exception as e:
            self._option_ltp_blocked_until = time.monotonic() + 15.0
            chain_ltp = self._option_ltp_from_chain_cache(option_key)
            if chain_ltp > 0:
                return chain_ltp
            upstox = self._get_secondary_broker()
            if upstox:
                try:
                    up_ltp = float(upstox.get_option_ltp(option_key) or 0.0)
                    if up_ltp > 0:
                        self._option_ltp_cache[option_key] = (up_ltp, time.monotonic())
                        return up_ltp
                except Exception:
                    pass
            rec = self._recover_from_parquet_cache(option_key)
            if rec > 0:
                return rec
            logger.warning(
                f"[DhanBroker] get_option_ltp({option_key}): "
                f"{type(e).__name__}: {e}"
            )
            return cached_ltp

    def get_india_vix(self) -> float:
        """Get India VIX using security_id='21' in NSE_INDEX with chart & yfinance fallbacks."""
        vix = float(self.get_ltp("INDIA VIX") or 0.0)
        if 8.0 <= vix <= 80.0:
            self._last_known_vix = vix
            return vix

        cached_vix = float(self._index_ltp_cache.get("INDIA VIX", (0.0, 0.0))[0] or 0.0)
        if 8.0 <= cached_vix <= 80.0:
            self._last_known_vix = cached_vix
            return cached_vix

        # Tier 1 fallback: Intraday 1-min chart close
        chart_vix = float(self._ltp_from_intraday_chart("INDIA VIX", "21") or 0.0)
        if 8.0 <= chart_vix <= 80.0:
            self._index_ltp_cache["INDIA VIX"] = (chart_vix, time.monotonic())
            self._last_known_vix = chart_vix
            return chart_vix

        # Tier 2 fallback: yfinance ^INDIAVIX
        yf_vix = float(self._fallback_yfinance_ltp("INDIA VIX") or 0.0)
        if 8.0 <= yf_vix <= 80.0:
            self._index_ltp_cache["INDIA VIX"] = (yf_vix, time.monotonic())
            self._last_known_vix = yf_vix
            return yf_vix

        # Tier 3 fallback: Last known valid VIX
        if hasattr(self, "_last_known_vix") and 8.0 <= float(self._last_known_vix or 0.0) <= 80.0:
            return float(self._last_known_vix)

        now = time.monotonic()
        if now - self._vix_fallback_last_log_at >= 300.0:
            self._vix_fallback_last_log_at = now
            logger.warning(f"[DhanBroker] India VIX unavailable from all feeds (ltp={vix:.2f}); using fallback 14.0")
        return 14.0

    # ── INSTRUMENT LOOKUP ─────────────────────────────────────────────────────

    def get_instrument_key(self, symbol: str, exchange: str = "NSE") -> str:
        """Return Dhan security ID for a symbol.

        For MCX commodities (SILVERMIC, GOLD, CRUDEOIL, etc.) and contract symbols,
        resolves the Dhan numeric security ID.
        For index symbols (NIFTY, BANKNIFTY, etc.) returns the static mapping.
        For option trading symbols (e.g. NIFTY26AUG1824300PE) looks up the
        numeric security_id from the cached option chain.
        """
        sym = str(symbol).upper().strip()
        # 0. Already a numeric security ID
        if sym.isdigit():
            return sym

        # 1. Static mapping lookup
        if sym in DHAN_SECURITY_IDS:
            return DHAN_SECURITY_IDS[sym]

        # 2. Fast pre-indexed security ID cache lookup (< 0.01 ms)
        if sym in self._security_id_cache:
            return self._security_id_cache[sym]

        # 3. Check local scrip master cache for exact trading symbol or custom symbol
        try:
            from pathlib import Path
            import pandas as pd
            scrip_cache = Path("data/cache/dhan_scrip_master.csv")
            if scrip_cache.exists():
                df = pd.read_csv(scrip_cache, low_memory=False)
                match = df[
                    (df["SEM_TRADING_SYMBOL"].astype(str).str.upper() == sym)
                    | (df["SEM_CUSTOM_SYMBOL"].astype(str).str.upper() == sym)
                ]
                if not match.empty:
                    sec_id_str = str(match.iloc[0]["SEM_SMST_SECURITY_ID"])
                    self._security_id_cache[sym] = sec_id_str
                    return sec_id_str
        except Exception:
            pass

        # 4. Commodity / MCX underlying futures resolution (prefix or exact symbol match)
        for comm in ("SILVERMIC", "SILVERM", "SILVER", "GOLDM", "GOLD", "CRUDEOILM", "CRUDEOIL", "NATGASMINI", "NATURALGAS", "NATGAS", "NATGASM"):
            if sym.startswith(comm) or sym == comm:
                sec = DHAN_SECURITY_IDS.get(comm)
                if sec:
                    self._security_id_cache[sym] = sec
                    return sec

        # 3. Option contract — parse and look up from cached option chain
        parsed = self._parse_trading_option_symbol(sym)
        if parsed:
            underlying, expiry_date, strike, option_type = parsed
            expiry_str = expiry_date.isoformat()
            cache_key = (underlying, expiry_str)
            _, chain_data = self._option_chain_cache.get(cache_key, (0.0, {}))
            if not chain_data:
                _, chain_data = self._option_chain_cache.get(f"{underlying}_{expiry_str}", (0.0, {}))

            if chain_data:
                side_key = "ce" if option_type == "CE" else "pe"
                for strike_key in [f"{float(strike):.6f}", str(strike), str(float(strike))]:
                    strike_data = chain_data.get(strike_key, {})
                    cdata = strike_data.get(side_key, {})
                    sec_id = cdata.get("security_id")
                    if sec_id:
                        sec_id_str = str(sec_id)
                        self._security_id_cache[sym] = sec_id_str
                        logger.debug(
                            f"[DhanBroker] Resolved option security_id | "
                            f"{sym} → {sec_id_str}"
                        )
                        return sec_id_str

            # 4. Fallback: fetch fresh option chain if not cached
            try:
                underlying_id = DHAN_SECURITY_IDS.get(underlying, "")
                if underlying_id:
                    import requests as _req
                    resp = _req.post(
                        f"{DHAN_BASE_URL}/v2/optionchain",
                        headers=self._headers(),
                        json={
                            "UnderlyingScrip": int(underlying_id),
                            "UnderlyingSeg": "IDX_I",
                            "Expiry": expiry_str,
                        },
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        fresh_chain = resp.json().get("data", {}).get("oc", {}) or {}
                        now_mono = time.monotonic()
                        self._option_chain_cache[cache_key] = (now_mono, fresh_chain)
                        self._option_chain_cache[f"{underlying}_{expiry_str}"] = (now_mono, fresh_chain)
                        self._pre_index_security_ids(underlying, expiry_date, fresh_chain)
                        if sym in self._security_id_cache:
                            return self._security_id_cache[sym]
            except Exception as exc:
                logger.debug(f"[DhanBroker] Option chain fetch for security_id failed: {exc}")

            # 4. Secondary Fallback: look up standard NSE exchange token from local parquet cache
            try:
                import pandas as pd
                from pathlib import Path
                cache_file = Path("data/cache/instruments_upstox_NSE.parquet")
                if cache_file.exists():
                    df = pd.read_parquet(cache_file)
                    work = df[
                        (df["name"].astype(str).str.upper() == underlying)
                        & (df["strike_price"].astype(float) == float(strike))
                        & (df["instrument_type"].astype(str).str.upper() == option_type)
                    ]
                    if not work.empty:
                        # Match expiry date if present
                        if "expiry" in work.columns:
                            raw_expiry = work["expiry"]
                            numeric_expiry = pd.to_numeric(raw_expiry, errors="coerce")
                            if numeric_expiry.notna().any() and float(numeric_expiry.dropna().median() or 0.0) > 10000000000:
                                exp_series = pd.to_datetime(numeric_expiry, errors="coerce", unit="ms").dt.date
                            else:
                                exp_series = pd.to_datetime(raw_expiry, errors="coerce").dt.date
                            exp_match = work[exp_series == expiry_date]
                            if not exp_match.empty:
                                work = exp_match
                        key = str(work.iloc[0].get("instrument_key", "") or "")
                        if "|" in key:
                            token = key.split("|")[-1]
                            if token.isdigit():
                                logger.info(f"[DhanBroker] Resolved option security_id via master cache | {sym} → {token}")
                                return token
            except Exception as exc:
                logger.debug(f"[DhanBroker] Master cache token lookup error: {exc}")

            logger.warning(f"[DhanBroker] Could not resolve security_id for option {sym}")

        return symbol

    def get_option_instrument_key(
        self,
        symbol:      str,
        expiry:      str,
        strike:      int,
        option_type: str,
    ) -> str:
        """
        Build Dhan trading symbol for an options contract.
        Dhan format is same as NSE standard: NIFTY26APR2522500CE
        Returns the trading symbol string (used as the key for option LTP).
        """
        from utils.option_utils import build_option_symbol
        expiry_date = date.fromisoformat(expiry)
        return build_option_symbol(symbol, expiry_date, strike, option_type)

    # ── ORDER MANAGEMENT ──────────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol:      str,
        quantity:    int,
        transaction: str,    # "BUY" or "SELL"
        product:     str = "MIS",
        exchange:    str = "NSE_FNO",
        security_id: str = "",
    ) -> OrderResult:
        """
        Place a market order via Dhan.
        Endpoint: POST /v2/orders

        product: "MIS" → INTRADAY | "NRML" / "D" → MARGIN (overnight)
        exchange: "NSE_FNO" for options, "NSE_EQ" for equity
        """
        # Dhan uses different product type strings
        product_map = {
            "MIS": "INTRADAY",
            "I":   "INTRADAY",
            "NRML":"MARGIN",
            "D":   "MARGIN",
        }
        dhan_product = product_map.get(product.upper(), "INTRADAY")

        # Dhan uses different exchange segment strings
        seg_map = {
            "MCX":      Seg.MCX_COMM,
            "MCX_COMM": Seg.MCX_COMM,
            "MCX_FUT":  Seg.MCX_COMM,
            "NSE_FNO":  Seg.NSE_FNO,
            "NFO":      Seg.NSE_FNO,
            "NSE":      Seg.NSE_EQ,
            "NSE_EQ":   Seg.NSE_EQ,
            # SENSEX / BSE F&O
            "BSE_FNO":  SENSEX_DHAN_OPTION_SEG,
            "BFO":      SENSEX_DHAN_OPTION_SEG,
        }
        dhan_seg = seg_map.get(
            exchange.upper(),
            Seg.MCX_COMM if "MCX" in exchange.upper() else Seg.NSE_FNO
        )

        # Resolve and validate numeric securityId
        sec_id = str(security_id or "") if str(security_id or "").isdigit() else (symbol if str(symbol).isdigit() else self.get_instrument_key(symbol, exchange=exchange))
        if not str(sec_id).isdigit():
            logger.error(
                f"[DhanBroker] Pre-flight validation FAILED | securityId is non-numeric "
                f"('{sec_id}') for symbol '{symbol}' — rejecting order to prevent API error."
            )
            return OrderResult(
                order_id="",
                symbol=symbol,
                quantity=quantity,
                order_type="MARKET",
                status="REJECTED",
                message=f"Invalid non-numeric securityId '{sec_id}' for {symbol}",
            )

        # Pre-flight lot size validation & alignment
        try:
            from utils.instrument_selector import get_instrument_config
            cfg = get_instrument_config(symbol)
            default_lot = cfg.lot_size if cfg else 1
            if default_lot > 1 and quantity % default_lot != 0:
                aligned_qty = max(1, round(quantity / default_lot)) * default_lot
                logger.warning(
                    f"[DhanBroker] Pre-flight aligned quantity from {quantity} to {aligned_qty} "
                    f"(multiple of lot_size={default_lot}) for {symbol}"
                )
                quantity = aligned_qty
        except Exception:
            pass

        try:
            import requests
            order_body = {
                "dhanClientId":    self._client_id,
                "transactionType": transaction.upper(),      # "BUY" or "SELL"
                "exchangeSegment": dhan_seg,
                "productType":     dhan_product,
                "orderType":       "MARKET",
                "validity":        "DAY",
                "tradingSymbol":   symbol,
                "securityId":      sec_id,
                "quantity":        quantity,
                "price":           0,
                "triggerPrice":    0,
                "afterMarketOrder": False,
                "amoTime":         "OPEN",
                "boProfitValue":   0,
                "boStopLossValue": 0,
                "correlationId":   f"SF_{int(time.time())}",
            }

            session = getattr(self, "_session", requests)
            resp = session.post(
                f"{DHAN_BASE_URL}/v2/orders",
                headers=self._headers(),
                json=order_body,
                timeout=BROKER_DHAN_TIMEOUT_10,
            )
            data = resp.json()

            # Dynamic in-flight token reload & retry on authentication failure
            if (
                resp.status_code == 401
                or "DH-901" in str(data)
                or "Invalid_Authentication" in str(data)
                or "expired" in str(data).lower()
            ):
                if self._reload_token_from_env(reason="place_market_order_auth_fail"):
                    order_body["dhanClientId"] = self._client_id
                    resp = session.post(
                        f"{DHAN_BASE_URL}/v2/orders",
                        headers=self._headers(),
                        json=order_body,
                        timeout=BROKER_DHAN_TIMEOUT_10,
                    )
                    data = resp.json()

            if resp.status_code in (200, 201) and data.get("orderId"):
                order_id = str(data["orderId"])
                logger.success(
                    f"[DhanBroker] Order placed | {transaction} {symbol} "
                    f"qty={quantity} | order_id={order_id}"
                )
                return OrderResult(
                    order_id   = order_id,
                    symbol     = symbol,
                    quantity   = quantity,
                    order_type = "MARKET",
                    status     = "PLACED",
                )
            else:
                err_msg = str(data.get("remarks", data.get("message", str(data))))
                safe_err = err_msg.replace("{", "{{").replace("}", "}}")
                logger.error(
                    f"[DhanBroker] place_order FAILED | "
                    f"{transaction} {symbol} qty={quantity} | "
                    f"HTTP={resp.status_code} | error={safe_err}"
                )
                return OrderResult(
                    order_id   = "",
                    symbol     = symbol,
                    quantity   = quantity,
                    order_type = "MARKET",
                    status     = "REJECTED",
                    message    = err_msg,
                )
        except Exception as e:
            logger.error(
                f"[DhanBroker] place_order exception | "
                f"{transaction} {symbol} | "
                f"{type(e).__name__}: {e}"
            )
            return OrderResult(
                order_id="", symbol=symbol, quantity=quantity,
                order_type="MARKET", status="ERROR", message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        Endpoint: DELETE /v2/orders/{order-id}
        """
        try:
            import requests
            resp = requests.delete(
                f"{DHAN_BASE_URL}/v2/orders/{order_id}",
                headers=self._headers(),
                timeout=BROKER_DHAN_TIMEOUT_10,
            )
            if resp.status_code == 200:
                logger.info(f"[DhanBroker] Order cancelled: {order_id}")
                return True
            else:
                data    = resp.json() if resp.content else {}
                err_msg = data.get("remarks", data.get("message", "no detail"))
                logger.error(
                    f"[DhanBroker] cancel_order({order_id}) FAILED | "
                    f"HTTP={resp.status_code} | error={err_msg}"
                )
                return False
        except Exception as e:
            logger.error(
                f"[DhanBroker] cancel_order({order_id}): {type(e).__name__}: {e}"
            )
            return False

    def get_positions(self) -> list[PositionInfo]:
        """
        Get all open positions.
        Endpoint: GET /v2/positions
        """
        try:
            import requests
            resp = requests.get(
                f"{DHAN_BASE_URL}/v2/positions",
                headers=self._headers(),
                timeout=BROKER_DHAN_TIMEOUT_10,
            )
            if resp.status_code != 200:
                if self._is_token_error(resp):
                    if self._reload_token_from_env(reason="get_positions"):
                        return self.get_positions()
                logger.error(
                    f"[DhanBroker] get_positions HTTP {resp.status_code}: {self._response_snippet(resp, 120)}"
                )
                return []
            data = resp.json()
            positions = []
            pos_list = data.get("data", []) if isinstance(data, dict) else data
            for p in pos_list:
                qty = int(p.get("netQty", 0))
                if qty == 0:
                    continue
                positions.append(PositionInfo(
                    symbol      = p.get("tradingSymbol", ""),
                    quantity    = qty,
                    avg_price   = float(p.get("costPrice", 0)),
                    ltp         = float(p.get("lastTradedPrice", 0)),
                    pnl         = float(p.get("unrealizedProfit", 0)),
                    product     = p.get("productType", ""),
                    security_id = str(p.get("securityId", "") or ""),
                ))
            return positions
        except Exception as e:
            logger.error(f"[DhanBroker] get_positions: {type(e).__name__}: {e}")
            return []

    # ── PRIVATE HELPERS ───────────────────────────────────────────────────────

    def _headers(self) -> dict:
        """Common request headers for all Dhan API calls."""
        return {
            "access-token": self._access_token,
            "client-id":    self._client_id,
            "Content-Type": "application/json",
            "Accept":       "application/json",
        }

    def _parse_historical_dict(self, data: dict) -> pd.DataFrame:
        """
        Parse Dhan historical response when data is parallel arrays:
        {"open": [...], "high": [...], "low": [...], "close": [...],
         "volume": [...], "timestamp": [...]}
        """
        try:
            timestamps = data.get("timestamp", [])
            opens      = data.get("open",      [])
            highs      = data.get("high",      [])
            lows       = data.get("low",       [])
            closes     = data.get("close",     [])
            volumes    = data.get("volume",    [])

            if not timestamps:
                return pd.DataFrame()

            records = []
            for i, ts in enumerate(timestamps):
                try:
                    # Dhan timestamps can be epoch seconds or epoch milliseconds
                    ts_int = int(ts)
                    ts_sec = ts_int / 1000.0 if ts_int > 10000000000 else float(ts_int)
                    dt = datetime.fromtimestamp(ts_sec, tz=IST)
                    records.append({
                        "datetime": dt,
                        "open":     float(opens[i])   if i < len(opens)   else 0.0,
                        "high":     float(highs[i])   if i < len(highs)   else 0.0,
                        "low":      float(lows[i])    if i < len(lows)    else 0.0,
                        "close":    float(closes[i])  if i < len(closes)  else 0.0,
                        "volume":   int(volumes[i])   if i < len(volumes) else 0,
                    })
                except (IndexError, ValueError):
                    continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records).set_index("datetime").sort_index()
            has_intraday_times = any(ts.hour != 0 or ts.minute != 0 for ts in df.index)
            if has_intraday_times:
                # MCX commodity market hours run 09:00 to 23:30/23:55
                return df.between_time("09:00", "23:35")
            return df

        except Exception as e:
            logger.error(f"[DhanBroker] _parse_historical_dict: {e}")
            return pd.DataFrame()

    def _parse_historical_candles(self, candles: list) -> pd.DataFrame:
        """
        Parse Dhan historical response when data is list of candle arrays:
        [[timestamp_epoch_ms, open, high, low, close, volume], ...]
        """
        try:
            records = []
            for c in candles:
                if len(c) < 6:
                    continue
                try:
                    # c[0] may be epoch ms or epoch s — detect by magnitude
                    ts_raw = int(c[0])
                    if ts_raw > 1e11:
                        dt = datetime.fromtimestamp(ts_raw / 1000.0, tz=IST)
                    else:
                        dt = datetime.fromtimestamp(ts_raw, tz=IST)
                    records.append({
                        "datetime": dt,
                        "open":     float(c[1]),
                        "high":     float(c[2]),
                        "low":      float(c[3]),
                        "close":    float(c[4]),
                        "volume":   int(c[5]),
                    })
                except (ValueError, IndexError):
                    continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records).set_index("datetime").sort_index()
            has_intraday_times = any(ts.hour != 0 or ts.minute != 0 for ts in df.index)
            if has_intraday_times:
                return df.between_time("09:00", "23:35")
            return df

        except Exception as e:
            logger.error(f"[DhanBroker] _parse_historical_candles: {e}")
            return pd.DataFrame()

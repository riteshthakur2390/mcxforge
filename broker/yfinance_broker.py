"""
broker/yfinance_broker.py — Yahoo Finance Broker (No Account Needed)
=====================================================================
Use this while waiting for Upstox/Kite/Groww account approval.
Zero setup — just pip install yfinance and set BROKER=yfinance

What works:
  ✅ NIFTY historical OHLCV (5-min: last 60 days, daily: 15+ years)
  ✅ All 5 strategies
  ✅ Regime detection (ADX, Choppiness Index)
  ✅ Signal journal + EOD reports
  ✅ Dashboard live
  ✅ Replay script
  ✅ ML training on daily candles

What does NOT work (needs real broker):
  ❌ Live streaming / real-time LTP
  ❌ Option chain data
  ❌ Order placement
  ❌ India VIX live (uses static fallback)

Yahoo Finance symbols:
  NIFTY 50   → ^NSEI
  BANKNIFTY  → ^NSEBANK
  India VIX  → ^INDIAVIX (not always available)

Intervals supported by yfinance:
  1m  → last 7 days only
  5m  → last 60 days
  15m → last 60 days
  1d  → 15+ years
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
from data.historical_store import HistoricalCandleStore

IST = pytz.timezone("Asia/Kolkata")

# Yahoo Finance symbol map for Indian indices
YFINANCE_SYMBOL_MAP = {
    "NIFTY":     "^NSEI",
    "NIFTY 50":  "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX",
    "SENSEX":    "^BSESN",
}

YFINANCE_UNSUPPORTED_LTP_SYMBOLS = {"GIFTNIFTY"}

# yfinance interval codes
INTERVAL_MAP = {
    "1minute":  "1m",
    "5minute":  "5m",
    "15minute": "15m",
    "30minute": "30m",
    "60minute": "60m",
    "day":      "1d",
}

# Max days yfinance allows per interval
MAX_DAYS = {
    "1m":  7,
    "5m":  60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "1d":  3650,
}


class YFinanceBroker(BaseBroker):
    """
    Yahoo Finance data source — zero account required.
    For development and testing only.
    """

    def __init__(self) -> None:
        try:
            import yfinance as yf
            self._yf = yf
            logger.info("[YFinanceBroker] Initialized. Ready for testing.")
        except ImportError:
            raise ImportError(
                "yfinance not installed.\n"
                "Run: pip install yfinance"
            )
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        # Cache last known LTP per symbol so we can still show something
        # if a later poll fails. Keys are the same as the `symbol` argument
        # passed into get_ltp() (e.g. "NIFTY", "BANKNIFTY").
        self._last_ltp_by_symbol: dict[str, float] = {}

    @property
    def broker_name(self) -> str:
        return "yfinance"

    # ── AUTH (no-op for yfinance) ─────────────────────────────────────────────

    def get_login_url(self) -> str:
        return "https://finance.yahoo.com"   # not needed

    def generate_session(self, auth_code: str = "") -> str:
        logger.info("[YFinanceBroker] No auth needed for yfinance.")
        return "yfinance-no-token"

    def set_access_token(self, token: str) -> None:
        pass   # no-op

    # ── MARKET DATA ───────────────────────────────────────────────────────────

    def get_ltp(self, symbol: str) -> float:
        """
        Get latest price from yfinance.
        NOTE: Not real-time — delayed ~15 minutes.
        For live testing, this gives you a close enough value.
        """
        if symbol in YFINANCE_UNSUPPORTED_LTP_SYMBOLS:
            return float(self._last_ltp_by_symbol.get(symbol, 0.0))
        try:
            yf_symbol = YFINANCE_SYMBOL_MAP.get(symbol, symbol)
            ticker    = self._yf.Ticker(yf_symbol)
            hist      = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                ltp = float(hist["Close"].iloc[-1])
                self._last_ltp_by_symbol[symbol] = ltp
                return ltp
        except Exception as e:
            logger.warning(f"[YFinanceBroker] get_ltp({symbol}): {e}")
        return float(self._last_ltp_by_symbol.get(symbol, 0.0))

    def get_historical_data(
        self,
        symbol:    str,
        interval:  str,
        from_date: str,
        to_date:   str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Yahoo Finance.
        Automatically handles yfinance date limits per interval.
        """
        try:
            yf_symbol    = YFINANCE_SYMBOL_MAP.get(symbol, symbol)
            yf_interval  = INTERVAL_MAP.get(interval, "5m")
            max_days     = MAX_DAYS.get(yf_interval, 60)

            # Clip from_date to yfinance limit
            from_dt = pd.Timestamp(from_date)
            limit   = pd.Timestamp.now() - pd.Timedelta(days=max_days - 1)
            if from_dt < limit:
                logger.warning(
                    f"[YFinanceBroker] {yf_interval} limited to {max_days} days. "
                    f"Adjusting from_date to {limit.date()}"
                )
                from_dt = limit

            ticker = self._yf.Ticker(yf_symbol)
            # yfinance `end` is exclusive and can miss same-day intraday candles.
            # Use period-based fetch for minute intervals, then clip locally.
            if "m" in yf_interval:
                df = ticker.history(
                    period   = f"{max_days}d",
                    interval = yf_interval,
                )
            else:
                df = ticker.history(
                    start    = from_dt.strftime("%Y-%m-%d"),
                    end      = to_date,
                    interval = yf_interval,
                )

            if df.empty:
                logger.warning(f"[YFinanceBroker] No data for {symbol} {interval}")
                return pd.DataFrame()

            # Standardise columns
            df = df.rename(columns={
                "Open":   "open",
                "High":   "high",
                "Low":    "low",
                "Close":  "close",
                "Volume": "volume",
            })
            df = df[["open", "high", "low", "close", "volume"]]

            # Convert to IST
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(IST)

            # Filter to market hours for intraday
            if "m" in yf_interval:
                # Keep requested date window for period-based fetch.
                to_dt = pd.Timestamp(to_date).tz_localize(IST) + pd.Timedelta(days=1)
                from_dt_ist = pd.Timestamp(from_dt).tz_localize(IST)
                df = df[(df.index >= from_dt_ist) & (df.index < to_dt)]
                df = df.between_time("09:15", "15:30")

            df = df.sort_index().dropna()
            logger.info(
                f"[YFinanceBroker] {symbol} {interval}: "
                f"{len(df)} candles ({df.index[0].date()} → {df.index[-1].date()})"
            )
            return df

        except Exception as e:
            logger.error(f"[YFinanceBroker] get_historical_data: {e}")
            return pd.DataFrame()

    def get_option_ltp(self, option_symbol: str) -> float:
        """
        yfinance doesn't have Indian F&O option data.
        Returns 0 — Agent 6 will use delta-gamma estimation instead.
        """
        logger.debug(
            f"[YFinanceBroker] Option LTP not available via yfinance. "
            f"Using delta-gamma estimation for {option_symbol}."
        )
        return 0.0

    def get_india_vix(self) -> float:
        """Attempt to get India VIX — fallback to 14.0 if unavailable."""
        try:
            ticker = self._yf.Ticker("^INDIAVIX")
            hist   = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            pass
        logger.debug("[YFinanceBroker] India VIX unavailable — using default 14.0")
        return 14.0   # safe fallback — typical NIFTY VIX on normal day

    # ── INSTRUMENT LOOKUP ─────────────────────────────────────────────────────

    def get_instrument_key(self, symbol: str, exchange: str = "NSE") -> str:
        return YFINANCE_SYMBOL_MAP.get(symbol, f"^{symbol}")

    def get_option_instrument_key(
        self, symbol: str, expiry: str,
        strike: int, option_type: str,
    ) -> str:
        # Not supported — return tradingsymbol format for reference
        from utils.option_utils import build_option_symbol
        expiry_date = date.fromisoformat(expiry)
        return build_option_symbol(symbol, expiry_date, strike, option_type)

    # ── ORDER MANAGEMENT (all return DRY_RUN — yfinance is data only) ─────────

    def place_market_order(
        self, symbol: str, quantity: int,
        transaction: str, product: str = "MIS",
        exchange: str = "NFO",
    ) -> OrderResult:
        logger.warning(
            "[YFinanceBroker] Order placement not supported. "
            "Switch to BROKER=kite/upstox/groww for live trading."
        )
        return OrderResult(
            order_id   = f"YFIN_DRYRUN_{int(time.time())}",
            symbol     = symbol,
            quantity   = quantity,
            order_type = "MARKET",
            status     = "ERROR",
            message    = "yfinance does not support order placement",
        )

    def cancel_order(self, order_id: str) -> bool:
        return False

    def get_positions(self) -> list[PositionInfo]:
        return []

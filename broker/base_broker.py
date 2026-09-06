from config.settings.modules.system_thresholds import *
"""
broker/base_broker.py — Abstract Broker Interface
===================================================
Both UpstoxBroker and KiteBroker implement this interface.
All agents use BaseBroker — they never import a specific broker directly.

Rule: Agent code never says "kite." or "upstox." anywhere.
      It only calls methods defined here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
import pandas as pd

from utils.market_calendar import is_trading_day


@dataclass
class OrderResult:
    """Returned after placing an order."""
    order_id:    str
    symbol:      str
    quantity:    int
    order_type:  str      # MARKET / LIMIT
    status:      str      # PLACED / REJECTED / ERROR
    message:     str = ""


@dataclass
class PositionInfo:
    """Current open position details."""
    symbol:        str
    quantity:      int
    avg_price:     float
    ltp:           float
    pnl:           float
    product:       str    # MIS / NRML
    security_id:   str = ""


@dataclass
class OptionContract:
    """Snapshot for one option contract candidate."""
    symbol: str
    strike: int
    option_type: str
    expiry_date: str
    last_price: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    volume: int = 0
    open_interest: int = 0
    oi_change: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    theta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "strike": self.strike,
            "option_type": self.option_type,
            "expiry_date": self.expiry_date,
            "last_price": self.last_price,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "volume": self.volume,
            "open_interest": self.open_interest,
            "oi_change": self.oi_change,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "theta": self.theta,
            "gamma": self.gamma,
            "vega": self.vega,
            "source": self.source,
        }


class BaseBroker(ABC):
    """
    Abstract base class for all broker integrations.
    Implement all methods below for each broker.
    """

    # ── AUTHENTICATION ────────────────────────────────────────────────────────

    @abstractmethod
    def get_login_url(self) -> str:
        """Return the OAuth login URL for the broker."""
        ...

    @abstractmethod
    def generate_session(self, auth_code: str) -> str:
        """
        Exchange auth code for access token.
        Returns the access token string.
        Called once daily after login.
        """
        ...

    @abstractmethod
    def set_access_token(self, token: str) -> None:
        """Set the access token on the broker client."""
        ...

    # ── MARKET DATA ───────────────────────────────────────────────────────────

    @abstractmethod
    def get_ltp(self, symbol: str) -> float:
        """
        Get last traded price for a symbol.
        Symbol format is broker-specific — use get_instrument_key() to convert.
        """
        ...

    @abstractmethod
    def get_historical_data(
        self,
        symbol:    str,
        interval:  str,
        from_date: str,
        to_date:   str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV historical candles.
        Returns DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex (IST timezone)
        """
        ...

    @abstractmethod
    def get_option_ltp(self, option_symbol: str) -> float:
        """Get current LTP of an options contract."""
        ...

    @abstractmethod
    def get_india_vix(self) -> float:
        """Get current India VIX value."""
        ...

    # ── INSTRUMENT LOOKUP ─────────────────────────────────────────────────────

    @abstractmethod
    def get_instrument_key(self, symbol: str, exchange: str = "NSE") -> str:
        """
        Get broker-specific instrument key/token for a symbol.
        Kite   → integer token e.g. 256265
        Upstox → string key  e.g. "NSE_INDEX|Nifty 50"
        """
        ...

    @abstractmethod
    def get_option_instrument_key(
        self,
        symbol:      str,
        expiry:      str,
        strike:      int,
        option_type: str,
    ) -> str:
        """
        Get broker-specific key for an options contract.
        option_type: "CE" or "PE"
        """
        ...

    # ── ORDER MANAGEMENT ──────────────────────────────────────────────────────

    @abstractmethod
    def place_market_order(
        self,
        symbol:      str,
        quantity:    int,
        transaction: str,    # "BUY" or "SELL"
        product:     str,    # "MIS" (intraday) or "NRML"
        exchange:    str,    # "NFO"
    ) -> OrderResult:
        """Place a market order. Returns OrderResult."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if successful."""
        ...

    @abstractmethod
    def get_positions(self) -> list[PositionInfo]:
        """Get all current open positions."""
        ...

    def get_option_contracts(
        self,
        symbol: str,
        expiry: date,
        option_type: str,
        strikes: list[int],
    ) -> list[OptionContract]:
        """
        Best-effort option contract snapshots for contract selection.
        Brokers that do not expose chain data may return [].
        """
        return []

    @property
    def supports_bracket_orders(self) -> bool:
        return False

    def place_bracket_order(
        self,
        symbol: str,
        quantity: int,
        transaction: str,
        stop_loss_price: float,
        target_price: float,
        product: str,
        exchange: str,
    ) -> OrderResult:
        return OrderResult(
            order_id="",
            symbol=symbol,
            quantity=quantity,
            order_type="BRACKET",
            status="ERROR",
            message="Bracket orders are not supported by this broker adapter.",
        )

    # ── UTILITY ───────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Return broker name string: 'kite' or 'upstox'"""
        ...

    def download_and_cache(
        self,
        symbol:   str = "NIFTY",
        interval: str = "5minute",
    ) -> pd.DataFrame:
        """
        Download recent history for backtesting and persist it under the
        active broker cache name.
        """
        from datetime import datetime, timedelta
        import pytz
        from pathlib import Path
        from loguru import logger
        from config.settings import DATA_CACHE_DIR
        from data.historical_store import HistoricalCandleStore

        IST = pytz.timezone("Asia/Kolkata")
        end_date = datetime.now(IST)
        
        # Determine lookback based on interval and broker capability
        if interval in {"1minute", "5minute", "15minute"}:
            lookback_days = BROKER_BASE_LOOKBACK_31
        elif interval == "30minute":
            lookback_days = BROKER_BASE_LOOKBACK_90
        else:
            lookback_days = BROKER_BASE_LOOKBACK_365
            
        start_date = end_date - timedelta(days=lookback_days)
        df = self.get_historical_data(
            symbol=symbol,
            interval=interval,
            from_date=start_date.strftime("%Y-%m-%d"),
            to_date=end_date.strftime("%Y-%m-%d"),
        )
        if df.empty:
            if interval == "day":
                logger.info(
                    f"[{self.broker_name.upper()}] No daily data downloaded for {symbol}; "
                    "keeping existing day cache if available"
                )
            else:
                logger.warning(f"[{self.broker_name.upper()}] No data downloaded for {symbol} {interval}")
            return df
            
        symbol_key = str(symbol or "NIFTY").upper().replace(" ", "")
        if symbol_key in {"NIFTY50", "NIFTY_50"}:
            symbol_key = "NIFTY"
        cache_path = Path(DATA_CACHE_DIR) / f"{symbol_key}_{interval}_{self.broker_name}.parquet"
        df.to_parquet(cache_path)
        try:
            HistoricalCandleStore().upsert_candles(
                df,
                symbol=symbol,
                interval=interval,
                broker=self.broker_name,
                source="broker_cache_refresh",
            )
        except Exception as exc:
            logger.warning(f"[{self.broker_name.upper()}] Local history persist failed: {exc}")
        logger.success(f"[{self.broker_name.upper()}] Cached {len(df)} candles -> {cache_path}")
        return df

    def is_market_open(self, current_dt: Optional[datetime] = None, exchange: str = "MCX") -> bool:
        """
        Check if market is currently open. Supports MCX sessions (09:00 - 23:30)
        and optional current_dt parameter for testing.
        """
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
        now = current_dt or datetime.now(IST)
        if not is_trading_day(now.date(), exchange=exchange):
            return False
        t = now.strftime("%H:%M")
        if exchange.upper() == "MCX":
            return "09:00" <= t <= "23:30"
        return "09:15" <= t <= "15:30"

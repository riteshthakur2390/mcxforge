"""
utils/option_utils.py — Option maths utilities
================================================
Premium estimation and expiry calculation.
Used by Agent 4 (Trade Planner) and Agent 6 (Position Manager).
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd

from utils.market_calendar import is_trading_day
from config.settings import (
    NIFTY_LOT_SIZE,
    NIFTY_STRIKE_STEP,
    SENSEX_LOT_SIZE,
    SENSEX_STRIKE_STEP,
    MIN_DAYS_TO_EXPIRY,
)

INDEX_SPECS = {
    "NIFTY": {"weekly_expiry_weekday": 1, "lot_size": NIFTY_LOT_SIZE, "strike_step": NIFTY_STRIKE_STEP},
    "SENSEX": {"weekly_expiry_weekday": 4, "lot_size": SENSEX_LOT_SIZE, "strike_step": SENSEX_STRIKE_STEP},
}
NIFTY_WEEKLY_EXPIRY_WEEKDAY = INDEX_SPECS["NIFTY"]["weekly_expiry_weekday"]
OPTION_SYMBOL_RE = re.compile(r"^[A-Z]+(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$")
PAT_DHAN = re.compile(r"^([A-Z0-9]+)-(\d{1,2}[A-Za-z]{3}\d{4})-(\d+)-(CE|PE)$")

MONTH_3_TO_NUM = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}
NSE_WEEKLY_MONTH_CHAR_TO_NUM = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
    "7": 7, "8": 8, "9": 9, "O": 10, "N": 11, "D": 12
}
NUM_TO_3 = {v: k for k, v in MONTH_3_TO_NUM.items()}

# Pattern A: Standard 3-letter month with day: e.g. NIFTY26SEP0823900PE
PAT_A = re.compile(r"^([A-Z]+)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})(\d+)(CE|PE)$")
# Pattern B: NSE weekly 1-char month with day: e.g. NIFTY2690823900PE, NIFTY26O1524000CE
PAT_B = re.compile(r"^([A-Z]+)(\d{2})([1-9OND])(\d{2})(\d+)(CE|PE)$")
# Pattern C: Monthly contract without day: e.g. NIFTY26SEP23900PE
PAT_C = re.compile(r"^([A-Z]+)(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d+)(CE|PE)$")


def canonical_option_sym(sym: str) -> str:
    """
    Normalizes option symbol representations across brokers (Dhan, Upstox, NSE, Zerodha).
    Transforms NSE weekly format (e.g. NIFTY2690823900PE) into canonical 3-letter format (NIFTY26SEP0823900PE).
    Strips broker/exchange prefixes and whitespace.
    """
    if not sym:
        return ""
    clean = str(sym).upper().strip()
    for prefix in ["NSE_FO|", "BFO|", "NSE_INDEX|", "BSE_INDEX|", "NSE:", "BSE:"]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    clean = re.sub(r"[\s\-_]+", "", clean)

    m = PAT_A.match(clean)
    if m:
        underlying, yy, mon_str, dd, strike, opt_type = m.groups()
        return f"{underlying}{yy}{mon_str}{dd}{strike}{opt_type}"

    m = PAT_B.match(clean)
    if m:
        underlying, yy, m_char, dd, strike, opt_type = m.groups()
        mon_num = NSE_WEEKLY_MONTH_CHAR_TO_NUM[m_char]
        mon_str = NUM_TO_3[mon_num]
        return f"{underlying}{yy}{mon_str}{dd}{strike}{opt_type}"

    m = PAT_C.match(clean)
    if m:
        underlying, yy, mon_str, strike, opt_type = m.groups()
        return f"{underlying}{yy}{mon_str}{strike}{opt_type}"

    return clean



@dataclass(frozen=True)
class OptionValidationResult:
    valid: bool
    reason: str = ""
    symbol: str = ""
    expiry_date: Optional[date] = None
    strike: int = 0
    option_type: str = ""
    moneyness: str = ""


def estimate_atm_premium(
    underlying: float,
    days_to_expiry: int,
    iv: float = 0.14
) -> float:
    """
    ATM option premium approximation using simplified Black-Scholes.
    Formula: premium ≈ 0.4 × IV × sqrt(T) × S
    where T = days_to_expiry / 365

    Args:
        underlying:       NIFTY spot price
        days_to_expiry:   calendar days until expiry
        iv:               implied volatility (default 14% = normal NIFTY)

    Returns:
        Estimated premium rounded to nearest ₹5
    """
    T       = max(days_to_expiry, 1) / 365
    premium = 0.4 * iv * np.sqrt(T) * underlying
    return float(round(round(premium / 5) * 5, 1))


def estimate_option_pnl(
    entry_premium: float,
    underlying_change_pct: float,
    direction: str,
    delta: float = 0.5,
    gamma: float = 0.001,
    underlying_spot: float = 24000.0,
) -> float:
    """
    Estimate current option premium from underlying price change.
    Uses delta-gamma approximation for position tracking.

    Returns: estimated current premium (not % change).
    """
    if "CALL" in str(direction).upper():
        pnl_pct = (
            delta * underlying_change_pct
            + 0.5 * gamma * (underlying_change_pct ** 2)
        )
    else:
        pnl_pct = (
            delta * (-underlying_change_pct)
            + 0.5 * gamma * (underlying_change_pct ** 2)
        )
    return round(entry_premium * (1 + pnl_pct), 1)


def compute_atr(df: pd.DataFrame, period: int = 14) -> tuple[float, float]:
    """
    Compute ATR and ATR percentage using OHLC candles.
    Returns (atr_points, atr_pct_of_close).
    """
    if df is None or len(df) < max(period + 1, 5):
        return 0.0, 0.0
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.rolling(period).mean().iloc[-1] or 0.0)
    last_close = float(close.iloc[-1] or 0.0)
    atr_pct = atr / last_close if last_close > 0 else 0.0
    return round(atr, 4), round(atr_pct, 6)


def apply_option_slippage(premium: float, side: str, slippage_pct: float) -> float:
    if premium <= 0 or slippage_pct <= 0:
        return round(premium, 1)
    side = str(side or "").upper()
    if side == "BUY":
        premium *= 1 + slippage_pct / 100
    else:
        premium *= max(0.0, 1 - slippage_pct / 100)
    return round(premium, 1)


def estimate_round_trip_costs(
    entry_premium: float,
    exit_premium: float,
    quantity: int,
    brokerage_per_order: float,
    transaction_cost_pct: float,
) -> float:
    trade_value = max(entry_premium, 0.0) * quantity
    variable_cost = trade_value * max(transaction_cost_pct, 0.0) / 100
    flat_cost = max(brokerage_per_order, 0.0) * 2
    return round(variable_cost + flat_cost, 2)


def get_nearest_expiry(
    min_days: int = MIN_DAYS_TO_EXPIRY,
    reference_date: Union[date, datetime, None] = None,
    symbol: str = "NIFTY",
) -> tuple[date, int]:
    """
    Returns nearest valid weekly expiry with at least min_days remaining.
    For MCX commodities (SILVERMIC, GOLD, CRUDEOIL, etc.), delegates to instrument selector.
    """
    sym_upper = str(symbol or "").upper()
    if sym_upper not in INDEX_SPECS:
        try:
            from utils.instrument_selector import get_nearest_expiry as mcx_get_nearest_expiry
            ref = reference_date.date() if isinstance(reference_date, datetime) else (reference_date or date.today())
            return mcx_get_nearest_expiry(min_days=min_days, ref_date=ref, symbol=sym_upper)
        except Exception:
            pass

    if isinstance(reference_date, datetime):
        today = reference_date.date()
    elif isinstance(reference_date, date):
        today = reference_date
    else:
        today = date.today()
    expiry = get_weekly_expiry(today, symbol=symbol)
    days_to_exp = (expiry - today).days

    if days_to_exp < min_days:
        next_week_ref = today + timedelta(days=7)
        expiry = get_weekly_expiry(next_week_ref, symbol=symbol)
        days_to_exp = (expiry - today).days

    return expiry, days_to_exp


def get_weekly_expiry(reference_day: date, symbol: str = "NIFTY") -> date:
    sym_upper = str(symbol or "").upper()
    if sym_upper not in INDEX_SPECS:
        try:
            from utils.instrument_selector import get_weekly_expiry as mcx_get_weekly_expiry
            return mcx_get_weekly_expiry(trade_date=reference_day, symbol=sym_upper)
        except Exception:
            pass

    weekday = expiry_weekday(symbol)
    days_to_expiry = (weekday - reference_day.weekday()) % 7
    expiry = reference_day + timedelta(days=days_to_expiry)
    exchange = "BSE" if symbol.upper() == "SENSEX" else "NSE"
    if days_to_expiry == 0 and not is_trading_day(expiry, exchange=exchange):
        expiry += timedelta(days=7)
    return previous_trading_day(expiry, symbol=symbol)


def expiry_weekday(symbol: str) -> int:
    spec = INDEX_SPECS.get(str(symbol or "NIFTY").upper())
    if spec is None:
        return 4  # Default Friday if not specified
    return int(spec["weekly_expiry_weekday"])


def is_expiry_day(day: Union[date, datetime], symbol: str = "NIFTY") -> bool:
    sym_upper = str(symbol or "").upper()
    if sym_upper not in INDEX_SPECS:
        try:
            from utils.instrument_selector import is_expiry_day as mcx_is_expiry_day
            ref = day.date() if isinstance(day, datetime) else day
            return mcx_is_expiry_day(day=ref, symbol=sym_upper)
        except Exception:
            pass
    ref = day.date() if isinstance(day, datetime) else day
    return ref == get_weekly_expiry(ref, symbol=symbol)


def get_index_lot_size(symbol: str) -> int:
    spec = INDEX_SPECS.get(str(symbol or "NIFTY").upper())
    if spec is None:
        try:
            from utils.instrument_selector import get_instrument
            return get_instrument(str(symbol or "").upper()).lot_size
        except Exception:
            return 1
    return int(spec["lot_size"])


def get_index_strike_step(symbol: str) -> int:
    spec = INDEX_SPECS.get(str(symbol or "NIFTY").upper())
    if spec is None:
        try:
            from utils.instrument_selector import get_instrument
            return get_instrument(str(symbol or "").upper()).strike_step
        except Exception:
            return 500
    return int(spec["strike_step"])


def previous_trading_day(day: date, symbol: str = "NIFTY") -> date:
    exchange = "BSE" if symbol.upper() == "SENSEX" else "NSE"
    current = day
    while not is_trading_day(current, exchange=exchange):
        current -= timedelta(days=1)
    return current


def build_option_symbol(
    symbol:      str,
    expiry_date: date,
    strike:      int,
    option_type: str,         # "CE" or "PE"
) -> str:
    """
    Build canonical option symbol.
    For Dhan MCX: Format {SYMBOL}-{DDMonYYYY}-{STRIKE}-{CE/PE}
    Example: CRUDEOIL-17Sep2026-7000-CE, SILVERM-24Sep2026-285000-PE
    For NSE: NIFTY25MAR2722500CE
    """
    sym_upper = str(symbol).upper().strip()
    is_mcx = (
        sym_upper in ("CRUDEOIL", "CRUDEOILM", "NATURALGAS", "NATGASMINI", "SILVER", "SILVERM", "SILVERMIC", "GOLD", "GOLDM")
        or any(k in sym_upper for k in ("SILVER", "GOLD", "CRUDE", "NAT"))
    )
    if is_mcx:
        root = "SILVERM" if sym_upper == "SILVERMIC" else sym_upper
        exp_str = expiry_date.strftime("%d%b%Y")
        return f"{root}-{exp_str}-{int(strike)}-{option_type.upper()}"

    expiry_str = expiry_date.strftime("%y%b%d").upper()
    return f"{symbol}{expiry_str}{int(strike)}{option_type}"


def get_atm_strike(ltp: float, strike_step: int = NIFTY_STRIKE_STEP) -> int:
    """Round LTP to nearest strike step to get ATM strike."""
    step = max(int(strike_step or 50), 1)
    return int(round(ltp / step) * step)


def classify_strike_moneyness(
    *,
    underlying: float,
    strike: int,
    option_type: str,
    strike_step: int = NIFTY_STRIKE_STEP,
) -> str:
    atm = get_atm_strike(underlying, strike_step)
    if strike == atm:
        return "ATM"
    if option_type == "CE":
        return "ITM" if strike < atm else "OTM"
    return "ITM" if strike > atm else "OTM"


def get_nearest_expiry_for_instrument(
    instrument_name: str = "NIFTY",
    min_days: int = MIN_DAYS_TO_EXPIRY,
) -> tuple["date", int]:
    today = date.today()
    target_weekday = expiry_weekday(instrument_name)
    days_ahead = (target_weekday - today.weekday()) % 7
    expiry = today + timedelta(days=days_ahead)
    dte    = (expiry - today).days
    if dte < min_days:
        expiry = expiry + timedelta(weeks=1)
        dte    = (expiry - today).days
    return expiry, dte


def build_option_symbol_for_instrument(
    instrument_name: str,
    expiry_date:     "date",
    strike:          int,
    option_type:     str,
) -> str:
    return build_option_symbol(instrument_name, expiry_date, strike, option_type)


def validate_option_contract(
    *,
    symbol: str,
    expiry_date: date,
    strike: int,
    option_type: str,
    underlying: float,
    strike_step: int = 50,
    expected_symbol: Optional[str] = None,
) -> OptionValidationResult:
    sym_upper = str(symbol or "").upper()
    is_mcx = (
        sym_upper not in INDEX_SPECS
        or any(k in sym_upper for k in ("SILVER", "GOLD", "CRUDE", "NAT"))
        or (expected_symbol and ("-" in expected_symbol or "OPTFUT" in expected_symbol))
    )

    opt_type_upper = str(option_type).upper()
    if opt_type_upper not in {"CE", "PE"}:
        return OptionValidationResult(False, "invalid_option_type", expected_symbol or "", expiry_date, strike, option_type)
    if strike <= 0:
        return OptionValidationResult(False, "non_tradable_strike", expected_symbol or "", expiry_date, strike, option_type)

    if is_mcx:
        # MCX commodity option validation
        moneyness = classify_strike_moneyness(
            underlying=underlying,
            strike=strike,
            option_type=opt_type_upper,
            strike_step=max(strike_step, 1),
        )
        final_symbol = expected_symbol or build_option_symbol(symbol, expiry_date, strike, opt_type_upper)
        return OptionValidationResult(
            True,
            "",
            final_symbol,
            expiry_date,
            strike,
            opt_type_upper,
            moneyness,
        )

    expected = expected_symbol or build_option_symbol(symbol, expiry_date, strike, opt_type_upper)
    if build_option_symbol(symbol, expiry_date, strike, opt_type_upper) != expected:
        return OptionValidationResult(False, "symbol_mismatch", expected, expiry_date, strike, opt_type_upper)
    if not (OPTION_SYMBOL_RE.match(expected) or PAT_DHAN.match(expected)):
        return OptionValidationResult(False, "invalid_symbol_format", expected, expiry_date, strike, opt_type_upper)
    strike_step = int(strike_step or get_index_strike_step(symbol))
    if strike <= 0 or strike % strike_step != 0:
        return OptionValidationResult(False, "non_tradable_strike", expected, expiry_date, strike, opt_type_upper)
    if expiry_date != previous_trading_day(expiry_date):
        return OptionValidationResult(False, "expiry_not_trading_day", expected, expiry_date, strike, opt_type_upper)
    if expiry_date != get_weekly_expiry(expiry_date, symbol=symbol):
        return OptionValidationResult(False, "invalid_weekly_expiry", expected, expiry_date, strike, opt_type_upper)

    moneyness = classify_strike_moneyness(
        underlying=underlying,
        strike=strike,
        option_type=opt_type_upper,
        strike_step=strike_step,
    )
    return OptionValidationResult(
        True,
        "",
        expected,
        expiry_date,
        strike,
        opt_type_upper,
        moneyness,
    )

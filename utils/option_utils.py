"""
utils/option_utils.py — Option maths utilities
================================================
Premium estimation and expiry calculation.
Used by Agent 4 (Trade Planner) and Agent 6 (Position Manager).
"""

from __future__ import annotations

import os
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
    "NIFTY": {"weekly_expiry_weekday": 1, "lot_size": NIFTY_LOT_SIZE, "strike_step": 50},
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


COMMODITY_DEFAULT_IV = {
    "SILVER": 0.38,
    "SILVERM": 0.38,
    "SILVERMIC": 0.38,
    "GOLD": 0.18,
    "GOLDM": 0.18,
    "CRUDE": 0.42,
    "CRUDEOIL": 0.42,
    "CRUDEOILM": 0.42,
    "NATGAS": 0.65,
    "NATGASMINI": 0.65,
    "NATURALGAS": 0.65,
    "NIFTY": 0.14,
    "BANKNIFTY": 0.18,
    "SENSEX": 0.14,
}


def estimate_atm_premium(
    underlying: float,
    days_to_expiry: int,
    iv: Optional[float] = None,
    symbol: str = "",
) -> float:
    """
    ATM option premium approximation using simplified Black-Scholes.
    Formula: premium ≈ 0.4 × IV × sqrt(T) × S
    where T = days_to_expiry / 365

    Args:
        underlying:       Spot price
        days_to_expiry:   calendar days until expiry
        iv:               implied volatility (if None or 0.14, uses commodity-specific IV)
        symbol:           commodity / instrument symbol to determine appropriate IV

    Returns:
        Estimated premium rounded to nearest ₹5
    """
    T = max(days_to_expiry, 1) / 365
    sym_clean = str(symbol or "").upper().strip()
    effective_iv = iv

    if effective_iv is None or effective_iv <= 0.0 or effective_iv == 0.14:
        matched_iv = None
        for k, v in COMMODITY_DEFAULT_IV.items():
            if k in sym_clean or sym_clean.startswith(k):
                matched_iv = v
                break
        if matched_iv is not None:
            effective_iv = matched_iv
        elif underlying > 150000.0:  # Silver
            effective_iv = 0.38
        elif underlying > 50000.0:   # Gold
            effective_iv = 0.18
        elif 3000.0 < underlying < 15000.0:  # Crude
            effective_iv = 0.42
        elif underlying < 1000.0:    # NatGas
            effective_iv = 0.65
        else:
            effective_iv = iv if (iv is not None and iv > 0) else 0.14

    premium = 0.4 * effective_iv * np.sqrt(T) * underlying
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
    symbol: str | None = None,
) -> tuple[date, int]:
    """
    Returns nearest valid weekly/monthly expiry with at least min_days remaining.
    For MCX commodities (SILVERM, SILVERMIC, GOLD, CRUDEOIL, etc.), queries active Dhan
    scrip master to resolve real live option expiry dates (e.g. 24Sep2026).
    """
    sym = str(symbol or "").upper().strip()
    if not sym:
        ref_dt = reference_date.date() if isinstance(reference_date, datetime) else (reference_date or date.today())
        if ref_dt < date(2026, 6, 1):
            sym = "NIFTY"
        else:
            sym = str(os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper().strip()

    if sym not in INDEX_SPECS:
        ref = reference_date.date() if isinstance(reference_date, datetime) else (reference_date or date.today())
        try:
            from broker.factory import get_broker
            broker = get_broker()
            if hasattr(broker, "_get_mcx_scrip_master"):
                df_mcx = broker._get_mcx_scrip_master()
                if not df_mcx.empty:
                    names = ["SILVERM", "SILVER"] if "SILVER" in sym else ([sym + "M", sym] if not sym.endswith("M") else [sym, sym[:-1]])
                    sub = df_mcx[(df_mcx["SM_SYMBOL_NAME"].isin(names)) & (df_mcx["SEM_OPTION_TYPE"].isin(["CE", "PE"]))]
                    if not sub.empty:
                        ref_str = ref.isoformat()
                        future_exp = sorted([
                            exp[:10] for exp in sub["SEM_EXPIRY_DATE"].astype(str).unique()
                            if exp[:10] >= ref_str
                        ])
                        if future_exp:
                            exp_dt = date.fromisoformat(future_exp[0])
                            dte = (exp_dt - ref).days
                            if dte >= min_days:
                                return exp_dt, dte
                            elif len(future_exp) > 1:
                                exp_dt = date.fromisoformat(future_exp[1])
                                return exp_dt, (exp_dt - ref).days
        except Exception:
            pass

        try:
            from utils.instrument_selector import get_nearest_expiry as mcx_get_nearest_expiry
            return mcx_get_nearest_expiry(min_days=min_days, ref_date=ref, symbol=sym)
        except Exception:
            pass

    if isinstance(reference_date, datetime):
        today = reference_date.date()
    elif isinstance(reference_date, date):
        today = reference_date
    else:
        today = date.today()
    expiry = get_weekly_expiry(today, symbol=sym)
    days_to_exp = (expiry - today).days

    if days_to_exp < min_days:
        next_week_ref = today + timedelta(days=7)
        expiry = get_weekly_expiry(next_week_ref, symbol=sym)
        days_to_exp = (expiry - today).days

    return expiry, days_to_exp


def get_weekly_expiry(reference_day: date, symbol: str | None = None) -> date:
    sym = str(symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
    if sym not in INDEX_SPECS:
        try:
            from utils.instrument_selector import get_weekly_expiry as mcx_get_weekly_expiry
            return mcx_get_weekly_expiry(trade_date=reference_day, symbol=sym)
        except Exception:
            pass

    weekday = expiry_weekday(sym)
    days_to_expiry = (weekday - reference_day.weekday()) % 7
    expiry = reference_day + timedelta(days=days_to_expiry)
    exchange = "BSE" if sym == "SENSEX" else "NSE"
    if days_to_expiry == 0 and not is_trading_day(expiry, exchange=exchange):
        expiry += timedelta(days=7)
    if is_trading_day(expiry, exchange=exchange):
        return expiry
    return previous_trading_day(expiry, symbol=sym)


def expiry_weekday(symbol: str | None = None) -> int:
    sym = str(symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
    spec = INDEX_SPECS.get(sym)
    if spec is None:
        return 4  # Default Friday if not specified
    return int(spec["weekly_expiry_weekday"])


def is_expiry_day(day: Union[date, datetime], symbol: str | None = None) -> bool:
    sym = str(symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
    if sym not in INDEX_SPECS:
        try:
            from utils.instrument_selector import is_expiry_day as mcx_is_expiry_day
            ref = day.date() if isinstance(day, datetime) else day
            return mcx_is_expiry_day(day=ref, symbol=sym)
        except Exception:
            pass
    ref = day.date() if isinstance(day, datetime) else day
    return ref == get_weekly_expiry(ref, symbol=sym)


def get_index_lot_size(symbol: str | None = None) -> int:
    sym = str(symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
    spec = INDEX_SPECS.get(sym)
    if spec is None:
        try:
            from utils.instrument_selector import get_instrument
            return get_instrument(sym).lot_size
        except Exception:
            return 1
    return int(spec["lot_size"])


def get_index_strike_step(symbol: str | None = None) -> int:
    sym = str(symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
    if "SILVER" in sym:
        return 1000  # MCX Silver options trade in 1000 strike steps
    spec = INDEX_SPECS.get(sym)
    if spec is None:
        try:
            from utils.instrument_selector import get_instrument
            return get_instrument(sym).strike_step
        except Exception:
            return 1000 if "SILVER" in sym else 500
    return int(spec["strike_step"])


def previous_trading_day(day: date, symbol: str | None = None) -> date:
    sym = str(symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
    exchange = "BSE" if sym == "SENSEX" else "NSE"
    current = day - timedelta(days=1)
    while not is_trading_day(current, exchange=exchange):
        current -= timedelta(days=1)
    return current


def build_option_symbol(
    symbol:      str | None = None,
    expiry_date: "date" = None,
    strike:      int = 0,
    option_type: str = "CE",
) -> str:
    """
    Build canonical option symbol.
    For Dhan MCX: Format {SYMBOL}-{DDMonYYYY}-{STRIKE}-{CE/PE}
    Example: CRUDEOIL-17Sep2026-7000-CE, SILVERM-24Sep2026-285000-PE
    For NSE: NIFTY25MAR2722500CE
    """
    sym_upper = str(symbol or os.getenv("INSTRUMENT", "SILVERM")).upper().strip()
    is_mcx = (
        sym_upper in ("CRUDEOIL", "CRUDEOILM", "NATURALGAS", "NATGASMINI", "SILVER", "SILVERM", "SILVERMIC", "GOLD", "GOLDM")
        or any(k in sym_upper for k in ("SILVER", "GOLD", "CRUDE", "NAT"))
    )
    if is_mcx:
        root = "SILVERM" if sym_upper == "SILVERMIC" else sym_upper
        exp_str = expiry_date.strftime("%d%b%Y") if expiry_date else ""
        return f"{root}-{exp_str}-{int(strike)}-{option_type.upper()}"

    expiry_str = expiry_date.strftime("%y%b%d").upper() if expiry_date else ""
    return f"{sym_upper}{expiry_str}{int(strike)}{option_type}"


def get_atm_strike(ltp: float, strike_step: int | None = None, symbol: str | None = None) -> int:
    """Round LTP to nearest strike step to get ATM strike."""
    if strike_step is None:
        strike_step = get_index_strike_step(symbol)
    step = max(int(strike_step or 500), 1)
    return int(round(ltp / step) * step)


def classify_strike_moneyness(
    *,
    underlying: float,
    strike: int,
    option_type: str,
    strike_step: int | None = None,
    symbol: str | None = None,
) -> str:
    atm = get_atm_strike(underlying, strike_step, symbol=symbol)
    if strike == atm:
        return "ATM"
    if option_type == "CE":
        return "ITM" if strike < atm else "OTM"
    return "ITM" if strike > atm else "OTM"


def get_nearest_expiry_for_instrument(
    instrument_name: str | None = None,
    min_days: int = MIN_DAYS_TO_EXPIRY,
) -> tuple["date", int]:
    sym = str(instrument_name or os.getenv("INSTRUMENT", "SILVERM")).upper()
    return get_nearest_expiry(min_days=min_days, symbol=sym)


def build_option_symbol_for_instrument(
    instrument_name: str | None = None,
    expiry_date:     "date" = None,
    strike:          int = 0,
    option_type:     str = "CE",
) -> str:
    sym = str(instrument_name or os.getenv("INSTRUMENT", "SILVERM")).upper()
    return build_option_symbol(sym, expiry_date, strike, option_type)


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
    if expiry_date != get_weekly_expiry(expiry_date, symbol=symbol):
        return OptionValidationResult(False, "invalid_weekly_expiry", expected, expiry_date, strike, opt_type_upper)
    if not is_trading_day(expiry_date):
        return OptionValidationResult(False, "expiry_not_trading_day", expected, expiry_date, strike, opt_type_upper)

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

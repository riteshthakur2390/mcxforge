"""
utils/instrument_selector.py — MCX Multi-Instrument Futures Selector & Specifications
=====================================================================================
Multi-instrument platform abstraction supporting:
  - SILVERMIC (Silver Micro, 1 kg) — Default initial instrument
  - GOLD / GOLDM (Gold Mini / Gold Regular)
  - CRUDEOIL (Crude Oil, 100 Barrels)
  - NATURALGAS (Natural Gas, 1250 mmBtu)
  - Future MCX instruments
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Optional, List, Tuple
import pytz

from utils.brokerage_calculator import calculate_commodity_trade_charges, TradeCharges

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class InstrumentConfig:
    """
    Complete specification for an MCX commodity or exchange instrument.
    Decouples all strategy, risk, execution, and planner code from specific instrument assumptions.
    """
    name:                       str           # e.g. "SILVERMIC", "GOLD", "CRUDEOIL", "NATURALGAS"
    exchange:                   str           # "MCX"
    segment:                    str           # Broker segment: "MCX_COMM", "MCX_FO"
    underlying:                 str           # "SILVER", "GOLD", "CRUDEOIL", "NATURALGAS"
    symbol_prefix:              str           # e.g. "SILVERMIC"
    lot_size:                   int           # Contract multiplier: SILVERMIC=1, GOLD=100, CRUDEOIL=100
    tick_size:                  float         # Minimum price movement: 1.0 for Silver/Gold/Crude, 0.10 for NatGas
    tick_value:                 float         # Rupee PnL value per tick per lot
    round_number_step:          float         # Psychological round numbers (SILVERMIC=500, CRUDE=50)
    strike_step:                int           # Strike step alias for compatibility
    margin_pct_estimate:        float         # Estimated SPAN + Exposure margin requirement (e.g. 0.13 = 13%)
    tender_days_before_expiry:  int           # Compulsory delivery tender window (e.g. 5 days for metals, 0 for cash)
    contract_cycle_months:      List[int]     # Active expiry months: e.g. [2,4,6,8,11] for Silver Micro
    session_start:              str = "09:00" # Normal morning start IST
    session_end:                str = "23:30" # Normal close IST (23:55 during winter DST)
    evening_session_start:      str = "17:00" # US / European active session
    # Broker-specific keys
    dhan_segment:               str = "MCX_COMM"
    dhan_security_id:           str = ""
    upstox_key:                 str = ""
    kite_symbol:                str = ""
    reason:                     str = "default"
    expiry_day:                 str = "Active"

    def __post_init__(self) -> None:
        if self.strike_step == 0:
            self.strike_step = int(self.round_number_step)

    # ── PRICE & TICK MATH ───────────────────────────────────────────────────────
    def round_to_tick(self, price: float) -> float:
        """Round price to valid exchange tick increments."""
        if self.tick_size <= 0:
            return round(price, 2)
        return round(round(price / self.tick_size) * self.tick_size, 2)

    def get_round_number(self, price: float) -> float:
        """Find nearest psychological round number."""
        step = max(self.round_number_step, 1.0)
        return round(round(price / step) * step, 2)

    def get_atm_strike(self, ltp: float) -> int:
        """Backward compatibility alias for round_number."""
        return int(self.get_round_number(ltp))

    # ── P&L & TURNOVER ──────────────────────────────────────────────────────────
    def calculate_turnover(self, price: float, quantity: int) -> float:
        """Calculate total trade contract turnover in INR."""
        mult = (self.tick_value / self.tick_size) if self.tick_size > 0 else 1.0
        return price * quantity * mult

    def calculate_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        direction: str = "BUY",
    ) -> float:
        """Calculate gross profit/loss in INR for long or short futures."""
        is_long = str(direction).upper() in ("BUY", "BUY_CALL", "LONG")
        mult = (self.tick_value / self.tick_size) if self.tick_size > 0 else 1.0
        diff = (exit_price - entry_price) if is_long else (entry_price - exit_price)
        return round(diff * quantity * mult, 2)

    def calculate_charges(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        direction: str = "BUY",
    ) -> TradeCharges:
        """Calculate CTT, MCX exchange fees, stamp duty, SEBI turnover, and brokerage."""
        return calculate_commodity_trade_charges(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            direction=direction,
            tick_size=self.tick_size,
            tick_value=self.tick_value,
        )

    def calculate_margin(self, price: float, lots: int = 1) -> float:
        """Estimate total required margin for positions."""
        turnover = self.calculate_turnover(price, lots * self.lot_size)
        return round(turnover * self.margin_pct_estimate, 2)

    # ── CONTRACT & EXPIRY LOGIC ─────────────────────────────────────────────────
    def get_active_contract(self, from_date: Optional[date] = None) -> str:
        """
        Derive active front-month contract symbol.
        e.g. For SILVERMIC on 2026-09-04 -> SILVERMIC26NOVFUT
        """
        d = from_date or date.today()
        month_codes = {
            1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
            7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC",
        }
        yy = str(d.year)[-2:]
        # Find next active contract month
        target_month = None
        target_year = d.year
        for m in self.contract_cycle_months:
            if m >= d.month:
                target_month = m
                break
        if target_month is None:
            target_month = self.contract_cycle_months[0]
            target_year += 1
            yy = str(target_year)[-2:]

        mon_str = month_codes.get(target_month, "FUT")
        return f"{self.symbol_prefix}{yy}{mon_str}FUT"

    def get_next_expiry(self, from_date: Optional[date] = None) -> date:
        """Find next expiry date based on contract cycle months."""
        d = from_date or date.today()
        target_month = None
        target_year = d.year
        for m in self.contract_cycle_months:
            if m >= d.month:
                target_month = m
                break
        if target_month is None:
            target_month = self.contract_cycle_months[0]
            target_year += 1

        exp_day = 19 if self.underlying in ("CRUDEOIL", "NATURALGAS") else 5
        try:
            return date(target_year, target_month, exp_day)
        except ValueError:
            return date(target_year, target_month, 28)

    def get_dte(self, from_date: Optional[date] = None) -> int:
        """Days to next contract expiry."""
        d = from_date or date.today()
        exp = self.get_next_expiry(d)
        return max(0, (exp - d).days)

    def is_tender_period(self, check_date: Optional[date] = None, expiry_date: Optional[date] = None) -> bool:
        """Returns True if within delivery tender period (last N days before expiry)."""
        if self.tender_days_before_expiry <= 0:
            return False
        d = check_date or date.today()
        exp = expiry_date or self.get_next_expiry(d)
        days_left = (exp - d).days
        return 0 <= days_left <= self.tender_days_before_expiry

    def is_session_active(self, current_dt: Optional[datetime] = None) -> bool:
        """True if current IST time is within MCX trading session."""
        dt = current_dt or datetime.now(IST)
        now_time = dt.strftime("%H:%M")
        return self.session_start <= now_time <= self.session_end

    @property
    def symbol(self) -> str:
        return self.name

    @property
    def is_commodity(self) -> bool:
        return self.exchange == "MCX"

    @property
    def is_deliverable(self) -> bool:
        return self.tender_days_before_expiry > 0

    def __str__(self) -> str:
        return (
            f"Instrument({self.name} | ex={self.exchange} | lot={self.lot_size} | "
            f"tick={self.tick_size} | tick_val=₹{self.tick_value} | session={self.session_start}-{self.session_end})"
        )


# ── PREDEFINED COMMODITY CONFIGURATIONS ───────────────────────────────────────

SILVERMIC_CONFIG = InstrumentConfig(
    name                      = "SILVERMIC",
    exchange                  = "MCX",
    segment                   = "MCX_COMM",
    underlying                = "SILVER",
    symbol_prefix             = "SILVERMIC",
    lot_size                  = 1,             # 1 kg
    tick_size                 = 1.0,           # ₹1.00
    tick_value                = 1.0,           # ₹1.00 per point
    round_number_step         = 1000.0,        # ₹1000 round levels
    strike_step               = 1000,          # MCX Silver options have 1000 strike steps
    margin_pct_estimate       = 0.13,          # ~13% SPAN + Exposure
    tender_days_before_expiry = 5,             # 5 tender days
    contract_cycle_months     = [2, 4, 6, 8, 11], # Feb, Apr, Jun, Aug, Nov
    session_start             = "09:00",
    session_end               = "23:30",
    dhan_segment              = "MCX_COMM",
    upstox_key                = "MCX_FO|SILVERMIC",
    kite_symbol               = "MCX:SILVERMIC",
    reason                    = "mcx_silver_micro",
)

SILVERM_CONFIG = InstrumentConfig(
    name                      = "SILVERM",
    exchange                  = "MCX",
    segment                   = "MCX_COMM",
    underlying                = "SILVERM",
    symbol_prefix             = "SILVERM",
    lot_size                  = 5,             # 5 kg
    tick_size                 = 1.0,           # ₹1.00
    tick_value                = 5.0,           # ₹5.00 per point
    round_number_step         = 1000.0,        # ₹1000 round levels
    strike_step               = 1000,          # MCX Silver options have 1000 strike steps
    margin_pct_estimate       = 0.13,          # ~13% SPAN + Exposure
    tender_days_before_expiry = 5,             # 5 tender days
    contract_cycle_months     = [2, 4, 6, 8, 11], # Feb, Apr, Jun, Aug, Nov
    session_start             = "09:00",
    session_end               = "23:30",
    dhan_segment              = "MCX_COMM",
    upstox_key                = "MCX_FO|SILVERM",
    kite_symbol               = "MCX:SILVERM",
    reason                    = "default_mcx_silver_mini",
)

GOLD_CONFIG = InstrumentConfig(
    name                      = "GOLD",
    exchange                  = "MCX",
    segment                   = "MCX_COMM",
    underlying                = "GOLD",
    symbol_prefix             = "GOLD",
    lot_size                  = 100,           # 100g contract (quoted per 10g)
    tick_size                 = 1.0,           # ₹1.00
    tick_value                = 10.0,          # ₹10.00 per ₹1 move in quote
    round_number_step         = 500.0,
    strike_step               = 500,
    margin_pct_estimate       = 0.10,
    tender_days_before_expiry = 5,
    contract_cycle_months     = [2, 4, 6, 8, 10, 12],
    session_start             = "09:00",
    session_end               = "23:30",
    dhan_segment              = "MCX_COMM",
    upstox_key                = "MCX_FO|GOLD",
    kite_symbol               = "MCX:GOLD",
    reason                    = "mcx_gold",
)

CRUDEOIL_CONFIG = InstrumentConfig(
    name                      = "CRUDEOIL",
    exchange                  = "MCX",
    segment                   = "MCX_COMM",
    underlying                = "CRUDEOIL",
    symbol_prefix             = "CRUDEOIL",
    lot_size                  = 100,           # 100 barrels
    tick_size                 = 1.0,
    tick_value                = 100.0,         # ₹100 per ₹1 move
    round_number_step         = 50.0,
    strike_step               = 50,
    margin_pct_estimate       = 0.25,
    tender_days_before_expiry = 0,             # Cash settled
    contract_cycle_months     = list(range(1, 13)),
    session_start             = "09:00",
    session_end               = "23:30",
    dhan_segment              = "MCX_COMM",
    upstox_key                = "MCX_FO|CRUDEOIL",
    kite_symbol               = "MCX:CRUDEOIL",
    reason                    = "mcx_crudeoil",
)

NATURALGAS_CONFIG = InstrumentConfig(
    name                      = "NATURALGAS",
    exchange                  = "MCX",
    segment                   = "MCX_COMM",
    underlying                = "NATURALGAS",
    symbol_prefix             = "NATURALGAS",
    lot_size                  = 1250,          # 1250 mmBtu
    tick_size                 = 0.10,
    tick_value                = 125.0,         # ₹125 per 0.10 move
    round_number_step         = 5.0,
    strike_step               = 5,
    margin_pct_estimate       = 0.30,
    tender_days_before_expiry = 0,             # Cash settled
    contract_cycle_months     = list(range(1, 13)),
    session_start             = "09:00",
    session_end               = "23:30",
    dhan_segment              = "MCX_COMM",
    upstox_key                = "MCX_FO|NATURALGAS",
    kite_symbol               = "MCX:NATURALGAS",
    reason                    = "mcx_naturalgas",
)

# Legacy fallbacks for SignalForge compatibility
NIFTY_CONFIG = InstrumentConfig(
    name                      = "NIFTY",
    exchange                  = "NFO",
    segment                   = "NSE_FNO",
    underlying                = "NIFTY 50",
    symbol_prefix             = "NIFTY",
    lot_size                  = 65,
    tick_size                 = 0.05,
    tick_value                = 3.25,
    round_number_step         = 50.0,
    strike_step               = 50,
    margin_pct_estimate       = 0.12,
    tender_days_before_expiry = 0,
    contract_cycle_months     = list(range(1, 13)),
    session_start             = "09:15",
    session_end               = "15:30",
    reason                    = "legacy_nifty",
)

SENSEX_CONFIG = InstrumentConfig(
    name                      = "SENSEX",
    exchange                  = "BFO",
    segment                   = "BSE_FNO",
    underlying                = "SENSEX",
    symbol_prefix             = "SENSEX",
    lot_size                  = 10,
    tick_size                 = 0.05,
    tick_value                = 0.50,
    round_number_step         = 100.0,
    strike_step               = 100,
    margin_pct_estimate       = 0.12,
    tender_days_before_expiry = 0,
    contract_cycle_months     = list(range(1, 13)),
    session_start             = "09:15",
    session_end               = "15:30",
    reason                    = "legacy_sensex",
)


INSTRUMENT_REGISTRY = {
    "SILVERM":    SILVERM_CONFIG,
    "SILVERMIC":  SILVERMIC_CONFIG,
    "SILVER":     SILVERM_CONFIG,
    "GOLD":       GOLD_CONFIG,
    "GOLDM":      GOLD_CONFIG,
    "CRUDEOIL":   CRUDEOIL_CONFIG,
    "CRUDEOILM":  CRUDEOIL_CONFIG,
    "CRUDE":      CRUDEOIL_CONFIG,
    "NATURALGAS": NATURALGAS_CONFIG,
    "NATGAS":     NATURALGAS_CONFIG,
    "NIFTY":      NIFTY_CONFIG,
    "SENSEX":     SENSEX_CONFIG,
}


# ── SELECTOR FUNCTIONS ────────────────────────────────────────────────────────

def get_instrument(
    symbol: Optional[str] = None,
    trade_date: Optional[date] = None,
    verbose: bool = False,
    override: Optional[str] = None,
) -> InstrumentConfig:
    """
    Get the configured instrument specification.
    Defaults to SILVERM (Silver Mini, 5 kg).
    """
    sym = override or symbol
    chosen = (sym or os.getenv("INSTRUMENT", "SILVERM")).strip().upper()
    cfg = INSTRUMENT_REGISTRY.get(chosen, SILVERM_CONFIG)
    if verbose:
        try:
            from loguru import logger
            logger.info(f"[InstrumentSelector] Selected: {cfg}")
        except Exception:
            print(f"[InstrumentSelector] Selected: {cfg}")
    return cfg


def get_today_instrument() -> InstrumentConfig:
    """Convenience accessor for current session instrument."""
    return get_instrument()


get_instrument_config = get_instrument


def get_instrument_for_signal(signal_payload: dict) -> InstrumentConfig:
    """Derive instrument config from signal dictionary or fallback to default."""
    sym = signal_payload.get("symbol") or signal_payload.get("instrument")
    return get_instrument(sym)


def compute_atr(df, period: int = 14) -> tuple[float, float]:
    """
    Compute ATR and ATR percentage using OHLC candles.
    Returns (atr_points, atr_pct_of_close).
    """
    if df is None or len(df) < max(period + 1, 5):
        return 0.0, 0.0
    try:
        import pandas as pd
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
        atr_pct = (atr / last_close * 100.0) if last_close > 0 else 0.0
        return round(atr, 2), round(atr_pct, 4)
    except Exception:
        return 0.0, 0.0


def apply_futures_slippage(price: float, direction: str, slippage_pct: float = 0.05) -> float:
    """
    Apply realistic execution slippage for futures.
    BUY orders slip higher; SELL orders slip lower.
    """
    p = float(price or 0.0)
    if p <= 0:
        return 0.0
    mult = slippage_pct / 100.0
    is_buy = str(direction).upper() in ("BUY", "BUY_CALL", "LONG")
    if is_buy:
        return round(p * (1.0 + mult), 2)
    return round(p * (1.0 - mult), 2)


# Backward-compatible alias for existing callers
apply_option_slippage = apply_futures_slippage


def estimate_round_trip_costs(
    entry_price: float,
    exit_price: float,
    quantity: int,
    order_brokerage: float = 20.0,
    txn_cost_pct: float = 0.0,
) -> float:
    """Calculate round-trip brokerage and taxes for futures."""
    try:
        charges = calculate_commodity_trade_charges(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            order_brokerage=order_brokerage,
        )
        return charges.total_charges
    except Exception:
        return round(order_brokerage * 2.0, 2)


def estimate_option_pnl(
    entry_premium: float,
    underlying_chg_pct: float,
    direction: str = "BUY",
    delta: float = 0.50,
    underlying_spot: float = 0.0,
) -> float:
    """Direction-aware P&L estimation for underlying price movement."""
    ep = float(entry_premium or 0.0)
    if ep <= 0:
        return 0.0
    is_short = str(direction).upper() in ("SELL", "BUY_PUT", "SHORT")
    mult = -1.0 if is_short else 1.0
    pnl_factor = 1.0 + (mult * (underlying_chg_pct / 100.0))
    return round(max(0.05, ep * pnl_factor), 2)


def canonical_option_sym(sym: str) -> str:
    """Normalize commodity/futures/option symbol strings across brokers."""
    if not sym:
        return ""
    import re
    clean = str(sym).upper().strip()
    for prefix in ["MCX_FO|", "MCX_COMM|", "MCX:", "NSE_FO|", "BFO|", "NSE_INDEX|", "BSE_INDEX|", "NSE:", "BSE:"]:
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
    return re.sub(r"[\s\-_]+", "", clean)


def build_option_symbol(symbol: str, expiry_date: date, strike: int, option_type: str = "FUT") -> str:
    """Return canonical symbol for commodity contract or option."""
    mon = expiry_date.strftime("%b").upper()
    yy = str(expiry_date.year)[-2:]
    clean_sym = symbol.replace("MCX:", "").replace("MCX_FO|", "").strip()
    if not option_type or option_type.upper() in ("FUT", "FUTURE"):
        return f"{clean_sym}{yy}{mon}FUT"
    return f"{clean_sym}{yy}{mon}{int(strike)}{option_type.upper()}"


def get_nearest_expiry(min_days: int = 0, ref_date: Optional[date] = None, symbol: str = "SILVERMIC") -> tuple[date, int]:
    cfg = get_instrument(symbol)
    d = ref_date or date.today()
    exp = cfg.get_next_expiry(d)
    dte = cfg.get_dte(d)
    return exp, dte


def get_weekly_expiry(trade_date: Optional[date] = None, symbol: str = "SILVERMIC") -> date:
    cfg = get_instrument(symbol)
    return cfg.get_next_expiry(trade_date or date.today())


def is_expiry_day(day: Optional[date] = None, symbol: str = "SILVERMIC") -> bool:
    cfg = get_instrument(symbol)
    d = day or date.today()
    return cfg.is_tender_period(d) or cfg.get_dte(d) == 0


def get_atm_strike(ltp: float, step: int = 500) -> int:
    s = max(step, 1)
    return int(round(ltp / s) * s)


def get_index_lot_size(symbol: str) -> int:
    return get_instrument(symbol).lot_size


def get_index_strike_step(symbol: str) -> int:
    return get_instrument(symbol).strike_step


def estimate_atm_premium(underlying: float, days_to_expiry: int = 0, iv: float = 0.14) -> float:
    """Estimate basis or price for underlying."""
    return round(float(underlying or 0.0), 2)


@dataclass(frozen=True)
class OptionValidationResult:
    valid: bool
    reason: str = ""
    symbol: str = ""
    expiry_date: Optional[date] = None
    strike: int = 0
    option_type: str = ""
    moneyness: str = ""


def validate_option_contract(contract_symbol: str, current_spot: float = 0.0, lot_size: int = 1) -> OptionValidationResult:
    sym = canonical_option_sym(contract_symbol)
    return OptionValidationResult(
        valid=bool(sym),
        symbol=sym,
        strike=int(current_spot),
        option_type="FUT" if "FUT" in sym else ("CE" if sym.endswith("CE") else "PE"),
        moneyness="ATM",
    )

"""
instruments/silverm.py — MCX Silver Mini Futures & Options Underlying (SILVERM)
===============================================================================
SILVERM is the primary Silver instrument of MCXForge.
- Contract Unit: 5 Kilograms (5 kg)
- Tick Size: ₹1.00 per kg
- Tick Value: ₹5.00 per point per lot (5 kg × ₹1.00)
- Options Chain: Supported by MCX (CE & PE contracts available)
- Expiry Date: Last calendar/trading day of February, April, June, August, November
- Tender Period: 5 days prior to expiry (Compulsory physical delivery)
- Trading Session: 09:00 - 23:30 IST (23:55 during US DST)
"""

from datetime import date, datetime, timedelta
import calendar
from typing import Optional
import pytz

from instruments.base import InstrumentConfig, CommoditySector, SessionSpec, MarginSpec, ContractSpec

IST = pytz.timezone("Asia/Kolkata")

SILVERM_CONFIG = InstrumentConfig(
    symbol="SILVERM",
    name="Silver Mini Futures",
    exchange="MCX",
    segment="MCX_COMM",
    sector=CommoditySector.PRECIOUS_METALS,
    lot_size=5,                         # 5 kg (Mini with options)
    strike_step=1000,                   # ₹1000 strike interval for MCX Silver options
    tick_size=1.0,                      # ₹1.0
    tick_value=5.0,                     # ₹5.0 per point per lot
    contract_unit="5 kg",
    quotation_unit="1 kg",
    currency="INR",
    session=SessionSpec(
        open_time="09:00",
        close_time="23:30",
        signal_start="09:15",
        signal_cutoff="23:00",
        intraday_squareoff="23:15",
    ),
    margin=MarginSpec(
        initial_margin_pct=11.5,
        extreme_loss_margin_pct=3.5,
        additional_margin_pct=0.0,
    ),
    contract_cycle_months=[2, 4, 6, 8, 11], # Feb, Apr, Jun, Aug, Nov
    tender_period_days=5,
    rollover_dte_threshold=3,
    default_stop_atr_mult=1.5,
    default_target_atr_mult=3.0,
    is_deliverable=True,
)

# For backward compatibility during migration, SILVERMIC_CONFIG references SILVERM_CONFIG
SILVERMIC_CONFIG = SILVERM_CONFIG


def get_silverm_expiry(year: int, month: int) -> date:
    """Calculate the last business day of the expiry month for SILVERM."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def get_active_silverm_contract(as_of: Optional[date] = None) -> ContractSpec:
    """
    Resolves the front-month active SILVERM contract for trading.
    Automatically advances to next contract if within tender period or DTE <= 3.
    """
    if as_of is None:
        as_of = datetime.now(IST).date()

    cycle_months = SILVERM_CONFIG.contract_cycle_months
    year = as_of.year

    for y in [year, year + 1]:
        for m in cycle_months:
            candidate_expiry = get_silverm_expiry(y, m)
            if candidate_expiry < as_of:
                continue
            dte = (candidate_expiry - as_of).days
            # Rollover check
            if dte > SILVERM_CONFIG.rollover_dte_threshold and not SILVERM_CONFIG.is_in_tender_period(candidate_expiry, as_of):
                trading_symbol = f"SILVERM-{candidate_expiry.strftime('%d%b%Y')}-FUT"
                return ContractSpec(
                    symbol="SILVERM",
                    trading_symbol=trading_symbol,
                    expiry_date=candidate_expiry,
                    lot_size=SILVERM_CONFIG.lot_size,
                    tick_size=SILVERM_CONFIG.tick_size,
                    tick_value=SILVERM_CONFIG.tick_value,
                )

    # Fallback to nearest upcoming cycle month
    next_m = cycle_months[0]
    next_y = year + 1
    fallback_expiry = get_silverm_expiry(next_y, next_m)
    return ContractSpec(
        symbol="SILVERM",
        trading_symbol=f"SILVERM-{fallback_expiry.strftime('%d%b%Y')}-FUT",
        expiry_date=fallback_expiry,
        lot_size=SILVERM_CONFIG.lot_size,
        tick_size=SILVERM_CONFIG.tick_size,
        tick_value=SILVERM_CONFIG.tick_value,
    )


# Compatibility alias
get_active_silvermic_contract = get_active_silverm_contract


def get_active_silverm_option_expiry(as_of: Optional[date] = None, rollover_days: int = 2) -> date:
    """
    Resolves the active monthly option expiry for SILVERM.
    Rule:
      1. Trades in the same monthly expiry whenever possible.
      2. If within rollover_days (default 5 days) of expiry, advances to the next month's expiry.
    """
    if as_of is None:
        as_of = datetime.now(IST).date()

    known_expiries = [
        date(2026, 7, 28),
        date(2026, 8, 26),
        date(2026, 9, 24),
        date(2026, 10, 27),
        date(2026, 11, 23),
        date(2026, 12, 28),
        date(2027, 1, 25),
        date(2027, 2, 19),
        date(2027, 3, 25),
        date(2027, 4, 23),
        date(2027, 5, 26),
        date(2027, 6, 23),
    ]
    upcoming = [e for e in known_expiries if e >= as_of]
    if not upcoming:
        m = as_of.month + 1 if as_of.month < 12 else 1
        y = as_of.year if as_of.month < 12 else as_of.year + 1
        d = date(y, m, 24)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    front = upcoming[0]
    dte = (front - as_of).days
    if dte <= rollover_days and len(upcoming) > 1:
        return upcoming[1]
    return front


def resolve_silverm_option_symbol(
    as_of: date,
    strike: int,
    opt_type: str,
    rollover_days: int = 2,
) -> tuple[str, date, int]:
    """
    Constructs the canonical Dhan/MCX option trading symbol, e.g. SILVERM-24Sep2026-244500-PE.
    Returns (trading_symbol, expiry_date, dte).
    """
    exp = get_active_silverm_option_expiry(as_of, rollover_days)
    dte = max(1, (exp - as_of).days)
    exp_str = exp.strftime("%d%b%Y")
    sym = f"SILVERM-{exp_str}-{int(strike)}-{opt_type.upper()}"
    return sym, exp, dte

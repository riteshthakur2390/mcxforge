"""
instruments/silvermic.py — MCX Silver Micro Futures (SILVERMIC)

SILVERMIC is the primary launch instrument of MCXForge.
- Contract Unit: 1 Kilogram (1 kg)
- Tick Size: ₹1.00 per kg
- Tick Value: ₹1.00 per point per lot
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

SILVERMIC_CONFIG = InstrumentConfig(
    symbol="SILVERMIC",
    name="Silver Micro Futures",
    exchange="MCX",
    segment="MCX_COMM",
    sector=CommoditySector.PRECIOUS_METALS,
    lot_size=1,                         # 1 kg
    tick_size=1.0,                      # ₹1.0
    tick_value=1.0,                     # ₹1.0 per point per lot
    contract_unit="1 kg",
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

SILVERM_CONFIG = InstrumentConfig(
    symbol="SILVERM",
    name="Silver Mini Futures",
    exchange="MCX",
    segment="MCX_COMM",
    sector=CommoditySector.PRECIOUS_METALS,
    lot_size=5,                         # 5 kg
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


def get_silvermic_expiry(year: int, month: int) -> date:
    """Calculate the last business day of the expiry month for SILVERMIC."""
    last_day = calendar.monthrange(year, month)[1]
    d = date(year, month, last_day)
    # Adjust for weekend
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def get_active_silvermic_contract(as_of: Optional[date] = None) -> ContractSpec:
    """
    Resolves the front-month active SILVERMIC contract for trading.
    Automatically advances to next contract if within tender period or DTE <= 3.
    """
    if as_of is None:
        as_of = datetime.now(IST).date()

    cycle_months = SILVERMIC_CONFIG.contract_cycle_months
    year = as_of.year

    for y in [year, year + 1]:
        for m in cycle_months:
            candidate_expiry = get_silvermic_expiry(y, m)
            if candidate_expiry < as_of:
                continue
            dte = (candidate_expiry - as_of).days
            # Rollover check
            if dte > SILVERMIC_CONFIG.rollover_dte_threshold and not SILVERMIC_CONFIG.is_in_tender_period(candidate_expiry, as_of):
                mon_abbr = calendar.month_abbr[m]
                trading_symbol = f"SILVERMIC-{candidate_expiry.strftime('%d%b%Y')}-FUT"
                return ContractSpec(
                    symbol="SILVERMIC",
                    trading_symbol=trading_symbol,
                    expiry_date=candidate_expiry,
                    lot_size=SILVERMIC_CONFIG.lot_size,
                    tick_size=SILVERMIC_CONFIG.tick_size,
                    tick_value=SILVERMIC_CONFIG.tick_value,
                )

    # Fallback to nearest upcoming cycle month
    next_m = cycle_months[0]
    next_y = year + 1
    fallback_expiry = get_silvermic_expiry(next_y, next_m)
    return ContractSpec(
        symbol="SILVERMIC",
        trading_symbol=f"SILVERMIC-{fallback_expiry.strftime('%d%b%Y')}-FUT",
        expiry_date=fallback_expiry,
        lot_size=SILVERMIC_CONFIG.lot_size,
        tick_size=SILVERMIC_CONFIG.tick_size,
        tick_value=SILVERMIC_CONFIG.tick_value,
    )

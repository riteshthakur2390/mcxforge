"""
instruments/gold.py — MCX Gold (GOLD) & Gold Mini (GOLDM) Futures

- GOLD: 1 kg (1000 grams), tick size ₹1.00 / 10 grams, quotation ₹ per 10g
- GOLDM: 100 grams, tick size ₹1.00 / 10 grams, quotation ₹ per 10g
- Expiry: 5th of February, April, June, August, October, December (GOLD)
          5th of every month (GOLDM)
"""

from datetime import date, datetime, timedelta
import calendar
from typing import Optional
import pytz

from instruments.base import InstrumentConfig, CommoditySector, SessionSpec, MarginSpec, ContractSpec

IST = pytz.timezone("Asia/Kolkata")

GOLDM_CONFIG = InstrumentConfig(
    symbol="GOLDM",
    name="Gold Mini Futures",
    exchange="MCX",
    segment="MCX_COMM",
    sector=CommoditySector.PRECIOUS_METALS,
    lot_size=10,                        # 10 units (100 grams, quoted per 10 grams -> 10 multiplier)
    tick_size=1.0,                      # ₹1.0
    tick_value=10.0,                    # ₹10.0 per point move
    contract_unit="100 grams",
    quotation_unit="10 grams",
    currency="INR",
    session=SessionSpec(
        open_time="09:00",
        close_time="23:30",
        signal_start="09:15",
        signal_cutoff="23:00",
        intraday_squareoff="23:15",
    ),
    margin=MarginSpec(
        initial_margin_pct=10.0,
        extreme_loss_margin_pct=3.0,
        additional_margin_pct=0.0,
    ),
    contract_cycle_months=list(range(1, 13)), # Monthly cycle
    tender_period_days=5,
    rollover_dte_threshold=3,
    default_stop_atr_mult=1.5,
    default_target_atr_mult=3.0,
    is_deliverable=True,
)

GOLD_CONFIG = InstrumentConfig(
    symbol="GOLD",
    name="Gold Futures (Standard)",
    exchange="MCX",
    segment="MCX_COMM",
    sector=CommoditySector.PRECIOUS_METALS,
    lot_size=100,                       # 1 kg = 1000g, quoted per 10g -> 100 multiplier
    tick_size=1.0,
    tick_value=100.0,
    contract_unit="1 kg",
    quotation_unit="10 grams",
    currency="INR",
    session=SessionSpec(
        open_time="09:00",
        close_time="23:30",
        signal_start="09:15",
        signal_cutoff="23:00",
        intraday_squareoff="23:15",
    ),
    margin=MarginSpec(
        initial_margin_pct=10.0,
        extreme_loss_margin_pct=3.0,
    ),
    contract_cycle_months=[2, 4, 6, 8, 10, 12], # Bi-monthly cycle
    tender_period_days=5,
    rollover_dte_threshold=3,
    is_deliverable=True,
)

"""
instruments/crudeoil.py — MCX Crude Oil (CRUDEOIL) & Crude Oil Mini (CRUDEOILM) Futures

- CRUDEOIL: 100 Barrels, tick size ₹1.00 / barrel
- CRUDEOILM: 10 Barrels, tick size ₹1.00 / barrel
- Expiry: Around 19th of every month (Cash settled, non-deliverable)
- Session: 09:00 - 23:30 IST (23:55 during US DST)
"""

from instruments.base import InstrumentConfig, CommoditySector, SessionSpec, MarginSpec

CRUDEOILM_CONFIG = InstrumentConfig(
    symbol="CRUDEOILM",
    name="Crude Oil Mini Futures",
    exchange="MCX",
    segment="MCX_COMM",
    sector=CommoditySector.ENERGY,
    lot_size=10,                        # 10 barrels
    tick_size=1.0,                      # ₹1.0
    tick_value=10.0,                    # ₹10.0 per point move
    contract_unit="10 barrels",
    quotation_unit="1 barrel",
    currency="INR",
    session=SessionSpec(
        open_time="09:00",
        close_time="23:30",
        signal_start="09:15",
        signal_cutoff="23:00",
        intraday_squareoff="23:15",
    ),
    margin=MarginSpec(
        initial_margin_pct=15.0,
        extreme_loss_margin_pct=5.0,
    ),
    contract_cycle_months=list(range(1, 13)), # Monthly cycle
    tender_period_days=0,               # Cash settled — no compulsory physical tender lockout
    rollover_dte_threshold=2,
    default_stop_atr_mult=1.5,
    default_target_atr_mult=3.0,
    is_deliverable=False,
)

CRUDEOIL_CONFIG = InstrumentConfig(
    symbol="CRUDEOIL",
    name="Crude Oil Futures (Standard)",
    exchange="MCX",
    segment="MCX_COMM",
    sector=CommoditySector.ENERGY,
    lot_size=100,                       # 100 barrels
    tick_size=1.0,
    tick_value=100.0,
    contract_unit="100 barrels",
    quotation_unit="1 barrel",
    currency="INR",
    session=SessionSpec(
        open_time="09:00",
        close_time="23:30",
        signal_start="09:15",
        signal_cutoff="23:00",
        intraday_squareoff="23:15",
    ),
    margin=MarginSpec(
        initial_margin_pct=15.0,
        extreme_loss_margin_pct=5.0,
    ),
    contract_cycle_months=list(range(1, 13)),
    tender_period_days=0,
    rollover_dte_threshold=2,
    is_deliverable=False,
)

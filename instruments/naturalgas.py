"""
instruments/naturalgas.py — MCX Natural Gas (NATURALGAS) & Mini (NATGASM) Futures

- NATURALGAS: 1250 mmBtu, tick size ₹0.10, quotation per mmBtu
- NATGASM: 250 mmBtu, tick size ₹0.10, quotation per mmBtu
- Expiry: ~25th–28th of every month (Cash settled)
- Session: 09:00 - 23:30 IST (23:55 during US DST)
"""

from instruments.base import InstrumentConfig, CommoditySector, SessionSpec, MarginSpec

NATURALGAS_CONFIG = InstrumentConfig(
    symbol="NATURALGAS",
    name="Natural Gas Futures",
    exchange="MCX",
    segment="MCX_COMM",
    sector=CommoditySector.ENERGY,
    lot_size=1250,                      # 1250 mmBtu
    tick_size=0.10,                     # ₹0.10
    tick_value=125.0,                   # ₹125.0 per 1.0 point move (12.5 per tick)
    contract_unit="1250 mmBtu",
    quotation_unit="1 mmBtu",
    currency="INR",
    session=SessionSpec(
        open_time="09:00",
        close_time="23:30",
        signal_start="09:15",
        signal_cutoff="23:00",
        intraday_squareoff="23:15",
    ),
    margin=MarginSpec(
        initial_margin_pct=16.0,
        extreme_loss_margin_pct=5.0,
    ),
    contract_cycle_months=list(range(1, 13)),
    tender_period_days=0,
    rollover_dte_threshold=2,
    default_stop_atr_mult=1.5,
    default_target_atr_mult=3.0,
    is_deliverable=False,
)

NATGASM_CONFIG = InstrumentConfig(
    symbol="NATGASM",
    name="Natural Gas Mini Futures",
    exchange="MCX",
    segment="MCX_COMM",
    sector=CommoditySector.ENERGY,
    lot_size=250,                       # 250 mmBtu
    tick_size=0.10,
    tick_value=25.0,
    contract_unit="250 mmBtu",
    quotation_unit="1 mmBtu",
    currency="INR",
    session=SessionSpec(
        open_time="09:00",
        close_time="23:30",
        signal_start="09:15",
        signal_cutoff="23:00",
        intraday_squareoff="23:15",
    ),
    margin=MarginSpec(
        initial_margin_pct=16.0,
        extreme_loss_margin_pct=5.0,
    ),
    contract_cycle_months=list(range(1, 13)),
    tender_period_days=0,
    rollover_dte_threshold=2,
    is_deliverable=False,
)

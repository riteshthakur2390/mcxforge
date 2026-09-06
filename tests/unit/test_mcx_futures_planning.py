"""
tests/unit/test_mcx_futures_planning.py
=======================================
Unit tests validating MCX Commodity Futures integration:
- Instrument configuration & contract cycle resolution (SILVERMIC, GOLDM, CRUDEOILM)
- Futures Greeks filter bypass (delta=1.0, theta=0.0)
- Futures margin-based position sizing
- TradePlan dataclass futures compatibility
- Upstox MCX broker configurations
"""

import pytest
from datetime import date, datetime
import pytz

from utils.instrument_selector import (
    get_instrument,
    InstrumentConfig,
    canonical_option_sym,
    get_nearest_expiry,
    is_expiry_day,
)
from utils.greeks_filter import GreeksFilter
from utils.greeks_position_sizer import size_by_delta, PositionSize
from core.models import TradePlan, RawSignal, Direction, Regime
from broker.upstox_broker import UpstoxBroker

IST = pytz.timezone("Asia/Kolkata")


def test_mcx_instrument_config_specs():
    silver = get_instrument("SILVERMIC")
    assert silver.symbol == "SILVERMIC"
    assert silver.exchange == "MCX"
    assert silver.lot_size == 1
    assert silver.tick_size == 1.0
    assert silver.tick_value == 1.0
    assert silver.is_commodity is True

    gold = get_instrument("GOLDM")
    assert gold.symbol in ("GOLD", "GOLDM")
    assert gold.exchange == "MCX"
    assert gold.lot_size == 100
    assert gold.tick_value == 10.0

    crude = get_instrument("CRUDEOILM")
    assert crude.symbol in ("CRUDEOIL", "CRUDEOILM")
    assert crude.exchange == "MCX"
    assert crude.lot_size == 100
    assert crude.tick_value == 100.0


def test_mcx_active_contract_resolution():
    silver = get_instrument("SILVERMIC")
    ref_date = date(2026, 6, 1)
    contract_sym = silver.get_active_contract(ref_date)
    assert isinstance(contract_sym, str)
    assert "SILVERMIC" in contract_sym
    assert "FUT" in contract_sym
    assert silver.get_dte(ref_date) > 0


def test_mcx_futures_greeks_filter_bypasses_options_math():
    flt = GreeksFilter()
    result = flt.check(
        nifty_ltp=75000.0,
        opt_type="FUT",
        premium_source="COMMODITY_FUTURES",
    )
    assert result["tradeable"] is True
    assert result["delta"] == 1.0
    assert result["theta"] == 0.0
    assert result["greeks_ok"] is True
    assert "futures" in result["reason"].lower()


def test_mcx_futures_position_sizer_margin_based():
    res = size_by_delta(
        premium=75000.0,
        premium_source="COMMODITY_FUTURES",
        capital=200000.0,
        daily_cap_pct=15.0,
    )
    # Available capital = 15% of 200,000 / 3 = 10,000
    # Minimum 1 lot returned
    assert res.lots >= 1
    assert res.net_delta >= 1.0
    assert res.capital_used > 0
    assert "futures" in res.reason.lower()


def test_mcx_trade_plan_futures_compatibility():
    signal = RawSignal(
        symbol="SILVERMIC",
        direction=Direction.BUY,
        confidence=0.85,
        votes=2,
        strategies_fired=["S1_ORB", "S17_SMC"],
        spot_ltp=76500.0,
        regime=Regime.TRENDING,
        timestamp=datetime.now(IST),
    )

    plan = TradePlan(
        signal=signal,
        contract_symbol="SILVERMIC26NOVFUT",
        lot_size=1,
        quantity=1,
        desired_lots=1,
        entry_price=76500.0,
        sl_price=76100.0,
        target_price=77500.0,
        tick_size=1.0,
        tick_value=1.0,
        total_invested=11475.0,  # ~15% margin
        ml_confidence=0.72,
        ml_rank_score=0.80,
    )

    d = plan.to_dict()
    assert d["symbol"] == "SILVERMIC"
    assert d["contract_symbol"] == "SILVERMIC26NOVFUT"
    assert d["tick_size"] == 1.0
    assert d["tick_value"] == 1.0
    assert d["entry_price"] == 76500.0
    assert d["sl_price"] == 76100.0
    assert d["target_price"] == 77500.0
    assert d["quantity"] == 1
    assert d["is_long"] is True


def test_mcx_upstox_broker_market_hours():
    broker = UpstoxBroker()
    # MCX evening session is active at 20:00 IST on a normal trading day
    day_dt = datetime(2026, 6, 2, 20, 0, tzinfo=IST)
    assert broker.is_market_open(day_dt) is True

    # After close (23:45 IST)
    night_dt = datetime(2026, 6, 2, 23, 45, tzinfo=IST)
    assert broker.is_market_open(night_dt) is False

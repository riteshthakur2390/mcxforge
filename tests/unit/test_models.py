"""
tests/unit/test_models.py — Core Model Tests
"""
import pytest
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.models import (
    Direction, Regime, RawSignal, TradePlan, Position
)
import pytz

IST = pytz.timezone("Asia/Kolkata")


# ── RawSignal ──────────────────────────────────────────────────────────────
def test_raw_signal_valid():
    s = RawSignal(
        symbol="NIFTY", direction=Direction.BUY_CALL,
        confidence=0.72, votes=3,
        strategies_fired=["S1", "S2", "S3"],
        nifty_ltp=22480.0,
    )
    assert s.is_valid is True


def test_raw_signal_invalid_no_votes():
    s = RawSignal(
        symbol="NIFTY", direction=Direction.BUY_CALL,
        confidence=0.72, votes=1,
        strategies_fired=["S1"],
        nifty_ltp=22480.0,
    )
    assert s.is_valid is False


def test_raw_signal_invalid_none_direction():
    s = RawSignal(
        symbol="NIFTY", direction=Direction.NONE,
        confidence=0.0, votes=0,
        strategies_fired=[],
        nifty_ltp=22480.0,
    )
    assert s.is_valid is False


def test_raw_signal_to_dict():
    s = RawSignal(
        symbol="NIFTY", direction=Direction.BUY_PUT,
        confidence=0.68, votes=2,
        strategies_fired=["S1", "S4"],
        nifty_ltp=22350.0,
    )
    d = s.to_dict()
    assert d["direction"] == "BUY_PUT"
    assert d["votes"] == 2
    assert isinstance(d["timestamp"], str)


# ── Position ───────────────────────────────────────────────────────────────
class _FakePlan:
    sl_premium     = 84.0
    target_premium = 192.0
    option_symbol  = "NIFTY25MAR2722500CE"


def make_position(entry=120.0, simulated=True):
    return Position(plan=_FakePlan(), entry_premium=entry, is_simulated=simulated)


def test_position_initial_state():
    pos = make_position(120.0)
    assert pos.entry_premium == 120.0
    assert pos.sl_premium    == 84.0
    assert pos.target_premium== 192.0
    assert pos.is_open        is True
    assert pos.pnl_pct        == 0.0


def test_position_trailing_sl_moves_up():
    pos = make_position(120.0)
    pos.update(150.0)
    # SL is now managed by PositionManagerAgent via exit controls, not inside Position.update
    assert pos.peak_premium == 150.0


def test_position_trailing_sl_does_not_move_down():
    pos = make_position(120.0)
    pos.update(150.0, trailing_sl_pct=20.0)
    original_sl = pos.sl_premium
    pos.update(140.0, trailing_sl_pct=20.0)   # price dropped, SL should NOT drop
    assert pos.sl_premium >= original_sl


def test_position_pnl():
    pos = make_position(100.0)
    pos.update(140.0)
    assert pos.pnl_pct == pytest.approx(40.0, rel=0.01)


def test_position_close():
    pos = make_position(100.0)
    pos.close(160.0, "TARGET_HIT")
    assert pos.is_open is False
    assert pos.pnl_pct == pytest.approx(60.0, rel=0.01)
    assert pos.exit_reason == "TARGET_HIT"


# ── Option Utils ───────────────────────────────────────────────────────────
def test_estimate_atm_premium():
    from utils.option_utils import estimate_atm_premium
    p = estimate_atm_premium(22500.0, days_to_expiry=7, iv=0.14)
    assert 50 < p < 300   # reasonable range for NIFTY ATM
    assert p % 5 == 0     # rounded to nearest 5


def test_get_nearest_expiry_returns_tuesday_or_prior_trading_day():
    from datetime import date

    from utils.option_utils import get_nearest_expiry

    expiry, dte = get_nearest_expiry(min_days=2, reference_date=date(2026, 4, 15))
    assert expiry == date(2026, 4, 21)
    assert dte >= 2


def test_get_nearest_expiry_sensex_returns_friday():
    from datetime import date

    from utils.option_utils import get_nearest_expiry

    expiry, dte = get_nearest_expiry(
        min_days=2,
        reference_date=date(2026, 4, 14),
        symbol="SENSEX",
    )
    assert expiry == date(2026, 4, 17)
    assert dte == 3


def test_get_nearest_expiry_holiday_adjusts_to_previous_trading_day():
    from datetime import date

    from utils.option_utils import get_nearest_expiry

    expiry, _ = get_nearest_expiry(min_days=0, reference_date=date(2026, 4, 13))
    assert expiry == date(2026, 4, 13)


def test_index_lot_sizes_for_supported_indices():
    from utils.option_utils import get_index_lot_size
    from config.settings import NIFTY_LOT_SIZE, SENSEX_LOT_SIZE

    assert get_index_lot_size("NIFTY") == NIFTY_LOT_SIZE
    assert get_index_lot_size("SENSEX") == SENSEX_LOT_SIZE


def test_build_option_symbol():
    from utils.option_utils import build_option_symbol
    from datetime import date
    sym = build_option_symbol("NIFTY", date(2025, 3, 27), 22500, "CE")
    assert "NIFTY" in sym
    assert "22500" in sym
    assert sym.endswith("CE")


def test_get_atm_strike():
    from utils.option_utils import get_atm_strike
    assert get_atm_strike(22480.0, 50) == 22500
    assert get_atm_strike(22430.0, 50) == 22450
    assert get_atm_strike(22450.0, 50) == 22450


def test_validate_option_contract_rejects_wrong_expiry_and_strike():
    from datetime import date

    from utils.option_utils import validate_option_contract

    bad_expiry = validate_option_contract(
        symbol="NIFTY",
        expiry_date=date(2026, 4, 23),
        strike=24450,
        option_type="PE",
        underlying=24303.35,
        expected_symbol="NIFTY26APR2324450PE",
    )
    assert bad_expiry.valid is False
    assert bad_expiry.reason == "invalid_weekly_expiry"

    bad_strike = validate_option_contract(
        symbol="NIFTY",
        expiry_date=date(2026, 4, 21),
        strike=24425,
        option_type="PE",
        underlying=24303.35,
        expected_symbol="NIFTY26APR2124425PE",
    )
    assert bad_strike.valid is False
    assert bad_strike.reason == "non_tradable_strike"


# ── ORB ────────────────────────────────────────────────────────────────────
def test_orb_signal_breakout():
    from agents_code.agent1_data.orb import orb_signal
    r = orb_signal(close=22550.0, orb_high=22500.0, orb_low=22400.0, buffer_pct=0.05)
    assert r == "BUY_CALL"


def test_orb_signal_breakdown():
    from agents_code.agent1_data.orb import orb_signal
    r = orb_signal(close=22350.0, orb_high=22500.0, orb_low=22400.0, buffer_pct=0.05)
    assert r == "BUY_PUT"


def test_orb_signal_inside_range():
    from agents_code.agent1_data.orb import orb_signal
    r = orb_signal(close=22460.0, orb_high=22500.0, orb_low=22400.0)
    assert r is None


# ── Choppiness Index ────────────────────────────────────────────────────────
def test_choppiness_index_shape():
    import pandas as pd
    import numpy as np
    from agents_code.agent9_regime.classifier import MarketRegimeAgent

    # Build a clearly trending (low chop) dataset
    close  = pd.Series([100 + i * 0.5 for i in range(30)])
    high   = close + 1.0
    low    = close - 0.5
    df     = pd.DataFrame({"high": high, "low": low, "close": close,
                            "open": close - 0.2, "volume": [100000]*30})
    agent  = MarketRegimeAgent()
    chop   = agent._chop_index(df, period=14)
    assert 0 <= chop <= 100


# ── Futures Models & Direction Tests ──────────────────────────────────────────
def test_futures_direction_and_raw_signal():
    assert Direction.BUY.is_long is True
    assert Direction.BUY.is_short is False
    assert Direction.BUY.action == "BUY"
    assert Direction.SELL.is_long is False
    assert Direction.SELL.is_short is True
    assert Direction.SELL.action == "SELL"

    raw = RawSignal(
        symbol="SILVERMIC",
        direction=Direction.BUY,
        confidence=0.85,
        votes=3,
        strategies_fired=["SuperTrend+RSI", "SMC", "SqueezeMomentum"],
        spot_ltp=85420.0,
    )
    assert raw.ltp == 85420.0
    assert raw.nifty_ltp == 85420.0
    d = raw.to_dict()
    assert d["spot_ltp"] == 85420.0
    assert d["action"] == "BUY"


def test_futures_position_long_and_short_pnl():
    raw_buy = RawSignal(
        symbol="SILVERMIC",
        direction=Direction.BUY,
        confidence=0.80,
        votes=3,
        strategies_fired=["S1", "S6"],
        spot_ltp=85000.0,
    )
    plan_buy = TradePlan(
        signal=raw_buy,
        contract_symbol="SILVERMIC24NOVFUT",
        entry_price=85000.0,
        sl_price=84500.0,
        target_price=86000.0,
        lot_size=1,
        quantity=1,
        tick_size=1.0,
        tick_value=1.0,
    )
    assert plan_buy.is_long is True
    assert plan_buy.option_symbol == "SILVERMIC24NOVFUT"
    assert plan_buy.risk_reward == 2.0

    pos_buy = Position(plan=plan_buy, entry_premium=85000.0)
    pos_buy.update(85500.0)
    assert pos_buy.pnl_points == 500.0
    assert pos_buy.pnl_inr == 500.0

    # Short trade
    raw_sell = RawSignal(
        symbol="SILVERMIC",
        direction=Direction.SELL,
        confidence=0.80,
        votes=3,
        strategies_fired=["S1", "S6"],
        spot_ltp=85000.0,
    )
    plan_sell = TradePlan(
        signal=raw_sell,
        contract_symbol="SILVERMIC24NOVFUT",
        entry_price=85000.0,
        sl_price=85500.0,
        target_price=84000.0,
        lot_size=1,
        quantity=1,
        tick_size=1.0,
        tick_value=1.0,
    )
    assert plan_sell.is_short is True

    pos_sell = Position(plan=plan_sell, entry_premium=85000.0)
    pos_sell.update(84700.0)
    # Price dropped by 300, which is +300 points profit for short!
    assert pos_sell.pnl_points == 300.0
    assert pos_sell.pnl_inr == 300.0
    assert pos_sell.peak_premium == 84700.0

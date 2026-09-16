import pytest
from datetime import datetime
import pytz
from core.models import Position, TradePlan, RawSignal, Direction, Regime
from agents_code.agent6_position.manager import PositionManagerAgent

IST = pytz.timezone("Asia/Kolkata")

def _make_dummy_position(entry_premium=100.0, sl_premium=75.0, target_premium=140.0):
    sig = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_CALL,
        confidence=0.85,
        votes=4,
        strategies_fired=["SuperTrend+RSI", "VWAP+EMA", "CPR", "Ichimoku"],
        nifty_ltp=24500.0,
        timestamp=datetime(2026, 8, 27, 9, 30, tzinfo=IST),
        regime=Regime.TRENDING,
    )
    plan = TradePlan(
        signal=sig,
        option_symbol="NIFTY26SEP0124500CE",
        option_type="CE",
        strike=24500,
        expiry_date="2026-09-01",
        days_to_expiry=4,
        est_premium=entry_premium,
        sl_premium=sl_premium,
        target_premium=target_premium,
        quantity=65,
        lot_size=65,
    )
    pos = Position(plan=plan, entry_premium=entry_premium, entry_time=datetime(2026, 8, 27, 9, 30, tzinfo=IST))
    return pos

@pytest.mark.asyncio
async def test_momentum_breakeven_lock_at_5pct():
    """Verify that when peak PnL reaches >= 5.0%, SL is moved to Breakeven."""
    manager = PositionManagerAgent()
    pos = _make_dummy_position(entry_premium=100.0, sl_premium=75.0)
    manager._pos = pos
    manager._entry_nifty_ltp = 24500.0
    manager._quantity = 65
    manager._initial_quantity = 65
    manager.data_agent = type("DummyDataAgent", (), {"get_option_ltp": lambda *args, **kwargs: 106.0})()

    # Premium surges to 106.0 (+6.0% gain)
    ts = datetime(2026, 8, 27, 9, 35, tzinfo=IST)
    msg = type("Message", (), {
        "topic": "CANDLES_READY",
        "payload": {
            "timestamp": ts.isoformat(),
            "ltp": 24530.0,
            "candles": [{"open": 24500.0, "high": 24535.0, "low": 24495.0, "close": 24530.0}],
            "option_ltp": 106.0,
        }
    })()

    manager._intrabar_option_ltp = lambda spot: 106.0

    await manager.on_candle(msg)

    assert manager._peak_pnl >= 5.0
    assert manager._tsl_active is True
    # Breakeven floor: 100 * 1.002 = 100.2
    assert manager._pos.sl_premium >= 100.2


@pytest.mark.asyncio
async def test_tiered_profit_protection_at_12pct_and_25pct():
    """Verify tiered profit protection locks +3% at 12% peak and +12% at 25% peak."""
    manager = PositionManagerAgent()
    pos = _make_dummy_position(entry_premium=100.0, sl_premium=75.0)
    manager._pos = pos
    manager._entry_nifty_ltp = 24500.0
    manager._quantity = 65

    # Surge to +12.2%
    ts1 = datetime(2026, 8, 27, 9, 40, tzinfo=IST)
    msg1 = type("Message", (), {
        "topic": "CANDLES_READY",
        "payload": {
            "timestamp": ts1.isoformat(),
            "ltp": 24560.0,
            "candles": [{"open": 24530.0, "high": 24565.0, "low": 24525.0, "close": 24560.0}],
            "option_ltp": 115.0,
        }
    })()

    await manager.on_candle(msg1)
    # At 12% peak, floor should be at least entry * 1.03 = 103.0
    assert manager._pos.sl_premium >= 103.0

    # Surge to +30.6% (ltp=24650)
    ts2 = datetime(2026, 8, 27, 9, 45, tzinfo=IST)
    msg2 = type("Message", (), {
        "topic": "CANDLES_READY",
        "payload": {
            "timestamp": ts2.isoformat(),
            "ltp": 24650.0,
            "candles": [{"open": 24600.0, "high": 24655.0, "low": 24595.0, "close": 24650.0}],
            "option_ltp": 130.0,
        }
    })()

    await manager.on_candle(msg2)
    # At >25% peak, floor should be at least entry * 1.12 = 112.0
    assert manager._pos.sl_premium >= 112.0

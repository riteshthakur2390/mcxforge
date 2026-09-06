import sys
from datetime import datetime
from pathlib import Path
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents_code.agent6_position.manager import PositionManagerAgent
from core.models import Direction, Position, RawSignal, TradePlan


IST = pytz.timezone("Asia/Kolkata")


class FakeDataAgent:
    def __init__(self, close: float) -> None:
        self.close = close
        self.calls = []

    def get_option_candle_close(self, option_symbol: str, timestamp, interval: str = "5minute") -> float:
        self.calls.append((option_symbol, timestamp, interval))
        return self.close


def _position(*, simulated: bool, execution_mode: str = "DRY_RUN") -> Position:
    signal = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_CALL,
        confidence=0.9,
        votes=3,
        strategies_fired=["SuperTrend+RSI"],
        nifty_ltp=24195.65,
    )
    plan = TradePlan(
        signal=signal,
        option_symbol="NIFTY26JUL1424200CE",
        strike=24200,
        option_type="CE",
        expiry_date="2026-07-14",
        days_to_expiry=4,
        est_premium=116.8,
        sl_premium=88.57,
        target_premium=158.25,
        lot_size=65,
        quantity=65,
    )
    return Position(
        plan=plan,
        entry_premium=116.8,
        is_simulated=simulated,
        execution_mode=execution_mode,
        entry_time=IST.localize(datetime(2026, 7, 10, 9, 40)),
    )


def test_simulated_exit_prefers_option_candle_close():
    agent = PositionManagerAgent.__new__(PositionManagerAgent)
    agent._pos = _position(simulated=True)
    agent.data_agent = FakeDataAgent(close=108.75)
    ts = IST.localize(datetime(2026, 7, 10, 10, 35))

    exit_price = agent._paper_exit_premium("NIFTY26JUL1424200CE", 116.3, ts)

    assert exit_price == 108.75
    assert agent.data_agent.calls == [("NIFTY26JUL1424200CE", ts, "1minute")]


def test_real_exit_keeps_broker_price():
    agent = PositionManagerAgent.__new__(PositionManagerAgent)
    agent._pos = _position(simulated=False)
    agent.data_agent = FakeDataAgent(close=108.75)

    exit_price = agent._paper_exit_premium(
        "NIFTY26JUL1424200CE",
        116.3,
        IST.localize(datetime(2026, 7, 10, 10, 35)),
    )

    assert exit_price == 116.3
    assert agent.data_agent.calls == []


def test_missing_option_candle_close_keeps_quote_price():
    agent = PositionManagerAgent.__new__(PositionManagerAgent)
    agent._pos = _position(simulated=True)
    agent.data_agent = FakeDataAgent(close=0.0)

    exit_price = agent._paper_exit_premium(
        "NIFTY26JUL1424200CE",
        116.3,
        IST.localize(datetime(2026, 7, 10, 10, 35)),
    )

    assert exit_price == 116.3

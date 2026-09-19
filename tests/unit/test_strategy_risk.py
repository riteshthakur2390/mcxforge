from datetime import datetime
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents_code.agent2_strategy.runner import StrategyAgent
from agents_code.agent10_risk.guard import RiskGuardAgent
from core.bus import get_bus, reset_bus, Topic
from core.models import RawSignal, Direction
from config.settings import PAPER_TRADING_CAPITAL, MAX_DAILY_LOSS_PCT


@pytest.fixture(autouse=True)
def clean_bus():
    reset_bus()
    yield
    reset_bus()


def test_strategy_duplicate_signal_key_suppressed():
    agent = StrategyAgent()
    signal = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_CALL,
        confidence=0.72,
        votes=3,
        strategies_fired=["S1", "S2"],
        nifty_ltp=22480.0,
    )

    assert agent._is_duplicate_signal(signal) is False
    agent._remember_signal(signal)
    assert agent._is_duplicate_signal(signal) is True


def test_strategy_signal_key_changes_with_direction():
    agent = StrategyAgent()
    call_signal = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_CALL,
        confidence=0.72,
        votes=3,
        strategies_fired=["S1", "S2"],
        nifty_ltp=22480.0,
    )
    put_signal = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_PUT,
        confidence=0.72,
        votes=3,
        strategies_fired=["S1", "S2"],
        nifty_ltp=22480.0,
    )

    assert agent._signal_key(call_signal) != agent._signal_key(put_signal)


def test_risk_guard_halts_after_daily_loss_breach():
    statuses = []

    async def run():
        bus = get_bus()
        agent = RiskGuardAgent()
        agent.register()

        async def capture(msg):
            statuses.append(msg.payload)

        bus.subscribe(Topic.SYSTEM_STATUS, capture)
        loss_amount = float(PAPER_TRADING_CAPITAL * (MAX_DAILY_LOSS_PCT + 5) / 100.0)
        await bus.publish(Topic.POSITION_CLOSED, {
            "realized_pnl": -loss_amount,
            "timestamp": datetime.now().isoformat(),
        }, "test")

    asyncio.run(run())

    assert any(status.get("status") == "HALTED" for status in statuses)
    assert any(status.get("trading_enabled") is False for status in statuses)

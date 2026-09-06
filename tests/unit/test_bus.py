"""
tests/unit/test_bus.py — Message Bus Unit Tests
"""
import asyncio
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.bus import MessageBus, reset_bus, get_bus, Topic


@pytest.fixture(autouse=True)
def clean_bus():
    reset_bus()
    yield
    reset_bus()


# ── Basic pub/sub ──────────────────────────────────────────────────────────
def test_subscribe_and_publish():
    received = []
    bus = MessageBus()

    async def handler(msg):
        received.append(msg.payload)

    bus.subscribe(Topic.CANDLES_READY, handler)

    async def run():
        await bus.publish(Topic.CANDLES_READY, {"ltp": 22480.5}, "test")

    asyncio.run(run())
    assert len(received) == 1
    assert received[0]["ltp"] == 22480.5


def test_multiple_subscribers():
    results = []
    bus = MessageBus()

    async def h1(msg): results.append("h1")
    async def h2(msg): results.append("h2")

    bus.subscribe(Topic.RAW_SIGNAL, h1)
    bus.subscribe(Topic.RAW_SIGNAL, h2)

    async def run():
        await bus.publish(Topic.RAW_SIGNAL, {}, "test")

    asyncio.run(run())
    assert set(results) == {"h1", "h2"}


def test_handler_error_does_not_crash_bus():
    """A failing handler should not stop other handlers."""
    results = []
    bus = MessageBus()

    async def bad_handler(msg): raise RuntimeError("intentional")
    async def good_handler(msg): results.append("ok")

    bus.subscribe(Topic.ALERT, bad_handler)
    bus.subscribe(Topic.ALERT, good_handler)

    async def run():
        await bus.publish(Topic.ALERT, {"text": "test"}, "test")

    asyncio.run(run())
    assert results == ["ok"]


def test_history_limit():
    bus = MessageBus()
    bus._max_history = 5

    async def handler(msg): pass
    bus.subscribe(Topic.SYSTEM_STATUS, handler)

    async def run():
        for i in range(10):
            await bus.publish(Topic.SYSTEM_STATUS, {"i": i}, "test")

    asyncio.run(run())
    assert len(bus._history) == 5


def test_stats_counting():
    bus = MessageBus()

    async def handler(msg): pass
    bus.subscribe(Topic.CANDLES_READY, handler)

    async def run():
        for _ in range(3):
            await bus.publish(Topic.CANDLES_READY, {}, "test")

    asyncio.run(run())
    stats = bus.get_stats()
    assert stats["by_topic"][Topic.CANDLES_READY] == 3


def test_get_history_filtered():
    bus = MessageBus()

    async def h(msg): pass
    bus.subscribe(Topic.CANDLES_READY, h)
    bus.subscribe(Topic.ALERT, h)

    async def run():
        await bus.publish(Topic.CANDLES_READY, {"a": 1}, "test")
        await bus.publish(Topic.ALERT, {"b": 2}, "test")
        await bus.publish(Topic.CANDLES_READY, {"c": 3}, "test")

    asyncio.run(run())
    history = bus.get_history(Topic.CANDLES_READY)
    assert len(history) == 2
    assert all(m["topic"] == Topic.CANDLES_READY for m in history)

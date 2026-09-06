import asyncio
import os
import sys
from datetime import datetime as real_datetime

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents_code.agent1_data.fetcher import DataFetcherAgent
from agents_code.agent1_data import fetcher as fetcher_module
from core.bus import Topic, get_bus, reset_bus


@pytest.fixture(autouse=True)
def clean_bus():
    reset_bus()
    yield
    reset_bus()


def _session_df() -> pd.DataFrame:
    day1 = pd.date_range("2026-04-08 15:15", periods=3, freq="5min", tz="Asia/Kolkata")
    day2 = pd.date_range("2026-04-09 09:15", periods=2, freq="5min", tz="Asia/Kolkata")
    index = day1.append(day2)
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 101.2, 101.3],
            "high": [101.0, 102.0, 103.0, 101.6, 101.7],
            "low": [99.5, 100.5, 101.5, 100.9, 101.0],
            "close": [100.5, 101.5, 102.5, 101.4, 101.5],
            "volume": [1000, 1000, 1000, 1200, 1200],
        },
        index=index,
    )


def _daily_df() -> pd.DataFrame:
    index = pd.to_datetime(
        ["2026-04-08 00:00:00+05:30", "2026-04-09 00:00:00+05:30"]
    )
    return pd.DataFrame(
        {
            "open": [23997.35, 23880.55],
            "high": [24025.15, 23990.75],
            "low": [23828.50, 23682.80],
            "close": [23997.35, 23775.10],
            "volume": [0, 0],
        },
        index=index,
    )


def test_resolve_premarket_uses_session_open_and_prev_close():
    class FakeDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            dt = real_datetime(2026, 4, 9, 9, 20)
            return tz.localize(dt) if tz else dt

    agent = DataFetcherAgent()
    agent._fetch_candles = lambda n=200: _session_df()
    agent._fetch_daily_candles = lambda days=30: None
    agent._get_prev_close_from_cache = lambda: 999.0
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fetcher_module, "datetime", FakeDateTime)

    values = agent._resolve_premarket_session_values()
    monkeypatch.undo()

    assert values["prev_close"] == pytest.approx(102.5)
    assert values["today_open"] == pytest.approx(101.2)


def test_publish_premarket_bias_uses_open_not_ltp():
    events = []

    async def run():
        bus = get_bus()
        agent = DataFetcherAgent()
        agent.broker.get_india_vix = lambda: 20.4
        agent.broker.get_ltp = lambda symbol: 105.35  # intentionally misleading
        agent._resolve_premarket_session_values = lambda: {
            "prev_close": 102.5,
            "today_open": 101.2,
        }

        async def capture(msg):
            events.append(msg.payload)

        bus.subscribe(Topic.PREMARKET_BIAS, capture)
        await agent._publish_premarket_bias()

    asyncio.run(run())

    assert len(events) == 1
    payload = events[0]
    assert payload["prev_close"] == pytest.approx(102.5)
    assert payload["today_open"] == pytest.approx(101.2)
    assert payload["gap_pct"] == pytest.approx(round((101.2 - 102.5) / 102.5 * 100, 3))
    assert payload["bias"] == "BEARISH"
    assert payload["gap_ready"] is True


def test_gap_bias_thresholds_are_neutral_inside_band():
    assert DataFetcherAgent._classify_gap_bias(0.14) == "NEUTRAL"
    assert DataFetcherAgent._classify_gap_bias(-0.14) == "NEUTRAL"
    assert DataFetcherAgent._classify_gap_bias(0.16) == "BULLISH"
    assert DataFetcherAgent._classify_gap_bias(-0.16) == "BEARISH"


def test_resolve_premarket_does_not_treat_latest_available_day_as_today(monkeypatch):
    class FakeDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            dt = real_datetime(2026, 4, 10, 8, 55)
            return tz.localize(dt) if tz else dt

    agent = DataFetcherAgent()
    agent._fetch_candles = lambda n=200: _session_df()
    agent._fetch_daily_candles = lambda days=30: _daily_df()
    agent._get_prev_close_from_cache = lambda as_of_date=None: None
    monkeypatch.setattr(fetcher_module, "datetime", FakeDateTime)

    values = agent._resolve_premarket_session_values()

    assert values["today_open"] is None
    assert values["prev_close"] == pytest.approx(23775.10)
    assert values["prev_close_source"] == f"{agent.broker.broker_name}:day_prior_close"
    assert values["today_open_source"] == ""


def test_prev_close_cache_uses_latest_completed_session(monkeypatch, tmp_path):
    class FakeDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            dt = real_datetime(2026, 4, 10, 8, 55)
            return tz.localize(dt) if tz else dt

    cache_df = _daily_df()
    cache_path = tmp_path / "NIFTY_day_yfinance.parquet"
    cache_df.to_parquet(cache_path)

    agent = DataFetcherAgent()
    monkeypatch.setattr(fetcher_module, "DATA_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(fetcher_module, "datetime", FakeDateTime)

    assert agent._get_prev_close_from_cache() == pytest.approx(23775.10)

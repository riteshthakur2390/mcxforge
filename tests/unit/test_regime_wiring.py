import asyncio
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents_code.agent3_ml.filter import MLFilterAgent
from agents_code.agent9_regime.classifier import MarketRegimeAgent
from agents_code.agent9_regime.regime_detector import RegimeResult
from core.bus import Message, Topic, get_bus, reset_bus


@pytest.fixture(autouse=True)
def clean_bus():
    reset_bus()
    yield
    reset_bus()


def _candles(n: int = 10) -> list[dict]:
    idx = pd.date_range("2026-04-09 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    return [
        {
            "datetime": ts.isoformat(),
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 1000 + i,
        }
        for i, ts in enumerate(idx)
    ]


def test_regime_agent_uses_detailed_detector_for_top_level_gate():
    published = []

    async def run():
        bus = get_bus()
        agent = MarketRegimeAgent()
        agent.register()

        async def capture(msg):
            published.append((msg.topic, msg.payload))

        bus.subscribe(Topic.SIGNAL_SUPPRESSED, capture)
        bus.subscribe(Topic.MARKET_REGIME, capture)

        agent._regime_detector.compute = lambda df, india_vix: RegimeResult(
            label="HIGH_VOLATILITY",
            confidence=0.87,
            sub_label="ATR_SPIKE",
            adx=14.0,
            atr_pct=1.2,
            atr_ratio=1.8,
            chop=55.0,
            di_spread=2.0,
            is_trending=False,
            is_ranging=False,
            is_high_vol=True,
        )
        agent._classify = lambda df: (agent._last_regime, {"adx": 12.0, "chop_index": 55.0})

        await bus.publish(
            Topic.CANDLES_READY,
            {"candles": _candles(12), "timestamp": "2026-04-09T10:15:00+05:30"},
            "test",
        )

    asyncio.run(run())

    assert len(published) == 2
    topics = [topic for topic, _payload in published]
    assert Topic.MARKET_REGIME in topics
    assert Topic.SIGNAL_SUPPRESSED in topics
    market_payload = next(payload for topic, payload in published if topic == Topic.MARKET_REGIME)
    suppressed_payload = next(payload for topic, payload in published if topic == Topic.SIGNAL_SUPPRESSED)
    assert market_payload["regime"] == "HIGH_VOL"
    assert market_payload["signal_permitted"] is False
    assert market_payload["regime_details"]["detailed_regime"]["label"] == "HIGH_VOLATILITY"
    assert "market_structure" in market_payload["regime_details"]
    assert suppressed_payload["details"]["regime_confidence"] == pytest.approx(0.87)


def test_ml_agent_updates_regime_info_from_suppressed_payload():
    agent = MLFilterAgent(enable_file_watcher=False)
    payload = {
        "details": {
            "detailed_regime": {
                "label": "RANGING",
                "confidence": 0.74,
                "atr_ratio": 0.66,
            }
        }
    }

    asyncio.run(
        agent.on_regime_update(
            Message(topic=Topic.SIGNAL_SUPPRESSED, payload=payload, source="test")
        )
    )

    assert agent._latest_regime_info["label"] == "RANGING"
    assert agent._latest_regime_info["confidence"] == pytest.approx(0.74)
    assert agent._latest_regime_info["atr_ratio"] == pytest.approx(0.66)

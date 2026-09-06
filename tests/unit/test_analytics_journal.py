import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents_code.agent7_analytics.journal import AnalyticsAgent
from core.bus import Message, Topic


def test_suppressed_stats_deduplicate_repeated_runtime_events():
    agent = AnalyticsAgent(backtest_mode=True)
    payload = {
        "timestamp": "2026-04-27T11:25:00+05:30",
        "regime": "RANGING",
        "reason": "detailed=CHOPPY | conf=0.51 | chop=45.7",
    }

    asyncio.run(agent._on_suppressed(Message(topic=Topic.SIGNAL_SUPPRESSED, payload=payload, source="test")))
    asyncio.run(agent._on_suppressed(Message(topic=Topic.SIGNAL_SUPPRESSED, payload=payload, source="test")))

    stats = agent.get_daily_stats()
    assert stats["suppressed"] == 1
    assert len(agent._journal) == 1
    assert agent._journal[0]["suppression_reason"] == payload["reason"]

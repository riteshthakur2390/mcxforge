from datetime import date, datetime
from pathlib import Path

import pytest
import pytz

from core.bus import Topic
from agents_code.agent1_data.fetcher import DataFetcherAgent


IST = pytz.timezone("Asia/Kolkata")


def test_orb_state_persists_and_loads_from_disk(tmp_path: Path):
    agent = DataFetcherAgent.__new__(DataFetcherAgent)
    agent._orb_dir = tmp_path
    agent._session_date = date(2026, 4, 27)
    agent._orb_high = None
    agent._orb_low = None

    payload = {
        "orb_high": 24210.5,
        "orb_low": 24162.0,
        "orb_range": 48.5,
        "session_date": "2026-04-27",
        "formed_at": "2026-04-27T09:30:00+05:30",
    }

    agent._persist_orb_state(payload)
    loaded = agent.get_orb_state()

    assert loaded is not None
    assert loaded["orb_high"] == 24210.5
    assert loaded["orb_low"] == 24162.0
    assert loaded["orb_range"] == 48.5


@pytest.mark.asyncio
async def test_restore_orb_state_republishes_after_restart(tmp_path: Path):
    published: list[tuple[object, dict, str]] = []

    class StubBus:
        async def publish(self, topic, payload, source):
            published.append((topic, payload, source))

    agent = DataFetcherAgent.__new__(DataFetcherAgent)
    agent.bus = StubBus()
    agent._orb_dir = tmp_path
    agent._session_date = date(2026, 4, 28)
    agent._orb_high = None
    agent._orb_low = None
    agent._orb_published = False

    agent._persist_orb_state({
        "orb_high": 24133.65,
        "orb_low": 23999.25,
        "orb_range": 134.4,
        "session_date": "2026-04-28",
        "formed_at": "2026-04-28T09:28:00+05:30",
    })

    restored = await agent._restore_orb_state_if_available(
        IST.localize(datetime(2026, 4, 28, 11, 33))
    )

    assert restored is True
    assert agent._orb_high == 24133.65
    assert agent._orb_low == 23999.25
    assert agent._orb_published is True
    assert published == [(
        Topic.ORB_FORMED,
        {
            "orb_high": 24133.65,
            "orb_low": 23999.25,
            "orb_range": 134.4,
            "session_date": "2026-04-28",
            "formed_at": "2026-04-28T09:28:00+05:30",
        },
        DataFetcherAgent.NAME,
    )]

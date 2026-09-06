import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import utils.llm as llm_utils
from core.llm_router import TaskType


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch):
    llm_utils._LLM_CACHE.clear()
    monkeypatch.setattr(llm_utils, "TRADING_MODE", "OBSERVE")
    monkeypatch.setattr(llm_utils, "LLM_ENABLED", True)
    monkeypatch.setattr(llm_utils, "LLM_CONTEXT_MIN_RANK", 0.78)
    monkeypatch.setattr(llm_utils, "LLM_CONTEXT_MIN_PROB", 0.62)
    yield
    llm_utils._LLM_CACHE.clear()


def test_llm_context_skips_low_confidence(monkeypatch):
    called = {"count": 0}

    async def fake_call(*args, **kwargs):
        called["count"] += 1
        return "should not be used"

    monkeypatch.setattr(llm_utils, "call_llm_async", fake_call)
    text = asyncio.run(
        llm_utils.call_llm_context_async(
            "test",
            cache_key="k1",
            task_type=TaskType.GENERAL,
            rank_score=0.6,
            success_prob=0.7,
        )
    )

    assert text == ""
    assert called["count"] == 0


def test_llm_context_uses_cache(monkeypatch):
    called = {"count": 0}

    async def fake_call(*args, **kwargs):
        called["count"] += 1
        return "cached text"

    monkeypatch.setattr(llm_utils, "call_llm_async", fake_call)

    first = asyncio.run(
        llm_utils.call_llm_context_async(
            "test",
            cache_key="k2",
            task_type=TaskType.GENERAL,
            rank_score=0.82,
            success_prob=0.7,
        )
    )
    second = asyncio.run(
        llm_utils.call_llm_context_async(
            "test",
            cache_key="k2",
            task_type=TaskType.GENERAL,
            rank_score=0.82,
            success_prob=0.7,
        )
    )

    assert first == "cached text"
    assert second == "cached text"
    assert called["count"] == 1

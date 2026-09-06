import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents_code.agent3_ml.filter import MLFilterAgent, _CORRELATED_PAIRS
from core.bus import Message, Topic, get_bus, reset_bus


@pytest.fixture(autouse=True)
def clean_bus():
    reset_bus()
    yield
    reset_bus()


def test_ml_filter_ranks_signal_without_rejecting():
    events = []

    async def run():
        bus = get_bus()
        agent = MLFilterAgent(enable_file_watcher=False)
        agent._score = lambda conf, votes, direction: (0.42, "ML")
        agent._latest_regime_info = {
            "label": "TRENDING",
            "confidence": 0.76,
            "atr_ratio": 1.18,
        }

        async def capture(msg):
            events.append((msg.topic, msg.payload))

        bus.subscribe(Topic.SIGNAL_APPROVED, capture)
        bus.subscribe(Topic.SIGNAL_REJECTED, capture)

        payload = {
            "direction": "BUY_CALL",
            "confidence": 0.84,
            "votes": 4,
            "timestamp": "2026-04-09T10:15:00+05:30",
            "strategies_fired": ["ADX+PSAR", "Ichimoku", "SuperTrend", "VWAP+EMA"],
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "trend_pullback",
                        "setup_strength": 0.82,
                        "expected_move": 22.0,
                    },
                    "weighted_vote": {
                        "winner_avg_weight": 1.14,
                    }
                }
            },
        }

        await agent.on_raw_signal(
            Message(topic=Topic.RAW_SIGNAL, payload=payload, source="test")
        )

    asyncio.run(run())

    assert len(events) == 1
    topic, data = events[0]
    assert topic == Topic.SIGNAL_APPROVED
    assert data["ml_approved"] is True
    assert data["ml_confidence"] == pytest.approx(0.42)
    assert 0.0 < data["ml_rank_score"] < 1.0
    assert data["ml_rank_tier"] in {"LOW", "MEDIUM", "HIGH"}
    assert "ml_rank_components" in data
    assert data["ml_rank_components"]["setup_type"] == "trend_pullback"
    assert data["ml_rank_components"]["setup_strength"] == pytest.approx(0.82)


def test_ml_filter_rejects_bottom_rank_weak_structure():
    blocked, reason = MLFilterAgent._fails_low_quality_rank_gate(
        data={
            "direction": "BUY_PUT",
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "vote_aligned",
                        "setup_strength": 0.65,
                    },
                    "market_structure": {
                        "entry_validation": {
                            "valid": True,
                            "avoid": False,
                            "middle_range": False,
                            "by_direction": {
                                "BUY_PUT": {"valid": True}
                            },
                        }
                    },
                }
            },
        },
        rank_score=0.51,
        votes=2,
    )

    assert blocked is True
    assert "vote_aligned" in reason


def test_ml_filter_rejects_fallback_when_model_is_trained():
    events = []

    async def run():
        bus = get_bus()
        agent = MLFilterAgent(enable_file_watcher=False)
        agent._score = lambda conf, votes, direction: (0.52, "FALLBACK")
        agent._latest_regime_info = {
            "label": "TRENDING",
            "confidence": 0.76,
            "atr_ratio": 1.18,
        }

        async def capture(msg):
            events.append((msg.topic, msg.payload))

        bus.subscribe(Topic.SIGNAL_APPROVED, capture)
        bus.subscribe(Topic.SIGNAL_REJECTED, capture)

        payload = {
            "direction": "BUY_PUT",
            "confidence": 0.84,
            "votes": 4,
            "timestamp": "2026-04-09T10:15:00+05:30",
            "strategies_fired": ["ADX+PSAR", "BBSqueeze", "SuperTrend", "VWAP+EMA"],
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "vote_aligned",
                        "setup_strength": 0.82,
                        "expected_move": 22.0,
                    },
                    "weighted_vote": {
                        "winner_avg_weight": 1.10,
                    }
                }
            },
        }

        await agent.on_raw_signal(
            Message(topic=Topic.RAW_SIGNAL, payload=payload, source="test")
        )

    asyncio.run(run())

    assert len(events) == 1
    topic, data = events[0]
    assert topic == Topic.SIGNAL_REJECTED
    assert data["ml_approved"] is False
    assert "fallback disabled" in data["rejection_reason"].lower()


def test_ml_filter_keeps_high_consensus_or_strong_setup():
    blocked, reason = MLFilterAgent._fails_low_quality_rank_gate(
        data={
            "direction": "BUY_CALL",
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "breakout",
                        "setup_strength": 0.83,
                    }
                }
            },
        },
        rank_score=0.58,
        votes=3,
    )

    assert blocked is False
    assert reason == ""


def test_ml_rank_penalizes_exact_triple_correlated_call_combo():
    agent = MLFilterAgent(enable_file_watcher=False)
    agent._latest_regime_info = {
        "label": "TRENDING",
        "confidence": 0.75,
    }

    penalized_rank, meta = agent._rank_signal(
        data={
            "timestamp": "2026-03-18T10:50:00+05:30",
            "direction": "BUY_CALL",
            "strategies_fired": ["VWAP+EMA", "ADX+PSAR", "BBSqueeze"],
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "vote_aligned",
                        "setup_strength": 0.81,
                    }
                }
            },
        },
        success_prob=0.20,
        raw_conf=0.78,
        votes=3,
        decision_type="ML",
    )
    neutral_rank, _ = agent._rank_signal(
        data={
            "timestamp": "2026-03-18T10:50:00+05:30",
            "direction": "BUY_CALL",
            "strategies_fired": ["VWAP+EMA", "Ichimoku", "BBSqueeze"],
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "vote_aligned",
                        "setup_strength": 0.81,
                    }
                }
            },
        },
        success_prob=0.20,
        raw_conf=0.78,
        votes=3,
        decision_type="ML",
    )

    assert meta["components"]["loss_pattern_penalty"] == pytest.approx(0.055)
    assert penalized_rank < neutral_rank


def test_ml_rank_penalizes_non_adx_trend_pullback_cluster():
    agent = MLFilterAgent(enable_file_watcher=False)
    agent._latest_regime_info = {
        "label": "TRENDING",
        "confidence": 0.78,
    }

    penalized_rank, meta = agent._rank_signal(
        data={
            "timestamp": "2026-03-23T14:30:00+05:30",
            "direction": "BUY_PUT",
            "strategies_fired": ["ADX+PSAR", "SuperTrend+RSI"],
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "trend_pullback",
                        "setup_strength": 0.80,
                    }
                }
            },
        },
        success_prob=0.36,
        raw_conf=0.74,
        votes=2,
        decision_type="ML",
    )
    control_rank, _ = agent._rank_signal(
        data={
            "timestamp": "2026-03-23T14:30:00+05:30",
            "direction": "BUY_PUT",
            "strategies_fired": ["ADX+PSAR"],
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "trend_pullback",
                        "setup_strength": 0.80,
                    }
                }
            },
        },
        success_prob=0.36,
        raw_conf=0.74,
        votes=2,
        decision_type="ML",
    )

    assert meta["components"]["loss_pattern_penalty"] == pytest.approx(0.09)
    assert penalized_rank < control_rank


def test_vwap_bb_pair_is_classified_as_correlated():
    assert frozenset({"VWAP+EMA", "BBSqueeze"}) in _CORRELATED_PAIRS


def test_ml_rank_penalizes_regime_structure_mismatch():
    agent = MLFilterAgent(enable_file_watcher=False)
    agent._latest_regime_info = {
        "label": "TRENDING",
        "confidence": 0.78,
    }

    mismatch_rank, meta = agent._rank_signal(
        data={
            "timestamp": "2026-03-16T12:45:00+05:30",
            "direction": "BUY_PUT",
            "strategies_fired": ["SuperTrend+RSI", "BBSqueeze"],
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "vote_aligned",
                        "setup_strength": 0.77,
                    },
                    "market_structure": {
                        "structure_state": {
                            "bias": "RANGING",
                        }
                    },
                }
            },
        },
        success_prob=0.41,
        raw_conf=0.76,
        votes=2,
        decision_type="ML",
    )
    aligned_rank, _ = agent._rank_signal(
        data={
            "timestamp": "2026-03-16T12:45:00+05:30",
            "direction": "BUY_PUT",
            "strategies_fired": ["SuperTrend+RSI", "BBSqueeze"],
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "vote_aligned",
                        "setup_strength": 0.77,
                    },
                    "market_structure": {
                        "structure_state": {
                            "bias": "BEARISH",
                        }
                    },
                }
            },
        },
        success_prob=0.41,
        raw_conf=0.76,
        votes=2,
        decision_type="ML",
    )

    assert meta["components"]["regime_structure_penalty"] == pytest.approx(0.035)
    assert mismatch_rank < aligned_rank

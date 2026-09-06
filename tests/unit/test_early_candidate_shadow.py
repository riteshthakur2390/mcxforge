import pytest
from datetime import datetime
import pytz
from core.models import Direction
from agents_code.agent2_strategy.runner import StrategyAgent, WeightedVoteSummary

IST = pytz.timezone("Asia/Kolkata")

def test_early_candidate_inception_and_excursion():
    """Verify early candidate inception on 2+ votes and excursion tracking."""
    agent = StrategyAgent()
    ts1 = datetime(2026, 8, 27, 9, 30, tzinfo=IST)
    call_votes = [
        {"name": "SuperTrend+RSI", "confidence": 0.80, "_vote_weight": 1.0, "_weighted_score": 0.80},
        {"name": "VWAP+EMA", "confidence": 0.75, "_vote_weight": 1.0, "_weighted_score": 0.75},
    ]
    summary = StrategyAgent._summarize_votes(call_votes)

    agent._process_early_candidate_cycle(
        current_ts=ts1,
        ltp=24500.0,
        candle_high=24510.0,
        candle_low=24495.0,
        call_votes=call_votes,
        put_votes=[],
        call_summary=summary,
        put_summary=WeightedVoteSummary(count=0, weight_total=0.0, weighted_score=0.0, avg_weight=0.0),
    )

    cand = agent._active_candidates[Direction.BUY_CALL.value]
    assert cand is not None
    assert cand["lifecycle_status"] == "CANDIDATE"
    assert cand["candidate_price"] == 24500.0
    assert cand["raw_vote_count"] == 2
    assert cand["independent_category_count"] == 1 # Both are trend / momentum

    # Next candle moves favorably to 24525 (high=24530, low=24505)
    ts2 = datetime(2026, 8, 27, 9, 35, tzinfo=IST)
    agent._process_early_candidate_cycle(
        current_ts=ts2,
        ltp=24525.0,
        candle_high=24530.0,
        candle_low=24505.0,
        call_votes=call_votes,
        put_votes=[],
        call_summary=summary,
        put_summary=WeightedVoteSummary(count=0, weight_total=0.0, weighted_score=0.0, avg_weight=0.0),
    )

    cand = agent._active_candidates[Direction.BUY_CALL.value]
    assert cand["bars_active"] == 2
    assert cand["max_favorable_excursion_pts"] == 30.0 # 24530 - 24500
    assert cand["max_adverse_excursion_pts"] == 0.0


def test_early_candidate_confirmation_and_metrics():
    """Verify confirmation lifecycle transition, time saved, and price improvement."""
    agent = StrategyAgent()
    ts1 = datetime(2026, 8, 27, 9, 30, tzinfo=IST)
    call_votes = [
        {"name": "SuperTrend+RSI", "confidence": 0.80, "_vote_weight": 1.0, "_weighted_score": 0.80},
        {"name": "VWAP+EMA", "confidence": 0.75, "_vote_weight": 1.0, "_weighted_score": 0.75},
    ]
    summary = StrategyAgent._summarize_votes(call_votes)

    agent._process_early_candidate_cycle(
        current_ts=ts1,
        ltp=24500.0,
        candle_high=24505.0,
        candle_low=24498.0,
        call_votes=call_votes,
        put_votes=[],
        call_summary=summary,
        put_summary=WeightedVoteSummary(count=0, weight_total=0.0, weighted_score=0.0, avg_weight=0.0),
    )

    # 10 minutes later (9:40), 4 votes confirm at 24535.0
    ts_conf = datetime(2026, 8, 27, 9, 40, tzinfo=IST)
    confirmed = agent._confirm_early_candidate(
        direction=Direction.BUY_CALL,
        signal_id="SIG_20260827_094000",
        confirmed_price=24535.0,
        confirmed_ts=ts_conf,
    )

    assert confirmed is not None
    assert confirmed["lifecycle_status"] == "CONFIRMED"
    assert confirmed["confirmed_signal_id"] == "SIG_20260827_094000"
    assert confirmed["time_saved_sec"] == 600.0 # 10 mins
    assert confirmed["price_improvement_pts"] == 35.0 # Entered 35 pts earlier / cheaper for CALL
    assert agent._active_candidates[Direction.BUY_CALL.value] is None
    assert len(agent._candidate_shadow_history) == 1


def test_early_candidate_expiration():
    """Verify candidate expires after 6 bars without confirmation."""
    agent = StrategyAgent()
    ts = datetime(2026, 8, 27, 9, 30, tzinfo=IST)
    call_votes = [
        {"name": "SuperTrend+RSI", "confidence": 0.80, "_vote_weight": 1.0, "_weighted_score": 0.80},
        {"name": "VWAP+EMA", "confidence": 0.75, "_vote_weight": 1.0, "_weighted_score": 0.75},
    ]
    summary = StrategyAgent._summarize_votes(call_votes)

    # Inception
    agent._process_early_candidate_cycle(
        current_ts=ts,
        ltp=24500.0,
        candle_high=24505.0,
        candle_low=24498.0,
        call_votes=call_votes,
        put_votes=[],
        call_summary=summary,
        put_summary=WeightedVoteSummary(count=0, weight_total=0.0, weighted_score=0.0, avg_weight=0.0),
    )

    # Simulate 5 subsequent bars (total 6 bars) without full confirmation
    for k in range(1, 6):
        ts_next = datetime(2026, 8, 27, 9, 30 + k * 5, tzinfo=IST)
        agent._process_early_candidate_cycle(
            current_ts=ts_next,
            ltp=24502.0,
            candle_high=24508.0,
            candle_low=24495.0,
            call_votes=call_votes,
            put_votes=[],
            call_summary=summary,
            put_summary=WeightedVoteSummary(count=0, weight_total=0.0, weighted_score=0.0, avg_weight=0.0),
        )

    # Should be expired and moved to history
    assert agent._active_candidates[Direction.BUY_CALL.value] is None
    assert len(agent._candidate_shadow_history) == 1
    assert agent._candidate_shadow_history[0]["lifecycle_status"] == "EXPIRED"


def test_early_candidate_rejection():
    """Verify candidate is marked REJECTED when late-entry or other filters block it."""
    agent = StrategyAgent()
    ts = datetime(2026, 8, 27, 9, 30, tzinfo=IST)
    call_votes = [
        {"name": "SuperTrend+RSI", "confidence": 0.80, "_vote_weight": 1.0, "_weighted_score": 0.80},
        {"name": "VWAP+EMA", "confidence": 0.75, "_vote_weight": 1.0, "_weighted_score": 0.75},
    ]
    summary = StrategyAgent._summarize_votes(call_votes)

    agent._process_early_candidate_cycle(
        current_ts=ts,
        ltp=24500.0,
        candle_high=24505.0,
        candle_low=24498.0,
        call_votes=call_votes,
        put_votes=[],
        call_summary=summary,
        put_summary=WeightedVoteSummary(count=0, weight_total=0.0, weighted_score=0.0, avg_weight=0.0),
    )

    rejected = agent._reject_early_candidate(
        direction=Direction.BUY_CALL,
        rejection_reason="late_entry_move_exhausted: EXHAUSTION_RISK",
        current_ts=ts,
    )

    assert rejected is not None
    assert rejected["lifecycle_status"] == "REJECTED"
    assert "EXHAUSTION_RISK" in rejected["rejection_reason"]
    assert agent._active_candidates[Direction.BUY_CALL.value] is None
    assert len(agent._candidate_shadow_history) == 1

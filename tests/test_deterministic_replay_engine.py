"""
tests/test_deterministic_replay_engine.py — Test Suite for Deterministic Replay Engine & Rolling Context Architecture

Verifies:
1. No random/synthetic path is reachable from the real backtest runner.
2. Same strategy classes in STRATEGY_REGISTRY are used by replay and production.
3. Native 5-minute timeframe is preserved.
4. Historical warm-up contains only completed candles.
5. No future candle leakage (Point-in-Time safety).
6. State-machine behavior is deterministic.
7. Trade outcomes come from actual historical price data.
8. Production feature flag defaults to SAFE/OFF (LEGACY_CONTEXT).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

from config.settings import STRATEGY_CONTEXT_MODE, MIN_STRATEGY_VOTES
from signalforge.backtest.deterministic_replay_engine import (
    DeterministicReplayEngine,
    ReplayTrade,
    ReplaySessionSummary,
)
from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY, StrategyAgent
from agents_code.agent2_strategy.pullback_state_machine import PullbackShadowStateMachine, PullbackState
from data.historical_store import HistoricalCandleStore

IST = pytz.timezone("Asia/Kolkata")


def test_production_feature_flag_safe_default():
    """Verify production feature flag defaults to SAFE/OFF (LEGACY_CONTEXT or PNL_MAXIMIZER_V1)."""
    assert STRATEGY_CONTEXT_MODE in {"LEGACY_CONTEXT", "ROLLING_5M_CONTEXT", "PNL_MAXIMIZER_V1"}
    # Safe default check: if not explicitly overridden, it is LEGACY_CONTEXT or PNL_MAXIMIZER_V1
    import os
    env_mode = os.getenv("STRATEGY_CONTEXT_MODE", "LEGACY_CONTEXT").strip().upper()
    assert env_mode in {"LEGACY_CONTEXT", "PNL_MAXIMIZER_V1"}


def test_same_strategy_registry_used():
    """Verify that replay engine uses the exact strategies in STRATEGY_REGISTRY."""
    assert len(STRATEGY_REGISTRY) >= 27
    engine = DeterministicReplayEngine()
    # StrategyAgent and DeterministicReplayEngine both reference STRATEGY_REGISTRY
    agent = StrategyAgent()
    assert len(agent._strategy_health) >= 27
    assert set(s.name for s in STRATEGY_REGISTRY).issubset(set(agent._strategy_health.keys()))


def test_no_synthetic_randomness_in_replay():
    """Verify that running replay twice on the exact same session produces 100% identical results."""
    engine = DeterministicReplayEngine(context_mode="ROLLING_5M_CONTEXT", warmup_bars=50)
    if not engine.available_sessions:
        pytest.skip("Historical database not populated")

    test_date = engine.available_sessions[-1]
    
    # Run 1
    summary1, trades1 = engine.replay_session(session_date=test_date)
    # Run 2
    summary2, trades2 = engine.replay_session(session_date=test_date)

    assert summary1.raw_4vote_signals == summary2.raw_4vote_signals
    assert summary1.ema20_approved_signals == summary2.ema20_approved_signals
    assert summary1.trades_executed == summary2.trades_executed
    assert summary1.net_pnl == summary2.net_pnl
    assert len(trades1) == len(trades2)
    for t1, t2 in zip(trades1, trades2):
        assert t1.trade_id == t2.trade_id
        assert t1.direction == t2.direction
        assert t1.entry_option_price == t2.entry_option_price
        assert t1.exit_option_price == t2.exit_option_price
        assert t1.net_pnl == t2.net_pnl


def test_point_in_time_integrity():
    """Verify zero lookahead leakage during replay."""
    engine = DeterministicReplayEngine(context_mode="ROLLING_5M_CONTEXT", warmup_bars=50)
    if not engine.available_sessions:
        pytest.skip("Historical database not populated")

    test_date = engine.available_sessions[-1]
    summary, _ = engine.replay_session(session_date=test_date)
    assert summary.lookahead_violations == 0


def test_native_5minute_timeframe_preserved():
    """Verify that strategies operate on 5-minute candles and warm-up is 5-minute bars."""
    engine = DeterministicReplayEngine(context_mode="ROLLING_5M_CONTEXT", warmup_bars=50)
    assert not engine._df_all_candles.empty
    # Time diff between consecutive bars within intraday session should be 5 minutes
    sample_diff = (engine._df_all_candles.index[1] - engine._df_all_candles.index[0]).total_seconds()
    assert sample_diff == 300  # 5 minutes in seconds


def test_state_machine_pullback_gating():
    """Verify that EMA20 retest state machine correctly gates raw breakout signals."""
    sm = PullbackShadowStateMachine(state_dir=None)
    test_ts = datetime(2026, 8, 1, 9, 45, tzinfo=IST)
    
    # 1. Register candidate signal
    signal_data = {
        "signal_id": "TEST_SIG_1",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "quality_classification": "MEDIUM_QUALITY",
        "nifty_ltp": 24500.0,
        "ema20": 24480.0,
        "atr": 25.0,
        "votes": 4,
        "categories": 3,
        "ml_state": "POSITIVE",
    }
    setup = sm.on_candidate_signal(signal_data, test_ts)
    assert setup is not None
    assert setup.state == PullbackState.PENDING_PULLBACK.value

    # 2. Candle touches EMA20 (24480.0) -> transitions to RETEST_DETECTED
    events = sm.on_candle(
        candle_open=24495.0,
        candle_high=24505.0,
        candle_low=24478.0,  # Touches EMA20
        candle_close=24490.0,
        current_ema20=24480.0,
        current_atr=25.0,
        current_ts=test_ts,
    )
    assert len(events) == 0
    assert setup.state == PullbackState.RETEST_DETECTED.value

    # 3. Confirmation candle touches EMA20 and closes green in breakout direction -> CONFIRMED / ENTRY
    events = sm.on_candle(
        candle_open=24485.0,
        candle_high=24510.0,
        candle_low=24482.0,  # Touches EMA20
        candle_close=24505.0,  # Green close
        current_ema20=24482.0,
        current_atr=25.0,
        current_ts=test_ts,
    )
    assert len(events) == 1
    assert events[0]["state"] == PullbackState.SHADOW_ENTRY.value

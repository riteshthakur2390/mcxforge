import os
import sys
from datetime import datetime

import pandas as pd
import pytz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backtesting.engine import load_cached_data
from agents_code.agent7_analytics.journal import AnalyticsAgent
from agents_code.agent3_ml.filter import MLFilterAgent
from agents_code.agent4_planner.planner import TradePlannerAgent
from agents_code.agent2_strategy.runner import StrategyAgent

IST = pytz.timezone("Asia/Kolkata")


def test_load_cached_data_filters_single_date(tmp_path, monkeypatch):
    idx = pd.date_range("2026-03-20 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    idx = idx.append(pd.date_range("2026-03-21 09:15", periods=4, freq="5min", tz="Asia/Kolkata"))
    df = pd.DataFrame({
        "open": [1.0] * len(idx),
        "high": [2.0] * len(idx),
        "low": [0.5] * len(idx),
        "close": [1.5] * len(idx),
        "volume": [100] * len(idx),
    }, index=idx)

    monkeypatch.setattr("backtesting.engine._refresh_cache_if_needed", lambda _: df)

    result = load_cached_data(target_date="2026-03-21")

    assert not result.empty
    # Data will include warmup candles, so only the selected_days attr should be exactly 2026-03-21
    assert [str(d) for d in result.attrs["selected_days"]] == ["2026-03-21"]
    assert result.attrs["live_parity_date"] is None


def test_load_cached_data_days_excludes_partial_today(tmp_path, monkeypatch):
    idx = pd.date_range("2026-03-20 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    idx = idx.append(pd.date_range("2026-03-21 09:15", periods=4, freq="5min", tz="Asia/Kolkata"))
    idx = idx.append(pd.date_range("2026-03-22 09:15", periods=2, freq="5min", tz="Asia/Kolkata"))
    df = pd.DataFrame({
        "open": [1.0] * len(idx),
        "high": [2.0] * len(idx),
        "low": [0.5] * len(idx),
        "close": [1.5] * len(idx),
        "volume": [100] * len(idx),
    }, index=idx)

    monkeypatch.setattr("backtesting.engine._refresh_cache_if_needed", lambda _: df)

    class FakeDateTime:
        @classmethod
        def now(cls, tz=None):
            return IST.localize(datetime(2026, 3, 22, 10, 0))

    monkeypatch.setattr("backtesting.engine.datetime", FakeDateTime)

    result = load_cached_data(days=2)

    assert [str(d) for d in result.attrs["selected_days"]] == ["2026-03-20", "2026-03-21"]
    assert result.attrs["live_parity_date"] is None


def test_load_cached_data_days_uses_calendar_window_not_trading_count(tmp_path, monkeypatch):
    idx = pd.date_range("2026-03-10 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    idx = idx.append(pd.date_range("2026-03-18 09:15", periods=4, freq="5min", tz="Asia/Kolkata"))
    idx = idx.append(pd.date_range("2026-03-19 09:15", periods=4, freq="5min", tz="Asia/Kolkata"))
    df = pd.DataFrame({
        "open": [1.0] * len(idx),
        "high": [2.0] * len(idx),
        "low": [0.5] * len(idx),
        "close": [1.5] * len(idx),
        "volume": [100] * len(idx),
    }, index=idx)

    monkeypatch.setattr("backtesting.engine._refresh_cache_if_needed", lambda _: df)

    class FakeDateTime:
        @classmethod
        def now(cls, tz=None):
            return IST.localize(datetime(2026, 3, 20, 10, 0))

    monkeypatch.setattr("backtesting.engine.datetime", FakeDateTime)

    result = load_cached_data(days=3)

    assert [str(d) for d in result.attrs["selected_days"]] == ["2026-03-18", "2026-03-19"]
    assert "2026-03-10" not in [str(d) for d in result.attrs["selected_days"]]


def test_load_cached_data_trading_days_uses_session_count(tmp_path, monkeypatch):
    idx = pd.date_range("2026-03-10 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    idx = idx.append(pd.date_range("2026-03-18 09:15", periods=4, freq="5min", tz="Asia/Kolkata"))
    idx = idx.append(pd.date_range("2026-03-19 09:15", periods=4, freq="5min", tz="Asia/Kolkata"))
    df = pd.DataFrame({
        "open": [1.0] * len(idx),
        "high": [2.0] * len(idx),
        "low": [0.5] * len(idx),
        "close": [1.5] * len(idx),
        "volume": [100] * len(idx),
    }, index=idx)

    monkeypatch.setattr("backtesting.engine._refresh_cache_if_needed", lambda _: df)

    class FakeDateTime:
        @classmethod
        def now(cls, tz=None):
            return IST.localize(datetime(2026, 3, 20, 10, 0))

    monkeypatch.setattr("backtesting.engine.datetime", FakeDateTime)

    result = load_cached_data(trading_days=3)

    assert [str(d) for d in result.attrs["selected_days"]] == [
        "2026-03-10",
        "2026-03-18",
        "2026-03-19",
    ]
    assert result.attrs["requested_trading_days"] == 3


def test_load_cached_data_today_enables_live_parity(tmp_path, monkeypatch):
    idx = pd.date_range("2026-03-21 09:15", periods=4, freq="5min", tz="Asia/Kolkata")
    idx = idx.append(pd.date_range("2026-03-22 09:15", periods=4, freq="5min", tz="Asia/Kolkata"))
    df = pd.DataFrame({
        "open": [1.0] * len(idx),
        "high": [2.0] * len(idx),
        "low": [0.5] * len(idx),
        "close": [1.5] * len(idx),
        "volume": [100] * len(idx),
    }, index=idx)

    monkeypatch.setattr("backtesting.engine._refresh_cache_if_needed", lambda _: df)

    class FakeDateTime:
        @classmethod
        def now(cls, tz=None):
            return IST.localize(datetime(2026, 3, 22, 10, 0))

    monkeypatch.setattr("backtesting.engine.datetime", FakeDateTime)

    result = load_cached_data(target_date="2026-03-22")

    assert [str(d) for d in result.attrs["selected_days"]] == ["2026-03-22"]
    assert str(result.attrs["live_parity_date"]) == "2026-03-22"


def test_strategy_resolves_message_timestamp():
    ts = StrategyAgent._resolve_timestamp("2026-03-26T10:05:00+05:30")
    assert ts.strftime("%H:%M") == "10:05"


def test_analytics_marks_backtest_entries_with_backtest_mode():
    agent = AnalyticsAgent(backtest_mode=True)

    entry = agent._base_entry(
        {
            "timestamp": "2026-03-26T10:05:00+05:30",
            "strategies_fired": ["ADX+PSAR", "BBSqueeze"],
            "ml_rank_score": 0.58,
            "confidence": 0.74,
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "vote_aligned",
                        "setup_strength": 0.79,
                    }
                }
            },
        },
        status="SUPPRESSED",
    )

    assert entry["mode"] == "BACKTEST"
    assert entry["strategy_combo"] == "ADX+PSAR|BBSqueeze"
    assert entry["entry_timing"] == "mid"
    assert entry["ml_score_bucket"] == "0.56-0.60"
    assert entry["setup_type"] == "vote_aligned"


def test_analytics_generates_post_trade_groupings():
    agent = AnalyticsAgent(backtest_mode=True)
    agent._journal = [
        {
            "lifecycle_status": "CLOSED",
            "regime": "TRENDING",
            "strategy_combo": "ADX+PSAR|BBSqueeze|VWAP+EMA",
            "ml_score_bucket": "0.52-0.56",
            "realized_pnl": -500.0,
        },
        {
            "lifecycle_status": "CLOSED",
            "regime": "TRENDING",
            "strategy_combo": "BBSqueeze|VWAP+EMA",
            "ml_score_bucket": "<0.52",
            "realized_pnl": 1200.0,
        },
    ]

    summary = agent._post_trade_diagnostics()

    assert summary["by_regime"]["TRENDING"]["count"] == 2
    assert summary["by_strategy_combo"]["ADX+PSAR|BBSqueeze|VWAP+EMA"]["losses"] == 1
    assert summary["by_ml_score_bucket"]["<0.52"]["wins"] == 1


def test_ml_filter_allows_low_consensus_early_trigger(monkeypatch):
    monkeypatch.setattr("agents_code.agent3_ml.filter.ALLOW_SUBMIN_VOTE_EARLY_TRIGGER", False)
    assert MLFilterAgent._allows_low_consensus_signal(
        {"metadata": {"_context": {"early_trigger": True}}}
    ) is False
    assert MLFilterAgent._allows_low_consensus_signal({}) is False


def test_ml_fallback_blocks_weak_breakout():
    allowed, reason = MLFilterAgent._passes_fallback_gate(
        {
            "votes": 4,
            "regime": "TRENDING",
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "breakout",
                        "setup_strength": 0.73,
                    }
                }
            },
        },
        fallback_conf=0.72,
    )

    assert allowed is False
    assert "weak breakout" in reason


def test_ml_fallback_blocks_weak_high_consensus_trend_pullback():
    allowed, reason = MLFilterAgent._passes_fallback_gate(
        {
            "votes": 4,
            "regime": "TRENDING",
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "trend_pullback",
                        "setup_strength": 0.72,
                    }
                }
            },
        },
        fallback_conf=0.78,
    )

    assert allowed is False
    assert "weak high-consensus trend_pullback" in reason


def test_ml_fallback_allows_stronger_trend_pullback():
    allowed, reason = MLFilterAgent._passes_fallback_gate(
        {
            "votes": 4,
            "regime": "TRENDING",
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_type": "trend_pullback",
                        "setup_strength": 0.94,
                    }
                }
            },
        },
        fallback_conf=0.81,
    )

    assert allowed is True
    assert reason == ""


def test_planner_allows_low_consensus_early_trigger(monkeypatch):
    monkeypatch.setattr("agents_code.agent4_planner.planner.ALLOW_SUBMIN_VOTE_EARLY_TRIGGER", False)
    assert TradePlannerAgent._allows_low_consensus_plan(
        {"metadata": {"_context": {"early_trigger": True}}}
    ) is False
    assert TradePlannerAgent._allows_low_consensus_plan({}) is False


def test_planner_blocks_weak_breakout_backtest_fallback():
    allowed, reason = TradePlannerAgent._should_allow_backtest_fallback(
        candidate={"score": 0.61},
        setup={"setup_type": "breakout", "setup_strength": 0.73},
        ml_rank_score=0.56,
    )

    assert allowed is False
    assert "weak breakout fallback blocked" in reason


def test_planner_allows_stronger_backtest_fallback():
    allowed, reason = TradePlannerAgent._should_allow_backtest_fallback(
        candidate={"score": 0.66},
        setup={"setup_type": "vote_aligned", "setup_strength": 0.81},
        ml_rank_score=0.54,
    )

    assert allowed is True
    assert reason == ""


def test_planner_execution_confidence_uses_predictive_inputs_not_rank_bonus():
    exec_conf = TradePlannerAgent._execution_confidence(
        {
            "confidence": 0.88,
            "metadata": {
                "_context": {
                    "setup": {
                        "setup_strength": 0.76,
                    }
                }
            },
        },
        rank_score=0.92,
        success_prob=0.36,
    )

    assert exec_conf == 0.584

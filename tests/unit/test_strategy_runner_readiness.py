import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY, StrategyAgent


def _build_df(periods: int, start: str, freq: str = "5min") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="Asia/Kolkata")
    base = list(range(periods))
    return pd.DataFrame(
        {
            "open": [100 + x for x in base],
            "high": [101 + x for x in base],
            "low": [99 + x for x in base],
            "close": [100.5 + x for x in base],
            "volume": [1000 + x for x in base],
        },
        index=index,
    )


def test_strategy_registry_has_all_unique_strategies():
    names = [meta.name for meta in STRATEGY_REGISTRY]

    assert len(STRATEGY_REGISTRY) >= 16
    assert len(names) == len(set(names))


def test_strategy_readiness_skips_with_clear_reasons():
    agent = StrategyAgent()
    agent.set_backtest_mode(True)
    df = _build_df(periods=30, start="2026-04-09 09:15")

    eligible, skipped = agent._get_eligible(df, has_orb=False)
    skipped_by_name = {item["name"]: item for item in skipped}

    assert "SuperTrend+RSI" in [meta.name for meta in eligible]
    assert skipped_by_name["ORB"]["reason"] == "ORB levels not available yet"
    assert "need at least" in skipped_by_name["Ichimoku"]["reason"]
    assert "need at least" in skipped_by_name["VolumeProfile"]["reason"]
    assert skipped_by_name["CPR"]["reason"] == "previous session data not available"


def test_strategy_readiness_allows_all_when_data_is_available():
    agent = StrategyAgent()
    agent.set_backtest_mode(False)
    day1 = _build_df(periods=60, start="2026-04-08 09:15")
    day2 = _build_df(periods=60, start="2026-04-09 09:15")
    df = pd.concat([day1, day2])

    eligible, skipped = agent._get_eligible(df, has_orb=True)

    assert len(eligible) == len(STRATEGY_REGISTRY)
    assert skipped == []


def test_weighted_voting_boosts_trend_strategies_in_trending_regime():
    agent = StrategyAgent()
    annotated = agent._annotate_votes(
        [
            {"name": "ADX+PSAR", "confidence": 0.70},
            {"name": "PriceAction", "confidence": 0.70},
        ],
        "TRENDING",
    )

    by_name = {item["name"]: item for item in annotated}

    assert by_name["ADX+PSAR"]["_vote_weight"] > by_name["PriceAction"]["_vote_weight"]
    assert by_name["ADX+PSAR"]["_weighted_score"] > by_name["PriceAction"]["_weighted_score"]


def test_weighted_vote_summary_uses_adjusted_scores():
    agent = StrategyAgent()
    call_votes = agent._annotate_votes(
        [{"name": "Ichimoku", "confidence": 0.68}],
        "TRENDING",
    )
    put_votes = agent._annotate_votes(
        [{"name": "LiqSweep", "confidence": 0.72}],
        "TRENDING",
    )

    call_summary = agent._summarize_votes(call_votes)
    put_summary = agent._summarize_votes(put_votes)

    assert call_summary.count == 1
    assert put_summary.count == 1
    assert call_summary.weighted_score > put_summary.weighted_score


def test_early_trigger_allows_strong_single_vote_setup():
    agent = StrategyAgent()
    setup = type(
        "Setup",
        (),
        {"setup_type": "trend_pullback", "setup_strength": 0.73},
    )()

    assert agent._should_allow_early_trigger(
        best_conf=0.87,
        weighted_score=0.87,
        setup=setup,
    ) is True


def test_early_trigger_rejects_weak_single_vote_setup():
    agent = StrategyAgent()
    setup = type(
        "Setup",
        (),
        {"setup_type": "none", "setup_strength": 0.45},
    )()

    assert agent._should_allow_early_trigger(
        best_conf=0.61,
        weighted_score=0.70,
        setup=setup,
    ) is False

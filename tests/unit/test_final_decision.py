import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.final_decision import FinalDecisionLayer


def test_final_decision_positive_expectancy_sizes_trade():
    layer = FinalDecisionLayer(journal_dir="/tmp/does-not-exist", capital=100000, slippage_pct=0.5)

    decision = layer.evaluate(
        entry_premium=100.0,
        target_premium=160.0,
        stop_premium=75.0,
        ml_success_prob=0.64,
        ml_rank_score=0.71,
        atr_pct=0.003,
        strategies=["ADX+PSAR", "Ichimoku"],
        regime="TRENDING",
        setup={"setup_type": "trend_pullback", "setup_strength": 0.84, "expected_move": 24.0},
    )

    assert decision.tradeable is True
    assert 1.0 <= decision.risk_pct <= 2.0
    assert decision.expectancy_inr > 0
    assert decision.desired_lots >= 1
    assert decision.quantity >= 65


def test_final_decision_rejects_negative_expectancy_trade():
    layer = FinalDecisionLayer(journal_dir="/tmp/does-not-exist", capital=100000, slippage_pct=1.0)

    decision = layer.evaluate(
        entry_premium=120.0,
        target_premium=126.0,
        stop_premium=90.0,
        ml_success_prob=0.34,
        ml_rank_score=0.30,
        atr_pct=0.009,
        strategies=["PriceAction"],
        regime="HIGH_VOL",
        setup={"setup_type": "mean_reversion", "setup_strength": 0.42, "expected_move": 5.0},
    )

    assert decision.tradeable is False
    assert decision.expectancy_inr <= 0 or "reward" in decision.reason or "expectancy" in decision.reason
    assert decision.desired_lots == 0


def test_final_decision_setup_strength_improves_signal_strength():
    layer = FinalDecisionLayer(journal_dir="/tmp/does-not-exist", capital=100000, slippage_pct=0.0)

    weak = layer.evaluate(
        entry_premium=100.0,
        target_premium=150.0,
        stop_premium=80.0,
        ml_success_prob=0.58,
        ml_rank_score=0.58,
        atr_pct=0.0035,
        strategies=["PriceAction"],
        regime="RANGING",
        setup={"setup_type": "mean_reversion", "setup_strength": 0.45, "expected_move": 10.0},
    )
    strong = layer.evaluate(
        entry_premium=100.0,
        target_premium=150.0,
        stop_premium=80.0,
        ml_success_prob=0.58,
        ml_rank_score=0.58,
        atr_pct=0.0035,
        strategies=["PriceAction"],
        regime="RANGING",
        setup={"setup_type": "mean_reversion", "setup_strength": 0.88, "expected_move": 24.0},
    )

    assert strong.risk_pct >= weak.risk_pct
    assert strong.expectancy_inr >= weak.expectancy_inr


def test_final_decision_does_not_reblock_ranked_setup_on_raw_ml_prob_alone():
    layer = FinalDecisionLayer(journal_dir="/tmp/does-not-exist", capital=100000, slippage_pct=0.0)

    decision = layer.evaluate(
        entry_premium=95.0,
        target_premium=160.0,
        stop_premium=65.0,
        ml_success_prob=0.20,
        ml_rank_score=0.44,
        atr_pct=0.0024,
        strategies=["ADX+PSAR"],
        regime="TRENDING",
        setup={"setup_type": "trend_pullback", "setup_strength": 0.71, "expected_move": 45.9},
    )

    assert decision.expectancy_inr > 0
    assert decision.tradeable is True

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from core.models import Direction
from agents_code.agent2_strategy.runner import StrategyAgent

IST = pytz.timezone("Asia/Kolkata")

def _make_df(prices):
    n = len(prices)
    dates = pd.date_range("2026-08-27 09:15", periods=n, freq="5min", tz=IST)
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 5.0 for p in prices],
        "low": [p - 5.0 for p in prices],
        "close": prices,
        "volume": [1000] * n,
    }, index=dates)
    return df

def test_remaining_opportunity_excellent():
    """Verify EXCELLENT classification when estimated R:R >= 2.5."""
    prices = [24400.0 + i * 2.0 for i in range(20)]
    df = _make_df(prices)
    ltp = 24440.0
    atr_val = 20.0
    
    # Class call with large target distance
    class DummySetup:
        setup_type = "breakout"
        setup_strength = 0.8
        anchor_price = 24410.0
        context = {"target_level": 24520.0} # 80 pts target, ~20 pts stop -> R:R ~ 4.0
        
    opp = StrategyAgent._evaluate_remaining_opportunity(
        df=df,
        ltp=ltp,
        direction=Direction.BUY_CALL,
        atr_val=atr_val,
        setup=DummySetup(),
        hybrid_5m={},
    )
    
    assert opp["entry_price"] == 24440.0
    assert opp["stop_risk_distance"] > 0
    assert opp["distance_to_potential_target"] == 80.0
    assert opp["estimated_remaining_rr"] >= 2.5
    assert opp["remaining_opportunity_class"] == "EXCELLENT"
    assert opp["shadow_opportunity_decision"] == "ACCEPT"


def test_remaining_opportunity_acceptable_and_marginal():
    """Verify ACCEPTABLE (1.5 - 2.5) and MARGINAL (1.0 - 1.5) classifications."""
    prices = [24400.0 + i * 2.0 for i in range(20)]
    df = _make_df(prices)
    ltp = 24440.0
    atr_val = 20.0
    
    # Target 50 pts away, stop 29 pts -> R:R = 1.72 -> ACCEPTABLE
    class DummySetupAcceptable:
        setup_type = "pullback"
        setup_strength = 0.6
        context = {"target_level": 24490.0}

    opp1 = StrategyAgent._evaluate_remaining_opportunity(
        df=df,
        ltp=ltp,
        direction=Direction.BUY_CALL,
        atr_val=atr_val,
        setup=DummySetupAcceptable(),
        hybrid_5m={},
    )
    assert opp1["remaining_opportunity_class"] == "ACCEPTABLE"
    assert opp1["shadow_opportunity_decision"] == "ACCEPT"

    # Target 35 pts away, stop 29 pts -> R:R = 1.20 -> MARGINAL
    class DummySetupMarginal:
        setup_type = "reversal"
        setup_strength = 0.5
        context = {"target_level": 24475.0}

    opp2 = StrategyAgent._evaluate_remaining_opportunity(
        df=df,
        ltp=ltp,
        direction=Direction.BUY_CALL,
        atr_val=atr_val,
        setup=DummySetupMarginal(),
        hybrid_5m={},
    )
    assert opp2["remaining_opportunity_class"] == "MARGINAL"
    assert opp2["shadow_opportunity_decision"] == "CAUTION"


def test_remaining_opportunity_poor():
    """Verify POOR classification when estimated R:R < 1.0."""
    prices = [24400.0 + i * 2.0 for i in range(20)]
    df = _make_df(prices)
    ltp = 24440.0
    atr_val = 20.0
    
    # Target directly into heavy resistance 8 pts away, stop 20 pts -> R:R = 0.4 -> POOR
    class DummySetupPoor:
        setup_type = "trend"
        setup_strength = 0.4
        context = {"target_level": 24448.0}
        
    opp = StrategyAgent._evaluate_remaining_opportunity(
        df=df,
        ltp=ltp,
        direction=Direction.BUY_CALL,
        atr_val=atr_val,
        setup=DummySetupPoor(),
        hybrid_5m={},
    )
    
    assert opp["estimated_remaining_rr"] < 1.0
    assert opp["remaining_opportunity_class"] == "POOR"
    assert opp["shadow_opportunity_decision"] == "REJECT"

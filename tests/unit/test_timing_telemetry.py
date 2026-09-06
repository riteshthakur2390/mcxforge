import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import pytz

from agents_code.agent2_strategy.runner import StrategyAgent, STRATEGY_CATEGORY_MAPPING
from core.models import Direction, Regime, RawSignal

IST = pytz.timezone("Asia/Kolkata")

def create_synthetic_candles(n=30, trend="up", base_price=24000.0, step=10.0):
    now = datetime(2026, 8, 27, 10, 0, tzinfo=IST)
    rows = []
    price = base_price
    for i in range(n):
        ts = now + timedelta(minutes=5 * i)
        if trend == "up":
            o = price
            c = price + step
            h = c + 5.0
            l = o - 2.0
            price = c
        elif trend == "down":
            o = price
            c = price - step
            h = o + 2.0
            l = c - 5.0
            price = c
        else:
            o = price
            c = price + (5.0 if i % 2 == 0 else -5.0)
            h = max(o, c) + 3.0
            l = min(o, c) - 3.0
        rows.append({
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": 10000 + i * 100,
            "timestamp": ts,
        })
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def test_classify_entry_timing_thresholds():
    """Verify non-trading classification boundaries."""
    # EARLY: extension <= 0.65 and dist_ema <= 0.75
    assert StrategyAgent._classify_entry_timing(0.4, 0.5, 2) == "EARLY"
    assert StrategyAgent._classify_entry_timing(0.65, 0.75, 1) == "EARLY"

    # VALID: 0.65 < extension <= 1.35 and dist_ema <= 1.50
    assert StrategyAgent._classify_entry_timing(1.0, 1.2, 3) == "VALID"
    assert StrategyAgent._classify_entry_timing(1.35, 1.50, 4) == "VALID"

    # EXTENDED: 1.35 < extension <= 2.10 and dist_ema <= 2.25
    assert StrategyAgent._classify_entry_timing(1.8, 1.9, 4) == "EXTENDED"
    assert StrategyAgent._classify_entry_timing(2.10, 2.20, 5) == "EXTENDED"

    # EXHAUSTED: extension > 2.10 or dist_ema > 2.25
    assert StrategyAgent._classify_entry_timing(2.5, 2.0, 6) == "EXHAUSTED"
    assert StrategyAgent._classify_entry_timing(1.5, 2.5, 5) == "EXHAUSTED"


def test_timing_telemetry_and_entry_quality_generation():
    """Verify complete timing and entry quality telemetry structure."""
    df = create_synthetic_candles(n=25, trend="up", base_price=24000.0, step=15.0)
    current_ts = df.index[-1]
    ltp = float(df["close"].iloc[-1])
    direction = Direction.BUY_CALL

    cand_ts = current_ts - timedelta(minutes=10)
    cand_price = ltp - 30.0
    cand_info = {"ts": cand_ts, "price": cand_price, "votes": 2}

    class DummySetup:
        setup_type = "trend_pullback"
        setup_strength = 0.75
        anchor_price = 24100.0
        context = {"breakout_level": 24100.0, "target_level": 24450.0}

    hybrid_5m = {"vwap": 24200.0}

    timing_telemetry, entry_quality, timing_class = StrategyAgent._build_timing_and_quality_telemetry(
        df=df,
        ltp=ltp,
        direction=direction,
        current_ts=current_ts,
        cand_info=cand_info,
        setup=DummySetup(),
        hybrid_5m=hybrid_5m,
    )

    # 1. Check Timing Telemetry
    assert timing_telemetry["candidate_ts"] == cand_ts.isoformat()
    assert timing_telemetry["confirmation_ts"] == current_ts.isoformat()
    assert timing_telemetry["raw_signal_ts"] == current_ts.isoformat()
    assert timing_telemetry["entry_ts"] is None
    assert timing_telemetry["candidate_price"] == round(cand_price, 2)
    assert timing_telemetry["confirmation_price"] == round(ltp, 2)
    assert timing_telemetry["raw_signal_price"] == round(ltp, 2)
    assert timing_telemetry["confirmation_delay_sec"] == 600.0
    assert timing_telemetry["candidate_to_confirmation_pts"] == 30.0

    # 2. Check Entry Quality
    assert entry_quality["atr"] > 0
    assert entry_quality["extension_atr"] >= 0
    assert entry_quality["consecutive_directional_candles"] >= 1
    assert entry_quality["recent_directional_move_pts"] > 0
    assert 0 <= entry_quality["move_consumed_pct"] <= 100
    assert entry_quality["dist_from_vwap"] == round(ltp - 24200.0, 2)
    assert entry_quality["dist_from_breakout_level"] == round(abs(ltp - 24100.0), 2)
    assert entry_quality["opposing_level_dist"] == round(abs(24450.0 - ltp), 2)
    assert entry_quality["timing_classification"] in {"EARLY", "VALID", "EXTENDED", "EXHAUSTED"}
    assert timing_class == entry_quality["timing_classification"]


def test_timing_telemetry_graceful_missing_data():
    """Verify that empty dataframe or None objects do not crash and produce valid defaults."""
    empty_df = pd.DataFrame()
    current_ts = datetime(2026, 8, 27, 10, 0, tzinfo=IST)
    ltp = 24500.0

    timing_telemetry, entry_quality, timing_class = StrategyAgent._build_timing_and_quality_telemetry(
        df=empty_df,
        ltp=ltp,
        direction=Direction.BUY_PUT,
        current_ts=current_ts,
        cand_info=None,
        setup=None,
        hybrid_5m=None,
    )

    assert timing_telemetry["candidate_ts"] == current_ts.isoformat()
    assert timing_telemetry["confirmation_delay_sec"] == 0.0
    assert timing_telemetry["candidate_to_confirmation_pts"] == 0.0
    assert entry_quality["atr"] == 20.0
    assert entry_quality["timing_classification"] in {"EARLY", "VALID", "EXTENDED", "EXHAUSTED"}

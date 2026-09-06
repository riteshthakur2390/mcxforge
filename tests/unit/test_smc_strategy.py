import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents_code.agent2_strategy.s17_smc import MarketStructure, SMCStrategy, SwingPoint


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-04-09 09:15", periods=8, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106, 107],
            "high": [101, 102, 103, 104, 105, 106, 107, 108],
            "low": [99, 100, 101, 102, 103, 104, 105, 106],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5],
            "volume": [1000] * 8,
        },
        index=index,
    )


def test_classify_structure_detects_bullish_recent_hh_hl():
    strategy = SMCStrategy()
    swings = [
        SwingPoint("high", 112.0, 2),
        SwingPoint("low", 106.0, 3),
        SwingPoint("high", 108.0, 5),
        SwingPoint("low", 102.0, 6),
    ]

    structure = strategy._classify_structure(_frame(), swings)

    assert structure == MarketStructure(
        trend="bullish",
        last_hh=112.0,
        last_hl=106.0,
        last_lh=None,
        last_ll=None,
        swings=swings,
        bos_level=112.0,
        choch_level=106.0,
    )


def test_classify_structure_detects_bearish_recent_lh_ll():
    strategy = SMCStrategy()
    swings = [
        SwingPoint("high", 108.0, 2),
        SwingPoint("low", 102.0, 3),
        SwingPoint("high", 112.0, 5),
        SwingPoint("low", 106.0, 6),
    ]

    structure = strategy._classify_structure(_frame(), swings)

    assert structure == MarketStructure(
        trend="bearish",
        last_hh=None,
        last_hl=None,
        last_lh=108.0,
        last_ll=102.0,
        swings=swings,
        bos_level=102.0,
        choch_level=108.0,
    )

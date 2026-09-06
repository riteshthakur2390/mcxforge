import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents_code.agent2_strategy.market_structure import MarketStructureLiquidityEngine


def _df(rows: list[dict], start: str = "2026-04-09 09:15") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(rows), freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(rows, index=index)


def _series(value: float, length: int) -> pd.Series:
    return pd.Series([value] * length)


def test_market_structure_detects_bullish_liquidity_sweep():
    rows = []
    for idx in range(18):
        base = 100.0 + (0.18 if idx % 2 else -0.15)
        rows.append({"open": base + 0.1, "high": base + 0.7, "low": base - 0.7, "close": base, "volume": 900})
    rows.append({"open": 100.1, "high": 100.6, "low": 99.1, "close": 99.8, "volume": 950})
    rows.append({"open": 99.4, "high": 100.7, "low": 98.2, "close": 100.5, "volume": 1300})
    df = _df(rows)
    cache = SimpleNamespace(atr_14=_series(1.0, len(df)))

    snapshot = MarketStructureLiquidityEngine().evaluate(df=df, cache=cache)

    assert snapshot["liquidity_event"]["direction"] == "BUY_CALL"
    assert snapshot["liquidity_event"]["type"] in {"liquidity_sweep_low", "structure_break"}
    assert snapshot["entry_validation"]["by_direction"]["BUY_CALL"]["valid"] is True


def test_market_structure_avoids_mid_range_without_confirmation():
    rows = []
    for idx in range(24):
        drift = 100.0 + (0.4 if idx % 2 else -0.35)
        rows.append({"open": drift, "high": drift + 0.8, "low": drift - 0.8, "close": 100.1, "volume": 850})
    df = _df(rows)
    cache = SimpleNamespace(atr_14=_series(1.1, len(df)))

    snapshot = MarketStructureLiquidityEngine().evaluate(df=df, cache=cache)

    assert snapshot["entry_validation"]["middle_range"] is True
    assert snapshot["entry_validation"]["valid"] is False
    assert snapshot["liquidity_event"]["type"] == "NONE"

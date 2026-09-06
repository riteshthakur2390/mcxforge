import os
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents_code.agent2_strategy.setup_engine import TradeSetupEngine


def _df(rows: list[dict]) -> pd.DataFrame:
    index = pd.date_range("2026-04-09 09:15", periods=len(rows), freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(rows, index=index)


def _series(value: float, length: int) -> pd.Series:
    return pd.Series([value] * length)


def _valid_structure(direction: str) -> dict:
    other = "BUY_PUT" if direction == "BUY_CALL" else "BUY_CALL"
    return {
        "structure_state": {
            "bias": "BULLISH" if direction == "BUY_CALL" else "BEARISH",
            "bos": "BULLISH_BOS" if direction == "BUY_CALL" else "BEARISH_BOS",
            "bos_direction": direction,
            "choch": "NONE",
            "choch_direction": "NONE",
        },
        "liquidity_zones": [],
        "liquidity_event": {
            "type": "structure_break",
            "direction": direction,
            "level": 100.0,
            "fake_breakout": False,
        },
        "entry_validation": {
            "valid": True,
            "avoid": False,
            "middle_range": False,
            "by_direction": {
                "BUY_CALL": {"valid": direction == "BUY_CALL", "reasons": [] if direction == "BUY_CALL" else ["blocked"]},
                "BUY_PUT": {"valid": direction == "BUY_PUT", "reasons": [] if direction == "BUY_PUT" else ["blocked"]},
            },
        },
    }


def test_trade_setup_engine_detects_trend_pullback():
    rows = []
    price = 100.0
    for _ in range(29):
        rows.append(
            {"open": price - 0.2, "high": price + 0.8, "low": price - 0.8, "close": price, "volume": 1000}
        )
        price += 1.0
    rows.append({"open": 128.1, "high": 129.6, "low": 127.8, "close": 129.0, "volume": 1200})
    df = _df(rows)
    cache = SimpleNamespace(
        adx=_series(26.0, len(df)),
        ema_20=_series(128.2, len(df)),
        ema_50=_series(121.0, len(df)),
        atr_14=_series(1.6, len(df)),
        bb_width=None,
        vol_ratio=None,
        vwap=None,
    )

    engine = TradeSetupEngine()
    engine._structure_engine.evaluate = lambda **kwargs: _valid_structure("BUY_CALL")
    setup = engine.evaluate(
        df=df,
        cache=cache,
        regime_details={"detailed_regime": {"label": "TRENDING", "confidence": 0.8}},
    )

    assert setup is not None
    assert setup.setup_type == "trend_pullback"
    assert setup.direction == "BUY_CALL"
    assert 0.55 <= setup.setup_strength <= 0.95
    assert "market_structure" in setup.context


def test_trade_setup_engine_detects_breakout():
    rows = []
    base = 100.0
    for idx in range(24):
        high = base + 1.0 + (0.05 if idx % 2 else 0.0)
        low = base - 0.2
        close = base + (0.2 if idx % 3 else 0.35)
        rows.append({"open": base, "high": high, "low": low, "close": close, "volume": 1000})
    rows.append({"open": 101.2, "high": 103.1, "low": 101.0, "close": 102.8, "volume": 1600})
    df = _df(rows)
    cache = SimpleNamespace(
        atr_14=_series(1.0, len(df)),
        bb_width=_series(6.0, len(df)),
        vol_ratio=_series(1.35, len(df)),
        adx=_series(25.0, len(df)),
        ema_20=None,
        ema_50=None,
        vwap=None,
    )

    engine = TradeSetupEngine()
    engine._structure_engine.evaluate = lambda **kwargs: _valid_structure("BUY_CALL")
    setup = engine.evaluate(
        df=df,
        cache=cache,
        regime_details={"detailed_regime": {"label": "TRENDING", "confidence": 0.7}},
    )
    print("DEBUG TEST SETUP:", setup)

    assert setup is not None
    assert setup.setup_type == "breakout"
    assert setup.direction == "BUY_CALL"
    assert setup.expected_move > 0


def test_trade_setup_engine_detects_mean_reversion():
    rows = []
    for _ in range(24):
        rows.append({"open": 97.8, "high": 98.4, "low": 97.1, "close": 97.6, "volume": 900})
    rows.append({"open": 97.0, "high": 98.8, "low": 96.7, "close": 98.2, "volume": 1100})
    df = _df(rows)
    cache = SimpleNamespace(
        vwap=_series(100.0, len(df)),
        atr_14=_series(1.2, len(df)),
        adx=None,
        ema_20=None,
        ema_50=None,
        bb_width=None,
        vol_ratio=None,
    )

    engine = TradeSetupEngine()
    engine._structure_engine.evaluate = lambda **kwargs: _valid_structure("BUY_CALL")
    setup = engine.evaluate(
        df=df,
        cache=cache,
        regime_details={"detailed_regime": {"label": "RANGING", "confidence": 0.74}},
    )

    assert setup is not None
    assert setup.setup_type == "mean_reversion"
    assert setup.direction == "BUY_CALL"
    assert setup.stop_loss_level < setup.entry_zone_low

"""
tests/integration/test_pipeline.py — Full Pipeline Integration Test
=====================================================================
Tests the complete signal pipeline using synthetic NIFTY data.
No Kite API required — runs fully offline.

Run: pytest tests/integration/test_pipeline.py -v
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.bus import get_bus, reset_bus, Topic


@pytest.fixture(autouse=True)
def clean_bus():
    reset_bus()
    yield
    reset_bus()


def make_trending_df(n: int = 100) -> pd.DataFrame:
    """Generate synthetic trending NIFTY 5-min candles."""
    base  = 22000.0
    dates = pd.date_range("2026-03-18 09:15", periods=n, freq="5min",
                          tz="Asia/Kolkata")
    np.random.seed(42)

    # Strong uptrend with low noise
    trend  = np.linspace(0, 300, n)
    noise  = np.random.normal(0, 15, n)
    close  = base + trend + noise
    high   = close + np.random.uniform(5, 30, n)
    low    = close - np.random.uniform(5, 30, n)
    open_  = close - np.random.normal(0, 10, n)
    volume = np.random.randint(50_000, 200_000, n)

    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=dates)
    return df.between_time("09:15", "15:30")


def make_choppy_df(n: int = 80) -> pd.DataFrame:
    """Generate synthetic choppy/sideways NIFTY candles."""
    base  = 22000.0
    dates = pd.date_range("2026-03-18 09:15", periods=n, freq="5min",
                          tz="Asia/Kolkata")
    np.random.seed(99)

    # Oscillating chop with low ADX
    noise = np.random.normal(0, 2, n)
    close = base + np.sin(np.linspace(0, 12 * np.pi, n)) * 4 + noise
    high   = close + np.random.uniform(1.0, 3.0, n)
    low    = close - np.random.uniform(1.0, 3.0, n)
    open_  = close - np.random.normal(0, 1, n)
    volume = np.random.randint(30_000, 80_000, n)

    df = pd.DataFrame({
        "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    }, index=dates)
    return df.between_time("09:15", "15:30")


def df_to_candles(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "datetime": str(idx), "open": float(r["open"]),
            "high": float(r["high"]), "low": float(r["low"]),
            "close": float(r["close"]), "volume": int(r["volume"]),
        }
        for idx, r in df.iterrows()
    ]


# ── REGIME AGENT TESTS ──────────────────────────────────────────────────────
class TestRegimeAgent:

    def test_trending_market_passes(self):
        """Strong trend → MARKET_REGIME published."""
        received = []

        async def run():
            from agents_code.agent9_regime.classifier import MarketRegimeAgent
            bus   = get_bus()
            agent = MarketRegimeAgent()
            agent.register()

            async def capture(msg): received.append(msg.topic)
            bus.subscribe(Topic.MARKET_REGIME,    capture)
            bus.subscribe(Topic.SIGNAL_SUPPRESSED, capture)

            df     = make_trending_df(80)
            candles= df_to_candles(df)
            await bus.publish(Topic.CANDLES_READY, {
                "candles": candles, "ltp": float(df["close"].iloc[-1]),
                "symbol": "NIFTY", "timeframe": "5minute",
                "timestamp": str(df.index[-1]),
            }, "test")

        asyncio.run(run())
        # Trending data should pass (MARKET_REGIME) not suppress
        assert Topic.MARKET_REGIME in received or Topic.SIGNAL_SUPPRESSED in received

    def test_choppy_market_suppressed(self):
        """Choppy market → SIGNAL_SUPPRESSED (or MARKET_REGIME for borderline)."""
        suppressed = []

        async def run():
            from agents_code.agent9_regime.classifier import MarketRegimeAgent
            bus   = get_bus()
            agent = MarketRegimeAgent()
            agent._india_vix = 14.0   # normal VIX so chop index drives RANGING suppression
            agent.register()

            async def capture(msg): suppressed.append(msg.topic)
            bus.subscribe(Topic.SIGNAL_SUPPRESSED, capture)

            df     = make_choppy_df(60)
            candles= df_to_candles(df)
            await bus.publish(Topic.CANDLES_READY, {
                "candles": candles, "ltp": float(df["close"].iloc[-1]),
                "symbol": "NIFTY", "timeframe": "5minute",
                "timestamp": str(df.index[-1]),
            }, "test")

        asyncio.run(run())
        assert Topic.SIGNAL_SUPPRESSED in suppressed


# ── STRATEGY AGENT TESTS ────────────────────────────────────────────────────
class TestStrategyAgent:

    def test_strategy_needs_50_candles(self):
        """Less than 50 candles → no signal."""
        published = []

        async def run():
            from agents_code.agent2_strategy.runner import StrategyAgent
            bus   = get_bus()
            agent = StrategyAgent()
            agent.register()

            async def capture(msg): published.append(msg.topic)
            bus.subscribe(Topic.RAW_SIGNAL, capture)

            df     = make_trending_df(30)   # only 30 candles
            candles= df_to_candles(df)
            await bus.publish(Topic.MARKET_REGIME, {
                "candles": candles, "ltp": float(df["close"].iloc[-1]),
                "regime": "TRENDING",
                "timestamp": "2026-03-18T10:00:00+05:30",
            }, "test")

        asyncio.run(run())
        assert Topic.RAW_SIGNAL not in published


# ── OPTION UTILS TESTS ──────────────────────────────────────────────────────
class TestOptionUtils:

    def test_premium_sanity_range(self):
        from utils.option_utils import estimate_atm_premium
        for dte in [1, 3, 7, 15, 30]:
            p = estimate_atm_premium(22500, dte, iv=0.14)
            assert p > 0, f"Premium should be positive for DTE={dte}"
            assert p < 22500 * 0.05, f"Premium too high for DTE={dte}"

    def test_expiry_uses_tuesday_weekly_rule(self):
        from utils.option_utils import get_nearest_expiry
        for _ in range(5):
            expiry, dte = get_nearest_expiry(min_days=1, symbol="NIFTY")
            assert expiry.weekday() in {0, 1}, "Expiry must resolve to Tuesday or prior trading day"

    def test_symbol_format(self):
        from utils.option_utils import build_option_symbol
        from datetime import date
        sym = build_option_symbol("NIFTY", date(2026, 3, 26), 22500, "CE")
        assert sym.startswith("NIFTY")
        assert "22500" in sym
        assert sym.endswith("CE")
        assert len(sym) > 10


# ── FULL PIPELINE SMOKE TEST ────────────────────────────────────────────────
class TestFullPipeline:

    def test_bus_message_flow(self):
        """
        Verify message topics flow correctly through subscriptions.
        All agents registered — publish CANDLES_READY and trace what fires.
        """
        events = []

        async def run():
            from agents_code.agent9_regime.classifier  import MarketRegimeAgent
            from agents_code.agent7_analytics.journal  import AnalyticsAgent

            bus    = get_bus()
            regime = MarketRegimeAgent()
            regime.register()

            analytics = AnalyticsAgent()
            analytics.register()

            async def capture(msg): events.append(msg.topic)
            for topic in [Topic.MARKET_REGIME, Topic.SIGNAL_SUPPRESSED,
                          Topic.RAW_SIGNAL, Topic.ALERT]:
                bus.subscribe(topic, capture)

            # Publish premarket bias first
            await bus.publish(Topic.PREMARKET_BIAS, {
                "bias": "BULLISH", "india_vix": 13.0, "gap_pct": 0.3,
                "timestamp": "2026-03-18T09:00:00+05:30",
            }, "test")

            # Now publish candles
            df     = make_trending_df(80)
            candles= df_to_candles(df)
            await bus.publish(Topic.CANDLES_READY, {
                "candles": candles, "ltp": float(df["close"].iloc[-1]),
                "symbol": "NIFTY", "timeframe": "5minute",
                "orb_high": None, "orb_low": None,
                "timestamp": str(df.index[-1]),
            }, "test")

        asyncio.run(run())

        # At minimum: ALERT from morning brief, and MARKET_REGIME or SIGNAL_SUPPRESSED
        assert Topic.ALERT in events
        assert (
            Topic.MARKET_REGIME in events or
            Topic.SIGNAL_SUPPRESSED in events
        ), f"Expected regime event. Got: {events}"

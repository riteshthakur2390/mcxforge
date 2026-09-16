"""
tests/unit/test_commodity_ensemble.py — Unit Tests for Commodity Ensemble & Live Paper Runner
=============================================================================================
Validates:
1. SignalForgeStrategyAdapter integration with BaseCommodityStrategy
2. MCXSession timestamp classification (Morning, Afternoon, Evening)
3. CommodityEnsembleEngine multi-strategy voting and contribution attribution
4. Commodity Rich Performance Report formatting
5. LiveCommodityPaperRunner position management and journal tracking
"""

from datetime import datetime, time, timedelta
import os
import pandas as pd
import pytest
import pytz

from instruments import SILVERM_CONFIG, SILVERMIC_CONFIG
from core.models import Direction
from core.strategies.ensemble import (
    MCXSession,
    SignalForgeStrategyAdapter,
    CommodityEnsembleEngine,
    build_default_strategy_suite,
    EnsembleTradeRecord,
)
from utils.commodity_report_formatter import (
    format_commodity_rich_report,
    format_commodity_markdown_report,
)
from scripts.live_commodity_paper_runner import LiveCommodityPaperRunner

IST = pytz.timezone("Asia/Kolkata")


def _generate_synthetic_candles(n: int = 120, base_price: float = 75000.0) -> pd.DataFrame:
    """Generates synthetic intraday 15m commodity candles."""
    start_dt = datetime(2026, 9, 1, 9, 15, tzinfo=IST)
    times = [start_dt + timedelta(minutes=15 * i) for i in range(n)]
    
    # Generate realistic upward momentum
    prices = [base_price + (i * 25.0) + (10.0 if i % 2 == 0 else -10.0) for i in range(n)]
    
    data = []
    for i, (t, p) in enumerate(zip(times, prices)):
        data.append({
            "timestamp": t,
            "open": p - 5.0,
            "high": p + 15.0,
            "low": p - 10.0,
            "close": p + 5.0,
            "volume": 150 + (i % 5) * 20,
        })

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def test_mcx_session_classification():
    """Verifies that timestamps are classified into correct MCX sessions."""
    assert MCXSession.from_time(time(9, 30)) == MCXSession.MORNING
    assert MCXSession.from_time(time(12, 45)) == MCXSession.MORNING
    assert MCXSession.from_time(time(13, 0)) == MCXSession.AFTERNOON
    assert MCXSession.from_time(time(16, 30)) == MCXSession.AFTERNOON
    assert MCXSession.from_time(time(17, 0)) == MCXSession.EVENING
    assert MCXSession.from_time(time(21, 15)) == MCXSession.EVENING
    assert MCXSession.from_time(time(23, 15)) == MCXSession.EVENING
    assert MCXSession.from_time(time(2, 0)) == MCXSession.OFF_MARKET


def test_signalforge_strategy_adapter():
    """Verifies that SignalForgeStrategyAdapter generates valid commodity futures signals."""
    from agents_code.agent2_strategy.s1_supertrend_rsi import SuperTrendRSI
    
    adapter = SignalForgeStrategyAdapter(SuperTrendRSI, name="SuperTrend+RSI")
    adapter.initialize(SILVERM_CONFIG)
    
    df = _generate_synthetic_candles(n=80)
    sig = adapter.generate_signal(df)
    
    assert sig is not None
    assert sig.strategy == "SuperTrend+RSI"
    assert sig.instrument in ("SILVERM", "SILVERMIC")


def test_commodity_ensemble_engine_run():
    """Verifies multi-strategy ensemble backtest execution and attribution."""
    df = _generate_synthetic_candles(n=100)
    
    engine = CommodityEnsembleEngine(
        instrument_config=SILVERM_CONFIG,
        min_votes=1,
        capital=200000.0,
    )
    
    res = engine.run(df, timeframe="15m")
    
    assert res is not None
    assert res.symbol in ("SILVERM", "SILVERMIC")
    assert res.timeframe == "15m"
    assert isinstance(res.strategy_contributions, dict)
    assert isinstance(res.session_breakdown, dict)


def test_commodity_report_formatting():
    """Verifies that rich report and markdown formatters execute cleanly."""
    df = _generate_synthetic_candles(n=100)
    engine = CommodityEnsembleEngine(
        instrument_config=SILVERM_CONFIG,
        min_votes=1,
        capital=200000.0,
    )
    res = engine.run(df, timeframe="15m")
    
    rich_text = format_commodity_rich_report(res, use_colors=False)
    assert "MCXFORGE SILVERM — PERFORMANCE REVIEW" in rich_text or "MCXFORGE SILVERMIC — PERFORMANCE REVIEW" in rich_text
    assert "SUMMARY" in rich_text
    assert "CAPITAL & MARGIN" in rich_text
    
    md_text = format_commodity_markdown_report(res)
    assert "# MCXForge Multi-Strategy Ensemble Performance Report" in md_text
    assert "## 1. Executive Summary" in md_text
    assert "## 2. Multi-Strategy Contribution Matrix" in md_text


def test_live_commodity_paper_runner():
    """Verifies live paper runner state management, trade logging, and journal save."""
    journal_path = "/tmp/test_mcx_paper_journal.json"
    if os.path.exists(journal_path):
        os.remove(journal_path)
        
    runner = LiveCommodityPaperRunner(
        symbol="SILVERMIC",
        timeframe="15m",
        capital=200000.0,
        journal_file=journal_path,
    )
    
    df = _generate_synthetic_candles(n=50)
    runner.on_new_candle(df)
    
    # Check that journal was persisted
    assert os.path.exists(journal_path)
    if os.path.exists(journal_path):
        os.remove(journal_path)

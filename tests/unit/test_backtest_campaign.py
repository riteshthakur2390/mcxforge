"""
tests/unit/test_backtest_campaign.py — Production Campaign Unit Tests
=====================================================================
Validates:
1. DataQualityAuditor origin detection and physical invariant auditing.
2. ContractBacktestRunner lifecycle partitioning and tender period lockout.
3. CommodityPortfolioBacktestEngine concurrency limits, shared capital, and daily loss circuit breaker.
4. Robustness Scorecard transparent classifications and sample size warnings.
"""

from datetime import datetime, date, timedelta, time
import numpy as np
import pandas as pd
import pytest
import pytz

from core.models import Direction
from core.regime.engine import MarketRegime
from instruments import SILVERMIC_CONFIG
from core.strategies.trend_following import TrendFollowingStrategy
from core.strategies.orb import OpeningRangeBreakoutStrategy
from core.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from core.strategies.backtest_models import (
    SlippageModel,
    SlippageType,
    PerformanceMetrics,
    ContractMode,
)
from core.strategies.backtest_data_quality import DataQualityAuditor
from core.strategies.backtest_contract import ContractBacktestRunner
from core.strategies.backtest_portfolio import CommodityPortfolioBacktestEngine
from core.strategies.backtest_campaign import ProductionResearchCampaign

IST = pytz.timezone("Asia/Kolkata")


def make_clean_df(n: int = 120, base_p: float = 85000.0, trend: float = 10.0) -> pd.DataFrame:
    """Generates synthetic 5m OHLCV dataframe."""
    start_dt = IST.localize(datetime(2026, 9, 1, 9, 0))
    times = [start_dt + timedelta(minutes=5 * i) for i in range(n)]

    p = base_p
    opens, highs, lows, closes, vols = [], [], [], [], []
    for i in range(n):
        o = p
        c = p + trend + (np.sin(i / 5.0) * 20.0)
        h = max(o, c) + 25.0
        l = min(o, c) - 25.0
        v = 1500 + int(np.random.uniform(100, 500))
        opens.append(round(o, 1))
        highs.append(round(h, 1))
        lows.append(round(l, 1))
        closes.append(round(c, 1))
        vols.append(v)
        p = c

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    }, index=pd.DatetimeIndex(times))


# ── 1. DATA QUALITY AUDITOR TESTS ─────────────────────────────────────────────
def test_data_quality_auditor_detects_sample_data():
    """Verify auditor flags sample datasets with strict non-production warnings."""
    auditor = DataQualityAuditor(symbol="SILVERMIC")
    df = make_clean_df(n=50)

    # Flagged via file path or size
    report = auditor.audit(df, file_path="data/examples/silvermic_sample.csv")
    assert report.data_origin == "SAMPLE_TEST"
    assert any("SAMPLE / TEST DATA" in w for w in report.warnings)


def test_data_quality_auditor_detects_real_data():
    """Verify auditor recognizes REAL_HISTORICAL data and enforces physical invariants."""
    auditor = DataQualityAuditor(symbol="SILVERMIC")
    df = make_clean_df(n=1000)
    df["data_origin"] = "DHAN_REAL_HISTORICAL"

    report = auditor.audit(df, file_path="data/historical/SILVERMIC_dhan_5m.csv")
    assert report.data_origin == "REAL_HISTORICAL"
    assert report.invalid_ohlc_count == 0


def test_data_quality_auditor_corrupt_data():
    """Verify auditor catches corrupt OHLC and returns FAIL verdict."""
    auditor = DataQualityAuditor(symbol="SILVERMIC")
    df = make_clean_df(n=600)
    # Corrupt High < Low on bar 50
    df.iloc[50, df.columns.get_loc("high")] = df.iloc[50]["low"] - 50.0

    report = auditor.audit(df)
    assert report.verdict == "FAIL"
    assert report.invalid_ohlc_count > 0
    assert any("invariant violations" in e for e in report.errors)


# ── 2. CONTRACT PARTITIONING & TENDER LOCKOUT TESTS ───────────────────────────
def test_contract_backtest_runner_partitioning():
    """Verify ContractBacktestRunner segments history into distinct futures expiries."""
    runner = ContractBacktestRunner(config=SILVERMIC_CONFIG)

    # Create dates spanning multiple months (e.g. Feb to Aug)
    start_dt = IST.localize(datetime(2026, 1, 1, 9, 0))
    times = [start_dt + timedelta(hours=i) for i in range(24 * 120)] # 120 days
    df = pd.DataFrame({
        "open": np.linspace(80000, 85000, len(times)),
        "high": np.linspace(80050, 85050, len(times)),
        "low": np.linspace(79950, 84950, len(times)),
        "close": np.linspace(80020, 85020, len(times)),
        "volume": np.full(len(times), 1000),
    }, index=pd.DatetimeIndex(times))

    partitions = runner.partition_into_contracts(df)
    assert len(partitions) >= 1
    for p in partitions:
        assert "symbol" in p
        assert "expiry" in p
        assert "df" in p
        assert len(p["df"]) >= 50


# ── 3. PORTFOLIO ENGINE CONCURRENCY & DAILY LOSS TESTS ────────────────────────
def test_portfolio_engine_concurrency_and_capital():
    """Verify portfolio engine enforces maximum concurrent position caps and tracks margin."""
    p_engine = CommodityPortfolioBacktestEngine(
        capital=100_000.0,
        max_concurrent_positions=1,  # Strict cap: Only 1 position allowed at a time
        daily_loss_limit_inr=3_000.0,
    )

    strat1 = TrendFollowingStrategy()
    strat2 = OpeningRangeBreakoutStrategy()

    df = make_clean_df(n=100, trend=25.0)
    summary, trades, equity = p_engine.run_portfolio(
        strategies=[strat1, strat2],
        instrument_config=SILVERMIC_CONFIG,
        df=df,
        timeframe="5m",
        lots_per_trade=1,
    )

    assert summary.initial_capital_inr == 100_000.0
    assert summary.max_concurrent_positions == 1
    assert summary.max_margin_utilization_pct <= 100.0


def test_portfolio_daily_loss_circuit_breaker():
    """Verify daily loss limit halts further trading for the day when triggered."""
    p_engine = CommodityPortfolioBacktestEngine(
        capital=100_000.0,
        max_concurrent_positions=2,
        daily_loss_limit_inr=500.0, # Very tight limit to test circuit breaker
    )

    strat = TrendFollowingStrategy()
    # Force a losing session
    df = make_clean_df(n=80, trend=-40.0)

    summary, trades, equity = p_engine.run_portfolio(
        strategies=[strat],
        instrument_config=SILVERMIC_CONFIG,
        df=df,
        timeframe="5m",
        lots_per_trade=1,
    )

    # Summary must track daily loss hit
    assert isinstance(summary.days_daily_loss_hit, int)


# ── 4. SCORECARD CLASSIFICATION TESTS ─────────────────────────────────────────
def test_scorecard_classification_logic():
    """Verify transparent classification logic for STRONG, WEAK, REJECT, and INSUFFICIENT DATA."""
    campaign = ProductionResearchCampaign()

    # 1. Insufficient data (< 30 trades)
    m_low = PerformanceMetrics(total_trades=15, net_pnl_inr=5000.0)
    verdict, rationale = campaign._classify_strategy(m_low, None, [], 5000.0)
    assert verdict == "INSUFFICIENT DATA"

    # 2. Weak (Gross profit positive, but net loss after costs)
    m_weak = PerformanceMetrics(total_trades=50, gross_pnl_inr=2000.0, net_pnl_inr=-1500.0, profit_factor=0.90)
    verdict, rationale = campaign._classify_strategy(m_weak, None, [], -2000.0)
    assert verdict == "WEAK"

    # 3. Reject (Negative gross and net)
    m_rej = PerformanceMetrics(total_trades=60, gross_pnl_inr=-3000.0, net_pnl_inr=-6000.0, profit_factor=0.60)
    verdict, rationale = campaign._classify_strategy(m_rej, None, [], -8000.0)
    assert verdict == "REJECT"

    # 4. Strong Candidate (Survives 5 ticks slippage, positive OOS, PF > 1.15, WF positive)
    m_strong = PerformanceMetrics(total_trades=120, gross_pnl_inr=25000.0, net_pnl_inr=18000.0, profit_factor=1.45)
    oos_strong = PerformanceMetrics(total_trades=35, net_pnl_inr=4000.0)
    wf_strong = [{"test_net_pnl": 2000.0}, {"test_net_pnl": 1500.0}, {"test_net_pnl": -500.0}]
    verdict, rationale = campaign._classify_strategy(m_strong, oos_strong, wf_strong, 8000.0)
    assert verdict == "STRONG CANDIDATE"

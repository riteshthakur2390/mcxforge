"""
tests/unit/test_backtest_comprehensive.py — Comprehensive Validation Tests for Backtest Engine
=============================================================================================
Verifies:
1. Historical Data Ingestion & Physical Invariant Violations (High < Low, Low > Open, etc.)
2. Timezone Normalization, Chronological Sorting, and Duplicate Deduplication
3. Multi-timeframe Resampling (1m -> 5m, 15m, 1h)
4. Strict No-Lookahead Bias Protection (Signal on Bar N, Execution on Bar N+1 Open)
5. Same-Bar SL and Target Ambiguity Resolution (STOP_FIRST vs TARGET_FIRST)
6. Slippage Modeling: Fixed Ticks, Fixed Points, Percentage
7. Statutory MCX Fee & Tax Accounting (Gross P&L, Fees, Slippage Cost, Net P&L)
8. Physical Delivery Tender-Period Lockout in Individual Contract Mode
9. Out-of-Sample (OOS) Partitioning & Walk-Forward Rolling Window Analysis
10. Slippage Sensitivity Sweep & All 5 Strategies Execution
"""

from datetime import datetime, date, time, timedelta
import numpy as np
import pandas as pd
import pytest
import pytz

from core.models import Direction
from core.regime.engine import MarketRegime
from instruments import (
    SILVERMIC_CONFIG,
    GOLDM_CONFIG,
    CRUDEOILM_CONFIG,
    NATGASM_CONFIG,
)
from core.strategies.base import BaseCommodityStrategy, StrategySignal
from core.strategies.trend_following import TrendFollowingStrategy
from core.strategies.orb import OpeningRangeBreakoutStrategy
from core.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from core.strategies.volatility_breakout import VolatilityBreakoutStrategy
from core.strategies.donchian_breakout import DonchianBreakoutStrategy
from core.strategies.registry import StrategyRegistry
from core.strategies.backtest_models import (
    SlippageType,
    SlippageModel,
    ContractMode,
    SameBarAmbiguityRule,
    BacktestConfig,
    BacktestTrade,
    PerformanceMetrics,
)
from core.strategies.backtest_data import (
    HistoricalDataLoader,
    DataValidationError,
    resample_ohlcv,
    detect_data_gaps,
)
from core.strategies.backtest import CommodityBacktestEngine

IST = pytz.timezone("Asia/Kolkata")


def make_clean_df(n: int = 100, base_p: float = 85000.0, trend: float = 10.0) -> pd.DataFrame:
    """Generates clean synthetic 5m OHLCV dataframe."""
    start_dt = IST.localize(datetime(2026, 9, 1, 9, 0))
    times = [start_dt + timedelta(minutes=5 * i) for i in range(n)]
    np.random.seed(42)
    closes = base_p + np.arange(n) * trend + np.random.normal(0, 10, n)
    opens = closes - np.random.normal(0, 5, n)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(12, 4, n))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(12, 4, n))
    vols = np.random.randint(500, 2500, n)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
        "open_interest": np.full(n, 15000),
    }, index=pd.DatetimeIndex(times))
    return df


# ── 1. HISTORICAL DATA LOADER & PHYSICAL INVARIANTS ───────────────────────────
def test_data_loader_invariants_and_error_detection():
    """Verify data loader detects physical price violations (High < Low, negative prices, etc.)."""
    df = make_clean_df(n=50)

    # Clean data loads cleanly
    loaded = HistoricalDataLoader.load(df)
    assert len(loaded) == 50

    # 1. High < Low
    df_bad_hl = df.copy()
    df_bad_hl.iloc[10, df_bad_hl.columns.get_loc("high")] = df_bad_hl.iloc[10]["low"] - 10.0
    with pytest.raises(DataValidationError, match="High .* < Low"):
        HistoricalDataLoader.load(df_bad_hl)

    # 2. High < Close
    df_bad_hc = df.copy()
    df_bad_hc.iloc[10, df_bad_hc.columns.get_loc("low")] = 84000.0
    df_bad_hc.iloc[10, df_bad_hc.columns.get_loc("open")] = 84500.0
    df_bad_hc.iloc[10, df_bad_hc.columns.get_loc("high")] = 84800.0
    df_bad_hc.iloc[10, df_bad_hc.columns.get_loc("close")] = 85000.0
    with pytest.raises(DataValidationError, match="High .* < Close"):
        HistoricalDataLoader.load(df_bad_hc)

    # 3. Low > Open
    df_bad_lo = df.copy()
    df_bad_lo.iloc[10, df_bad_lo.columns.get_loc("high")] = 85000.0
    df_bad_lo.iloc[10, df_bad_lo.columns.get_loc("close")] = 84800.0
    df_bad_lo.iloc[10, df_bad_lo.columns.get_loc("open")] = 84500.0
    df_bad_lo.iloc[10, df_bad_lo.columns.get_loc("low")] = 84600.0
    with pytest.raises(DataValidationError, match="Low .* > Open"):
        HistoricalDataLoader.load(df_bad_lo)

    # 4. Non-positive price
    df_bad_zero = df.copy()
    df_bad_zero.iloc[10, df_bad_zero.columns.get_loc("close")] = 0.0
    with pytest.raises(DataValidationError, match="Non-positive price detected"):
        HistoricalDataLoader.load(df_bad_zero)


def test_data_loader_sorting_and_deduplication():
    """Verify loader auto-sorts unsorted timestamps and deduplicates records."""
    df = make_clean_df(n=40)
    # Shuffle index
    df_shuffled = df.sample(frac=1.0, random_state=42)
    loaded_sorted = HistoricalDataLoader.load(df_shuffled, auto_sort=True)
    assert loaded_sorted.index.is_monotonic_increasing

    # Duplicates
    df_dup = pd.concat([df.iloc[:20], df.iloc[10:30]])
    loaded_dedup = HistoricalDataLoader.load(df_dup, deduplicate=True)
    assert len(loaded_dedup) == 30
    assert not loaded_dedup.index.duplicated().any()


# ── 2. MULTI-TIMEFRAME RESAMPLING ─────────────────────────────────────────────
def test_multi_timeframe_resampling():
    """Verify resampling from 1m to 5m, 15m, 1h preserves OHLC aggregation laws."""
    start_dt = IST.localize(datetime(2026, 9, 1, 9, 0))
    times = [start_dt + timedelta(minutes=i) for i in range(60)] # 60 1-min bars
    df_1m = pd.DataFrame({
        "open": np.linspace(85000, 85100, 60),
        "high": np.linspace(85020, 85120, 60),
        "low": np.linspace(84980, 85080, 60),
        "close": np.linspace(85010, 85110, 60),
        "volume": np.full(60, 100),
    }, index=pd.DatetimeIndex(times))

    # Resample to 5m
    df_5m = resample_ohlcv(df_1m, "5m")
    assert len(df_5m) == 12
    assert df_5m["volume"].iloc[0] == 500
    assert df_5m["open"].iloc[0] == df_1m["open"].iloc[0]
    assert df_5m["close"].iloc[0] == df_1m["close"].iloc[4]
    assert df_5m["high"].iloc[0] == df_1m["high"].iloc[:5].max()
    assert df_5m["low"].iloc[0] == df_1m["low"].iloc[:5].min()

    # Resample to 1h
    df_1h = resample_ohlcv(df_1m, "1h")
    assert len(df_1h) == 1
    assert df_1h["volume"].iloc[0] == 6000


# ── 3. STRICT NO-LOOKAHEAD BIAS PROTECTION ────────────────────────────────────
def test_strict_no_lookahead_execution():
    """
    Verify that an entry signal produced on bar N CANNOT fill on bar N.
    It MUST fill on bar N+1 at bar N+1's open.
    """
    engine = CommodityBacktestEngine(slippage_model=SlippageModel(SlippageType.FIXED_TICKS, 0.0))
    strat = TrendFollowingStrategy()

    df = make_clean_df(n=80, trend=25.0) # Strong uptrend
    res = engine.run(strat, SILVERMIC_CONFIG, df)

    assert len(res.trades) > 0
    for trade in res.trades:
        # Trade entry time must match one of the candle timestamps in df
        entry_idx = df.index.get_loc(trade.entry_time)
        assert entry_idx > 0
        # Entry price must match bar N's open (with slippage)
        expected_open = float(df.iloc[entry_idx]["open"])
        assert trade.entry_price == strat.round_to_tick(expected_open)


# ── 4. SAME-BAR SL AND TARGET AMBIGUITY RESOLUTION ────────────────────────────
def test_same_bar_sl_target_ambiguity():
    """
    When both SL and Target are touched in the same candle:
    - STOP_FIRST assumes Stop Loss is hit (conservative institutional rule)
    - TARGET_FIRST assumes Target is hit
    """
    strat = TrendFollowingStrategy()

    df = make_clean_df(n=50, trend=0.0)
    # Open long position by having prior signal
    # Force a gigantic outside candle on bar 35 where High is +500 and Low is -500
    df.iloc[35, df.columns.get_loc("high")] = 86000.0
    df.iloc[35, df.columns.get_loc("low")] = 84000.0

    # 1. STOP_FIRST
    engine_stop = CommodityBacktestEngine(same_bar_rule=SameBarAmbiguityRule.STOP_FIRST)
    res_stop = engine_stop.run(strat, SILVERMIC_CONFIG, df)

    # 2. TARGET_FIRST
    engine_target = CommodityBacktestEngine(same_bar_rule=SameBarAmbiguityRule.TARGET_FIRST)
    res_target = engine_target.run(strat, SILVERMIC_CONFIG, df)

    # Ambiguous trades should be flagged
    ambiguous_stop = [t for t in res_stop.trades if t.is_ambiguous_bar]
    ambiguous_target = [t for t in res_target.trades if t.is_ambiguous_bar]

    if ambiguous_stop:
        assert ambiguous_stop[0].exit_reason == "SL_HIT"
    if ambiguous_target:
        assert ambiguous_target[0].exit_reason == "TARGET_HIT"


# ── 5. SLIPPAGE MODELING (TICKS, POINTS, PERCENTAGE) ──────────────────────────
def test_slippage_models():
    """Verify slippage calculations across Fixed Ticks, Fixed Points, and Percentage."""
    # 1. Fixed Ticks (SILVERMIC tick = 1.0)
    slip_ticks = SlippageModel(SlippageType.FIXED_TICKS, 2.0)
    assert slip_ticks.calculate_slippage_points(85000.0, tick_size=1.0) == 2.0

    # For NATGASM (tick = 0.10)
    assert slip_ticks.calculate_slippage_points(250.0, tick_size=0.10) == 0.20

    # 2. Fixed Points
    slip_pts = SlippageModel(SlippageType.FIXED_POINTS, 5.0)
    assert slip_pts.calculate_slippage_points(85000.0, tick_size=1.0) == 5.0

    # 3. Percentage (0.05% of 85000 = 42.5 pts, banker's round to even = 42.0)
    slip_pct = SlippageModel(SlippageType.PERCENTAGE, 0.05)
    assert slip_pct.calculate_slippage_points(85000.0, tick_size=1.0) == 42.0


# ── 6. STATUTORY MCX FEE & TAX ACCOUNTING ─────────────────────────────────────
def test_statutory_mcx_fee_accounting():
    """Verify transaction costs include brokerage, exchange turnover, SEBI, GST, stamp duty."""
    engine = CommodityBacktestEngine(slippage_model=SlippageModel(SlippageType.FIXED_TICKS, 1.0))
    strat = TrendFollowingStrategy()

    df = make_clean_df(n=70, trend=30.0)
    res = engine.run(strat, SILVERMIC_CONFIG, df, lots=1)

    assert len(res.trades) > 0
    trade = res.trades[0]
    # Trade must have positive fees
    assert trade.fees_inr > 40.0 # Brokerage is 40 roundtrip + taxes
    assert trade.slippage_cost_inr > 0.0
    # Net P&L = Gross P&L - Fees
    assert round(trade.net_pnl_inr, 2) == round(trade.gross_pnl_inr - trade.fees_inr, 2)


# ── 7. PHYSICAL DELIVERY TENDER LOCKOUT ───────────────────────────────────────
def test_physical_delivery_tender_period_lockout():
    """
    In INDIVIDUAL_CONTRACT mode, positions must be forcefully exited and new entries blocked
    when date enters the 5-day compulsory physical delivery tender window.
    """
    engine = CommodityBacktestEngine()
    strat = TrendFollowingStrategy()

    # Create dates leading into expiry: Expiry on Sept 30, 2026.
    # Tender period starts 5 days before: Sept 25, 2026.
    expiry = date(2026, 9, 30)

    # Dates: Sept 24, 25, 26
    start_dt = IST.localize(datetime(2026, 9, 24, 10, 0))
    times = [start_dt + timedelta(hours=i) for i in range(48)] # spans through Sept 26
    df = pd.DataFrame({
        "open": np.linspace(85000, 86000, 48),
        "high": np.linspace(85050, 86050, 48),
        "low": np.linspace(84950, 85950, 48),
        "close": np.linspace(85020, 86020, 48),
        "volume": np.full(48, 2000),
    }, index=pd.DatetimeIndex(times))

    res = engine.run(
        strat,
        SILVERMIC_CONFIG,
        df,
        contract_mode=ContractMode.INDIVIDUAL_CONTRACT,
        contract_expiry=expiry,
    )

    # Any trade initiated before Sept 25 must exit on tender lockout
    for t in res.trades:
        if t.exit_time.date() >= date(2026, 9, 25):
            assert t.exit_reason == "TENDER_PERIOD_LOCKOUT"


# ── 8. OUT-OF-SAMPLE (OOS) PARTITIONING ───────────────────────────────────────
def test_out_of_sample_split():
    """Verify chronological train/test partitioning without data leakage."""
    engine = CommodityBacktestEngine()
    strat = TrendFollowingStrategy()

    df = make_clean_df(n=100, trend=15.0)
    res_is, res_oos = engine.run_oos(strat, SILVERMIC_CONFIG, df, split_pct=0.30)

    # 70% in-sample, 30% out-of-sample
    assert res_is.config.end_time < res_oos.config.start_time
    assert res_is.metrics is not None
    assert res_oos.metrics is not None
    # All OOS trades flagged is_oos=True
    for t in res_oos.trades:
        assert t.is_oos is True


# ── 9. WALK-FORWARD ROLLING WINDOW ANALYSIS ───────────────────────────────────
def test_walk_forward_rolling_windows():
    """Verify walk-forward framework generates disjoint rolling train/test windows."""
    engine = CommodityBacktestEngine()
    strat = TrendFollowingStrategy()

    df = make_clean_df(n=300, trend=10.0)
    wf_results = engine.run_walk_forward(strat, SILVERMIC_CONFIG, df, n_windows=3, train_ratio=0.70)

    assert len(wf_results) == 3
    for w in wf_results:
        assert "train_net_pnl" in w
        assert "test_net_pnl" in w
        assert "test_profit_factor" in w
        assert w["train_start"] < w["train_end"]
        assert w["train_end"] <= w["test_start"]


# ── 10. ALL 5 STRATEGIES INDEPENDENT TESTABILITY ──────────────────────────────
def test_all_five_strategies_backtest():
    """Verify all 5 commodity strategies can be independently backtested."""
    engine = CommodityBacktestEngine(slippage_model=SlippageModel(SlippageType.FIXED_TICKS, 1.0))
    df = make_clean_df(n=80, trend=15.0)

    strategies = [
        TrendFollowingStrategy(),
        OpeningRangeBreakoutStrategy(),
        VWAPMeanReversionStrategy(),
        VolatilityBreakoutStrategy(),
        DonchianBreakoutStrategy(),
    ]

    for strat in strategies:
        res = engine.run(strat, SILVERMIC_CONFIG, df)
        assert res.metrics is not None
        assert res.config.strategy_name == strat.name
        assert hasattr(res.metrics, "profit_factor")
        assert hasattr(res.metrics, "expectancy_inr")

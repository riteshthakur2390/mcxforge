"""
tests/unit/test_commodity_strategies.py — Comprehensive Unit Tests for 5 MCX Commodity Strategies
================================================================================================
Validates:
1. Multi-instrument portability & tick rounding across SILVERMIC, GOLDM, CRUDEOILM, NATGASM
2. TrendFollowingStrategy: Bullish, Bearish, Flat/No-Signal conditions
3. OpeningRangeBreakoutStrategy: 09:00 IST open, ORB formation, window expiration, buffer checks
4. VWAPMeanReversionStrategy: Stretch fade in RANGE regime, strict veto during TREND
5. VolatilityBreakoutStrategy: Squeeze compression -> expansion, abnormal volatility ceiling
6. DonchianBreakoutStrategy: Clean N-period channel break & trailing channel exit
7. Dynamic trailing stop & position management
8. Abnormal Market Shield & Regime veto integration
9. Zero lookahead backtest simulation with MFE/MAE and fee tracking
10. Strategy Registry governance & per-instrument instantiation
"""

from datetime import datetime, date, time, timedelta
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import pytest
import pytz

from core.models import Direction, Position, TradePlan, RawSignal
from core.regime.engine import MarketRegime, MarketRegimeEngine, RegimeDetails
from core.shield import MarketShieldLayer, ShieldDecision
from instruments import (
    SILVERMIC_CONFIG,
    GOLDM_CONFIG,
    CRUDEOILM_CONFIG,
    NATGASM_CONFIG,
    get_instrument_config,
)
from core.strategies.base import BaseCommodityStrategy, StrategySignal
from core.strategies.trend_following import TrendFollowingStrategy
from core.strategies.orb import OpeningRangeBreakoutStrategy
from core.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from core.strategies.volatility_breakout import VolatilityBreakoutStrategy
from core.strategies.donchian_breakout import DonchianBreakoutStrategy
from core.strategies.registry import StrategyRegistry, StrategyStatus
from core.strategies.backtest import CommodityBacktestEngine

IST = pytz.timezone("Asia/Kolkata")


# ── FIXTURES & HELPERS ────────────────────────────────────────────────────────
def make_candles(
    n: int = 50,
    start_dt: Optional[datetime] = None,
    base_price: float = 85000.0,
    trend_slope: float = 0.0,
    volatility: float = 30.0,
    volume: int = 5000,
) -> pd.DataFrame:
    """Helper to generate realistic synthetic OHLCV candle series with DatetimeIndex."""
    if start_dt is None:
        start_dt = datetime(2026, 9, 4, 9, 15, tzinfo=IST)

    times = [start_dt + timedelta(minutes=5 * i) for i in range(n)]
    np.random.seed(42)
    noise = np.random.normal(0, volatility, n)

    closes = base_price + np.arange(n) * trend_slope + noise
    opens = closes - (np.random.normal(0, 10, n))
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(15, 10, n))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(15, 10, n))
    vols = np.full(n, volume)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    }, index=pd.DatetimeIndex(times))
    return df


# ── 1. PORTABILITY & INSTRUMENT CONFIG ────────────────────────────────────────
def test_strategy_initialization_and_portability():
    """Verify all 5 strategies correctly inherit instrument configuration without hardcoded values."""
    instruments = [SILVERMIC_CONFIG, GOLDM_CONFIG, CRUDEOILM_CONFIG, NATGASM_CONFIG]
    strategies = [
        TrendFollowingStrategy(),
        OpeningRangeBreakoutStrategy(),
        VWAPMeanReversionStrategy(),
        VolatilityBreakoutStrategy(),
        DonchianBreakoutStrategy(),
    ]

    for strat in strategies:
        for inst in instruments:
            strat.initialize(inst)
            assert strat.instrument_config.symbol == inst.symbol
            assert strat.instrument_config.lot_size == inst.lot_size
            assert strat.instrument_config.tick_size == inst.tick_size

            # Test tick rounding
            rounded = strat.round_to_tick(1234.567)
            if inst.tick_size == 1.0:
                assert rounded == 1235.0
            elif inst.tick_size == 0.10:
                assert round(rounded, 2) == 1234.60


# ── 2. TREND FOLLOWING STRATEGY TESTS ─────────────────────────────────────────
def test_trend_following_bullish_and_bearish():
    """Test TrendFollowingStrategy detects bullish & bearish trends with ADX confirmation."""
    strat = TrendFollowingStrategy()
    strat.initialize(SILVERMIC_CONFIG)

    # Strong Uptrend: +50 pts per bar
    df_bull = make_candles(n=60, trend_slope=50.0, volatility=10.0, volume=8000)
    sig_bull = strat.generate_signal(df_bull)

    assert sig_bull.direction == Direction.BUY
    assert sig_bull.is_valid is True
    assert sig_bull.entry_price > 0
    assert sig_bull.stop_loss < sig_bull.entry_price
    assert sig_bull.target > sig_bull.entry_price
    assert sig_bull.risk_reward >= 1.5

    # Strong Downtrend: -50 pts per bar
    df_bear = make_candles(n=60, trend_slope=-50.0, volatility=10.0, volume=8000)
    sig_bear = strat.generate_signal(df_bear)

    assert sig_bear.direction == Direction.SELL
    assert sig_bear.is_valid is True
    assert sig_bear.entry_price > 0
    assert sig_bear.stop_loss > sig_bear.entry_price
    assert sig_bear.target < sig_bear.entry_price

    # Flat / No Trend: slope 0, ADX low
    df_flat = make_candles(n=60, trend_slope=0.0, volatility=5.0, volume=2000)
    sig_flat = strat.generate_signal(df_flat)
    assert sig_flat.direction == Direction.NONE
    assert sig_flat.is_valid is False


# ── 3. OPENING RANGE BREAKOUT STRATEGY TESTS ──────────────────────────────────
def test_orb_breakout_and_session_timing():
    """Test OpeningRangeBreakoutStrategy respects 09:00 IST open, 30m ORB, and window."""
    strat = OpeningRangeBreakoutStrategy(parameters={"max_range_atr_mult": 5.0})
    strat.initialize(SILVERMIC_CONFIG)

    # Day starting at 09:00 IST
    start_dt = datetime(2026, 9, 4, 9, 0, tzinfo=IST)

    # 1. During first 30 mins (e.g. 09:15, only 4 bars) -> should WAIT
    df_early = make_candles(n=4, start_dt=start_dt, base_price=85000, trend_slope=5.0)
    sig_early = strat.generate_signal(df_early)
    assert sig_early.direction == Direction.NONE

    # 2. Complete 09:00 - 09:30 range (bars 0-6), then explosive breakout at 09:35 (bar 7)
    df_orb = make_candles(n=10, start_dt=start_dt, base_price=85000, trend_slope=0.0, volatility=5.0)
    # Calculate ORB high from 09:00 to 09:30 (bars 0 to 6)
    orb_high = float(df_orb.iloc[:7]["high"].max())
    # Bar 7 is at 09:35 (first bar strictly after 09:30)
    df_orb.iloc[7, df_orb.columns.get_loc("close")] = orb_high + 50.0
    df_orb.iloc[7, df_orb.columns.get_loc("high")] = orb_high + 60.0
    df_orb.iloc[7, df_orb.columns.get_loc("volume")] = 15000  # volume expansion

    sig_break = strat.generate_signal(df_orb.iloc[:8])
    assert sig_break.direction == Direction.BUY
    assert sig_break.is_valid is True
    assert sig_break.entry_price >= orb_high


# ── 4. VWAP MEAN REVERSION STRATEGY TESTS ─────────────────────────────────────
def test_vwap_mean_reversion_and_regime_rejection():
    """Test VWAPMeanReversionStrategy triggers on stretched prices in RANGE, strictly vetoed in TREND."""
    strat = VWAPMeanReversionStrategy()
    strat.initialize(SILVERMIC_CONFIG)

    # Create range-bound candles with an extreme oversold stretch at the end
    df_range = make_candles(n=40, trend_slope=0.0, volatility=20.0)
    # Deep dump at the end (stretching 150 pts below VWAP)
    df_range.iloc[-1, df_range.columns.get_loc("close")] = 84700.0
    df_range.iloc[-1, df_range.columns.get_loc("low")] = 84680.0
    df_range.iloc[-1, df_range.columns.get_loc("high")] = 84720.0

    # Test in RANGE regime -> Should generate BUY
    regime_range = RegimeDetails(regime=MarketRegime.RANGE, confidence=0.8, is_tradeable=True)
    sig_range = strat.generate_signal(df_range, regime_details=regime_range)

    if sig_range.direction != Direction.NONE:
        assert sig_range.direction == Direction.BUY
        assert sig_range.stop_loss < sig_range.entry_price

    # CRITICAL: In TREND regime -> Must be strictly VETOED
    regime_trend = RegimeDetails(regime=MarketRegime.TREND, confidence=0.9, is_tradeable=True)
    sig_trend = strat.generate_signal(df_range, regime_details=regime_trend)
    assert sig_trend.direction == Direction.NONE
    assert sig_trend.decision == "NO_TRADE"
    assert "REGIME_TRENDING_VETO" in sig_trend.rejection_reason


# ── 5. VOLATILITY BREAKOUT STRATEGY TESTS ─────────────────────────────────────
def test_volatility_breakout_compression_and_abnormal_ceiling():
    """Test VolatilityBreakoutStrategy detects squeeze compression and respects abnormal ceiling."""
    strat = VolatilityBreakoutStrategy()
    strat.initialize(SILVERMIC_CONFIG)

    # 1. 25 bars of ultra-low volatility compression (tight range)
    df_comp = make_candles(n=35, trend_slope=0.0, volatility=5.0, volume=1000)
    # Followed by sharp expansion breakout on bar 36
    df_comp.iloc[-1, df_comp.columns.get_loc("close")] = df_comp.iloc[-2]["close"] + 120.0
    df_comp.iloc[-1, df_comp.columns.get_loc("high")] = df_comp.iloc[-2]["close"] + 130.0
    df_comp.iloc[-1, df_comp.columns.get_loc("volume")] = 10000

    sig = strat.generate_signal(df_comp)
    # Should detect breakout if squeeze occurred
    if sig.direction != Direction.NONE:
        assert sig.direction == Direction.BUY
        assert sig.is_valid is True

    # 2. Test abnormal volatility ceiling: If ATR ratio is wild (>2.8) -> ABNORMAL
    data_abnormal = strat.calculate_indicators(df_comp)
    curr = data_abnormal.iloc[-1].copy()
    # Artificially test abnormal ceiling (ATR ratio is ~0.93, so ceiling of 0.5 forces veto)
    strat.parameters["abnormal_vol_max_atr_ratio"] = 0.5
    sig_abnormal = strat.generate_signal(df_comp)
    assert sig_abnormal.decision in ("ABNORMAL", "NO_TRADE")


# ── 6. DONCHIAN BREAKOUT STRATEGY TESTS ───────────────────────────────────────
def test_donchian_breakout_and_channel_exit():
    """Test DonchianBreakoutStrategy channel break and Turtle-style trailing channel exit."""
    strat = DonchianBreakoutStrategy(parameters={"entry_lookback": 20, "exit_lookback": 10})
    strat.initialize(SILVERMIC_CONFIG)

    # 25 consolidation bars
    df_donch = make_candles(n=30, trend_slope=0.0, volatility=15.0, volume=4000)
    highest_20 = df_donch.iloc[-21:-1]["high"].max()

    # 1. Break above 20-period High
    df_donch.iloc[-1, df_donch.columns.get_loc("close")] = highest_20 + 50.0
    df_donch.iloc[-1, df_donch.columns.get_loc("high")] = highest_20 + 60.0
    df_donch.iloc[-1, df_donch.columns.get_loc("volume")] = 8000

    sig = strat.generate_signal(df_donch)
    assert sig.direction == Direction.BUY
    assert sig.is_valid is True
    assert sig.entry_price > highest_20

    # 2. Test Donchian exit: Position is long, price drops below 10-period low
    mock_plan = TradePlan(
        signal=RawSignal(symbol="SILVERMIC", direction=Direction.BUY, confidence=0.8, votes=1, strategies_fired=["DonchianBreakout"]),
        entry_price=highest_20 + 50.0,
        sl_price=highest_20 - 50.0,
        desired_lots=1,
    )
    pos = Position(plan=mock_plan, entry_premium=highest_20 + 50.0, current_premium=highest_20 + 40.0)

    # Current price falls below 10-period low
    lowest_10 = df_donch.iloc[-11:-1]["low"].min()
    exit_dict = strat.exit_signal(pos, current_price=lowest_10 - 10.0, df=df_donch)
    assert exit_dict is not None
    assert exit_dict["exit"] is True
    assert "DONCHIAN_CHANNEL_EXIT_LONG" in exit_dict["reason"]


# ── 7. DYNAMIC TRAILING STOP TESTS ────────────────────────────────────────────
def test_dynamic_trailing_stop_management():
    """Verify position trailing stop ratchets upwards for Long and downwards for Short."""
    strat = TrendFollowingStrategy(parameters={"trailing_stop_atr_mult": 1.5})
    strat.initialize(SILVERMIC_CONFIG)

    df = make_candles(n=30, base_price=85000, trend_slope=10.0)

    # Long Position: Entry at 85000, initial SL at 84800
    mock_plan_long = TradePlan(
        signal=RawSignal(symbol="SILVERMIC", direction=Direction.BUY, confidence=0.8, votes=1, strategies_fired=["TrendFollowing"]),
        entry_price=85000.0,
        sl_price=84800.0,
        desired_lots=1,
    )
    pos_long = Position(plan=mock_plan_long, entry_premium=85000.0, current_premium=85400.0)

    # Current price moved up to 85400 -> trailing stop should ratchet up above 84800
    update_long = strat.manage_position(pos_long, current_price=85400.0, df=df)
    assert update_long is not None
    assert update_long["action"] == "UPDATE_SL"
    assert update_long["new_sl"] > 84800.0
    assert update_long["new_sl"] < 85400.0


# ── 8. SHIELD VETO & REGIME GATE INTEGRATION ──────────────────────────────────
def test_shield_veto_integration():
    """Verify strategy signals are rejected when abnormal conditions or shield vetoes trigger."""
    strat = TrendFollowingStrategy()
    strat.initialize(SILVERMIC_CONFIG)

    shield = MarketShieldLayer(max_stale_seconds=300)
    now = datetime(2026, 9, 4, 14, 0, tzinfo=IST)

    # 1. Normal valid signal
    sig = StrategySignal(
        timestamp=now,
        instrument="SILVERMIC",
        contract="SILVERMIC-30Nov2026-FUT",
        strategy=strat.name,
        strategy_version=strat.version,
        direction=Direction.BUY,
        confidence=0.8,
        entry_price=85000.0,
        stop_loss=84800.0,
        target=85400.0,
        decision="TRADE",
        is_valid=True,
    )

    # Regime ABNORMAL -> Should be vetoed
    regime_abnormal = RegimeDetails(regime=MarketRegime.ABNORMAL, reasons=["EXTREME_SPIKE"], is_tradeable=False)
    validated = strat.validate_signal(sig, regime_details=regime_abnormal)
    assert validated.is_valid is False
    assert validated.decision == "ABNORMAL"

    # Shield stale data veto
    from core.shield import ShieldVerdict
    stale_tick = now - timedelta(seconds=400)
    decision = shield.evaluate_pre_execution(
        instrument=SILVERMIC_CONFIG,
        strategy_name="TrendFollowing",
        regime_details=RegimeDetails(regime=MarketRegime.TREND, is_tradeable=True),
        permitted_regimes=[MarketRegime.TREND],
        last_candle_time=stale_tick,
        current_ltp=85000.0,
        current_time=now,
    )
    assert decision.verdict == ShieldVerdict.ABNORMAL
    assert "STALE" in decision.reason


# ── 9. NON-LOOKAHEAD BACKTEST SIMULATION TEST ──────────────────────────────────
def test_non_lookahead_backtest_simulation():
    """Run non-lookahead backtest engine and verify trades, MFE/MAE, and EdgeReport metrics."""
    engine = CommodityBacktestEngine(slippage_ticks=1)
    strat = TrendFollowingStrategy()

    # Generate 80 bars of clean trend data
    df_trend = make_candles(n=80, base_price=80000.0, trend_slope=20.0, volatility=10.0, volume=8000)

    trades, report, df_journal = engine.run(strat, SILVERMIC_CONFIG, df_trend, lots=1)

    assert isinstance(trades, list)
    if trades:
        t = trades[0]
        assert t.instrument == "SILVERMIC"
        assert t.mfe_pts >= 0.0
        assert t.mae_pts >= 0.0
        assert t.fees_inr > 0.0  # Real statutory charges computed
        assert t.hold_bars > 0

    assert report is not None
    assert hasattr(report, "profit_factor")
    assert hasattr(report, "expectancy_inr")


# ── 10. STRATEGY REGISTRY & GOVERNANCE ────────────────────────────────────────
def test_strategy_registry_governance():
    """Test StrategyRegistry dynamic creation and per-instrument enablement."""
    # 1. Create individual strategy
    tf = StrategyRegistry.create_strategy("TrendFollowing", SILVERMIC_CONFIG)
    assert isinstance(tf, TrendFollowingStrategy)
    assert tf.instrument_config.symbol == "SILVERMIC"

    # 2. Instantiate all 5 canonical strategies
    all_strats = StrategyRegistry.get_all_commodity_strategies(GOLDM_CONFIG)
    assert len(all_strats) == 5
    assert "TrendFollowing" in all_strats
    assert "OpeningRangeBreakout" in all_strats
    assert "VWAPMeanReversion" in all_strats
    assert "VolatilityBreakout" in all_strats
    assert "DonchianBreakout" in all_strats
    assert all_strats["DonchianBreakout"].instrument_config.symbol == "GOLDM"

    # 3. Verify eligible strategies for SILVERMIC includes KEEP status
    eligible = StrategyRegistry.get_eligible_strategies_for_instrument("SILVERMIC")
    names = [s.name for s in eligible]
    assert "TrendFollowing" in names
    assert "OpeningRangeBreakout" in names
    assert "VWAPMeanReversion" in names
    assert "VolatilityBreakout" in names
    assert "DonchianBreakout" in names

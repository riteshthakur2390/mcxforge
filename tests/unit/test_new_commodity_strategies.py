"""
tests/unit/test_new_commodity_strategies.py — Unit Tests for 8 New Commodity Strategies
=======================================================================================
Verifies initialization, indicator calculations, invariant safety, and signal generation for:
1. BollingerBandMeanReversionStrategy
2. MovingAverageCrossoverStrategy
3. RSIDivergenceStrategy
4. MomentumVolumeBreakoutStrategy
5. GoldSilverPairsStrategy
6. OrderFlowDeltaStrategy
7. TimeOfDaySeasonalityStrategy
8. RSI2MeanReversionStrategy
9. Enhanced VWAPMeanReversionStrategy
10. Unified Suite Integration
"""

from datetime import datetime, timedelta, time
import numpy as np
import pandas as pd
import pytest
import pytz

from instruments import SILVERMIC_CONFIG
from core.models import Direction
from core.strategies.bb_mean_reversion import BollingerBandMeanReversionStrategy
from core.strategies.ma_crossover import MovingAverageCrossoverStrategy
from core.strategies.rsi_divergence import RSIDivergenceStrategy
from core.strategies.momentum_volume_breakout import MomentumVolumeBreakoutStrategy
from core.strategies.gold_silver_pairs import GoldSilverPairsStrategy
from core.strategies.order_flow_delta import OrderFlowDeltaStrategy
from core.strategies.time_of_day_seasonality import TimeOfDaySeasonalityStrategy
from core.strategies.rsi2_mean_reversion import RSI2MeanReversionStrategy
from core.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from core.strategies.ensemble import build_default_strategy_suite, CommodityEnsembleEngine

IST = pytz.timezone("Asia/Kolkata")


def _generate_test_candles(n: int = 150, trend: float = 1.0) -> pd.DataFrame:
    """Generates synthetic OHLCV candles with a specified trend slope."""
    start_dt = datetime(2026, 9, 1, 17, 0, tzinfo=IST)
    times = [start_dt + timedelta(minutes=15 * i) for i in range(n)]
    
    base_price = 75000.0
    prices = [base_price + (i * 20.0 * trend) + (15.0 if i % 2 == 0 else -15.0) for i in range(n)]
    
    rows = []
    for i, (t, p) in enumerate(zip(times, prices)):
        rows.append({
            "timestamp": t,
            "open": p - 5.0,
            "high": p + 25.0,
            "low": p - 20.0,
            "close": p + 10.0,
            "volume": 200 + (i % 7) * 30,
        })
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def test_bb_mean_reversion_strategy():
    strat = BollingerBandMeanReversionStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    df = _generate_test_candles(100, trend=0.0)  # Ranging
    ind = strat.calculate_indicators(df)
    assert "bb_upper" in ind.columns
    assert "bb_lower" in ind.columns
    assert "adx" in ind.columns
    sig = strat.generate_signal(df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)
    assert sig.strategy == "BBMeanReversion"


def test_ma_crossover_strategy():
    strat = MovingAverageCrossoverStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    df = _generate_test_candles(100, trend=2.0)  # Strong uptrend
    ind = strat.calculate_indicators(df)
    assert "ema_fast" in ind.columns
    assert "ema_slow" in ind.columns
    sig = strat.generate_signal(df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)


def test_rsi_divergence_strategy():
    strat = RSIDivergenceStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    df = _generate_test_candles(120, trend=0.2)
    ind = strat.calculate_indicators(df)
    assert "rsi" in ind.columns
    assert "atr" in ind.columns
    sig = strat.generate_signal(df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)


def test_momentum_volume_breakout_strategy():
    strat = MomentumVolumeBreakoutStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    df = _generate_test_candles(100, trend=3.0)
    # Inject volume spike on last bar
    df.iloc[-1, df.columns.get_loc("volume")] = 2500
    ind = strat.calculate_indicators(df)
    assert "channel_high" in ind.columns
    assert "macd_hist" in ind.columns
    sig = strat.generate_signal(df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)


def test_gold_silver_pairs_strategy():
    strat = GoldSilverPairsStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    silver_df = _generate_test_candles(100, trend=1.0)
    gold_df = _generate_test_candles(100, trend=1.5)
    strat.set_gold_data(gold_df)
    ind = strat.calculate_indicators(silver_df)
    assert "zscore" in ind.columns
    sig = strat.generate_signal(silver_df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)


def test_order_flow_delta_strategy():
    strat = OrderFlowDeltaStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    df = _generate_test_candles(100, trend=1.5)
    ind = strat.calculate_indicators(df)
    assert "bar_delta" in ind.columns
    assert "cvd_zscore" in ind.columns
    sig = strat.generate_signal(df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)


def test_time_of_day_seasonality_strategy():
    strat = TimeOfDaySeasonalityStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    # Candles starting at 17:00 (inside evening window)
    df = _generate_test_candles(100, trend=1.0)
    sig = strat.generate_signal(df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)


def test_rsi2_mean_reversion_strategy():
    strat = RSI2MeanReversionStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    df = _generate_test_candles(100, trend=0.5)
    ind = strat.calculate_indicators(df)
    assert "rsi2" in ind.columns
    assert "sma_trend" in ind.columns
    sig = strat.generate_signal(df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)


def test_enhanced_vwap_mean_reversion():
    strat = VWAPMeanReversionStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    df = _generate_test_candles(100, trend=0.0)
    ind = strat.calculate_indicators(df)
    assert "vwap_pct" in ind.columns
    assert "volume_sma" in ind.columns
    sig = strat.generate_signal(df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)


def test_calendar_seasonality_strategy():
    from core.strategies.calendar_seasonality import CalendarSeasonalityStrategy
    strat = CalendarSeasonalityStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    df = _generate_test_candles(100, trend=1.0)
    sig = strat.generate_signal(df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)
    assert sig.strategy == "CalendarSeasonality"


def test_term_structure_strategy():
    from core.strategies.term_structure import TermStructureStrategy
    strat = TermStructureStrategy()
    strat.initialize(SILVERMIC_CONFIG)
    near_df = _generate_test_candles(100, trend=1.0)
    far_df = _generate_test_candles(100, trend=0.8)
    strat.set_far_month_data(far_df)
    ind = strat.calculate_indicators(near_df)
    assert "calendar_spread" in ind.columns
    sig = strat.generate_signal(near_df)
    assert sig.direction in (Direction.BUY, Direction.SELL, Direction.NONE)
    assert sig.strategy == "TermStructure"


def test_currency_macro_filter():
    from core.strategies.currency_macro_filter import CurrencyMacroFilter, MacroVolatilityLevel
    filter_engine = CurrencyMacroFilter()
    # Normal move
    norm = filter_engine.evaluate(usdinr_change_pct=0.15)
    assert norm.level == MacroVolatilityLevel.NORMAL
    assert norm.position_size_multiplier == 1.0
    assert norm.is_safe_to_trade is True

    # Elevated move (> 0.5%) -> cut position size by half
    elev = filter_engine.evaluate(usdinr_change_pct=0.65)
    assert elev.level == MacroVolatilityLevel.ELEVATED
    assert elev.position_size_multiplier == 0.50
    assert elev.is_safe_to_trade is True

    # Extreme move (> 1.2%)
    extr = filter_engine.evaluate(usdinr_change_pct=-1.35)
    assert extr.level == MacroVolatilityLevel.EXTREME
    assert extr.position_size_multiplier == 0.25
    assert extr.is_safe_to_trade is False


def test_expanded_suite_ensemble_execution():
    suite = build_default_strategy_suite()
    # Ensure all new strategies are loaded
    names = [s.name for s in suite]
    assert "BBMeanReversion" in names
    assert "MACrossover" in names
    assert "RSIDivergence" in names
    assert "MomentumVolumeBreakout" in names
    assert "GoldSilverPairs" in names
    assert "OrderFlowDelta" in names
    assert "TimeOfDaySeasonality" in names
    assert "RSI2MeanReversion" in names
    assert "CalendarSeasonality" in names
    assert "TermStructure" in names

    df = _generate_test_candles(120, trend=1.0)
    engine = CommodityEnsembleEngine(strategies=suite, min_votes=1)
    res = engine.run(df, timeframe="15m")
    assert res.total_days >= 1
    assert isinstance(res.strategy_contributions, dict)

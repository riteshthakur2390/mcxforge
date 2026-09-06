"""
tests/test_rolling_defects_fixes.py

Comprehensive regression test suite for the 4 rolling-context defects:
1. CPR (S8): True completed previous session OHLC, gap handling, truncation rejection.
2. GammaExposure (S32): Fail-closed on missing GEX data, zero synthetic proxy signals.
3. Ichimoku (S9): Mathematical alignment of 26-bar displacement and causal projection.
4. ElliottWave (S34): Confirmed pivots only, zero lookahead or tail-candle false reversals.
"""

import os
import sys
import datetime
import pytest
import pandas as pd
import numpy as np
import pytz

IST = pytz.timezone("Asia/Kolkata")
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from core.models import Direction
from agents_code.agent2_strategy.s8_cpr import CPRStrategy, CPRLevels
from agents_code.agent2_strategy.s32_gamma_exposure import GammaExposureStrategy
from agents_code.agent2_strategy.s9_ichimoku import IchimokuStrategy
from agents_code.agent2_strategy.s34_elliott_wave import ElliottWaveStrategy, Pivot


# ─────────────────────────────────────────────────────────────────────────────
# 1. CPR TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_cpr_normal_day_and_monday():
    """Verify CPR identifies Friday for Monday, and previous day for normal trading days."""
    strat = CPRStrategy(db_path="non_existent.sqlite3")
    strat._daily_cache = {
        datetime.date(2024, 1, 11): (21725.25, 21594.00, 21665.55),
        datetime.date(2024, 1, 12): (21927.20, 21718.75, 21908.00), # Friday
        datetime.date(2024, 1, 15): (22115.55, 22001.00, 22097.45), # Monday
        datetime.date(2024, 1, 16): (22124.15, 21969.80, 22032.30), # Tuesday
    }

    # Test Monday 2024-01-15 -> must pick Friday 2024-01-12
    idx_mon = pd.date_range("2024-01-15 09:15", periods=20, freq="5min", tz=IST)
    df_mon = pd.DataFrame({"high": [22050]*20, "low": [22000]*20, "close": [22020]*20, "volume": [1000]*20}, index=idx_mon)
    cpr_mon = strat._compute_cpr(df_mon)
    assert cpr_mon is not None
    # Expected pivot from Friday: (21927.20 + 21718.75 + 21908.00) / 3 = 21851.32
    expected_pivot = round((21927.20 + 21718.75 + 21908.00) / 3, 2)
    assert abs(cpr_mon.pivot - expected_pivot) < 0.05

    # Test Tuesday 2024-01-16 -> must pick Monday 2024-01-15
    idx_tue = pd.date_range("2024-01-16 09:15", periods=20, freq="5min", tz=IST)
    df_tue = pd.DataFrame({"high": [22100]*20, "low": [22000]*20, "close": [22050]*20, "volume": [1000]*20}, index=idx_tue)
    cpr_tue = strat._compute_cpr(df_tue)
    assert cpr_tue is not None
    expected_pivot_tue = round((22115.55 + 22001.00 + 22097.45) / 3, 2)
    assert abs(cpr_tue.pivot - expected_pivot_tue) < 0.05


def test_cpr_holiday_gap_and_expiry():
    """Verify CPR handles holiday multi-day gaps and expiry day transitions."""
    strat = CPRStrategy(db_path="non_existent.sqlite3")
    strat._daily_cache = {
        datetime.date(2024, 1, 19): (21670.0, 21500.0, 21622.0), # Friday
        # Jan 22 was Ram Mandir holiday
        datetime.date(2024, 1, 23): (21750.0, 21550.0, 21600.0), # Tuesday
        datetime.date(2024, 1, 24): (21500.0, 21300.0, 21450.0), # Wednesday
        datetime.date(2024, 1, 25): (21480.0, 21250.0, 21350.0), # Thursday (Expiry)
    }

    # Tuesday Jan 23 after holiday -> must pick Friday Jan 19
    idx_gap = pd.date_range("2024-01-23 09:15", periods=20, freq="5min", tz=IST)
    df_gap = pd.DataFrame({"high": [21600]*20, "low": [21500]*20, "close": [21550]*20, "volume": [1000]*20}, index=idx_gap)
    cpr_gap = strat._compute_cpr(df_gap)
    assert cpr_gap is not None
    expected_gap_pivot = round((21670.0 + 21500.0 + 21622.0) / 3, 2)
    assert abs(cpr_gap.pivot - expected_gap_pivot) < 0.05

    # Expiry Thursday Jan 25 -> must pick Wednesday Jan 24
    idx_exp = pd.date_range("2024-01-25 09:15", periods=20, freq="5min", tz=IST)
    df_exp = pd.DataFrame({"high": [21400]*20, "low": [21300]*20, "close": [21350]*20, "volume": [1000]*20}, index=idx_exp)
    cpr_exp = strat._compute_cpr(df_exp)
    assert cpr_exp is not None
    expected_exp_pivot = round((21500.0 + 21300.0 + 21450.0) / 3, 2)
    assert abs(cpr_exp.pivot - expected_exp_pivot) < 0.05


def test_cpr_rejects_truncated_rolling_slice():
    """Verify that CPR strictly rejects incomplete / truncated session slices when daily cache is absent."""
    strat = CPRStrategy(db_path="non_existent.sqlite3")
    strat._daily_cache.clear()

    # Create a truncated slice where yesterday only has 30 bars (e.g. 13:00 to 15:30)
    idx_yest = pd.date_range("2024-01-15 13:00", "2024-01-15 15:30", freq="5min", tz=IST)
    idx_today = pd.date_range("2024-01-16 09:15", "2024-01-16 11:00", freq="5min", tz=IST)
    full_idx = idx_yest.append(idx_today)

    df_truncated = pd.DataFrame({
        "high": [22000] * len(full_idx),
        "low": [21900] * len(full_idx),
        "close": [21950] * len(full_idx),
        "volume": [1000] * len(full_idx),
    }, index=full_idx)

    # Must return None (fail closed) because yesterday is truncated (< 65 bars)
    res = strat._compute_cpr(df_truncated)
    assert res is None, "CPR must reject truncated rolling slices (< 65 bars)!"


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAMMA EXPOSURE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_gamma_exposure_fails_closed_when_data_missing():
    """Verify GammaExposure fails closed (returns NONE) when GEX data is missing."""
    strat = GammaExposureStrategy()
    assert len(strat._gex_walls) == 0

    idx = pd.date_range("2026-08-28 09:15", periods=30, freq="5min", tz=IST)
    df = pd.DataFrame({
        "close": [24500.0] * 30,
        "volume": [10000.0] * 30,
    }, index=idx)

    res = strat.evaluate(df)
    assert res["direction"] == Direction.NONE
    assert res["confidence"] == 0.0


def test_gamma_exposure_evaluates_when_real_gex_provided():
    """Verify GammaExposure evaluates accurately when genuine GEX walls are updated."""
    strat = GammaExposureStrategy()
    strat.update_gex({
        24500: 5000.0, # Strong positive GEX wall at 24500
        24400: -2000.0,
    })

    idx = pd.date_range("2026-08-28 09:15", periods=30, freq="5min", tz=IST)
    # Price is approaching 24500 from below (resistance)
    df = pd.DataFrame({
        "close": list(np.linspace(24450, 24495, 30)),
        "volume": [10000.0] * 30,
    }, index=idx)

    res = strat.evaluate(df)
    assert res["direction"] == Direction.BUY_PUT
    assert res["meta"]["signal"] == "GEX_WALL_RESISTANCE"
    assert res["meta"]["nearest_strike"] == 24500

    # Reset clears state
    strat.reset()
    assert len(strat._gex_walls) == 0
    assert strat.evaluate(df)["direction"] == Direction.NONE


# ─────────────────────────────────────────────────────────────────────────────
# 3. ICHIMOKU TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_ichimoku_displacement_and_causality():
    """Verify Ichimoku cloud displacement alignment (-27) and point-in-time causal projection."""
    strat = IchimokuStrategy()
    idx = pd.date_range("2026-08-28 09:15", periods=100, freq="5min", tz=IST)
    prices = np.linspace(24000, 24500, 100)
    df = pd.DataFrame({
        "open": prices - 5,
        "high": prices + 10,
        "low": prices - 10,
        "close": prices,
        "volume": np.full(100, 50000),
    }, index=idx)

    levels = strat._compute(df)
    assert levels is not None
    # Current close
    assert levels.close == prices[-1]
    # Verify Chikou reference is exactly price 26 bars ago (index -27)
    expected_chikou_ref = float(df["close"].iloc[-27])
    assert levels.chikou_ref == expected_chikou_ref
    # Verify future cloud is projected causally
    assert levels.future_a > 0
    assert levels.future_b > 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. ELLIOTT WAVE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_elliott_wave_rejects_unconfirmed_tail():
    """Verify ElliottWave uses only confirmed pivots and rejects active unconfirmed candle tails."""
    strat = ElliottWaveStrategy()

    # Monotonically increasing data (running trend, no reversal)
    idx = pd.date_range("2026-08-28 09:15", periods=50, freq="5min", tz=IST)
    prices = np.linspace(100, 150, 50)
    df_trend = pd.DataFrame({
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1000] * 50,
    }, index=idx)

    # In a pure trend with no reversals, only the origin anchor exists (no confirmed swing turns)
    confirmed_pivots = strat._zigzag(df_trend, include_unconfirmed=False)
    assert len(confirmed_pivots) == 1, "Running trend must have only origin anchor (no confirmed swing turns)!"

    # With include_unconfirmed=True, the unfinished tail high is also force-appended
    tail_pivots = strat._zigzag(df_trend, include_unconfirmed=True)
    assert len(tail_pivots) == 2
    assert tail_pivots[-1].idx == 49
    assert tail_pivots[-1].kind == "HIGH"

    # Evaluate must return NONE (insufficient confirmed pivots for wave 5 or ABC)
    res = strat.evaluate(df_trend)
    assert res["direction"] == Direction.NONE


if __name__ == "__main__":
    pytest.main(["-v", __file__])

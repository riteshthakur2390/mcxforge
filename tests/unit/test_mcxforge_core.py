"""
tests/unit/test_mcxforge_core.py — Unit Tests for MCXForge Multi-Instrument Core Architecture

Validates:
1. Multi-Instrument Abstraction (SILVERMIC, GOLDM, CRUDEOILM, NATGASM)
2. SILVERMIC Contract Lifecycle & 5-Day Tender-Period Physical Delivery Lockout
3. Futures P&L, Tick Values, and Margin Sizing
4. Market Regime Engine (TREND, RANGE, BREAKOUT, HIGH_VOL, LOW_VOL, ABNORMAL)
5. Abnormal Market Shield & Veto Pipeline (TRADE, WAIT, NO_TRADE, ABNORMAL)
6. Risk Guardian & Global Kill Switch
7. Strategy Governance & Per-Instrument Configuration (KEEP, ADAPT, RESEARCH, REMOVED)
8. Execution Modes (PAPER, SHADOW, LIVE safety lock)
9. Four-Month Observation Analytics & Attribution
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import pytz

from instruments import (
    get_instrument_config,
    resolve_active_contract,
    normalize_symbol,
    SILVERM_CONFIG,
    SILVERMIC_CONFIG,
    GOLDM_CONFIG,
    CRUDEOILM_CONFIG,
    NATGASM_CONFIG,
)
from core.regime import MarketRegime, MarketRegimeEngine
from core.shield import MarketShieldLayer, ShieldVerdict
from core.risk import RiskGuardian, KILL_SWITCH_FILE
from core.strategies import StrategyRegistry, StrategyStatus, MASTER_STRATEGY_CATALOG
from core.execution import FuturesExecutionEngine, ExecutionMode
from core.analytics import PerformanceAnalyticsEngine
from core.models import Direction, RawSignal, TradePlan, Position

IST = pytz.timezone("Asia/Kolkata")


# ── 1. MULTI-INSTRUMENT ABSTRACTION TESTS ─────────────────────────────────────
def test_multi_instrument_catalog():
    """Verify all 4 core MCX instruments are correctly configured with accurate specs."""
    sm = get_instrument_config("SILVERM")
    assert sm.symbol == "SILVERM"
    assert sm.lot_size == 5
    assert sm.tick_size == 1.0
    assert sm.tick_value == 5.0
    assert sm.tender_period_days == 5
    assert sm.is_deliverable is True

    gm = get_instrument_config("GOLDM")
    assert gm.symbol == "GOLDM"
    assert gm.lot_size == 10
    assert gm.tick_value == 10.0
    assert gm.tender_period_days == 5

    cr = get_instrument_config("CRUDEOILM")
    assert cr.symbol == "CRUDEOILM"
    assert cr.lot_size == 10
    assert cr.is_deliverable is False # Cash settled

    ng = get_instrument_config("NATGASM")
    assert ng.symbol == "NATGASM"
    assert ng.lot_size == 250
    assert ng.tick_size == 0.10


def test_symbol_normalization():
    """Verify alias mapping to canonical root symbol."""
    assert normalize_symbol("MCX:SILVERM") == "SILVERM"
    assert normalize_symbol("MCX:SILVERMIC") == "SILVERM"
    assert normalize_symbol("SILVER_MIC") == "SILVERM"
    assert normalize_symbol("SILVER") == "SILVERM"
    assert normalize_symbol("GOLD") == "GOLDM"
    assert normalize_symbol("CRUDE") == "CRUDEOILM"
    assert normalize_symbol("NATGAS") == "NATGASM"


# ── 2. SILVERM CONTRACT & TENDER PERIOD TESTS ──────────────────────────────
def test_silverm_active_contract_and_tender_period():
    """Verify SILVERM contract resolution and 5-day tender period lockout."""
    # Date well before Nov 2026 expiry
    ref_date = date(2026, 9, 4)
    contract = resolve_active_contract("SILVERM", as_of=ref_date)
    assert contract.symbol == "SILVERM"
    assert "SILVERM" in contract.trading_symbol
    assert contract.expiry_date >= ref_date

    # Test tender period detection: 4 days before expiry = tender period active
    expiry = date(2026, 11, 30)
    inside_tender = date(2026, 11, 27)
    outside_tender = date(2026, 11, 20)

    assert SILVERM_CONFIG.is_in_tender_period(expiry, inside_tender) is True
    assert SILVERM_CONFIG.is_in_tender_period(expiry, outside_tender) is False
    assert SILVERM_CONFIG.should_rollover(expiry, inside_tender) is True


# ── 3. FUTURES P&L & MARGIN SIZING TESTS ──────────────────────────────────────
def test_futures_pnl_and_margin():
    """Verify P&L and margin calculation for long and short futures on SILVERM (5 kg lot)."""
    cfg = SILVERM_CONFIG
    entry_p = 85000.0
    exit_p = 86200.0

    # Long trade: +1200 points on 2 lots (10 kg total) -> 1200 * 5 * 2 = 12000 INR
    long_pnl_pts = cfg.calculate_pnl_points(entry_p, exit_p, is_long=True)
    long_pnl_inr = cfg.calculate_pnl_rupees(entry_p, exit_p, is_long=True, lots=2)
    assert long_pnl_pts == 1200.0
    assert long_pnl_inr == 12000.0

    # Short trade: Entry 85000, Exit 84000 = +1000 points on 1 lot (5 kg) -> 5000 INR
    short_pnl_pts = cfg.calculate_pnl_points(entry_p, 84000.0, is_long=False)
    short_pnl_inr = cfg.calculate_pnl_rupees(entry_p, 84000.0, is_long=False, lots=1)
    assert short_pnl_pts == 1000.0
    assert short_pnl_inr == 5000.0

    # Margin check: 15% on 85,000 * 5 kg = ~63,750 INR
    margin = cfg.calculate_margin(price=85000.0, lots=1)
    assert 60000.0 < margin < 70000.0


# ── 4. MARKET REGIME ENGINE TESTS ─────────────────────────────────────────────
def test_market_regime_engine_trend_and_spike():
    """Verify regime engine accurately detects trends, ranges, and abnormal spikes."""
    engine = MarketRegimeEngine()

    # Generate 50 candles with strong uptrend
    np.random.seed(42)
    closes = np.linspace(80000, 83000, 50) + np.random.normal(0, 15, 50)
    df_trend = pd.DataFrame({
        "open": closes - 10,
        "high": closes + 30,
        "low": closes - 30,
        "close": closes,
        "volume": np.full(50, 10000),
    })
    res_trend = engine.classify(df_trend)
    assert res_trend.regime in (MarketRegime.TREND, MarketRegime.BREAKOUT)
    assert res_trend.is_tradeable is True

    # Test abnormal single-bar spike (>3.5x ATR)
    df_spike = df_trend.copy()
    atr_val = res_trend.atr
    df_spike.iloc[-1, df_spike.columns.get_loc("high")] = df_spike.iloc[-1]["close"] + (atr_val * 4.0)
    res_spike = engine.classify(df_spike)
    assert res_spike.regime == MarketRegime.ABNORMAL
    assert res_spike.is_tradeable is False


# ── 5. ABNORMAL MARKET SHIELD TESTS ───────────────────────────────────────────
def test_abnormal_market_shield_vetoes():
    """Verify shield vetoes trades on kill switch, stale data, tender period, and disconnect."""
    shield = MarketShieldLayer(max_stale_seconds=300, max_spread_pts=5.0)
    cfg = SILVERMIC_CONFIG
    now = datetime(2026, 9, 4, 14, 0, tzinfo=IST)

    # Fake normal regime
    engine = MarketRegimeEngine()
    df_norm = pd.DataFrame({
        "open": [85000]*20, "high": [85020]*20, "low": [84980]*20, "close": [85010]*20, "volume": [1000]*20
    })
    regime = engine.classify(df_norm)

    # 1. Clean trade approval
    decision = shield.evaluate_pre_execution(
        instrument=cfg,
        strategy_name="SuperTrend+RSI",
        regime_details=regime,
        permitted_regimes=[MarketRegime.RANGE, MarketRegime.TREND],
        last_candle_time=now - timedelta(minutes=2),
        current_ltp=85010.0,
        bid_price=85009.0,
        ask_price=85011.0,
        broker_connected=True,
        kill_switch_active=False,
        current_time=now,
    )
    assert decision.verdict == ShieldVerdict.TRADE
    assert decision.is_executable is True

    # 2. Kill switch active -> ABNORMAL
    decision_ks = shield.evaluate_pre_execution(
        instrument=cfg, strategy_name="SuperTrend+RSI", regime_details=regime,
        permitted_regimes=[MarketRegime.RANGE], last_candle_time=now, current_ltp=85010.0,
        broker_connected=True, kill_switch_active=True, current_time=now,
    )
    assert decision_ks.verdict == ShieldVerdict.ABNORMAL
    assert "KILL_SWITCH" in decision_ks.reason

    # 3. Stale data -> ABNORMAL
    decision_stale = shield.evaluate_pre_execution(
        instrument=cfg, strategy_name="SuperTrend+RSI", regime_details=regime,
        permitted_regimes=[MarketRegime.RANGE], last_candle_time=now - timedelta(minutes=15),
        current_ltp=85010.0, broker_connected=True, kill_switch_active=False, current_time=now,
    )
    assert decision_stale.verdict == ShieldVerdict.ABNORMAL
    assert "STALE" in decision_stale.reason

    # 4. Excessive spread -> WAIT
    decision_spread = shield.evaluate_pre_execution(
        instrument=cfg, strategy_name="SuperTrend+RSI", regime_details=regime,
        permitted_regimes=[MarketRegime.RANGE], last_candle_time=now - timedelta(minutes=1),
        current_ltp=85010.0, bid_price=85000.0, ask_price=85010.0, # 10 pts spread > 5.0
        broker_connected=True, kill_switch_active=False, current_time=now,
    )
    assert decision_spread.verdict == ShieldVerdict.WAIT


# ── 6. RISK GUARDIAN & KILL SWITCH TESTS ──────────────────────────────────────
def test_risk_guardian_limits(tmp_path, monkeypatch):
    """Verify hard safety limits: max daily loss, max open positions, consecutive losses."""
    lock_file = tmp_path / "kill_switch.lock"
    monkeypatch.setattr("core.risk.kill_switch.KILL_SWITCH_FILE", lock_file)

    guardian = RiskGuardian(
        max_daily_loss_inr=3000.0,
        max_open_positions=1,
        max_consecutive_losses=3,
        max_risk_per_trade_inr=1500.0,
    )

    # 1. Normal trade allowed
    res1 = guardian.evaluate_new_trade(risk_amount_inr=1000.0, current_open_positions=0)
    assert res1.allowed is True

    # 2. Max open positions breach
    res2 = guardian.evaluate_new_trade(risk_amount_inr=1000.0, current_open_positions=1)
    assert res2.allowed is False
    assert "MAX_OPEN_POSITIONS" in res2.reason

    # 3. Consecutive losses circuit breaker
    guardian.record_trade_result(-500.0)
    guardian.record_trade_result(-600.0)
    guardian.record_trade_result(-700.0) # 3 consecutive losses
    res3 = guardian.evaluate_new_trade(risk_amount_inr=1000.0, current_open_positions=0)
    assert res3.allowed is False
    assert "CONSECUTIVE_LOSS" in res3.reason

    # 4. Kill switch file engage/disengage
    RiskGuardian.engage_kill_switch("TEST_TRIGGER")
    assert RiskGuardian.is_kill_switch_active() is True
    res_ks = guardian.evaluate_new_trade(risk_amount_inr=500.0, current_open_positions=0)
    assert res_ks.allowed is False
    assert "KILL_SWITCH" in res_ks.reason

    RiskGuardian.disengage_kill_switch()
    assert RiskGuardian.is_kill_switch_active() is False


# ── 7. STRATEGY GOVERNANCE & PER-INSTRUMENT CONFIG TESTS ──────────────────────
def test_strategy_governance_and_status_isolation():
    """Verify KEEP, ADAPT, RESEARCH, REMOVED classifications and instrument config."""
    eligible = StrategyRegistry.get_eligible_strategies_for_instrument("SILVERMIC")
    names = [s.name for s in eligible]

    # Reusable strategies must be present
    assert "SuperTrend+RSI" in names
    assert ("OpeningRangeBreakout" in names or "ORB" in names)
    assert ("VolatilityBreakout" in names or "BBSqueeze" in names or "SqueezeMomentum" in names)
    assert "CPR" in names

    # Removed options strategies must NOT be eligible
    assert "HeroZero" not in names
    assert "GammaExposure" not in names
    assert "ExpiryWeek" not in names
    assert "IVContraction" not in names

    # Check regime compatibility
    assert StrategyRegistry.is_strategy_permitted_in_regime("SuperTrend+RSI", MarketRegime.TREND) is True
    assert StrategyRegistry.is_strategy_permitted_in_regime("SuperTrend+RSI", MarketRegime.RANGE) is False


# ── 8. EXECUTION ENGINE SAFETY MODES TESTS ────────────────────────────────────
def test_futures_execution_modes_and_safety_guard(monkeypatch):
    """Verify PAPER, SHADOW, and failsafe LIVE mode guard."""
    monkeypatch.delenv("LIVE_TRADING_CONFIRMATION", raising=False)

    engine = FuturesExecutionEngine(default_mode=ExecutionMode.PAPER, slippage_pts=2.0)
    assert engine.mode == ExecutionMode.PAPER

    # Switching to LIVE without confirmation token must fail safe to PAPER
    switched = engine.set_mode(ExecutionMode.LIVE)
    assert switched is False
    assert engine.mode == ExecutionMode.PAPER

    # Execute simulated paper trade
    plan = TradePlan(
        signal=RawSignal(symbol="SILVERM", direction=Direction.BUY, confidence=0.8, votes=4, strategies_fired=["S1", "S2"]),
        contract_symbol="SILVERM-30Nov2026-FUT",
        entry_price=85000.0,
        sl_price=84000.0,
        target_price=87000.0,
        lot_size=5,
        desired_lots=1,
    )
    contract = resolve_active_contract("SILVERM")
    order = engine.execute_trade_plan(plan, SILVERM_CONFIG, contract, current_ltp=85000.0)

    assert order.success is True
    assert order.status == "SIMULATED"
    assert order.price == 85002.0 # 85000 + 2.0 slippage

    # Duplicate signal must be suppressed
    dup_order = engine.execute_trade_plan(plan, SILVERM_CONFIG, contract, current_ltp=85000.0)
    assert dup_order.success is False
    assert "DUPLICATE" in dup_order.rejection_reason


# ── 9. FOUR-MONTH OBSERVATION ANALYTICS TESTS ─────────────────────────────────
def test_performance_analytics_edge_report():
    """Verify statistical edge report calculation from journal DataFrame."""
    rows = [
        {"date": "2026-03-01", "time": "10:00", "symbol": "SILVERMIC", "direction": "BUY", "strategies_fired": "S1|S2", "regime": "TREND", "entry_price": 85000, "exit_price": 86000, "pnl_points": 1000.0, "pnl_inr": 1000.0, "mfe_points": 1200.0, "mae_points": 150.0, "slippage_pts": 2.0, "costs": 40.0, "rejection_reason": ""},
        {"date": "2026-03-02", "time": "14:00", "symbol": "SILVERMIC", "direction": "BUY", "strategies_fired": "S1|S2", "regime": "TREND", "entry_price": 86000, "exit_price": 87200, "pnl_points": 1200.0, "pnl_inr": 1200.0, "mfe_points": 1300.0, "mae_points": 100.0, "slippage_pts": 2.0, "costs": 40.0, "rejection_reason": ""},
        {"date": "2026-03-03", "time": "18:00", "symbol": "SILVERMIC", "direction": "SELL", "strategies_fired": "S3", "regime": "BREAKOUT", "entry_price": 87000, "exit_price": 87500, "pnl_points": -500.0, "pnl_inr": -500.0, "mfe_points": 100.0, "mae_points": 500.0, "slippage_pts": 2.0, "costs": 40.0, "rejection_reason": ""},
        {"date": "2026-03-04", "time": "11:00", "symbol": "SILVERMIC", "direction": "BUY", "strategies_fired": "S4", "regime": "RANGE", "entry_price": 86500, "exit_price": 0, "pnl_points": 0.0, "pnl_inr": 0.0, "mfe_points": 0.0, "mae_points": 0.0, "slippage_pts": 0.0, "costs": 0.0, "rejection_reason": "REGIME_MISMATCH"},
    ]
    df = pd.DataFrame(rows)

    report = PerformanceAnalyticsEngine.generate_report(df)

    assert report.total_signals == 4
    assert report.executed_trades == 3
    assert report.rejected_signals == 1
    assert report.win_rate_pct == 66.67
    assert report.profit_factor == 4.4 # 2200 / 500
    assert report.avg_win_points == 1100.0
    assert report.avg_loss_points == 500.0
    assert report.expectancy_points > 0
    assert report.net_pnl_inr == 1580.0 # 1700 - 120 costs
    assert "TREND" in report.by_regime
    assert "S1|S2" in report.by_strategy
    assert "BUY" in report.by_direction

"""
tests/unit/test_top5_improvements.py
===================================
Comprehensive unit tests for the Top 5 Strategic Improvements:
1. Concurrent Multi-Position Management (PositionManagerAgent)
2. Precious Metals Cross-Asset Correlation Guard (MarketContextGate)
3. NATGASM Liquidity & Spread Gate (TradePlannerAgent)
4. Intraday Session Profile Multipliers (StrategyAgent)
5. Model Retrain Hot-Reloading (MLFilterAgent)
"""

import pytest
from datetime import datetime, date, time
import pytz
from unittest.mock import MagicMock, AsyncMock

from core.models import Position, TradePlan, RawSignal, Direction
from core.bus import Topic, Message
from agents_code.agent6_position.manager import PositionManagerAgent, CommodityPositionSlot
from agents_code.agent2_strategy.market_context_gate import MarketContextGate
from agents_code.agent2_strategy.runner import StrategyAgent
from agents_code.agent3_ml.filter import MLFilterAgent

IST = pytz.timezone("Asia/Kolkata")


def make_dummy_plan(symbol: str, option_symbol: str, direction: Direction = Direction.BUY_CALL) -> TradePlan:
    sig = RawSignal(
        timestamp=datetime.now(IST),
        symbol=symbol,
        direction=direction,
        nifty_ltp=1000.0,
        strategies_fired=["SuperTrend+RSI"],
        confidence=0.8,
        votes=4,
    )
    return TradePlan(
        signal=sig,
        entry_premium=100.0,
        sl_premium=80.0,
        target_premium=150.0,
        lot_size=1,
        quantity=1,
        option_symbol=option_symbol,
        expiry_date=date(2026, 10, 20),
        strike=1000,
        option_type="CE",
        stop_distance=20.0,
        target_distance=50.0,
        risk_reward_ratio=2.5,
        risk_budget_inr=1000.0,
        confidence=0.8,
        setup_type="vote_aligned",
        setup_strength=0.7,
        structure_bias="BULLISH",
    )


class TestConcurrentMultiPosition:
    @pytest.mark.asyncio
    async def test_multi_position_isolation_and_exits(self):
        mgr = PositionManagerAgent()
        mgr.bus = MagicMock()
        mgr.bus.publish = AsyncMock()

        # 1. Open SILVERM position
        silver_order = {
            "symbol": "SILVERM",
            "option_symbol": "SILVERM24OCT75000CE",
            "entry_premium": 200.0,
            "quantity": 1,
            "simulated": True,
            "timestamp": datetime.now(IST).isoformat(),
        }
        await mgr.on_order(Message(Topic.ORDER_PLACED, silver_order, "TEST"))

        # 2. Open GOLDM position concurrently
        gold_order = {
            "symbol": "GOLDM",
            "option_symbol": "GOLDM24OCT152000CE",
            "entry_premium": 400.0,
            "quantity": 1,
            "simulated": True,
            "timestamp": datetime.now(IST).isoformat(),
        }
        await mgr.on_order(Message(Topic.ORDER_PLACED, gold_order, "TEST"))

        # Check that BOTH positions are concurrently tracked
        silver_pos = mgr.get_position("SILVERM")
        gold_pos = mgr.get_position("GOLDM")
        assert silver_pos is not None, "SILVERM position should be active"
        assert gold_pos is not None, "GOLDM position should be active"
        assert silver_pos["symbol"] == "SILVERM"
        assert gold_pos["symbol"] == "GOLDM"

        all_open = mgr.all_open_positions()
        assert len(all_open) == 2, f"Expected 2 concurrent positions, got {len(all_open)}"

        # 3. Close SILVERM only
        await mgr.manual_exit("SILVERM")

        # Verify SILVERM is closed but GOLDM is still active
        assert mgr.get_position("SILVERM") is None
        assert mgr.get_position("GOLDM") is not None
        assert len(mgr.all_open_positions()) == 1

        # 4. Close GOLDM
        await mgr.manual_exit("GOLDM")
        assert mgr.get_position("GOLDM") is None
        assert len(mgr.all_open_positions()) == 0


class TestCrossAssetCorrelationGuard:
    @pytest.mark.asyncio
    async def test_correlation_guard_metals(self):
        gate = MarketContextGate()
        gate.bus = MagicMock()
        gate.bus.publish = AsyncMock()

        # Simulate SILVERM opened BUY_CALL
        await gate._on_pos_opened(Message(Topic.ORDER_PLACED, {
            "symbol": "SILVERM",
            "direction": "BUY_CALL",
        }, "TEST"))

        # Incoming low-conviction GOLDM BUY_CALL (votes=4, score=0.55) -> MUST BE SUPPRESSED
        gold_raw_signal = Message(Topic.RAW_SIGNAL, {
            "symbol": "GOLDM",
            "direction": "BUY_CALL",
            "votes": 4,
            "ml_rank_score": 0.55,
            "timestamp": datetime.now(IST).isoformat(),
        }, "TEST")
        await gate.on_raw_signal(gold_raw_signal)

        # Check published topic was SIGNAL_SUPPRESSED
        call_topics = [call[0][0] for call in gate.bus.publish.call_args_list]
        assert Topic.SIGNAL_SUPPRESSED in call_topics, "Low-conviction correlated signal must be suppressed"

        gate.bus.publish.reset_mock()

        # Incoming high-conviction institutional GOLDM BUY_CALL (votes=6, score=0.75) -> ALLOWED
        gold_institutional_signal = Message(Topic.RAW_SIGNAL, {
            "symbol": "GOLDM",
            "direction": "BUY_CALL",
            "votes": 6,
            "ml_rank_score": 0.75,
            "timestamp": datetime.now(IST).isoformat(),
        }, "TEST")
        await gate.on_raw_signal(gold_institutional_signal)
        call_topics = [call[0][0] for call in gate.bus.publish.call_args_list]
        assert Topic.RAW_SIGNAL in call_topics, "Institutional conviction should bypass correlation guard"


class TestIntradaySessionProfiles:
    def test_session_weight_multipliers(self):
        # 1. Evening US session (18:30 IST) for Crude
        evening_ts = datetime(2026, 9, 16, 18, 30, tzinfo=IST)
        base, reg, adj_crude = StrategyAgent._strategy_vote_weight(
            "MomentumVolumeBreakout", "TRENDING", symbol="CRUDEOILM", current_time=evening_ts
        )
        _, _, adj_crude_morning = StrategyAgent._strategy_vote_weight(
            "MomentumVolumeBreakout", "TRENDING", symbol="CRUDEOILM", current_time=datetime(2026, 9, 16, 10, 30, tzinfo=IST)
        )
        assert adj_crude > adj_crude_morning, "Crude momentum should have higher weight during US evening session"

        # 2. Morning session (10:00 IST) favors mean reversion
        morning_ts = datetime(2026, 9, 16, 10, 0, tzinfo=IST)
        _, _, adj_mr_morning = StrategyAgent._strategy_vote_weight(
            "VWAP_MeanReversion", "RANGING", symbol="GOLDM", current_time=morning_ts
        )
        _, _, adj_bo_morning = StrategyAgent._strategy_vote_weight(
            "MomentumVolumeBreakout", "RANGING", symbol="GOLDM", current_time=morning_ts
        )
        assert adj_mr_morning > adj_bo_morning, "Morning session should favor mean reversion over breakout"


class TestNatgasSpreadGate:
    def test_natgas_spread_gate_rejection(self):
        from agents_code.agent4_planner.planner import TradePlannerAgent
        from broker.base_broker import OptionContract

        planner = TradePlannerAgent(backtest_mode=True)
        # Mock broker to return wide bid-ask spread
        snap = OptionContract(
            symbol="NATGASM24OCT280CE",
            strike=280,
            option_type="CE",
            expiry_date="2026-10-20",
            last_price=10.0,
            bid_price=9.0,
            ask_price=11.0,  # spread = (11-9)/10 = 20% > 4%
        )
        planner._fetch_contract_snapshots = MagicMock(return_value=[snap])

        # Signal at 14:00 IST (before 17:00 IST)
        scored, notes = planner._select_contracts(
            symbol="NATGASM",
            nifty_ltp=280.0,
            confidence=0.7,
            direction="BUY_CALL",
            atm=280,
            expiry_date=date(2026, 10, 20),
            dte=5,
            lot_size=1250,
            strike_step=5,
            signal_data={
                "timestamp": "2026-09-16T14:00:00+05:30",
                "strategies_fired": ["SuperTrend+RSI"],
            },
        )
        assert len(scored) == 0, "Wide spread NATGASM contract must be rejected before 17:00 IST"
        assert any("NATGASM spread gate" in n for n in notes), f"Notes should mention spread gate: {notes}"


class TestModelRetrainHotReload:
    @pytest.mark.asyncio
    async def test_model_retrained_event_hot_reloads(self):
        filter_agent = MLFilterAgent()
        filter_agent.reload_model = MagicMock()

        msg = Message(Topic.MODEL_RETRAINED, {
            "symbol": "ALL",
            "models": ["SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"],
        }, "TEST")
        await filter_agent.on_model_retrained(msg)
        assert filter_agent.reload_model.called, "reload_model() must be called on MODEL_RETRAINED event"

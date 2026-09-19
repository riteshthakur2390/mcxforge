from __future__ import annotations

"""
agents_code/agent4_planner/planner.py  Trade Planner Agent
"""
import asyncio
from loguru import logger
from datetime import date, datetime, timedelta
from typing import Union
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bus import get_bus, Topic, Message
from core.models import RawSignal, TradePlan, Direction
from config.settings.planner_thresholds import *
from config.settings import (
    NIFTY_STRIKE_STEP, NIFTY_LOT_SIZE, HIGH_CONF_THRESHOLD,
    OTM_OFFSET_POINTS, STOP_LOSS_PCT, TARGET_PCT,
    MIN_DAYS_TO_EXPIRY, LLM_ENABLED, TRADING_MODE,
    OPTION_CANDIDATE_STEPS, OPTION_MIN_PREMIUM, OPTION_MAX_PREMIUM,
    ATR_STOP_MULTIPLIER, ATR_TARGET_MULTIPLIER, BREAKEVEN_R_TRIGGER,
    TIME_STOP_MINUTES, TIME_STOP_MIN_PNL_PCT, HIGH_VOL_ATR_PCT,
    HIGH_VOL_TARGET_COMPRESSION, PAPER_TRADING_CAPITAL, RISK_BUDGET_PER_TRADE_PCT,
    MAX_POSITION_LOTS, TWO_LOT_CONFIDENCE_THRESHOLD,
    MIN_ACCEPTABLE_PROFIT_PCT, MIN_ENTRY_MINUTES_BEFORE_CLOSE,
    MARKET_CLOSE_TIME, MIN_RR_RATIO,
    MIN_EXPECTED_TARGET_PNL_PCT,
    EXPIRY_EARLY_ENTRY_MINUTE, EXPIRY_EARLY_MIN_RANK,
    EXPIRY_EARLY_MIN_SETUP, EXPIRY_EARLY_MIN_VOTES,
    EXECUTION_MIN_RANK_SCORE, EXECUTION_LAST_ENTRY_MINUTE,
    BACKTEST_BROKERAGE_PER_ORDER, BACKTEST_TRANSACTION_COST_PCT,
    PLANNER_ENTRY_SLIPPAGE_PCT, PLANNER_MIN_EXECUTABLE_RANK,
    PLANNER_MIN_STRATEGY_CONFIDENCE, SUPPORTED_INDEX_SYMBOLS,
    DEPLOYED_CAPITAL, STRONG_SIGNAL_CAPITAL_PCT,
)
from config.settings.strategy import (
    ALLOW_SUBMIN_VOTE_EARLY_TRIGGER,
    MIN_STRATEGY_VOTES,
)
from broker.factory import get_broker
from broker.base_broker import OptionContract
from utils.option_utils import (
    apply_option_slippage, compute_atr,
    estimate_atm_premium, get_nearest_expiry,
    build_option_symbol, get_atm_strike, validate_option_contract,
    get_index_lot_size, get_index_strike_step, is_expiry_day,
    estimate_round_trip_costs,
)
from utils.instrument_selector import get_instrument, InstrumentConfig
from utils.market_calendar import latest_expected_trading_day
from utils.final_decision import FinalDecisionLayer
from utils.llm import TaskType, call_llm_async, call_llm_context_async
from utils.greeks_filter import GreeksFilter
from utils.market_intelligence import MarketIntelligence
from utils.mtf_data_fetcher import MTFDataFetcher
from utils.dynamic_exit_manager import compute_atr_sl, get_current_iv_proxy, select_strike_em
from utils.greeks_position_sizer import size_by_delta
from utils.news_sentiment import get_news_filter
from utils.smart_entry_filter import get_smart_entry_filter
from backtesting.realistic_assumptions import apply_realistic_entry
from utils.pipeline_logging import log_pipeline_stage
from utils.advanced_filters import get_expiry_theta
from instruments.registry import get_instrument_strategy_config, normalize_symbol

BACKTEST_MAX_TRADE_INVESTMENT_INR = float(os.getenv("BACKTEST_MAX_TRADE_INVESTMENT_INR", "40000"))
MAX_TRADE_INVESTMENT_INR = BACKTEST_MAX_TRADE_INVESTMENT_INR
STRONG_TRADE_INVESTMENT_INR = float(os.getenv("BACKTEST_STRONG_TRADE_INVESTMENT_INR", "40000"))
BASE_TRADE_INVESTMENT_INR = float(os.getenv("BACKTEST_BASE_TRADE_INVESTMENT_INR", "25000"))
REDUCED_BUDGET_MAX_TRADE_INR = float(os.getenv("REDUCED_BUDGET_MAX_TRADE_INR", "15000"))
REDUCED_BUDGET_MIN_VOTES = int(os.getenv("REDUCED_BUDGET_MIN_VOTES", "4"))
REDUCED_BUDGET_MIN_RANK = float(os.getenv("REDUCED_BUDGET_MIN_RANK", "0.55"))
REDUCED_BUDGET_MIN_SETUP = float(os.getenv("REDUCED_BUDGET_MIN_SETUP", "0.60"))
REDUCED_BUDGET_STRONG_MIN_VOTES = int(os.getenv("REDUCED_BUDGET_STRONG_MIN_VOTES", "4"))
REDUCED_BUDGET_STRONG_MIN_RANK = float(os.getenv("REDUCED_BUDGET_STRONG_MIN_RANK", "0.70"))
REDUCED_BUDGET_STRONG_MIN_SETUP = float(os.getenv("REDUCED_BUDGET_STRONG_MIN_SETUP", "0.84"))
REDUCED_BUDGET_TARGET_PCT = float(os.getenv("REDUCED_BUDGET_TARGET_PCT", "30"))
REDUCED_BUDGET_MIN_PROB = float(os.getenv("REDUCED_BUDGET_MIN_PROB", "0.25"))
CAUTION_TARGET_PCT = float(os.getenv("CAUTION_TARGET_PCT", "20"))
CAUTION_TARGET_MAX_RANK = float(os.getenv("CAUTION_TARGET_MAX_RANK", "0.70"))
CAUTION_TARGET_MAX_ML_CONF = float(os.getenv("CAUTION_TARGET_MAX_ML_CONF", "0.45"))
LOW_VOTE_MIN_VOTES = int(os.getenv("LOW_VOTE_MIN_VOTES", "5"))
LOW_VOTE_MIN_RANK = float(os.getenv("LOW_VOTE_MIN_RANK", "0.75"))
LOW_VOTE_MIN_PROB = float(os.getenv("LOW_VOTE_MIN_PROB", "0.50"))
LATE_SESSION_TARGET_START_MINUTE = int(
    os.getenv("LATE_SESSION_TARGET_START_MINUTE", str(22 * 60 + 30))
)


class TradePlannerAgent:
    NAME = "TradePlannerAgent"

    @staticmethod
    def _reduced_budget_meta(reason: str) -> dict:
        return {
            "reduced_budget_lane": True,
            "reduced_budget_reason": reason,
            "max_trade_investment_inr": REDUCED_BUDGET_MAX_TRADE_INR,
        }

    @staticmethod
    def _reduced_budget_requested(data: dict) -> tuple[bool, float, str, float]:
        metadata = data.get("metadata") or {}
        requested = bool(metadata.get("reduced_budget_lane", False))
        max_investment = float(
            metadata.get("max_trade_investment_inr", REDUCED_BUDGET_MAX_TRADE_INR)
            or REDUCED_BUDGET_MAX_TRADE_INR
        )
        reason = str(metadata.get("reduced_budget_reason", "") or "")
        min_ml_prob = float(
            metadata.get("reduced_budget_min_ml_prob", REDUCED_BUDGET_MIN_PROB)
            or REDUCED_BUDGET_MIN_PROB
        )
        return requested, max_investment, reason, min_ml_prob

    @staticmethod
    def _target_pct(entry_premium: float, target_premium: float) -> float:
        entry = float(entry_premium or 0.0)
        target = float(target_premium or 0.0)
        if entry <= 0:
            return 0.0
        return max(0.0, (target / entry - 1.0) * 100.0)

    @staticmethod
    def _compress_target(
        *,
        entry_premium: float,
        target_premium: float,
        target_pct: float,
    ) -> float:
        entry = float(entry_premium or 0.0)
        if entry <= 0:
            return float(target_premium or 0.0)
        compressed = round(entry * (1.0 + float(target_pct) / 100.0), 2)
        return min(float(target_premium or compressed), compressed)

    def __init__(self, data_agent=None, backtest_mode: bool = False) -> None:
        self.bus        = get_bus()
        self.data_agent = data_agent
        self.backtest_mode = backtest_mode
        self.broker     = getattr(data_agent, "broker", None) or get_broker()
        self.final_decision = FinalDecisionLayer(
            slippage_pct=0.0 if backtest_mode else PLANNER_ENTRY_SLIPPAGE_PCT
        )
        self.market_intelligence = MarketIntelligence()
        self.mtf_fetcher = MTFDataFetcher(self.broker)
        self.news_filter = get_news_filter()

    def register(self) -> None:
        self.bus.subscribe(Topic.SIGNAL_APPROVED, self.on_approved)
        logger.info(f"[{self.NAME}] Registered.")

    async def _publish_planner_rejection(
        self,
        signal: dict,
        *,
        reason: str,
        details: str = "",
    ) -> None:
        payload = dict(signal)
        if "regime" in payload and hasattr(payload["regime"], "value"):
            payload["regime"] = payload["regime"].value
        payload["rejection_reason"] = reason if not details else f"{reason}: {details}"
        payload["planner_rejection_reason"] = reason
        if details:
            payload["planner_rejection_details"] = details
        try:
            await self.bus.publish(Topic.SIGNAL_REJECTED, payload, self.NAME)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Planner rejection journal publish skipped: {exc}")

    @staticmethod
    def _allows_low_consensus_plan(data: dict) -> bool:
        context = ((data.get("metadata") or {}).get("_context") or {})
        return ALLOW_SUBMIN_VOTE_EARLY_TRIGGER and bool(context.get("early_trigger", False))

    @staticmethod
    def _execution_confidence(
        data: dict,
        *,
        rank_score: float,
        success_prob: float,
    ) -> float:
        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        strategy_conf = float(data.get("confidence", 0.0) or 0.0)
        blended = (
            success_prob * 0.50
            + setup_strength * 0.30
            + strategy_conf * 0.20
        )
        return round(max(0.0, min(0.99, min(rank_score, blended))), 4)

    @staticmethod
    def _allowed_moneyness(direction: str, confidence: float) -> set[str]:
        # Moneyness is a coarse safety check; GreeksFilter and risk scoring
        # already reject far-OTM contracts. Allowing near-ATM OTM avoids dropping
        # valid signals when ATM rounding or historical-chain coverage shifts.
        return {"ATM", "ITM", "OTM"}

    def _use_historical_option_premium(self) -> bool:
        if not self.backtest_mode:
            return False
        return os.getenv(
            "BACKTEST_ENABLE_HISTORICAL_OPTION_PREMIUM",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _require_historical_option_premium(self) -> bool:
        if not self.backtest_mode:
            return False
        return os.getenv(
            "BACKTEST_REQUIRE_HISTORICAL_OPTION_PREMIUM",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _should_allow_backtest_fallback(
        *,
        candidate: dict,
        setup: dict,
        ml_rank_score: float,
    ) -> tuple[bool, str]:
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        if setup_type == "breakout" and setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_74:
            return False, (
                f"weak breakout fallback blocked (rank={ml_rank_score:.2f}, "
                f"setup={setup_strength:.2f}, score={float(candidate.get('score', 0.0)):.2f})"
            )
        return True, ""

    @staticmethod
    def _structure_bias_for_signal(data: dict) -> str:
        context = ((data.get("metadata") or {}).get("_context") or {})
        setup = context.get("setup", {}) or {}
        setup_context = setup.get("context", {}) or {}
        market_structure = (
            context.get("market_structure")
            or setup_context.get("market_structure")
            or {}
        )
        structure_state = market_structure.get("structure_state", {}) or {}
        return str(structure_state.get("bias", "UNKNOWN") or "UNKNOWN").upper()

    @staticmethod
    def _weak_entry_block_reason(
        *,
        data: dict,
        signal_ts: datetime,
        rank_score: float,
        success_prob: float,
        setup: dict,
        votes: int,
        direction: str,
    ) -> str:
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        signal_minute = signal_ts.hour * 60 + signal_ts.minute
        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        bias = TradePlannerAgent._structure_bias_for_signal(data)
        opposite_bias = (
            direction == "BUY_CALL" and bias == "BEARISH"
        ) or (
            direction == "BUY_PUT" and bias == "BULLISH"
        )
        context = ((data.get("metadata") or {}).get("_context") or {})
        setup_context = setup.get("context", {}) or {}
        market_structure = (
            context.get("market_structure")
            or setup_context.get("market_structure")
            or {}
        )
        liquidity_event = market_structure.get("liquidity_event", {}) or {}
        entry_validation = market_structure.get("entry_validation", {}) or {}
        direction_validation = (
            (entry_validation.get("by_direction", {}) or {}).get(direction, {})
        )
        event_type = str(liquidity_event.get("type", "NONE") or "NONE").lower()
        event_direction = str(liquidity_event.get("direction", "NONE") or "NONE").upper()
        direction_entry_valid = bool(
            direction_validation.get("valid", entry_validation.get("valid", False))
        )
        hybrid_5m = context.get("hybrid_5m", {}) or {}
        hybrid_bias = str(hybrid_5m.get("bias", "NEUTRAL") or "NEUTRAL").upper()
        hybrid_conf = float(hybrid_5m.get("confidence", 0.0) or 0.0)
        hybrid_aligned = (
            hybrid_conf >= PLANNER_HYBRID_CONF_THRESH_0_58
            and (
                (hybrid_bias == "BULLISH" and direction == "BUY_CALL")
                or (hybrid_bias == "BEARISH" and direction == "BUY_PUT")
            )
        )
        hard_auction_trigger = bool(
            strategies.intersection({"OIAnalysis", "VolumeProfile", "StrikeMomentum", "GammaExposure", "RangeSpread", "ValueArea"})
        )
        wyckoff = context.get("wyckoff", {}) or {}
        wyckoff_rec = str(wyckoff.get("recommendation", "") or "").upper()
        wyckoff_conf = float(wyckoff.get("confidence", 0.0) or 0.0)
        weighted_vote = context.get("weighted_vote", {}) or {}
        winning_side = weighted_vote.get("call" if direction == "BUY_CALL" else "put", {}) or {}
        instrument_symbol = str(data.get("symbol") or os.getenv("INSTRUMENT", "SILVERM")).upper()
        is_mcx = instrument_symbol in SUPPORTED_INDEX_SYMBOLS or any(k in instrument_symbol for k in ("SILVER", "GOLD", "CRUDE", "NAT"))
        expiry_session = is_expiry_day(signal_ts.date(), symbol=instrument_symbol) or ("HeroZero" in strategies)
        if is_mcx:
            late_entry_cutoff = (23 * 60) if expiry_session else (22 * 60 + 30)
        else:
            late_entry_cutoff = (15 * 60 + 5) if expiry_session else (15 * 60)
        if signal_minute >= late_entry_cutoff:
            return (
                f"blocked late-day entry after {late_entry_cutoff // 60}:{late_entry_cutoff % 60:02d} "
                f"(signal_minute={signal_minute})"
            )
        wyckoff_opposes_direction = (
            ("FAVOR_CALL" in wyckoff_rec and direction == "BUY_PUT")
            or ("FAVOR_PUT" in wyckoff_rec and direction == "BUY_CALL")
        )
        if (
            setup_type == "vote_aligned"
            and wyckoff_opposes_direction
            and wyckoff_conf >= 60.0
            and not hard_auction_trigger
            and not (
                rank_score >= PLANNER_RANK_SCORE_THRESH_0_65
                and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_88
                and votes >= 8
                and weighted_score >= 5.0
            )
        ):
            return (
                f"blocked vote_aligned setup against Wyckoff recommendation "
                f"(rec={wyckoff_rec}, conf={wyckoff_conf:.0f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and signal_minute < 11 * 60
            and "ExpiryWeek" in strategies
            and not strategies.intersection({"FVG", "OIAnalysis", "ValueArea", "VolumeProfile", "ORB"})
        ):
            return (
                f"blocked early expiry PUT without flow/value confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        cas_ctx = context.get("cas") or {}
        cas_score = float(context.get("cas_score", 0.0) or cas_ctx.get("score", 0.0) or 0.0)
        cas_override = bool(context.get("cas_override") or cas_ctx.get("allows_structural_override", False) or cas_score >= 0.65)
        hero_zero_active = "HeroZero" in strategies

        # ── Chop Index Gate ──────────────────────────────────────────────────
        # Block entries when Choppiness Index > 60.0 unless there is a confirmed volume/ORB breakout or high CAS
        det_regime = context.get("detailed_regime") or {}
        chop_val = float(det_regime.get("chop", 0.0) or context.get("chop_index", 0.0) or 0.0)
        has_breakout_confirm = bool(strategies.intersection({"ORB", "VolumeProfile", "Breakout", "CPR"}))
        if (
            chop_val > 60.0
            and not has_breakout_confirm
            and not (cas_score >= 0.72 or (hero_zero_active and expiry_session))
        ):
            return (
                f"blocked entry in high chop regime (Chop={chop_val:.1f} > 60.0 "
                f"without confirmed volume/ORB breakout, CAS={cas_score:.2f})"
            )

        # ── Higher-Timeframe (HTF) Trend Confirmation Filter ─────────────────
        # Suppress non-expiry entries when Daily or 60m trend conflicts with intraday 5m signal unless CAS >= 0.72
        mtf_data = context.get("mtf") or {}
        daily_trend = str(mtf_data.get("daily_trend", "") or "").upper()
        trend_60m = str(mtf_data.get("trend_60m", "") or "").upper()
        htf_conflicts = (
            (direction == "BUY_CALL" and (daily_trend == "DOWN" or trend_60m == "BEARISH"))
            or (direction == "BUY_PUT" and (daily_trend == "UP" or trend_60m == "BULLISH"))
        )
        if (
            htf_conflicts
            and not expiry_session
            and not (hero_zero_active and expiry_session)
            and cas_score < 0.72
            and votes < 5
            and rank_score < 0.70
            and setup_strength < 0.75
        ):
            return (
                f"blocked counter-HTF trend entry without high CAS alignment "
                f"(HTF daily={daily_trend}, 60m={trend_60m}, CAS={cas_score:.2f} < 0.72)"
            )

        # ── RSI Climax Exhaustion Filter ────────────────────────────────────
        rsi_val = float(context.get("rsi_14", 0.0) or 0.0)
        if rsi_val > 0.0:
            if direction == "BUY_PUT" and rsi_val < 28.0:
                return (
                    f"blocked oversold RSI exhaustion for BUY_PUT (RSI_5m={rsi_val:.1f} < 28.0)"
                )
            elif direction == "BUY_CALL" and rsi_val > 72.0:
                return (
                    f"blocked overbought RSI exhaustion for BUY_CALL (RSI_5m={rsi_val:.1f} > 72.0)"
                )

        # ── Timing Climax Exhaustion Guard ──────────────────────────────────
        timing_class = str(context.get("timing_classification", "") or "").upper()
        if timing_class == "EXHAUSTED" and votes < 7:
            return (
                f"blocked timing exhaustion with weak consensus (timing={timing_class}, votes={votes} < 7)"
            )
        if timing_class == "EXTENDED" and votes < 5:
            return (
                f"blocked extended move entry without high consensus (timing={timing_class}, votes={votes} < 5)"
            )

        # ── Anti-Chasing & Mean-Extension Guard ──────────────────────────────
        setup_ctx = setup.get("context", {}) or {}
        stretch = float(setup_ctx.get("stretch", 0.0) or 0.0)
        nifty_ltp_val = float(data.get("nifty_ltp", 0.0) or 0.0)
        if stretch != 0.0 and nifty_ltp_val > 0:
            stretch_pct = abs(stretch) / nifty_ltp_val * 100.0
            # If price is already extended >0.35% from VWAP without compression/pullback
            if (
                stretch_pct >= 0.35
                and setup_type in {"breakout", "vote_aligned", "trend_pullback", "none"}
                and not (cas_override or (hero_zero_active and expiry_session))
                and not {"ORB", "CPR", "FVG"}.intersection(strategies)
            ):
                return (
                    f"blocked overextended entry chase (stretch={stretch:.1f}pts, "
                    f"{stretch_pct:.2f}% from VWAP/EMA without pullback/retest)"
                )

        if not (cas_override or (hero_zero_active and expiry_session)):
            if (
                setup_type == "vote_aligned"
                and direction == "BUY_CALL"
                and bias == "BEARISH"
                and not strategies.intersection({"ORB", "CPR"})
            ):
                return (
                    f"blocked bearish-bias CALL without ORB/CPR confirmation "
                    f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                    f"strategies={'+'.join(sorted(strategies))})"
                )
            if (
                setup_type == "trend_pullback"
                and (
                    (direction == "BUY_CALL" and bias == "BEARISH")
                    or (direction == "BUY_PUT" and bias == "BULLISH")
                )
                and "ORB" not in strategies
            ):
                return (
                    f"blocked counter-bias trend_pullback without ORB displacement "
                    f"(bias={bias}, direction={direction}, rank={rank_score:.2f}, "
                    f"setup={setup_strength:.2f}, strategies={'+'.join(sorted(strategies))})"
                )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and {"FVG", "Ichimoku"}.issubset(strategies)
            and not strategies.intersection({"ORB", "CPR", "ValueArea", "StrikeMomentum", "RangeSpread"})
        ):
            return (
                f"blocked CALL FVG/Ichimoku continuation without auction confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and {"FVG", "OIAnalysis", "GammaExposure"}.issubset(strategies)
            and not strategies.intersection({"ORB", "CPR", "ValueArea", "StrikeMomentum", "RangeSpread"})
        ):
            return (
                f"blocked CALL flow stack without auction confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "breakout"
            and direction == "BUY_CALL"
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
            and not strategies.intersection({"ORB", "CPR", "OIAnalysis", "GammaExposure", "RangeSpread"})
        ):
            return (
                f"blocked low-rank CALL breakout without institutional confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "breakout"
            and direction == "BUY_CALL"
            and {"VWAP+EMA", "ADX+PSAR", "Ichimoku", "ValueArea"}.issubset(strategies)
            and not strategies.intersection({"FVG", "ORB", "CPR", "OIAnalysis", "GammaExposure", "RangeSpread"})
        ):
            return (
                f"blocked CALL breakout without flow confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "breakout"
            and direction == "BUY_PUT"
            and "SkewHunter" in strategies
            and not strategies.intersection({"FVG", "ValueArea", "ADX+PSAR", "RangeSpread"})
        ):
            return (
                f"blocked PUT breakout without trend/value confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and {"CPR", "PriceAction", "GammaExposure"}.issubset(strategies)
            and not strategies.intersection({"FVG", "OIAnalysis", "ValueArea", "ORB", "RangeSpread"})
        ):
            return (
                f"blocked CALL CPR/PriceAction stack without flow/value confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and {"SuperTrend+RSI", "BBSqueeze", "ValueArea"}.issubset(strategies)
            and not strategies.intersection({"FVG", "ORB", "CPR", "OIAnalysis", "RangeSpread"})
        ):
            return (
                f"blocked CALL squeeze/value continuation without displacement or flow "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and {"ADX+PSAR", "ExpiryWeek", "ValueArea", "EMASlope"}.issubset(strategies)
            and not strategies.intersection({"FVG", "OIAnalysis", "VolumeProfile", "ORB", "CPR", "RangeSpread"})
        ):
            return (
                f"blocked expiry-week PUT value/EMA continuation without flow "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and "ExpiryWeek" in strategies
            and "ValueArea" in strategies
            and not strategies.intersection({"FVG", "OIAnalysis", "VolumeProfile", "ORB", "CPR"})
            and (
                direction == "BUY_PUT"
                or bias == "RANGING"
                or "EMASlope" in strategies
            )
        ):
            return (
                f"blocked expiry/value continuation without flow or auction trigger "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"direction={direction}, bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type in {"vote_aligned", "trend_pullback"}
            and direction == "BUY_PUT"
            and {"SuperTrend+RSI", "SkewHunter", "EMASlope"}.issubset(strategies)
            and "ADX+PSAR" not in strategies
            and not strategies.intersection({"ValueArea", "StochRSI", "FVG", "OIAnalysis", "RangeSpread"})
        ):
            return (
                f"blocked PUT skew/EMA continuation without ADX or value confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            direction == "BUY_PUT"
            and votes < 6
            and setup_type in {"breakout", "trend_pullback"}
            and "EMASlope" in strategies
            and not strategies.intersection({"FVG", "OIAnalysis", "VolumeProfile", "ORB", "CPR", "ElliottWave", "RangeSpread", "StrikeMomentum"})
            and (
                (
                    setup_type == "trend_pullback"
                    and {"SuperTrend+RSI", "ADX+PSAR"}.issubset(strategies)
                )
                or {"ADX+PSAR", "SkewHunter"}.issubset(strategies)
                or {"SuperTrend+RSI", "RangeSpread"}.issubset(strategies)
                or {"BBSqueeze", "ADX+PSAR"}.issubset(strategies)
            )
            and not (
                "ValueArea" in strategies
                and "StochRSI" in strategies
                and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_84
            )
        ):
            return (
                f"blocked PUT technical pullback/breakout without flow confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and {"BBSqueeze", "ADX+PSAR", "ValueArea"}.issubset(strategies)
            and not strategies.intersection({"FVG", "OIAnalysis", "VolumeProfile", "ORB", "CPR", "RangeSpread"})
        ):
            return (
                f"blocked CALL squeeze/ADX/value continuation without flow confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and "ValueArea" in strategies
            and bias == "RANGING"
            and not strategies.intersection({"FVG", "OIAnalysis", "VolumeProfile", "ORB", "CPR", "RangeSpread"})
        ):
            return (
                f"blocked ranging CALL value continuation without auction/flow trigger "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and strategies == {"SuperTrend+RSI", "ValueArea"}
        ):
            return (
                f"blocked thin CALL SuperTrend/value continuation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, bias={bias})"
            )
        if (
            setup_type == "trend_pullback"
            and direction == "BUY_CALL"
            and {"SuperTrend+RSI", "ValueArea", "EMASlope"}.issubset(strategies)
            and "ADX+PSAR" not in strategies
            and "HeikinAshi" not in strategies
        ):
            return (
                f"blocked CALL value/EMA pullback without trend or candle confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "trend_pullback"
            and direction == "BUY_PUT"
            and {"SuperTrend+RSI", "ADX+PSAR", "SkewHunter", "ValueArea"}.issubset(strategies)
            and not strategies.intersection({"EMASlope", "StochRSI", "FVG", "OIAnalysis", "RangeSpread"})
        ):
            return (
                f"blocked PUT trend/skew pullback without momentum confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and bias == "BEARISH"
            and {"SuperTrend+RSI", "ADX+PSAR", "SkewHunter", "ValueArea", "EMASlope"}.issubset(strategies)
            and "StochRSI" not in strategies
            and not strategies.intersection({"FVG", "OIAnalysis", "RangeSpread"})
        ):
            return (
                f"blocked BEARISH PUT skew/value continuation without flow or oscillator confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        validated_counter_structure = (
            votes >= 2
            and rank_score >= PLANNER_RANK_SCORE_THRESH_0_48
            and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_78
            and setup_type != "trend_pullback"
            and direction_entry_valid
            and not bool(entry_validation.get("avoid", False))
            and event_direction == direction
            and event_type in {"structure_break", "liquidity_sweep_high", "liquidity_sweep_low"}
        )
        auction_confirmed_counter_structure = (
            hard_auction_trigger
            and votes >= 6
            and rank_score >= PLANNER_RANK_SCORE_THRESH_0_58
            and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_78
            and strategies.intersection({"OIAnalysis", "StrikeMomentum", "GammaExposure", "RangeSpread"})
            and setup_type in {"vote_aligned", "breakout"}
        )
        hybrid_confirmed_quality = (
            hybrid_aligned
            and direction_entry_valid
            and not bool(entry_validation.get("avoid", False))
            and event_direction == direction
            and event_type in {"structure_break", "liquidity_sweep_high", "liquidity_sweep_low"}
            and votes >= 3
            and weighted_score >= PLANNER_WEIGHTED_SCORE_THRESH_1_9
            and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_72
            and rank_score >= PLANNER_RANK_SCORE_THRESH_0_52
            and success_prob >= PLANNER_SUCCESS_PROB_THRESH_0_3
        )

        instrument_symbol = str(data.get("symbol") or os.getenv("INSTRUMENT", "SILVERM")).upper()
        is_mcx = instrument_symbol in SUPPORTED_INDEX_SYMBOLS or any(k in instrument_symbol for k in ("SILVER", "GOLD", "CRUDE", "NAT"))
        expiry_session = is_expiry_day(signal_ts.date(), symbol=instrument_symbol) or ("HeroZero" in strategies)
        if is_mcx:
            late_entry_cutoff = (23 * 60) if expiry_session else (22 * 60 + 30)
        else:
            late_entry_cutoff = (15 * 60 + 5) if expiry_session else (15 * 60)
        if signal_minute >= late_entry_cutoff:
            return (
                f"blocked late-day entry after {late_entry_cutoff // 60}:{late_entry_cutoff % 60:02d} "
                f"(signal_minute={signal_minute})"
            )
        late_supertrend_cutoff = (21 * 60 + 15) if is_mcx else (13 * 60 + 15)
        if (
            signal_minute >= late_supertrend_cutoff
            and votes <= 2
            and strategies == {"SuperTrend+RSI", "ADX+PSAR"}
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
            and setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_86
        ):
            return (
                f"blocked late weak SuperTrend/ADX pair "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            "ExpiryWeek" in strategies
            and strategies.issubset({"SuperTrend+RSI", "ADX+PSAR", "ExpiryWeek"})
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
        ):
            return (
                f"blocked weak expiry-week SuperTrend/ADX pair "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if rank_score < PLANNER_MIN_EXECUTABLE_RANK:
            return (
                f"blocked low-rank executable setup "
                f"(rank={rank_score:.2f} < {PLANNER_MIN_EXECUTABLE_RANK:.2f}, "
                f"p={success_prob:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type in {"vote_aligned", "trend_pullback", "breakout"}
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_58
            and not (
                votes >= 5
                and setup_strength >= 0.65
                and {"OIAnalysis", "VolumeProfile", "FVG", "CPR", "ORB", "ValueArea", "SuperTrend+RSI"}.intersection(strategies)
            )
            and not (
                votes >= 8
            )
            and not (
                direction == "BUY_CALL"
                and {"BBSqueeze", "FVG", "ValueArea"}.issubset(strategies)
                and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_72
            )
        ):
            return (
                f"blocked low-rank vote_aligned setup "
                f"(rank={rank_score:.2f} < 0.58, p={success_prob:.2f}, "
                f"setup={setup_strength:.2f}, direction={direction}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type in {"vote_aligned", "trend_pullback", "breakout"}
            and direction == "BUY_PUT"
            and "ExpiryWeek" not in strategies
            and votes < 6
            and not (
                votes >= 4
                and rank_score >= PLANNER_RANK_SCORE_THRESH_0_65
                and {"VWAP+EMA", "ADX+PSAR", "OIAnalysis", "VolumeProfile"}.intersection(strategies)
            )
            and not (
                {"OIAnalysis", "VolumeProfile"}.intersection(strategies)
                and votes >= 4
                and rank_score >= PLANNER_RANK_SCORE_THRESH_0_58
            )
            and not (
                {"ADX+PSAR", "SkewHunter", "ValueArea"}.issubset(strategies)
                and votes >= 5
                and rank_score >= PLANNER_RANK_SCORE_THRESH_0_58
                and success_prob >= PLANNER_SUCCESS_PROB_THRESH_0_3
                and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_84
                and signal_minute >= (12 * 60)
            )
            and not (
                rank_score >= 0.70
                or (rank_score >= PLANNER_RANK_SCORE_THRESH_0_65 and success_prob >= 0.60)
            )
        ):
            return (
                f"blocked non-expiry PUT without institutional confirmation "
                f"(rank={rank_score:.2f}, p={success_prob:.2f}, votes={votes}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and (
                "ADXRising" in strategies
                or ("SkewHunter" in strategies and "ExpiryWeek" not in strategies)
            )
            and not {"OIAnalysis", "VolumeProfile"}.intersection(strategies)
            and not {"ADX+PSAR", "ExpiryWeek"}.issubset(strategies)
            and votes <= 3
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
        ):
            return (
                f"blocked weak PUT ADXRising/Skew continuation without flow confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and strategies == {"ADX+PSAR", "SkewHunter"}
            and (
                rank_score < PLANNER_RANK_SCORE_THRESH_0_62
                or setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_84
                or success_prob < PLANNER_SUCCESS_PROB_THRESH_0_312
            )
        ):
            return (
                f"blocked weak pure ADX/Skew PUT continuation "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, bias={bias})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and strategies == {"BBSqueeze", "ADX+PSAR", "SkewHunter"}
            and (rank_score < PLANNER_RANK_SCORE_THRESH_0_6 or success_prob < PLANNER_SUCCESS_PROB_THRESH_0_312)
        ):
            return (
                f"blocked weak squeeze/ADX/Skew PUT continuation "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, bias={bias})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and strategies.issubset({"BBSqueeze", "ADX+PSAR", "ExpiryWeek"})
            and {"BBSqueeze", "ADX+PSAR"}.issubset(strategies)
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
        ):
            return (
                f"blocked weak BBSqueeze/ADX CALL pair "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and {"ADX+PSAR", "ValueArea"}.issubset(strategies)
            and not strategies.intersection({"FVG", "ORB", "Ichimoku", "OIAnalysis", "VolumeProfile", "CPR"})
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_62
        ):
            return (
                f"blocked weak ADX/value CALL continuation without independent confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and "ADXRising" in strategies
            and not strategies.intersection({"FVG", "ORB", "Ichimoku", "OIAnalysis", "VolumeProfile", "CPR"})
            and bias in {"BEARISH", "RANGING"}
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
        ):
            return (
                f"blocked weak ADXRising CALL against {bias} structure "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and strategies == {"FVG", "Ichimoku"}
            and votes <= 2
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_58
        ):
            return (
                f"blocked weak FVG/Ichimoku two-vote setup "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, direction={direction})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and strategies == {"SuperTrend+RSI", "BBSqueeze"}
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_305
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_56
        ):
            return (
                f"blocked low-probability SuperTrend/BBSqueeze CALL "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, bias={bias})"
            )
        if (
            setup_type == "vote_aligned"
            and strategies == {"SuperTrend+RSI", "ADX+PSAR"}
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_305
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_58
        ):
            return (
                f"blocked low-probability SuperTrend/ADX pair "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, bias={bias}, direction={direction})"
            )
        if (
            setup_type in {"vote_aligned", "trend_pullback", "breakout"}
            and opposite_bias
            and not validated_counter_structure
            and not auction_confirmed_counter_structure
        ):
            return f"{setup_type} setup conflicts with structure bias {bias}"
        if hybrid_confirmed_quality:
            return ""
        if (
            setup_type == "trend_pullback"
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_31
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_62
        ):
            return (
                f"blocked low-probability trend_pullback "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, bias={bias})"
            )
        if (
            setup_type == "vote_aligned"
            and votes == 4
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_304
            and "ORB" not in strategies
            and not {"FVG", "OIAnalysis"}.issubset(strategies)
        ):
            return (
                f"blocked weak four-vote continuation "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, bias={bias}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and bias == "RANGING"
            and "Ichimoku" in strategies
            and "ExpiryWeek" in strategies
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_32
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_62
        ):
            return (
                f"blocked ranging PUT expiry/Ichimoku trap "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and strategies == {"VolumeProfile", "OIAnalysis"}
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_56
        ):
            return (
                f"blocked weak PUT volume/OI-only setup "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, bias={bias})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and strategies == {"SuperTrend+RSI", "Ichimoku"}
            and votes <= 2
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_58
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_318
        ):
            return (
                f"blocked weak PUT SuperTrend/Ichimoku continuation "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, setup={setup_strength:.2f})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and strategies == {"SuperTrend+RSI", "BBSqueeze"}
            and votes <= 2
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_56
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_305
        ):
            return (
                f"blocked weak PUT SuperTrend/BBSqueeze continuation "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, setup={setup_strength:.2f})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and "Ichimoku" in strategies
            and "FVG" not in strategies
            and "ADX+PSAR" not in strategies
            and "SkewHunter" not in strategies
            and signal_minute >= 13 * 60
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_57
        ):
            return (
                f"blocked late weak PUT Ichimoku stack without displacement "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and votes >= 3
            and "Ichimoku" in strategies
            and {"OIAnalysis", "VolumeProfile"}.intersection(strategies)
            and "FVG" not in strategies
            and "SkewHunter" not in strategies
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_305
        ):
            return (
                f"blocked PUT institutional stack without displacement confirmation "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            votes <= 3
            and setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and {"SuperTrend+RSI", "BBSqueeze", "SkewHunter"}.issubset(strategies)
            and not {"ADX+PSAR", "VWAP+EMA", "VolumeProfile", "OIAnalysis"}.intersection(strategies)
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_32
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
            and not (
                bias == "BEARISH"
                and rank_score >= PLANNER_RANK_SCORE_THRESH_0_625
                and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_84
                and success_prob >= PLANNER_SUCCESS_PROB_THRESH_0_31
            )
        ):
            return (
                f"blocked weak PUT squeeze/skew continuation "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, bias={bias})"
            )
        if (
            setup_type in {"vote_aligned", "trend_pullback"}
            and direction == "BUY_PUT"
            and strategies == {"SuperTrend+RSI", "ADX+PSAR", "SkewHunter"}
            and (
                rank_score < PLANNER_RANK_SCORE_THRESH_0_62
                or setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_88
                or bias != "BEARISH"
            )
        ):
            return (
                f"blocked unconfirmed SuperTrend/ADX/Skew PUT continuation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        put_cont_cutoff = (21 * 60) if is_mcx else (13 * 60)
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and signal_minute >= put_cont_cutoff
            and {"SuperTrend+RSI", "FVG", "Ichimoku", "OIAnalysis"}.issubset(strategies)
            and not {"ADX+PSAR", "ORB", "SkewHunter"}.intersection(strategies)
            and (rank_score < PLANNER_RANK_SCORE_THRESH_0_6 or setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_86)
        ):
            return (
                f"blocked late PUT continuation without trend/ORB confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_CALL"
            and signal_minute < (17 * 60 if is_mcx else 13 * 60)
            and {"UTBot", "OIAnalysis", "FVG", "VWAP+EMA", "Ichimoku"}.issubset(strategies)
            and "ORB" not in strategies
            and (rank_score < PLANNER_RANK_SCORE_THRESH_0_65 or setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_88)
        ):
            return (
                f"blocked CALL continuation stack without ORB confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            direction == "BUY_CALL"
            and signal_minute <= 10 * 60
            and setup_type in {"vote_aligned", "breakout"}
            and "Ichimoku" in strategies
            and "BBSqueeze" not in strategies
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_62
        ):
            return (
                f"blocked early Ichimoku CALL without squeeze confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        late_rank_cutoff = (21 * 60 + 15) if is_mcx else (13 * 60 + 15)
        if signal_minute >= late_rank_cutoff and rank_score < PLANNER_RANK_SCORE_THRESH_0_5:
            return (
                f"late setup needs stronger rank: {rank_score:.2f} < 0.58 "
                f"after {late_rank_cutoff // 60}:{late_rank_cutoff % 60:02d}"
            )
        if (
            setup_type == "breakout"
            and votes <= 2
            and {"SuperTrend+RSI", "ADX+PSAR"}.issubset(strategies)
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_25
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_48
        ):
            return (
                f"weak two-vote breakout pair blocked "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, setup={setup_strength:.2f})"
            )
        if (
            votes <= 2
            and direction == "BUY_PUT"
            and strategies == {"SuperTrend+RSI", "ADX+PSAR"}
            and (
                (setup_type == "breakout" and (setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_74 or bias == "RANGING"))
                or (
                    setup_type == "vote_aligned"
                    and rank_score < PLANNER_RANK_SCORE_THRESH_0_61
                    and setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_84
                    and not (
                        bias == "BEARISH"
                        and rank_score >= PLANNER_RANK_SCORE_THRESH_0_57
                        and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_82
                    )
                )
            )
        ):
            return (
                f"blocked weak two-vote PUT trend pair "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            votes <= 2
            and direction == "BUY_CALL"
            and setup_type == "vote_aligned"
            and bias == "RANGING"
            and strategies in (
                {"SuperTrend+RSI", "ADX+PSAR"},
                {"VWAP+EMA", "ADX+PSAR"},
            )
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_305
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_55
            and setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_8
        ):
            return (
                f"blocked low-edge ranging CALL trend pair "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            votes <= 2
            and direction == "BUY_PUT"
            and strategies == {"SuperTrend+RSI", "SkewHunter"}
            and not (rank_score >= PLANNER_RANK_SCORE_THRESH_0_53 and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_86)
            and (rank_score < PLANNER_RANK_SCORE_THRESH_0_62 or setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_84)
        ):
            return (
                f"blocked weak two-vote SkewHunter PUT "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            votes <= 2
            and direction == "BUY_CALL"
            and strategies == {"SuperTrend+RSI", "VWAP+EMA"}
            and (rank_score < PLANNER_RANK_SCORE_THRESH_0_56 or bias == "RANGING")
        ):
            return (
                f"blocked weak two-vote CALL trend pair "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type in {"breakout", "vote_aligned"}
            and direction == "BUY_PUT"
            and bias == "RANGING"
            and signal_minute < (9 * 60 + 45)
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_62
        ):
            return (
                f"blocked early ranging PUT breakdown trap "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type in {"vote_aligned", "trend_pullback"}
            and direction == "BUY_PUT"
            and "SkewHunter" in strategies
            and "ADX+PSAR" in strategies
            and "BBSqueeze" not in strategies
            and "VWAP+EMA" not in strategies
            and success_prob < PLANNER_SUCCESS_PROB_THRESH_0_3
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
        ):
            return (
                f"blocked low-probability SkewHunter+ADX PUT continuation "
                f"(p={success_prob:.2f}, rank={rank_score:.2f}, "
                f"setup={setup_strength:.2f}, bias={bias}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "vote_aligned"
            and direction == "BUY_PUT"
            and bias == "RANGING"
            and "SkewHunter" in strategies
            and "ADX+PSAR" not in strategies
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
        ):
            return (
                f"blocked ranging PUT vote stack without ADX confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type == "breakout"
            and direction == "BUY_PUT"
            and "SkewHunter" in strategies
            and votes < 5
            and (
                rank_score < PLANNER_RANK_SCORE_THRESH_0_62
                or setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_74
            )
        ):
            return (
                f"blocked negative-edge PUT breakout with SkewHunter "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"strategies={'+'.join(sorted(strategies))})"
            )
        if (
            setup_type in {"vote_aligned", "trend_pullback"}
            and direction == "BUY_PUT"
            and "SkewHunter" in strategies
            and "VWAP+EMA" not in strategies
            and "ADX+PSAR" not in strategies
            and (
                rank_score < PLANNER_RANK_SCORE_THRESH_0_56
                or setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_68
                or opposite_bias
                or (bias == "RANGING" and "BBSqueeze" not in strategies)
            )
        ):
            return (
                f"blocked weak SkewHunter PUT without trend confirmation "
                f"(rank={rank_score:.2f}, setup={setup_strength:.2f}, "
                f"bias={bias}, strategies={'+'.join(sorted(strategies))})"
            )
        late_squeeze_cutoff = (22 * 60) if is_mcx else (14 * 60)
        if (
            direction == "BUY_PUT"
            and setup_type == "vote_aligned"
            and signal_minute >= late_squeeze_cutoff
            and {"IVContraction", "BBSqueeze"}.intersection(strategies)
            and not {"ADX+PSAR", "VolumeProfile", "OIAnalysis"}.intersection(strategies)
        ):
            return (
                f"blocked late-session unconfirmed squeeze PUT "
                f"(strategies={'+'.join(sorted(strategies))})"
            )
        return ""

    def _validate_market_freshness(self, signal_ts: datetime) -> tuple[bool, str]:
        if self.backtest_mode:
            return True, ""
        expected_day = latest_expected_trading_day(signal_ts.date())
        if signal_ts.date() != expected_day:
            return False, f"stale_signal_timestamp={signal_ts.date()} expected={expected_day}"
        now_ist = datetime.now(signal_ts.tzinfo) if signal_ts.tzinfo else datetime.now()
        age_minutes = max(int((now_ist - signal_ts).total_seconds() // 60), 0)
        if age_minutes > 20:
            return False, f"stale_signal_age={age_minutes}m"
        return True, ""

    def _validate_selected_contract(
        self,
        *,
        symbol: str,
        option_symbol: str,
        expiry_date: date,
        strike: int,
        option_type: str,
        underlying_ltp: float,
        confidence: float,
        premium_source: str,
        direction: str,
        strike_step: int,
    ) -> tuple[bool, str]:
        if premium_source == "COMMODITY_FUTURES" or option_symbol.endswith("FUT") or option_type in ("FUT", "FUTURE"):
            return True, ""
        validation = validate_option_contract(
            symbol=symbol,
            expiry_date=expiry_date,
            strike=strike,
            option_type=option_type,
            underlying=underlying_ltp,
            strike_step=strike_step,
            expected_symbol=option_symbol,
        )
        if not validation.valid:
            return False, validation.reason
        allowed_moneyness = self._allowed_moneyness(direction, confidence)
        if validation.moneyness not in allowed_moneyness:
            return False, (
                f"strike_moneyness={validation.moneyness} not_allowed="
                f"{'/'.join(sorted(allowed_moneyness))}"
            )
        if not self.backtest_mode and premium_source != "LIVE":
            current_mode = str(os.getenv("TRADING_MODE", TRADING_MODE)).strip().upper()
            if current_mode == "OBSERVE" or premium_source in ("ESTIMATED_OBSERVE", "ESTIMATED"):
                return True, ""
            return False, "no_live_option_quote"
        return True, ""

    @staticmethod
    def _extract_votes_count(votes_obj) -> int:
        if isinstance(votes_obj, dict):
            return sum(1 for v in votes_obj.values() if v)
        try:
            return int(votes_obj or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _should_block_late_session_setup(
        *,
        minutes_to_close: int,
        rank_score: float,
        setup_strength: float,
    ) -> bool:
        return (
            minutes_to_close <= 75
            and rank_score < PLANNER_RANK_SCORE_THRESH_0_6
            and setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_82
        )

    @staticmethod
    def _expiry_trade_allowed(
        *,
        symbol: str,
        signal_ts: datetime,
        dte: int,
        rank_score: float,
        setup: dict,
        votes: int,
        direction: str,
    ) -> tuple[bool, str]:
        if dte > 0:
            return True, ""
        if not is_expiry_day(signal_ts, symbol=symbol):
            return False, f"{symbol} DTE=0 but signal day is not weekly expiry"
        from config.settings.modules.session_policy import is_expiry_option_freeze
        is_frozen, freeze_reason = is_expiry_option_freeze(signal_ts, signal_ts.date())
        if is_frozen:
            return False, freeze_reason

        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        mins = signal_ts.hour * 60 + signal_ts.minute
        if setup_type not in {"breakout", "trend_pullback", "vote_aligned"}:
            return False, f"expiry-day setup_type={setup_type} not allowed"
        if direction not in {"BUY_CALL", "BUY_PUT"}:
            return False, "expiry-day direction invalid"
        if mins < 13 * 60:
            early_quality_ok = (
                mins >= EXPIRY_EARLY_ENTRY_MINUTE
                and rank_score >= EXPIRY_EARLY_MIN_RANK
                and setup_strength >= EXPIRY_EARLY_MIN_SETUP
                and votes >= EXPIRY_EARLY_MIN_VOTES
            )
            if early_quality_ok:
                return True, ""
            return False, "expiry-day trade before 13:00 blocked"
        if rank_score < PLANNER_RANK_SCORE_THRESH_0_45 or setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_5 or votes < 2:
            return False, (
                f"expiry-day setup too weak rank={rank_score:.2f} "
                f"setup={setup_strength:.2f} votes={votes}"
            )
        return True, ""

    def _resolve_instrument_spot(self, data: dict, instrument: InstrumentConfig) -> float:
        """
        Resolve the spot used for strike selection.
        Checks instrument-specific keys (e.g. crudeoil_ltp, silvermic_ltp, spot_ltp, ltp, close).
        """
        lower = instrument.name.lower()
        keys = (f"{lower}_ltp", f"{lower}_price", "spot_ltp", "ltp", "price", "close", "nifty_ltp", "nifty_price")
        for key in keys:
            value = float(data.get(key, 0) or 0)
            if value > 0:
                return value
        try:
            if self.data_agent and hasattr(self.data_agent, "get_ltp"):
                value = float(self.data_agent.get_ltp(instrument.name) or 0)
                if value > 0:
                    return value
            if self.broker and hasattr(self.broker, "get_ltp"):
                value = float(self.broker.get_ltp(instrument.name) or 0)
                if value > 0:
                    return value
        except Exception:
            pass
        return 0.0

    async def on_approved(self, msg: Message) -> None:
        try:
            data       = msg.payload
            signal_ts  = self._resolve_signal_ts(data.get("timestamp"))
            metadata = data.get("metadata", {}) or {}
            instrument_override = data.get("instrument", None) or data.get("symbol", None)
            if not instrument_override:
                for meta in metadata.values():
                    if isinstance(meta, dict):
                        m_cand = str(meta.get("instrument") or meta.get("symbol") or "").upper()
                        if m_cand in SUPPORTED_INDEX_SYMBOLS:
                            instrument_override = m_cand
                            break
            if not instrument_override:
                instrument_override = os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))
            # ── Instrument selection (Commodity Futures / Options) ───────────────────
            instrument: InstrumentConfig = get_instrument(
                override=instrument_override,
                trade_date=signal_ts.date(),
                verbose=True,
            )
            logger.info(
                f"[{self.NAME}] Instrument: {instrument.name} | "
                f"lot={instrument.lot_size} | step={instrument.strike_step} | "
                f"expiry={instrument.expiry_day}"
            )
            # ── Expiry: instrument-specific (Tuesday for NIFTY, Friday for SENSEX) ──
            expiry_date = instrument.get_next_expiry(signal_ts.date())
            dte = instrument.get_dte(signal_ts.date())

            # Expiry Engine: In LIVE trading on Expiry Day (0 DTE) after 11:30 AM, switch option buying to Next Weekly Expiry (D+7)
            is_backtest_mode = str(os.getenv("TRADING_MODE", "")).upper() == "BACKTEST"
            if not is_backtest_mode and dte == 0 and signal_ts.time() >= datetime.strptime("11:30", "%H:%M").time():
                from utils.option_utils import get_weekly_expiry
                expiry_date = get_weekly_expiry(signal_ts.date() + timedelta(days=7), symbol=instrument.name)
                dte = (expiry_date - signal_ts.date()).days
                logger.info(
                    f"[{self.NAME}] 🔄 Expiry Engine: Switched to Next Weekly Expiry (D+7) "
                    f"due to Expiry Day after 11:30 AM | target_expiry={expiry_date} | dte={dte}"
                )


            if not bool(data.get("ml_approved", False)):
                reason = str(data.get("ml_decision_reason", "") or data.get("rejection_reason", "") or "ml_not_approved")
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason=reason,
                    market_ts=signal_ts.isoformat(),
                    direction=str(data.get("direction", "NONE")),
                    votes=self._extract_votes_count(data.get("votes", 0)),
                )
                logger.warning(
                    f"[{self.NAME}] Skipping unapproved signal | "
                    f"market_ts={signal_ts.isoformat()} | reason={reason}"
                )
                return
            symbol = instrument.name
            if symbol not in SUPPORTED_INDEX_SYMBOLS:
                logger.info(f"[{self.NAME}] Unsupported symbol for planner: {symbol}")
                return
            lot_size = instrument.lot_size
            strike_step = instrument.strike_step
            nifty_ltp = self._resolve_instrument_spot(data, instrument)
            rank_score = float(data.get("ml_rank_score", data.get("ml_confidence", data.get("confidence", 0.65))))
            success_prob = float(data.get("ml_confidence", data.get("confidence", 0.65)))
            confidence = self._execution_confidence(
                data,
                rank_score=rank_score,
                success_prob=success_prob,
            )
            direction  = data.get("direction", "NONE")
            if direction == "NONE" or nifty_ltp == 0:
                return
            if self.backtest_mode:
                use_adaptive_edge_filters = os.getenv(
                    "BACKTEST_ENABLE_ADAPTIVE_EDGE_FILTERS",
                    "false",
                ).strip().lower() in {"1", "true", "yes", "on"}
            else:
                use_adaptive_edge_filters = os.getenv(
                    "LIVE_ENABLE_ADAPTIVE_EDGE_FILTERS",
                    "false",
                ).strip().lower() in {"1", "true", "yes", "on"}
            theta = get_expiry_theta().check() if use_adaptive_edge_filters else None
            if theta and (theta.force_close_now or theta.avoid_new_entries):
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="expiry_theta_gate",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    theta_note=theta.note,
                    premium_remaining=f"{theta.premium_remaining_pct:.3f}",
                )
                logger.warning(f"[{self.NAME}] Expiry theta gate blocked new trade | {theta.note}")
                return
            market_fresh, fresh_reason = self._validate_market_freshness(signal_ts)

            if not market_fresh:
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason=fresh_reason,
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                )
                logger.warning(
                    f"[{self.NAME}] Skipping stale signal context | "
                    f"market_ts={signal_ts.isoformat()} | reason={fresh_reason}"
                )
                return
            reduced_budget_requested, reduced_budget_cap, reduced_budget_reason, reduced_budget_min_ml_prob = self._reduced_budget_requested(data)
            votes = self._extract_votes_count(data.get("votes", 0))
            early_trigger = bool(
                (((data.get("metadata") or {}).get("_context") or {}).get("early_trigger", False))
            )
            hero_zero_active = "HeroZero" in set(data.get("strategies_fired", []) or [])
            if votes < MIN_STRATEGY_VOTES and not self._allows_low_consensus_plan(data) and not reduced_budget_requested:
                await self._publish_planner_rejection(
                    data,
                    reason="low_consensus_setup",
                    details=f"votes={votes} < {MIN_STRATEGY_VOTES}",
                )
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="low_consensus_setup",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    votes=votes,
                    min_votes=MIN_STRATEGY_VOTES,
                    ml_rank=f"{rank_score:.2f}",
                    ml_prob=f"{success_prob:.2f}",
                )
                logger.info(
                    f"[{self.NAME}] Skipping low-consensus setup | "
                    f"market_ts={signal_ts.isoformat()} | votes={votes} < {MIN_STRATEGY_VOTES}"
                )
                return
            minutes_to_close = self._minutes_to_close(signal_ts)
            if minutes_to_close < MIN_ENTRY_MINUTES_BEFORE_CLOSE and not hero_zero_active:
                await self._publish_planner_rejection(
                    data,
                    reason="late_session_setup",
                    details=f"minutes_to_close={minutes_to_close} < {MIN_ENTRY_MINUTES_BEFORE_CLOSE}",
                )
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="late_session_setup",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    minutes_to_close=minutes_to_close,
                    min_required=MIN_ENTRY_MINUTES_BEFORE_CLOSE,
                )
                logger.info(
                    f"[{self.NAME}] Skipping late-session setup | "
                    f"market_ts={signal_ts.isoformat()} | minutes_to_close={minutes_to_close} "
                    f"< required {MIN_ENTRY_MINUTES_BEFORE_CLOSE}"
                )
                return
            if hero_zero_active and minutes_to_close < MIN_ENTRY_MINUTES_BEFORE_CLOSE:
                logger.info(
                    f"[{self.NAME}] HeroZero late-session override | "
                    f"market_ts={signal_ts.isoformat()} | minutes_to_close={minutes_to_close} "
                    f"< generic_required {MIN_ENTRY_MINUTES_BEFORE_CLOSE}"
                )
            setup_ctx = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
            setup_strength = float(setup_ctx.get("setup_strength", 0.0) or 0.0)
            strategy_confidence = float(data.get("confidence", 0.0) or 0.0)
            signal_minute = signal_ts.hour * 60 + signal_ts.minute
            if (
                PLANNER_MIN_STRATEGY_CONFIDENCE > 0
                and strategy_confidence < PLANNER_MIN_STRATEGY_CONFIDENCE
            ):
                await self._publish_planner_rejection(
                    data,
                    reason="strategy_confidence_below_min",
                    details=(
                        f"strategy_conf={strategy_confidence:.2f} "
                        f"< {PLANNER_MIN_STRATEGY_CONFIDENCE:.2f}"
                    ),
                )
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="strategy_confidence_below_min",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    strategy_conf=f"{strategy_confidence:.2f}",
                    min_conf=f"{PLANNER_MIN_STRATEGY_CONFIDENCE:.2f}",
                    ml_rank=f"{rank_score:.2f}",
                    ml_prob=f"{success_prob:.2f}",
                    setup_strength=f"{setup_strength:.2f}",
                    votes=votes,
                )
                logger.info(
                    f"[{self.NAME}] Skipping low-confidence execution | "
                    f"market_ts={signal_ts.isoformat()} | strategy_conf={strategy_confidence:.2f} "
                    f"< {PLANNER_MIN_STRATEGY_CONFIDENCE:.2f}"
                )
                return
            is_mcx = symbol in SUPPORTED_INDEX_SYMBOLS or any(k in str(symbol).upper() for k in ("SILVER", "GOLD", "CRUDE", "NAT"))
            late_rank_cutoff = (21 * 60 + 15) if is_mcx else (13 * 60 + 15)
            if (
                signal_minute >= late_rank_cutoff
                and strategy_confidence < 0.55
                and rank_score < PLANNER_RANK_SCORE_THRESH_0_55
                and votes <= 3
            ):
                await self._publish_planner_rejection(
                    data,
                    reason="late_low_confidence_setup",
                    details=f"strategy_conf={strategy_confidence:.2f}, rank={rank_score:.2f}, votes={votes}",
                )
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="late_low_confidence_setup",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    strategy_conf=f"{strategy_confidence:.2f}",
                    ml_rank=f"{rank_score:.2f}",
                    ml_prob=f"{success_prob:.2f}",
                    setup_strength=f"{setup_strength:.2f}",
                    votes=votes,
                )
                logger.info(
                    f"[{self.NAME}] Skipping late low-confidence setup | "
                    f"market_ts={signal_ts.isoformat()} | strategy_conf={strategy_confidence:.2f} "
                    f"rank={rank_score:.2f} votes={votes}"
                )
                return
            if rank_score < EXECUTION_MIN_RANK_SCORE and not reduced_budget_requested:
                await self._publish_planner_rejection(
                    data,
                    reason="execution_rank_below_min",
                    details=f"rank={rank_score:.2f} < {EXECUTION_MIN_RANK_SCORE:.2f}",
                )
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="execution_rank_below_min",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    ml_rank=f"{rank_score:.2f}",
                    min_rank=f"{EXECUTION_MIN_RANK_SCORE:.2f}",
                    setup_strength=f"{setup_strength:.2f}",
                    votes=votes,
                )
                logger.info(
                    f"[{self.NAME}] Skipping low execution-rank setup | "
                    f"market_ts={signal_ts.isoformat()} | ml_rank={rank_score:.2f} "
                    f"< {EXECUTION_MIN_RANK_SCORE:.2f}"
                )
                return
            weak_entry_reason = self._weak_entry_block_reason(
                data=data,
                signal_ts=signal_ts,
                rank_score=rank_score,
                success_prob=success_prob,
                setup=setup_ctx,
                votes=votes,
                direction=direction,
            )
            if weak_entry_reason and (votes < 6 and rank_score < 0.65):
                # Structural blocks, counter-trend entries, and climax exhaustions must NEVER be bypassed to reduced budget
                is_hard_structural_block = (
                    "counter-bias trend_pullback" in weak_entry_reason
                    or "counter-HTF" in weak_entry_reason
                    or "RSI exhaustion" in weak_entry_reason
                    or "timing exhaustion" in weak_entry_reason
                    or "high chop" in weak_entry_reason
                    or "overextended" in weak_entry_reason
                    or "Wyckoff" in weak_entry_reason
                    or "late-day" in weak_entry_reason
                    or weak_entry_reason.startswith("blocked ")
                )
                reduced_budget_weak_entry_standard = (
                    not is_hard_structural_block
                    and votes >= REDUCED_BUDGET_MIN_VOTES
                    and rank_score >= REDUCED_BUDGET_MIN_RANK
                    and setup_strength >= REDUCED_BUDGET_MIN_SETUP
                )
                reduced_budget_weak_entry_strong = (
                    not is_hard_structural_block
                    and votes >= REDUCED_BUDGET_STRONG_MIN_VOTES
                    and rank_score >= REDUCED_BUDGET_STRONG_MIN_RANK
                    and setup_strength >= REDUCED_BUDGET_STRONG_MIN_SETUP
                )
                reduced_budget_weak_entry = (
                    reduced_budget_weak_entry_standard
                    or reduced_budget_weak_entry_strong
                )
                if reduced_budget_weak_entry:
                    reduced_reason = (
                        f"reduced_budget_weak_entry: {weak_entry_reason}; "
                        f"votes={votes}, rank={rank_score:.2f}, setup={setup_strength:.2f}"
                    )
                    data["metadata"] = {
                        **(data.get("metadata") or {}),
                        **self._reduced_budget_meta(reduced_reason),
                    }
                    reduced_budget_requested = True
                    logger.info(
                        f"[{self.NAME}] Weak entry allowed with reduced budget | "
                        f"market_ts={signal_ts.isoformat()} | {reduced_reason}"
                    )
                    log_pipeline_stage(
                        self.NAME,
                        "trade_planning",
                        "passed",
                        reason="reduced_budget_weak_entry",
                        market_ts=signal_ts.isoformat(),
                        direction=direction,
                        ml_rank=f"{rank_score:.2f}",
                        ml_prob=f"{success_prob:.2f}",
                        setup_strength=f"{setup_strength:.2f}",
                        votes=votes,
                        max_investment=f"{REDUCED_BUDGET_MAX_TRADE_INR:.0f}",
                        details=weak_entry_reason,
                    )
                else:
                    await self._publish_planner_rejection(
                        data,
                        reason="weak_entry_quality",
                        details=weak_entry_reason,
                    )
                    log_pipeline_stage(
                        self.NAME,
                        "trade_planning",
                        "filtered",
                        reason="weak_entry_quality",
                        market_ts=signal_ts.isoformat(),
                        direction=direction,
                        ml_rank=f"{rank_score:.2f}",
                        ml_prob=f"{success_prob:.2f}",
                        setup_strength=f"{setup_strength:.2f}",
                        votes=votes,
                        details=weak_entry_reason,
                    )
                    logger.info(
                        f"[{self.NAME}] Skipping weak entry | "
                        f"market_ts={signal_ts.isoformat()} | {weak_entry_reason}"
                    )
                    return
            reduced_budget_requested, _, _, reduced_budget_min_ml_prob = self._reduced_budget_requested(data)
            if reduced_budget_requested and success_prob < reduced_budget_min_ml_prob:
                details = (
                    f"reduced_budget_low_ml_prob: prob={success_prob:.2f} "
                    f"< {reduced_budget_min_ml_prob:.2f}, rank={rank_score:.2f}, votes={votes}"
                )
                await self._publish_planner_rejection(
                    data,
                    reason="reduced_budget_low_ml_prob",
                    details=details,
                )
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="reduced_budget_low_ml_prob",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    ml_rank=f"{rank_score:.2f}",
                    ml_prob=f"{success_prob:.2f}",
                    setup_strength=f"{setup_strength:.2f}",
                    votes=votes,
                )
                logger.info(
                    f"[{self.NAME}] Skipping reduced-budget low-probability setup | "
                    f"market_ts={signal_ts.isoformat()} | {details}"
                )
                return
            if (
                not reduced_budget_requested
                and votes <= LOW_VOTE_MIN_VOTES
                and setup_strength < 0.75
                and rank_score < LOW_VOTE_MIN_RANK
                and success_prob < LOW_VOTE_MIN_PROB
            ):
                reduced_budget_low_vote_eligible = (
                    votes >= REDUCED_BUDGET_MIN_VOTES
                    and rank_score >= REDUCED_BUDGET_MIN_RANK
                    and setup_strength >= REDUCED_BUDGET_MIN_SETUP
                )
                if reduced_budget_low_vote_eligible:
                    reduced_reason = (
                        f"reduced_budget_low_vote_demotion: votes={votes} <= {LOW_VOTE_MIN_VOTES}, "
                        f"rank={rank_score:.2f}, prob={success_prob:.2f}, setup={setup_strength:.2f}"
                    )
                    data["metadata"] = {
                        **(data.get("metadata") or {}),
                        **self._reduced_budget_meta(reduced_reason),
                    }
                    reduced_budget_requested = True
                    logger.info(
                        f"[{self.NAME}] Low-vote/low-ML setup allowed with reduced budget | "
                        f"market_ts={signal_ts.isoformat()} | {reduced_reason}"
                    )
                    log_pipeline_stage(
                        self.NAME,
                        "trade_planning",
                        "passed",
                        reason="reduced_budget_low_vote_demotion",
                        market_ts=signal_ts.isoformat(),
                        direction=direction,
                        ml_rank=f"{rank_score:.2f}",
                        ml_prob=f"{success_prob:.2f}",
                        setup_strength=f"{setup_strength:.2f}",
                        votes=votes,
                        max_investment=f"{REDUCED_BUDGET_MAX_TRADE_INR:.0f}",
                        details=reduced_reason,
                    )
                else:
                    details = (
                        f"low_vote_low_ml_quality: votes={votes} <= {LOW_VOTE_MIN_VOTES}, "
                        f"rank={rank_score:.2f} < {LOW_VOTE_MIN_RANK:.2f}, "
                        f"prob={success_prob:.2f} < {LOW_VOTE_MIN_PROB:.2f}"
                    )
                    await self._publish_planner_rejection(
                        data,
                        reason="low_vote_low_ml_quality",
                        details=details,
                    )
                    log_pipeline_stage(
                        self.NAME,
                        "trade_planning",
                        "filtered",
                        reason="low_vote_low_ml_quality",
                        market_ts=signal_ts.isoformat(),
                        direction=direction,
                        ml_rank=f"{rank_score:.2f}",
                        ml_prob=f"{success_prob:.2f}",
                        setup_strength=f"{setup_strength:.2f}",
                        votes=votes,
                    )
                    logger.info(
                        f"[{self.NAME}] Skipping low-vote low-ML setup | "
                        f"market_ts={signal_ts.isoformat()} | {details}"
                    )
                    return
            is_mcx = symbol in SUPPORTED_INDEX_SYMBOLS or any(k in str(symbol).upper() for k in ("SILVER", "GOLD", "CRUDE", "NAT"))
            late_cond_cutoff = (22 * 60 + 30) if is_mcx else (15 * 60)
            if rank_score < PLANNER_RANK_SCORE_THRESH_0_48 and not reduced_budget_requested:
                if success_prob < PLANNER_SUCCESS_PROB_THRESH_0_4 or signal_minute >= late_cond_cutoff:
                    await self._publish_planner_rejection(
                        data,
                        reason="conditional_rank_guard",
                        details=f"rank={rank_score:.2f}, prob={success_prob:.2f}",
                    )
                    log_pipeline_stage(
                        self.NAME,
                        "trade_planning",
                        "filtered",
                        reason="conditional_rank_guard",
                        market_ts=signal_ts.isoformat(),
                        direction=direction,
                        ml_rank=f"{rank_score:.2f}",
                        ml_prob=f"{success_prob:.2f}",
                        setup_strength=f"{setup_strength:.2f}",
                        votes=votes,
                    )
                    logger.info(
                        f"[{self.NAME}] Skipping conditional rank setup | "
                        f"market_ts={signal_ts.isoformat()} | ml_rank={rank_score:.2f} "
                        f"prob={success_prob:.2f}"
                    )
                    return
            quality_class = str(
                data.get("quality_classification")
                or (data.get("metadata") or {}).get("quality_classification")
                or (data.get("signal") or {}).get("quality_classification")
                or ""
            ).upper()
            if quality_class == "LOW_QUALITY" and not reduced_budget_requested and (votes < 5 and rank_score < 0.60):
                reduced_reason = (
                    f"reduced_budget_low_quality_demotion: quality_classification={quality_class}, "
                    f"rank={rank_score:.2f}, prob={success_prob:.2f}, votes={votes}"
                )
                data["metadata"] = {
                    **(data.get("metadata") or {}),
                    **self._reduced_budget_meta(reduced_reason),
                }
                reduced_budget_requested = True
                logger.info(
                    f"[{self.NAME}] LOW_QUALITY setup demoted to reduced budget | "
                    f"market_ts={signal_ts.isoformat()} | {reduced_reason}"
                )

            if signal_minute > EXECUTION_LAST_ENTRY_MINUTE:
                await self._publish_planner_rejection(
                    data,
                    reason="execution_entry_after_cutoff",
                    details=f"entry_minute={signal_minute} > {EXECUTION_LAST_ENTRY_MINUTE}",
                )
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="execution_entry_after_cutoff",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    entry_minute=signal_minute,
                    cutoff_minute=EXECUTION_LAST_ENTRY_MINUTE,
                    ml_rank=f"{rank_score:.2f}",
                )
                logger.info(
                    f"[{self.NAME}] Skipping post-cutoff setup | "
                    f"market_ts={signal_ts.isoformat()} | entry_minute={signal_minute} "
                    f"> {EXECUTION_LAST_ENTRY_MINUTE}"
                )
                return
            if not reduced_budget_requested and self._should_block_late_session_setup(
                minutes_to_close=minutes_to_close,
                rank_score=rank_score,
                setup_strength=setup_strength,
            ):
                await self._publish_planner_rejection(
                    data,
                    reason="late_session_low_momentum",
                    details=(
                        f"minutes_to_close={minutes_to_close}, "
                        f"rank={rank_score:.2f}, setup={setup_strength:.2f}"
                    ),
                )
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="late_session_low_momentum",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    minutes_to_close=minutes_to_close,
                    ml_rank=f"{rank_score:.2f}",
                    setup_strength=f"{setup_strength:.2f}",
                )
                logger.info(
                    f"[{self.NAME}] Skipping late-session low-momentum setup | "
                    f"market_ts={signal_ts.isoformat()} | minutes_to_close={minutes_to_close} | "
                    f"ml_rank={rank_score:.2f} | setup={setup_strength:.2f}"
                )
                return

            atm    = get_atm_strike(nifty_ltp, strike_step)
            opt_t  = "CE" if direction == "BUY_CALL" else "PE"

            # Expiry
            expiry_min_days = 0 if is_expiry_day(signal_ts, symbol=symbol) else MIN_DAYS_TO_EXPIRY
            expiry_date, dte = get_nearest_expiry(
                expiry_min_days,
                reference_date=signal_ts,
                symbol=symbol,
            )
            expiry_ok, expiry_reason = self._expiry_trade_allowed(
                symbol=symbol,
                signal_ts=signal_ts,
                dte=dte,
                rank_score=rank_score,
                setup=setup_ctx,
                votes=votes,
                direction=direction,
            )
            if not expiry_ok and not reduced_budget_requested:
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="expiry_safety_gate",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    dte=dte,
                    details=expiry_reason,
                )
                logger.warning(f"[{self.NAME}] Expiry safety gate blocked trade: {expiry_reason}")
                return

            candidates, selection_notes = self._select_contracts(
                symbol=symbol,
                nifty_ltp=nifty_ltp,
                confidence=confidence,
                direction=direction,
                atm=atm,
                expiry_date=expiry_date,
                dte=dte,
                lot_size=lot_size,
                strike_step=strike_step,
                signal_data=data,
            )
            if not candidates:
                details = " | ".join(selection_notes[:6]) if selection_notes else "no candidate diagnostics"
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="no_viable_contracts",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    ml_rank=f"{rank_score:.2f}",
                    ml_prob=f"{success_prob:.2f}",
                    notes=details,
                )
                logger.warning(f"[{self.NAME}] No viable option contracts found. {details}")
                return

            validation_confidence = max(confidence, float(data.get("confidence", 0.0) or 0.0))
            valid_candidates: list[dict] = []
            for candidate in candidates:
                cand_sym = str(candidate["symbol"])
                cand_source = str(candidate.get("premium_source", ""))
                if not self.backtest_mode and cand_source != "LIVE":
                    refreshed_ltp = 0.0
                    if self.broker:
                        try:
                            refreshed_ltp = float(self.broker.get_option_ltp(cand_sym) or 0.0)
                        except Exception:
                            refreshed_ltp = 0.0
                    if refreshed_ltp <= 0 and self.data_agent:
                        try:
                            refreshed_ltp = float(self.data_agent.get_option_ltp(cand_sym) or 0.0)
                        except Exception:
                            refreshed_ltp = 0.0
                    if refreshed_ltp > 0:
                        candidate["premium"] = refreshed_ltp
                        candidate["premium_source"] = "LIVE"
                        cand_source = "LIVE"
                        if isinstance(candidate.get("snapshot"), dict):
                            candidate["snapshot"]["last_price"] = refreshed_ltp
                            candidate["snapshot"]["source"] = "DHAN_LIVE"
                        logger.info(
                            f"[{self.NAME}] Successfully recovered live quote for candidate | "
                            f"symbol={cand_sym} | ltp={refreshed_ltp}"
                        )
                candidate_valid, candidate_reason = self._validate_selected_contract(
                    symbol=symbol,
                    option_symbol=cand_sym,
                    expiry_date=expiry_date,
                    strike=int(candidate["strike"]),
                    option_type=opt_t,
                    underlying_ltp=nifty_ltp,
                    confidence=validation_confidence,
                    premium_source=cand_source,
                    direction=direction,
                    strike_step=strike_step,
                )
                if candidate_valid:
                    valid_candidates.append(candidate)
                else:
                    selection_notes.append(
                        f"Skipped {candidate['symbol']}: validation failed ({candidate_reason})"
                    )
                    logger.info(
                        f"[{self.NAME}] Candidate validation skipped | "
                        f"market_ts={signal_ts.isoformat()} | contract={candidate['symbol']} | "
                        f"reason={candidate_reason}"
                    )
            if not valid_candidates:
                details = " | ".join(selection_notes[-6:]) if selection_notes else "all candidates failed validation"
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="no_valid_contract_after_validation",
                    market_ts=signal_ts.isoformat(),
                    direction=direction,
                    ml_rank=f"{rank_score:.2f}",
                    notes=details,
                )
                logger.warning(f"[{self.NAME}] No valid option contracts after validation. {details}")
                return

            best = valid_candidates[0]
            is_comm = "SILVER" in str(symbol).upper() or "GOLD" in str(symbol).upper() or "CRUDE" in str(symbol).upper()
            min_contract_score = 0.30 if is_comm else (0.40 if (reduced_budget_requested or votes >= 6 or rank_score >= 0.65) else 0.54)
            if float(best["score"]) < min_contract_score:
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="low_contract_score",
                    market_ts=signal_ts.isoformat(),
                    contract_score=f"{float(best['score']):.3f}",
                    direction=direction,
                )
                logger.info(f"[{self.NAME}] Skipping trade due to low contract score: {best['score']:.3f} < {min_contract_score}")
                return

            strike = best["strike"]
            option_symbol = best["symbol"]
            est_premium = float(best["premium"])
            premium_source = best["premium_source"]
            spot_sl_info: dict = {}
            recent_df = None
            try:
                recent_df = self.data_agent.get_recent_candles(120) if self.data_agent else None
                iv_proxy = get_current_iv_proxy(recent_df, float(data.get("vix", 0) or 0))
                em_sel = select_strike_em(
                    spot=nifty_ltp,
                    direction=direction,
                    dte=dte,
                    iv=iv_proxy,
                    step=strike_step,
                )
                if em_sel.strike != strike:
                    selection_notes.append(
                        f"EM strike advisory: {em_sel.strike} ({em_sel.note}); "
                        f"kept scored contract {strike}"
                    )
                if best.get("premium_source") == "COMMODITY_FUTURES" or best.get("opt_type") == "FUT":
                    atr_pts = 0.0
                    if recent_df is not None and not recent_df.empty:
                        from utils.instrument_selector import compute_atr
                        atr_pts, _ = compute_atr(recent_df, 14)
                    inst_cfg = get_instrument(symbol)
                    stop_dist = max(inst_cfg.tick_size * 20.0, atr_pts if atr_pts > 0 else nifty_ltp * 0.005)
                    is_buy = direction in ("BUY", "BUY_CALL")
                    sl_premium_override = round(nifty_ltp - stop_dist if is_buy else nifty_ltp + stop_dist, 2)
                    target_premium_override = round(nifty_ltp + (stop_dist * 2.0) if is_buy else nifty_ltp - (stop_dist * 2.0), 2)
                    best["desired_lots"] = 1
                else:
                    exit_params = compute_atr_sl(
                        recent_df,
                        entry_premium=est_premium,
                        regime=str(data.get("regime", "TRENDING")),
                        adx=float((((data.get("metadata") or {}).get("_context") or {}).get("adx", 20.0)) or 20.0),
                    )
                    sl_premium_override = float(exit_params.sl_premium)
                    target_premium_override = float(exit_params.target_prem)
                    delta_size = size_by_delta(
                        spot=nifty_ltp,
                        strike=strike,
                        dte=dte,
                        iv=iv_proxy,
                        option_type=opt_t,
                        entry_premium=est_premium,
                        lot_size=lot_size,
                        capital_budget=float(best.get("risk_budget_inr", 0.0) or 0.0),
                    )
                    if delta_size.lots <= 0:
                        if reduced_budget_requested:
                            best["desired_lots"] = 1
                        else:
                            logger.info(f"[{self.NAME}] Delta sizing rejected contract | {delta_size.note}")
                            return
                    elif not (int(best.get("desired_lots", 1) or 1) >= 2 or self._extract_votes_count(data.get("votes", 0)) >= 5 or rank_score >= 0.65):
                        best["desired_lots"] = min(int(best["desired_lots"]), int(delta_size.lots))
                wyckoff_size = float(
                    (((data.get("metadata") or {}).get("_context") or {}).get("wyckoff_size_multiplier", 1.0))
                    or 1.0
                )
                if wyckoff_size < 1.0:
                    before_lots = int(best["desired_lots"])
                    if self._should_apply_wyckoff_size_cap(
                        direction=direction,
                        strategies=data.get("strategies_fired", []),
                        setup=((data.get("metadata") or {}).get("_context") or {}).get("setup", {}),
                        ml_rank_score=rank_score,
                        structure_bias=str(data.get("structure_bias", "") or ""),
                        dte=dte,
                        signal_ts=signal_ts,
                    ):
                        best["desired_lots"] = max(1, int(before_lots * wyckoff_size))
                    if int(best["desired_lots"]) < before_lots:
                        selection_notes.append(
                            f"Wyckoff size cap: {before_lots}->{int(best['desired_lots'])} lots "
                            f"(mult={wyckoff_size:.2f})"
                        )
                    else:
                        policy_lots_after_delta = self._apply_lot_policy(
                            desired_lots=int(best["desired_lots"]),
                            lot_size=lot_size,
                            direction=direction,
                            strategies=data.get("strategies_fired", []),
                            setup=((data.get("metadata") or {}).get("_context") or {}).get("setup", {}),
                            ml_rank_score=rank_score,
                            structure_bias=self._structure_bias_for_signal(data),
                            dte=dte,
                            signal_ts=signal_ts,
                        )
                        if policy_lots_after_delta > int(best["desired_lots"]):
                            selection_notes.append(
                                f"Strategy lot restore after Wyckoff bypass: "
                                f"{int(best['desired_lots'])}->{policy_lots_after_delta}"
                            )
                            best["desired_lots"] = policy_lots_after_delta
                        selection_notes.append(
                            f"Wyckoff cap bypassed for confirmed edge "
                            f"(mult={wyckoff_size:.2f})"
                        )
                else:
                    if not reduced_budget_requested:
                        policy_lots_after_delta = self._apply_lot_policy(
                            desired_lots=int(best["desired_lots"]),
                            lot_size=lot_size,
                            direction=direction,
                            strategies=data.get("strategies_fired", []),
                            setup=((data.get("metadata") or {}).get("_context") or {}).get("setup", {}),
                            ml_rank_score=rank_score,
                            structure_bias=self._structure_bias_for_signal(data),
                            dte=dte,
                            signal_ts=signal_ts,
                        )
                        if policy_lots_after_delta > int(best["desired_lots"]):
                            selection_notes.append(
                                f"Strategy lot restore after delta sizing: "
                                f"{int(best['desired_lots'])}->{policy_lots_after_delta}"
                            )
                            best["desired_lots"] = policy_lots_after_delta
                best["quantity"] = int(best["desired_lots"]) * lot_size
                best["selection_note_delta"] = delta_size.note
                selection_notes.append(f"Dynamic exit: {exit_params.method} | {exit_params.note}")
                selection_notes.append(f"Delta sizing: {delta_size.note}")
            except Exception as exc:
                logger.debug(f"[{self.NAME}] Dynamic exit/delta sizing skipped: {exc}")
                sl_premium_override = None
                target_premium_override = None
            selected_valid, selected_reason = self._validate_selected_contract(
                symbol=symbol,
                option_symbol=option_symbol,
                expiry_date=expiry_date,
                strike=strike,
                option_type=opt_t,
                underlying_ltp=nifty_ltp,
                confidence=validation_confidence,
                premium_source=premium_source,
                direction=direction,
                strike_step=strike_step,
            )
            if not selected_valid:
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason=selected_reason,
                    market_ts=signal_ts.isoformat(),
                    contract=option_symbol,
                    direction=direction,
                    strike=strike,
                    premium_source=premium_source,
                )
                logger.warning(
                    f"[{self.NAME}] Skipping invalid contract | "
                    f"market_ts={signal_ts.isoformat()} | contract={option_symbol} | "
                    f"reason={selected_reason}"
                )
                return

            executable = premium_source == "LIVE"
            block_reason = ""
            if not executable and TRADING_MODE.upper() != "OBSERVE":
                block_reason = (
                    "Live option premium unavailable from active broker. "
                    "Trade kept for observation only."
                )

            sl_premium = float(sl_premium_override if sl_premium_override is not None else best["sl_premium"])
            target_premium = float(target_premium_override if target_premium_override is not None else best["target2_premium"])
            quantity = int(best["quantity"])
            total_invested = round(est_premium * quantity, 2)

            # TASK Integration: Smart Money Filter + Invalidation SL
            try:
                # Find key support/resistance from market structure
                market_structure = ((data.get("metadata", {}) or {}).get("_context", {}) or {}).get("market_structure", {}) or {}
                zones = market_structure.get("liquidity_zones", [])
                
                if direction == "BUY_CALL":
                    lower_zones = [z for z in zones if float(z["price"]) < nifty_ltp]
                    key_level = max([float(z["price"]) for z in lower_zones], default=nifty_ltp * 0.99)
                else:
                    upper_zones = [z for z in zones if float(z["price"]) > nifty_ltp]
                    key_level = min([float(z["price"]) for z in upper_zones], default=nifty_ltp * 1.01)

                snap = best.get("snapshot", {})
                snap_source = str(snap.get("source", premium_source) or premium_source).upper()
                raw_delta = float(snap.get("delta", 0.48) or 0.48)
                raw_gamma = float(snap.get("gamma", 0.0) or 0.0)
                raw_oi = float(snap.get("open_interest", 0.0) or 0.0)
                try:
                    from utils.spot_based_sl import compute_spot_sl_for_trade
                    spot_sl_info = compute_spot_sl_for_trade(
                        df=recent_df,
                        spot=nifty_ltp,
                        direction=direction,
                        entry_premium=est_premium,
                        delta=raw_delta,
                        lot_size=lot_size,
                    )
                    selection_notes.append(
                        f"Spot SL: {spot_sl_info.get('sl_source')} @ "
                        f"{float(spot_sl_info.get('sl_spot_level', 0.0) or 0.0):.1f} "
                        f"({float(spot_sl_info.get('distance_pts', 0.0) or 0.0):.0f}pts)"
                    )
                except Exception as spot_sl_exc:
                    logger.debug(f"[{self.NAME}] Spot SL unavailable: {spot_sl_exc}")
                has_live_greeks = (
                    snap_source not in {"ESTIMATED", "SIMULATED", "FALLBACK"}
                    and raw_gamma > 0.0
                    and raw_oi > 0.0
                )
                sef_result = get_smart_entry_filter().evaluate(
                    df=recent_df,
                    direction=direction,
                    entry_spot=nifty_ltp,
                    entry_premium=est_premium,
                    delta=raw_delta,
                    gamma=raw_gamma if has_live_greeks else 0.0009,
                    oi=raw_oi if has_live_greeks else 150000,
                    invalidation_spot=key_level,
                    lot_size=lot_size
                )

                if has_live_greeks and not sef_result["allow_buy"]:
                    log_pipeline_stage(
                        self.NAME, "trade_planning", "filtered",
                        reason="smart_money_filter_blocked",
                        market_ts=signal_ts.isoformat(),
                        details=sef_result.get("oi_reason", "OI/Delta/Gamma block")
                    )
                    logger.info(f"[{self.NAME}] Smart Money Filter blocked trade | {sef_result.get('oi_reason')}")
                    return

                # Override SL from Smart Entry Filter; lots still pass the
                # final hard cap below so invalidation sizing cannot bypass
                # max-lot or max-capital policy.
                sl_premium = sef_result["final_sl_premium"]
                policy_lots_before_sef = int(best.get("desired_lots", 1) or 1)
                current_setup_for_size = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
                current_setup_type_for_size = str(current_setup_for_size.get("setup_type", "unknown") or "unknown").lower()
                current_setup_strength_for_size = float(current_setup_for_size.get("setup_strength", 0.0) or 0.0)
                strategy_set_for_size = {
                    str(s).strip()
                    for s in data.get("strategies_fired", [])
                    if str(s).strip()
                }
                confirmed_breakout_size_floor = (
                    current_setup_type_for_size == "breakout"
                    and current_setup_strength_for_size >= PLANNER_SETUP_STRENGTH_THRESH_0_72
                    and rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_57
                    and {"BBSqueeze", "FVG", "ValueArea"}.issubset(strategy_set_for_size)
                    and "SkewHunter" not in strategy_set_for_size
                )
                high_conviction_size_floor = (
                    confirmed_breakout_size_floor
                    or policy_lots_before_sef >= 2
                    or votes >= 5
                    or rank_score >= 0.65
                )
                if has_live_greeks:
                    sef_capped_lots = self._cap_lots_by_policy(
                        desired_lots=int(sef_result["final_lots"] or 1),
                        entry_premium=est_premium,
                        lot_size=lot_size,
                    )
                    if high_conviction_size_floor and policy_lots_before_sef > sef_capped_lots:
                        selection_notes.append(
                            f"SmartFilter size floor kept high-conviction lots: "
                            f"{sef_capped_lots}->{policy_lots_before_sef}"
                        )
                        best["desired_lots"] = policy_lots_before_sef
                    else:
                        best["desired_lots"] = min(policy_lots_before_sef, sef_capped_lots)
                else:
                    best["desired_lots"] = policy_lots_before_sef
                quantity = int(best["desired_lots"]) * lot_size
                total_invested = round(est_premium * quantity, 2)
                
                # Confidence boost
                if has_live_greeks:
                    data["confidence"] = float(data.get("confidence", 0.65)) + sef_result["oi_confidence_boost"]
                    selection_notes.append(f"SmartFilter: {sef_result.get('oi_reason')}")
                else:
                    selection_notes.append("SmartFilter: OI/greeks hard gate skipped for estimated snapshot")
                selection_notes.append(f"Setup: {sef_result.get('setup_type')} | SL override: ₹{sl_premium:.1f}")

            except Exception as sef_exc:
                logger.error(f"[{self.NAME}] SmartEntryFilter error: {sef_exc}")

            micro_ctx = (((data.get("metadata") or {}).get("_context") or {}).get("microstructure") or {})
            micro_lot_multiplier = float(micro_ctx.get("lot_multiplier", 1.0) or 1.0)
            if micro_lot_multiplier != 1.0 and votes < 6 and rank_score < 0.70:
                before_micro_lots = int(best.get("desired_lots", 1) or 1)
                micro_lots = max(1, int(before_micro_lots * micro_lot_multiplier))
                if micro_lots != before_micro_lots:
                    selection_notes.append(
                        f"Microstructure size: {before_micro_lots}->{micro_lots} lots "
                        f"(mult={micro_lot_multiplier:.2f}; {micro_ctx.get('rv_iv_regime', 'NA')})"
                    )
                    best["desired_lots"] = micro_lots

            current_setup_for_cap = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
            cap_strategy_set = {
                str(s).strip()
                for s in data.get("strategies_fired", [])
                if str(s).strip()
            }
            cap_override_pct = None
            if (
                int(best.get("desired_lots", 1) or 1) >= 2
                and (
                    {"FVG", "OIAnalysis", "GammaExposure"}.issubset(cap_strategy_set)
                    or {"BBSqueeze", "FVG", "ValueArea"}.issubset(cap_strategy_set)
                    or {"ADX+PSAR", "SkewHunter", "ValueArea"}.issubset(cap_strategy_set)
                )
                and float(current_setup_for_cap.get("setup_strength", 0.0) or 0.0) >= PLANNER_SETUP_STRENGTH_THRESH_0_72
                and rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_57
            ):
                cap_override_pct = 30.0
            reduced_budget_requested, reduced_budget_cap, reduced_budget_reason, _ = self._reduced_budget_requested(data)
            capped_lots = self._cap_lots_by_policy(
                desired_lots=int(best.get("desired_lots", 1) or 1),
                entry_premium=est_premium,
                lot_size=lot_size,
                cap_pct_override=cap_override_pct,
                absolute_cap_override=reduced_budget_cap if reduced_budget_requested else None,
            )
            if capped_lots != int(best.get("desired_lots", 1) or 1):
                selection_notes.append(
                    f"Hard lot cap: {best.get('desired_lots')}->{capped_lots} "
                    f"(max_lots={MAX_POSITION_LOTS}, max_capital={self._lot_cap_pct(int(best.get('desired_lots', 1) or 1)):.1f}%)"
                )
            if reduced_budget_requested:
                rb_lots = 1
                capped_lots = 1
                best["desired_lots"] = 1
                selection_notes.append(
                    f"Reduced budget lane: 1 lot(s) (invested=₹{est_premium * lot_size:.0f}) | {reduced_budget_reason}"
                )
            elif (not is_backtest_mode) and dte == 0:
                capped_lots = 1
                best["desired_lots"] = 1
                selection_notes.append("0 DTE Expiry Day Cap: max 1 lot enforced (Live Mode)")
            else:
                best["desired_lots"] = capped_lots
            if theta and theta.size_multiplier < 1.0 and votes < 6 and rank_score < 0.70:
                theta_lots = max(1, round(int(best["desired_lots"]) * float(theta.size_multiplier)))
                if theta_lots <= 0:
                    log_pipeline_stage(
                        self.NAME,
                        "trade_planning",
                        "filtered",
                        reason="expiry_theta_size_zero",
                        market_ts=signal_ts.isoformat(),
                        direction=direction,
                        theta_note=theta.note,
                    )
                    logger.warning(f"[{self.NAME}] Expiry theta size gate blocked trade | {theta.note}")
                    return
                if theta_lots < int(best["desired_lots"]):
                    selection_notes.append(
                        f"Expiry theta size cap: {best['desired_lots']}->{theta_lots} lots "
                        f"(mult={theta.size_multiplier:.2f}; premium_remaining={theta.premium_remaining_pct:.0%})"
                    )
                    best["desired_lots"] = theta_lots
            quantity = int(best["desired_lots"]) * lot_size
            total_invested = round(est_premium * quantity, 2)
            current_setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
            current_setup_type = str(current_setup.get("setup_type", "unknown") or "unknown").lower()
            if reduced_budget_requested:
                rb_sl_floor = round(est_premium * 0.90, 2)
                if sl_premium < rb_sl_floor:
                    selection_notes.append(
                        f"Reduced budget tight SL floor: ₹{sl_premium:.1f}->₹{rb_sl_floor:.1f} (10% max risk)"
                    )
                    sl_premium = rb_sl_floor
            elif (success_prob < PLANNER_SUCCESS_PROB_THRESH_0_32 or rank_score < PLANNER_RANK_SCORE_THRESH_0_6) and current_setup_type != "breakout":
                low_edge_entry = success_prob < PLANNER_SUCCESS_PROB_THRESH_0_32 or rank_score < PLANNER_RANK_SCORE_THRESH_0_62
                sl_floor_pct = 0.965 if low_edge_entry else 0.95
                medium_quality_floor = round(est_premium * sl_floor_pct, 2)
                if sl_premium < medium_quality_floor:
                    selection_notes.append(
                        f"Medium-quality SL cap: ₹{sl_premium:.1f}->₹{medium_quality_floor:.1f} "
                        f"(rank={rank_score:.2f}, p={success_prob:.2f}, floor={sl_floor_pct:.1%})"
                    )
                    sl_premium = medium_quality_floor
            current_target_pct = self._target_pct(est_premium, target_premium)
            compressed_target_pct = None
            compressed_target_reason = ""
            signal_minute_for_target = signal_ts.hour * 60 + signal_ts.minute
            if reduced_budget_requested:
                compressed_target_pct = REDUCED_BUDGET_TARGET_PCT
                compressed_target_reason = "reduced_budget"
            elif (
                rank_score < 0.70
                and votes < 6
                and (
                    self._structure_bias_for_signal(data) == "RANGING"
                    or signal_minute_for_target <= 9 * 60 + 40
                    or success_prob < CAUTION_TARGET_MAX_ML_CONF
                )
            ):
                compressed_target_pct = CAUTION_TARGET_PCT
                compressed_target_reason = "ranging_or_caution"
            if signal_minute_for_target >= LATE_SESSION_TARGET_START_MINUTE:
                compressed_target_pct = (
                    min(compressed_target_pct, LATE_SESSION_TARGET_PCT)
                    if compressed_target_pct is not None
                    else LATE_SESSION_TARGET_PCT
                )
                compressed_target_reason = (
                    f"{compressed_target_reason}+late_session"
                    if compressed_target_reason
                    else "late_session"
                )
            max_intraday_cap = float(os.getenv("MAX_INTRADAY_TARGET_PCT", "22.0"))
            if max_intraday_cap > 0:
                compressed_target_pct = (
                    min(compressed_target_pct, max_intraday_cap)
                    if compressed_target_pct is not None
                    else max_intraday_cap
                )
                if not compressed_target_reason:
                    compressed_target_reason = "max_intraday_cap"
                elif "max_intraday_cap" not in compressed_target_reason:
                    compressed_target_reason += "+max_intraday_cap"
            if compressed_target_pct is not None and current_target_pct > compressed_target_pct:
                compressed_target = self._compress_target(
                    entry_premium=est_premium,
                    target_premium=target_premium,
                    target_pct=compressed_target_pct,
                )
                if compressed_target < target_premium:
                    selection_notes.append(
                        f"Target compression: ₹{target_premium:.1f}->₹{compressed_target:.1f} "
                        f"({current_target_pct:.0f}%->{compressed_target_pct:.0f}%; "
                        f"{compressed_target_reason})"
                    )
                    target_premium = compressed_target
            if reduced_budget_requested:
                rb_lots = 1
                best["desired_lots"] = 1
                capped_lots = 1
            quantity = int(best["desired_lots"]) * lot_size
            total_invested = round(est_premium * quantity, 2)

            if hero_zero_active:
                if "metadata" not in data or data["metadata"] is None:
                    data["metadata"] = {}
                data["metadata"]["hero_zero_lane"] = True
                data["metadata"]["budget_lane"] = "HERO_ZERO"
                data["budget_lane"] = "HERO_ZERO"

            # Reconstruct signal
            sig = RawSignal(
                symbol           = symbol,
                direction        = Direction(direction),
                confidence       = float(data.get("confidence", 0.65)),
                votes            = self._extract_votes_count(data.get("votes", 2)),
                strategies_fired = data.get("strategies_fired", []),
                nifty_ltp        = nifty_ltp,
                timestamp        = signal_ts,
                metadata         = data.get("metadata", {}),
                ml_rank_score    = rank_score,
                ml_rank_tier     = data.get("ml_rank_tier", ""),
                ml_decision_reason = data.get("ml_decision_reason", ""),
            )
            plan = TradePlan(
                signal          = sig,
                option_symbol   = option_symbol,
                strike          = strike,
                option_type     = opt_t,
                expiry_date     = expiry_date.isoformat(),
                days_to_expiry  = dte,
                est_premium     = est_premium,
                sl_premium      = sl_premium,
                target_premium  = target_premium,
                lot_size        = lot_size,
                quantity        = quantity,
                desired_lots    = int(best["desired_lots"]),
                total_invested  = total_invested,
                tick_size       = float(getattr(instrument, "tick_size", 1.0)),
                tick_value      = float(getattr(instrument, "tick_value", 1.0)),
                contract_symbol = option_symbol,
                entry_price     = est_premium,
                sl_price        = sl_premium,
                target_price    = target_premium,
                ml_confidence   = success_prob,
                ml_rank_score   = rank_score,
                ml_approved     = True,
                premium_source  = premium_source,
                executable      = executable,
                execution_block_reason = block_reason,
                contract_score  = float(best["score"]),
                contract_snapshot = best["snapshot"],
                alternate_contracts = [c["snapshot"] for c in candidates[1:4]],
                selection_notes = selection_notes,
                broker          = self.broker.broker_name if self.broker else "",
                atr_points      = float(best["atr_points"]),
                atr_pct         = float(best["atr_pct"]),
                stop_distance   = float(best["stop_distance"]),
                target1_premium = float(best["target1_premium"]),
                target2_premium = float(best["target2_premium"]),
                breakeven_trigger_premium = float(best["breakeven_trigger_premium"]),
                trailing_stop_distance = float(best["trailing_stop_distance"]),
                time_stop_minutes = int(best["time_stop_minutes"]),
                time_stop_min_pnl_pct = float(best["time_stop_min_pnl_pct"]),
                risk_budget_inr = float(best["risk_budget_inr"]),
                sl_spot_level = float(spot_sl_info.get("sl_spot_level", 0.0) or 0.0),
                sl_source = str(spot_sl_info.get("sl_source", "") or ""),
                sl_note = str(spot_sl_info.get("note", "") or ""),
                sl_distance_pts = float(spot_sl_info.get("distance_pts", 0.0) or 0.0),
                sl_structural_premium = float(spot_sl_info.get("sl_premium", 0.0) or 0.0),
                protection_mode = "LOCAL_BRACKET",
                management_template = "ATM_INTRADAY_V1",
                timestamp       = signal_ts,
            )
            logger.info(
                f"[{self.NAME}]  market_ts={data.get('timestamp', '')} | "
                f"{option_symbol} | {est_premium} ({premium_source}) | "
                f"lots={plan.desired_lots} qty={plan.quantity} invested={total_invested:.0f} | "
                f"score={best['score']:.3f} | ml_rank={rank_score:.3f} | ml_p={success_prob:.3f} | exec_conf={confidence:.3f} | SL {sl_premium} | "
                f"T1 {plan.target1_premium} | T2 {target_premium} | RR {plan.risk_reward}"
            )
            log_pipeline_stage(
                self.NAME,
                "trade_planning",
                "passed",
                market_ts=data.get("timestamp", ""),
                direction=direction,
                contract=option_symbol,
                premium=f"{est_premium:.1f}",
                lots=plan.desired_lots,
                quantity=plan.quantity,
                invested=f"{total_invested:.2f}",
                expiry_date=plan.expiry_date,
                dte=plan.days_to_expiry,
                contract_score=f"{best['score']:.3f}",
                ml_rank=f"{rank_score:.3f}",
                ml_prob=f"{success_prob:.3f}",
                exec_conf=f"{confidence:.3f}",
                rr=f"{plan.risk_reward:.2f}",
                risk_budget=f"{float(best['risk_budget_inr']):.0f}",
            )
            if block_reason:
                logger.warning(f"[{self.NAME}] {block_reason}")

            # Skip low-quality trades not worth broker costs (unless target compression or reduced budget is active)
            if (
                hasattr(plan, 'risk_reward')
                and plan.risk_reward < MIN_RR_RATIO
                and not compressed_target_reason
                and not reduced_budget_requested
            ):
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="low_risk_reward",
                    market_ts=data.get("timestamp", ""),
                    contract=option_symbol,
                    rr=f"{plan.risk_reward:.2f}",
                    min_rr=MIN_RR_RATIO,
                )
                logger.info(
                    f"[{self.NAME}]   Skipped: RR={plan.risk_reward:.2f} < "
                    f"MIN_RR_RATIO={MIN_RR_RATIO}"
                )
                return
            expected_target_pnl_pct = self._projected_target_pnl_pct(
                entry_premium=plan.est_premium,
                exit_premium=plan.target_premium,
                lot_size=plan.quantity,
            )
            if expected_target_pnl_pct < MIN_EXPECTED_TARGET_PNL_PCT:
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="low_expected_target_pnl",
                    market_ts=data.get("timestamp", ""),
                    contract=option_symbol,
                    expected_target_pnl=f"{expected_target_pnl_pct:.2f}",
                    min_target_pnl=MIN_EXPECTED_TARGET_PNL_PCT,
                )
                logger.info(
                    f"[{self.NAME}]   Skipped: expected_target_pnl={expected_target_pnl_pct:.2f}% "
                    f"< MIN_EXPECTED_TARGET_PNL_PCT={MIN_EXPECTED_TARGET_PNL_PCT}"
                )
                return

            extra_ok, extra_reason = await self._extra_market_gates(plan, data)
            if not extra_ok:
                log_pipeline_stage(
                    self.NAME,
                    "trade_planning",
                    "filtered",
                    reason="market_intelligence_filter",
                    market_ts=data.get("timestamp", ""),
                    contract=option_symbol,
                    details=extra_reason,
                )
                logger.info(f"[{self.NAME}] Market gate blocked trade | {extra_reason}")
                return
            await self.bus.publish(Topic.TRADE_PLAN_READY, plan.to_dict(), self.NAME)
            if LLM_ENABLED:
                asyncio.create_task(self._sanity_check(plan))
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    async def _extra_market_gates(self, plan: TradePlan, data: dict) -> tuple[bool, str]:
        if not self.data_agent:
            return True, "no_data_agent"
        try:
            df = self.data_agent.get_recent_candles(120)
        except Exception:
            df = None
        if df is None or len(df) < 20:
            return True, "insufficient_recent_candles"

        context = ((data.get("metadata") or {}).get("_context") or {})
        india_vix = float(
            data.get("india_vix")
            or data.get("vix")
            or context.get("india_vix")
            or 18.0
        )
        mi = self.market_intelligence.should_trade(
            direction=plan.signal.direction.value,
            df=df,
            india_vix=india_vix,
        )
        plan.selection_notes.append(f"Market intelligence: {mi.get('summary', 'n/a')}")
        if False: # mi disabled 
            return False, mi.get("summary", "market intelligence blocked")

        mtf = await self.mtf_fetcher.get_context(plan.signal.symbol, df_5min=df)
        plan.selection_notes.append(f"MTF: {mtf.note} | score={mtf.confluence_score:.2f}")
        direction = plan.signal.direction.value
        if False: # mtf disabled 
            return False, mtf.note
        if False: # mtf disabled 
            return False, mtf.note
        try:
            news = await asyncio.wait_for(self.news_filter.scan(), timeout=1.5)
            plan.selection_notes.append(f"News filter: {news.note}")
            if news.halt_trading:
                return False, f"news halt: {news.note}"
            if news.caution:
                plan.selection_notes.append("News caution active: executor should use reduced discretion")
        except Exception as exc:
            logger.debug(f"[{self.NAME}] News filter fail-open: {exc}")
        return True, "ok"

    async def _sanity_check(self, plan: TradePlan) -> None:
        setup = ((plan.signal.metadata or {}).get("_context") or {}).get("setup", {}) or {}
        structure = ((plan.signal.metadata or {}).get("_context") or {}).get("market_structure", {}) or {}
        prompt = (
            f"Review this {plan.option_symbol} commodity options trade plan in one sentence. Flag any concern.\n"
            f"Option: {plan.option_symbol} | Underlying: {plan.signal.nifty_ltp}\n"
            f"Premium: {plan.est_premium} | DTE: {plan.days_to_expiry} | RR: {plan.risk_reward}\n"
            f"ML conf: {plan.ml_confidence:.0%} | Strategies: {plan.signal.strategies_fired}\n"
            f"Setup: {setup.get('setup_type', 'unknown')} ({float(setup.get('setup_strength', 0.0) or 0.0):.2f}) | "
            f"Structure event: {structure.get('liquidity_event', {}).get('type', 'NONE')}"
        )
        rank_score = float(plan.contract_snapshot.get("ml_rank_score", plan.ml_confidence) or plan.ml_confidence)
        text = await call_llm_context_async(
            prompt,
            cache_key=f"plan_sanity|{plan.signal.timestamp.isoformat()}|{plan.option_symbol}",
            task_type=TaskType.TRADE_SANITY,
            max_tokens=80,
            rank_score=rank_score,
            success_prob=float(plan.ml_confidence or 0.0),
        )
        if text:
            await self.bus.publish(Topic.ALERT,
                {"type": "trade_sanity", "text": text, "option": plan.option_symbol},
                self.NAME)

    def _select_contracts(
        self,
        *,
        symbol: str,
        nifty_ltp: float,
        confidence: float,
        direction: str,
        atm: int,
        expiry_date: date,
        dte: int,
        lot_size: int,
        strike_step: int,
        signal_data: dict,
    ) -> tuple[list[dict], list[str]]:
        option_type = "CE" if "CALL" in str(direction).upper() or str(direction).upper() in ("BUY", "LONG") else "PE"
        market_context = self._market_context(signal_data)
        ml_success_prob = float(
            signal_data.get("ml_success_prob", signal_data.get("ml_confidence", confidence))
        )
        ml_rank_score = float(
            signal_data.get("ml_rank_score", signal_data.get("ml_confidence", confidence))
        )
        setup = (((signal_data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        signal_ts = self._resolve_signal_ts(signal_data.get("timestamp"))
        hero_zero_meta = (signal_data.get("metadata") or {}).get("HeroZero") or {}
        hero_zero_strike = int(hero_zero_meta.get("strike") or 0)
        hero_zero_active = "HeroZero" in set(signal_data.get("strategies_fired", []) or [])
        structure_bias_for_quality = self._structure_bias_for_signal(signal_data)
        strikes = self._candidate_strikes(
            atm=atm,
            direction=direction,
            confidence=confidence,
            strike_step=strike_step,
            preferred_strike=hero_zero_strike if hero_zero_active else 0,
        )
        snapshots = self._fetch_contract_snapshots(
            symbol=symbol,
            expiry_date=expiry_date,
            option_type=option_type,
            strikes=strikes,
            nifty_ltp=nifty_ltp,
            dte=dte,
            strike_step=strike_step,
        )
        scored = []
        notes = []
        risk_budget_inr = 0.0
        for snap in snapshots:
            raw_premium = float(snap.last_price or 0.0)
            if raw_premium <= 0:
                raw_premium = estimate_atm_premium(nifty_ltp, dte, symbol=symbol) * max(0.55, 1 - abs(snap.strike - atm) / max(strike_step * 4, 400))
                snap.source = snap.source or "ESTIMATED"
            raw_premium = round(raw_premium, 1)

            if self.backtest_mode:
                real_entry = apply_realistic_entry(raw_premium, dte, time_of_day=signal_ts.strftime("%H:%M"))
                premium = real_entry.adjusted_premium
            else:
                premium = apply_option_slippage(raw_premium, "BUY", PLANNER_ENTRY_SLIPPAGE_PCT)

            quality_block = self._contract_quality_block_reason(
                direction=direction,
                setup=setup,
                structure_bias=structure_bias_for_quality,
                strategies=signal_data.get("strategies_fired", []),
                premium=premium,
                ml_rank_score=ml_rank_score,
            )
            if quality_block:
                notes.append(f"Skipped {snap.strike}{option_type}: {quality_block}")
                continue

            # Natural Gas (NATGASM) Liquidity & Spread Gate (pre-17:00 IST NYMEX session)
            bid = float(getattr(snap, "bid_price", getattr(snap, "bid", 0.0)) or 0.0)
            ask = float(getattr(snap, "ask_price", getattr(snap, "ask", 0.0)) or 0.0)
            if symbol.upper() in ("NATGASM", "NATGAS", "NATURALGAS") and bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                spread_pct = (ask - bid) / mid if mid > 0 else 0.0
                if signal_ts.hour < 17 and spread_pct > 0.04:
                    notes.append(
                        f"Skipped {snap.strike}{option_type}: NATGASM spread gate "
                        f"(bid={bid:.1f}, ask={ask:.1f}, spread={spread_pct:.1%} > 4.0% before 17:00 IST)"
                    )
                    continue

            gf = GreeksFilter()
            allow_expiry_day = (dte <= 1) or is_expiry_day(
                self._resolve_signal_ts(signal_data.get("timestamp")),
                symbol=symbol,
            )
            strats_fired = list(signal_data.get("strategies_fired", []) or [])
            greek_res = gf.check(
                nifty_ltp,
                snap.strike,
                option_type,
                premium,
                dte,
                allow_expiry_day=allow_expiry_day,
                is_hero_zero=hero_zero_active,
            )
            if not greek_res.tradeable:
                notes.append(f"Skipped {snap.strike}{option_type}: {greek_res.reason}")
                continue

            exit_profile = self._build_exit_profile(
                premium=premium,
                snap=snap,
                confidence=confidence,
                votes=signal_data.get("votes", {}),
                market_context=market_context,
                is_hero_zero=hero_zero_active,
                is_expiry_day=allow_expiry_day,
                symbol=symbol,
            )
            decision = self.final_decision.evaluate(
                entry_premium=premium,
                target_premium=exit_profile["target2_premium"],
                stop_premium=exit_profile["sl_premium"],
                ml_success_prob=ml_success_prob,
                ml_rank_score=ml_rank_score,
                atr_pct=float(market_context.get("atr_pct", 0.0) or 0.0),
                strategies=signal_data.get("strategies_fired", []),
                regime=str(signal_data.get("regime", "TRENDING")),
                setup=setup,
                vix_regime=str(market_context.get("vix_regime", "FLAT")),
                lot_size=lot_size,
            )
            is_commodity_trade = any(k in str(symbol).upper() for k in ("SILVER", "GOLD", "CRUDE", "NAT", "MCX"))
            allow_one_lot_fallback = (
                (self.backtest_mode or is_commodity_trade or str(os.getenv("TRADING_MODE", TRADING_MODE)).upper() == "OBSERVE")
                and decision.reason == "risk budget too small for one lot"
                and decision.expectancy_inr > 0
            )
            if not decision.tradeable:
                if allow_one_lot_fallback:
                    notes.append(
                        f"1-lot sizing fallback {snap.strike}{option_type}: "
                        f"EV {decision.expectancy_inr:.0f}/trade positive but one-lot risk "
                        f"{decision.avg_loss_inr:.0f} exceeds budget {decision.risk_budget_inr:.0f}"
                    )
                else:
                    notes.append(
                        f"Skipped {snap.strike}{option_type}: {decision.reason} "
                        f"(EV {decision.expectancy_inr:.0f}/trade)"
                    )
                    continue
            expansion_ok, expansion_reason = self._has_expansion_potential(
                market_context=market_context,
                ml_rank_score=ml_rank_score,
                confidence=confidence,
            )
            if not expansion_ok:
                notes.append(
                    f"Skipped {snap.strike}{option_type}: {expansion_reason}"
                )
                continue
            risk_budget_inr = float(decision.risk_budget_inr)
            risk_per_lot = float(decision.avg_loss_inr)
            planned_stop_risk_per_lot = self._planned_stop_risk_per_lot(
                entry_premium=decision.entry_premium,
                stop_premium=decision.stop_premium,
                lot_size=lot_size,
            )
            if allow_one_lot_fallback:
                desired_lots = 1
                quantity = lot_size
            else:
                desired_lots = int(decision.desired_lots)
                quantity = int(decision.quantity)
            structure_bias = str(market_context.get("structure_bias", "") or "").upper()
            if structure_bias in {"", "UNKNOWN"}:
                structure_bias = self._structure_bias_for_signal(signal_data)
            reduced_budget_requested_c, rb_cap_c, _, _ = self._reduced_budget_requested(signal_data)
            policy_lots = self._apply_lot_policy(
                desired_lots=desired_lots,
                lot_size=lot_size,
                direction=direction,
                strategies=signal_data.get("strategies_fired", []),
                setup=setup,
                ml_rank_score=ml_rank_score,
                structure_bias=structure_bias,
                dte=dte,
                signal_ts=signal_ts,
                nifty_ltp=nifty_ltp,
            )
            if policy_lots >= 2:
                desired_lots = policy_lots
                if isinstance(signal_data.get("metadata"), dict):
                    signal_data["metadata"].pop("reduced_budget_lane", None)
                    signal_data["metadata"].pop("reduced_budget_reason", None)
            elif reduced_budget_requested_c:
                max_affordable = max(1, int(rb_cap_c // max(float(decision.entry_premium) * lot_size, 1.0)))
                desired_lots = max(1, min(desired_lots, max_affordable, MAX_POSITION_LOTS))
            else:
                desired_lots = policy_lots
            if hero_zero_active:
                before_lots = desired_lots
                hero_recommended_lots = int(hero_zero_meta.get("recommended_lots") or 1)
                hero_budget = float(hero_zero_meta.get("max_loss_total") or 2500.0)
                if hero_budget <= 0:
                    hero_budget = 2500.0
                hero_budget = max(2500.0, min(hero_budget, 4000.0))
                premium_lot_cost = max(float(decision.entry_premium) * lot_size, 1.0)
                if premium_lot_cost > hero_budget:
                    desired_lots = 1
                    notes.append(
                        f"HERO_ZERO single-lot allocation applied | strike={snap.strike}{option_type} | "
                        f"lot_cost=₹{premium_lot_cost:.0f}"
                    )
                else:
                    budget_lots = max(1, int(hero_budget // premium_lot_cost))
                    desired_lots = max(1, min(desired_lots, hero_recommended_lots, budget_lots, 2))
                    notes.append(
                        f"HERO_ZERO lane active: {desired_lots} lot(s) (budget≈₹{hero_budget:.0f}, lot_cost≈₹{premium_lot_cost:.0f})"
                    )
            wyckoff_size = float(
                (((signal_data.get("metadata") or {}).get("_context") or {}).get("wyckoff_size_multiplier", 1.0))
                or 1.0
            )
            if wyckoff_size < 1.0:
                before_lots = desired_lots
                if self._should_apply_wyckoff_size_cap(
                    direction=direction,
                    strategies=signal_data.get("strategies_fired", []),
                    setup=setup,
                    ml_rank_score=ml_rank_score,
                    structure_bias=structure_bias,
                    dte=dte,
                    signal_ts=signal_ts,
                ):
                    desired_lots = max(1, int(desired_lots * wyckoff_size))
                if desired_lots < before_lots:
                    notes.append(
                        f"Wyckoff size cap: {before_lots}->{desired_lots} lots "
                        f"(mult={wyckoff_size:.2f})"
                    )
                else:
                    notes.append(
                        f"Wyckoff cap bypassed for confirmed edge "
                        f"(mult={wyckoff_size:.2f})"
                    )
            quantity = desired_lots * lot_size
            score, note = self._score_contract(
                snap=snap,
                atm=atm,
                confidence=confidence,
                premium=premium,
                direction=direction,
                risk_budget_inr=risk_budget_inr,
                risk_per_lot=planned_stop_risk_per_lot,
            )
            expectancy_bonus = min(0.18, max(0.0, decision.expectancy_inr) / max(risk_budget_inr, 1.0) * 0.12)
            score = round(score + expectancy_bonus, 4)
            scored.append({
                "symbol": snap.symbol,
                "strike": snap.strike,
                "premium": decision.entry_premium,
                "premium_source": (
                    "HISTORICAL" if self._use_historical_option_premium() and snap.last_price > 0
                    else ("LIVE" if snap.last_price > 0 else "ESTIMATED")
                ),
                "score": score,
                "desired_lots": desired_lots,
                "lot_size": lot_size,
                "quantity": quantity,
                "total_invested": round(decision.entry_premium * quantity, 2),
                "risk_budget_inr": round(risk_budget_inr, 2),
                "risk_per_lot": round(risk_per_lot, 2),
                "planned_stop_risk_per_lot": round(planned_stop_risk_per_lot, 2),
                "expectancy_inr": decision.expectancy_inr,
                "win_prob": decision.win_prob,
                "reward_risk": decision.reward_risk,
                **exit_profile,
                "snapshot": {
                    **snap.to_dict(),
                    "premium_used": decision.entry_premium,
                    "lots": desired_lots,
                    "lot_size": lot_size,
                    "quantity": quantity,
                    "total_invested": round(decision.entry_premium * quantity, 2),
                    "score": round(score, 4),
                    "score_note": f"{note} | ev={decision.expectancy_inr:.0f} | wp={decision.win_prob:.2f}",
                    "risk_per_lot": round(risk_per_lot, 2),
                    "planned_stop_risk_per_lot": round(planned_stop_risk_per_lot, 2),
                    "risk_budget_inr": round(risk_budget_inr, 2),
                    "setup": setup,
                    "risk_pct": decision.risk_pct,
                    "expectancy_inr": decision.expectancy_inr,
                    "win_prob": decision.win_prob,
                    "avg_win_inr": decision.avg_win_inr,
                    "avg_loss_inr": decision.avg_loss_inr,
                    "reward_risk": decision.reward_risk,
                    "historical_stats": {
                        "sample_size": decision.historical_stats.sample_size,
                        "win_rate": decision.historical_stats.win_rate,
                        "avg_win_pct": decision.historical_stats.avg_win_pct,
                        "avg_loss_pct": decision.historical_stats.avg_loss_pct,
                        "source": decision.historical_stats.source,
                    },
                },
            })
        viable = [c for c in scored if c["planned_stop_risk_per_lot"] <= c["risk_budget_inr"]]
        if not viable and scored:
            # Commodity & positive-expectancy 1-lot parity allocation:
            # When trading commodities or in OBSERVE mode, 1 lot is the minimum atomic trade unit.
            # If candidate contracts have positive expectancy and total_invested fits the commodity
            # capital allocation (<= ₹40,000 per trade, matching the 30-day institutional backtest),
            # allow 1 lot rather than dropping the setup.
            commodity_cap = float(os.getenv("MAX_COMMODITY_TRADE_CAPITAL", "40000.0"))
            best_ev_cand = max(scored, key=lambda c: c.get("score", 0.0))
            if (
                best_ev_cand.get("expectancy_inr", 0.0) > 0
                and best_ev_cand.get("total_invested", 0.0) <= commodity_cap
            ):
                best_ev_cand["desired_lots"] = 1
                best_ev_cand["quantity"] = best_ev_cand["lot_size"]
                best_ev_cand["risk_budget_inr"] = max(best_ev_cand["risk_budget_inr"], best_ev_cand["planned_stop_risk_per_lot"])
                viable = [best_ev_cand]
                notes.append(
                    f"Commodity 1-lot parity allocation active: 1 lot (invested≈₹{best_ev_cand['total_invested']:.0f} <= ₹{commodity_cap:.0f}, EV≈₹{best_ev_cand['expectancy_inr']:.0f})"
                )
            elif self.backtest_mode:
                fallback_candidate = max(scored, key=lambda c: c["score"])
                _, fallback_reason = self._should_allow_backtest_fallback(
                    candidate=fallback_candidate,
                    setup=setup,
                    ml_rank_score=ml_rank_score,
                )
                details = fallback_reason or (
                    "risk budget fallback disabled; one-lot stop risk must fit budget"
                )
                notes.append(f"Skipped fallback candidate: {details}")
        if not viable:
            notes.append(
                f"No contract fits hard planned-stop risk budget of {risk_budget_inr:.0f} per trade."
            )
            return [], notes
        viable.sort(key=lambda c: c["score"], reverse=True)
        scored = viable
        if scored:
            notes.append(
                f"Selected from {len(scored)} {option_type} candidates around ATM {atm}."
            )
            notes.append(
                "Priority: live premium, strike distance, positive expectancy, affordable premium band, lower theta drag, and stop-risk within budget."
            )
            if setup:
                notes.append(
                    f"Setup gate active: {setup.get('setup_type', 'unknown')} "
                    f"| strength={float(setup.get('setup_strength', 0.0) or 0.0):.2f}"
                )
            if hero_zero_active and hero_zero_strike:
                notes.append(
                    f"HeroZero strike priority active | strike={hero_zero_strike} "
                    f"| exit_by={hero_zero_meta.get('exit_by', '15:28')}"
                )
            max_selected_lots = max(int(c.get("desired_lots", 1) or 1) for c in scored)
            notes.append(
                f"Lot policy active | selected_lots={max_selected_lots} | "
                f"lot_size={lot_size} | max_lots={MAX_POSITION_LOTS}"
            )
        return scored, notes

    def _candidate_strikes(
        self,
        *,
        atm: int,
        direction: str,
        confidence: float,
        strike_step: int,
        preferred_strike: int = 0,
    ) -> list[int]:
        offsets = []
        for step in range(-OPTION_CANDIDATE_STEPS, OPTION_CANDIDATE_STEPS + 1):
            offsets.append(step * strike_step)
        preferred_shift = 0 if confidence >= HIGH_CONF_THRESHOLD else (
            OTM_OFFSET_POINTS if direction == "BUY_CALL" else -OTM_OFFSET_POINTS
        )
        ordered = sorted(
            {atm + off for off in offsets},
            key=lambda strike: abs((strike - atm) - preferred_shift),
        )
        if preferred_strike > 0:
            preferred_strike = int(round(preferred_strike / strike_step) * strike_step)
            ordered = [preferred_strike] + [strike for strike in ordered if strike != preferred_strike]
        return ordered

    def _fetch_contract_snapshots(
        self,
        *,
        symbol: str,
        expiry_date: date,
        option_type: str,
        strikes: list[int],
        nifty_ltp: float,
        dte: int,
        strike_step: int,
    ) -> list[OptionContract]:
        contracts = []
        use_historical_option_premium = self._use_historical_option_premium()
        require_historical_option_premium = self._require_historical_option_premium()
        if use_historical_option_premium and self.data_agent and hasattr(self.data_agent, "get_option_contracts"):
            try:
                contracts = self.data_agent.get_option_contracts(
                    symbol=symbol,
                    expiry=expiry_date,
                    option_type=option_type,
                    strikes=strikes,
                )
            except Exception as exc:
                logger.debug(f"[{self.NAME}] historical option chain fetch failed: {exc}")
                contracts = []
        elif self.broker:
            try:
                contracts = self.broker.get_option_contracts(
                    symbol=symbol,
                    expiry=expiry_date,
                    option_type=option_type,
                    strikes=strikes,
                )
            except Exception as exc:
                logger.warning(f"[{self.NAME}] option chain fetch failed: {exc}")
                contracts = []
        by_strike = {c.strike: c for c in contracts}
        snapshots = []
        for strike in strikes:
            contract = by_strike.get(strike)
            if contract is not None and (contract.last_price is None or contract.last_price <= 0):
                recovered_ltp = 0.0
                if self.broker:
                    try:
                        recovered_ltp = float(self.broker.get_option_ltp(contract.symbol) or 0.0)
                    except Exception:
                        recovered_ltp = 0.0
                if recovered_ltp <= 0 and self.data_agent:
                    try:
                        recovered_ltp = float(self.data_agent.get_option_ltp(contract.symbol) or 0.0)
                    except Exception:
                        recovered_ltp = 0.0
                if recovered_ltp > 0:
                    contract.last_price = recovered_ltp
                    contract.source = "DHAN_LIVE" if "DHAN" in str(contract.source) else "BROKER_LTP"

            if contract is None:
                option_symbol = build_option_symbol(symbol, expiry_date, strike, option_type)
                validation = validate_option_contract(
                    symbol=symbol,
                    expiry_date=expiry_date,
                    strike=strike,
                    option_type=option_type,
                    underlying=nifty_ltp,
                    strike_step=strike_step,
                    expected_symbol=option_symbol,
                )
                if not validation.valid:
                    logger.debug(
                        f"[{self.NAME}] Skipping contract build | symbol={option_symbol} | "
                        f"reason={validation.reason}"
                    )
                    continue
                live_ltp = 0.0
                if use_historical_option_premium and self.data_agent:
                    live_ltp = float(self.data_agent.get_option_ltp(option_symbol) or 0.0)
                    if require_historical_option_premium and live_ltp <= 0:
                        logger.debug(
                            f"[{self.NAME}] Historical option premium required but missing | "
                            f"symbol={option_symbol}"
                        )
                        continue
                elif self.data_agent and not self.backtest_mode:
                    live_ltp = float(self.data_agent.get_option_ltp(option_symbol) or 0.0)
                elif self.broker:
                    live_ltp = float(self.broker.get_option_ltp(option_symbol) or 0.0)
                est_delta = self._estimate_delta(
                    nifty_ltp=nifty_ltp,
                    strike=strike,
                    option_type=option_type,
                )
                contract = OptionContract(
                    symbol=option_symbol,
                    strike=strike,
                    option_type=option_type,
                    expiry_date=expiry_date.isoformat(),
                    last_price=live_ltp,
                    implied_volatility=0.14,
                    delta=est_delta,
                    theta=max(0.8, 4.5 - dte * 0.1),
                    source="HISTORICAL_OPTION_CACHE" if use_historical_option_premium and live_ltp > 0 else ("BROKER_LTP" if live_ltp > 0 else "ESTIMATED"),
                )
            snapshots.append(contract)
        return snapshots

    def _score_contract(
        self,
        *,
        snap: OptionContract,
        atm: int,
        confidence: float,
        premium: float,
        direction: str,
        risk_budget_inr: float,
        risk_per_lot: float,
    ) -> tuple[float, str]:
        sym = getattr(snap, "symbol", "") or ""
        sym_u = sym.upper()
        is_commodity = any(c in sym_u for c in ("SILVER", "GOLD", "CRUDE", "NAT")) or premium > 500
        if "SILVER" in sym_u:
            step = 1000
            min_prem, max_prem = 1000.0, 9000.0
        elif "GOLD" in sym_u:
            step = 100
            min_prem, max_prem = 200.0, 3000.0
        elif "CRUDE" in sym_u:
            step = 50
            min_prem, max_prem = 30.0, 500.0
        elif "NAT" in sym_u:
            step = 5
            min_prem, max_prem = 5.0, 100.0
        else:
            step = NIFTY_STRIKE_STEP
            min_prem, max_prem = OPTION_MIN_PREMIUM, OPTION_MAX_PREMIUM

        strike_distance = abs(snap.strike - atm)
        distance_penalty = min(0.45, strike_distance / max(step * 4, 1) * 0.35)

        if is_commodity:
            premium_bonus = 0.18 if min_prem <= premium <= max_prem else -0.12
            iv_penalty = 0.0
            if snap.implied_volatility > 0.38:
                iv_penalty = min(0.12, (snap.implied_volatility - 0.38) * 0.5)
            oi_bonus = min(0.15, snap.open_interest / 1_000) if snap.open_interest > 0 else 0.0
        else:
            premium_bonus = 0.18 if OPTION_MIN_PREMIUM <= premium <= OPTION_MAX_PREMIUM else -0.12
            iv_penalty = 0.0
            if snap.implied_volatility > 0.28:
                iv_penalty = min(0.12, (snap.implied_volatility - 0.28) * 0.5)
            oi_bonus = min(0.15, snap.open_interest / 1_000_000) if snap.open_interest > 0 else 0.0

        live_bonus = 0.02 if snap.last_price > 0 else 0.0
        theta_penalty = min(0.12, max(0.0, snap.theta) / 20)
        delta_bonus = 0.0
        if snap.delta:
            desired_delta = 0.42 if confidence < HIGH_CONF_THRESHOLD else 0.55
            delta_bonus = max(0.0, 0.14 - abs(abs(snap.delta) - desired_delta))
        score = max(
            0.0,
            confidence + premium_bonus + live_bonus + oi_bonus + delta_bonus
            - distance_penalty - iv_penalty - theta_penalty
        )
        if not is_commodity and risk_per_lot > risk_budget_inr > 0:
            score -= min(0.22, (risk_per_lot / risk_budget_inr - 1.0) * 0.18)
        note = (
            f"premium={premium}, distance={strike_distance}, "
            f"source={'live' if snap.last_price > 0 else 'estimated'}, "
            f"risk/lot={risk_per_lot:.0f}"
        )
        return round(score, 4), note

    @staticmethod
    def _contract_quality_block_reason(
        *,
        direction: str,
        setup: dict,
        structure_bias: str,
        strategies: list[str],
        premium: float,
        ml_rank_score: float,
    ) -> str:
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        bias = str(structure_bias or "").upper()
        strategy_set = {str(name).strip() for name in strategies if str(name).strip()}
        hard_displacement = {
            "FVG",
            "CPR",
            "OIAnalysis",
            "GammaExposure",
            "StrikeMomentum",
            "SqueezeMomentum",
        }

        if setup_type == "breakout" and bias == "RANGING":
            return "planner quality gate: ranging breakout has poor historical follow-through"

        if setup_type == "trend_pullback" and premium >= 250 and not (ml_rank_score >= 0.58 or len(strategy_set) >= 5):
            return (
                "planner quality gate: high-premium trend pullback "
                f"(premium={premium:.1f}) decays poorly"
            )

        if (
            direction == "BUY_CALL"
            and setup_type == "breakout"
            and bias == "BEARISH"
            and premium >= 180
        ):
            return (
                "planner quality gate: counter-bias CALL breakout with expensive premium "
                f"(bias={bias}, premium={premium:.1f})"
            )

        if (
            direction == "BUY_CALL"
            and setup_type == "vote_aligned"
            and bias == "BEARISH"
            and premium >= 250
            and not (ml_rank_score >= 0.58 or len(strategy_set) >= 5)
            and not strategy_set.intersection(hard_displacement)
        ):
            return (
                "planner quality gate: bearish-bias CALL lacks displacement confirmation "
                f"(premium={premium:.1f}, rank={ml_rank_score:.2f})"
            )

        return ""

    def _market_context(self, signal_data: dict) -> dict:
        metadata = signal_data.get("metadata", {}) or {}
        context = metadata.get("_context", {}) or {}
        setup = context.get("setup", {}) or {}
        setup_ctx = setup.get("context", {}) or {}
        atr_points = float(context.get("atr_points", 0.0) or 0.0)
        atr_pct = float(context.get("atr_pct", 0.0) or 0.0)
        vix_regime = signal_data.get("regime_details", {}).get("vix_regime", "FLAT")
        adx = float(context.get("adx", setup_ctx.get("adx", 0.0)) or 0.0)
        chop = float(context.get("chop", 0.0) or 0.0)
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        expected_move = float(setup.get("expected_move", 0.0) or 0.0)

        if (atr_points <= 0 or atr_pct <= 0) and self.data_agent:
            try:
                recent = self.data_agent.get_recent_candles(60)
                if recent is not None and len(recent) >= 20:
                    atr_points, atr_pct = compute_atr(recent, period=14)
            except Exception:
                pass
        return {
            "atr_points": atr_points,
            "atr_pct": atr_pct,
            "vix_regime": vix_regime,
            "adx": adx,
            "chop": chop,
            "setup_type": str(setup.get("setup_type", "unknown") or "unknown").lower(),
            "setup_strength": setup_strength,
            "expected_move": expected_move,
        }

    @staticmethod
    def _has_expansion_potential(
        *,
        market_context: dict,
        ml_rank_score: float,
        confidence: float,
    ) -> tuple[bool, str]:
        if max(float(ml_rank_score or 0.0), float(confidence or 0.0)) >= 0.55:
            return True, ""

        setup_strength = float(market_context.get("setup_strength", 0.0) or 0.0)
        adx = float(market_context.get("adx", 0.0) or 0.0)
        chop = float(market_context.get("chop", 0.0) or 0.0)
        expected_move = float(market_context.get("expected_move", 0.0) or 0.0)
        atr_points = float(market_context.get("atr_points", 0.0) or 0.0)
        expansion_ratio = (
            expected_move / max(atr_points, 1.0)
            if expected_move > 0 and atr_points > 0
            else 1.0
        )

        weak_momentum = (
            setup_strength < PLANNER_SETUP_STRENGTH_THRESH_0_6
            and 0.0 < adx < 18.0
            and chop >= 55.0
            and expansion_ratio < 1.15
        )
        if weak_momentum:
            return (
                False,
                (
                    f"low_momentum (setup={setup_strength:.2f}, adx={adx:.1f}, "
                    f"chop={chop:.1f}, move/atr={expansion_ratio:.2f})"
                ),
            )
        return True, ""

    def _build_exit_profile(
        self,
        *,
        premium: float,
        snap: OptionContract,
        confidence: float,
        market_context: dict,
        votes: int | dict = 0,
        is_hero_zero: bool = False,
        is_expiry_day: bool = False,
        symbol: str = "",
    ) -> dict:
        if isinstance(votes, dict):
            counts = []
            for v in votes.values():
                if isinstance(v, dict):
                    counts.append(int(v.get("count", 0) or 0))
                elif isinstance(v, (int, float)):
                    counts.append(int(v))
            votes_count = max(counts) if counts else 0
        else:
            try:
                votes_count = int(votes or 0)
            except (ValueError, TypeError):
                votes_count = 0
        votes = votes_count

        atr_points = float(market_context.get("atr_points", 0.0) or 0.0)
        atr_pct = float(market_context.get("atr_pct", 0.0) or 0.0)
        adx = float(market_context.get("adx", 0.0) or 0.0)
        chop = float(market_context.get("chop", 0.0) or 0.0)
        setup_type = str(market_context.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(market_context.get("setup_strength", 0.0) or 0.0)
        structure_bias = str(market_context.get("structure_bias", "UNKNOWN") or "UNKNOWN").upper()
        delta = abs(float(snap.delta or 0.45))
        trend_quality = max(0.0, min(1.0, (adx - 14.0) / 16.0)) if adx > 0 else 0.5
        if chop > 0:
            trend_quality = max(0.0, trend_quality - min(0.35, max(chop - 55.0, 0.0) / 35.0))
        quality_score = (
            max(0.0, min(1.0, confidence)) * 0.45
            + max(0.0, min(1.0, setup_strength)) * 0.35
            + trend_quality * 0.20
        )
        option_atr = max(premium * 0.14, atr_points * max(delta, 0.25))

        # Resolve instrument-specific configuration
        inst_sym = (
            symbol
            or market_context.get("symbol")
            or getattr(snap, "symbol", "")
            or os.getenv("INSTRUMENT", "CRUDEOILM")
        )
        inst_cfg = get_instrument_strategy_config(str(inst_sym))
        base_sl_pct = float(inst_cfg.get("stop_loss_pct", STOP_LOSS_PCT)) / 100.0
        base_tgt1_pct = float(inst_cfg.get("target1_pct", 12.0)) / 100.0
        base_tgt2_pct = float(inst_cfg.get("target2_pct", TARGET_PCT)) / 100.0
        base_be_pct = float(inst_cfg.get("breakeven_trigger_pct", 10.0)) / 100.0
        base_tsl_pct = float(inst_cfg.get("trailing_sl_pct", 8.0)) / 100.0

        if is_hero_zero:
            # 25% Stop Loss for HeroZero 0DTE options with multiplier targets
            stop_distance = premium * 0.25
            target1_distance = stop_distance * 1.8   # ~1.45x
            target2_distance = stop_distance * 3.5   # ~1.85x - 2.5x
            trailing_distance = max(stop_distance * 0.85, premium * 0.25)
        elif is_expiry_day or (getattr(snap, "dte", None) is not None and snap.dte <= 1):
            # Adaptive breathing SL for Expiry / 0DTE options to absorb normal tick oscillation
            sl_pct = max(0.20, base_sl_pct * 1.5) if atr_pct >= HIGH_VOL_ATR_PCT else max(0.15, base_sl_pct * 1.25)
            stop_distance = premium * sl_pct
            target_r = max(ATR_TARGET_MULTIPLIER, 2.0)
            target1_distance = max(stop_distance * 1.5, premium * base_tgt1_pct)
            target2_distance = max(stop_distance * target_r, premium * base_tgt2_pct)
            trailing_distance = max(stop_distance * 0.85, premium * sl_pct)
        else:
            sl_pct = base_sl_pct
            if atr_pct >= HIGH_VOL_ATR_PCT:
                sl_pct = max(sl_pct, base_sl_pct * 1.25)  # Breathing room in high volatility
            stop_distance = premium * sl_pct
            target_r = ATR_TARGET_MULTIPLIER
            if atr_pct >= HIGH_VOL_ATR_PCT:
                target_r *= HIGH_VOL_TARGET_COMPRESSION
            # Vote-based target scaling: realistic intraday R:R scaling
            if votes >= 3:
                target_r *= 1.25
            elif votes >= 2:
                target_r *= 1.15
            elif votes <= 2:
                target_r *= 1.0
            if confidence >= TWO_LOT_CONFIDENCE_THRESHOLD:
                target_r *= 1.05
            if quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_72:
                target_r *= 1.10
            elif quality_score < PLANNER_QUALITY_SCORE_THRESH_0_56:
                target_r *= 0.92
            if quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_8:
                target_r *= 1.05
            if structure_bias == "RANGING":
                target_r *= 0.92
            if setup_type in {"breakout", "trend_pullback"} and quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_68 and structure_bias != "RANGING":
                target_r *= 1.05
            target1_distance = max(stop_distance * BREAKEVEN_R_TRIGGER, premium * base_tgt1_pct)
            target2_distance = max(stop_distance * min(max(target_r * 0.6, 1.2), 2.0), premium * base_tgt2_pct)
            trailing_distance = max(stop_distance * 0.85, premium * base_tsl_pct)
        if quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_72:
            trailing_distance *= 1.25
        elif quality_score < PLANNER_QUALITY_SCORE_THRESH_0_56:
            trailing_distance *= 0.90
        elif quality_score < PLANNER_QUALITY_SCORE_THRESH_0_6:
            trailing_distance *= 0.95
        if quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_8:
            trailing_distance *= 1.08
        if structure_bias == "RANGING":
            trailing_distance *= 0.92
        if setup_type in {"breakout", "trend_pullback"} and quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_68 and structure_bias != "RANGING":
            trailing_distance *= 1.08
        if setup_type == "vote_aligned" and quality_score < PLANNER_QUALITY_SCORE_THRESH_0_6:
            stop_distance *= 0.94
            trailing_distance *= 0.93
        # Vote-based stop tightening (DISABLED for better PnL stability)
        # if votes <= 2:
        #     stop_distance *= 0.78
        #     trailing_distance *= 0.82
        pass
        time_stop_minutes = TIME_STOP_MINUTES
        # Minimal penalties - allow trades more time to develop
        if quality_score < PLANNER_QUALITY_SCORE_THRESH_0_55:
            time_stop_minutes = max(30, time_stop_minutes - 5)
        elif quality_score < PLANNER_QUALITY_SCORE_THRESH_0_62:
            time_stop_minutes = max(35, time_stop_minutes - 3)
        if quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_76:
            time_stop_minutes += 30
        elif quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_66:
            time_stop_minutes += 15
        elif quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_74:
            time_stop_minutes += 5
        if quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_8:
            time_stop_minutes += 10
        if structure_bias == "RANGING":
            time_stop_minutes = max(25, time_stop_minutes - 5)  # lighter penalty
        if setup_type in {"breakout", "trend_pullback"} and quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_68 and structure_bias != "RANGING":
            time_stop_minutes += 10
        # Time-decay threshold should be lighter than hard profit lock.
        # Using MIN_ACCEPTABLE_PROFIT_PCT here was forcing many +1~4% exits.
        time_stop_min_pnl_pct = max(TIME_STOP_MIN_PNL_PCT, 1.0)
        if quality_score < PLANNER_QUALITY_SCORE_THRESH_0_58:
            time_stop_min_pnl_pct = max(0.5, time_stop_min_pnl_pct + 0.5)
        elif quality_score < PLANNER_QUALITY_SCORE_THRESH_0_62:
            time_stop_min_pnl_pct = max(0.75, time_stop_min_pnl_pct + 0.25)
        if quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_76:
            time_stop_min_pnl_pct = max(2.0, time_stop_min_pnl_pct - 2.0)
        elif quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_66:
            time_stop_min_pnl_pct = max(3.0, time_stop_min_pnl_pct - 1.0)
        elif quality_score >= PLANNER_QUALITY_SCORE_THRESH_0_74:
            time_stop_min_pnl_pct = max(2.5, time_stop_min_pnl_pct - 0.5)
        if structure_bias == "RANGING":
            time_stop_min_pnl_pct = max(0.75, time_stop_min_pnl_pct - 0.5)
        breakeven_trigger = premium + target1_distance
        if structure_bias == "RANGING":
            breakeven_trigger = premium + (target1_distance * 0.88)
        return {
            "atr_points": round(atr_points, 4),
            "atr_pct": round(atr_pct, 6),
            "stop_distance": round(stop_distance, 2),
            "sl_premium": round(max(1.0, premium - stop_distance), 1),
            "target1_premium": round(premium + target1_distance, 1),
            "target2_premium": round(premium + target2_distance, 1),
            "breakeven_trigger_premium": round(breakeven_trigger, 1),
            "trailing_stop_distance": round(trailing_distance, 1),
            "time_stop_minutes": int(time_stop_minutes),
            "time_stop_min_pnl_pct": round(time_stop_min_pnl_pct, 2),
        }

    @staticmethod
    def _position_size(
        *,
        confidence: float,
        score_hint: float,
        stop_distance: float,
        risk_budget_inr: float,
        vix_regime: str = "FLAT",
    ) -> tuple[int, int]:
        # Centralized Lot Control: Strictly follow MAX_POSITION_LOTS from settings.
        # This removes dynamic 2-lot scaling and respects the user-defined fixed lot count.
        lots = max(1, MAX_POSITION_LOTS)
        return lots, lots * NIFTY_LOT_SIZE

    @staticmethod
    def _planned_stop_risk_per_lot(
        *,
        entry_premium: float,
        stop_premium: float,
        lot_size: int,
    ) -> float:
        """Hard risk gate uses actual planned SL loss, not expectancy-model avg loss."""
        quantity = max(int(lot_size or NIFTY_LOT_SIZE), 1)
        entry = max(float(entry_premium or 0.0), 0.0)
        stop = max(float(stop_premium or 0.0), 0.0)
        premium_loss = max(entry - stop, 0.0) * quantity
        costs = estimate_round_trip_costs(
            entry,
            stop,
            quantity,
            BACKTEST_BROKERAGE_PER_ORDER,
            BACKTEST_TRANSACTION_COST_PCT,
        )
        return round(premium_loss + costs, 2)

    @staticmethod
    def _apply_lot_policy(
        *,
        desired_lots: int,
        lot_size: int,
        direction: str,
        strategies: list[str],
        setup: dict | None,
        ml_rank_score: float,
        structure_bias: str,
        dte: int | None = None,
        signal_ts: datetime | None = None,
        nifty_ltp: float | None = None,
    ) -> int:
        from config.settings.modules.session_policy import get_session_policy
        session_policy = get_session_policy(signal_ts)
        max_lots = min(max(1, int(MAX_POSITION_LOTS)), session_policy.max_lots)
        desired_lots = max(1, min(int(desired_lots or 1), max_lots))
        if max_lots <= 1:
            return 1

        setup = setup or {}
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        setup_strength_gate = round(setup_strength, 2)
        strategy_set = {str(s).strip() for s in strategies if str(s).strip()}
        bias = str(structure_bias or "UNKNOWN").upper()
        aligned_bias = (
            (direction == "BUY_CALL" and bias == "BULLISH")
            or (direction == "BUY_PUT" and bias == "BEARISH")
        )
        signal_minute = (
            signal_ts.hour * 60 + signal_ts.minute
            if isinstance(signal_ts, datetime)
            else 0
        )
        setup_context = setup.get("context", {}) or {}
        market_structure = setup_context.get("market_structure", {}) or {}
        structure_state = market_structure.get("structure_state", {}) or {}
        liquidity_event = market_structure.get("liquidity_event", {}) or {}
        bos_dir = str(structure_state.get("bos_direction", "") or "").upper()
        choch_dir = str(structure_state.get("choch_direction", "") or "").upper()
        range_break_confirmed = (
            bias == "RANGING"
            and int(dte or 0) > 0
            and signal_minute <= 10 * 60
            and setup_type == "vote_aligned"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_84
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_58
            and {"SuperTrend+RSI", "VWAP+EMA", "ADX+PSAR"}.issubset(strategy_set)
            and bos_dir == direction
            and choch_dir == direction
            and str(liquidity_event.get("type", "") or "") == "structure_break"
            and not bool(liquidity_event.get("fake_breakout", False))
        )



        # Early structure / compression breakout (tight to mean, fresh displacement)
        is_early_structure = bool(
            {"ORB", "CPR", "FVG", "ValueArea"}.intersection(strategy_set)
        )

        institutional_flow_present = bool(
            {"OIAnalysis", "VolumeProfile", "FVG", "CPR", "ORB", "ValueArea"}.intersection(strategy_set)
        )
        high_conviction_full_budget = (
            (setup_strength_gate >= 0.60 or len(strategy_set) >= 5)
            and ml_rank_score >= 0.60
            and institutional_flow_present
        )
        if high_conviction_full_budget:
            if len(strategy_set) >= 6 or ml_rank_score >= 0.70 or (is_early_structure and ml_rank_score >= 0.65):
                return min(3, max_lots)
            return min(2, max_lots)
        midday_skew_value_put = (
            direction == "BUY_PUT"
            and setup_type == "vote_aligned"
            and signal_minute >= 12 * 60
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_84
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_58
            and {"SuperTrend+RSI", "ADX+PSAR", "SkewHunter", "ValueArea"}.issubset(strategy_set)
            and bias != "BULLISH"
        )
        if midday_skew_value_put:
            return min(2, max_lots)
        if (
            direction == "BUY_CALL"
            and bias == "BEARISH"
            and "ORB" not in strategy_set
        ):
            return 1
        if (
            direction == "BUY_PUT"
            and int(dte or 0) == 0
            and signal_minute < 12 * 60
            and not {"ADX+PSAR", "OIAnalysis", "SkewHunter"}.intersection(strategy_set)
        ):
            return 1

        if (
            direction == "BUY_CALL"
            and setup_type == "vote_aligned"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_83
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_57
            and {"SuperTrend+RSI", "VWAP+EMA", "ADX+PSAR"}.issubset(strategy_set)
        ):
            return min(2, max_lots)
        if (
            direction == "BUY_CALL"
            and setup_type == "vote_aligned"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_8
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_5
            and {"SuperTrend+RSI", "ADX+PSAR"}.issubset(strategy_set)
            and "SkewHunter" not in strategy_set
            and bias != "BEARISH"
        ):
            return min(2, max_lots)
        if (
            direction == "BUY_CALL"
            and setup_type in {"vote_aligned", "breakout"}
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_82
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_57
            and {"SuperTrend+RSI", "BBSqueeze"}.issubset(strategy_set)
            and "SkewHunter" not in strategy_set
        ):
            return min(2, max_lots)
        if (
            direction == "BUY_CALL"
            and setup_type == "vote_aligned"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_84
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_6
            and {"VWAP+EMA", "ORB", "FVG", "OIAnalysis"}.issubset(strategy_set)
        ):
            return min(2, max_lots)
        if (
            direction == "BUY_CALL"
            and setup_type == "vote_aligned"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_8
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_58
            and {"BBSqueeze", "ValueArea"}.issubset(strategy_set)
            and "SuperTrend+RSI" not in strategy_set
            and not {"ADX+PSAR", "StochRSI", "OpeningRangeBias"}.intersection(strategy_set)
        ):
            return min(2, max_lots)
        if (
            direction == "BUY_CALL"
            and setup_type == "trend_pullback"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_8
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_6
            and {"SuperTrend+RSI", "ADX+PSAR", "ValueArea"}.issubset(strategy_set)
            and not {"OpeningRangeBias", "StochRSI", "BBSqueeze"}.intersection(strategy_set)
            and bias != "BEARISH"
        ):
            return min(2, max_lots)
        if (
            direction == "BUY_PUT"
            and aligned_bias
            and setup_type == "vote_aligned"
            and signal_minute < 10 * 60
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_8
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_56
            and {"ADX+PSAR", "ExpiryWeek"}.issubset(strategy_set)
        ):
            return min(2, max_lots)
        if (
            setup_type == "breakout"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_7
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_52
            and {"BBSqueeze", "FVG"}.issubset(strategy_set)
        ):
            return min(2, max_lots)
        if (
            direction == "BUY_CALL"
            and setup_type == "breakout"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_THRESH_0_72
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_57
            and {"BBSqueeze", "FVG", "ValueArea"}.issubset(strategy_set)
            and "SkewHunter" not in strategy_set
        ):
            return min(2, max_lots)
        if (
            setup_type == "vote_aligned"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_82
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_6
            and len(strategy_set) >= 6
            and {"ORB", "FVG", "OIAnalysis"}.issubset(strategy_set)
        ):
            return min(2, max_lots)
        if range_break_confirmed:
            return min(2, max_lots)
        if (
            direction == "BUY_PUT"
            and aligned_bias
            and setup_type == "vote_aligned"
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_8
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_54
            and {"FVG", "Ichimoku"}.issubset(strategy_set)
            and "SkewHunter" not in strategy_set
        ):
            return min(2, max_lots)
        if (
            direction == "BUY_PUT"
            and (aligned_bias or bias == "RANGING")
            and setup_type == "vote_aligned"
            and signal_minute <= 11 * 60
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_8
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_58
            and {"ADX+PSAR", "SkewHunter", "ExpiryWeek"}.issubset(strategy_set)
            and "SuperTrend+RSI" not in strategy_set
        ):
            return min(2, max_lots)
        if bias == "RANGING" or ml_rank_score < PLANNER_ML_RANK_SCORE_THRESH_0_62 or setup_strength_gate < PLANNER_SETUP_STRENGTH_GATE_THRESH_0_82:
            return 1
        if (
            direction == "BUY_PUT"
            and "SkewHunter" in strategy_set
            and "ADX+PSAR" in strategy_set
            and aligned_bias
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_63
            and setup_strength_gate >= PLANNER_SETUP_STRENGTH_GATE_THRESH_0_82
            and setup_type in {"vote_aligned", "trend_pullback"}
        ):
            return min(2, max_lots)
        if direction == "BUY_PUT" and "SkewHunter" in strategy_set:
            return 1
        if not aligned_bias:
            return 1

        trend_core = (
            len(strategy_set) >= 3
            and (
                {"SuperTrend+RSI", "ADX+PSAR"}.issubset(strategy_set)
                or {"VWAP+EMA", "ADX+PSAR"}.issubset(strategy_set)
                or {"SuperTrend+RSI", "VWAP+EMA", "BBSqueeze"}.issubset(strategy_set)
            )
            and setup_type in {"vote_aligned", "trend_pullback", "breakout"}
        )
        if trend_core:
            return min(2, max_lots)
        return 1

    @staticmethod
    def _should_apply_wyckoff_size_cap(
        *,
        direction: str,
        strategies: list[str],
        setup: dict | None,
        ml_rank_score: float,
        structure_bias: str,
        dte: int | None = None,
        signal_ts: datetime | None = None,
    ) -> bool:
        """Use Wyckoff as a risk reducer, not as a veto on confirmed edge."""
        setup = setup or {}
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = round(float(setup.get("setup_strength", 0.0) or 0.0), 2)
        strategy_set = {str(s).strip() for s in strategies if str(s).strip()}
        bias = str(structure_bias or "UNKNOWN").upper()
        signal_minute = (
            signal_ts.hour * 60 + signal_ts.minute
            if isinstance(signal_ts, datetime)
            else 0
        )
        aligned_bias = (
            (direction == "BUY_CALL" and bias == "BULLISH")
            or (direction == "BUY_PUT" and bias == "BEARISH")
        )

        clean_trend_call = (
            direction == "BUY_CALL"
            and setup_type == "vote_aligned"
            and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_8
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_5
            and {"SuperTrend+RSI", "ADX+PSAR"}.issubset(strategy_set)
            and "SkewHunter" not in strategy_set
            and bias != "BEARISH"
        )
        confirmed_breakout = (
            setup_type == "breakout"
            and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_72
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_54
            and {"BBSqueeze", "FVG"}.issubset(strategy_set)
            and (aligned_bias or bias == "RANGING")
        )
        fvg_value_breakout = (
            direction == "BUY_CALL"
            and setup_type == "breakout"
            and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_72
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_57
            and {"BBSqueeze", "FVG", "ValueArea"}.issubset(strategy_set)
            and "SkewHunter" not in strategy_set
        )
        institutional_orb = (
            setup_type == "vote_aligned"
            and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_84
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_6
            and {"ORB", "FVG", "OIAnalysis"}.issubset(strategy_set)
        )
        expiry_skew_put = (
            direction == "BUY_PUT"
            and setup_type == "vote_aligned"
            and signal_minute <= 11 * 60
            and setup_strength >= PLANNER_SETUP_STRENGTH_THRESH_0_81
            and ml_rank_score >= PLANNER_ML_RANK_SCORE_THRESH_0_59
            and int(dte or 0) <= 3
            and {"ADX+PSAR", "SkewHunter", "ExpiryWeek"}.issubset(strategy_set)
            and "SuperTrend+RSI" not in strategy_set
        )
        if len(strategy_set) >= 8 or ml_rank_score >= 0.70 or (len(strategy_set) >= 5 and ml_rank_score >= 0.65):
            return False
        return not (
            clean_trend_call
            or confirmed_breakout
            or fvg_value_breakout
            or institutional_orb
            or expiry_skew_put
        )

    @staticmethod
    def _cap_lots_by_policy(
        *,
        desired_lots: int,
        entry_premium: float,
        lot_size: int,
        cap_pct_override: float | None = None,
        absolute_cap_override: float | None = None,
    ) -> int:
        max_lots = max(1, int(MAX_POSITION_LOTS))
        premium = max(float(entry_premium or 0.0), 0.0)
        lot = max(int(lot_size or NIFTY_LOT_SIZE), 1)
        lots = max(1, min(int(desired_lots or 1), max_lots))
        is_bt = (os.getenv("TRADING_MODE") == "BACKTEST") or (cap_pct_override is not None and cap_pct_override >= 30.0)
        max_trade_value = float(BACKTEST_MAX_TRADE_INVESTMENT_INR if is_bt else (DEPLOYED_CAPITAL or 40000.0))
        absolute_cap = max(
            premium * lot,
            min(
                BACKTEST_MAX_TRADE_INVESTMENT_INR if is_bt else MAX_TRADE_INVESTMENT_INR,
                STRONG_TRADE_INVESTMENT_INR if lots >= 2 else BASE_TRADE_INVESTMENT_INR,
            ),
        )
        if absolute_cap_override is not None:
            absolute_cap = max(premium * lot, min(absolute_cap, float(absolute_cap_override)))
        max_trade_value = min(max_trade_value, absolute_cap)
        max_lots_by_capital = max(1, int(max_trade_value // max(premium * lot, 1.0)))
        return max(1, min(lots, max_lots_by_capital, max_lots))

    @staticmethod
    def _lot_cap_pct(desired_lots: int) -> float:
        lots = int(desired_lots or 1)
        if lots >= 3:
            return 30.0
        elif lots == 2:
            return 20.0
        return 15.0

    @staticmethod
    def _projected_target_pnl_pct(
        *,
        entry_premium: float,
        exit_premium: float,
        lot_size: int,
    ) -> float:
        quantity = max(int(lot_size or 0), 1)
        gross_entry_value = max(entry_premium * quantity, 1.0)
        transaction_pct = max(BACKTEST_TRANSACTION_COST_PCT, 0.0) / 100.0
        costs = (
            max(BACKTEST_BROKERAGE_PER_ORDER, 0.0) * 2.0
            + (entry_premium * quantity + exit_premium * quantity) * transaction_pct
        )
        realized = (exit_premium - entry_premium) * quantity - costs
        return round(realized / gross_entry_value * 100, 2)

    @staticmethod
    def _estimate_delta(
        *,
        nifty_ltp: float,
        strike: int,
        option_type: str,
    ) -> float:
        moneyness = (nifty_ltp - strike) / max(nifty_ltp, 1)
        if option_type == "CE":
            return round(max(0.15, min(0.75, 0.5 + moneyness * 6)), 3)
        return round(-max(0.15, min(0.75, 0.5 - moneyness * 6)), 3)

    @staticmethod
    def _resolve_signal_ts(raw: Union[str, None]) -> datetime:
        if raw:
            return datetime.fromisoformat(str(raw))
        return datetime.now()

    @staticmethod
    def _minutes_to_close(signal_ts: datetime) -> int:
        close_hour, close_minute = [
            int(part) for part in str(MARKET_CLOSE_TIME).split(":", 1)
        ]
        close_ts = signal_ts.replace(
            hour=close_hour,
            minute=close_minute,
            second=0,
            microsecond=0,
        )
        return max(int((close_ts - signal_ts).total_seconds() // 60), 0)

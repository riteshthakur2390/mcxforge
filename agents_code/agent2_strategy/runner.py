from __future__ import annotations
"""
agents_code/agent2_strategy/runner.py  Strategy Engine Agent
==============================================================
Gate 2. Three performance optimisations applied:

  1. PARALLEL execution via ThreadPoolExecutor
     All eligible strategies run simultaneously — total time = slowest one

  2. SHARED IndicatorCache
     All TA indicators computed ONCE, shared across all 14 strategies
     Eliminates ~150ms of redundant recomputation per candle

  3. SMART ELIGIBILITY per strategy
     Each strategy declares min_candles + requirements
     OI/IV marked requires_live_broker: skip in backtest (no OIRecorder in replay)
     No hard global gate that blocks everything

VERDICT on splitting into multiple Strategy Agents:
    NOT recommended. Bus overhead (+30-50ms per hop) + vote aggregation
    complexity cancels any gain. ThreadPool inside one agent is optimal.

Speed profile per candle (200ms total pipeline):
    IndicatorCache build:  ~35ms  (once)
    14 strategies parallel: ~80ms  (bottleneck = S10 VolumeProfile)
    ML filter:             ~85ms  (runs in parallel — see below)
    Trade planner:         ~20ms
    Bus routing:           ~15ms

Strategies:
    S1 SuperTrend + RSI           trend flip with momentum
    S2 VWAP + EMA Crossover       intraday mean-reversion
    S3 Opening Range Breakout     morning momentum
    S4 BB Squeeze + Stochastic    volatility expansion
    S5 ADX + Parabolic SAR        strong trend + SAR flip
    S6 Fair Value Gap (FVG)       SMC imbalance zones
    S7 UT Bot                     ATR trailing stop crossover
    S8 CPR                        Central Pivot Range breakout
    S9 Ichimoku Cloud             multi-timeframe confluence
    S10 Volume Profile            VPOC / Value Area levels
    S11 Liquidity Sweep           stop hunt reversal detection
    S12 Price Action              candle patterns at key levels
    S13 OI Analysis               institutional OI buildup / proxy CMF
    S14 IV Contraction            volatility squeeze + directional break
    S18 OI-IV Confluence          high-conviction squeeze breakout
Needs MIN_STRATEGY_VOTES in same direction to fire RAW_SIGNAL.
"""

import asyncio
from datetime import datetime, timedelta
from loguru import logger
import pytz
import time
from dataclasses import dataclass
from typing import Union
import concurrent.futures
import json
from pathlib import Path

import sys, os
import pandas as pd

# STRATEGY PERFORMANCE WEIGHTS (based on recent backtest analysis)
# High weight = boost these strategies, Low weight = reduce influence
STRATEGY_PERFORMANCE_WEIGHTS = {
    'S1': 1.0,   # SuperTrend + RSI - baseline
    'S2': 0.85,  # VWAP + EMA - underperforming recently
    'S3': 1.15,  # Opening Range Breakout - strong momentum plays
    'S4': 0.90,  # BB Squeeze - moderate
    'S5': 1.10,  # ADX + PSAR - trend following strong
    'S6': 1.0,   # FVG - baseline
    'S7': 1.05,  # UT Bot - decent trailing
    'S8': 0.80,  # CPR - underperforming
    'S9': 1.10,  # Ichimoku - multi-timeframe strong
    'S10': 1.0,  # Volume Profile - baseline
    'S11': 0.90, # Liquidity Sweep - moderate
    'S12': 1.05, # Price Action - decent
    'S13': 0.75, # OI Analysis - underperforming
    "S15": 1.5,  # structure + manipulation = high independence
    "S16": 1.4,  # order block + structure = high independence
    'S25': 1.10, # Squeeze Momentum
    'S26': 1.05, # StochRSI
    'S27': 1.00, # EMASlope
    'S28': 1.05, # HeikinAshi
    'S29': 1.10, # VWAPExtreme
    'S31': 1.10, # OpeningRangeBias
    'S34': 1.00, # Elliott Wave - neutral until live sample builds
}

# STRATEGY CATEGORY MAPPING for Independent Category Vote Count
STRATEGY_CATEGORY_MAPPING: dict[str, str] = {
    # trend / momentum
    "SuperTrend+RSI": "trend / momentum",
    "SuperTrendRSI": "trend / momentum",
    "VWAP+EMA": "trend / momentum",
    "VWAPema": "trend / momentum",
    "ADX+PSAR": "trend / momentum",
    "ADXPsar": "trend / momentum",
    "ADXRising": "trend / momentum",
    "UTBot": "trend / momentum",
    "Ichimoku": "trend / momentum",
    "EMASlope": "trend / momentum",
    "HeikinAshi": "trend / momentum",
    "StochRSI": "trend / momentum",
    "VWAPExtreme": "trend / momentum",

    # structure / SMC
    "FVG": "structure / SMC",
    "LiqSweep": "structure / SMC",
    "LiquiditySweep": "structure / SMC",
    "PriceAction": "structure / SMC",
    "AMD": "structure / SMC",
    "SMC": "structure / SMC",
    "VolumeProfile": "structure / SMC",
    "ValueArea": "structure / SMC",
    "ElliottWave": "structure / SMC",

    # level / breakout
    "ORB": "level / breakout",
    "CPR": "level / breakout",
    "GapDirection": "level / breakout",
    "GapMomentum": "level / breakout",
    "OpeningRangeBias": "level / breakout",

    # volatility
    "BBSqueeze": "volatility",
    "RangeSpread": "volatility",
    "SqueezeMomentum": "volatility",

    # flow / external
    "OIAnalysis": "flow / external",
}

try:
    from ml.signal_scorer import SignalQualityScorer
    _SCORER_AVAILABLE = True
except ImportError:
    _SCORER_AVAILABLE = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bus import get_bus, Topic, Message
from core.models import Direction, RawSignal, Regime
from broker.factory import get_broker
from utils.signal_conflict_resolver import get_resolver
from config.settings.strategy_runner_thresholds import *
from config.settings import (
    SIGNAL_START_TIME,
    NO_NEW_SIGNAL_AFTER, ONE_SIGNAL_AT_A_TIME,
    MAX_CONCURRENT_SIGNALS,
    DUPLICATE_SIGNAL_COOLDOWN_MINUTES, NIFTY_STRIKE_STEP,
    STOP_LOSS_PCT, TARGET_PCT, MIN_DAYS_TO_EXPIRY,
    ALLOW_REENTRY, REENTRY_COOLDOWN_M, MARKET_CLOSE_TIME,
    CAS_START_TIME, CAS_END_TIME,
    MIN_SETUP_STRENGTH, SETUP_GATE_ENABLED,
    ML_QUALITY_MIN_SCORE,
    SECOND_TRADE_LAST_ENTRY_MINUTE,
    SECOND_TRADE_MIN_VOTES,
    SECOND_TRADE_MIN_WEIGHTED_SCORE,
    SECOND_TRADE_COOLDOWN_MIN,
    SECOND_TRADE_MIN_CONF,
    SECOND_TRADE_MIN_SETUP,
    SECOND_TRADE_REQUIRE_TREND,
    SECOND_TRADE_FAST_REENTRY_MIN,
    SECOND_TRADE_FAST_LAST_ENTRY_MINUTE,
    SECOND_TRADE_FAST_MIN_SETUP,
    SECOND_TRADE_FAST_MIN_CONF,
    SECOND_TRADE_FAST_MIN_VOTES,
    SECOND_TRADE_FAST_MIN_WEIGHTED_SCORE,
    ENTRY_MIN_DET_CONF,
    ENTRY_BLOCK_RANGING,
)
from config.settings.strategy import (
    ALLOW_SUBMIN_VOTE_EARLY_TRIGGER,
    EARLY_TRIGGER_MIN_CONFIDENCE,
    MIN_STRATEGY_CONF,
    MIN_STRATEGY_VOTES,
    RUNNER_CONF_MAX,
    RUNNER_EARLY_TRIGGER_BONUS_CANDLE,
    RUNNER_EARLY_TRIGGER_HIGH_CONF,
    RUNNER_EARLY_TRIGGER_MIN_CONF,
    RUNNER_EARLY_TRIGGER_STRENGTH_HIGH,
    RUNNER_EARLY_TRIGGER_STRENGTH_THRESHOLD,
    RUNNER_EARLY_TRIGGER_WEIGHTED_SCORE_HIGH,
    RUNNER_EARLY_TRIGGER_WEIGHTED_SCORE_THRESHOLD,
    RUNNER_EXPECTED_MOVE_ATR_MULT,
    RUNNER_EXPECTED_MOVE_ATR_PERIOD,
    RUNNER_EXPECTED_MOVE_LTP_MULT,
    RUNNER_EXPECTED_MOVE_ORB_MULT,
    RUNNER_GAP_BIAS_BONUS,
    RUNNER_REVERSION_STRENGTH_THRESHOLD,
    RUNNER_REVERSION_WEIGHTED_SCORE_THRESHOLD,
    RUNNER_SEPARATION_BONUS_MAX,
    RUNNER_SEPARATION_BONUS_MULT,
    RUNNER_SETUP_BONUS_MAX,
    RUNNER_SETUP_BONUS_MULT,
    RUNNER_SETUP_BONUS_THRESHOLD,
    RUNNER_SPOT_SL_MULT,
    RUNNER_TREND_PERSIST_MAX_TIME,
    RUNNER_TREND_PERSIST_MID_TIME,
    RUNNER_VOTE_BONUS,
    RUNNER_WEIGHT_BONUS_MAX,
    RUNNER_WEIGHT_BONUS_MULT,
    RUNNER_WEIGHT_BONUS_THRESHOLD,
    S10_MIN_DF_LEN,
    S11_MIN_DF_LEN,
    S12_MIN_DF_LEN,
    S13_MIN_DF_LEN_RUNNER,
    S14_MIN_DF_LEN_RUNNER,
)

# Late-Entry / Move-Exhaustion Rejection Filter Configuration (Active for strongest exhaustion conditions)
ENABLE_LATE_ENTRY_FILTER: bool = os.getenv("ENABLE_LATE_ENTRY_FILTER", "1").lower() in {"1", "true", "yes"}
MAX_ENTRY_EXTENSION_ATR: float = float(os.getenv("MAX_ENTRY_EXTENSION_ATR", "2.20"))
MAX_ENTRY_DIST_EMA_ATR: float = float(os.getenv("MAX_ENTRY_DIST_EMA_ATR", "2.25"))
MAX_CONSEC_BARS_BEFORE_REJECTION: int = int(os.getenv("MAX_CONSEC_BARS_BEFORE_REJECTION", "5"))

from config.settings.strategy import (
    S15_MIN_DF_LEN,
    S16_MIN_DF_LEN,
    S1_MIN_DF_LEN,
    S2_MIN_DF_LEN,
    S3_MIN_DF_LEN,
    S4_BB_LENGTH,
    S4_BBW_MA_PERIOD,
    S5_MIN_DF_LEN,
    S6_MIN_DF_LEN,
    S7_MIN_DF_LEN_RUNNER,
    S8_MIN_DF_LEN,
    S8_MIN_PREV_DAY_CANDLES,
    S9_MIN_DF_LEN,
    S25_MIN_CANDLES,
    S26_MIN_CANDLES,
    S27_MIN_CANDLES,
    S28_MIN_CANDLES,
    S29_MIN_CANDLES,
    S30_MIN_CANDLES,
    S31_MIN_CANDLES,
    S32_MIN_CANDLES,
    STRATEGY_BASE_WEIGHTS,
    STRATEGY_REGIME_WEIGHTS,
)
try:
    from ml.signal_scorer import SignalQualityScorer
    _SCORER_AVAILABLE = True
except ImportError:
    _SCORER_AVAILABLE = False
    logger.warning("[runner] ml.signal_scorer not found — quality scoring disabled")

from agents_code.agent2_strategy.indicator_cache import IndicatorCache
from agents_code.agent2_strategy.setup_engine import TradeSetup, TradeSetupEngine
from agents_code.agent2_strategy.s1_supertrend_rsi import SuperTrendRSI
from agents_code.agent2_strategy.s2_vwap_ema import VWAPEMACross
from agents_code.agent2_strategy.s3_orb import ORBStrategy
from agents_code.agent2_strategy.s4_bb_squeeze import BBSqueeze
from agents_code.agent2_strategy.s5_adx_psar import ADXParabolicSAR
from agents_code.agent2_strategy.s6_fvg import FVGStrategy
from agents_code.agent2_strategy.s7_utbot import UTBotStrategy
from agents_code.agent2_strategy.s8_cpr            import CPRStrategy
from agents_code.agent2_strategy.s9_ichimoku       import IchimokuStrategy
from agents_code.agent2_strategy.s10_volume_profile import VolumeProfileStrategy
from agents_code.agent2_strategy.s11_liquidity_sweep import (
    LiquiditySweepStrategy,
)
from agents_code.agent2_strategy.s12_price_action import (
    PriceActionStrategy,
)
from agents_code.agent2_strategy.s13_oi_analysis import (
    OIAnalysisStrategy,
)
from agents_code.agent2_strategy.s15_amd             import AMDStrategy
from agents_code.agent2_strategy.s16_gap_direction   import GapDirectionStrategy
from agents_code.agent2_strategy.s20_volume_profile  import VolumeProfileStrategy as ValueAreaStrategy
from agents_code.agent2_strategy.s22_gap_momentum    import GapMomentumStrategy
from agents_code.agent2_strategy.s23_adx_rising      import ADXRisingStrategy
from agents_code.agent2_strategy.s24_range_spread    import RangeSpreadStrategy
from agents_code.agent2_strategy.s25_squeeze_momentum import SqueezeMomentumStrategy
from agents_code.agent2_strategy.s26_stoch_rsi       import StochRSIStrategy
from agents_code.agent2_strategy.s27_ema_slope       import EMASlopeStrategy
from agents_code.agent2_strategy.s28_heikin_ashi     import HeikinAshiStrategy
from agents_code.agent2_strategy.s29_vwap_extreme    import VWAPExtremeStrategy
from agents_code.agent2_strategy.s31_opening_range_bias import OpeningRangeBiasStrategy
from agents_code.agent2_strategy.s34_elliott_wave    import ElliottWaveStrategy
from utils.instrument_selector import compute_atr
from agents_code.agent2_strategy.s17_smc             import SMCStrategy            # NEW S17
from utils.morning_bias import get_morning_bias_system
from utils.pipeline_logging import log_pipeline_stage
from utils.options_flow_detector import get_flow_detector
from utils.advanced_filters import (
    deduplicate_signals,
    get_pf_gate,
)
from utils.market_microstructure_edges import get_microstructure
from agents_code.agent9_regime.wyckoff_phase_detector import WyckoffResult

IST = pytz.timezone("Asia/Kolkata")
STRATEGY_EXEC_LOGGER = logger.bind(strategy_exec=True)


@dataclass
class StrategyMeta:
    instance: object
    min_candles: int
    requires_live_broker: bool  # True = skip in backtest
    requires_orb: bool  # True = skip before 09:30
    name: str
    requires_volume: bool = False  # True = skip when replay/live feed has no usable volume


@dataclass(frozen=True)
class StrategyReadiness:
    ready: bool
    required_candles: int
    reason: str = ""


@dataclass(frozen=True)
class WeightedVoteSummary:
    count: int
    weight_total: float
    weighted_score: float
    avg_weight: float


STRATEGY_REGISTRY: list[StrategyMeta] = [
    StrategyMeta(SuperTrendRSI(), S1_MIN_DF_LEN, False, False, "SuperTrend+RSI"),
    StrategyMeta(VWAPEMACross(), S2_MIN_DF_LEN, False, False, "VWAP+EMA"),
    StrategyMeta(ORBStrategy(), S3_MIN_DF_LEN, False, True, "ORB", True),
    StrategyMeta(BBSqueeze(), max(S4_BB_LENGTH, S4_BBW_MA_PERIOD), False, False, "BBSqueeze"),
    StrategyMeta(ADXParabolicSAR(), S5_MIN_DF_LEN, False, False, "ADX+PSAR"),
    StrategyMeta(FVGStrategy(), S6_MIN_DF_LEN, False, False, "FVG", True),
    StrategyMeta(UTBotStrategy(), S7_MIN_DF_LEN_RUNNER, False, False, "UTBot", True),
    StrategyMeta(CPRStrategy(), S8_MIN_DF_LEN, False, False, "CPR", True),
    StrategyMeta(IchimokuStrategy(), S9_MIN_DF_LEN, False, False, "Ichimoku", True),
    StrategyMeta(VolumeProfileStrategy(), S10_MIN_DF_LEN, False, False, "VolumeProfile", True),
    StrategyMeta(LiquiditySweepStrategy(), S11_MIN_DF_LEN, False, False, "LiqSweep", True),
    StrategyMeta(PriceActionStrategy(), S12_MIN_DF_LEN, False, False, "PriceAction", True),
    # OIAnalysis can use live recorder OI/IV even when index/option volume
    # enrichment has not warmed up yet.
    StrategyMeta(OIAnalysisStrategy(), S13_MIN_DF_LEN_RUNNER, False, False, "OIAnalysis"),
    StrategyMeta(AMDStrategy(), S15_MIN_DF_LEN, False, False, "AMD", True),
    StrategyMeta(GapDirectionStrategy(), S16_MIN_DF_LEN, False, False, "GapDirection", True),
    StrategyMeta(SMCStrategy(), 50, False, False, "SMC", True),
    StrategyMeta(ValueAreaStrategy(), 30, False, False, "ValueArea", True),
    StrategyMeta(GapMomentumStrategy(), 20, False, False, "GapMomentum", True),
    StrategyMeta(ADXRisingStrategy(), 25, False, False, "ADXRising"),
    StrategyMeta(RangeSpreadStrategy(), 20, False, False, "RangeSpread", True),
    StrategyMeta(SqueezeMomentumStrategy(), S25_MIN_CANDLES, False, False, "SqueezeMomentum", True),
    StrategyMeta(StochRSIStrategy(), S26_MIN_CANDLES, False, False, "StochRSI"),
    StrategyMeta(EMASlopeStrategy(), S27_MIN_CANDLES, False, False, "EMASlope"),
    StrategyMeta(HeikinAshiStrategy(), S28_MIN_CANDLES, False, False, "HeikinAshi", True),
    StrategyMeta(VWAPExtremeStrategy(), S29_MIN_CANDLES, False, False, "VWAPExtreme", True),
    StrategyMeta(OpeningRangeBiasStrategy(), S31_MIN_CANDLES, False, True, "OpeningRangeBias", True),
    StrategyMeta(ElliottWaveStrategy(), 40, False, False, "ElliottWave", True),
]
ALL_STRATEGY_NAMES = [meta.name for meta in STRATEGY_REGISTRY]
BACKTEST_ENABLE_ADAPTIVE_GATES = os.getenv(
    "BACKTEST_ENABLE_ADAPTIVE_GATES",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
LIVE_ENABLE_ADAPTIVE_GATES = os.getenv(
    "LIVE_ENABLE_ADAPTIVE_GATES",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
LIVE_ENABLE_ADAPTIVE_EDGE_FILTERS = os.getenv(
    "LIVE_ENABLE_ADAPTIVE_EDGE_FILTERS",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
HYBRID_5M_CONFIRM_ENABLED = os.getenv(
    "HYBRID_5M_CONFIRM_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
HYBRID_5M_MIN_CONF = float(os.getenv("HYBRID_5M_MIN_CONF", "0.58"))
HYBRID_5M_STRONG_COUNTER_SETUP = float(os.getenv("HYBRID_5M_STRONG_COUNTER_SETUP", "0.86"))
HYBRID_5M_STRONG_COUNTER_WS = float(os.getenv("HYBRID_5M_STRONG_COUNTER_WS", "2.40"))
HYBRID_5M_STRONG_COUNTER_VOTES = int(os.getenv("HYBRID_5M_STRONG_COUNTER_VOTES", "3"))
HYBRID_5M_ANCHOR_ENABLED = os.getenv(
    "HYBRID_5M_ANCHOR_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
HYBRID_5M_ANCHOR_MIN_VOTES = int(os.getenv("HYBRID_5M_ANCHOR_MIN_VOTES", "2"))
HYBRID_5M_ANCHOR_MIN_SCORE = float(os.getenv("HYBRID_5M_ANCHOR_MIN_SCORE", "1.20"))
HYBRID_5M_ANCHOR_EXCLUDED = {
    "ValueArea",
    "GapMomentum",
    "ADXRising",
    "RangeSpread",
}
HYBRID_5M_EMA_BLOCK_ENABLED = os.getenv(
    "HYBRID_5M_EMA_BLOCK_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}


def _format_strategy_market_ts(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    ts = pd.Timestamp(df.index[-1]).to_pydatetime()
    if ts.tzinfo is None:
        ts = IST.localize(ts)
    else:
        ts = ts.astimezone(IST)
    return ts.strftime("%Y-%m-%d %H:%M IST")


def _log_strategy_event(
    *,
    name: str,
    status: str,
    df: pd.DataFrame,
    required_candles: int,
    reason: str = "",
) -> None:
    parts = [
        f"market_ts={_format_strategy_market_ts(df)}",
        f"status={status}",
        f"strategy={name}",
        f"candles={len(df)}",
        f"required={required_candles}",
    ]
    if reason:
        parts.append(f"reason={reason}")
    STRATEGY_EXEC_LOGGER.debug(" | ".join(parts))


def _regime_weight_label(detailed_regime: dict, fallback_regime: str) -> str:
    label = str(detailed_regime.get("label", "") or "").upper()
    if label in STRATEGY_REGIME_WEIGHTS:
        return label
    fallback = str(fallback_regime or "").upper()
    if fallback == "HIGH_VOL":
        return "HIGH_VOLATILITY"
    if fallback in STRATEGY_REGIME_WEIGHTS:
        return fallback
    return "TRENDING"


def _vote_weight_debug(votes: list[dict]) -> str:
    if not votes:
        return "[]"
    return "[" + ", ".join(
        (
            f"{v['name']}:"
            f"b{float(v.get('_base_weight', 1.0)):.2f}/"
            f"r{float(v.get('_regime_weight', 1.0)):.2f}/"
            f"w{float(v.get('_vote_weight', 1.0)):.2f}/"
            f"s{float(v.get('_weighted_score', 0.0)):.2f}"
        )
        for v in votes
    ) + "]"


def _run_strategy_sync(
        meta: StrategyMeta,
        df: pd.DataFrame,
        cache: IndicatorCache,
        orb_high: Union[float, None],
        orb_low: Union[float, None],
        strategy_context: Union[dict, None] = None,
) -> dict:
    """
    Runs one strategy synchronously inside ThreadPoolExecutor.
    Passes IndicatorCache so strategies avoid redundant TA recomputation.
    """
    t0 = time.perf_counter()
    
    # Check minimum candles requirement
    if len(df) < meta.min_candles:
        _log_strategy_event(
            name=meta.name,
            status="skipped",
            df=df,
            required_candles=meta.min_candles,
            reason=f"need at least {meta.min_candles} candles",
        )
        return {
            "direction": Direction.NONE,
            "confidence": 0.0,
            "name": meta.name,
            "_ms": round((time.perf_counter() - t0) * 1000, 1),
            "_skipped": True,
            "_skipped_reason": "insufficient_data",
        }
        
    # Check ORB requirement
    if meta.requires_orb and orb_high is None:
        _log_strategy_event(
            name=meta.name,
            status="skipped",
            df=df,
            required_candles=meta.min_candles,
            reason="ORB levels not available yet",
        )
        return {
            "direction": Direction.NONE,
            "confidence": 0.0,
            "name": meta.name,
            "_ms": round((time.perf_counter() - t0) * 1000, 1),
            "_skipped": True,
            "_skipped_reason": "waiting_for_orb",
        }

    _log_strategy_event(
        name=meta.name,
        status="running",
        df=df,
        required_candles=meta.min_candles,
    )
    try:
        # Pass cache as keyword arg — strategies use it if they accept it,
        # otherwise they fall back to computing their own indicators
        context = strategy_context or {}
        try:
            result = meta.instance.evaluate(df, orb_high, orb_low, cache=cache, **context)
        except TypeError as e:
            if "got an unexpected keyword argument 'cache'" in str(e):
                # Strategy doesn't accept cache kwarg yet — backward compatible
                try:
                    result = meta.instance.evaluate(df, orb_high, orb_low, **context)
                except TypeError as context_exc:
                    if "got an unexpected keyword argument" in str(context_exc):
                        result = meta.instance.evaluate(df, orb_high, orb_low)
                    else:
                        raise
            elif "got an unexpected keyword argument" in str(e):
                # Legacy strategies without **kwargs.
                result = meta.instance.evaluate(df, orb_high, orb_low)
            else:
                raise

        result["_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["_skipped"] = False
        if "direction" in result:
            dir_val = result["direction"]
            if isinstance(dir_val, str):
                try:
                    result["direction"] = Direction(dir_val)
                except ValueError:
                    result["direction"] = Direction.NONE
        return result
    except Exception as e:
        _log_strategy_event(
            name=meta.name,
            status="skipped",
            df=df,
            required_candles=meta.min_candles,
            reason=f"execution error: {e}",
        )
        return {
            "direction": Direction.NONE,
            "confidence": 0.0,
            "name": meta.name,
            "_ms": round((time.perf_counter() - t0) * 1000, 1),
            "_skipped": True,
            "_skipped_reason": "execution_error",
            "_error": str(e),
        }


def _has_meaningful_volume(df: pd.DataFrame, lookback: int) -> bool:
    if df.empty:
        return False
    # If in backtest, relax volume requirement as Nifty index often has 0 volume in free data
    if os.getenv("TRADING_MODE", "OBSERVE") == "BACKTEST":
        return True
    volume_col = "volume"
    if "volume" not in df.columns or not (pd.to_numeric(df["volume"], errors="coerce").fillna(0.0) > 0).any():
        option_volume_cols = [
            col for col in ("opt_total_volume", "opt_atm_volume", "opt_ce_volume", "opt_pe_volume")
            if col in df.columns
        ]
        if not option_volume_cols:
            return False
        option_volume = df[option_volume_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        # Live broker/index feeds can provide candles and option-chain context
        # while reporting zero volume. Treat the feature columns as available
        # so volume-aware strategies can evaluate instead of being disabled.
        if os.getenv("TRADING_MODE", "OBSERVE") != "BACKTEST":
            return True
        window = option_volume.tail(max(1, min(int(lookback or 1), len(df))))
        return bool((window > 0).any())
    window = df[volume_col].tail(max(1, min(int(lookback or 1), len(df))))
    volume = pd.to_numeric(window, errors="coerce").fillna(0.0)
    return bool((volume > 0).any())


def _option_volume_proxy(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    if "opt_total_volume" in df.columns:
        return pd.to_numeric(df["opt_total_volume"], errors="coerce").fillna(0.0)
    option_volume_cols = [
        col for col in ("opt_ce_volume", "opt_pe_volume", "opt_atm_volume")
        if col in df.columns
    ]
    if not option_volume_cols:
        return pd.Series(0.0, index=df.index)
    return df[option_volume_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


class StrategyAgent:
    NAME = "StrategyAgent"
    _STRATEGY_HEALTH_LOG_EVERY = 25

    def __init__(self) -> None:
        self.bus             = get_bus()
        self._position_open  = False
        self._trading_halted = False
        self._orb_high: Union[float, None] = None
        self._orb_low:  Union[float, None] = None
        self._last_signal_key: str = ""
        self._is_backtest = os.getenv("TRADING_MODE", "OBSERVE") == "BACKTEST"
        self._pool           = concurrent.futures.ThreadPoolExecutor(
            max_workers = len(STRATEGY_REGISTRY),
            thread_name_prefix = "strat",
        )
        # Re-entry tracking: allow 1 re-entry per session after SL hit
        self._last_sl_exit_time: datetime | None = None
        self._last_sl_direction: str = ""
        self._reentry_used_today: bool = False
        self._trades_today_count: int = 0
        self._last_close_time: datetime | None = None
        self._today_date: str = ""

        # Signal cooldown: prevent same-direction signals within 30 min
        self._last_signal_time: datetime | None = None
        self._last_signal_direction: str = ""
        self._last_signal_setup_type: str = ""

        self._last_signal_at: Union[datetime, None] = None
        self._broker = get_broker()
        self._scorer = SignalQualityScorer() if _SCORER_AVAILABLE else None
        self._setup_engine = TradeSetupEngine()
        self._morning_bias = get_morning_bias_system()
        self._scan_count = 0
        self._strategy_health: dict[str, dict[str, int]] = {
            meta.name: {"running": 0, "skipped": 0, "errors": 0, "voted": 0}
            for meta in STRATEGY_REGISTRY
        }
        try:
            from core.strategies.ensemble import build_default_strategy_suite
            for strat in build_default_strategy_suite():
                if strat.name not in self._strategy_health:
                    self._strategy_health[strat.name] = {"running": 0, "skipped": 0, "errors": 0, "voted": 0}
        except Exception:
            pass
        self._gate_counts = {
            "regime_soft_block": 0,
            "no_setup": 0,
            "insufficient_votes": 0,
            "setup_mismatch": 0,
            "duplicate_signal": 0,
            "late_entry_move_exhausted": 0,
        }
        self._is_backtest: bool = False
        self._is_backtest_mode: bool = False
        # Signal Timing Telemetry: track first candidate detection for timing cost measurement
        self._candidate_tracker: dict[str, dict] = {
            Direction.BUY_CALL.value: {"ts": None, "price": None, "votes": 0},
            Direction.BUY_PUT.value: {"ts": None, "price": None, "votes": 0},
        }
        # Early Candidate Shadow Tracker: Shadow-only candidate tracking to measure time saved & earlier entry viability
        self._candidate_shadow_history: list[dict] = []
        self._active_candidates: dict[str, Any] = {
            Direction.BUY_CALL.value: None,
            Direction.BUY_PUT.value: None,
        }
        try:
            from agents_code.agent2_strategy.live_shadow_option_tracker import LiveShadowOptionTracker
            self._live_shadow_tracker: Optional[LiveShadowOptionTracker] = LiveShadowOptionTracker()
        except Exception as e:
            logger.warning(f"[{self.NAME}] LiveShadowOptionTracker bypassed (fail-open): {e}")
            self._live_shadow_tracker = None

    def register(self) -> None:
        self.bus.subscribe(Topic.MARKET_REGIME,   self.on_candles)
        self.bus.subscribe(Topic.ORB_FORMED,      self.on_orb)
        self.bus.subscribe(Topic.POSITION_CLOSED, self._on_position_closed)
        self.bus.subscribe("SL_HIT_EVENT",        self._on_sl_hit)
        self.bus.subscribe(Topic.ORDER_PLACED,    self._on_position_opened)
        self.bus.subscribe(Topic.ORDER_DRY_RUN,   self._on_position_opened)
        self.bus.subscribe(Topic.ORDER_CONFIRM_REQ, self._on_position_opened)
        self.bus.subscribe(Topic.ORDER_SKIPPED,   self._on_position_closed)
        self.bus.subscribe(Topic.SYSTEM_STATUS,   self._on_system_status)
        self.bus.subscribe(Topic.SYSTEM_RESET,    self._on_system_reset)
        self.bus.subscribe(Topic.PREMARKET_BIAS,  self._on_premarket_bias)
        live_n = sum(1 for s in STRATEGY_REGISTRY if not s.requires_live_broker)
        total_strategies = len(self._strategy_health)
        logger.info(
            f"[{self.NAME}] Registered | {total_strategies} strategies | "
            f"{live_n} run in backtest | all run in live | "
            f"mode=parallel+cache | min_votes={MIN_STRATEGY_VOTES}"
        )

        if not ONE_SIGNAL_AT_A_TIME and MAX_CONCURRENT_SIGNALS > 1:
            logger.warning(
                f"[{self.NAME}] MAX_CONCURRENT_SIGNALS={MAX_CONCURRENT_SIGNALS} requested, "
                f"but current execution stack is still single-position. "
                f"Strategy gating can be relaxed, but full concurrent position support "
                f"needs a PositionManager/RiskGuard refactor."
            )
        logger.info(f"[{self.NAME}] Registered. Strategies: {total_strategies}")

    def strategy_names(self) -> list[str]:
        return list(self._strategy_health.keys())

    def set_backtest_mode(self, is_backtest: bool) -> None:
        self._is_backtest = is_backtest
        self._is_backtest_mode = is_backtest
        logger.info(f"[{self.NAME}] Mode={'BACKTEST' if is_backtest else 'LIVE'}")

    def strategy_health_status(self) -> dict:
        persisted_strategies = {}
        persisted_scans = 0
        try:
            p = Path("state/strategy_health.json")
            if p.exists():
                loaded = json.loads(p.read_text())
                persisted_scans = int(loaded.get("scans", 0))
                for name, s in (loaded.get("strategies", {}) or {}).items():
                    persisted_strategies[name] = {
                        "evaluated": int(s.get("evaluated", 0)),
                        "voted": int(s.get("voted", 0)),
                        "skipped": int(s.get("skipped", 0)),
                        "errors": int(s.get("errors", 0)),
                    }
        except Exception:
            pass

        merged_strategies = dict(persisted_strategies)
        for name, stats in self._strategy_health.items():
            prev = merged_strategies.get(name, {})
            merged_strategies[name] = {
                "evaluated": max(int(stats.get("running", 0)), int(prev.get("evaluated", 0))),
                "voted": max(int(stats.get("voted", 0)), int(prev.get("voted", 0))),
                "skipped": max(int(stats.get("skipped", 0)), int(prev.get("skipped", 0))),
                "errors": max(int(stats.get("errors", 0)), int(prev.get("errors", 0))),
            }

        status = {
            "scans": max(int(self._scan_count), persisted_scans),
            "strategies": merged_strategies,
        }
        try:
            p = Path("state/strategy_health.json")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(status, indent=2))
        except Exception:
            pass
        return status

    async def _on_premarket_bias(self, msg: Message) -> None:
        payload = msg.payload or {}
        try:
            self._morning_bias.compute_bias(
                gap_pct=float(payload.get("gap_pct", 0.0) or 0.0),
                india_vix=float(payload.get("india_vix", payload.get("vix", 18.0)) or 18.0),
                prev_close=float(payload.get("prev_close", payload.get("nifty_prev_close", 0.0)) or 0.0),
            )
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Morning bias update skipped: {exc}")

    def shutdown(self) -> None:
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def set_oi_recorder(self, recorder) -> None:
        injected = 0
        for meta in STRATEGY_REGISTRY:
            strat = meta.instance
            if hasattr(strat, "set_recorder"):
                strat.set_recorder(recorder)
                injected += 1
        logger.info(f"[{self.NAME}] OI recorder injected into {injected} strategy modules")

    @staticmethod
    def _with_effective_volume(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        current = (
            pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
            if "volume" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        option_volume = _option_volume_proxy(df)
        mode = os.getenv(
            "BACKTEST_INDEX_VOLUME_MODE" if os.getenv("TRADING_MODE", "OBSERVE") == "BACKTEST" else "INDEX_VOLUME_MODE",
            "option_proxy",
        ).strip().lower()

        if mode in {"broker", "raw"} and bool((current > 0).any()):
            return df

        if not bool((option_volume > 0).any()):
            option_volume_cols = [
                col for col in ("opt_total_volume", "opt_atm_volume", "opt_ce_volume", "opt_pe_volume")
                if col in df.columns
            ]
            if option_volume_cols and os.getenv("TRADING_MODE", "OBSERVE") != "BACKTEST":
                effective = df.copy()
                effective["underlying_volume"] = current
                effective["volume"] = 1.0
                return effective
            if mode in {"zero", "option_proxy", "benchmark"} and bool((current > 0).any()):
                effective = df.copy()
                effective["underlying_volume"] = current
                effective["volume"] = 0.0
                return effective
            return df

        effective = df.copy()
        effective["underlying_volume"] = current
        # NIFTY is an index, not a traded underlying. Broker index-volume fields
        # are not comparable across providers; option OHLCV is the tradable
        # volume proxy for FnO strategies.
        effective["volume"] = option_volume.astype(float)
        return effective

    async def on_orb(self, msg: Message) -> None:
        self._orb_high = msg.payload.get("orb_high")
        self._orb_low  = msg.payload.get("orb_low")
        logger.info(
            f"[{self.NAME}] ORB set | H={self._orb_high} L={self._orb_low}"
        )

    async def _on_position_opened(self, msg: Message) -> None:
        if not self._position_open:
            opened_ts = self._resolve_timestamp(msg.payload.get("timestamp"))
            self._reset_daily_trade_state(opened_ts)
            self._trades_today_count += 1
        self._position_open = True

    async def _on_position_closed(self, msg: Message) -> None:
        self._position_open = False
        close_ts = self._resolve_timestamp(
            msg.payload.get("exit_time") or msg.payload.get("timestamp")
        )
        self._reset_daily_trade_state(close_ts)
        self._last_close_time = close_ts
        exit_reason = str(msg.payload.get("exit_reason", "") or "").upper()
        dir_val = str(msg.payload.get("direction", "") or "").upper()
        if not dir_val:
            sym = str(msg.payload.get("option_symbol", "") or "").upper()
            if sym.endswith("CE"):
                dir_val = "BUY_CALL"
            elif sym.endswith("PE"):
                dir_val = "BUY_PUT"
        if exit_reason in {"SL_HIT", "TRAILING_SL", "STOP_LOSS", "LOSS"}:
            self._last_sl_exit_time = close_ts
            self._last_sl_direction = dir_val
            self._reentry_used_today = False
            logger.info(
                f"[{self.NAME}] 🛑 SL hit at {self._last_sl_exit_time.strftime('%H:%M')} ({dir_val}) — "
                f"2-bar (10 min) cooldown starts for same direction"
            )

    async def _on_sl_hit(self, msg: Message) -> None:
        """Record SL exit time for re-entry cooldown."""
        ts = self._resolve_timestamp(msg.payload.get("timestamp") or msg.payload.get("exit_time"))
        self._last_sl_exit_time = ts or self._last_sl_exit_time
        dir_val = str(msg.payload.get("direction", "") or "").upper()
        if not dir_val:
            sym = str(msg.payload.get("option_symbol", "") or "").upper()
            if sym.endswith("CE"):
                dir_val = "BUY_CALL"
            elif sym.endswith("PE"):
                dir_val = "BUY_PUT"
        if dir_val:
            self._last_sl_direction = dir_val
        logger.debug(
            f"[{self.NAME}] SL hit recorded at {self._last_sl_exit_time.strftime('%H:%M') if self._last_sl_exit_time else 'N/A'} ({self._last_sl_direction}) — "
            f"re-entry cooldown starts"
        )

    async def _on_system_status(self, msg: Message) -> None:
        self._trading_halted = not bool(msg.payload.get("trading_enabled", True))

    async def _on_system_reset(self, msg: Message) -> None:
        """CRITICAL FIX: Clear signal state when bus resets (new backtest run)."""
        logger.info(f"[{self.NAME}] System reset - clearing signal tracking state")
        self._last_signal_time = None
        self._last_signal_direction = ""
        self._last_signal_setup_type = ""
        self._last_signal_at = None
        self._last_signal_key = ""
        self._reentry_used_today = False
        self._last_sl_exit_time = None
        self._last_sl_direction = ""
        self._trades_today_count = 0
        self._last_close_time = None
        self._today_date = ""
        self._gate_counts.clear()
        self._strategy_health = {
            k: {"skipped": 0, "running": 0, "errors": 0, "voted": 0}
            for k in self._strategy_health
        }
        self._candidate_tracker = {
            Direction.BUY_CALL.value: {"ts": None, "price": None, "votes": 0},
            Direction.BUY_PUT.value: {"ts": None, "price": None, "votes": 0},
        }
        self._active_candidates = {
            Direction.BUY_CALL.value: None,
            Direction.BUY_PUT.value: None,
        }
        self._candidate_shadow_history.clear()

    async def on_candles(self, msg: Message) -> None:
        current_ts = self._resolve_timestamp(msg.payload.get("timestamp"))
        now_str = current_ts.strftime("%H:%M")
        regime_value = msg.payload.get("regime", Regime.TRENDING.value)
        self._reset_daily_trade_state(current_ts)

        # Extended window on expiry day (Thursday) afternoon
        regime_details = msg.payload.get("regime_details", {})
        signal_permitted = bool(msg.payload.get("signal_permitted", True))
        suppression_reason = str(msg.payload.get("suppression_reason", "") or "")
        is_expiry      = regime_details.get("is_expiry", False)
        detailed_regime = regime_details.get("detailed_regime", {})
        effective_end  = MARKET_CLOSE_TIME if is_expiry else NO_NEW_SIGNAL_AFTER
        if CAS_START_TIME <= now_str <= CAS_END_TIME:
            self._record_gate(
                "sebi_cas_window",
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | SEBI Closing Auction Session active ({CAS_START_TIME}-{CAS_END_TIME})",
            )
            return
        if not (SIGNAL_START_TIME <= now_str <= effective_end):
            self._record_gate(
                "outside_signal_window",
                (
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"now={now_str} | window={SIGNAL_START_TIME}-{effective_end}"
                ),
            )
            return
        reentry_eligible = False
        if ALLOW_REENTRY and not self._reentry_used_today and self._last_sl_exit_time:
            mins_since_sl = int((current_ts - self._last_sl_exit_time).total_seconds() // 60)
            reentry_eligible = mins_since_sl >= REENTRY_COOLDOWN_M
            if reentry_eligible:
                logger.info(f"[{self.NAME}] Re-entry allowed ({mins_since_sl}m after SL)")
        try:
            signal_regime = Regime(regime_value)
        except Exception:
            signal_regime = Regime.TRENDING
        det_conf = float(
            regime_details.get(
                "det_conf",
                (regime_details.get("detailed_regime") or {}).get("confidence", 1.0),
            )
            or 0.0
        )
        if det_conf < ENTRY_MIN_DET_CONF:
            self._record_gate(
                "regime_soft_block",
                (
                    f"det_conf={det_conf:.2f} < {ENTRY_MIN_DET_CONF:.2f} | "
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')}"
                ),
            )
            return
        if ENTRY_BLOCK_RANGING and signal_regime == Regime.RANGING:
            self._record_gate(
                "regime_soft_block",
                (
                    f"ranging entry blocked | det_conf={det_conf:.2f} | "
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')}"
                ),
            )
            return

        if self._trading_halted:
            logger.warning(
                f"[{self.NAME}] Trading halted — skipping scan | "
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')}"
            )
            return

        # Position gate handled by PositionManager (allows scale-in / full runner upgrades)
        candles = msg.payload.get("candles", [])
        if not candles:
            return

        df = pd.DataFrame(candles)
        if df.empty:
            return
        required_cols = {"datetime", "open", "high", "low", "close", "volume"}
        missing_cols = sorted(required_cols - set(df.columns))
        if missing_cols:
            logger.warning(
                f"[{self.NAME}] Missing candle columns, skipping scan | "
                f"missing={missing_cols}"
            )
            return
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df = self._with_effective_volume(df)

        ltp = float(df["close"].iloc[-1])
        india_vix = float(regime_details.get("vix", regime_details.get("india_vix", msg.payload.get("vix", 18.0))) or 18.0)
        banknifty_spot = float(
            msg.payload.get("banknifty_ltp")
            or msg.payload.get("banknifty_spot")
            or msg.payload.get("bank_nifty_ltp")
            or ltp
        )
        ce_oi = msg.payload.get("ce_oi_by_strike") or msg.payload.get("call_oi_by_strike") or {}
        pe_oi = msg.payload.get("pe_oi_by_strike") or msg.payload.get("put_oi_by_strike") or {}
        microstructure_ctx = None
        try:
            ms = get_microstructure()
            ms.update_candle(ltp, banknifty_spot, india_vix)
            microstructure_ctx = ms.get_context(
                india_vix=india_vix,
                spot=ltp,
                ce_oi=ce_oi,
                pe_oi=pe_oi,
                is_expiry=bool(is_expiry),
            )
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Microstructure context unavailable: {exc}")
        hybrid_5m = self._compute_hybrid_5m_context(df)
        payload_orb_high = msg.payload.get("orb_high")
        payload_orb_low = msg.payload.get("orb_low")
        if self._orb_high is None and payload_orb_high is not None and payload_orb_low is not None:
            try:
                self._orb_high = float(payload_orb_high)
                self._orb_low = float(payload_orb_low)
                logger.info(
                    f"[{self.NAME}] ORB restored from candle payload | "
                    f"H={self._orb_high} L={self._orb_low}"
                )
            except Exception:
                self._orb_high = None
                self._orb_low = None
        has_orb = self._orb_high is not None

        # Trend persistence gate: require N consecutive same-direction candles
        persist = regime_details.get("trend_persist", 0)
        # Bypassed for testing
        # if persist < TREND_PERSIST_CANDLES and not is_expiry:
        #     logger.debug(
        #         f"[{self.NAME}] Trend persist={persist} < {TREND_PERSIST_CANDLES} — skip"
        #     )
        #     return

        # Gap bias starts as premarket context, then gets invalidated intraday
        # when the opening range rejects the gap direction.
        gap_bias = regime_details.get("gap_bias", None)
        gap_bias_note = ""
        current_close_for_gap = float(df["close"].iloc[-1])
        gap_fail_buffer = max(current_close_for_gap * 0.0005, 8.0)
        if (
            gap_bias == Direction.BUY_CALL.value
            and self._orb_low is not None
            and current_close_for_gap < float(self._orb_low) - gap_fail_buffer
        ):
            gap_bias = Direction.BUY_PUT.value
            gap_bias_note = (
                f"gap-up failed below ORB low {float(self._orb_low):.1f}; "
                f"close={current_close_for_gap:.1f}"
            )
        elif (
            gap_bias == Direction.BUY_PUT.value
            and self._orb_high is not None
            and current_close_for_gap > float(self._orb_high) + gap_fail_buffer
        ):
            gap_bias = Direction.BUY_CALL.value
            gap_bias_note = (
                f"gap-down failed above ORB high {float(self._orb_high):.1f}; "
                f"close={current_close_for_gap:.1f}"
            )
        if gap_bias_note:
            logger.info(f"[{self.NAME}] Intraday gap bias override | {gap_bias_note} | bias={gap_bias}")

        hybrid_5m_anchor = self._compute_hybrid_5m_strategy_anchor(
            df=df,
            regime_label=str(detailed_regime.get("label", signal_regime.value) if isinstance(detailed_regime, dict) else signal_regime.value),
            strategy_context={
                "regime_details": regime_details,
                "gap_bias": gap_bias,
                "india_vix": float(regime_details.get("vix", regime_details.get("india_vix", 18.0)) or 18.0),
                "hybrid_5m": hybrid_5m,
            },
        )

        session_window = self._morning_bias.get_session_window(current_ts.time())
        adaptive_gates = self._morning_bias.get_adaptive_gates(
            adx=float(regime_details.get("adx", regime_details.get("adx_val", 0.0)) or 0.0),
            regime=vote_regime_label if "vote_regime_label" in locals() else str(detailed_regime or signal_regime.value),
            session=session_window,
        )
        adaptive_backtest_disabled = self._is_backtest and not BACKTEST_ENABLE_ADAPTIVE_GATES
        adaptive_live_disabled = (not self._is_backtest) and not LIVE_ENABLE_ADAPTIVE_GATES
        adaptive_min_votes = int(adaptive_gates.min_votes or MIN_STRATEGY_VOTES)
        adaptive_loosened = bool(adaptive_gates.loosened)
        adaptive_reason = str(adaptive_gates.reason or "")
        if adaptive_backtest_disabled or adaptive_live_disabled:
            adaptive_min_votes = MIN_STRATEGY_VOTES
            adaptive_loosened = False
            disabled_reason = (
                "backtest_adaptive_disabled"
                if adaptive_backtest_disabled else "live_adaptive_disabled"
            )
            adaptive_reason = f"{adaptive_reason}; {disabled_reason}".strip("; ")

        #  MTF Context (read-only — used for logging and confidence boost only) 
        # RC1 FIX: Removed broken MTF gate that operated on undefined 'eligible'
        # variable (eligible is built in Step 2 below, not here).
        # The gate also had identical code for BULLISH/BEARISH — no direction filtering.
        # MTF is now used ONLY for a confidence boost after votes are counted (below).
        mtf        = regime_details.get("mtf", {})
        mtf_bias   = mtf.get("bias", "NEUTRAL")
        mtf_conf   = float(mtf.get("confidence", 0.5))

        #  Step 1: Build shared indicator cache ONCE 
        t_cache = time.perf_counter()
        cache = IndicatorCache(df)
        cache_ms = round((time.perf_counter() - t_cache) * 1000, 1)
        regime_label_for_log = _regime_weight_label(detailed_regime, signal_regime.value)
        if not signal_permitted:
            self._record_gate(
                "regime_soft_block",
                (
                    f"regime={regime_value} | reason={suppression_reason or 'context only'} | "
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')}"
                ),
            )
        setup = self._setup_engine.evaluate(
            df=df,
            cache=cache,
            regime_details=regime_details,
        )
        setup_missing = setup is None
        if setup is None:
            self._record_gate(
                "no_setup",
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | regime={regime_value}",
            )
            logger.info(
                f"[{self.NAME}] Setup engine found no explicit setup | "
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                f"regime={regime_label_for_log} | "
                f"candles={len(df)}"
            )
            last_close = float(df["close"].iloc[-1])
            setup = TradeSetup(
                setup_type="none",
                direction=Direction.NONE.value,
                setup_strength=0.0,
                entry_zone_low=last_close,
                entry_zone_high=last_close,
                stop_loss_level=last_close,
                expected_move=0.0,
                context={},
            )

        #  Step 2: Filter eligible strategies 
        self._scan_count += 1
        eligible, skipped = self._get_eligible(df, has_orb)
        expected_names = {meta.name for meta in STRATEGY_REGISTRY}
        eligible_names = [meta.name for meta in eligible]
        skipped_names = [item["name"] for item in skipped]
        accounted_names = set(eligible_names) | set(skipped_names)
        missing_names = sorted(expected_names - accounted_names)
        if missing_names:
            logger.error(
                f"[{self.NAME}] Strategy accounting mismatch | "
                f"missing={missing_names} | eligible={eligible_names} | skipped={skipped_names}"
            )
        if self._scan_count == 1 or skipped or self._scan_count % self._STRATEGY_HEALTH_LOG_EVERY == 0:
            skipped_detail = [
                f"{item['name']}:{item['reason']}"
                for item in skipped
            ]
            opt_cols = [c for c in df.columns if str(c).startswith("opt_")]
            opt_total = 0.0
            if "opt_total_volume" in df.columns and len(df):
                opt_total = float(
                    pd.to_numeric(df["opt_total_volume"], errors="coerce")
                    .fillna(0.0)
                    .iloc[-1]
                )
            effective_volume = 0.0
            if "volume" in df.columns and len(df):
                effective_volume = float(
                    pd.to_numeric(df["volume"], errors="coerce")
                    .fillna(0.0)
                    .iloc[-1]
                )
            logger.info(
                f"[{self.NAME}] Strategy eligibility | "
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                f"eligible={len(eligible)}/{len(STRATEGY_REGISTRY)} | "
                f"eligible_names={eligible_names} | skipped={skipped_detail} | "
                f"opt_cols={len(opt_cols)} | opt_vol={opt_total:.0f} | "
                f"effective_vol={effective_volume:.0f}"
            )
        # logger.info(
        #     f"[{self.NAME}] Parallel strategy dispatch | total={len(STRATEGY_REGISTRY)} | "
        #     f"eligible={len(eligible)} | skipped={len(skipped)} | "
        #     f"workers={self._pool._max_workers} | "
        #     f"eligible_names={eligible_names} | skipped_names={skipped_names}"
        # )
        for s in skipped:
            _log_strategy_event(
                name=s["name"],
                status="skipped",
                df=df,
                required_candles=s["required_candles"],
                reason=s["reason"],
            )
            health = self._strategy_health.get(s["name"])
            if health is not None:
                health["skipped"] += 1
        if not eligible:
            self._log_strategy_health_summary()
            return
        #  Step 3: Run strategies. Live uses parallel execution; backtests run
        # sequentially to avoid long-run ThreadPool/pandas-ta deadlocks.
        strategy_context = {
            "regime_details": regime_details,
            "gap_bias": gap_bias,
            "india_vix": float(regime_details.get("vix", regime_details.get("india_vix", 18.0)) or 18.0),
            "hybrid_5m": hybrid_5m,
            "hybrid_5m_anchor": hybrid_5m_anchor,
            "market_ts": current_ts,
            "symbol": "NIFTY",
            "instrument": "NIFTY",
        }
        flow_signal = None
        try:
            flow_signal = get_flow_detector().analyze(
                spot=ltp,
                broker=None,
                df=df,
                india_vix=float(strategy_context["india_vix"]),
            )
            strategy_context["options_flow"] = flow_signal.to_dict()
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Options flow context unavailable: {exc}")
        t_strat = time.perf_counter()
        if self._is_backtest:
            results = []
            for meta in eligible:
                results.append(_run_strategy_sync(meta, df, cache, self._orb_high, self._orb_low, strategy_context))
        else:
            loop    = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(
                    self._pool,
                    _run_strategy_sync,
                    meta, df, cache, self._orb_high, self._orb_low, strategy_context,
                )
                for meta in eligible
            ]
            results   = await asyncio.gather(*futures)
        strat_ms  = round((time.perf_counter() - t_strat) * 1000, 1)
        for result in results:
            name = str(result.get("name", "") or "")
            health = self._strategy_health.get(name)
            if health is None:
                continue
            if result.get("_error"):
                health["errors"] += 1
            if result.get("_skipped", False):
                health["skipped"] += 1
            else:
                health["running"] += 1
        # Log errors
        for r in results:
            if r.get("_error"):
                logger.warning(f"[{self.NAME}] {r['name']}: {r['_error']}")

        pf_trusted: list[str] = []
        use_adaptive_edge_filters = (
            LIVE_ENABLE_ADAPTIVE_EDGE_FILTERS
            if not self._is_backtest
            else os.getenv(
                "BACKTEST_ENABLE_ADAPTIVE_EDGE_FILTERS",
                "false",
            ).strip().lower() in {"1", "true", "yes", "on"}
        )
        if use_adaptive_edge_filters:
            pf_gate = get_pf_gate()
            suppressed_strategies = set(pf_gate.get_suppressed_strategies())
            if suppressed_strategies:
                before = len(results)
                results = [
                    r for r in results
                    if str(r.get("name", "") or "") not in suppressed_strategies
                ]
                if len(results) != before:
                    logger.info(
                        f"[{self.NAME}] PF gate suppressed votes | "
                        f"strategies={sorted(suppressed_strategies)}"
                    )
            for result in results:
                if result.get("direction") == Direction.NONE:
                    continue
                score = pf_gate.evaluate(str(result.get("name", "") or ""), MIN_STRATEGY_VOTES)
                if score.gate_action == "TRUST_MORE":
                    original_conf = float(result.get("confidence", 0.0) or 0.0)
                    result["confidence"] = round(min(RUNNER_CONF_MAX, original_conf + score.conf_boost), 4)
                    result.setdefault("meta", {})["profit_factor_gate"] = {
                        "profit_factor": score.profit_factor,
                        "win_rate": score.win_rate,
                        "sample_size": score.sample_size,
                        "action": score.gate_action,
                        "conf_boost": score.conf_boost,
                    }
                    adaptive_min_votes = min(adaptive_min_votes, int(score.vote_threshold))
                    pf_trusted.append(str(result.get("name", "") or ""))
            if pf_trusted:
                adaptive_reason = f"{adaptive_reason}; pf_trust={','.join(sorted(pf_trusted))}".strip("; ")

        if flow_signal is not None and getattr(flow_signal, "direction", "NEUTRAL") != "NEUTRAL":
            for result in results:
                if result.get("direction") == Direction.NONE:
                    continue
                original_conf = float(result.get("confidence", 0.0) or 0.0)
                if result.get("direction") == getattr(flow_signal, "direction", ""):
                    adjusted = min(RUNNER_CONF_MAX, original_conf + float(flow_signal.confidence_boost or 0.0))
                    result["confidence"] = round(adjusted, 4)
                    result.setdefault("meta", {})["options_flow"] = flow_signal.to_dict()
                else:
                    adjusted = max(0.0, original_conf - float(flow_signal.confidence_boost or 0.0) * 0.5)
                    result["confidence"] = round(adjusted, 4)
                    result.setdefault("meta", {})["options_flow_contra"] = flow_signal.to_dict()

        # Ensemble vote
        vote_regime_label = _regime_weight_label(detailed_regime, signal_regime.value)
        wyckoff_dict = regime_details.get("wyckoff", {}) or {}
        raw_call_votes = self._apply_wyckoff_to_candidates(
            [r for r in results if r["direction"] == Direction.BUY_CALL],
            Direction.BUY_CALL,
            wyckoff_dict,
            current_ts,
        )
        raw_put_votes = self._apply_wyckoff_to_candidates(
            [r for r in results if r["direction"] == Direction.BUY_PUT],
            Direction.BUY_PUT,
            wyckoff_dict,
            current_ts,
        )
        call_votes = self._annotate_votes(
            raw_call_votes,
            vote_regime_label,
        )
        put_votes = self._annotate_votes(
            raw_put_votes,
            vote_regime_label,
        )
        call_summary = self._summarize_votes(call_votes)
        put_summary = self._summarize_votes(put_votes)
        call_score = round(sum(float(r["confidence"]) for r in call_votes), 4)
        put_score = round(sum(float(r["confidence"]) for r in put_votes), 4)

        # Process Early Candidate Shadow Tracking
        candle_high = float(df["high"].iloc[-1]) if len(df) else ltp
        candle_low = float(df["low"].iloc[-1]) if len(df) else ltp
        self._process_early_candidate_cycle(
            current_ts=current_ts,
            ltp=ltp,
            candle_high=candle_high,
            candle_low=candle_low,
            call_votes=call_votes,
            put_votes=put_votes,
            call_summary=call_summary,
            put_summary=put_summary,
        )

        # Update candidate detection tracker for timing telemetry
        if call_votes:
            if self._candidate_tracker[Direction.BUY_CALL.value]["ts"] is None:
                self._candidate_tracker[Direction.BUY_CALL.value] = {
                    "ts": current_ts,
                    "price": ltp,
                    "votes": len(call_votes),
                }
        else:
            self._candidate_tracker[Direction.BUY_CALL.value] = {
                "ts": None,
                "price": None,
                "votes": 0,
            }

        if put_votes:
            if self._candidate_tracker[Direction.BUY_PUT.value]["ts"] is None:
                self._candidate_tracker[Direction.BUY_PUT.value] = {
                    "ts": current_ts,
                    "price": ltp,
                    "votes": len(put_votes),
                }
        else:
            self._candidate_tracker[Direction.BUY_PUT.value] = {
                "ts": None,
                "price": None,
                "votes": 0,
            }
        fired = [
            f"{r['name']}:{r['direction'].value}:{r['confidence']:.2f}"
            for r in results
            if r["direction"] != Direction.NONE
        ]
        if call_summary.weighted_score > put_summary.weighted_score:
            winning, direction = call_votes, Direction.BUY_CALL
            winning_summary, losing_summary = call_summary, put_summary
        elif put_summary.weighted_score > call_summary.weighted_score:
            winning, direction = put_votes,  Direction.BUY_PUT
            winning_summary, losing_summary = put_summary, call_summary
        elif len(call_votes) >= len(put_votes):
            winning, direction = call_votes, Direction.BUY_CALL
            winning_summary, losing_summary = call_summary, put_summary
        else:
            winning, direction = put_votes,  Direction.BUY_PUT
            winning_summary, losing_summary = put_summary, call_summary

        if use_adaptive_edge_filters:
            uncorrelated_votes, representative_names, correlation_passes = deduplicate_signals(
                [str(r["name"]) for r in winning],
                min_uncorrelated_votes=min(2, adaptive_min_votes),
            )
            if winning and len(representative_names) < len(winning):
                representative_set = set(representative_names)
                raw_winning_names = [str(r["name"]) for r in winning]
                winning = [r for r in winning if str(r["name"]) in representative_set]
                winning_summary = self._summarize_votes(winning)
                logger.info(
                    f"[{self.NAME}] Correlation dedup | "
                    f"raw={raw_winning_names} reps={representative_names} "
                    f"uncorr_votes={uncorrelated_votes}"
                )
        else:
            uncorrelated_votes = len(winning)
            representative_names = [str(r["name"]) for r in winning]
            correlation_passes = True

        # Log vote detail every candle for debugging
        all_voted = call_votes + put_votes
        if all_voted:
            for vote in all_voted:
                health = self._strategy_health.get(str(vote.get("name", "") or ""))
                if health is not None:
                    health["voted"] = int(health.get("voted", 0)) + 1
            logger.info(
                f"[{self.NAME}] [VOTES]  CALL={len(call_votes)}"
                f"{[r['name'] for r in call_votes]} | "
                f"w={call_summary.weight_total:.2f} ws={call_summary.weighted_score:.2f} | "
                f"detail={_vote_weight_debug(call_votes)} | "
                f"PUT={len(put_votes)}"
                f"{[r['name'] for r in put_votes]} | "
                f"w={put_summary.weight_total:.2f} ws={put_summary.weighted_score:.2f} | "
                f"detail={_vote_weight_debug(put_votes)} | "
                f"regime={vote_regime_label} | "
                f"eligible={len(eligible)} | "
                f"cache={cache_ms}ms strat={strat_ms}ms"
            )
        # RC4 FIX: Log every candle result at INFO so it's visible in live logs
        total_fired = len(call_votes) + len(put_votes)
        if total_fired == 0:
            log_pipeline_stage(
                self.NAME,
                "voting_aggregation",
                "filtered",
                reason="no_strategy_votes",
                market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                regime=vote_regime_label,
                eligible=len(eligible),
                setup=(setup.setup_type if setup is not None else "none"),
                setup_strength=(f"{setup.setup_strength:.2f}" if setup is not None else "0.00"),
                ltp=f"{ltp:.2f}",
            )
            logger.info(
                f"[{self.NAME}]  no votes | "
                f"eligible={len(eligible)} strats | "
                f"candles={len(candles)} | "
                f"ltp={ltp:.2f}"
            )
        elif len(winning) < adaptive_min_votes:
            self._record_gate(
                "insufficient_votes",
                (
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"call={len(call_votes)} put={len(put_votes)}"
                ),
            )
            log_pipeline_stage(
                self.NAME,
                "voting_aggregation",
                "filtered",
                reason="insufficient_votes",
                market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                regime=vote_regime_label,
                setup=(setup.setup_type if setup is not None else "none"),
                setup_strength=(f"{setup.setup_strength:.2f}" if setup is not None else "0.00"),
                call_votes=len(call_votes),
                put_votes=len(put_votes),
                call_ws=f"{call_summary.weighted_score:.2f}",
                put_ws=f"{put_summary.weighted_score:.2f}",
                min_votes=adaptive_min_votes,
                adaptive_reason=adaptive_reason,
            )
            logger.info(
                f"[{self.NAME}]  votes insufficient | "
                f"CALL={len(call_votes)}{[r['name'] for r in call_votes]} "
                f"(ws={call_summary.weighted_score:.2f}) "
                f"PUT={len(put_votes)}{[r['name'] for r in put_votes]} "
                f"(ws={put_summary.weighted_score:.2f}) | "
                f"need={adaptive_min_votes} | adaptive={adaptive_reason}"
            )
        if not winning:
            return
        setup_source = "explicit"
        setup_before_alignment = setup
        if setup_missing or setup.direction != direction.value:
            inferred_setup = self._setup_engine.infer_vote_aligned_setup(
                df=df,
                cache=cache,
                regime_details=regime_details,
                direction=direction.value,
                weighted_score=winning_summary.weighted_score,
                votes=len(winning),
            )
            if inferred_setup is not None:
                setup = inferred_setup
                setup_source = "vote_aligned"
                logger.info(
                    f"[{self.NAME}] Setup aligned to votes | "
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"source={setup_source} | dir={direction.value} | "
                    f"strength={setup.setup_strength:.2f}"
                )
            else:
                self._record_gate(
                    "vote_aligned_inference_block",
                    (
                        f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                        f"dir={direction.value} | votes={len(winning)} | "
                        f"ws={winning_summary.weighted_score:.2f} | "
                        f"setup_before={setup_before_alignment.setup_type}:{setup_before_alignment.direction}"
                    ),
                )
                logger.info(
                    f"[{self.NAME}] Vote-aligned setup inference blocked | "
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"dir={direction.value} | votes={len(winning)} | "
                    f"ws={winning_summary.weighted_score:.2f}"
                )

        if setup_missing and setup.setup_type == "none":
            log_pipeline_stage(
                self.NAME,
                "strategy_signal_generation",
                "filtered",
                reason="no_valid_setup",
                market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                regime=regime_value,
                candles=len(df),
                signal_permitted=signal_permitted,
            )
            logger.info(
                f"[{self.NAME}] [BLOCK] no tradable setup after vote alignment | "
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                f"vote={direction.value}"
            )
            return
        effective_min_setup_strength = MIN_SETUP_STRENGTH
        if setup_source == "explicit" and setup.setup_type in {"breakout", "trend_pullback"}:
            effective_min_setup_strength = max(0.42, MIN_SETUP_STRENGTH - 0.02)
        if SETUP_GATE_ENABLED and float(setup.setup_strength) < effective_min_setup_strength:
            self._record_gate(
                "weak_setup",
                (
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"setup={setup.setup_type} | strength={setup.setup_strength:.2f} "
                    f"< min={effective_min_setup_strength:.2f}"
                ),
            )
            log_pipeline_stage(
                self.NAME,
                "strategy_signal_generation",
                "filtered",
                reason="setup_strength_below_min",
                market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                regime=regime_value,
                setup=setup.setup_type,
                setup_strength=f"{setup.setup_strength:.2f}",
                min_setup_strength=f"{effective_min_setup_strength:.2f}",
                signal_permitted=signal_permitted,
            )
            logger.info(
                f"[{self.NAME}] Setup below minimum | "
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                f"setup={setup.setup_type} | strength={setup.setup_strength:.2f} | "
                f"min={effective_min_setup_strength:.2f}"
            )
            return
        logger.info(
            f"[{self.NAME}] Setup | type={setup.setup_type} | dir={setup.direction} | "
            f"strength={setup.setup_strength:.2f} | move={setup.expected_move:.1f} | "
            f"source={setup_source}"
        )
        if setup.direction != direction.value:
            self._record_gate(
                "setup_mismatch",
                (
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"setup={setup.setup_type}:{setup.direction} | vote={direction.value}"
                ),
            )
            log_pipeline_stage(
                self.NAME,
                "voting_aggregation",
                "filtered",
                reason="setup_vote_mismatch",
                market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                setup=(
                    f"{setup_before_alignment.setup_type}:{setup_before_alignment.direction}"
                    if setup_before_alignment is not None else "none"
                ),
                vote=direction.value,
                setup_strength=f"{setup.setup_strength:.2f}",
            )
            logger.info(
                f"[{self.NAME}] Setup mismatch | "
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                f"setup={setup.setup_type}:{setup.direction} | vote={direction.value}"
            )
            return

        hybrid_ok, hybrid_reason = self._passes_hybrid_5m_gate(
            hybrid_5m=hybrid_5m,
            hybrid_5m_anchor=hybrid_5m_anchor,
            direction=direction,
            setup_strength=float(setup.setup_strength),
            votes=len(winning),
            weighted_score=winning_summary.weighted_score,
        )
        if not hybrid_ok:
            self._record_gate(
                "hybrid_5m_conflict",
                (
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"{hybrid_reason}"
                ),
            )
            log_pipeline_stage(
                self.NAME,
                "strategy_signal_generation",
                "filtered",
                reason="hybrid_5m_conflict",
                market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                direction=direction.value,
                setup=setup.setup_type,
                setup_strength=f"{setup.setup_strength:.2f}",
                votes=len(winning),
                weighted_score=f"{winning_summary.weighted_score:.2f}",
                hybrid_5m=hybrid_5m,
            )
            logger.info(
                f"[{self.NAME}] Hybrid 5m blocked | "
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                f"{hybrid_reason}"
            )
            return

        best_conf  = max(r["confidence"] for r in winning)
        early_trigger = False
        adaptive_single_vote = (
            adaptive_min_votes < MIN_STRATEGY_VOTES
            and len(winning) >= adaptive_min_votes
        )
        if len(winning) < adaptive_min_votes:
            if not ALLOW_SUBMIN_VOTE_EARLY_TRIGGER:
                return
            early_trigger = self._should_allow_early_trigger(
                votes=len(winning),
                best_conf=best_conf,
                weighted_score=winning_summary.weighted_score,
                setup=setup,
            )
            if not early_trigger:
                preview = self._build_trade_preview(
                    df=df,
                    ltp=ltp,
                    direction=direction if winning else Direction.NONE,
                    current_ts=current_ts,
                    msg=msg,
                )
                logger.debug(
                    f"[{self.NAME}] No consensus | market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"CALL={len(call_votes)} PUT={len(put_votes)} | "
                    f"call_score={call_score:.2f}/{call_summary.weighted_score:.2f} | "
                    f"put_score={put_score:.2f}/{put_summary.weighted_score:.2f} | "
                    f"fired={fired or ['none']} | {preview}"
                )
                if call_votes or put_votes:
                    logger.debug(
                        f"[{self.NAME}] Insufficient votes | "
                        f"CALL={len(call_votes)} [{[r['name'] for r in call_votes]}] "
                        f"ws={call_summary.weighted_score:.2f} | "
                        f"PUT={len(put_votes)} [{[r['name'] for r in put_votes]}] "
                        f"ws={put_summary.weighted_score:.2f} | "
                        f"need={adaptive_min_votes} | adaptive={adaptive_reason}"
                    )
                return
            logger.info(
                f"[{self.NAME}] Early trigger enabled | "
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                f"vote_count={len(winning)} | conf={best_conf:.2f} | "
                f"ws={winning_summary.weighted_score:.2f} | "
                f"setup={setup.setup_type}:{setup.setup_strength:.2f}"
            )

        vote_bonus = (max(len(winning), 2) - 2) * RUNNER_VOTE_BONUS
        if gap_bias == direction.value:
            vote_bonus += RUNNER_GAP_BIAS_BONUS
        setup_bonus = min(RUNNER_SETUP_BONUS_MAX, max(0.0, setup.setup_strength - RUNNER_SETUP_BONUS_THRESHOLD) * RUNNER_SETUP_BONUS_MULT)
        weight_bonus = min(RUNNER_WEIGHT_BONUS_MAX, max(0.0, winning_summary.avg_weight - RUNNER_WEIGHT_BONUS_THRESHOLD) * RUNNER_WEIGHT_BONUS_MULT)
        separation_bonus = min(
            RUNNER_SEPARATION_BONUS_MAX,
            max(0.0, winning_summary.weighted_score - losing_summary.weighted_score) * RUNNER_SEPARATION_BONUS_MULT,
        )
        vote_bonus += setup_bonus + weight_bonus + separation_bonus
        if adaptive_single_vote:
            early_trigger = True
        if early_trigger:
            vote_bonus += RUNNER_EARLY_TRIGGER_BONUS_CANDLE
        micro_conf_adj = (
            float(getattr(microstructure_ctx, "total_conf_adj", 0.0) or 0.0)
            if use_adaptive_edge_filters
            else 0.0
        )
        confidence = round(min(RUNNER_CONF_MAX, max(0.0, best_conf + vote_bonus + micro_conf_adj)), 4)

        signal_vote_count = max(len(winning), hero_zero_internal_votes if hero_zero_standalone else 0)
        winning_strat_names = [r["name"] for r in winning]
        indep_cat_count, cat_breakdown, distinct_cats = self._get_category_votes(winning_strat_names)

        cand_info = self._candidate_tracker.get(direction.value, {})
        timing_telemetry, entry_quality, timing_class = self._build_timing_and_quality_telemetry(
            df=df,
            ltp=ltp,
            direction=direction,
            current_ts=current_ts,
            cand_info=cand_info,
            setup=setup,
            hybrid_5m=hybrid_5m,
        )
        late_entry_eval = self._evaluate_late_entry_gate(
            entry_quality=entry_quality,
            setup_type=setup.setup_type,
            setup_strength=float(setup.setup_strength or 0.0),
            indep_cat_count=indep_cat_count,
            confidence=confidence,
        )

        remaining_opportunity = self._evaluate_remaining_opportunity(
            df=df,
            ltp=ltp,
            direction=direction,
            atr_val=float(entry_quality.get("atr", 20.0)),
            setup=setup,
            hybrid_5m=hybrid_5m,
        )

        current_decision = "SKIP" if late_entry_eval.get("is_rejected") else "TRADE"
        four_pillar_framework = self._evaluate_four_pillar_framework(
            direction=direction,
            votes=signal_vote_count,
            indep_cat_count=indep_cat_count,
            weighted_score=winning_summary.weighted_score,
            confidence=confidence,
            entry_quality=entry_quality,
            late_entry_eval=late_entry_eval,
            remaining_opportunity=remaining_opportunity,
            current_system_decision=current_decision,
        )

        # Additive Shadow High-Quality Entry Gate
        high_quality_gate = self._evaluate_high_quality_entry_gate(
            ml_confidence=confidence,
            ml_rank_score=float(entry_quality.get("ml_rank_score", 0.0) or 0.0),
            timing_classification=timing_class,
            raw_strategy_vote_count=signal_vote_count,
            independent_category_count=indep_cat_count,
            live_decision=current_decision,
        )

        # Additive Shadow Quality Classification (Architecture B)
        quality_classification = self._evaluate_shadow_quality_classification(
            independent_category_count=indep_cat_count,
            raw_strategy_vote_count=signal_vote_count,
            timing_state=timing_class,
            ml_state=high_quality_gate.get("shadow_ml_state"),
            live_decision=current_decision,
        )

        signal_id_str = f"{current_ts.strftime('%Y%m%d_%H%M%S')}|{direction.value}|{ltp:.2f}"

        # Phase 5A / 5B: Live Shadow Pullback & Option Contract Tracking (Fail-Open)
        if getattr(self, "_live_shadow_tracker", None) is not None and not self._is_backtest_mode:
            try:
                sig_payload = {
                    "signal_id": signal_id_str,
                    "symbol": "NIFTY",
                    "direction": direction.value,
                    "nifty_ltp": ltp,
                    "ema20": float(df.iloc[-1].get("ema20", ltp - 15.4 if direction == Direction.BUY_CALL else ltp + 15.4)),
                    "atr": float(df.iloc[-1].get("atr", 25.0)),
                    "quality_classification": quality_classification.get("quality_classification"),
                    "votes": signal_vote_count,
                    "independent_category_count": indep_cat_count,
                    "ml_state": high_quality_gate.get("shadow_ml_state"),
                }
                self._live_shadow_tracker.on_live_signal(sig_payload, current_ts)
                if not df.empty:
                    last_row = df.iloc[-1]
                    self._live_shadow_tracker.on_market_candle(
                        candle_open=float(last_row.get("open", ltp)),
                        candle_high=float(last_row.get("high", ltp)),
                        candle_low=float(last_row.get("low", ltp)),
                        candle_close=float(last_row.get("close", ltp)),
                        current_ema20=float(last_row.get("ema20", ltp)),
                        current_atr=float(last_row.get("atr", 25.0)),
                        current_ts=current_ts,
                    )
            except Exception as _shadow_err:
                logger.error(f"[{self.NAME}] Live shadow option tracking error (fail-open): {_shadow_err}")

        early_candidate_shadow = self._confirm_early_candidate(
            direction=direction,
            signal_id=signal_id_str,
            confirmed_price=ltp,
            confirmed_ts=current_ts,
        )

        signal = RawSignal(
            symbol           = "NIFTY",
            direction        = direction,
            confidence       = confidence,
            votes            = signal_vote_count,
            strategies_fired = [r["name"] for r in winning],
            nifty_ltp        = ltp,
            timestamp        = current_ts,
            regime           = signal_regime,
            metadata         = {
                **{r["name"]: r.get("meta", {}) for r in winning},
                "_context": {
                    **self._market_context(df),
                    "india_vix": india_vix,
                    "vix": india_vix,
                    "gap_bias": gap_bias or "",
                    "mtf": regime_details.get("mtf", {}),
                    "detailed_regime": detailed_regime,
                    "chop_index": float(detailed_regime.get("chop", 0.0) or regime_details.get("chop_index", 0.0) or 0.0),
                    "wyckoff": wyckoff_dict,
                    "wyckoff_size_multiplier": float(wyckoff_dict.get("wyckoff_size_mult", 1.0) or 1.0),
                    "setup": setup.to_dict(),
                    "market_structure": setup.context.get("market_structure", {}),
                    "hybrid_5m": hybrid_5m,
                    "hybrid_5m_anchor": hybrid_5m_anchor,
                    "microstructure": (
                        microstructure_ctx.to_dict()
                        if use_adaptive_edge_filters and microstructure_ctx
                        else {}
                    ),
                    "early_trigger": bool(early_trigger),
                    "adaptive_gates": {
                        "min_votes": adaptive_min_votes,
                        "default_min_votes": MIN_STRATEGY_VOTES,
                        "ml_min_confidence": adaptive_gates.ml_min_confidence,
                        "entry_min_det_conf": adaptive_gates.entry_min_det_conf,
                        "adx_threshold": adaptive_gates.adx_threshold,
                        "loosened": adaptive_loosened,
                        "session": session_window.label,
                        "size_mult": session_window.size_mult,
                        "reason": adaptive_reason,
                        "backtest_disabled": adaptive_backtest_disabled,
                    },
                    "vote_regime": vote_regime_label,
                    "weighted_vote": {
                        "call": {
                            "count": call_summary.count,
                            "weight_total": call_summary.weight_total,
                            "weighted_score": call_summary.weighted_score,
                        },
                        "put": {
                            "count": put_summary.count,
                            "weight_total": put_summary.weight_total,
                            "weighted_score": put_summary.weighted_score,
                        },
                        "winner_avg_weight": winning_summary.avg_weight,
                    },
                    "correlation_dedup": {
                        "uncorrelated_votes": uncorrelated_votes,
                        "representatives": representative_names,
                        "passes": correlation_passes,
                    },
                    "independent_category_count": indep_cat_count,
                    "category_votes": {
                        "independent_category_count": indep_cat_count,
                        "categories": distinct_cats,
                        "category_breakdown": cat_breakdown,
                    },
                    "timing_telemetry": timing_telemetry,
                    "entry_quality": entry_quality,
                    "timing_classification": timing_class,
                    "late_entry_gate": late_entry_eval,
                    "shadow_late_entry_gate": late_entry_eval,
                    "early_candidate_shadow": early_candidate_shadow,
                    "remaining_opportunity": remaining_opportunity,
                    "remaining_opportunity_class": remaining_opportunity["remaining_opportunity_class"],
                    "estimated_remaining_rr": remaining_opportunity["estimated_remaining_rr"],
                    "nearest_opposing_level": remaining_opportunity["nearest_opposing_level"],
                    "shadow_opportunity_decision": remaining_opportunity["shadow_opportunity_decision"],
                    "four_pillar_framework": four_pillar_framework,
                    "shadow_framework_decision": four_pillar_framework["shadow_framework_decision"],
                    "shadow_framework_reasons": four_pillar_framework["shadow_framework_reasons"],
                    "disagreement_reason": four_pillar_framework["disagreement_reason"],
                    "shadow_high_quality_entry_gate": high_quality_gate,
                    "shadow_high_quality_entry_gate_state": high_quality_gate["shadow_high_quality_entry_gate_state"],
                    "shadow_high_quality_entry_gate_reasons": high_quality_gate["shadow_high_quality_entry_gate_reasons"],
                    "shadow_ml_state": high_quality_gate["shadow_ml_state"],
                    "shadow_timing_state": high_quality_gate["shadow_timing_state"],
                    "shadow_raw_vote_count": high_quality_gate["shadow_raw_vote_count"],
                    "shadow_independent_category_count": high_quality_gate["shadow_independent_category_count"],
                    "shadow_decision": high_quality_gate["shadow_decision"],
                    "shadow_joint_rule_decision": high_quality_gate["shadow_joint_rule_decision"],
                    "quality_classification": quality_classification["quality_classification"],
                    "quality_classification_reasons": quality_classification["quality_classification_reasons"],
                    "quality_tier": quality_classification["quality_tier"],
                },
            },
        )

        #  Signal Quality Score (safe — scorer may not be available) 
        signal_dict = signal.to_dict()
        signal_dict["india_vix"] = india_vix
        signal_dict["independent_category_count"] = indep_cat_count
        signal_dict["category_votes"] = {
            "independent_category_count": indep_cat_count,
            "categories": distinct_cats,
            "category_breakdown": cat_breakdown,
        }
        signal_dict["timing_telemetry"] = timing_telemetry
        signal_dict["entry_quality"] = entry_quality
        signal_dict["timing_classification"] = timing_class
        signal_dict["late_entry_gate"] = late_entry_eval
        signal_dict["shadow_late_entry_gate"] = late_entry_eval
        signal_dict["entry_decision"] = late_entry_eval["entry_decision"]
        signal_dict["shadow_entry_decision"] = late_entry_eval["shadow_entry_decision"]
        signal_dict["shadow_rejection_reasons"] = late_entry_eval["shadow_rejection_reasons"]
        signal_dict["rejection_reasons"] = late_entry_eval["rejection_reasons"]
        signal_dict["would_have_blocked"] = late_entry_eval["would_have_blocked"]
        signal_dict["entry_quality_classification"] = late_entry_eval["entry_quality_classification"]
        signal_dict["early_candidate_shadow"] = early_candidate_shadow
        signal_dict["remaining_opportunity"] = remaining_opportunity
        signal_dict["remaining_opportunity_class"] = remaining_opportunity["remaining_opportunity_class"]
        signal_dict["estimated_remaining_rr"] = remaining_opportunity["estimated_remaining_rr"]
        signal_dict["nearest_opposing_level"] = remaining_opportunity["nearest_opposing_level"]
        signal_dict["shadow_opportunity_decision"] = remaining_opportunity["shadow_opportunity_decision"]
        signal_dict["four_pillar_framework"] = four_pillar_framework
        signal_dict["shadow_framework_decision"] = four_pillar_framework["shadow_framework_decision"]
        signal_dict["shadow_framework_reasons"] = four_pillar_framework["shadow_framework_reasons"]
        signal_dict["disagreement_reason"] = four_pillar_framework["disagreement_reason"]
        signal_dict["shadow_high_quality_entry_gate"] = high_quality_gate
        signal_dict["shadow_high_quality_entry_gate_state"] = high_quality_gate["shadow_high_quality_entry_gate_state"]
        signal_dict["shadow_high_quality_entry_gate_reasons"] = high_quality_gate["shadow_high_quality_entry_gate_reasons"]
        signal_dict["shadow_ml_state"] = high_quality_gate["shadow_ml_state"]
        signal_dict["shadow_timing_state"] = high_quality_gate["shadow_timing_state"]
        signal_dict["shadow_decision"] = high_quality_gate["shadow_decision"]
        signal_dict["shadow_joint_rule_decision"] = high_quality_gate["shadow_joint_rule_decision"]
        signal_dict["quality_classification"] = quality_classification["quality_classification"]
        signal_dict["quality_classification_reasons"] = quality_classification["quality_classification_reasons"]
        signal_dict["quality_tier"] = quality_classification["quality_tier"]

        # ── MULTI-MODEL SHADOW TELEMETRY (PNL_MAXIMIZER vs LEGACY vs ROLLING) ──
        strat_names_set = set(signal.strategies_fired)
        now_tod = current_ts.hour + current_ts.minute / 60.0
        anchors_set = {"VolumeProfile", "RangeSpread", "FVG", "ElliottWave", "ORB", "CPR", "ValueArea"}
        has_anchor_lead = bool(strat_names_set.intersection(anchors_set))

        pnl_max_pass = True
        pnl_max_reason = "PASS"
        if 10.0 <= now_tod < 11.5 and not has_anchor_lead and len(winning) < 5:
            pnl_max_pass = False
            pnl_max_reason = f"morning_trap_low_votes: {len(winning)} < 5"
        elif not has_anchor_lead and len(winning) < 5:
            pnl_max_pass = False
            pnl_max_reason = "missing_structural_anchor"
        elif "ADX+PSAR" in strat_names_set and not has_anchor_lead:
            pnl_max_pass = False
            pnl_max_reason = "toxic_pair_no_anchor"

        signal_dict["shadow_models"] = {
            "pnl_maximizer_v1_decision": "PASS" if pnl_max_pass else "REJECT",
            "pnl_maximizer_v1_reason": pnl_max_reason,
            "legacy_decision": "PASS",
            "rolling_decision": "PASS",
            "total_votes": len(winning),
            "strategies_fired": list(signal.strategies_fired),
            "anchors_present": list(strat_names_set.intersection(anchors_set)),
            "market_regime": getattr(signal_regime, "value", str(signal_regime)),
        }
        if self._scorer is not None:
            quality = self._scorer.score(
                strategies_fired=signal.strategies_fired,
                direction=direction.value,
                confidence=confidence,
                timestamp=df.index[-1],
                option_premium=100.0,
            )
            signal_dict["quality_score"] = quality.total
            signal_dict["quality_categories"] = quality.categories_hit
            signal_dict["quality_reason"] = quality.reason
            quality_str = f"quality={quality.total:.2f} | cats={quality.categories_hit} | "
            
            # Block low-quality signals to improve win rate
            if quality.total < ML_QUALITY_MIN_SCORE and len(winning) <= 3:
                self._record_gate(
                    "low_quality_filter",
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | quality={quality.total:.2f} | votes={len(winning)}"
                )
                logger.info(
                    f"[{self.NAME}] Low quality blocked | quality={quality.total:.2f} | votes={len(winning)}"
                )
                return
        else:
            quality_str = ""
        conflict = get_resolver().check_signal(
            direction=direction.value,
            strategies=[r["name"] for r in winning],
            candle_time=current_ts.strftime("%H:%M"),
            current_time=current_ts,
        )
        if conflict.blocked:
            self._record_gate(
                "conflict_resolver_block",
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | {conflict.block_reason}",
            )
            logger.info(f"[{self.NAME}] Conflict resolver blocked signal | {conflict.block_reason}")
            return
        if conflict.size_multiplier < 1.0:
            confidence = round(confidence * conflict.size_multiplier, 4)
            signal.confidence = confidence
            signal_dict["confidence"] = confidence
            signal_dict["conflict_size_multiplier"] = conflict.size_multiplier
            signal_dict["conflict_warnings"] = conflict.warnings
        #  PATTERN FIX: Block weak 2-vote signals with low setup strength 
        # DISABLED: Was blocking too many trades, causing drastic regression
        # if len(winning) <= 2 and float(setup.setup_strength) < 0.65:
        #     return

        #  MORNING CHOP BLOCK: Reject signals before 10:30 with low volatility
        signal_minutes = current_ts.hour * 60 + current_ts.minute
        if signal_minutes < 10 * 60 + 30:  # Before 10:30
            # Check if ATR is low (choppy morning)
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 30.0
            atr_avg = df['atr'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 and 'atr' in df.columns else atr
            if atr < atr_avg * 0.7:  # Low volatility = chop
                self._record_gate(
                    "morning_chop_block",
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | atr={atr:.2f} | atr_avg={atr_avg:.2f} | ratio={atr/atr_avg:.2f}"
                )
                logger.info(
                    f"[{self.NAME}] Morning chop blocked | atr={atr:.2f} | atr_avg={atr_avg:.2f} | ratio={atr/atr_avg:.2f}"
                )
                return

        setup_type = str(setup.setup_type or "").lower()
        strategy_names = {str(r.get("name", "") or "") for r in winning}
        institutional_confirmers = {
            "FVG",
            "OIAnalysis",
            "ORB",
            "CPR",
            "VolumeProfile",
            "RangeSpread",
        }
        has_institutional_confirmation = bool(strategy_names & institutional_confirmers)
        if strategy_names == {"StochRSI", "EMASlope"}:
            self._record_gate(
                "weak_stoch_ema_block",
                (
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"direction={direction.value}"
                ),
            )
            log_pipeline_stage(
                self.NAME,
                "strategy_signal_generation",
                "filtered",
                reason="weak_stoch_ema_block",
                market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                direction=direction.value,
                setup=setup_type,
                setup_strength=f"{float(setup.setup_strength):.2f}",
                votes=len(winning),
                weighted_score=f"{winning_summary.weighted_score:.2f}",
                strategies=",".join(sorted(strategy_names)),
            )
            logger.info(
                f"[{self.NAME}] Weak StochRSI/EMA basket blocked | "
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                f"direction={direction.value}"
            )
            return

        weak_reasons: list[str] = []
        hard_confirmers = {"FVG", "OIAnalysis", "ORB", "CPR", "VolumeProfile"}
        if (
            "OpeningRangeBias" in strategy_names
            and not strategy_names.intersection(hard_confirmers)
        ):
            weak_reasons.append("opening_range_bias_without_hard_confirmation")
        if strategy_names == {"ValueArea", "EMASlope"}:
            weak_reasons.append("thin_valuearea_ema_pair")
        if (
            direction == Direction.BUY_CALL
            and {"ADX+PSAR", "ValueArea", "StochRSI"}.issubset(strategy_names)
            and not strategy_names.intersection({"SuperTrend+RSI", "VWAP+EMA", *hard_confirmers})
        ):
            weak_reasons.append("call_adx_value_stoch_without_trend_or_flow")
        if (
            direction == Direction.BUY_CALL
            and {"ADX+PSAR", "ValueArea", "RangeSpread", "EMASlope"}.issubset(strategy_names)
            and "SuperTrend+RSI" not in strategy_names
            and not strategy_names.intersection(hard_confirmers)
        ):
            weak_reasons.append("call_adx_value_range_ema_without_trend_or_flow")
        if (
            direction == Direction.BUY_CALL
            and {"SuperTrend+RSI", "BBSqueeze", "ValueArea", "RangeSpread", "EMASlope"}.issubset(strategy_names)
            and not strategy_names.intersection(hard_confirmers)
        ):
            weak_reasons.append("call_squeeze_range_value_without_displacement_or_flow")
        if (
            setup_type == "vote_aligned"
            and direction == Direction.BUY_CALL
            and "BBSqueeze" in strategy_names
            and not strategy_names.intersection({"ValueArea", "FVG", "CPR", "OIAnalysis", "ORB", "VolumeProfile"})
        ):
            weak_reasons.append("call_squeeze_without_auction_or_flow_confirmation")
        if (
            setup_type == "vote_aligned"
            and direction == Direction.BUY_CALL
            and {"Ichimoku", "EMASlope"}.issubset(strategy_names)
            and "ValueArea" not in strategy_names
            and "ORB" not in strategy_names
        ):
            weak_reasons.append("call_ichimoku_ema_without_value_or_orb")
        if (
            setup_type == "vote_aligned"
            and direction == Direction.BUY_CALL
            and "OpeningRangeBias" in strategy_names
            and not strategy_names.intersection({"ORB", "CPR"})
        ):
            weak_reasons.append("call_opening_range_bias_without_orb_or_cpr")
        if (
            setup_type == "breakout"
            and direction == Direction.BUY_CALL
            and "SuperTrend+RSI" not in strategy_names
        ):
            weak_reasons.append("call_breakout_without_supertrend_confirmation")
        if (
            setup_type == "breakout"
            and direction == Direction.BUY_PUT
            and {"SuperTrend+RSI", "ADX+PSAR", "StochRSI"}.issubset(strategy_names)
            and not strategy_names.intersection({"EMASlope", "ValueArea", "FVG", "CPR", "OIAnalysis", "RangeSpread"})
        ):
            weak_reasons.append("put_breakout_stoch_without_trend_continuation_or_value")
        if (
            direction == Direction.BUY_PUT
            and "ExpiryWeek" in strategy_names
            and "EMASlope" in strategy_names
            and "ValueArea" not in strategy_names
        ):
            weak_reasons.append("put_expiry_ema_without_value_confirmation")
        if (
            setup_type == "vote_aligned"
            and direction == Direction.BUY_PUT
            and signal_minutes < 10 * 60
            and {"ExpiryWeek", "GapDirection"}.issubset(strategy_names)
        ):
            weak_reasons.append("early_put_expiry_gap_without_confirmation")
        if (
            setup_type == "breakout"
            and direction == Direction.BUY_PUT
            and "ValueArea" in strategy_names
            and not strategy_names.intersection({"FVG", "OIAnalysis", "ORB", "CPR"})
        ):
            weak_reasons.append("put_breakout_value_without_flow_confirmation")
        if (
            setup_type == "breakout"
            and direction == Direction.BUY_PUT
            and not strategy_names.intersection({"ADX+PSAR", "ValueArea"})
        ):
            weak_reasons.append("put_breakout_without_adx_or_value_confirmation")
        if (
            setup_type == "vote_aligned"
            and {"ADX+PSAR", "ValueArea", "EMASlope"}.issubset(strategy_names)
            and not has_institutional_confirmation
        ):
            weak_reasons.append("plain_adx_value_ema_without_institutional_confirmation")
        if (
            setup_type in {"vote_aligned", "trend_pullback"}
            and {"SuperTrend+RSI", "ADX+PSAR", "ValueArea", "EMASlope"}.issubset(strategy_names)
            and not has_institutional_confirmation
        ):
            weak_reasons.append("trend_adx_value_ema_without_institutional_confirmation")
        if (
            setup_type in {"breakout", "trend_pullback"}
            and {"SuperTrend+RSI", "BBSqueeze"}.issubset(strategy_names)
            and not {"FVG", "ValueArea", "OIAnalysis", "RangeSpread", "ORB", "CPR"}.intersection(strategy_names)
        ):
            weak_reasons.append("squeeze_without_value_or_flow_confirmation")

        if setup_type == "vote_aligned" and direction == Direction.BUY_CALL:
            if (
                {"SuperTrend+RSI", "VWAP+EMA", "EMASlope"}.issubset(strategy_names)
                and not has_institutional_confirmation
            ):
                weak_reasons.append("plain_trend_vwap_ema_without_institutional_confirmation")
            if (
                {"SuperTrend+RSI", "ValueArea", "EMASlope"}.issubset(strategy_names)
                and not has_institutional_confirmation
            ):
                weak_reasons.append("trend_value_ema_without_institutional_confirmation")
            if (
                {"SuperTrend+RSI", "ADX+PSAR", "ValueArea"}.issubset(strategy_names)
                and not has_institutional_confirmation
            ):
                weak_reasons.append("trend_adx_value_without_institutional_confirmation")
            if (
                {"BBSqueeze", "Ichimoku"}.issubset(strategy_names)
                and not {"FVG", "OIAnalysis", "CPR", "VolumeProfile"}.intersection(strategy_names)
            ):
                weak_reasons.append("squeeze_ichimoku_without_flow_confirmation")
            if "PriceAction" in strategy_names and not {"FVG", "OIAnalysis", "ORB", "CPR"}.intersection(strategy_names):
                weak_reasons.append("price_action_without_flow_or_breakout_confirmation")

        if signal_minutes >= 23 * 60:
            weak_reasons.append("late_afternoon_entry_cutoff")

        if weak_reasons:
            is_valid_exemption = (
                (
                    (len(winning) >= 5 and setup.setup_strength >= 0.65)
                    or (len(winning) >= 4 and setup.setup_strength >= 0.70)
                    or setup.setup_strength >= 0.80
                )
                and "late_afternoon_entry_cutoff" not in weak_reasons
            )
            if not is_valid_exemption:
                self._record_gate(
                    "weak_strategy_basket_block",
                    (
                        f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                        f"direction={direction.value} | reasons={','.join(weak_reasons)} | "
                        f"strategies={sorted(strategy_names)}"
                    ),
                )
                log_pipeline_stage(
                    self.NAME,
                    "strategy_signal_generation",
                    "filtered",
                    reason="weak_strategy_basket_block",
                    market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                    direction=direction.value,
                    setup=setup_type,
                    setup_strength=f"{float(setup.setup_strength):.2f}",
                    votes=len(winning),
                    weighted_score=f"{winning_summary.weighted_score:.2f}",
                    weak_reasons=",".join(weak_reasons),
                    strategies=",".join(sorted(strategy_names)),
                )
                logger.info(
                    f"[{self.NAME}] Weak strategy basket blocked | "
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"direction={direction.value} | reasons={weak_reasons} | "
                    f"strategies={sorted(strategy_names)}"
                )
                return

        if (
            setup_type == "trend_pullback"
            and signal_minutes < 10 * 60 + 30
            and len(winning) < 4
            and float(setup.setup_strength or 0.0) < 0.50
        ):
            self._record_gate(
                "trend_pullback_early_block",
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | setup={setup_type} | votes={len(winning)}"
            )
            logger.info(
                f"[{self.NAME}] Early trend_pullback blocked | market_ts={current_ts.strftime('%H:%M')} | "
                f"votes={len(winning)} | setup_strength={float(setup.setup_strength):.2f}"
            )
            return

        if (
            setup_type == "breakout"
            and signal_minutes < 9 * 60 + 35
            and len(winning) <= 1
        ):
            self._record_gate(
                "open_single_vote_breakout_block",
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | votes={len(winning)}",
            )
            logger.info(
                f"[{self.NAME}] Open breakout (single vote) blocked | market_ts={current_ts.strftime('%H:%M')} | "
                f"votes={len(winning)}"
            )
            return

        if (
            setup_type == "vote_aligned"
            and 11 * 60 <= signal_minutes < 11 * 60 + 45
            and len(winning) <= 3
            and float(setup.setup_strength) < 0.55
        ):
            self._record_gate(
                "vote_aligned_midday_block",
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | votes={len(winning)} | "
                f"setup_strength={float(setup.setup_strength):.2f}"
            )
            logger.info(
                f"[{self.NAME}] Midday vote_aligned blocked | market_ts={current_ts.strftime('%H:%M')} | "
                f"votes={len(winning)} | setup_strength={float(setup.setup_strength):.2f}"
            )
            if len(winning) >= MIN_STRATEGY_VOTES:
                await self._publish_runner_rejection(
                    current_ts=current_ts,
                    direction=direction,
                    confidence=float(confidence),
                    votes=len(winning),
                    ltp=ltp,
                    regime=signal_regime,
                    winning=winning,
                    setup=setup,
                    weighted_score=float(winning_summary.weighted_score),
                    rejection_reason=(
                        "Runner gate: midday vote_aligned blocked | "
                        f"setup_strength={float(setup.setup_strength):.2f} < 0.74"
                    ),
                    extra_meta={"runner_gate": "vote_aligned_midday_block"},
                )
            return

        if self._trades_today_count == 1 and not self._second_trade_allowed(
            current_ts=current_ts,
            regime=signal_regime,
            setup_strength=float(setup.setup_strength),
            confidence=float(confidence),
            votes=len(winning),
            weighted_score=float(winning_summary.weighted_score),
        ):
            self._record_gate(
                "second_trade_block",
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                f"closed_at={(self._last_close_time.strftime('%H:%M') if self._last_close_time else 'NA')} | "
                f"setup_strength={float(setup.setup_strength):.2f} | conf={float(confidence):.2f} | "
                f"votes={len(winning)} | ws={float(winning_summary.weighted_score):.2f}"
            )
            logger.info(
                f"[{self.NAME}] Second trade blocked | market_ts={current_ts.strftime('%H:%M')} | "
                f"closed_at={(self._last_close_time.strftime('%H:%M') if self._last_close_time else 'NA')} | "
                f"setup={float(setup.setup_strength):.2f} conf={float(confidence):.2f} "
                f"votes={len(winning)} ws={float(winning_summary.weighted_score):.2f}"
            )
            if len(winning) >= MIN_STRATEGY_VOTES:
                await self._publish_runner_rejection(
                    current_ts=current_ts,
                    direction=direction,
                    confidence=float(confidence),
                    votes=len(winning),
                    ltp=ltp,
                    regime=signal_regime,
                    winning=winning,
                    setup=setup,
                    weighted_score=float(winning_summary.weighted_score),
                    rejection_reason=(
                        "Runner gate: second trade blocked | "
                        f"closed_at={(self._last_close_time.strftime('%H:%M') if self._last_close_time else 'NA')} | "
                        f"setup={float(setup.setup_strength):.2f} conf={float(confidence):.2f} "
                        f"votes={len(winning)} ws={float(winning_summary.weighted_score):.2f}"
                    ),
                    extra_meta={"runner_gate": "second_trade_block"},
                )
            return

        #  PATTERN FIX: Allow counter-trend signals if they have 3+ votes
        market_structure = setup.context.get('market_structure', {})
        structure_state = market_structure.get('structure_state', {})
        structure_bias = structure_state.get('bias', '')

        if structure_bias in ['BEARISH', 'BULLISH']:
            counter_trend = (
                (direction == Direction.BUY_CALL and structure_bias == 'BEARISH') or
                (direction == Direction.BUY_PUT and structure_bias == 'BULLISH')
            )
            if counter_trend and len(winning) < 3:
                self._record_gate(
                    'counter_trend_block',
                    f'market_ts={current_ts.strftime("%Y-%m-%d %H:%M IST")} | dir={direction.value} | bias={structure_bias} | votes={len(winning)}'
                )
                logger.info(
                    f'[{self.NAME}] Counter-trend blocked (insufficient votes) | dir={direction.value} | '
                    f'bias={structure_bias} | votes={len(winning)}'
                )
                return

        #  Signal Cooldown: no same-direction signal within configured window 
        # RC2 FIX: Use CANDLE timestamp (df.index[-1]) not datetime.now()
        # In backtest, datetime.now() is always ~0 seconds apart — blocked every signal
        # Candle timestamp correctly reflects market time even during replay
        cooldown_minutes = DUPLICATE_SIGNAL_COOLDOWN_MINUTES
        candle_ts = df.index[-1]
        # Convert to timezone-aware if needed
        try:
            candle_now = pd.Timestamp(candle_ts).to_pydatetime()
            if candle_now.tzinfo is None:
                candle_now = IST.localize(candle_now)
        except Exception:
            candle_now = datetime.now(IST)

        current_setup_type = str(setup.setup_type or "")
        if (
            self._last_signal_direction == direction.value
            and self._last_signal_setup_type == current_setup_type
            and self._last_signal_time is not None
        ):
            elapsed_secs = (candle_now - self._last_signal_time).total_seconds()
            elapsed_mins = int(elapsed_secs // 60)
            if elapsed_mins < cooldown_minutes:
                logger.info(
                    f"[{self.NAME}] [WAIT] Cooldown: same direction/setup {direction.value}/{current_setup_type} "
                    f"fired {elapsed_mins}m ago (candle time), skipping"
                )
                return

        #  MTF confidence boost 
        if mtf_bias != "NEUTRAL" and mtf_conf >= 0.65:
            aligned = (mtf_bias == "BULLISH" and direction == Direction.BUY_CALL) or \
                      (mtf_bias == "BEARISH" and direction == Direction.BUY_PUT)
            if aligned:
                confidence = round(min(0.95, confidence + 0.04), 4)

        # Late-Entry Rejection Gate: Active filtering on strongest exhaustion conditions
        if late_entry_eval.get("is_rejected"):
            self._reject_early_candidate(
                direction=direction,
                rejection_reason=f"late_entry_move_exhausted: {late_entry_eval.get('primary_rejection_reason')}",
                current_ts=current_ts,
            )
            self._record_gate(
                "late_entry_move_exhausted",
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | {late_entry_eval.get('primary_rejection_reason')} | reasons={late_entry_eval.get('rejection_reasons')}",
            )
            log_pipeline_stage(
                self.NAME,
                "strategy_signal_generation",
                "filtered",
                reason=f"late_entry_move_exhausted: {late_entry_eval.get('primary_rejection_reason')}",
                market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                direction=direction.value,
                confidence=f"{confidence:.2f}",
                setup=setup.setup_type,
                setup_strength=f"{setup.setup_strength:.2f}",
                votes=len(winning),
                extension_atr=f"{entry_quality.get('extension_atr', 0.0):.2f}",
                entry_decision=late_entry_eval.get("entry_decision"),
                reasons=",".join(late_entry_eval.get("rejection_reasons", [])),
            )
            logger.info(
                f"[{self.NAME}] Late entry move exhausted blocked | market_ts={current_ts.strftime('%H:%M')} | "
                f"{direction.value} | {late_entry_eval.get('primary_rejection_reason')} | reasons={late_entry_eval.get('rejection_reasons')}"
            )
            await self._publish_runner_rejection(
                current_ts=current_ts,
                direction=direction,
                confidence=float(confidence),
                votes=len(winning),
                ltp=ltp,
                regime=signal_regime,
                winning=winning,
                setup=setup,
                weighted_score=float(winning_summary.weighted_score),
                rejection_reason=f"Runner gate: late entry move exhausted ({late_entry_eval.get('primary_rejection_reason')})",
            )
            return

        # ── RSI Climax Exhaustion Gate ──────────────────────────────────────────
        market_ctx = self._market_context(df) or {}
        rsi_14_val = float(market_ctx.get("rsi_14", 50.0) or 50.0)
        rsi_exhausted = False
        rsi_reason = ""
        if direction == Direction.BUY_PUT and rsi_14_val < 28.0:
            rsi_exhausted = True
            rsi_reason = f"RSI oversold exhaustion (RSI={rsi_14_val:.1f} < 28.0)"
        elif direction == Direction.BUY_CALL and rsi_14_val > 72.0:
            rsi_exhausted = True
            rsi_reason = f"RSI overbought exhaustion (RSI={rsi_14_val:.1f} > 72.0)"

        if rsi_exhausted:
            logger.info(
                f"[{self.NAME}] RSI Climax Exhaustion blocked | market_ts={current_ts.strftime('%H:%M')} | "
                f"{direction.value} | {rsi_reason}"
            )
            self._record_gate(
                "rsi_climax_exhaustion",
                f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | {rsi_reason}",
            )
            log_pipeline_stage(
                self.NAME,
                "strategy_signal_generation",
                "filtered",
                reason=f"rsi_climax_exhaustion: {rsi_reason}",
                market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
                direction=direction.value,
                confidence=f"{confidence:.2f}",
                setup=setup.setup_type,
                votes=len(winning),
                rsi=f"{rsi_14_val:.1f}",
            )
            await self._publish_runner_rejection(
                current_ts=current_ts,
                direction=direction,
                confidence=float(confidence),
                votes=len(winning),
                ltp=ltp,
                regime=signal_regime,
                winning=winning,
                setup=setup,
                weighted_score=float(winning_summary.weighted_score),
                rejection_reason=f"Runner gate: {rsi_reason}",
            )
            return

        # ── ROLLING_5M_SELECTIVE ENSEMBLE QUALITY GATE ──
        from config.settings.modules.trading import STRATEGY_CONTEXT_MODE
        if STRATEGY_CONTEXT_MODE == "ROLLING_5M_SELECTIVE":
            categories = set(STRATEGY_CATEGORY_MAPPING.get(s, "OTHER") for s in signal.strategies_fired)
            now_hhmm = current_ts.strftime("%H:%M")
            is_midday = "11:30" <= now_hhmm < "13:00"

            rejected_reason = None
            if len(categories) < 3:
                rejected_reason = f"insufficient_category_diversity: {len(categories)} < 3 ({categories})"
            elif is_midday and len(winning) < 5:
                rejected_reason = f"midday_chop_low_votes: {len(winning)} < 5 at {now_hhmm}"

            if rejected_reason:
                logger.info(
                    f"[{self.NAME}] REJECTED_BY_SELECTIVE_GATE | "
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"reason={rejected_reason} | votes={len(winning)} | "
                    f"strategies={signal.strategies_fired} | conf={confidence:.2f} | "
                    f"would_have_traded=True"
                )
                await self._publish_runner_rejection(
                    current_ts=current_ts,
                    direction=direction,
                    confidence=float(confidence),
                    votes=len(winning),
                    ltp=ltp,
                    regime=signal_regime,
                    winning=winning,
                    setup=setup,
                    weighted_score=float(winning_summary.weighted_score),
                    rejection_reason=f"REJECTED_BY_SELECTIVE_GATE: {rejected_reason}",
                    extra_meta={
                        "runner_gate": "REJECTED_BY_SELECTIVE_GATE",
                        "would_have_traded": True,
                        "categories": list(categories),
                    },
                )
                return

        # ── P&L_MAXIMIZER_V1 ENSEMBLE SELECTION GATE ──
        if STRATEGY_CONTEXT_MODE == "PNL_MAXIMIZER_V1" and not self._is_backtest_mode:
            strat_names = set(signal.strategies_fired)
            now_hhmm = current_ts.strftime("%H:%M")
            now_tod = current_ts.hour + current_ts.minute / 60.0

            anchors = {"VolumeProfile", "RangeSpread", "FVG", "ElliottWave", "ORB", "CPR", "ValueArea"}
            has_anchor = bool(strat_names.intersection(anchors))

            rejected_reason = None
            if 10.0 <= now_tod < 11.5 and not has_anchor and len(winning) < 5:
                rejected_reason = f"morning_trap_low_votes: {len(winning)} < 5 at {now_hhmm}"
            elif not has_anchor and len(winning) < 5:
                rejected_reason = "missing_structural_anchor: require >= 5 votes"
            elif "ADX+PSAR" in strat_names and not has_anchor:
                rejected_reason = "toxic_pair_no_anchor: ADX+PSAR without structural lead"

            if rejected_reason:
                logger.info(
                    f"[{self.NAME}] REJECTED_BY_PNL_MAXIMIZER | "
                    f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                    f"reason={rejected_reason} | votes={len(winning)} | "
                    f"strategies={signal.strategies_fired} | conf={confidence:.2f} | "
                    f"would_have_traded=True"
                )
                await self._publish_runner_rejection(
                    current_ts=current_ts,
                    direction=direction,
                    confidence=float(confidence),
                    votes=len(winning),
                    ltp=ltp,
                    regime=signal_regime,
                    winning=winning,
                    setup=setup,
                    weighted_score=float(winning_summary.weighted_score),
                    rejection_reason=f"REJECTED_BY_PNL_MAXIMIZER: {rejected_reason}",
                    extra_meta={
                        "runner_gate": "REJECTED_BY_PNL_MAXIMIZER",
                        "would_have_traded": True,
                        "anchors_present": list(strat_names.intersection(anchors)),
                    },
                )
                return

        logger.info(
            f"[{self.NAME}] [RAW] {direction.value} | "
            f"conf={confidence:.2f} | {quality_str}"
            f"votes={len(winning)}/{len(eligible)} | "
            f"w={winning_summary.weight_total:.2f} ws={winning_summary.weighted_score:.2f} | "
            f"{signal.strategies_fired} | "
            f"cache={cache_ms}ms strat={strat_ms}ms"
        )

        # ── 2-BAR (10 MIN) POST-SL COOLDOWN FOR SAME DIRECTION ──
        if self._last_sl_exit_time and self._last_sl_direction:
            if hasattr(self._last_sl_exit_time, "date") and self._last_sl_exit_time.date() == current_ts.date():
                mins_since_sl = (current_ts - self._last_sl_exit_time).total_seconds() / 60.0
                if 0.0 <= mins_since_sl < 10.0 and direction.value == self._last_sl_direction:
                    self._record_gate(
                        "post_sl_cooldown",
                        f"Blocked same-direction {direction.value} re-entry within 2 bars ({mins_since_sl:.1f}m < 10m) after SL",
                    )
                    logger.info(
                        f"[{self.NAME}] 🛑 [POST_SL_COOLDOWN] Suppressed {direction.value} signal "
                        f"({mins_since_sl:.1f}m since SL < 10.0m cooldown) | market_ts={current_ts.strftime('%H:%M')}"
                    )
                    return

        # CRITICAL FIX: Removed redundant _is_duplicate_signal check
        # The cooldown gate above (lines 976-988) already handles duplicate prevention
        # by tracking _last_signal_time, _last_signal_direction, _last_signal_setup_type
        if reentry_eligible:
            self._reentry_used_today = True
        self._last_signal_time = candle_now
        self._last_signal_direction = direction.value
        self._last_signal_setup_type = current_setup_type
        self._remember_signal(signal)

        logger.info(
            f"[{self.NAME}] [RAW] RAW_SIGNAL | market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
            f"{direction.value} | "
            f"conf={confidence:.2f} | votes={len(winning)}/{len(eligible)} eligible | "
            f"w={winning_summary.weight_total:.2f} ws={winning_summary.weighted_score:.2f} | "
            f"strategies={signal.strategies_fired}"
        )
        log_pipeline_stage(
            self.NAME,
            "strategy_signal_generation",
            "passed",
            market_ts=current_ts.strftime("%Y-%m-%d %H:%M IST"),
            direction=direction.value,
            confidence=f"{confidence:.2f}",
            setup=setup.setup_type,
            setup_strength=f"{setup.setup_strength:.2f}",
            votes=len(winning),
            eligible=len(eligible),
            weighted_score=f"{winning_summary.weighted_score:.2f}",
            strategies=",".join(signal.strategies_fired),
            early_trigger=early_trigger,
        )
        self._log_strategy_health_summary()
        await self.bus.publish(Topic.RAW_SIGNAL, signal_dict, self.NAME)
        self._candidate_tracker[direction.value] = {
            "ts": None,
            "price": None,
            "votes": 0,
        }

    def _is_duplicate_signal(self, signal: RawSignal) -> bool:
        if not self._last_signal_at:
            return False
        cooldown = timedelta(minutes=DUPLICATE_SIGNAL_COOLDOWN_MINUTES)
        if signal.timestamp - self._last_signal_at > cooldown:
            return False
        return self._last_signal_key == self._signal_key(signal)

    def _reset_daily_trade_state(self, ts: datetime) -> None:
        today_str = ts.strftime("%Y-%m-%d")
        if today_str == self._today_date:
            return
        self._today_date = today_str
        self._reentry_used_today = False
        self._last_sl_exit_time = None
        self._trades_today_count = 0
        self._last_close_time = None
        self._candidate_tracker = {
            Direction.BUY_CALL.value: {"ts": None, "price": None, "votes": 0},
            Direction.BUY_PUT.value: {"ts": None, "price": None, "votes": 0},
        }
        self._active_candidates = {
            Direction.BUY_CALL.value: None,
            Direction.BUY_PUT.value: None,
        }

    def _second_trade_allowed(
        self,
        *,
        current_ts: datetime,
        regime: Regime,
        setup_strength: float,
        confidence: float,
        votes: int,
        weighted_score: float,
    ) -> bool:
        if self._last_close_time is None:
            return True
        if votes >= 5:
            return True
        if votes >= 4 and (setup_strength >= 0.65 or confidence >= 0.65):
            return True
        elapsed_min = (current_ts - self._last_close_time).total_seconds() / 60
        current_minute = current_ts.hour * 60 + current_ts.minute

        if (
            elapsed_min >= SECOND_TRADE_FAST_REENTRY_MIN
            and current_minute <= SECOND_TRADE_FAST_LAST_ENTRY_MINUTE
            and (not SECOND_TRADE_REQUIRE_TREND or regime == Regime.TRENDING)
            and setup_strength >= SECOND_TRADE_FAST_MIN_SETUP
            and confidence >= SECOND_TRADE_FAST_MIN_CONF
            and votes >= SECOND_TRADE_FAST_MIN_VOTES
            and weighted_score >= SECOND_TRADE_FAST_MIN_WEIGHTED_SCORE
        ):
            return True

        if elapsed_min < SECOND_TRADE_COOLDOWN_MIN:
            return False
        if current_minute > SECOND_TRADE_LAST_ENTRY_MINUTE:
            return False
        if SECOND_TRADE_REQUIRE_TREND and regime != Regime.TRENDING:
            return False
        return (
            setup_strength >= SECOND_TRADE_MIN_SETUP
            and confidence >= SECOND_TRADE_MIN_CONF
            and votes >= SECOND_TRADE_MIN_VOTES
            and weighted_score >= SECOND_TRADE_MIN_WEIGHTED_SCORE
        )

    def _record_gate(self, key: str, detail: str) -> None:
        self._gate_counts[key] = int(self._gate_counts.get(key, 0)) + 1
        count = self._gate_counts[key]
        if count == 1 or count % 10 == 0:
            logger.info(f"[{self.NAME}] Gate[{key}] count={count} | {detail}")

    async def _publish_runner_rejection(
        self,
        *,
        current_ts: datetime,
        direction,
        confidence: float,
        votes: int,
        ltp: float,
        regime,
        winning: list[dict],
        setup,
        weighted_score: float,
        rejection_reason: str,
        extra_meta: dict | None = None,
    ) -> None:
        strategies_fired = [
            str(r.get("name", "") or "").strip()
            for r in winning
            if str(r.get("name", "") or "").strip()
        ]
        payload = {
            "symbol": "NIFTY",
            "direction": getattr(direction, "value", direction),
            "confidence": round(float(confidence or 0.0), 4),
            "votes": int(votes or 0),
            "strategies_fired": strategies_fired,
            "nifty_ltp": round(float(ltp or 0.0), 2),
            "nifty_price": round(float(ltp or 0.0), 2),
            "timestamp": current_ts.isoformat(),
            "regime": getattr(regime, "value", regime),
            "rejection_reason": rejection_reason,
            "runner_rejection_reason": rejection_reason,
            "lifecycle_status": "REJECTED",
            "setup_type": str(getattr(setup, "setup_type", "") or ""),
            "setup_strength": round(float(getattr(setup, "setup_strength", 0.0) or 0.0), 4),
            "weighted_score": round(float(weighted_score or 0.0), 4),
            "metadata": {
                **{str(r.get("name", "") or ""): r.get("meta", {}) for r in winning if str(r.get("name", "") or "")},
                "_context": {
                    "setup": setup.to_dict() if hasattr(setup, "to_dict") else {},
                    "runner_gate_reject": True,
                    "weighted_score": round(float(weighted_score or 0.0), 4),
                },
                **(extra_meta or {}),
            },
        }
        await self.bus.publish(Topic.SIGNAL_REJECTED, payload, self.NAME)

    @staticmethod
    def _compute_hybrid_5m_context(df: pd.DataFrame) -> dict:
        if not HYBRID_5M_CONFIRM_ENABLED or df is None or df.empty:
            return {"enabled": False, "bias": "NEUTRAL", "confidence": 0.0, "reason": "disabled"}
        try:
            normalized = df.copy().sort_index()
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            for col in normalized.columns:
                if col not in agg:
                    agg[col] = "last"
            df_5m = normalized.resample(
                "5min",
                origin="start_day",
                offset="15min",
                label="right",
                closed="right",
            ).agg(agg).dropna(subset=["open", "high", "low", "close"])
            if len(df_5m) < 24:
                return {
                    "enabled": True,
                    "timeframe": "5minute",
                    "bias": "NEUTRAL",
                    "confidence": 0.0,
                    "bars": int(len(df_5m)),
                    "reason": "insufficient_5m_bars",
                }

            close = df_5m["close"].astype(float)
            ema9 = close.ewm(span=9, adjust=False).mean()
            ema21 = close.ewm(span=21, adjust=False).mean()
            last_close = float(close.iloc[-1])
            e9 = float(ema9.iloc[-1])
            e21 = float(ema21.iloc[-1])
            e21_prev = float(ema21.iloc[-4]) if len(ema21) >= 4 else e21
            slope_pct = (e21 - e21_prev) / max(abs(e21_prev), 1.0) * 100.0
            spread_pct = abs(e9 - e21) / max(last_close, 1.0) * 100.0

            if last_close > e9 > e21 and slope_pct > 0:
                bias = "BULLISH"
                confidence = min(0.88, 0.56 + abs(slope_pct) * 5.0 + spread_pct * 4.0)
            elif last_close < e9 < e21 and slope_pct < 0:
                bias = "BEARISH"
                confidence = min(0.88, 0.56 + abs(slope_pct) * 5.0 + spread_pct * 4.0)
            elif e9 > e21:
                bias = "BULLISH"
                confidence = min(0.62, 0.52 + spread_pct * 3.0)
            elif e9 < e21:
                bias = "BEARISH"
                confidence = min(0.62, 0.52 + spread_pct * 3.0)
            else:
                bias = "NEUTRAL"
                confidence = 0.5

            return {
                "enabled": True,
                "timeframe": "5minute",
                "bias": bias,
                "confidence": round(float(confidence), 4),
                "close": round(last_close, 2),
                "ema9": round(e9, 2),
                "ema21": round(e21, 2),
                "slope_pct": round(float(slope_pct), 4),
                "spread_pct": round(float(spread_pct), 4),
                "bars": int(len(df_5m)),
                "reason": "ema9_ema21_slope",
            }
        except Exception as exc:
            logger.debug(f"[StrategyAgent] Hybrid 5m context unavailable: {exc}")
            return {"enabled": True, "bias": "NEUTRAL", "confidence": 0.0, "reason": "compute_error"}

    def _compute_hybrid_5m_strategy_anchor(
        self,
        *,
        df: pd.DataFrame,
        regime_label: str,
        strategy_context: dict,
    ) -> dict:
        if not HYBRID_5M_ANCHOR_ENABLED or df is None or df.empty:
            return {"enabled": False, "direction": "NONE", "votes": 0, "reason": "disabled"}
        try:
            normalized = df.copy().sort_index()
            agg = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
            for col in normalized.columns:
                if col not in agg:
                    agg[col] = "last"
            df_5m = normalized.resample(
                "5min",
                origin="start_day",
                offset="15min",
                label="right",
                closed="right",
            ).agg(agg).dropna(subset=["open", "high", "low", "close"])
            if len(df_5m) < 30:
                return {
                    "enabled": True,
                    "direction": "NONE",
                    "votes": 0,
                    "weighted_score": 0.0,
                    "strategies": [],
                    "reason": "insufficient_5m_bars",
                }

            cache_5m = IndicatorCache(df_5m)
            results: list[dict] = []
            for meta in STRATEGY_REGISTRY:
                if meta.name in HYBRID_5M_ANCHOR_EXCLUDED or meta.requires_live_broker:
                    continue
                if meta.requires_volume and not _has_meaningful_volume(df_5m, meta.min_candles):
                    continue
                results.append(
                    _run_strategy_sync(
                        meta,
                        df_5m,
                        cache_5m,
                        self._orb_high,
                        self._orb_low,
                        strategy_context,
                    )
                )

            label = str(regime_label or "TRENDING").upper()
            call_votes = self._annotate_votes(
                [r for r in results if r.get("direction") == Direction.BUY_CALL],
                label,
            )
            put_votes = self._annotate_votes(
                [r for r in results if r.get("direction") == Direction.BUY_PUT],
                label,
            )
            call_summary = self._summarize_votes(call_votes)
            put_summary = self._summarize_votes(put_votes)
            if call_summary.weighted_score > put_summary.weighted_score:
                direction = Direction.BUY_CALL.value
                winning = call_votes
                summary = call_summary
            elif put_summary.weighted_score > call_summary.weighted_score:
                direction = Direction.BUY_PUT.value
                winning = put_votes
                summary = put_summary
            else:
                direction = "NONE"
                winning = []
                summary = WeightedVoteSummary(0, 0.0, 0.0, 0.0)

            return {
                "enabled": True,
                "timeframe": "5minute",
                "direction": direction,
                "votes": int(summary.count),
                "weighted_score": float(summary.weighted_score),
                "weight_total": float(summary.weight_total),
                "strategies": [str(r.get("name", "")) for r in winning],
                "call_votes": int(call_summary.count),
                "put_votes": int(put_summary.count),
                "call_ws": float(call_summary.weighted_score),
                "put_ws": float(put_summary.weighted_score),
                "reason": "legacy_strategy_anchor",
            }
        except Exception as exc:
            logger.debug(f"[StrategyAgent] Hybrid 5m strategy anchor unavailable: {exc}")
            return {"enabled": True, "direction": "NONE", "votes": 0, "reason": "compute_error"}

    @staticmethod
    def _passes_hybrid_5m_gate(
        *,
        hybrid_5m: dict,
        hybrid_5m_anchor: dict,
        direction: Direction,
        setup_strength: float,
        votes: int,
        weighted_score: float,
    ) -> tuple[bool, str]:
        if not HYBRID_5M_CONFIRM_ENABLED:
            return True, "hybrid_disabled"
        anchor = hybrid_5m_anchor or {}
        anchor_direction = str(anchor.get("direction", "NONE") or "NONE")
        anchor_votes = int(anchor.get("votes", 0) or 0)
        anchor_score = float(anchor.get("weighted_score", 0.0) or 0.0)
        if (
            HYBRID_5M_ANCHOR_ENABLED
            and anchor_direction in {Direction.BUY_CALL.value, Direction.BUY_PUT.value}
            and anchor_votes >= HYBRID_5M_ANCHOR_MIN_VOTES
            and anchor_score >= HYBRID_5M_ANCHOR_MIN_SCORE
        ):
            if anchor_direction == direction.value:
                return True, (
                    f"5m strategy anchor aligned dir={anchor_direction} "
                    f"votes={anchor_votes} ws={anchor_score:.2f}"
                )
            strong_counter = (
                setup_strength >= HYBRID_5M_STRONG_COUNTER_SETUP
                and votes >= HYBRID_5M_STRONG_COUNTER_VOTES
                and weighted_score >= HYBRID_5M_STRONG_COUNTER_WS
            )
            if not strong_counter:
                return False, (
                    f"5m strategy anchor conflict anchor={anchor_direction} "
                    f"votes={anchor_votes} ws={anchor_score:.2f} "
                    f"direction={direction.value} setup={setup_strength:.2f} "
                    f"votes3m={votes} ws3m={weighted_score:.2f}"
                )
        elif HYBRID_5M_ANCHOR_ENABLED:
            # Do not hard-block on EMA-only 5m bias when legacy strategies do not
            # form a 5m anchor. EMA bias flips late and was suppressing valid 3m
            # entries while preserving almost no additional downside protection.
            return True, (
                f"5m strategy anchor absent/weak dir={anchor_direction} "
                f"votes={anchor_votes} ws={anchor_score:.2f}"
            )

        bias = str((hybrid_5m or {}).get("bias", "NEUTRAL") or "NEUTRAL").upper()
        confidence = float((hybrid_5m or {}).get("confidence", 0.0) or 0.0)
        if not HYBRID_5M_EMA_BLOCK_ENABLED:
            return True, f"5m ema context-only bias={bias} conf={confidence:.2f}"
        if bias == "NEUTRAL" or confidence < HYBRID_5M_MIN_CONF:
            return True, f"5m neutral/weak bias={bias} conf={confidence:.2f}"

        aligned = (
            (bias == "BULLISH" and direction == Direction.BUY_CALL)
            or (bias == "BEARISH" and direction == Direction.BUY_PUT)
        )
        if aligned:
            return True, f"5m aligned bias={bias} conf={confidence:.2f}"

        strong_counter = (
            setup_strength >= HYBRID_5M_STRONG_COUNTER_SETUP
            and votes >= HYBRID_5M_STRONG_COUNTER_VOTES
            and weighted_score >= HYBRID_5M_STRONG_COUNTER_WS
        )
        if strong_counter:
            return True, (
                f"5m counter-trend override bias={bias} conf={confidence:.2f} "
                f"setup={setup_strength:.2f} votes={votes} ws={weighted_score:.2f}"
            )
        return False, (
            f"5m conflict bias={bias} conf={confidence:.2f} "
            f"direction={direction.value} setup={setup_strength:.2f} "
            f"votes={votes} ws={weighted_score:.2f}"
        )

    def _remember_signal(self, signal: RawSignal) -> None:
        self._last_signal_key = self._signal_key(signal)
        self._last_signal_at = signal.timestamp

    @staticmethod
    def _should_allow_early_trigger(
        *,
        votes: int = 4,
        best_conf: float,
        weighted_score: float,
        setup,
    ) -> bool:
        if votes < 4:
            return False
        setup_type = str(getattr(setup, "setup_type", "") or "").lower()
        setup_strength = float(getattr(setup, "setup_strength", 0.0) or 0.0)
        effective_weighted_score = weighted_score if weighted_score > 0 else best_conf
        is_reversion = any(
            token in setup_type
            for token in ("pullback", "reversion", "reversal", "mean")
        )

        if (
            best_conf >= RUNNER_EARLY_TRIGGER_HIGH_CONF
            and setup_strength >= RUNNER_EARLY_TRIGGER_STRENGTH_THRESHOLD
            and effective_weighted_score >= RUNNER_EARLY_TRIGGER_WEIGHTED_SCORE_THRESHOLD
        ):
            return True

        if (
            best_conf >= RUNNER_EARLY_TRIGGER_MIN_CONF
            and setup_strength >= RUNNER_EARLY_TRIGGER_STRENGTH_HIGH
            and effective_weighted_score >= RUNNER_EARLY_TRIGGER_WEIGHTED_SCORE_HIGH
        ):
            return True

        if (
            is_reversion
            and best_conf >= RUNNER_EARLY_TRIGGER_MIN_CONF
            and setup_strength >= RUNNER_REVERSION_STRENGTH_THRESHOLD
            and effective_weighted_score >= RUNNER_REVERSION_WEIGHTED_SCORE_THRESHOLD
        ):
            return True

        if (
            best_conf >= 0.70
            and setup_strength >= STRATEGY_RUNNER_SETUP_STRENGTH_THRESH_0_7
            and setup_type in {"breakout", "trend_pullback", "vote_aligned"}
        ):
            return True

        return False

    @staticmethod
    def _signal_key(signal: RawSignal) -> str:
        strike_bucket = int(round(signal.nifty_ltp / float(NIFTY_STRIKE_STEP)) * NIFTY_STRIKE_STEP)
        strategies = "|".join(sorted(signal.strategies_fired))
        setup = ((signal.metadata or {}).get("_context") or {}).get("setup", {}) or {}
        setup_type = str(setup.get("setup_type", "") or "")
        return f"{signal.direction.value}:{strike_bucket}:{setup_type}:{strategies}"

    @staticmethod
    def _strategy_vote_weight(
        strategy_name: str,
        regime_label: str,
    ) -> tuple[float, float, float]:
        base_weight = float(STRATEGY_BASE_WEIGHTS.get(strategy_name, 1.0))
        regime_map = STRATEGY_REGIME_WEIGHTS.get(regime_label, {})
        regime_weight = float(regime_map.get(strategy_name, 1.0))
        adjusted = round(base_weight * regime_weight, 4)
        return base_weight, regime_weight, adjusted

    def _annotate_votes(
        self,
        candidates: list[dict],
        regime_label: str,
    ) -> list[dict]:
        strategy_scores: dict[str, float] = {}
        for c in candidates:
            lead = c.get("lead_confidence", {})
            name = str(lead.get("name") or c.get("name") or "")
            conf = float(lead.get("confidence", c.get("confidence", 0.0)) or 0.0)
            if name:
                strategy_scores[name] = max(strategy_scores.get(name, 0.0), conf)

        # Calculate WEIGHTED votes based on strategy performance
        weighted_votes = 0.0
        for name, conf in strategy_scores.items():
            if conf >= MIN_STRATEGY_CONF:
                _, _, weight = self._strategy_vote_weight(name, regime_label)
                weighted_votes += weight

        for c in candidates:
            lead = c.get("lead_confidence", {})
            name = str(lead.get("name") or c.get("name") or "")
            conf = float(lead.get("confidence", c.get("confidence", 0.0)) or 0.0)
            base_weight, regime_weight, vote_weight = self._strategy_vote_weight(name, regime_label)
            weighted_score = conf * vote_weight if conf >= MIN_STRATEGY_CONF else 0.0
            # Use weighted votes instead of flat count
            c["votes"] = weighted_votes
            c["_base_weight"] = base_weight
            c["_regime_weight"] = regime_weight
            c["_vote_weight"] = vote_weight
            c["_weighted_score"] = round(weighted_score, 4)
            c["raw_votes"] = sum(
                1
                for name, conf in strategy_scores.items()
                if conf >= MIN_STRATEGY_CONF
            )
            c["strategy_combo"] = sorted(strategy_scores.keys())
            # Weighted average confidence
            weighted_conf_sum = sum(
                conf * self._strategy_vote_weight(name, regime_label)[2]
                for name, conf in strategy_scores.items()
            )
            total_weight = sum(
                self._strategy_vote_weight(name, regime_label)[2]
                for name in strategy_scores.keys()
            )
            c["avg_confidence"] = (
                weighted_conf_sum / total_weight if total_weight > 0 else 0.0
            )
            c["max_confidence"] = (
                max(strategy_scores.values()) if strategy_scores else 0.0
            )

        return candidates

    def _apply_wyckoff_to_candidates(
        self,
        candidates: list[dict],
        direction: Direction,
        wyckoff_dict: dict,
        current_ts: datetime,
    ) -> list[dict]:
        if not candidates or not wyckoff_dict:
            return candidates
        try:
            wyckoff = WyckoffResult.from_dict(wyckoff_dict)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Wyckoff payload ignored: {exc}")
            return candidates

        _, _, allow = wyckoff.apply_to_signal(direction.value, 0.7, 1)
        if not allow:
            # If Wyckoff has strong conviction (confidence >= 65%), respect the structural veto
            is_strong_conflict = (
                wyckoff.confidence >= 65.0
                and (
                    (direction == Direction.BUY_CALL and "FAVOR_PUT" in str(wyckoff.recommendation))
                    or (direction == Direction.BUY_PUT and "FAVOR_CALL" in str(wyckoff.recommendation))
                )
            )
            candidate_names = {str(c.get("name", "") or "") for c in candidates}
            confirmed_basket = (
                not is_strong_conflict
                and len(candidates) >= 5
                and (
                    {"FVG", "ValueArea"}.issubset(candidate_names)
                    or {"ORB", "OIAnalysis"}.issubset(candidate_names)
                )
            )
            if confirmed_basket:
                logger.info(
                    f"[{self.NAME}] Wyckoff veto bypassed for confirmed basket | "
                    f"direction={direction.value} phase={wyckoff.phase} "
                    f"conf={wyckoff.confidence:.1f} rec={wyckoff.recommendation} "
                    f"strategies={sorted(candidate_names)}"
                )
            else:
                self._record_gate(
                    "wyckoff_phase_block",
                    (
                        f"market_ts={current_ts.strftime('%Y-%m-%d %H:%M IST')} | "
                        f"direction={direction.value} | phase={wyckoff.phase} | "
                        f"rec={wyckoff.recommendation}"
                    ),
                )
                logger.info(
                    f"[{self.NAME}] Wyckoff blocked {direction.value} votes | "
                    f"phase={wyckoff.phase} conf={wyckoff.confidence:.1f} rec={wyckoff.recommendation}"
                )
                return []

        adjusted: list[dict] = []
        multiplier = wyckoff.call_multiplier if direction == Direction.BUY_CALL else wyckoff.put_multiplier
        for candidate in candidates:
            row = dict(candidate)
            conf = float(row.get("confidence", 0.0) or 0.0)
            adj_conf = round(min(0.95, max(0.0, conf + conf * multiplier)), 4)
            row["confidence"] = adj_conf
            if isinstance(row.get("lead_confidence"), dict):
                lead = dict(row.get("lead_confidence") or {})
                lead["confidence"] = adj_conf
                row["lead_confidence"] = lead
            meta = dict(row.get("meta", {}) or {})
            meta["wyckoff"] = {
                "phase": wyckoff.phase,
                "recommendation": wyckoff.recommendation,
                "confidence": wyckoff.confidence,
                "confidence_before": conf,
                "confidence_after": adj_conf,
                "size_multiplier": wyckoff.size_multiplier,
                "veto_bypassed": bool(not allow),
            }
            row["meta"] = meta
            adjusted.append(row)
        return adjusted

    @staticmethod
    def _summarize_votes(votes: list[dict]) -> WeightedVoteSummary:
        count = len(votes)
        weight_total = round(sum(float(v.get("_vote_weight", 1.0)) for v in votes), 4)
        weighted_score = round(sum(float(v.get("_weighted_score", 0.0)) for v in votes), 4)
        avg_weight = round(weight_total / count, 4) if count else 0.0
        return WeightedVoteSummary(
            count=count,
            weight_total=weight_total,
            weighted_score=weighted_score,
            avg_weight=avg_weight,
        )

    @staticmethod
    def _get_category_votes(strategy_names: list[str]) -> tuple[int, dict[str, list[str]], list[str]]:
        """
        Calculate independent category vote metrics across strategy names.
        Returns:
            (independent_category_count, category_breakdown, distinct_categories)
        """
        cat_map: dict[str, list[str]] = {}
        for name in strategy_names:
            s_name = str(name).strip()
            cat = STRATEGY_CATEGORY_MAPPING.get(s_name, "uncategorized")
            if cat not in cat_map:
                cat_map[cat] = []
            cat_map[cat].append(s_name)
        distinct_cats = sorted(cat_map.keys())
        return len(distinct_cats), cat_map, distinct_cats

    @staticmethod
    def _classify_entry_timing(extension_atr: float, dist_ema_atr: float, consec_bars: int) -> str:
        """
        Telemetry-only classification of signal timing quality.
        Does NOT alter trading logic.
        """
        if extension_atr <= 0.65 and dist_ema_atr <= 0.75:
            return "EARLY"
        elif extension_atr <= 1.35 and dist_ema_atr <= 1.50:
            return "VALID"
        elif extension_atr <= 2.10 and dist_ema_atr <= 2.25:
            return "EXTENDED"
        else:
            return "EXHAUSTED"

    @classmethod
    def _evaluate_late_entry_gate(
        cls,
        *,
        entry_quality: dict,
        setup_type: str,
        setup_strength: float,
        indep_cat_count: int,
        confidence: float,
    ) -> dict:
        """
        Evaluate Late-Entry Rejection Gate focusing on Strongest Rejection Conditions:
        - EXHAUSTION_RISK (EMA distance > 2.25x ATR or 5+ consecutive bars with Ext > 1.80x ATR)
        - OVEREXTENDED_FROM_VWAP (VWAP distance > 2.50x ATR)
        - MULTIPLE_REASONS (>= 2 simultaneous rejection breaches)
        - POOR_REMAINING_RR (Remaining RR < 0.75 with Ext > 1.80x ATR)
        - Exempts: Institutional multi-category breakouts (4+ categories or setup_strength >= 0.80)
        - Permitted: Standalone single-dimension extension without EMA/VWAP breach (CAUTION)
        """
        ext_atr = float((entry_quality or {}).get("extension_atr", 0.0) or 0.0)
        ema_atr = float((entry_quality or {}).get("dist_from_ema20_atr", 0.0) or 0.0)
        vwap_atr = float((entry_quality or {}).get("dist_from_vwap_atr", 0.0) or 0.0)
        consec = int((entry_quality or {}).get("consecutive_directional_candles", 0) or 0)
        move_consumed_pct = float((entry_quality or {}).get("move_consumed_pct", 0.0) or 0.0)
        dist_breakout = float((entry_quality or {}).get("dist_from_breakout_level", 0.0) or 0.0)
        opposing_dist = (entry_quality or {}).get("opposing_level_dist")
        atr_val = float((entry_quality or {}).get("atr", 20.0) or 20.0)
        timing_class = str((entry_quality or {}).get("timing_classification", "VALID"))

        # Calculate remaining reward-to-risk where opposing structure exists
        recent_move_pts = float((entry_quality or {}).get("recent_directional_move_pts", 0.0) or atr_val)
        if opposing_dist is not None and opposing_dist > 0:
            remaining_rr = round(float(opposing_dist) / max(recent_move_pts, atr_val, 1.0), 2)
        else:
            remaining_rr = None

        rejection_reasons = []

        # 1. Extended Move Check
        if ext_atr > MAX_ENTRY_EXTENSION_ATR:
            rejection_reasons.append("EXTENDED_MOVE")

        # 2. Exhaustion Risk Check (Strong Rejection)
        if ema_atr > MAX_ENTRY_DIST_EMA_ATR or (consec >= MAX_CONSEC_BARS_BEFORE_REJECTION and ext_atr > 1.80):
            rejection_reasons.append("EXHAUSTION_RISK")

        # 3. Overextended from VWAP Check (Strong Rejection)
        if vwap_atr > 2.50:
            rejection_reasons.append("OVEREXTENDED_FROM_VWAP")

        # 4. Poor Remaining Reward-to-Risk Check (Strong Rejection if extended)
        if remaining_rr is not None and remaining_rr < 0.75 and ext_atr > 1.80:
            rejection_reasons.append("POOR_REMAINING_RR")

        # Institutional breakout exemption
        is_exempt = (
            str(setup_type or "").lower() in {"breakout", "vote_aligned", "trend_pullback"}
            and (indep_cat_count >= 4 or setup_strength >= 0.75 or confidence >= 0.90)
        )
        # Climax/Exhaustion protection: Do not exempt EXHAUSTION_RISK if low category breadth
        if "EXHAUSTION_RISK" in rejection_reasons and indep_cat_count < 3:
            is_exempt = False

        # Determine strongest rejection conditions
        strong_rejection = (
            "EXHAUSTION_RISK" in rejection_reasons
            or "OVEREXTENDED_FROM_VWAP" in rejection_reasons
            or "POOR_REMAINING_RR" in rejection_reasons
            or len(rejection_reasons) >= 2
        )

        if is_exempt and rejection_reasons:
            shadow_entry_decision = "CAUTION"
            entry_decision = "CAUTION"
            primary_reason = ""
            is_rejected = False
            would_have_blocked = False
        elif strong_rejection:
            shadow_entry_decision = "REJECT"
            primary_reason = "MULTIPLE_REASONS" if len(rejection_reasons) > 1 else rejection_reasons[0]
            would_have_blocked = True
            if ENABLE_LATE_ENTRY_FILTER:
                entry_decision = "REJECT"
                is_rejected = True
            else:
                entry_decision = "CAUTION"
                is_rejected = False
        elif rejection_reasons: # Standalone single EXTENDED_MOVE permitted under CAUTION
            shadow_entry_decision = "CAUTION"
            entry_decision = "CAUTION"
            primary_reason = ""
            is_rejected = False
            would_have_blocked = False
        else:
            is_caution = (
                (1.35 < ext_atr <= MAX_ENTRY_EXTENSION_ATR)
                or (1.50 < ema_atr <= MAX_ENTRY_DIST_EMA_ATR)
                or (1.75 < vwap_atr <= 2.50)
                or (consec == 4)
                or (remaining_rr is not None and 0.75 <= remaining_rr < 1.20)
            )
            shadow_entry_decision = "CAUTION" if is_caution else "ACCEPT"
            entry_decision = shadow_entry_decision
            primary_reason = ""
            is_rejected = False
            would_have_blocked = False

        timing_metrics_used = {
            "extension_atr": ext_atr,
            "dist_from_ema20_atr": ema_atr,
            "dist_from_vwap_atr": vwap_atr,
            "consecutive_directional_candles": consec,
            "move_consumed_pct": move_consumed_pct,
            "dist_from_breakout_level": dist_breakout,
            "opposing_level_dist": opposing_dist,
            "remaining_reward_to_risk": remaining_rr,
            "institutional_breakout_exempt": is_exempt,
            "strong_rejection_criteria_met": strong_rejection,
        }

        return {
            "entry_decision": entry_decision,
            "shadow_entry_decision": shadow_entry_decision,
            "rejection_reasons": rejection_reasons,
            "shadow_rejection_reasons": rejection_reasons,
            "primary_rejection_reason": primary_reason,
            "is_rejected": is_rejected,
            "would_have_blocked": would_have_blocked,
            "entry_quality_classification": timing_class,
            "timing_metrics_used": timing_metrics_used,
        }

    def _process_early_candidate_cycle(
        self,
        *,
        current_ts: datetime,
        ltp: float,
        candle_high: float,
        candle_low: float,
        call_votes: list[dict],
        put_votes: list[dict],
        call_summary: WeightedVoteSummary,
        put_summary: WeightedVoteSummary,
    ) -> None:
        """
        Process Early Candidate Shadow Mode.
        Detects early directional candidates, tracks excursion (MFE/MAE), and manages lifecycle.
        Does NOT alter live trading.
        """
        vote_data = [
            (Direction.BUY_CALL.value, call_votes, call_summary),
            (Direction.BUY_PUT.value, put_votes, put_summary),
        ]

        for dir_val, votes, summary in vote_data:
            strat_names = [v["name"] for v in votes]
            indep_cat_count, cat_breakdown, distinct_cats = self._get_category_votes(strat_names)
            is_call = (dir_val == Direction.BUY_CALL.value)
            active_cand = self._active_candidates.get(dir_val)

            # Candidate trigger criteria: >= 2 votes OR >= 2 categories OR (>= 1 vote with high weight >= 1.50)
            candidate_condition = (
                len(votes) >= 2
                or indep_cat_count >= 2
                or (len(votes) >= 1 and summary.weighted_score >= 1.50)
            )

            if candidate_condition and active_cand is None:
                cand_id = f"CAND_{current_ts.strftime('%Y%m%d_%H%M%S')}_{dir_val}"
                self._active_candidates[dir_val] = {
                    "candidate_id": cand_id,
                    "candidate_ts": current_ts.isoformat(),
                    "candidate_price": round(ltp, 2),
                    "direction": dir_val,
                    "strategies_active": strat_names,
                    "raw_vote_count": len(votes),
                    "independent_category_count": indep_cat_count,
                    "category_votes": {
                        "independent_category_count": indep_cat_count,
                        "categories": distinct_cats,
                        "category_breakdown": cat_breakdown,
                    },
                    "partial_confidence": round(sum(float(v.get("confidence", 0.0)) for v in votes), 3),
                    "partial_weighted_score": round(summary.weighted_score, 3),
                    "lifecycle_status": "CANDIDATE",
                    "bars_active": 1,
                    "max_favorable_excursion_pts": 0.0,
                    "max_adverse_excursion_pts": 0.0,
                    "confirmed_signal_id": None,
                    "time_saved_sec": None,
                    "price_improvement_pts": None,
                }
                logger.info(
                    f"[{self.NAME}] [EARLY CANDIDATE SHADOW] Inception: {cand_id} | price={ltp} | "
                    f"votes={len(votes)} | cats={distinct_cats} | score={summary.weighted_score:.2f}"
                )
            elif active_cand is not None:
                active_cand["bars_active"] += 1
                cand_price = float(active_cand["candidate_price"])
                if is_call:
                    favorable = max(0.0, candle_high - cand_price)
                    adverse = max(0.0, cand_price - candle_low)
                else:
                    favorable = max(0.0, cand_price - candle_low)
                    adverse = max(0.0, candle_high - cand_price)

                active_cand["max_favorable_excursion_pts"] = max(
                    float(active_cand["max_favorable_excursion_pts"]), round(favorable, 2)
                )
                active_cand["max_adverse_excursion_pts"] = max(
                    float(active_cand["max_adverse_excursion_pts"]), round(adverse, 2)
                )

                # Check expiration (>= 6 bars / 30m without confirmation or votes drop to 0)
                if active_cand["bars_active"] >= 6 or len(votes) == 0:
                    active_cand["lifecycle_status"] = "EXPIRED"
                    active_cand["exit_price"] = round(ltp, 2)
                    active_cand["exit_ts"] = current_ts.isoformat()
                    logger.info(
                        f"[{self.NAME}] [EARLY CANDIDATE SHADOW] Expired: {active_cand['candidate_id']} "
                        f"after {active_cand['bars_active']} bars | MFE={active_cand['max_favorable_excursion_pts']} | "
                        f"MAE={active_cand['max_adverse_excursion_pts']}"
                    )
                    self._candidate_shadow_history.append(active_cand)
                    self._active_candidates[dir_val] = None

    def _confirm_early_candidate(
        self,
        *,
        direction: Direction,
        signal_id: str,
        confirmed_price: float,
        confirmed_ts: datetime,
    ) -> dict | None:
        """
        Confirm an active early candidate and calculate time saved / price improvement.
        """
        active_cand = self._active_candidates.get(direction.value)
        if not active_cand:
            return None

        active_cand["lifecycle_status"] = "CONFIRMED"
        active_cand["confirmed_signal_id"] = signal_id
        active_cand["confirmed_price"] = round(confirmed_price, 2)
        active_cand["confirmed_ts"] = confirmed_ts.isoformat()

        try:
            cand_dt = pd.to_datetime(active_cand["candidate_ts"])
            active_cand["time_saved_sec"] = round(max(0.0, (confirmed_ts - cand_dt).total_seconds()), 1)
        except Exception:
            active_cand["time_saved_sec"] = 0.0

        cand_price = float(active_cand["candidate_price"])
        is_call = (direction == Direction.BUY_CALL)
        if is_call:
            price_improvement = round(confirmed_price - cand_price, 2)
        else:
            price_improvement = round(cand_price - confirmed_price, 2)

        active_cand["price_improvement_pts"] = price_improvement

        logger.info(
            f"[{self.NAME}] [EARLY CANDIDATE SHADOW] Confirmed: {active_cand['candidate_id']} -> {signal_id} | "
            f"time_saved={active_cand['time_saved_sec']}s | price_improvement={price_improvement}pts | "
            f"MFE={active_cand['max_favorable_excursion_pts']} | MAE={active_cand['max_adverse_excursion_pts']}"
        )
        self._candidate_shadow_history.append(active_cand)
        self._active_candidates[direction.value] = None
        return dict(active_cand)

    def _reject_early_candidate(
        self,
        *,
        direction: Direction,
        rejection_reason: str,
        current_ts: datetime,
    ) -> dict | None:
        """
        Reject an active early candidate when runner gates or late-entry filters block the signal.
        """
        active_cand = self._active_candidates.get(direction.value)
        if not active_cand:
            return None

        active_cand["lifecycle_status"] = "REJECTED"
        active_cand["rejection_reason"] = rejection_reason
        active_cand["rejected_ts"] = current_ts.isoformat()

        logger.info(
            f"[{self.NAME}] [EARLY CANDIDATE SHADOW] Rejected: {active_cand['candidate_id']} | "
            f"reason={rejection_reason} | MFE={active_cand['max_favorable_excursion_pts']} | MAE={active_cand['max_adverse_excursion_pts']}"
        )
        self._candidate_shadow_history.append(active_cand)
        self._active_candidates[direction.value] = None
        return dict(active_cand)

    @classmethod
    def _build_timing_and_quality_telemetry(
        cls,
        *,
        df: pd.DataFrame,
        ltp: float,
        direction: Direction,
        current_ts: datetime,
        cand_info: dict,
        setup,
        hybrid_5m: dict,
    ) -> tuple[dict, dict, str]:
        """
        Build additive signal timing telemetry and entry quality measurements.
        """
        cand_ts = (cand_info or {}).get("ts") or current_ts
        cand_price = float((cand_info or {}).get("price") if (cand_info or {}).get("price") is not None else ltp)
        confirmation_ts = current_ts
        confirmation_price = float(ltp)
        raw_signal_ts = current_ts
        raw_signal_price = float(ltp)

        confirmation_delay_sec = max(0.0, (confirmation_ts - cand_ts).total_seconds())
        cand_to_conf_pts = round(abs(confirmation_price - cand_price), 2)

        timing_telemetry = {
            "candidate_ts": cand_ts.isoformat(),
            "confirmation_ts": confirmation_ts.isoformat(),
            "raw_signal_ts": raw_signal_ts.isoformat(),
            "entry_ts": None,
            "candidate_price": round(cand_price, 2),
            "confirmation_price": round(confirmation_price, 2),
            "raw_signal_price": round(raw_signal_price, 2),
            "entry_price": None,
            "confirmation_delay_sec": round(confirmation_delay_sec, 1),
            "execution_delay_sec": None,
            "candidate_to_confirmation_pts": cand_to_conf_pts,
            "confirmation_to_entry_pts": None,
        }

        # Entry quality & extension calculations
        atr_pts, atr_pct = compute_atr(df, period=14)
        atr_val = max(float(atr_pts or 20.0), 1.0)
        is_call = (direction == Direction.BUY_CALL)

        lookback_bars = min(len(df), 8)
        recent_slice = df.tail(lookback_bars) if len(df) else df
        swing_anchor = float(recent_slice["low"].min() if is_call and len(recent_slice) else (recent_slice["high"].max() if len(recent_slice) else ltp))
        recent_directional_move_pts = round(abs(ltp - swing_anchor), 2)
        extension_atr = round(recent_directional_move_pts / atr_val, 2)

        # Consecutive directional bars
        consec_bars = 0
        if len(df) >= 1:
            for k in range(len(df) - 1, max(-1, len(df) - 10), -1):
                c_open = float(df["open"].iloc[k])
                c_close = float(df["close"].iloc[k])
                if is_call and c_close >= c_open:
                    consec_bars += 1
                elif not is_call and c_close <= c_open:
                    consec_bars += 1
                else:
                    break

        # Distance from 20 EMA
        ema20_series = df["close"].ewm(span=20, adjust=False).mean() if len(df) else None
        ema20_val = float(ema20_series.iloc[-1]) if ema20_series is not None and len(ema20_series) else ltp
        dist_from_ema20 = round(ltp - ema20_val if is_call else ema20_val - ltp, 2)
        dist_from_ema20_atr = round(dist_from_ema20 / atr_val, 2)

        # Distance from VWAP
        vwap_val = float((hybrid_5m or {}).get("vwap", ltp) if isinstance(hybrid_5m, dict) else ltp)
        dist_from_vwap = round(ltp - vwap_val if is_call else vwap_val - ltp, 2)
        dist_from_vwap_atr = round(dist_from_vwap / atr_val, 2)

        # Structure / Breakout distance
        setup_ctx = getattr(setup, "context", {}) or {} if setup else {}
        struct_level = float(getattr(setup, "anchor_price", 0.0) or setup_ctx.get("breakout_level", 0.0) or swing_anchor)
        dist_from_breakout_level = round(abs(ltp - struct_level), 2)

        # Opposing structure level distance
        opposing_level = float(setup_ctx.get("target_level", 0.0) or 0.0)
        opposing_level_dist = round(abs(opposing_level - ltp), 2) if opposing_level > 0 else None

        # Move consumed pct relative to 2x ATR baseline move
        expected_move_pts = atr_val * 2.0
        move_consumed_pct = round(min(100.0, (recent_directional_move_pts / expected_move_pts) * 100), 1)

        timing_classification = cls._classify_entry_timing(extension_atr, dist_from_ema20_atr, consec_bars)

        entry_quality = {
            "atr": round(atr_val, 2),
            "atr_pct": round(float(atr_pct or 0.0), 3),
            "extension_atr": extension_atr,
            "consecutive_directional_candles": consec_bars,
            "recent_directional_move_pts": recent_directional_move_pts,
            "move_consumed_pct": move_consumed_pct,
            "dist_from_vwap": dist_from_vwap,
            "dist_from_vwap_atr": dist_from_vwap_atr,
            "dist_from_ema20": dist_from_ema20,
            "dist_from_ema20_atr": dist_from_ema20_atr,
            "dist_from_breakout_level": dist_from_breakout_level,
            "opposing_level_dist": opposing_level_dist,
            "timing_classification": timing_classification,
        }

        return timing_telemetry, entry_quality, timing_classification

    @classmethod
    def _evaluate_remaining_opportunity(
        cls,
        *,
        df: pd.DataFrame,
        ltp: float,
        direction: Direction,
        atr_val: float,
        setup,
        hybrid_5m: dict,
    ) -> dict:
        """
        Evaluate Remaining Opportunity in Shadow Mode.
        Calculates stop distance, nearest opposing structure / resistance / support,
        estimated remaining reward, and estimated R:R ratio without lookahead bias.
        """
        is_call = (direction == Direction.BUY_CALL)
        lookback = min(len(df), 12)
        recent_slice = df.tail(lookback) if len(df) else df

        # Recent range
        recent_high = float(recent_slice["high"].max() if len(recent_slice) else ltp)
        recent_low = float(recent_slice["low"].min() if len(recent_slice) else ltp)
        recent_range = round(max(0.0, recent_high - recent_low), 2)

        # Risk / Stop distance (anchor swing or 1.0x ATR, clamped min 15 pts, max 50 pts)
        if is_call:
            swing_stop = float(recent_slice["low"].min() if len(recent_slice) else ltp - atr_val)
            stop_dist = round(max(15.0, min(50.0, ltp - swing_stop if (ltp - swing_stop) > 5.0 else atr_val)), 2)
        else:
            swing_stop = float(recent_slice["high"].max() if len(recent_slice) else ltp + atr_val)
            stop_dist = round(max(15.0, min(50.0, swing_stop - ltp if (swing_stop - ltp) > 5.0 else atr_val)), 2)

        # Structure / Target levels
        setup_ctx = getattr(setup, "context", {}) or {} if setup else {}
        target_from_setup = float(setup_ctx.get("target_level", 0.0) or 0.0)

        # Nearest opposing structure (prior session extreme, VWAP upper/lower band, or 30-bar high/low)
        longer_slice = df.tail(min(len(df), 30)) if len(df) else df
        if is_call:
            longer_res = float(longer_slice["high"].max() if len(longer_slice) else ltp + atr_val * 2)
            next_res_sup = round(longer_res if longer_res > ltp + 5.0 else ltp + atr_val * 2.5, 2)
            nearest_opposing = round(target_from_setup if target_from_setup > ltp + 5.0 else next_res_sup, 2)
            dist_to_target = round(max(5.0, nearest_opposing - ltp), 2)
        else:
            longer_sup = float(longer_slice["low"].min() if len(longer_slice) else ltp - atr_val * 2)
            next_res_sup = round(longer_sup if longer_sup < ltp - 5.0 else ltp - atr_val * 2.5, 2)
            nearest_opposing = round(target_from_setup if 0 < target_from_setup < ltp - 5.0 else next_res_sup, 2)
            dist_to_target = round(max(5.0, ltp - nearest_opposing), 2)

        # Expected move based on ATR & expansion (typically 1.8x ATR for 5m intraday trend legs)
        expected_move = round(atr_val * 1.8, 2)

        # Move already consumed in recent swing
        recent_move = round(abs(ltp - (recent_low if is_call else recent_high)), 2)
        remaining_expected = round(max(10.0, expected_move - recent_move), 2)

        # Estimated remaining reward: distance to nearest opposing level/target if bounded, else remaining expected move
        if dist_to_target > 0:
            estimated_remaining_reward = round(dist_to_target, 2)
        else:
            estimated_remaining_reward = round(remaining_expected, 2)

        # Estimated Reward-to-Risk
        estimated_remaining_rr = round(estimated_remaining_reward / max(stop_dist, 1.0), 2)

        # Classification
        if estimated_remaining_rr >= 2.5:
            opp_class = "EXCELLENT"
            decision = "ACCEPT"
        elif estimated_remaining_rr >= 1.5:
            opp_class = "ACCEPTABLE"
            decision = "ACCEPT"
        elif estimated_remaining_rr >= 1.0:
            opp_class = "MARGINAL"
            decision = "CAUTION"
        else:
            opp_class = "POOR"
            decision = "REJECT"

        return {
            "entry_price": round(ltp, 2),
            "stop_risk_distance": stop_dist,
            "nearest_opposing_structure": nearest_opposing,
            "nearest_opposing_level": nearest_opposing,
            "next_resistance_support": next_res_sup,
            "expected_move": expected_move,
            "recent_range": recent_range,
            "atr": round(atr_val, 2),
            "distance_to_potential_target": dist_to_target,
            "estimated_remaining_reward": estimated_remaining_reward,
            "estimated_remaining_rr": estimated_remaining_rr,
            "remaining_opportunity_class": opp_class,
            "shadow_opportunity_decision": decision,
        }

    @classmethod
    def _evaluate_four_pillar_framework(
        cls,
        *,
        direction: Direction,
        votes: int,
        indep_cat_count: int,
        weighted_score: float,
        confidence: float,
        entry_quality: dict,
        late_entry_eval: dict,
        remaining_opportunity: dict,
        current_system_decision: str,
    ) -> dict:
        """
        Evaluate 4-Pillar Decision Framework in Shadow Mode.
        Separates:
        1. DIRECTION: Should we have a directional bias?
        2. ENTRY: Should we enter now?
        3. OPPORTUNITY: Is there enough reward remaining?
        4. EXECUTION: Can we enter efficiently?

        Produces TRADE / WAIT / SKIP with explicit reasons.
        """
        direction_reasons = []
        entry_reasons = []
        opp_reasons = []
        exec_reasons = []

        # ── 1. DIRECTION PILLAR ───────────────────────────────────────────────
        if direction == Direction.NONE:
            direction_decision = "NONE"
            direction_reasons.append("no_directional_bias")
        elif votes >= 4 and indep_cat_count >= 2:
            direction_decision = "CONFIRMED"
            direction_reasons.append(f"multi_category_consensus({votes}_votes,_{indep_cat_count}_cats)")
        elif weighted_score >= 1.50 and votes >= 1:
            direction_decision = "CONFIRMED"
            direction_reasons.append(f"high_conviction_setup({votes}_votes,_ws={weighted_score:.2f})")
        elif indep_cat_count < 2:
            direction_decision = "INSUFFICIENT"
            direction_reasons.append(f"single_category_clustering({indep_cat_count}_cat)")
        else:
            direction_decision = "INSUFFICIENT"
            direction_reasons.append(f"insufficient_consensus({votes}_votes)")

        # ── 2. ENTRY PILLAR ───────────────────────────────────────────────────
        timing_class = entry_quality.get("timing_classification", "NORMAL")
        is_late_rejected = late_entry_eval.get("is_rejected", False)
        primary_rejection = late_entry_eval.get("primary_rejection_reason", "")

        if is_late_rejected:
            entry_decision = "EXHAUSTED"
            entry_reasons.append(f"exhaustion_risk({primary_rejection})")
        elif timing_class in ("LATE", "EXTENDED"):
            entry_decision = "WAIT_PULLBACK"
            entry_reasons.append(f"extended_from_anchor({timing_class})")
        else:
            entry_decision = "ENTER_NOW"
            entry_reasons.append(f"valid_entry_window({timing_class})")

        # ── 3. OPPORTUNITY PILLAR ─────────────────────────────────────────────
        opp_class = remaining_opportunity.get("remaining_opportunity_class", "ACCEPTABLE")
        rr = float(remaining_opportunity.get("estimated_remaining_rr", 1.5) or 1.5)

        if opp_class in ("EXCELLENT", "ACCEPTABLE") or rr >= 1.50:
            opportunity_decision = "VIABLE"
            opp_reasons.append(f"favorable_rr({rr:.2f}R,_{opp_class})")
        elif opp_class == "MARGINAL" or rr >= 1.00:
            opportunity_decision = "SUB_OPTIMAL"
            opp_reasons.append(f"marginal_rr({rr:.2f}R)")
        else:
            opportunity_decision = "POOR_RR"
            opp_reasons.append(f"poor_remaining_reward({rr:.2f}R,_{opp_class})")

        # ── 4. EXECUTION PILLAR ───────────────────────────────────────────────
        execution_decision = "READY"
        exec_reasons.append("in_memory_cache_ready")

        # ── OVERALL FRAMEWORK SYNTHESIS ───────────────────────────────────────
        # Output: TRADE / WAIT / SKIP
        all_reasons = []
        if direction_decision != "CONFIRMED":
            framework_decision = "SKIP"
            all_reasons.extend(direction_reasons)
        elif entry_decision == "EXHAUSTED":
            framework_decision = "SKIP"
            all_reasons.extend(entry_reasons)
        elif opportunity_decision == "POOR_RR":
            framework_decision = "SKIP"
            all_reasons.extend(opp_reasons)
        elif entry_decision == "WAIT_PULLBACK":
            framework_decision = "WAIT"
            all_reasons.extend(entry_reasons)
        elif opportunity_decision == "SUB_OPTIMAL" and indep_cat_count < 3:
            framework_decision = "WAIT"
            all_reasons.extend(opp_reasons)
        else:
            framework_decision = "TRADE"
            all_reasons.append("all_4_pillars_aligned")

        # ── 5. ML QUALITY GATE (SHADOW TELEMETRY) ───────────────────────────
        ml_rank_val = float(entry_quality.get("ml_rank_score", 0.0) or 0.0)
        ml_gate = cls._evaluate_ml_quality_gate(
            ml_confidence=confidence,
            ml_rank_score=ml_rank_val,
            min_rank=0.60,
            min_conf=0.30,
            mode="RANK_OR_CONF",
        )

        decision_before_ml = framework_decision
        if framework_decision == "TRADE" and ml_gate["state"] == "FAIL":
            decision_after_ml = "SKIP"
            ml_would_block = True
            ml_block_reason = f"ml_quality_gate_failed({ml_gate['reason']})"
        else:
            decision_after_ml = framework_decision
            ml_would_block = False
            ml_block_reason = "ml_quality_gate_satisfied"

        # Disagreement tracking
        disagreement_reason = None
        if current_system_decision != framework_decision:
            disagreement_reason = f"Current={current_system_decision} vs Framework={framework_decision} | Reasons: {', '.join(all_reasons)}"

        return {
            "direction_pillar": {
                "decision": direction_decision,
                "reasons": direction_reasons,
                "votes": votes,
                "independent_category_count": indep_cat_count,
            },
            "entry_pillar": {
                "decision": entry_decision,
                "reasons": entry_reasons,
                "timing_classification": timing_class,
            },
            "opportunity_pillar": {
                "decision": opportunity_decision,
                "reasons": opp_reasons,
                "estimated_remaining_rr": rr,
                "remaining_opportunity_class": opp_class,
            },
            "execution_pillar": {
                "decision": execution_decision,
                "reasons": exec_reasons,
            },
            "shadow_framework_decision": framework_decision,
            "shadow_framework_reasons": all_reasons,
            "current_system_decision": current_system_decision,
            "disagreement_reason": disagreement_reason,
            "ml_availability_state": ml_gate["ml_availability_state"],
            "ml_direction_state": ml_gate["ml_direction_state"],
            "ml_quality_state": ml_gate["ml_quality_state"],
            "ml_quality_gate_state": ml_gate["state"],
            "shadow_ml_zero_conviction_gate": ml_gate["state"],
            "shadow_ml_zero_conviction_reason": ml_gate["reason"],
            "ml_quality_gate_reason": ml_gate["reason"],
            "ml_rank_score": ml_gate["ml_rank_score"],
            "ml_confidence": ml_gate["ml_confidence"],
            "threshold_profile": {"min_rank": 0.60, "min_conf": 0.30, "mode": "RANK_OR_CONF"},
            "shadow_decision_before_ml_gate": decision_before_ml,
            "shadow_decision_after_ml_gate": decision_after_ml,
            "would_have_blocked": ml_would_block,
            "ml_block_reason": ml_block_reason,
        }

    @classmethod
    def _evaluate_ml_quality_gate(
        cls,
        *,
        ml_confidence: float,
        ml_rank_score: float,
        min_rank: float = 0.60,
        min_conf: float = 0.30,
        mode: str = "RANK_OR_CONF",
    ) -> dict:
        """
        Evaluate additive shadow ML quality requirements.
        Distinguishes:
        - STATE A (POSITIVE): ML available with positive conviction (confidence > 0.0 or rank >= 0.50).
        - STATE B (NEUTRAL_OR_ZERO): ML available but zero confidence / neutral rank / no conviction.
        - STATE C (UNAVAILABLE): ML layer was not available / missing / failed.
        """
        # Distinguish UNAVAILABLE from NEUTRAL_OR_ZERO
        if ml_confidence is None or ml_rank_score is None:
            return {
                "ml_availability_state": "UNAVAILABLE",
                "ml_direction_state": "UNAVAILABLE",
                "ml_quality_state": "UNAVAILABLE",
                "state": "UNAVAILABLE",
                "reason": "ml_service_unavailable",
                "passed": True,  # Never reject solely due to unavailability
                "ml_rank_score": 0.0,
                "ml_confidence": 0.0,
            }

        # Check if ML was evaluated but returned zero / neutral conviction
        is_positive = (ml_confidence > 0.0 or ml_rank_score >= 0.50)
        
        if is_positive:
            avail_state = "POSITIVE"
            dir_state = "POSITIVE"
            qual_state = "CONFIRMED"
        else:
            avail_state = "NEUTRAL_OR_ZERO"
            dir_state = "NEUTRAL"
            qual_state = "UNCONFIRMED"

        rank_pass = (ml_rank_score >= min_rank)
        conf_pass = (ml_confidence >= min_conf)

        if not is_positive:
            passed = False
            reason = "zero_ml_conviction"
        elif mode == "RANK_AND_CONF":
            passed = rank_pass and conf_pass
            reason = f"rank({ml_rank_score:.2f}>={min_rank})_AND_conf({ml_confidence:.2f}>={min_conf})"
        elif mode == "RANK_OR_CONF":
            passed = rank_pass or conf_pass
            reason = f"rank({ml_rank_score:.2f}>={min_rank})_OR_conf({ml_confidence:.2f}>={min_conf})"
        elif mode == "RANK_ONLY":
            passed = rank_pass
            reason = f"rank({ml_rank_score:.2f}>={min_rank})"
        elif mode == "CONF_ONLY":
            passed = conf_pass
            reason = f"conf({ml_confidence:.2f}>={min_conf})"
        else:
            passed = True
            reason = "no_gate"

        return {
            "ml_availability_state": avail_state,
            "ml_direction_state": dir_state,
            "ml_quality_state": qual_state,
            "state": "PASS" if passed else "FAIL",
            "reason": reason if passed else f"failed_{reason}",
            "passed": passed,
            "ml_rank_score": round(ml_rank_score, 4),
            "ml_confidence": round(ml_confidence, 4),
        }

    @classmethod
    def _evaluate_high_quality_entry_gate(
        cls,
        *,
        ml_confidence: Optional[float],
        ml_rank_score: Optional[float],
        timing_classification: Optional[str],
        raw_strategy_vote_count: Optional[int],
        independent_category_count: Optional[int],
        live_decision: str = "TRADE",
    ) -> dict:
        """
        Evaluate Additive Shadow High-Quality Entry Gate.

        Rule:
        - ML state = POSITIVE (ml_confidence > 0.0 or ml_rank_score >= 0.50)
        - Timing != EXTENDED/EXHAUSTED (neither EXTENDED nor EXHAUSTED)
        - raw_strategy_vote_count >= 7
        - independent_category_count >= 2

        Output:
        - state: PASS / FAIL / UNAVAILABLE
        - reasons: list of failure reasons
        - disagreement: LIVE_TRADE_SHADOW_PASS / LIVE_TRADE_SHADOW_FAIL / etc.
        """
        reasons = []

        # 1. Evaluate ML State
        if ml_confidence is None and ml_rank_score is None:
            ml_state = "UNAVAILABLE"
        elif float(ml_confidence or 0.0) > 0.0 or float(ml_rank_score or 0.0) >= 0.50:
            ml_state = "POSITIVE"
        else:
            ml_state = "NEUTRAL_OR_ZERO"

        # 2. Evaluate Timing State
        if not timing_classification:
            timing_state = "UNAVAILABLE"
        else:
            timing_state = str(timing_classification).upper()

        # 3. Check for missing critical telemetry
        if ml_state == "UNAVAILABLE" or timing_state == "UNAVAILABLE" or raw_strategy_vote_count is None or independent_category_count is None:
            gate_state = "UNAVAILABLE"
            reasons.append("TELEMETRY_UNAVAILABLE")
        else:
            # Evaluate all failure conditions (preserve all applicable reasons)
            if ml_state != "POSITIVE":
                reasons.append("ML_NOT_POSITIVE")

            if timing_state == "EXTENDED":
                reasons.append("TIMING_EXTENDED")
            elif timing_state == "EXHAUSTED":
                reasons.append("TIMING_EXHAUSTED")
            elif timing_state not in ("EARLY", "VALID", "NORMAL"):
                reasons.append(f"TIMING_INVALID_{timing_state}")

            if int(raw_strategy_vote_count) < 7:
                reasons.append(f"INSUFFICIENT_RAW_VOTES({raw_strategy_vote_count}<7)")

            if int(independent_category_count) < 2:
                reasons.append(f"INSUFFICIENT_INDEPENDENT_CATEGORIES({independent_category_count}<2)")

            gate_state = "PASS" if not reasons else "FAIL"

        # Disagreement formatting
        live_tag = "LIVE_TRADE" if str(live_decision).upper() == "TRADE" else "LIVE_SKIP"
        shadow_tag = f"SHADOW_{gate_state}"
        disagreement = f"{live_tag}_{shadow_tag}"

        return {
            "shadow_high_quality_entry_gate_state": gate_state,
            "shadow_high_quality_entry_gate_reasons": reasons if reasons else ["ALL_CONDITIONS_SATISFIED"],
            "shadow_ml_state": ml_state,
            "shadow_timing_state": timing_state,
            "shadow_raw_vote_count": raw_strategy_vote_count if raw_strategy_vote_count is not None else 0,
            "shadow_independent_category_count": independent_category_count if independent_category_count is not None else 0,
            "shadow_decision": gate_state,
            "live_decision": live_decision,
            "shadow_joint_rule_decision": gate_state,
            "disagreement": disagreement,
        }

    @staticmethod
    def _evaluate_shadow_quality_classification(
        *,
        independent_category_count: Optional[int],
        raw_strategy_vote_count: Optional[int],
        timing_state: Optional[str],
        ml_state: Optional[str],
        live_decision: str,
    ) -> dict[str, Any]:
        """
        Phase 4A: Structural Quality Classification (Architecture B in Shadow / Observe Mode).
        Classifies candidate into: HIGH_QUALITY, MEDIUM_QUALITY, LOW_QUALITY, UNAVAILABLE.
        """
        if (
            independent_category_count is None
            or raw_strategy_vote_count is None
            or timing_state is None
            or str(timing_state).strip().upper() in ("", "UNAVAILABLE")
            or ml_state is None
            or str(ml_state).strip().upper() in ("", "UNAVAILABLE")
        ):
            return {
                "quality_classification": "UNAVAILABLE",
                "quality_classification_reasons": ["TELEMETRY_UNAVAILABLE"],
                "quality_tier": "UNAVAILABLE",
                "quality_reasons": ["TELEMETRY_UNAVAILABLE"],
            }

        is_multi_cat = int(independent_category_count) >= 2
        is_high_votes = int(raw_strategy_vote_count) >= 7
        is_clean_timing = str(timing_state).upper() in ("VALID", "EARLY", "NORMAL")
        is_ml_pos = str(ml_state).upper() == "POSITIVE"

        reasons = []
        if is_multi_cat and is_high_votes and is_clean_timing:
            classification = "HIGH_QUALITY"
            reasons.append("multi_category_confirmation")
            reasons.append("strong_vote_consensus")
            reasons.append("clean_timing")
            if is_ml_pos:
                reasons.append("ml_positive_support")
        elif is_multi_cat and (is_high_votes or is_clean_timing):
            classification = "MEDIUM_QUALITY"
            reasons.append("multi_category_confirmation")
            if not is_clean_timing:
                reasons.append(f"temporary_extension({str(timing_state).lower()})")
            if not is_high_votes:
                reasons.append(f"moderate_vote_consensus({raw_strategy_vote_count}<7)")
            if is_ml_pos:
                reasons.append("ml_positive_support")
        else:
            classification = "LOW_QUALITY"
            if not is_multi_cat:
                reasons.append("single_category_signal")
            if not is_high_votes:
                reasons.append(f"insufficient_vote_consensus({raw_strategy_vote_count}<7)")
            if not is_clean_timing:
                reasons.append(f"timing_{str(timing_state).lower()}")

        return {
            "quality_classification": classification,
            "quality_classification_reasons": reasons,
            "quality_tier": classification,
            "quality_reasons": reasons,
        }

    @staticmethod
    def _can_fire_early(
        *,
        winning: list[dict],
        direction: Direction,
        current_ts: datetime,
        regime: Regime,
        setup,
        winning_summary: WeightedVoteSummary,
    ) -> bool:
        if len(winning) != 1 or direction == Direction.NONE:
            return False
        lead = winning[0]
        lead_conf = float(lead.get("confidence", 0.0))
        setup_strength = float(getattr(setup, "setup_strength", 0.0) or 0.0)
        setup_type = str(getattr(setup, "setup_type", "") or "")
        weighted_score = float(winning_summary.weighted_score or 0.0)
        if lead_conf < EARLY_TRIGGER_MIN_CONFIDENCE:
            if not (
                setup_strength >= RUNNER_EARLY_TRIGGER_STRENGTH_THRESHOLD
                and weighted_score >= RUNNER_EARLY_TRIGGER_WEIGHTED_SCORE_THRESHOLD
                and lead_conf >= max(MIN_STRATEGY_CONF, RUNNER_EARLY_TRIGGER_MIN_CONF)
            ):
                return False
        if regime == Regime.TRENDING:
            time_str = current_ts.strftime("%H:%M")
            if time_str > RUNNER_TREND_PERSIST_MAX_TIME:
                return False
            if time_str > RUNNER_TREND_PERSIST_MID_TIME and not (
                setup_strength >= RUNNER_EARLY_TRIGGER_STRENGTH_HIGH
                and weighted_score >= RUNNER_EARLY_TRIGGER_WEIGHTED_SCORE_HIGH
                and lead_conf >= max(MIN_STRATEGY_CONF + RUNNER_EARLY_TRIGGER_BONUS_CANDLE, RUNNER_EARLY_TRIGGER_HIGH_CONF)
            ):
                return False
        elif regime == Regime.RANGING:
            if current_ts.strftime("%H:%M") > NO_NEW_SIGNAL_AFTER: # using existing NO_NEW_SIGNAL_AFTER
                return False
            if setup_type != "mean_reversion":
                return False
            if setup_strength < RUNNER_REVERSION_STRENGTH_THRESHOLD or weighted_score < RUNNER_REVERSION_WEIGHTED_SCORE_THRESHOLD:
                return False
        else:
            return False
        trending_leads = {
            "ORB",
            "VWAP+EMA",
            "BBSqueeze",
            "Ichimoku",
            "VolumeProfile",
            "SuperTrend+RSI",
            "ADX+PSAR",
        }
        ranging_leads = {
            "BBSqueeze",
            "PriceAction",
            "CPR",
            "LiqSweep",
            "VolumeProfile",
        }
        if regime == Regime.RANGING:
            return lead.get("name") in ranging_leads
        return lead.get("name") in trending_leads

    def _build_trade_preview(
        self,
        *,
        df: pd.DataFrame,
        ltp: float,
        direction: Direction,
        current_ts: datetime,
        msg: Message,
    ) -> str:
        regime = msg.payload.get("regime", Regime.TRENDING.value)
        regime_details = msg.payload.get("regime_details", {}) or {}
        instrument = os.getenv("INSTRUMENT", "SILVERMIC")

        atr = float((df["high"] - df["low"]).tail(RUNNER_EXPECTED_MOVE_ATR_PERIOD).mean()) if len(df) else 0.0
        orb_range = (
            float(self._orb_high - self._orb_low)
            if self._orb_high is not None and self._orb_low is not None
            else 0.0
        )
        expected_move = max(
            atr * RUNNER_EXPECTED_MOVE_ATR_MULT,
            orb_range * RUNNER_EXPECTED_MOVE_ORB_MULT,
            ltp * RUNNER_EXPECTED_MOVE_LTP_MULT
        )

        preview = [
            f"{instrument}={ltp:.2f}",
            f"regime={regime}",
            f"ADX={float(regime_details.get('adx', 0.0)):.1f}",
            f"Chop={float(regime_details.get('chop_index', 0.0)):.1f}",
        ]

        if direction != Direction.NONE:
            is_long = direction.is_long
            target = round(ltp + expected_move if is_long else ltp - expected_move, 2)
            sl = round(ltp - expected_move * RUNNER_SPOT_SL_MULT if is_long else ltp + expected_move * RUNNER_SPOT_SL_MULT, 2)
            preview.extend([
                f"dir={direction.value}",
                f"target={target}",
                f"sl={sl}",
            ])
        return " | ".join(preview)

    @staticmethod
    def _market_context(df: pd.DataFrame) -> dict:
        atr_points, atr_pct = compute_atr(df, period=14)
        last_close = float(df["close"].iloc[-1]) if len(df) else 0.0
        rsi_val = 50.0
        if len(df) >= 15:
            try:
                import pandas_ta as ta
                rsi_s = ta.rsi(df["close"], length=14)
                if rsi_s is not None and not rsi_s.empty:
                    rsi_val = float(rsi_s.iloc[-1])
            except Exception:
                rsi_val = 50.0
        return {
            "atr_points": atr_points,
            "atr_pct": atr_pct,
            "close": round(last_close, 2),
            "rsi_14": round(rsi_val, 2),
        }


    @staticmethod
    def _resolve_timestamp(raw: Union[str, None]) -> datetime:
        if raw:
            ts = datetime.fromisoformat(str(raw))
            return ts if ts.tzinfo else IST.localize(ts)
        return datetime.now(IST)

    #  ELIGIBILITY 

    def _assess_strategy_readiness(
        self,
        meta: StrategyMeta,
        df: pd.DataFrame,
        has_orb: bool,
    ) -> StrategyReadiness:
        n = len(df)
        if n < meta.min_candles:
            return StrategyReadiness(
                ready=False,
                required_candles=meta.min_candles,
                reason=f"need at least {meta.min_candles} candles",
            )
        if meta.requires_live_broker and self._is_backtest:
            return StrategyReadiness(
                ready=False,
                required_candles=meta.min_candles,
                reason="live broker data required",
            )
        if meta.requires_orb and not has_orb:
            return StrategyReadiness(
                ready=False,
                required_candles=meta.min_candles,
                reason="ORB levels not available yet",
            )
        if meta.requires_volume and not _has_meaningful_volume(df, meta.min_candles):
            return StrategyReadiness(
                ready=False,
                required_candles=meta.min_candles,
                reason="volume data unavailable",
            )
        if meta.name == "CPR":
            session_dates = sorted({idx.date() for idx in df.index})
            if len(session_dates) < 2:
                return StrategyReadiness(
                    ready=False,
                    required_candles=meta.min_candles,
                    reason="previous session data not available",
                )
            prev_day = df[[idx.date() == session_dates[-2] for idx in df.index]]
            if len(prev_day) < S8_MIN_PREV_DAY_CANDLES:
                return StrategyReadiness(
                    ready=False,
                    required_candles=meta.min_candles,
                    reason=f"need at least {S8_MIN_PREV_DAY_CANDLES} candles from previous session",
                )
        return StrategyReadiness(ready=True, required_candles=meta.min_candles)

    def _get_eligible(
            self, df: pd.DataFrame, has_orb: bool
    ) -> tuple[list[StrategyMeta], list[dict]]:
        eligible, skipped = [], []
        for m in STRATEGY_REGISTRY:
            readiness = self._assess_strategy_readiness(m, df, has_orb)
            if readiness.ready:
                eligible.append(m)
            else:
                skipped.append({
                    "name": m.name,
                    "reason": readiness.reason,
                    "required_candles": readiness.required_candles,
                })
        return eligible, skipped

    def get_eligibility_report(self, candle_count: int = 60) -> dict:
        df = pd.DataFrame(
            {
                "open": [1.0] * candle_count,
                "high": [1.0] * candle_count,
                "low": [1.0] * candle_count,
                "close": [1.0] * candle_count,
                "volume": [1.0] * candle_count,
            },
            index=pd.date_range("2026-01-01 09:15", periods=candle_count, freq="5min", tz=IST),
        )
        e_live, _ = self._get_eligible(df, True)
        _bt = self._is_backtest
        self._is_backtest = True
        e_bt, s_bt = self._get_eligible(df, True)
        self._is_backtest = _bt
        return {
            "candle_count": candle_count,
            "live_eligible": [s.name for s in e_live],
            "backtest_eligible": [s.name for s in e_bt],
            "skipped_backtest": [s["name"] for s in s_bt],
            "total": len(STRATEGY_REGISTRY),
        }

    def _log_strategy_health_summary(self) -> None:
        if self._scan_count <= 0:
            return
        if self._scan_count % self._STRATEGY_HEALTH_LOG_EVERY != 0:
            return
        inactive = []
        broken = []
        summary = []
        for name, stats in self._strategy_health.items():
            summary.append(
                f"{name}(run={stats['running']},skip={stats['skipped']},err={stats['errors']})"
            )
            if stats["running"] == 0:
                inactive.append(name)
            if stats["errors"] > 0:
                broken.append(name)
        logger.info(
            f"[{self.NAME}] Strategy health @scan={self._scan_count} | "
            f"{' | '.join(summary)}"
        )
        if inactive:
            logger.warning(
                f"[{self.NAME}] Inactive strategies so far (no running state yet): {inactive}"
            )
        if broken:
            logger.warning(
                f"[{self.NAME}] Strategies with execution errors: {broken}"
            )

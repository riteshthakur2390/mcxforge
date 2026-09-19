from __future__ import annotations

"""
agents_code/agent3_ml/filter.py  ML Signal Filter Agent
=========================================================
Key optimisation: ML feature extraction now starts as soon as
CANDLES_READY fires  in PARALLEL with strategy evaluation.

Old pipeline (sequential):
    CANDLES_READY
         Agent2 runs strategies (80ms)
         RAW_SIGNAL fires
         Agent3 extracts features (30ms) + runs ML (55ms) = 85ms
    Total: 80 + 85 = 165ms

New pipeline (parallel):
    CANDLES_READY
         Agent2 runs strategies (80ms)         parallel
         Agent3 pre-extracts features (30ms)  
         RAW_SIGNAL fires
         Agent3 runs ML on pre-extracted features (55ms)
    Total: max(80, 30) + 55 = 135ms   saved 30ms

Also uses MODEL_RETRAINED bus event + background file watcher
for hot-reload without container restart.
"""

import asyncio
import os
import sys
import time
from datetime import date, datetime, time as dt_time
from typing import Union
from pathlib import Path

import pandas as pd
import pytz
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings.ml_filter_thresholds import *
from config.settings import (
    JOURNAL_DIR,
    ML_BLOCKED_STRATEGY_PAIRS,
    LLM_ENABLED,
    ML_LIVE_MIN_EVAL_TRADES,
    ML_LIVE_MIN_PRECISION,
    ML_LIVE_PRECISION_WINDOW,
    ML_FALLBACK_BLOCKED_STRATEGY_PAIRS,
    ML_LOOKBACK_CANDLES,
    ML_FALLBACK_ALLOW_RANGING,
    ML_FALLBACK_MIN_CONFIDENCE,
    ML_MIN_CONFIDENCE,
    ML_MIN_MODEL_CONFIDENCE,
    ML_TREND_PULLBACK_MIN_CONFIDENCE,
    ML_BREAKOUT_MIN_CONFIDENCE,
    ML_MODELS_DIR,
    ML_RETRAIN_DAYS,
    ML_SECONDARY_ALLOWED_STRATEGY_PAIRS,
    ML_SECONDARY_MIN_RAW_CONFIDENCE,
    ML_SECONDARY_MIN_VOTES,
    ML_SECONDARY_THRESHOLD,
    ML_THRESHOLD_OVERRIDE,
    ML_RANK_SOFT_THRESHOLD,
    ML_RANK_HIGH_THRESHOLD,
    ML_RANK_MODEL_WEIGHT,
    ML_RANK_STRATEGY_WEIGHT,
    ML_RANK_VOTE_WEIGHT,
    ML_RANK_REGIME_WEIGHT,
    SIGNAL_APPROVAL_DAILY_BUDGET,
    SIGNAL_REGIME_BUDGETS,
    SIGNAL_SESSION_BUDGETS,
    LIVE_TIMEFRAME, # Changed from SIGNAL_TIMEFRAME
# Dynamic threshold settings  all tunable in settings.py
    ML_RANK_BASE_THRESHOLD,
    ML_RANK_TIER_LOW, ML_RANK_TIER_MID, ML_RANK_TIER_HIGH,
    ML_CALL_BIAS_PENALTY, ML_MORNING_CHOP_PENALTY,
    ML_LATE_SESSION_PENALTY, ML_WEAK_PAIR_PENALTY,
    ML_HIGH_CONSENSUS_RELIEF, ML_TRENDING_RELIEF, ML_STRONG_SETUP_RELIEF,
)
from config.settings.strategy import (
    ALLOW_SUBMIN_VOTE_EARLY_TRIGGER,
    MIN_STRATEGY_VOTES,
)
from core.bus import Message, Topic, get_bus
from ml.features import extract
from ml.model import SignalForgeEnsemble
from utils.llm import TaskType, call_llm_async, call_llm_context_async
from utils.pipeline_logging import log_pipeline_stage
from agents_code.agent3_ml.reduced_budget_evaluator import (
    evaluate_for_reduced_budget,
    reduced_budget_meta as rb_meta,
)

IST = pytz.timezone("Asia/Kolkata")
OPENING_END_TIME = dt_time(11, 0)
REDUCED_BUDGET_MAX_TRADE_INR = float(os.getenv("REDUCED_BUDGET_MAX_TRADE_INR", "15000"))
REDUCED_BUDGET_MIN_VOTES = int(os.getenv("REDUCED_BUDGET_MIN_VOTES", "4"))
REDUCED_BUDGET_MIN_RANK = float(os.getenv("REDUCED_BUDGET_MIN_RANK", "0.40"))
REDUCED_BUDGET_MIN_SETUP = float(os.getenv("REDUCED_BUDGET_MIN_SETUP", "0.50"))
REDUCED_BUDGET_NEAR_ML_DELTA = float(os.getenv("REDUCED_BUDGET_NEAR_ML_DELTA", "0.02"))
REDUCED_BUDGET_NEAR_ML_MIN_RANK = float(
    os.getenv("REDUCED_BUDGET_NEAR_ML_MIN_RANK", "0.39")
)
REDUCED_BUDGET_STRUCTURE_MIN_VOTES = int(
    os.getenv("REDUCED_BUDGET_STRUCTURE_MIN_VOTES", "7")
)
REDUCED_BUDGET_STRUCTURE_MIN_RANK = float(
    os.getenv("REDUCED_BUDGET_STRUCTURE_MIN_RANK", "0.60")
)
REDUCED_BUDGET_STRUCTURE_MIN_PROB = float(
    os.getenv("REDUCED_BUDGET_STRUCTURE_MIN_PROB", "0.60")
)
REDUCED_BUDGET_PRIMARY_MIN_PROB = float(
    os.getenv("REDUCED_BUDGET_PRIMARY_MIN_PROB", "0.30")
)
REDUCED_BUDGET_STRONG_MIN_SETUP = float(
    os.getenv("REDUCED_BUDGET_STRONG_MIN_SETUP", "0.50")
)

if not hasattr(Topic, "MODEL_RETRAINED"):
    Topic.MODEL_RETRAINED = "MODEL_RETRAINED"


def _extract_votes(val) -> int:
    if isinstance(val, dict):
        return sum(1 for v in val.values() if v)
    try:
        return int(val or 0)
    except (TypeError, ValueError):
        return 0

#  STRATEGY TIER CLASSIFICATION 
# HIGH-value strategy combos  these pairs have the most independent signals
_HIGH_VALUE_STRATEGIES = {"Ichimoku", "OIAnalysis", "VolumeProfile", "CPR", "LiqSweep",
                          "GapDirection",   # gap = pre-market context, always high tier
                          "SMC",           # order block + structure = high tier
                          "SkewHunter",    # IV skew = completely independent signal
                          "StrikeMomentum", "GapMomentum", "RangeSpread"}
_EXPERIMENTAL_PRIMARY_STRATEGIES = {"ValueArea", "ADXRising"}
_LEGACY_STRUCTURAL_CONFIRMATIONS = {
    "OIAnalysis",
    "Ichimoku",
    "VolumeProfile",
    "FVG",
    "ORB",
    "CPR",
    "LiqSweep",
    "PriceAction",
    "SMC",
    "GapDirection",
    "SkewHunter",
}
_STABLE_3M_EXECUTION_STRATEGIES = {
    "SuperTrend+RSI",
    "ADX+PSAR",
    "BBSqueeze",
    "VWAP+EMA",
    "FVG",
    "ORB",
    "Ichimoku",
    "OIAnalysis",
    "CPR",
    "VolumeProfile",
    "ExpiryWeek",
}

# WEAK pairs in this market (BBSqueeze+ADX, SuperTrend+ADX are correlated)
# These get weak_pair_penalty because they share ATR/ADX as base indicator
_CORRELATED_PAIRS = {
    frozenset({"BBSqueeze",     "ADX+PSAR"}),
    frozenset({"SuperTrend+RSI","ADX+PSAR"}),
    frozenset({"VWAP+EMA",      "ADX+PSAR"}),
    frozenset({"SuperTrend+RSI","BBSqueeze"}),
    frozenset({"VWAP+EMA",      "BBSqueeze"}),
    frozenset({"UTBot",         "ADX+PSAR"}),
    frozenset({"SuperTrend+RSI","UTBot"}),
}


def _signal_minutes(timestamp_raw: str) -> int | None:
    try:
        ts = pd.Timestamp(timestamp_raw)
        return ts.hour * 60 + ts.minute
    except Exception:
        return None

def _classify_tier(strategies: list[str], votes: int) -> str:
    """Classify signal into HIGH/MID/LOW tier for threshold selection."""
    strat_set = set(strategies)
    if votes >= 3:
        return "HIGH"
    if votes == 2:
        # RCA identified 2-vote signals as high-risk, requiring higher quality
        return "MID"
    if strat_set & _HIGH_VALUE_STRATEGIES:
        return "HIGH"
    pair = frozenset(strat_set)
    if pair in _CORRELATED_PAIRS:
        return "LOW"
    return "MID"


def _dynamic_threshold(
        strategies: list[str],
        direction: str,
        votes: int,
        regime_label: str,
        setup_strength: float,
        timestamp: pd.Timestamp | None,
) -> tuple[float, list[str]]:
    """
    Compute the minimum rank score required to approve this signal.
    Returns (threshold, [applied_tags]) for logging.

    Tags explain WHY the threshold is what it is  same format as live log:
      call_bias_penalty, weak_pair_X, morning_chop_penalty, etc.
    """
    tags: list[str] = []

    #  Base threshold from tier 
    tier = _classify_tier(strategies, votes)
    if tier == "HIGH":
        base = ML_RANK_TIER_HIGH
    elif tier == "MID":
        base = ML_RANK_TIER_MID
    else:
        base = ML_RANK_TIER_LOW
        tags.append("low_tier")

    threshold = base

    #  Penalties (raise threshold = harder to pass) 

    # 1. Call bias: market may have systematic direction
    #    Old value was 0.08 and blocked ALL CALLs during bearish Feb-Mar
    #    New value 0.03 still slightly prefers PUT when bears dominate
    if direction == "BUY_CALL":
        threshold += ML_CALL_BIAS_PENALTY
        tags.append("call_bias_penalty")

    # 2. Correlated pair penalty (less independent = lower confidence in signal)
    strat_set = set(strategies)
    pair_key = frozenset(strat_set)
    if pair_key in _CORRELATED_PAIRS:
        threshold += ML_WEAK_PAIR_PENALTY
        s_sorted = sorted(strat_set)
        tags.append(f"weak_pair_{'_'.join(s[:3].lower().replace('+', '_') for s in s_sorted)}")

    # 3. Session time & regime policy adjustments
    if timestamp is not None:
        try:
            h, m = timestamp.hour, timestamp.minute
            mins = h * 60 + m
            if 9 * 60 + 30 <= mins < 10 * 60:  # opening chop window
                threshold += ML_MORNING_CHOP_PENALTY
                tags.append("morning_chop_penalty")
            elif 22 * 60 + 30 <= mins < 23 * 60 + 15:  # MCX late session
                threshold += ML_LATE_SESSION_PENALTY
                tags.append("late_session_penalty")
        except Exception:
            pass

    try:
        from config.settings.modules.session_policy import get_session_policy
        session_policy = get_session_policy(timestamp)
        if session_policy.session_name == "MORNING":
            threshold += 0.04
            tags.append("session_morning_caution")
        elif session_policy.session_name == "EVENING":
            threshold -= 0.03
            tags.append("session_evening_boost")

        strat_set = set(strategies)
        if any(s in strat_set for s in session_policy.preferred_strategies):
            threshold -= 0.02
            tags.append("session_preferred_strategy")
        if any(s in strat_set for s in session_policy.discouraged_strategies):
            threshold += 0.04
            tags.append("session_discouraged_strategy")
    except Exception:
        pass

    #  Reliefs (lower threshold = easier to pass) 

    # 4. Trending regime relief: in clear trend  lower bar
    if regime_label == "TRENDING":
        threshold -= ML_TRENDING_RELIEF
        tags.append("trending_regime_relief")

    # 5. Strong setup relief: high setup_strength  lower bar
    if 0.72 <= setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_82:
        threshold -= 0.01
        tags.append("healthy_setup_relief")
    if setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_85:
        threshold -= ML_STRONG_SETUP_RELIEF
        tags.append("strong_setup_relief")
    elif setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_82:
        threshold -= 0.02
        tags.append("elite_setup_relief")

    # 6. High consensus relief: 3+ votes  lower bar
    if votes >= 3:
        threshold -= ML_HIGH_CONSENSUS_RELIEF
        tags.append("high_consensus_relief")

    # Strong non-correlated calls were getting clipped by 0.01-0.02 after the
    # recent penalty increase. Give back a small amount only when confluence is
    # decent and the pair is not in the weak/correlated bucket.
    if (
        direction == "BUY_CALL"
        and setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_75
        and votes >= MIN_STRATEGY_VOTES
        and frozenset(set(strategies)) not in _CORRELATED_PAIRS
    ):
        threshold -= 0.02
        tags.append("validated_call_relief")

    # Opening weak correlated pairs were the most consistent long-window drag,
    # but hard entry blocks removed too many good trades. Keep this as a small
    # rank penalty instead of rejecting the signal outright.
    if (
        timestamp is not None
        and 9 * 60 + 15 <= (timestamp.hour * 60 + timestamp.minute) < 10 * 60
        and votes <= 2
        and pair_key in _CORRELATED_PAIRS
        and setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_72
    ):
        threshold += 0.02
        tags.append("opening_weak_pair_penalty")

    # Restore some room for the stronger 3-vote opening signals that the hard
    # strategy-level blocks were clipping.
    if (
        timestamp is not None
        and (timestamp.hour * 60 + timestamp.minute) < 10 * 60
        and votes >= 3
        and setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_72
    ):
        threshold -= 0.01
        tags.append("opening_confluence_relief")

    # Keep the threshold dynamic; a hard floor at 0.48 was pinning most
    # trending PUT setups to the same required rank and nullifying small
    # ranking changes.
    return round(max(ML_RANK_BASE_THRESHOLD, threshold), 4), tags

class MLFilterAgent:
    NAME = "MLFilterAgent"

    @staticmethod
    def _candidate_model_paths(timeframe: str, symbol: str | None = None) -> list[tuple[Path, str, bool]]:
        models_dir = Path(ML_MODELS_DIR)
        requested = str(timeframe or "").strip() or "5minute"
        
        sym_clean = str(symbol or "").strip().lower()
        prefixes = []
        if sym_clean:
            if "silver" in sym_clean:
                prefixes.extend(["silverm", "silvermic"])
            elif "gold" in sym_clean:
                prefixes.extend(["goldm", "gold"])
            elif "crude" in sym_clean:
                prefixes.extend(["crudeoilm", "crudeoil"])
            elif "nat" in sym_clean:
                prefixes.extend(["natgasm", "natgasmini", "naturalgas"])
            else:
                prefixes.append(sym_clean)
        prefixes.extend(["silverm", "silvermic", "commodity", "nifty"])
        # Deduplicate while preserving order
        seen_p = set()
        deduped_prefixes = [p for p in prefixes if not (p in seen_p or seen_p.add(p))]

        candidates: list[tuple[Path, str, bool]] = []

        # 1. Exact match for requested timeframe across priority prefixes
        for prefix in deduped_prefixes:
            path = models_dir / f"{prefix}_{requested}.pkl"
            candidates.append((path, requested, False))

        # 2. Fallbacks
        fallback_order = {
            "1h": ["5minute", "15minute", "1minute"],
            "15minute": ["5minute", "1h", "1minute"],
            "3minute": ["1minute", "5minute"],
            "1minute": ["3minute", "5minute"],
            "5minute": ["15minute", "3minute", "1minute"],
        }.get(requested, ["5minute", "1minute"])

        for fallback_tf in fallback_order:
            for prefix in deduped_prefixes:
                path = models_dir / f"{prefix}_{fallback_tf}.pkl"
                if not any(c[0] == path for c in candidates):
                    candidates.append((path, fallback_tf, True))

        return candidates

    @staticmethod
    def _reduced_budget_meta(reason: str, *, min_ml_prob: float | None = None) -> dict:
        meta = {
            "reduced_budget_lane": True,
            "reduced_budget_reason": reason,
            "max_trade_investment_inr": REDUCED_BUDGET_MAX_TRADE_INR,
        }
        if min_ml_prob is not None:
            meta["reduced_budget_min_ml_prob"] = round(float(min_ml_prob), 4)
        return meta

    @staticmethod
    def _reduced_budget_direction_ok(
        *,
        data: dict,
        direction: str,
        regime_label: str,
    ) -> tuple[bool, str]:
        context = ((data.get("metadata") or {}).get("_context") or {})
        setup = context.get("setup", {}) or {}
        setup_context = setup.get("context", {}) or {}
        market_structure = (
            context.get("market_structure")
            or setup_context.get("market_structure")
            or {}
        )
        entry_validation = market_structure.get("entry_validation", {}) or {}
        direction_validation = (
            (entry_validation.get("by_direction", {}) or {}).get(direction, {})
        )
        if not bool(direction_validation.get("valid", entry_validation.get("valid", True))):
            return False, "invalid_structure_confirmation"
        if entry_validation.get("avoid", False) or entry_validation.get("middle_range", False):
            return False, "weak_entry_structure"

        signal_minutes = _signal_minutes(str(data.get("timestamp", "")))
        if signal_minutes is not None:
            if signal_minutes >= 15 * 60 + 20:
                return False, "late_session_penalty"
            if direction == "BUY_CALL" and 9 * 60 + 30 <= signal_minutes < 10 * 60:
                votes = _extract_votes(data.get("strategy_votes") or data.get("votes"))
                setup_strength = float(data.get("setup_strength") or 0.0)
                if not (votes >= 5 and setup_strength >= 0.50):
                    return False, "morning_chop_call_bias"

        structure_state = market_structure.get("structure_state", {}) or {}
        structure_bias = str(structure_state.get("bias", "UNKNOWN") or "UNKNOWN").upper()
        opposite_bias = (
            direction == "BUY_CALL" and structure_bias == "BEARISH"
        ) or (
            direction == "BUY_PUT" and structure_bias == "BULLISH"
        )
        if opposite_bias:
            return False, f"direction_vs_structure_bias:{structure_bias}"

        regime = str(regime_label or "UNKNOWN").upper()
        if regime == "TRENDING" and structure_bias == "RANGING":
            return False, "trending_regime_without_directional_structure"
        return True, ""

    def __init__(
        self,
        *,
        enable_file_watcher: bool = True,
        force_retrain: bool = False,
        live_parity_date: date | None = None,
    ) -> None:
        self.bus = get_bus()
        self.ensemble = SignalForgeEnsemble()
        self._enable_file_watcher = enable_file_watcher
        self._force_retrain = force_retrain
        self._live_parity_date = live_parity_date
        self._latest_df: Union[pd.DataFrame, None] = None
        self._requested_model_path = Path(ML_MODELS_DIR) / f"nifty_{LIVE_TIMEFRAME}.pkl"
        self._model_path = self._requested_model_path
        self._model_timeframe = LIVE_TIMEFRAME
        self._model_fallback_used = False
        self._model_mtime = 0.0
        # Pre-extracted features cache  populated on CANDLES_READY
        # so ML runs immediately when RAW_SIGNAL arrives
        self._prefetched_features: Union[dict, None] = None
        self._prefetch_conf:  float = 0.0
        self._prefetch_votes: int   = 0
        self._prefetch_dir:   str   = ""
        self._prefetch_df:    Union[pd.DataFrame, None] = None
        self._symbol_prefetched_features: dict[str, dict] = {}
        self._symbol_prefetch_df: dict[str, pd.DataFrame] = {}
        self._symbol_prefetch_dir: dict[str, str] = {}
        self._symbol_regime_info: dict[str, dict] = {}
        self._latest_regime_info: dict = {}
        self._latest_signal_context: dict = {}
        self._latest_strategies: list[str] = []
        self._approval_day = datetime.now(IST).date().isoformat()
        self._ensembles: dict[str, SignalForgeEnsemble] = {}
        self._symbol_dfs: dict[str, pd.DataFrame] = {}
        self._daily_approvals = 0
        self._session_approvals = {k: 0 for k in SIGNAL_SESSION_BUDGETS}
        self._regime_approvals = {k: 0 for k in SIGNAL_REGIME_BUDGETS}
        self._health = {
            "active": False,
            "model_age_days": None,
            "live_precision": None,
            "live_sample_size": 0,
            "retrain_needed": False,
            "retrain_reason": "",
            "daily_approvals": 0,
            "daily_budget": SIGNAL_APPROVAL_DAILY_BUDGET,
            "session_approvals": dict(self._session_approvals),
            "regime_approvals": dict(self._regime_approvals),
        }
        self._load_model()
        # CRITICAL FIX: Detect and reject stale models
        # Only clear ensemble if force_retrain is True. Let load_model handle the rest.
        # This prevents clearing perfectly good models during backtests.
        if self._force_retrain:
            logger.warning(f"[{self.NAME}] Retrain forced - clearing to fallback mode")
            self.ensemble = SignalForgeEnsemble()
        
        # Ensure _health matches current state before evaluating
        self._health["active"] = bool(self.ensemble.is_trained)
        self._evaluate_model_health()


    def register(self) -> None:
        self.bus.subscribe(Topic.CANDLES_READY, self.on_candles_update)
        self.bus.subscribe(Topic.CANDLES_READY, self.on_candles_prefetch)
        self.bus.subscribe(Topic.MARKET_REGIME, self.on_regime_update)
        self.bus.subscribe(Topic.SIGNAL_SUPPRESSED, self.on_regime_update)
        self.bus.subscribe(Topic.RAW_SIGNAL, self.on_raw_signal)
        self.bus.subscribe(Topic.SYSTEM_RESET, self.on_system_reset)
        self.bus.subscribe(Topic.MODEL_RETRAINED, self.on_model_retrained)
        self.bus.subscribe(Topic.PREMARKET_BIAS, self.on_new_session)
        self.bus.subscribe(Topic.POSITION_CLOSED, self.on_position_closed)
        status = (
            f"loaded ({', '.join(self.ensemble.models.keys())})"
            if self.ensemble.is_trained
            else "fallback mode (no model)"
        )
        logger.info(f"[{self.NAME}] Registered | {status}")
        if self._enable_file_watcher:
            asyncio.create_task(self._model_file_watcher())

    #  PRE-FETCH: runs on CANDLES_READY (parallel with strategy eval) 

    async def on_candles_prefetch(self, msg: Message) -> None:
        """
        Called on EVERY candle — pre-extracts features in background per commodity.
        When RAW_SIGNAL fires 80ms later, features are already ready.
        This hides the 30ms feature extraction cost completely.
        """
        candles = msg.payload.get("candles", [])
        sym = str(msg.payload.get("symbol") or os.getenv("COMMODITY", "SILVERM")).upper()
        if not candles or len(candles) < ML_LOOKBACK_CANDLES:
            self._prefetched_features = None
            self._prefetch_df = None
            if hasattr(self, "_symbol_prefetched_features"):
                self._symbol_prefetched_features.pop(sym, None)
            if hasattr(self, "_symbol_prefetch_df"):
                self._symbol_prefetch_df.pop(sym, None)
            return

        if os.environ.get("TRADING_MODE") == "BACKTEST":
            # Skip expensive feature extraction on every candle during backtests.
            # Only keep the df reference for on-demand extraction.
            try:
                df = pd.DataFrame(candles)
                dt_col = "datetime" if "datetime" in df.columns else ("timestamp" if "timestamp" in df.columns else df.columns[0])
                df["datetime"] = pd.to_datetime(df[dt_col])
                clean_df = df.set_index("datetime").sort_index()
                self._prefetch_df = clean_df
                if not hasattr(self, "_symbol_prefetch_df"):
                    self._symbol_prefetch_df = {}
                self._symbol_prefetch_df[sym] = clean_df
            except Exception:
                self._prefetch_df = None
            self._prefetched_features = None
            return

        try:
            df = pd.DataFrame(candles)
            dt_col = "datetime" if "datetime" in df.columns else ("timestamp" if "timestamp" in df.columns else df.columns[0])
            df["datetime"] = pd.to_datetime(df[dt_col])
            clean_df = df.set_index("datetime").sort_index()
            self._prefetch_df = clean_df
            if not hasattr(self, "_symbol_prefetch_df"):
                self._symbol_prefetch_df = {}
            self._symbol_prefetch_df[sym] = clean_df

            # Pre-extract with neutral direction — updated when signal fires
            reg_info = (
                getattr(self, "_symbol_regime_info", {}).get(sym)
                or self._latest_regime_info
            )
            feats = extract(
                df=clean_df,
                conf=0.65,  # placeholder, updated in on_raw_signal
                votes=2,
                direction="BUY_CALL",
                lookback=ML_LOOKBACK_CANDLES,
                regime_info=reg_info,
            )
            self._prefetched_features = feats
            if not hasattr(self, "_symbol_prefetched_features"):
                self._symbol_prefetched_features = {}
            self._symbol_prefetched_features[sym] = feats
            self._prefetch_dir = "BUY_CALL"
            if not hasattr(self, "_symbol_prefetch_dir"):
                self._symbol_prefetch_dir = {}
            self._symbol_prefetch_dir[sym] = "BUY_CALL"
        except Exception as e:
            logger.debug(f"[{self.NAME}] prefetch error for {sym}: {e}")
            self._prefetched_features = None
            if hasattr(self, "_symbol_prefetched_features"):
                self._symbol_prefetched_features.pop(sym, None)

    async def on_regime_update(self, msg: Message) -> None:
        """Keep df and regime info in sync from regime-passed candles too."""
        candles = msg.payload.get("candles", [])
        sym = str(msg.payload.get("symbol") or os.getenv("COMMODITY", "SILVERM")).upper()
        if candles and len(candles) >= ML_LOOKBACK_CANDLES:
            try:
                df = pd.DataFrame(candles)
                df["datetime"] = pd.to_datetime(df["datetime"])
                clean_df = df.set_index("datetime").sort_index()
                self._prefetch_df = clean_df
                if not hasattr(self, "_symbol_prefetch_df"):
                    self._symbol_prefetch_df = {}
                self._symbol_prefetch_df[sym] = clean_df
                if not hasattr(self, "_symbol_dfs"):
                    self._symbol_dfs = {}
                self._symbol_dfs[sym] = clean_df
            except Exception:
                pass
        # capture detailed_regime for ML feature engineering
        regime_details = msg.payload.get("regime_details", msg.payload.get("details", {}))
        reg_info = regime_details.get("detailed_regime", {})
        self._latest_regime_info = reg_info
        if not hasattr(self, "_symbol_regime_info"):
            self._symbol_regime_info = {}
        self._symbol_regime_info[sym] = reg_info

    async def on_system_reset(self, msg: Message) -> None:
        """CRITICAL FIX: Clear all cached state when bus resets (new backtest run)."""
        logger.info(f"[{self.NAME}] System reset - clearing all cached state")
        self._prefetched_features = None
        self._prefetch_df = None
        self._prefetch_conf = 0.0
        self._prefetch_votes = 0
        self._prefetch_dir = ""
        if hasattr(self, "_symbol_prefetched_features"):
            self._symbol_prefetched_features.clear()
        if hasattr(self, "_symbol_prefetch_df"):
            self._symbol_prefetch_df.clear()
        if hasattr(self, "_symbol_prefetch_dir"):
            self._symbol_prefetch_dir.clear()
        if hasattr(self, "_symbol_regime_info"):
            self._symbol_regime_info.clear()
        self._latest_df = None
        self._latest_regime_info = {}
        self._latest_signal_context = {}
        self._latest_strategies = []
        self._call_recovery_watch = {}
        self._daily_approvals = 0
        self._approval_day = datetime.now(IST).date().isoformat()

    async def on_candles_update(self, msg: Message) -> None:
        candles = msg.payload.get("candles", [])
        sym = str(msg.payload.get("symbol") or os.getenv("COMMODITY", "SILVERM")).upper()
        if candles:
            try:
                df = pd.DataFrame(candles)
                dt_col = "datetime" if "datetime" in df.columns else ("timestamp" if "timestamp" in df.columns else df.columns[0])
                df["datetime"] = pd.to_datetime(df[dt_col])
                clean_df = df.set_index("datetime").sort_index()
                self._latest_df = clean_df
                self._symbol_dfs[sym] = clean_df
            except Exception:
                pass

    async def on_new_session(self, msg: Message) -> None:
        ts_raw = msg.payload.get("timestamp")
        day = self._resolve_ts(ts_raw).date().isoformat()
        self._reset_budgets(day)
        self._call_recovery_watch = {}
        self._evaluate_model_health()

    async def on_position_closed(self, _msg: Message) -> None:
        await asyncio.sleep(0.2)
        self._evaluate_model_health()

    @staticmethod
    def _allows_low_consensus_signal(data: dict) -> bool:
        context = ((data.get("metadata") or {}).get("_context") or {})
        return ALLOW_SUBMIN_VOTE_EARLY_TRIGGER and bool(context.get("early_trigger", False))

    @staticmethod
    def _fails_low_quality_rank_gate(
        *,
        data: dict,
        rank_score: float,
        votes: int,
    ) -> tuple[bool, str]:
        # CRITICAL FIX: Check entry validation FIRST before any early returns
        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        market_structure = (
            ((data.get("metadata") or {}).get("_context") or {}).get("market_structure")
            or setup.get("context", {}).get("market_structure")
            or {}
        )
        entry_validation = market_structure.get("entry_validation", {}) or {}
        direction_validation = (
            (entry_validation.get("by_direction", {}) or {}).get(str(data.get("direction", "")), {})
        )
        # CRITICAL: Block ALL invalid entry validation signals - regardless of votes/rank
        if not bool(direction_validation.get("valid", entry_validation.get("valid", True))):
            return True, "ML quality gate: invalid structure confirmation"
        # CRITICAL: Block ALL avoid=True or middle_range=True signals
        if entry_validation.get("avoid", False) or entry_validation.get("middle_range", False):
            return True, "ML quality gate: weak entry structure"

        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        if LIVE_TIMEFRAME == "3minute" and setup_type == "vote_aligned":
            strategies = {
                str(name).strip()
                for name in data.get("strategies_fired", [])
                if str(name).strip()
            }
            independent_context = {
                "OIAnalysis",
                "Ichimoku",
                "VolumeProfile",
                "FVG",
                "ORB",
                "CPR",
                "LiqSweep",
                "PriceAction",
                "SMC",
                "StrikeMomentum",
                "GapMomentum",
                "RangeSpread",
            }
            clean_trend_pair = strategies == {"SuperTrend+RSI", "ADX+PSAR"}
            stable_legacy_stack = bool(strategies & _STABLE_3M_EXECUTION_STRATEGIES) and not (
                strategies & {"ValueArea", "ADXRising", "RangeSpread", "StrikeMomentum", "GapMomentum"}
            )
            has_independent_context = bool(strategies & independent_context)
            if clean_trend_pair:
                if rank_score >= ML_FILTER_RANK_SCORE_THRESH_0_5 and setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_45:
                    return False, ""
                if setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_75 or rank_score < ML_FILTER_RANK_SCORE_THRESH_0_62:
                    return True, (
                        f"ML quality gate: weak 3m trend-pair vote_aligned "
                        f"rank={rank_score:.2f} setup={setup_strength:.2f}"
                    )
            elif stable_legacy_stack and votes >= 2 and rank_score >= ML_FILTER_RANK_SCORE_THRESH_0_5 and setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_48:
                return False, ""
            elif (
                votes < 3
                or setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_82
                or rank_score < ML_FILTER_RANK_SCORE_THRESH_0_72
                or not has_independent_context
            ):
                return True, (
                    f"ML quality gate: 3m vote_aligned noise "
                    f"rank={rank_score:.2f} setup={setup_strength:.2f} "
                    f"votes={votes}"
                )

        if (
            setup_type == "vote_aligned"
            and rank_score < ML_FILTER_RANK_SCORE_THRESH_0_565
            and votes <= 3
            and setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_74
        ):
            return True, (
                f"ML quality gate: marginal vote_aligned rank={rank_score:.3f} "
                f"setup={setup_strength:.2f} votes={votes}"
            )

        if setup_type == "breakout" and rank_score < ML_FILTER_RANK_SCORE_THRESH_0_565:
            return True, (
                f"ML quality gate: low-rank breakout rank={rank_score:.3f} "
                f"setup={setup_strength:.2f} votes={votes}"
            )

        # Allow high-quality signals to pass
        if votes > 2:
            return False, ""
        if rank_score >= ML_FILTER_RANK_SCORE_THRESH_0_55:
            return False, ""

        if setup_strength > ML_FILTER_SETUP_STRENGTH_THRESH_0_78:
            return False, ""

        if setup_type == "breakout" and setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_7:
            context = ((data.get("metadata") or {}).get("_context") or {})
            weighted_vote = context.get("weighted_vote", {}) or {}
            winning_side = (
                weighted_vote.get("put", {})
                if str(data.get("direction", "")) == "BUY_PUT"
                else weighted_vote.get("call", {})
            )
            adx = float(
                context.get("adx", setup.get("context", {}).get("adx", 0.0)) or 0.0
            )
            weighted_score = float(winning_side.get("weighted_score", 0.0) or 0.0)
            if setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_72 and adx >= 28.0 and weighted_score >= 1.80:
                return False, ""
            return True, (
                f"ML quality gate: weak breakout rank={rank_score:.2f} "
                f"setup={setup_strength:.2f}"
            )
        if setup_type == "vote_aligned" and setup_strength <= ML_FILTER_SETUP_STRENGTH_THRESH_0_72:
            return True, (
                f"ML quality gate: low-conviction vote_aligned rank={rank_score:.2f} "
                f"setup={setup_strength:.2f}"
            )
        return False, ""

    @staticmethod
    def _passes_stable_3m_legacy_rescue(
        *,
        data: dict,
        rank_score: float,
        threshold: float,
        votes: int,
    ) -> bool:
        if LIVE_TIMEFRAME != "3minute":
            return False
        if threshold - rank_score > ML_FILTER_RANK_SCORE_THRESH_0_13 or rank_score < ML_FILTER_RANK_SCORE_THRESH_0_35 or votes < 2:
            return False

        context = ((data.get("metadata") or {}).get("_context") or {})
        setup = context.get("setup", {}) or {}
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        direction = str(data.get("direction", "") or "")
        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        experimental = {"ValueArea", "ADXRising", "RangeSpread", "StrikeMomentum", "GapMomentum"}
        if not strategies or strategies & experimental:
            return False

        anchor = context.get("hybrid_5m_anchor", {}) or {}
        anchor_direction = str(anchor.get("direction", "NONE") or "NONE")
        anchor_votes = _extract_votes(anchor.get("votes", 0))
        anchor_ws = float(anchor.get("weighted_score", 0.0) or 0.0)
        anchor_aligned = (
            anchor_direction == direction
            and anchor_votes >= 2
            and anchor_ws >= 1.20
        )

        stable_pair = bool(strategies & {"SuperTrend+RSI", "ADX+PSAR", "BBSqueeze", "VWAP+EMA"})
        structure_pair = bool(strategies & {"FVG", "ORB", "Ichimoku", "OIAnalysis", "CPR", "VolumeProfile"})
        expiry_skew = strategies == {"ADX+PSAR", "SkewHunter", "ExpiryWeek"}

        if expiry_skew and setup_type == "vote_aligned" and rank_score >= ML_FILTER_RANK_SCORE_THRESH_0_44:
            return True
        if direction == "BUY_PUT" and "SkewHunter" in strategies and not expiry_skew:
            return False
        if setup_type == "vote_aligned" and stable_pair and (anchor_aligned or structure_pair):
            return rank_score >= ML_FILTER_RANK_SCORE_THRESH_0_38 and setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_45
        if setup_type == "breakout" and structure_pair and rank_score >= ML_FILTER_RANK_SCORE_THRESH_0_4 and setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_5:
            return True
        return False

    @staticmethod
    def _experimental_strategy_block_reason(
        *,
        data: dict,
        success_prob: float,
        rank_score: float,
    ) -> str:
        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        if setup_type not in {"breakout", "trend_pullback"}:
            return ""

        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        if not (strategies & _EXPERIMENTAL_PRIMARY_STRATEGIES):
            return ""
        if strategies & _LEGACY_STRUCTURAL_CONFIRMATIONS:
            return ""

        if setup_type == "breakout" and (success_prob < ML_FILTER_SUCCESS_PROB_THRESH_0_34 or rank_score < ML_FILTER_RANK_SCORE_THRESH_0_6):
            return (
                "ML quality gate: experimental breakout needs legacy confirmation "
                f"p={success_prob:.2f} rank={rank_score:.2f}"
            )
        if setup_type == "trend_pullback" and success_prob < ML_FILTER_SUCCESS_PROB_THRESH_0_34 and rank_score < ML_FILTER_RANK_SCORE_THRESH_0_6:
            return (
                "ML quality gate: experimental trend_pullback needs stronger ML "
                f"p={success_prob:.2f} rank={rank_score:.2f}"
            )
        return ""

    @staticmethod
    def _hybrid_5m_block_reason(
        *,
        data: dict,
        success_prob: float,
        rank_score: float,
        votes: int,
    ) -> str:
        context = ((data.get("metadata") or {}).get("_context") or {})
        hybrid = context.get("hybrid_5m", {}) or {}
        if not bool(hybrid.get("enabled", False)):
            return ""
        bias = str(hybrid.get("bias", "NEUTRAL") or "NEUTRAL").upper()
        confidence = float(hybrid.get("confidence", 0.0) or 0.0)
        if bias == "NEUTRAL" or confidence < 0.58:
            return ""

        direction = str(data.get("direction", "") or "")
        aligned = (
            (bias == "BULLISH" and direction == "BUY_CALL")
            or (bias == "BEARISH" and direction == "BUY_PUT")
        )
        if aligned:
            return ""

        setup = context.get("setup", {}) or {}
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        strong_counter = (
            success_prob >= ML_FILTER_SUCCESS_PROB_THRESH_0_38
            and rank_score >= ML_FILTER_RANK_SCORE_THRESH_0_64
            and votes >= 3
            and setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_86
        )
        high_consensus = (
            votes >= 6
            and setup_strength >= 0.70
            and rank_score >= PLANNER_RANK_SCORE_THRESH_0_58
        ) or votes >= 8
        if strong_counter or high_consensus:
            return ""
        return (
            "ML quality gate: 3m signal conflicts with 5m confirmation "
            f"bias={bias} conf={confidence:.2f} p={success_prob:.2f} "
            f"rank={rank_score:.2f} setup={setup_strength:.2f} votes={votes}"
        )

    @staticmethod
    def _noisy_3m_stack_block_reason(
        *,
        data: dict,
        success_prob: float,
        rank_score: float,
        votes: int,
    ) -> str:
        if LIVE_TIMEFRAME != "3minute":
            return ""
        context = ((data.get("metadata") or {}).get("_context") or {})
        setup = context.get("setup", {}) or {}
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        direction = str(data.get("direction", "") or "")
        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        experimental = {"ValueArea", "ADXRising", "RangeSpread", "StrikeMomentum", "GapMomentum"}
        proven_structure = {"FVG", "ORB", "Ichimoku", "OIAnalysis", "CPR", "VolumeProfile"}
        anchor = context.get("hybrid_5m_anchor", {}) or {}
        anchor_direction = str(anchor.get("direction", "NONE") or "NONE")
        anchor_votes = _extract_votes(anchor.get("votes", 0))
        anchor_ws = float(anchor.get("weighted_score", 0.0) or 0.0)
        anchor_aligned = (
            anchor_direction == direction
            and anchor_votes >= 2
            and anchor_ws >= 1.20
        )

        if (
            direction == "BUY_PUT"
            and "SkewHunter" in strategies
            and setup_type in {"breakout", "trend_pullback"}
        ):
            if strategies & experimental:
                return (
                    "ML quality gate: noisy 3m SkewHunter+experimental PUT stack "
                    f"setup={setup_type} p={success_prob:.2f} rank={rank_score:.2f} "
                    f"votes={votes}"
                )
            if not anchor_aligned and not (("ExpiryWeek" in strategies) and setup_type == "vote_aligned"):
                return (
                    "ML quality gate: 3m SkewHunter PUT needs aligned 5m anchor "
                    f"anchor={anchor_direction}/{anchor_votes}/{anchor_ws:.2f} "
                    f"setup={setup_type} rank={rank_score:.2f}"
                )
            if setup_type == "breakout" and not (strategies & proven_structure):
                return (
                    "ML quality gate: 3m SkewHunter PUT breakout needs proven structure "
                    f"rank={rank_score:.2f} p={success_prob:.2f}"
                )
        return ""

    #  MAIN GATE 
    async def on_raw_signal(self, msg: Message) -> None:
        data = msg.payload
        direction = str(data.get("direction", "NONE"))
        conf = float(data.get("confidence", 0.0))
        votes = _extract_votes(data.get("votes", 0))
        market_ts = data.get("timestamp", "")

        sig_key = f"{market_ts}:{direction}"
        now_sec = time.time()
        if not hasattr(self, "_seen_raw_signals"):
            self._seen_raw_signals = {}
        self._seen_raw_signals = {k: t for k, t in self._seen_raw_signals.items() if now_sec - t < 30.0}
        if sig_key in self._seen_raw_signals:
            logger.debug(f"[{self.NAME}] Suppressed duplicate RAW_SIGNAL evaluation for {sig_key}")
            return
        self._seen_raw_signals[sig_key] = now_sec
        context = ((data.get("metadata") or {}).get("_context") or {})
        self._latest_signal_context = context
        self._latest_strategies = [
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        ]
        timestamp_str = data.get("timestamp", "")
        # Parse candle timestamp for session-time penalties
        try:
            ts = pd.Timestamp(timestamp_str)
        except Exception:
            ts = None

        sym = str(data.get("symbol") or os.getenv("COMMODITY", "SILVERM")).upper()
        from config.settings.modules.session_policy import (
            get_session_policy,
            is_macro_news_freeze_window,
        )
        session_policy = get_session_policy(ts, symbol=sym)

        # 1. Macro economic news freeze check (EIA Crude/Gas, CPI, NFP, etc.)
        is_frozen, freeze_reason = is_macro_news_freeze_window(ts, sym)
        if is_frozen:
            reason = f"ML gate: {freeze_reason}"
            enriched = {
                **data,
                "ml_confidence": 0.0,
                "ml_success_prob": 0.0,
                "ml_rank_score": 0.0,
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": reason,
                "rejection_reason": reason,
            }
            logger.warning(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {reason}")
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        from instruments.registry import get_instrument_strategy_config
        inst_strat_cfg = get_instrument_strategy_config(sym)
        min_required_votes = int(inst_strat_cfg.get("min_votes", session_policy.min_strategy_votes))
        early_trigger = bool(context.get("early_trigger", False))
        
        # Sub-threshold consensus (< 4 votes, or < 3 if early trigger on alpha setup) is noise
        cutoff_votes = 3 if early_trigger else min(4, min_required_votes)
        if votes < cutoff_votes:
            logger.debug(f"[{self.NAME}] Dropping sub-threshold signal for {sym} (votes {votes} < {cutoff_votes})")
            return

        if votes < min_required_votes and not self._allows_low_consensus_signal(data):
            reason = f"ML gate [{session_policy.session_name} | {sym}]: votes {votes} < required {min_required_votes}"
            enriched = {
                **data,
                "ml_confidence": 0.0,
                "ml_success_prob": 0.0,
                "ml_rank_score": 0.0,
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": self._decision_reason(
                    decision_type="RULE",
                    success_prob=0.0,
                    rank_score=0.0,
                    rank_tier="LOW",
                    threshold=ML_RANK_SOFT_THRESHOLD,
                    accepted=False,
                    extra_reason=reason,
                ),
                "rejection_reason": reason,
            }
            logger.info(
                f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {reason}"
            )
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
                early_trigger=early_trigger,
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        try:
            success_prob, decision_type = self._score(
                conf=conf,
                votes=votes,
                direction=direction,
                timestamp=ts,
                symbol=sym,
            )
        except TypeError:
            try:
                success_prob, decision_type = self._score(conf, votes, direction, symbol=sym)
            except TypeError:
                success_prob, decision_type = self._score(conf, votes, direction)
        rank_score, rank_meta = self._rank_signal(
            data=data,
            success_prob=success_prob,
            raw_conf=conf,
            votes=votes,
            decision_type=decision_type,
        )

        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_ctx = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        setup_strength_for_reduced = float(data.get("setup_strength") or setup.get("setup_strength") or setup_ctx.get("setup_strength") or 0.5)
        regime_label = str((self._latest_regime_info or {}).get("label", "RANGING")).upper()

        # Dynamic instrument-specific ML confidence threshold
        required_conf = float(inst_strat_cfg.get("min_ml_confidence", session_policy.ml_min_confidence))
        if setup_type == "trend_pullback":
            if votes < 4:
                required_conf = max(required_conf, ML_TREND_PULLBACK_MIN_CONFIDENCE)
        elif setup_type == "breakout":
            required_conf = max(required_conf, ML_BREAKOUT_MIN_CONFIDENCE)

        if decision_type == "ML" and success_prob < required_conf:
            reason = f"ML gate [{session_policy.session_name} | {sym}]: model confidence {success_prob:.3f} < minimum required {required_conf:.2f} for {setup_type}"
            setup_strength_for_reduced = float(data.get("setup_strength") or setup.get("setup_strength") or 0.5)
            # ── Last-chance: standalone reduced budget evaluator ──────────
            from agents_code.agent3_ml.reduced_budget_evaluator import (
                evaluate_for_reduced_budget,
                reduced_budget_meta as rb_meta,
            )
            rb_approved, rb_reason = evaluate_for_reduced_budget(
                data=data,
                direction=direction,
                votes=votes,
                success_prob=success_prob,
                rank_score=rank_score,
                setup_strength=setup_strength_for_reduced,
                regime_label=str((self._latest_regime_info or {}).get("label", "RANGING")).upper(),
                quality_threshold=quality_threshold if 'quality_threshold' in dir() else 0.45,
                required_conf=required_conf,
                rejection_source="ML_CONFIDENCE",
                entry_premium=float(data.get("est_premium") or data.get("entry_premium_est") or 0) or None,
            )
            if rb_approved:
                enriched = {
                    **data,
                    "ml_confidence": round(success_prob, 4),
                    "ml_success_prob": round(success_prob, 4),
                    "ml_rank_score": round(rank_score, 4),
                    "ml_rank_tier": rank_meta["tier"],
                    "ml_soft_pass": bool(rank_meta["soft_pass"]),
                    "ml_rank_components": rank_meta["components"],
                    "ml_decision": f"{decision_type}_REDUCED_BUDGET_ML_NEAR_MISS",
                    "ml_decision_reason": self._decision_reason(
                        decision_type=decision_type,
                        success_prob=success_prob,
                        rank_score=rank_score,
                        rank_tier=rank_meta["tier"],
                        threshold=required_conf,
                        accepted=True,
                        extra_reason=rb_reason,
                    ),
                    "ml_approved": True,
                    "early_trigger": early_trigger,
                    "metadata": {
                        **(data.get("metadata") or {}),
                        **rb_meta(rb_reason, min_ml_prob=0.06),
                    },
                }
                logger.info(
                    f"[{self.NAME}] [OK] REDUCED_BUDGET_ML | market_ts={market_ts} | "
                    f"p={success_prob:.2f} | rank={rank_score:.2f} | votes={votes} | "
                    f"setup={setup_strength_for_reduced:.2f}"
                )
                log_pipeline_stage(
                    self.NAME,
                    "ml_filtering_ranking",
                    "passed",
                    market_ts=market_ts,
                    direction=direction,
                    confidence=f"{conf:.2f}",
                    votes=votes,
                    prob=f"{success_prob:.2f}",
                    rank=f"{rank_score:.2f}",
                    tier=rank_meta["tier"],
                    decision="REDUCED_BUDGET_ML",
                    reduced_budget=f"{REDUCED_BUDGET_MAX_TRADE_INR:.0f}",
                    reason=rb_reason,
                )
                await self.bus.publish(Topic.SIGNAL_APPROVED, enriched, self.NAME)
                if LLM_ENABLED:
                    asyncio.create_task(self._explain(enriched, success_prob, rank_score))
                return
            enriched = {
                **data,
                "ml_confidence": round(success_prob, 4),
                "ml_success_prob": round(success_prob, 4),
                "ml_rank_score": round(rank_score, 4),
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": self._decision_reason(
                    decision_type=decision_type,
                    success_prob=success_prob,
                    rank_score=rank_score,
                    rank_tier=rank_meta["tier"],
                    threshold=required_conf,
                    accepted=False,
                    extra_reason=reason,
                ),
                "rejection_reason": reason,
            }
            logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {reason}")
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
                prob=f"{success_prob:.2f}",
                rank=f"{rank_score:.2f}",
                tier=rank_meta["tier"],
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        if decision_type == "FALLBACK":
            signal_day = None
            if ts is not None:
                try:
                    signal_day = pd.Timestamp(ts).date()
                except Exception:
                    signal_day = None
            allow_historical_fallback = (
                self._live_parity_date is not None
                and signal_day is not None
                and signal_day != self._live_parity_date
            )
            ens = self._get_ensemble_for_symbol(sym)
            if ens.is_trained and not allow_historical_fallback:
                reason = "ML fallback disabled while trained model is available"
                enriched = {
                    **data,
                    "ml_confidence": round(success_prob, 4),
                    "ml_success_prob": round(success_prob, 4),
                    "ml_rank_score": round(rank_score, 4),
                    "ml_approved": False,
                    "ml_decision": "RULE_REJECT",
                    "ml_decision_reason": self._decision_reason(
                        decision_type=decision_type,
                        success_prob=success_prob,
                        rank_score=rank_score,
                        rank_tier=rank_meta["tier"],
                        threshold=ML_RANK_SOFT_THRESHOLD,
                        accepted=False,
                        extra_reason=reason,
                    ),
                    "rejection_reason": reason,
                }
                logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {reason}")
                log_pipeline_stage(
                    self.NAME,
                    "ml_filtering_ranking",
                    "filtered",
                    reason=reason,
                    market_ts=market_ts,
                    direction=direction,
                    confidence=f"{conf:.2f}",
                    votes=votes,
                    prob=f"{success_prob:.2f}",
                    rank=f"{rank_score:.2f}",
                    tier=rank_meta["tier"],
                )
                await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
                return
            fallback_ok, fallback_reason = self._passes_fallback_gate(data, fallback_conf=conf)
            if not fallback_ok:
                reason = fallback_reason
                enriched = {
                    **data,
                    "ml_confidence": round(success_prob, 4),
                    "ml_success_prob": round(success_prob, 4),
                    "ml_rank_score": round(rank_score, 4),
                    "ml_approved": False,
                    "ml_decision": "RULE_REJECT",
                    "ml_decision_reason": self._decision_reason(
                        decision_type=decision_type,
                        success_prob=success_prob,
                        rank_score=rank_score,
                        rank_tier=rank_meta["tier"],
                        threshold=ML_RANK_SOFT_THRESHOLD,
                        accepted=False,
                        extra_reason=reason,
                    ),
                    "rejection_reason": reason,
                }
                logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {reason}")
                log_pipeline_stage(
                    self.NAME,
                    "ml_filtering_ranking",
                    "filtered",
                    reason=reason,
                    market_ts=market_ts,
                    direction=direction,
                    confidence=f"{conf:.2f}",
                    votes=votes,
                    prob=f"{success_prob:.2f}",
                    rank=f"{rank_score:.2f}",
                    tier=rank_meta["tier"],
                )
                await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
                return

        blocked_reason = self._blocked_strategy_reason(data=data)
        if blocked_reason:
            reason = blocked_reason
            enriched = {
                **data,
                "ml_confidence": round(success_prob, 4),
                "ml_success_prob": round(success_prob, 4),
                "ml_rank_score": round(rank_score, 4),
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": self._decision_reason(
                    decision_type=decision_type,
                    success_prob=success_prob,
                    rank_score=rank_score,
                    rank_tier=rank_meta["tier"],
                    threshold=ML_RANK_SOFT_THRESHOLD,
                    accepted=False,
                    extra_reason=reason,
                ),
                "rejection_reason": reason,
            }
            logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {reason}")
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
                prob=f"{success_prob:.2f}",
                rank=f"{rank_score:.2f}",
                tier=rank_meta["tier"],
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        setup_ctx = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        quality_threshold, threshold_tags = _dynamic_threshold(
            strategies=self._latest_strategies,
            direction=direction,
            votes=votes,
            regime_label=str((self._latest_regime_info or {}).get("label", "RANGING")).upper(),
            setup_strength=float(setup_ctx.get("setup_strength", 0.5) or 0.5),
            timestamp=ts,
        )
        adaptive_gates = context.get("adaptive_gates", {}) or {}
        if (
            early_trigger
            and bool(adaptive_gates.get("loosened", False))
            and int(adaptive_gates.get("min_votes", MIN_STRATEGY_VOTES) or MIN_STRATEGY_VOTES) < MIN_STRATEGY_VOTES
        ):
            quality_threshold = max(ML_RANK_BASE_THRESHOLD, quality_threshold - 0.04)
            threshold_tags.append("adaptive_gate_relief")
        quality_reason = f" ({','.join(threshold_tags)})" if threshold_tags else ""
        primary_pass = rank_score >= quality_threshold
        secondary_pass = False
        if not primary_pass:
            secondary_pass = self._passes_secondary_ml_gate(
                data=data,
                ml_conf=success_prob,
                raw_conf=conf,
                votes=votes,
            )
        near_miss_pass = False
        if not (primary_pass or secondary_pass):
            near_miss_pass = self._passes_near_miss_gate(
                data=data,
                rank_score=rank_score,
                threshold=quality_threshold,
                ml_conf=success_prob,
                raw_conf=conf,
                votes=votes,
            )
        stable_3m_legacy_rescue = False
        if not (primary_pass or secondary_pass or near_miss_pass):
            stable_3m_legacy_rescue = self._passes_stable_3m_legacy_rescue(
                data=data,
                rank_score=rank_score,
                threshold=quality_threshold,
                votes=votes,
            )
        recovery_pass = False
        if not (primary_pass or secondary_pass or near_miss_pass or stable_3m_legacy_rescue):
            recovery_pass = self._passes_call_recovery_gate(
                data=data,
                rank_score=rank_score,
                threshold=quality_threshold,
                ml_conf=success_prob,
                raw_conf=conf,
                votes=votes,
            )
        if not (primary_pass or secondary_pass or near_miss_pass or stable_3m_legacy_rescue or recovery_pass):
            reason = (
                f"ML gate: rank {rank_score:.3f} < required {quality_threshold:.3f}"
                f"{quality_reason}"
            )
            setup_strength_for_reduced = float(data.get("setup_strength") or setup_ctx.get("setup_strength") or 0.5)
            # ── Last-chance: standalone reduced budget evaluator ──────────
            from agents_code.agent3_ml.reduced_budget_evaluator import (
                evaluate_for_reduced_budget,
                reduced_budget_meta as rb_meta,
            )
            rb_approved, rb_reason = evaluate_for_reduced_budget(
                data=data,
                direction=direction,
                votes=votes,
                success_prob=success_prob,
                rank_score=rank_score,
                setup_strength=setup_strength_for_reduced,
                regime_label=str((self._latest_regime_info or {}).get("label", "RANGING")).upper(),
                quality_threshold=quality_threshold,
                required_conf=required_conf,
                rejection_source="RANK",
                entry_premium=float(data.get("est_premium") or data.get("entry_premium_est") or 0) or None,
            )
            if rb_approved:
                enriched = {
                    **data,
                    "ml_confidence": round(success_prob, 4),
                    "ml_success_prob": round(success_prob, 4),
                    "ml_rank_score": round(rank_score, 4),
                    "ml_rank_tier": rank_meta["tier"],
                    "ml_soft_pass": bool(rank_meta["soft_pass"]),
                    "ml_rank_components": rank_meta["components"],
                    "ml_decision": f"{decision_type}_RANK_REDUCED_BUDGET",
                    "ml_decision_reason": self._decision_reason(
                        decision_type=decision_type,
                        success_prob=success_prob,
                        rank_score=rank_score,
                        rank_tier=rank_meta["tier"],
                        threshold=quality_threshold,
                        accepted=True,
                        extra_reason=rb_reason,
                    ),
                    "ml_approved": True,
                    "early_trigger": early_trigger,
                    "metadata": {
                        **(data.get("metadata") or {}),
                        **rb_meta(rb_reason, min_ml_prob=0.06),
                    },
                }
                logger.info(
                    f"[{self.NAME}] [OK] REDUCED_BUDGET_RANK | market_ts={market_ts} | "
                    f"p={success_prob:.2f} | rank={rank_score:.2f} | votes={votes} | "
                    f"setup={setup_strength_for_reduced:.2f}"
                )
                log_pipeline_stage(
                    self.NAME,
                    "ml_filtering_ranking",
                    "passed",
                    market_ts=market_ts,
                    direction=direction,
                    confidence=f"{conf:.2f}",
                    votes=votes,
                    prob=f"{success_prob:.2f}",
                    rank=f"{rank_score:.2f}",
                    tier=rank_meta["tier"],
                    decision="REDUCED_BUDGET_RANK",
                    reduced_budget=f"{REDUCED_BUDGET_MAX_TRADE_INR:.0f}",
                    reason=rb_reason,
                )
                await self.bus.publish(Topic.SIGNAL_APPROVED, enriched, self.NAME)
                if LLM_ENABLED:
                    asyncio.create_task(self._explain(enriched, success_prob, rank_score))
                return
            enriched = {
                **data,
                "ml_confidence": round(success_prob, 4),
                "ml_success_prob": round(success_prob, 4),
                "ml_rank_score": round(rank_score, 4),
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": self._decision_reason(
                    decision_type=decision_type,
                    success_prob=success_prob,
                    rank_score=rank_score,
                    rank_tier=rank_meta["tier"],
                    threshold=quality_threshold,
                    accepted=False,
                    extra_reason=reason,
                ),
                "rejection_reason": reason,
            }
            logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {reason}")
            self._record_call_recovery_watch(
                data=data,
                rank_score=rank_score,
                threshold=quality_threshold,
                raw_conf=conf,
                votes=votes,
                reason=reason,
            )
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
                prob=f"{success_prob:.2f}",
                rank=f"{rank_score:.2f}",
                tier=rank_meta["tier"],
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        noisy_3m_reason = ""
        if not recovery_pass:
            noisy_3m_reason = self._noisy_3m_stack_block_reason(
                data=data,
                success_prob=success_prob,
                rank_score=rank_score,
                votes=votes,
            )
        if noisy_3m_reason:
            enriched = {
                **data,
                "ml_confidence": round(success_prob, 4),
                "ml_success_prob": round(success_prob, 4),
                "ml_rank_score": round(rank_score, 4),
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": self._decision_reason(
                    decision_type=decision_type,
                    success_prob=success_prob,
                    rank_score=rank_score,
                    rank_tier=rank_meta["tier"],
                    threshold=quality_threshold,
                    accepted=False,
                    extra_reason=noisy_3m_reason,
                ),
                "rejection_reason": noisy_3m_reason,
            }
            logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {noisy_3m_reason}")
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=noisy_3m_reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
                prob=f"{success_prob:.2f}",
                rank=f"{rank_score:.2f}",
                tier=rank_meta["tier"],
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        experimental_block_reason = ""
        if not recovery_pass:
            experimental_block_reason = self._experimental_strategy_block_reason(
                data=data,
                success_prob=success_prob,
                rank_score=rank_score,
            )
        if experimental_block_reason:
            enriched = {
                **data,
                "ml_confidence": round(success_prob, 4),
                "ml_success_prob": round(success_prob, 4),
                "ml_rank_score": round(rank_score, 4),
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": self._decision_reason(
                    decision_type=decision_type,
                    success_prob=success_prob,
                    rank_score=rank_score,
                    rank_tier=rank_meta["tier"],
                    threshold=quality_threshold,
                    accepted=False,
                    extra_reason=experimental_block_reason,
                ),
                "rejection_reason": experimental_block_reason,
            }
            logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {experimental_block_reason}")
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=experimental_block_reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
                prob=f"{success_prob:.2f}",
                rank=f"{rank_score:.2f}",
                tier=rank_meta["tier"],
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        hybrid_block_reason = ""
        if not recovery_pass:
            hybrid_block_reason = self._hybrid_5m_block_reason(
                data=data,
                success_prob=success_prob,
                rank_score=rank_score,
                votes=votes,
            )
        if hybrid_block_reason:
            from agents_code.agent3_ml.reduced_budget_evaluator import (
                evaluate_for_reduced_budget,
                reduced_budget_meta as rb_meta,
            )
            rb_approved, rb_reason = evaluate_for_reduced_budget(
                data=data,
                direction=direction,
                votes=votes,
                success_prob=success_prob,
                rank_score=rank_score,
                setup_strength=setup_strength_for_reduced,
                regime_label=str((self._latest_regime_info or {}).get("label", "RANGING")).upper(),
                quality_threshold=quality_threshold,
                required_conf=required_conf,
                rejection_source="ML_CONFIDENCE",
                entry_premium=float(data.get("est_premium") or data.get("entry_premium_est") or 0) or None,
            )
            if rb_approved:
                enriched = {
                    **data,
                    "ml_confidence": round(success_prob, 4),
                    "ml_success_prob": round(success_prob, 4),
                    "ml_rank_score": round(rank_score, 4),
                    "ml_rank_tier": rank_meta["tier"],
                    "ml_soft_pass": bool(rank_meta["soft_pass"]),
                    "ml_rank_components": rank_meta["components"],
                    "ml_decision": f"{decision_type}_QUALITY_REDUCED_BUDGET",
                    "ml_decision_reason": self._decision_reason(
                        decision_type=decision_type,
                        success_prob=success_prob,
                        rank_score=rank_score,
                        rank_tier=rank_meta["tier"],
                        threshold=quality_threshold,
                        accepted=True,
                        extra_reason=rb_reason,
                    ),
                    "ml_approved": True,
                    "early_trigger": early_trigger,
                    "metadata": {
                        **(data.get("metadata") or {}),
                        **rb_meta(rb_reason, min_ml_prob=0.06),
                    },
                }
                logger.info(
                    f"[{self.NAME}] [OK] REDUCED_BUDGET_ML_CONFIDENCE | market_ts={market_ts} | "
                    f"p={success_prob:.2f} | rank={rank_score:.2f} | votes={votes} | "
                    f"setup={setup_strength_for_reduced:.2f} | gate={hybrid_block_reason}"
                )
                log_pipeline_stage(
                    self.NAME,
                    "ml_filtering_ranking",
                    "passed",
                    reason=f"reduced_budget_ml_confidence_gate: {rb_reason}",
                    market_ts=market_ts,
                    direction=direction,
                )
                await self.bus.publish(Topic.SIGNAL_APPROVED, enriched, self.NAME)
                if LLM_ENABLED:
                    asyncio.create_task(self._explain(enriched, success_prob, rank_score))
                return

            enriched = {
                **data,
                "ml_confidence": round(success_prob, 4),
                "ml_success_prob": round(success_prob, 4),
                "ml_rank_score": round(rank_score, 4),
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": self._decision_reason(
                    decision_type=decision_type,
                    success_prob=success_prob,
                    rank_score=rank_score,
                    rank_tier=rank_meta["tier"],
                    threshold=quality_threshold,
                    accepted=False,
                    extra_reason=hybrid_block_reason,
                ),
                "rejection_reason": hybrid_block_reason,
            }
            logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {hybrid_block_reason}")
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=hybrid_block_reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
                prob=f"{success_prob:.2f}",
                rank=f"{rank_score:.2f}",
                tier=rank_meta["tier"],
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        low_quality_block, low_quality_reason = (False, "")
        if not (recovery_pass or stable_3m_legacy_rescue):
            low_quality_block, low_quality_reason = self._fails_low_quality_rank_gate(
                data=data,
                rank_score=rank_score,
                votes=votes,
            )
        if low_quality_block:
            # ── Last-chance: reduced budget evaluator for quality-blocked signals ──
            from agents_code.agent3_ml.reduced_budget_evaluator import (
                evaluate_for_reduced_budget,
                reduced_budget_meta as rb_meta,
            )
            rb_approved, rb_reason = evaluate_for_reduced_budget(
                data=data,
                direction=direction,
                votes=votes,
                success_prob=success_prob,
                rank_score=rank_score,
                setup_strength=setup_strength_for_reduced,
                regime_label=str((self._latest_regime_info or {}).get("label", "RANGING")).upper(),
                quality_threshold=quality_threshold,
                required_conf=required_conf,
                rejection_source="QUALITY_GATE",
                entry_premium=float(data.get("est_premium") or data.get("entry_premium_est") or 0) or None,
            )
            if rb_approved:
                enriched = {
                    **data,
                    "ml_confidence": round(success_prob, 4),
                    "ml_success_prob": round(success_prob, 4),
                    "ml_rank_score": round(rank_score, 4),
                    "ml_rank_tier": rank_meta["tier"],
                    "ml_soft_pass": bool(rank_meta["soft_pass"]),
                    "ml_rank_components": rank_meta["components"],
                    "ml_decision": f"{decision_type}_QUALITY_REDUCED_BUDGET",
                    "ml_decision_reason": self._decision_reason(
                        decision_type=decision_type,
                        success_prob=success_prob,
                        rank_score=rank_score,
                        rank_tier=rank_meta["tier"],
                        threshold=quality_threshold,
                        accepted=True,
                        extra_reason=rb_reason,
                    ),
                    "ml_approved": True,
                    "early_trigger": early_trigger,
                    "metadata": {
                        **(data.get("metadata") or {}),
                        **rb_meta(rb_reason, min_ml_prob=0.06),
                    },
                }
                logger.info(
                    f"[{self.NAME}] [OK] REDUCED_BUDGET_QUALITY | market_ts={market_ts} | "
                    f"p={success_prob:.2f} | rank={rank_score:.2f} | votes={votes} | "
                    f"setup={setup_strength_for_reduced:.2f} | gate={low_quality_reason}"
                )
                log_pipeline_stage(
                    self.NAME,
                    "ml_filtering_ranking",
                    "passed",
                    reason=f"reduced_budget_quality_gate: {rb_reason}",
                    market_ts=market_ts,
                    direction=direction,
                )
                await self.bus.publish(Topic.SIGNAL_APPROVED, enriched, self.NAME)
                if LLM_ENABLED:
                    asyncio.create_task(self._explain(enriched, success_prob, rank_score))
                return
            enriched = {
                **data,
                "ml_confidence": round(success_prob, 4),
                "ml_success_prob": round(success_prob, 4),
                "ml_rank_score": round(rank_score, 4),
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": self._decision_reason(
                    decision_type=decision_type,
                    success_prob=success_prob,
                    rank_score=rank_score,
                    rank_tier=rank_meta["tier"],
                    threshold=quality_threshold,
                    accepted=False,
                    extra_reason=low_quality_reason,
                ),
                "rejection_reason": low_quality_reason,
            }
            logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {low_quality_reason}")
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=low_quality_reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
                prob=f"{success_prob:.2f}",
                rank=f"{rank_score:.2f}",
                tier=rank_meta["tier"],
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        loss_pattern_reason = ""
        if not recovery_pass:
            loss_pattern_reason = self._loss_pattern_block_reason(
                data=data,
                rank_score=rank_score,
                votes=votes,
            )
        if loss_pattern_reason:
            enriched = {
                **data,
                "ml_confidence": round(success_prob, 4),
                "ml_success_prob": round(success_prob, 4),
                "ml_rank_score": round(rank_score, 4),
                "ml_approved": False,
                "ml_decision": "RULE_REJECT",
                "ml_decision_reason": self._decision_reason(
                    decision_type=decision_type,
                    success_prob=success_prob,
                    rank_score=rank_score,
                    rank_tier=rank_meta["tier"],
                    threshold=quality_threshold,
                    accepted=False,
                    extra_reason=loss_pattern_reason,
                ),
                "rejection_reason": loss_pattern_reason,
            }
            logger.info(f"[{self.NAME}] [X] REJECTED | market_ts={market_ts} | {loss_pattern_reason}")
            log_pipeline_stage(
                self.NAME,
                "ml_filtering_ranking",
                "filtered",
                reason=loss_pattern_reason,
                market_ts=market_ts,
                direction=direction,
                confidence=f"{conf:.2f}",
                votes=votes,
                prob=f"{success_prob:.2f}",
                rank=f"{rank_score:.2f}",
                tier=rank_meta["tier"],
            )
            await self.bus.publish(Topic.SIGNAL_REJECTED, enriched, self.NAME)
            return

        enriched = {
            **data,
            "ml_confidence": round(success_prob, 4),
            "ml_success_prob": round(success_prob, 4),
            "ml_rank_score": round(rank_score, 4),
            "ml_rank_tier": rank_meta["tier"],
            "ml_soft_pass": bool(rank_meta["soft_pass"]),
            "ml_rank_components": rank_meta["components"],
            "ml_decision": f"{decision_type}_RANK{'_SECONDARY' if secondary_pass and not primary_pass else ''}",
            "ml_decision_reason": self._decision_reason(
                decision_type=decision_type,
                success_prob=success_prob,
                rank_score=rank_score,
                rank_tier=rank_meta["tier"],
                threshold=quality_threshold,
                accepted=True,
                extra_reason=(
                    "secondary_lane_override"
                    if secondary_pass and not primary_pass
                    else (
                        "call_continuation_recovery"
                        if recovery_pass and not primary_pass
                        else (
                            "stable_3m_legacy_rescue"
                            if stable_3m_legacy_rescue and not primary_pass
                            else ("near_miss_strong_setup" if near_miss_pass and not primary_pass else "primary_rank_pass")
                        )
                    )
                ),
            ),
            "ml_approved": True,
            "early_trigger": early_trigger,
        }
        logger.info(
            f"[{self.NAME}] [OK] {decision_type}_RANK | market_ts={market_ts} | "
            f"p={success_prob:.2f} | rank={rank_score:.2f} | tier={rank_meta['tier']} | "
            f"dir={direction}"
        )
        log_pipeline_stage(
            self.NAME,
            "ml_filtering_ranking",
            "passed",
            market_ts=market_ts,
            direction=direction,
            confidence=f"{conf:.2f}",
            votes=votes,
            prob=f"{success_prob:.2f}",
            rank=f"{rank_score:.2f}",
            tier=rank_meta["tier"],
            decision=f"{decision_type}_RANK",
            regime=str((self._latest_regime_info or {}).get("label", "RANGING")),
        )
        await self.bus.publish(Topic.SIGNAL_APPROVED, enriched, self.NAME)
        if LLM_ENABLED:
            asyncio.create_task(self._explain(enriched, success_prob, rank_score))

    #  RANK SCORE COMPUTATION 

    def _compute_rank(self, conf: float, votes: int, direction: str) -> float:
        """
        Compute the signal's rank score.

        When model is trained: uses ML predict_proba blended with quality score.
        Fallback: uses strategy confidence directly.
        """
        try:
            df = self._prefetch_df
            if df is None or len(df) < ML_LOOKBACK_CANDLES:
                return conf

            features = extract(
                df=df,
                conf=conf,
                votes=votes,
                direction=direction,
                lookback=ML_LOOKBACK_CANDLES,
                regime_info=self._latest_regime_info,
            )
            if features is None:
                return conf

            ens = self._get_ensemble_for_symbol(symbol)
            if ens.is_trained:
                ml_prob = ens.predict_proba(features)
                # Blend: 60% ML probability + 40% strategy confidence
                # This prevents ML from being the sole arbiter when undertrained
                return round(0.60 * ml_prob + 0.40 * conf, 4)
            else:
                return conf
        except Exception as e:
            logger.debug(f"[{self.NAME}] rank score error: {e}")
            return conf

    def _score(
        self,
        conf: float,
        votes: int,
        direction: str,
        timestamp: pd.Timestamp | None = None,
        symbol: str | None = None,
    ) -> tuple[float, str]:
        sym = str(symbol or os.getenv("COMMODITY", "SILVERM")).upper()
        ens = self._get_ensemble_for_symbol(sym)
        target_df = None
        if hasattr(self, "_symbol_dfs") and self._symbol_dfs.get(sym) is not None:
            target_df = self._symbol_dfs.get(sym)
        elif hasattr(self, "_symbol_prefetch_df") and self._symbol_prefetch_df.get(sym) is not None:
            target_df = self._symbol_prefetch_df.get(sym)
        else:
            target_df = self._latest_df

        if (
            not ens.is_trained
            or target_df is None
            or len(target_df) < ML_LOOKBACK_CANDLES
        ):
            return conf, "FALLBACK"
        try:
            features = self._features_for_signal(
                conf=conf,
                votes=votes,
                direction=direction,
                symbol=sym,
            )
            if features is None:
                return conf, "FALLBACK"

            return ens.predict_proba(features), "ML"

        except Exception as e:
            logger.debug(f"[{self.NAME}] score error for {sym}: {e}")
            return conf, "FALLBACK"

    def _features_for_signal(
        self,
        *,
        conf: float,
        votes: int,
        direction: str,
        symbol: str | None = None,
    ) -> Union[dict, None]:
        sym = str(symbol or os.getenv("COMMODITY", "SILVERM")).upper()
        target_df = None
        if hasattr(self, "_symbol_dfs") and self._symbol_dfs.get(sym) is not None:
            target_df = self._symbol_dfs.get(sym)
        elif hasattr(self, "_symbol_prefetch_df") and self._symbol_prefetch_df.get(sym) is not None:
            target_df = self._symbol_prefetch_df.get(sym)
        else:
            target_df = self._latest_df

        if target_df is None or len(target_df) < ML_LOOKBACK_CANDLES:
            return None

        reg_info = (
            getattr(self, "_symbol_regime_info", {}).get(sym)
            or self._latest_regime_info
            or {}
        )
        sym_prefetch_df = getattr(self, "_symbol_prefetch_df", {}).get(sym)
        if sym_prefetch_df is None:
            sym_prefetch_df = self._prefetch_df

        sym_prefetched_feats = getattr(self, "_symbol_prefetched_features", {}).get(sym)
        if sym_prefetched_feats is None:
            sym_prefetched_feats = self._prefetched_features

        sym_prefetch_dir = getattr(self, "_symbol_prefetch_dir", {}).get(sym)
        if sym_prefetch_dir is None:
            sym_prefetch_dir = self._prefetch_dir

        if sym_prefetched_feats and sym_prefetch_df is not None:
            latest_prefetch_ts = sym_prefetch_df.index[-1]
            latest_signal_ts = target_df.index[-1]
            if latest_prefetch_ts == latest_signal_ts:
                # Verify prefetched direction matches signal direction
                prefetched_is_call = sym_prefetch_dir == "BUY_CALL"
                signal_is_call = direction == "BUY_CALL"
                if prefetched_is_call != signal_is_call:
                    logger.debug(f"[{self.NAME}] Direction mismatch for {sym} (prefetch={sym_prefetch_dir}, signal={direction}), re-extracting features")
                    return extract(
                        df=target_df,
                        conf=conf,
                        votes=votes,
                        direction=direction,
                        lookback=ML_LOOKBACK_CANDLES,
                        regime_info=reg_info,
                        signal_context=self._latest_signal_context,
                        strategies_fired=self._latest_strategies,
                    )
                features = dict(sym_prefetched_feats)
                features["strategy_conf"] = float(conf)
                features["votes"] = int(votes)
                features["is_call"] = int(direction == "BUY_CALL")
                
                lbl = reg_info.get("label", "RANGING")
                features["regime_label_enc"] = 1.0 if lbl == "TRENDING" else (2.0 if lbl == "HIGH_VOLATILITY" else 0.0)
                features["regime_confidence"] = float(reg_info.get("confidence", 0.5))
                features["regime_atr_ratio"] = float(reg_info.get("atr_ratio", 1.0))
                context_features = extract(
                    df=sym_prefetch_df,
                    conf=conf,
                    votes=votes,
                    direction=direction,
                    lookback=ML_LOOKBACK_CANDLES,
                    regime_info=reg_info,
                    signal_context=self._latest_signal_context,
                    strategies_fired=self._latest_strategies,
                ) or {}
                for key, value in context_features.items():
                    if key in (
                        "setup_type_enc",
                        "setup_strength",
                        "expected_move_pct",
                        "winner_avg_weight",
                        "structure_bias_enc",
                        "bos_flag",
                        "choch_flag",
                        "liquidity_event_flag",
                        "liquidity_nearest_upper_pct",
                        "liquidity_nearest_lower_pct",
                        "middle_range_flag",
                        "weak_pair_flag",
                    ):
                        features[key] = float(value)
                
                return features

        return extract(
            df=target_df,
            conf=conf,
            votes=votes,
            direction=direction,
            lookback=ML_LOOKBACK_CANDLES,
            regime_info=reg_info,
            signal_context=self._latest_signal_context,
            strategies_fired=self._latest_strategies,
        )



    def _decision_threshold(self, decision_type: str, symbol: str | None = None) -> float:
        penalty = self._health_threshold_penalty()
        ens = self._get_ensemble_for_symbol(symbol)
        inst_min_conf = ML_MIN_CONFIDENCE
        if symbol:
            try:
                from instruments.registry import get_instrument_strategy_config
                inst_cfg = get_instrument_strategy_config(symbol)
                inst_min_conf = float(inst_cfg.get("min_ml_confidence", ML_MIN_CONFIDENCE))
            except Exception:
                pass
        if decision_type == "ML" and ens.is_trained:
            base = ML_THRESHOLD_OVERRIDE if ML_THRESHOLD_OVERRIDE > 0 else ens.decision_threshold
            return min(0.95, base + penalty)
        return min(0.95, inst_min_conf + penalty)

    def _rank_signal(
        self,
        *,
        data: dict,
        success_prob: float,
        raw_conf: float,
        votes: int,
        decision_type: str,
    ) -> tuple[float, dict]:
        regime_info = self._latest_regime_info or {}
        regime_label = str(regime_info.get("label", "RANGING")).upper()
        regime_conf = float(regime_info.get("confidence", 0.5) or 0.5)
        weighted_vote = (
            ((data.get("metadata") or {}).get("_context") or {}).get("weighted_vote") or {}
        )
        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        market_structure = (
            ((data.get("metadata") or {}).get("_context") or {}).get("market_structure")
            or setup.get("context", {}).get("market_structure")
            or {}
        )
        winner_avg_weight = float(weighted_vote.get("winner_avg_weight", 1.0) or 1.0)
        setup_strength = max(0.0, min(1.0, float(setup.get("setup_strength", 0.5) or 0.5)))
        setup_type = str(setup.get("setup_type", "unknown") or "unknown")
        structure_state = market_structure.get("structure_state", {}) or {}
        liquidity_event = market_structure.get("liquidity_event", {}) or {}
        entry_validation = market_structure.get("entry_validation", {}) or {}
        direction_validation = (
            (entry_validation.get("by_direction", {}) or {}).get(str(data.get("direction", "")), {})
        )

        vote_strength = min(1.0, votes / 3.0)
        strategy_strength = max(0.0, min(1.0, raw_conf))
        model_strength = max(0.0, min(1.0, success_prob))
        regime_strength = self._regime_rank_factor(regime_label, regime_conf, winner_avg_weight)

        rank_score = (
            model_strength * ML_RANK_MODEL_WEIGHT
            + strategy_strength * ML_RANK_STRATEGY_WEIGHT
            + vote_strength * ML_RANK_VOTE_WEIGHT
            + regime_strength * ML_RANK_REGIME_WEIGHT
        )

        if decision_type == "FALLBACK":
            rank_score *= 0.97
        if self._health.get("retrain_needed", False):
            rank_score *= 0.96

        blocked_reason = self._blocked_strategy_reason(data=data)
        if blocked_reason:
            rank_score *= 0.93
        setup_bonus = min(0.06, max(0.0, setup_strength - 0.5) * 0.12)
        rank_score += setup_bonus
        if liquidity_event.get("type", "NONE") != "NONE":
            rank_score += 0.015
        if structure_state.get("choch", "NONE") != "NONE":
            rank_score += 0.012
        if structure_state.get("bos", "NONE") != "NONE":
            rank_score += 0.01
        if not bool(direction_validation.get("valid", entry_validation.get("valid", True))):
            rank_score *= 0.90
        if entry_validation.get("avoid", False):
            rank_score *= 0.86
        if entry_validation.get("middle_range", False):
            rank_score *= 0.94
        structure_bias = str(structure_state.get("bias", "UNKNOWN") or "UNKNOWN").upper()
        regime_structure_penalty = 0.0
        if structure_bias == "RANGING":
            regime_structure_penalty += 0.015
            if regime_label == "TRENDING":
                regime_structure_penalty += 0.02
        if regime_structure_penalty > 0:
            rank_score = max(0.01, rank_score - regime_structure_penalty)

        loss_pattern_penalty = 0.0
        signal_minutes = _signal_minutes(str(data.get("timestamp", "")))
        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        pair_key = frozenset(strategies)
        if (
            str(data.get("direction", "NONE")) == "BUY_CALL"
            and setup_type == "vote_aligned"
            and setup_strength <= ML_FILTER_SETUP_STRENGTH_THRESH_0_84
            and signal_minutes is not None
            and {"VWAP+EMA", "BBSqueeze", "ADX+PSAR"}.issubset(strategies)
        ):
            loss_pattern_penalty += 0.055
        elif (
            str(data.get("direction", "NONE")) == "BUY_CALL"
            and setup_type == "vote_aligned"
            and setup_strength <= ML_FILTER_SETUP_STRENGTH_THRESH_0_81
            and pair_key in _CORRELATED_PAIRS
            and signal_minutes is not None
            and signal_minutes >= 12 * 60
        ):
            loss_pattern_penalty += 0.035
            if signal_minutes >= 14 * 60:
                loss_pattern_penalty += 0.02

        if (
            setup_type == "trend_pullback"
            and pair_key != frozenset({"ADX+PSAR"})
            and rank_score < ML_FILTER_RANK_SCORE_THRESH_0_58
            and setup_strength <= ML_FILTER_SETUP_STRENGTH_THRESH_0_81
        ):
            loss_pattern_penalty += 0.05

        if (
            setup_type == "trend_pullback"
            and votes <= 2
            and setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_82
            and signal_minutes is not None
            and signal_minutes >= 14 * 60 + 15
        ):
            loss_pattern_penalty += 0.04

        if loss_pattern_penalty > 0:
            rank_score = max(0.01, rank_score - loss_pattern_penalty)

        tier = "HIGH" if rank_score >= ML_RANK_HIGH_THRESHOLD else (
            "MEDIUM" if rank_score >= ML_RANK_SOFT_THRESHOLD else "LOW"
        )
        components = {
            "model_strength": round(model_strength, 4),
            "strategy_strength": round(strategy_strength, 4),
            "vote_strength": round(vote_strength, 4),
            "regime_strength": round(regime_strength, 4),
            "setup_strength": round(setup_strength, 4),
            "setup_bonus": round(setup_bonus, 4),
            "setup_type": setup_type,
            "structure_bias": str(structure_state.get("bias", "UNKNOWN")),
            "bos": str(structure_state.get("bos", "NONE")),
            "choch": str(structure_state.get("choch", "NONE")),
            "liquidity_event": str(liquidity_event.get("type", "NONE")),
            "entry_valid": bool(direction_validation.get("valid", entry_validation.get("valid", True))),
            "entry_avoid": bool(entry_validation.get("avoid", False)),
            "middle_range": bool(entry_validation.get("middle_range", False)),
            "regime_structure_penalty": round(regime_structure_penalty, 4),
            "loss_pattern_penalty": round(loss_pattern_penalty, 4),
            "winner_avg_weight": round(winner_avg_weight, 4),
            "regime_label": regime_label,
        }
        return round(min(0.99, max(0.01, rank_score)), 4), {
            "tier": tier,
            "soft_pass": rank_score >= ML_RANK_SOFT_THRESHOLD,
            "components": components,
        }

    @staticmethod
    def _regime_rank_factor(
        regime_label: str,
        regime_conf: float,
        winner_avg_weight: float,
    ) -> float:
        regime_base = {
            "TRENDING": 0.72,
            "RANGING": 0.62,
            "HIGH_VOLATILITY": 0.52,
        }.get(regime_label, 0.60)
        weight_bonus = min(0.16, max(0.0, winner_avg_weight - 1.0) * 0.35)
        confidence_bonus = min(0.12, max(0.0, regime_conf - 0.5) * 0.30)
        return min(1.0, regime_base + weight_bonus + confidence_bonus)

    @staticmethod
    def _decision_reason(
        *,
        decision_type: str,
        success_prob: float,
        rank_score: float,
        rank_tier: str,
        threshold: float,
        accepted: bool,
        extra_reason: str,
    ) -> str:
        status = "APPROVED" if accepted else "REJECTED"
        return (
            f"{status} | path={decision_type} | "
            f"p={success_prob:.3f} | rank={rank_score:.3f} "
            f"(tier={rank_tier}, thr={threshold:.3f}) | {extra_reason}"
        )

    def _health_threshold_penalty(self) -> float:
        if not self._health.get("retrain_needed", False):
            return 0.0
        sample_size = int(self._health.get("live_sample_size", 0) or 0)
        live_precision = self._health.get("live_precision")
        if sample_size >= ML_LIVE_MIN_EVAL_TRADES and live_precision is not None:
            return 0.05
        return 0.03

    def _secondary_lane_enabled(self) -> bool:
        return not bool(self._health.get("retrain_needed", False))

    def _passes_signal_budget(self, data: dict) -> tuple[bool, str]:
        ts = self._resolve_ts(data.get("timestamp"))
        self._reset_budgets(ts.date().isoformat())
        regime = str(data.get("regime", "TRENDING")).upper()
        session_bucket = self._session_bucket(ts)
        if self._daily_approvals >= SIGNAL_APPROVAL_DAILY_BUDGET:
            return False, (
                f"Signal throttle: daily approval budget "
                f"{self._daily_approvals}/{SIGNAL_APPROVAL_DAILY_BUDGET} exhausted"
            )
        regime_budget = SIGNAL_REGIME_BUDGETS.get(regime, 0)
        if self._regime_approvals.get(regime, 0) >= regime_budget:
            return False, (
                f"Signal throttle: regime budget {regime} "
                f"{self._regime_approvals.get(regime, 0)}/{regime_budget} exhausted"
            )
        session_budget = SIGNAL_SESSION_BUDGETS.get(session_bucket, 0)
        if self._session_approvals.get(session_bucket, 0) >= session_budget:
            return False, (
                f"Signal throttle: session budget {session_bucket} "
                f"{self._session_approvals.get(session_bucket, 0)}/{session_budget} exhausted"
            )
        return True, ""

    def _consume_signal_budget(self, data: dict) -> None:
        ts = self._resolve_ts(data.get("timestamp"))
        self._reset_budgets(ts.date().isoformat())
        regime = str(data.get("regime", "TRENDING")).upper()
        session_bucket = self._session_bucket(ts)
        self._daily_approvals += 1
        self._session_approvals[session_bucket] = self._session_approvals.get(session_bucket, 0) + 1
        self._regime_approvals[regime] = self._regime_approvals.get(regime, 0) + 1
        self._health["daily_approvals"] = self._daily_approvals
        self._health["session_approvals"] = dict(self._session_approvals)
        self._health["regime_approvals"] = dict(self._regime_approvals)

    @staticmethod
    def _passes_fallback_gate(data: dict, fallback_conf: float) -> tuple[bool, str]:
        regime = str(data.get("regime", "TRENDING")).upper()
        votes = _extract_votes(data.get("votes", 0))
        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)

        if votes < MIN_STRATEGY_VOTES:
            return False, (
                f"Fallback gate: votes {votes} < required {MIN_STRATEGY_VOTES}"
            )
        if fallback_conf < ML_FALLBACK_MIN_CONFIDENCE:
            return False, (
                f"Fallback gate: conf {fallback_conf:.2f} < "
                f"{ML_FALLBACK_MIN_CONFIDENCE:.2f}"
            )
        if regime == "RANGING" and not ML_FALLBACK_ALLOW_RANGING:
            return False, "Fallback gate: ranging regime blocked without trained model"
        if setup_type == "breakout" and (votes < 3 or setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_76):
            return False, (
                f"Fallback gate: weak breakout setup={setup_strength:.2f} "
                f"votes={votes}"
            )
        if setup_type == "trend_pullback" and votes >= 4 and setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_75:
            return False, (
                f"Fallback gate: weak high-consensus trend_pullback "
                f"setup={setup_strength:.2f} votes={votes}"
            )
        blocked_reason = MLFilterAgent._blocked_strategy_reason(
            data=data,
            blocked_pairs=ML_FALLBACK_BLOCKED_STRATEGY_PAIRS,
            prefix="Fallback gate",
        )
        if blocked_reason:
            return False, blocked_reason
        return True, ""

    @staticmethod
    def _passes_secondary_ml_gate(
        data: dict,
        ml_conf: float,
        raw_conf: float,
        votes: int,
    ) -> bool:
        regime = str(data.get("regime", "TRENDING")).upper()
        if regime != "TRENDING":
            return False
        if votes < ML_SECONDARY_MIN_VOTES:
            return False
        if raw_conf < ML_SECONDARY_MIN_RAW_CONFIDENCE:
            return False
        if ml_conf < ML_SECONDARY_THRESHOLD:
            return False
        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        if not strategies:
            return False
        for raw_pair in ML_SECONDARY_ALLOWED_STRATEGY_PAIRS:
            pair = {part.strip() for part in str(raw_pair).split("|") if part.strip()}
            if pair and pair.issubset(strategies):
                return True
        return False

    @staticmethod
    def _passes_near_miss_gate(
        *,
        data: dict,
        rank_score: float,
        threshold: float,
        ml_conf: float,
        raw_conf: float,
        votes: int,
    ) -> bool:
        if votes < MIN_STRATEGY_VOTES:
            return False
        if rank_score < threshold - 0.03:
            return False
        regime = str(data.get("regime", "TRENDING")).upper()
        if regime != "TRENDING":
            return False

        direction = str(data.get("direction", "NONE"))
        if direction == "BUY_CALL":
            return False

        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        if setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_7:
            return False

        market_structure = (setup.get("context", {}) or {}).get("market_structure", {}) or {}
        entry_validation = market_structure.get("entry_validation", {}) or {}
        direction_validation = (
            (entry_validation.get("by_direction", {}) or {}).get(direction, {})
        )
        if not bool(direction_validation.get("valid", entry_validation.get("valid", True))):
            return False
        if entry_validation.get("avoid", False) or entry_validation.get("middle_range", False):
            return False

        if raw_conf < ML_FILTER_RAW_CONF_THRESH_0_74:
            return False
        if ml_conf < ML_FILTER_ML_CONF_THRESH_0_32:
            return False

        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        institutional_confirmers = {
            "FVG",
            "OIAnalysis",
            "GammaExposure",
            "ORB",
            "CPR",
            "RangeSpread",
            "StrikeMomentum",
            "VolumeProfile",
        }
        if (
            votes >= 5
            and raw_conf >= ML_FILTER_RAW_CONF_THRESH_0_88
            and ml_conf >= ML_FILTER_ML_CONF_THRESH_0_3
            and rank_score >= threshold - 0.10
            and len(strategies.intersection(institutional_confirmers)) >= 2
            and setup_strength >= ML_FILTER_SETUP_STRENGTH_THRESH_0_53
        ):
            return True
        # Keep the rescue lane narrow: only allow the stronger trending pairs
        # that are currently missing by a few basis points.
        allowed_pairs = (
            {"BBSqueeze", "ADX+PSAR"},
            {"SuperTrend+RSI", "ADX+PSAR"},
            {"SuperTrend+RSI", "BBSqueeze"},
        )
        return any(pair.issubset(strategies) for pair in allowed_pairs)

    def _record_call_recovery_watch(
        self,
        *,
        data: dict,
        rank_score: float,
        threshold: float,
        raw_conf: float,
        votes: int,
        reason: str,
    ) -> None:
        direction = str(data.get("direction", "NONE"))
        if direction != "BUY_CALL" or votes < MIN_STRATEGY_VOTES:
            return
        signal_minutes = _signal_minutes(str(data.get("timestamp", "")))
        if signal_minutes is None or not (9 * 60 + 50 <= signal_minutes <= 10 * 60 + 45):
            return
        spot = self._spot_from_signal(data)
        if spot <= 0:
            return
        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        watchable_stack = (
            "SuperTrend+RSI" in strategies
            and bool({"Ichimoku", "ADX+PSAR", "BBSqueeze"}.intersection(strategies))
        )
        if (
            not watchable_stack
            or setup_type not in {"vote_aligned", "trend_pullback"}
            or setup_strength > ML_FILTER_SETUP_STRENGTH_THRESH_0_6
            or rank_score < ML_FILTER_RANK_SCORE_THRESH_0_33
            or rank_score > threshold
            or raw_conf < ML_FILTER_RAW_CONF_THRESH_0_8
            or "call_bias_penalty" not in reason
        ):
            return

        day = str(pd.Timestamp(str(data.get("timestamp", ""))).date())
        current = self._call_recovery_watch.get(day, {})
        high = max(float(current.get("high", 0.0) or 0.0), spot)
        self._call_recovery_watch[day] = {
            "high": high,
            "timestamp": str(data.get("timestamp", "")),
            "rank": max(float(current.get("rank", 0.0) or 0.0), rank_score),
            "strategies": sorted(strategies),
        }
        logger.info(
            f"[{self.NAME}] CALL recovery watch armed | day={day} "
            f"high={high:.2f} rank={rank_score:.2f} setup={setup_strength:.2f}"
        )

    def _passes_call_recovery_gate(
        self,
        *,
        data: dict,
        rank_score: float,
        threshold: float,
        ml_conf: float,
        raw_conf: float,
        votes: int,
    ) -> bool:
        direction = str(data.get("direction", "NONE"))
        if direction != "BUY_CALL" or votes < 3:
            return False
        ts = pd.Timestamp(str(data.get("timestamp", "")))
        day = str(ts.date())
        watch = self._call_recovery_watch.get(day)
        if not watch:
            return False
        signal_minutes = ts.hour * 60 + ts.minute
        watch_minutes = _signal_minutes(str(watch.get("timestamp", "")))
        if watch_minutes is None or signal_minutes - watch_minutes > 90:
            return False

        spot = self._spot_from_signal(data)
        watch_high = float(watch.get("high", 0.0) or 0.0)
        displacement = max(22.0, watch_high * 0.0009)
        if spot < watch_high + displacement:
            return False

        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        trend_stack = (
            "SuperTrend+RSI" in strategies
            and bool({"ADX+PSAR", "BBSqueeze"}.intersection(strategies))
            and len(strategies) >= 3
        )
        regime = str(data.get("regime", "TRENDING")).upper()
        if (
            not trend_stack
            or regime != "TRENDING"
            or setup_type not in {"vote_aligned", "trend_pullback"}
            or setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_58
            or rank_score < ML_FILTER_RANK_SCORE_THRESH_0_36
            or ml_conf < ML_FILTER_ML_CONF_THRESH_0_32
            or raw_conf < ML_FILTER_RAW_CONF_THRESH_0_83
            or threshold - rank_score > ML_FILTER_RANK_SCORE_THRESH_0_13
        ):
            return False

        logger.info(
            f"[{self.NAME}] CALL recovery gate passed | spot={spot:.2f} "
            f"watch_high={watch_high:.2f} rank={rank_score:.2f} setup={setup_strength:.2f}"
        )
        return True

    @staticmethod
    def _spot_from_signal(data: dict) -> float:
        for key in ("nifty_price", "underlying_price", "spot", "ltp", "price"):
            try:
                value = float(data.get(key, 0.0) or 0.0)
            except Exception:
                value = 0.0
            if value > 0:
                return value
        return 0.0

    @staticmethod
    def _blocked_strategy_reason(
        data: dict,
        blocked_pairs: tuple[str, ...] = ML_BLOCKED_STRATEGY_PAIRS,
        prefix: str = "ML gate",
    ) -> str:
        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        for raw_pair in blocked_pairs:
            pair = {part.strip() for part in str(raw_pair).split("|") if part.strip()}
            if pair and pair.issubset(strategies):
                return (
                    f"{prefix}: blocked weak strategy pair "
                    f"{' + '.join(sorted(pair))}"
                )
        return ""

    @staticmethod
    def _loss_pattern_block_reason(
        *,
        data: dict,
        rank_score: float,
        votes: int,
    ) -> str:
        context = (data.get("metadata") or {}).get("_context", {}) or {}
        setup = context.get("setup", {}) or {}
        setup_context = setup.get("context", {}) or {}
        market_structure = (
            context.get("market_structure")
            or setup_context.get("market_structure")
            or {}
        )
        structure_state = market_structure.get("structure_state", {}) or {}
        structure_bias = str(structure_state.get("bias", "UNKNOWN") or "UNKNOWN").upper()
        setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        direction = str(data.get("direction", "NONE"))
        signal_minutes = _signal_minutes(str(data.get("timestamp", "")))
        strategies = {
            str(name).strip()
            for name in data.get("strategies_fired", [])
            if str(name).strip()
        }
        trend_stack = {"SuperTrend+RSI", "BBSqueeze", "ADX+PSAR"}
        full_trend_stack = {"SuperTrend+RSI", "VWAP+EMA", "BBSqueeze", "ADX+PSAR"}

        if (
            signal_minutes is not None
            and signal_minutes < 9 * 60 + 30
            and setup_type == "vote_aligned"
            and strategies.issubset(full_trend_stack)
        ):
            return "ML stability gate: opening trend-stack signal before 09:30"

        if (
            trend_stack.issubset(strategies)
            and "VWAP+EMA" not in strategies
            and setup_type in {"vote_aligned", "trend_pullback"}
        ):
            return "ML stability gate: correlated SuperTrend+BBSqueeze+ADX stack"

        if (
            strategies == full_trend_stack
            and direction == "BUY_CALL"
            and setup_type == "vote_aligned"
            and structure_bias == "RANGING"
            and setup_strength < ML_FILTER_SETUP_STRENGTH_THRESH_0_78
        ):
            return "ML stability gate: four-strategy CALL inside ranging structure"

        if (
            votes <= 2
            and rank_score < ML_FILTER_RANK_SCORE_THRESH_0_79
            and setup_type == "trend_pullback"
            and structure_bias == "BEARISH"
            and strategies == {"SuperTrend+RSI", "ADX+PSAR"}
        ):
            return "ML stability gate: weak two-vote trend pullback"

        return ""

    def _is_model_stale(self) -> bool:
        """Check if loaded model is too old or incompatible with current features."""
        if not self.ensemble.is_trained:
            return False

        if getattr(self, "_mode", "LIVE").upper() == "BACKTEST":
            return False

        trained_at = str(getattr(self.ensemble.meta, "trained_at", "") or "")
        if trained_at:
            try:
                ref_time = self._replay_time if hasattr(self, "_replay_time") and self._replay_time else datetime.now(IST)
                age_days = max((ref_time - self._resolve_ts(trained_at)).days, 0)
                if age_days > ML_RETRAIN_DAYS:
                    logger.warning(f"[{self.NAME}] Model is {age_days} days old - treating as stale")
                    return True
            except Exception:
                pass

        # Check if model has no feature cols (incompatible with current code)
        if hasattr(self.ensemble, 'meta') and self.ensemble.meta:
            if not getattr(self.ensemble.meta, 'feature_cols', []):
                logger.warning(f"[{self.NAME}] Model has no feature columns - incompatible")
                return True
        return False

    def _load_model(self) -> None:
        for path, timeframe, is_fallback in self._candidate_model_paths(LIVE_TIMEFRAME):
            if not path.exists():
                continue
            if self.ensemble.load(path):
                self._model_path = path
                self._model_timeframe = timeframe
                self._model_fallback_used = is_fallback
                self._model_mtime = path.stat().st_mtime
                if is_fallback:
                    logger.warning(
                        f"[{self.NAME}] Exact {LIVE_TIMEFRAME} model not found; "
                        f"using {timeframe} model at {path}."
                    )
                return

        logger.info(
            f"[{self.NAME}] No trained model found for {LIVE_TIMEFRAME}. "
            f"Checked: {[str(path) for path, _, _ in self._candidate_model_paths(LIVE_TIMEFRAME)]}. "
            "Entering FALLBACK mode."
        )

    def _get_ensemble_for_symbol(self, symbol: str | None = None) -> SignalForgeEnsemble:
        sym = (symbol or os.getenv("COMMODITY", "SILVERM")).strip().upper()
        if sym in self._ensembles and self._ensembles[sym].is_trained:
            return self._ensembles[sym]

        ens = SignalForgeEnsemble()
        for path, timeframe, is_fallback in self._candidate_model_paths(LIVE_TIMEFRAME, symbol=sym):
            if path.exists():
                if ens.load(path):
                    logger.info(f"[{self.NAME}] Loaded model for {sym} from {path}")
                    self._ensembles[sym] = ens
                    return ens
        if self.ensemble.is_trained:
            self._ensembles[sym] = self.ensemble
            return self.ensemble
        return ens

    def reload_model(self) -> None:
        self.ensemble = SignalForgeEnsemble()
        self._ensembles.clear()
        self._load_model()
        for sym in ("SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"):
            self._ensemble_for_symbol(sym)
        if self.ensemble.is_trained:
            self._evaluate_model_health()
            logger.info(
                f"[{self.NAME}] Model reloaded | "
                f"models={list(self.ensemble.models.keys())} | "
                f"features={len(self.ensemble.feature_cols)} | "
                f"lanes={list(self._ensembles.keys())}"
            )
        else:
            logger.warning(f"[{self.NAME}] reload_model: no compatible model for {LIVE_TIMEFRAME}")

    async def on_model_retrained(self, msg: Message) -> None:
        model_path = msg.payload.get("model_path")
        retrained_symbol = msg.payload.get("symbol", "ALL")
        logger.info(f"[{self.NAME}] MODEL_RETRAINED received | symbol={retrained_symbol} path={model_path} — hot-reloading ensembles")
        self.reload_model()

    async def _model_file_watcher(self) -> None:
        while True:
            await asyncio.sleep(300)
            try:
                if not self._model_path.exists():
                    continue
                mtime = self._model_path.stat().st_mtime
                if mtime > self._model_mtime + 1:
                        logger.info(f"[{self.NAME}] New model detected on disk  auto-reloading")
                        self.reload_model()
                else:
                    self._evaluate_model_health()
            except Exception as e:
                logger.debug(f"[{self.NAME}] File watcher error: {e}")

    async def _explain(self, data: dict, ml_conf: float, rank_score: float) -> None:
        sym = str(data.get("symbol") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
        setup = (((data.get("metadata") or {}).get("_context") or {}).get("setup") or {})
        structure = (((data.get("metadata") or {}).get("_context") or {}).get("market_structure") or {})
        prompt = (
            f"ML ranked a {sym} {data.get('direction')} signal.\n"
            f"Strategies: {data.get('strategies_fired')}\n"
            f"Success probability: {ml_conf:.0%} | Rank score: {rank_score:.0%} | Votes: {data.get('votes')}\n"
            f"Setup: {setup.get('setup_type', 'unknown')} ({float(setup.get('setup_strength', 0.0) or 0.0):.2f})\n"
            f"Structure: {structure.get('structure_state', {}).get('bias', 'UNKNOWN')} | "
            f"Event: {structure.get('liquidity_event', {}).get('type', 'NONE')}\n"
            f"In 2 sentences, explain why this rank is justified. Be direct."
        )
        cache_key = "|".join([
            "ml_explain",
            str(data.get("timestamp", "")),
            str(data.get("direction", "")),
            ",".join(data.get("strategies_fired", [])),
        ])
        text = await call_llm_context_async(
            prompt,
            cache_key=cache_key,
            task_type=TaskType.ML_EXPLANATION,
            max_tokens=120,
            rank_score=rank_score,
            success_prob=ml_conf,
        )
        if text:
            await self.bus.publish(
                Topic.ALERT,
                {"type": "ml_explanation", "text": text},
                self.NAME,
            )
    #  LLM EXPLANATION (non-blocking) 

    async def _explain_async(self, data: dict, ml_conf: float, decision: str) -> None:
        try:
            from llm.factory import get_llm
            llm_status = get_llm()
            if llm_status.provider_name == "disabled":
                return
            sym = str(data.get("symbol") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
            prompt = (
                f"ML {'model' if decision == 'ML' else 'fallback'} approved "
                f"a {sym} {data.get('direction')} signal.\n"
                f"Strategies: {data.get('strategies_fired')}\n"
                f"Confidence: {ml_conf:.0%} | Votes: {data.get('votes')}\n"
                f"In 2 sentences, why does this look like a good entry? Be direct."
            )
            resp = await llm_status.complete(prompt, max_tokens=120)
            if resp:
                await self.bus.publish(Topic.ALERT, {
                    "type": "ml_explanation", "text": resp.text,
                    "provider": resp.provider,
                }, self.NAME)
        except Exception as e:
            logger.debug(f"[{self.NAME}] explain skipped: {e}")

    def health_status(self) -> dict:
        return {
            **self._health,
            "is_stale": self._is_model_stale(),
            "threshold": self._decision_threshold("ML") if self.ensemble.is_trained else None,
            "base_threshold": (
                ML_THRESHOLD_OVERRIDE if ML_THRESHOLD_OVERRIDE > 0
                else self.ensemble.decision_threshold
            ) if self.ensemble.is_trained else None,
            "secondary_lane_enabled": self._secondary_lane_enabled(),
            "models": list(self.ensemble.models.keys()),
            "trained_at": getattr(self.ensemble.meta, "trained_at", ""),
            "requested_model_path": str(self._requested_model_path),
            "model_path": str(self._model_path),
            "model_timeframe": str(self._model_timeframe),
            "model_fallback_used": bool(self._model_fallback_used),
            "calibration_method": getattr(self.ensemble.meta, "calibration_method", ""),
        }

    def _reset_budgets(self, day: str) -> None:
        if day == self._approval_day:
            return
        self._approval_day = day
        self._daily_approvals = 0
        self._session_approvals = {k: 0 for k in SIGNAL_SESSION_BUDGETS}
        self._regime_approvals = {k: 0 for k in SIGNAL_REGIME_BUDGETS}
        self._health["daily_approvals"] = 0
        self._health["session_approvals"] = dict(self._session_approvals)
        self._health["regime_approvals"] = dict(self._regime_approvals)

    def _evaluate_model_health(self) -> None:
        trained_at = str(getattr(self.ensemble.meta, "trained_at", "") or "")
        age_days = None
        retrain_needed = False
        retrain_reason = ""
        if trained_at:
            try:
                age_days = max((datetime.now(IST) - self._resolve_ts(trained_at)).days, 0)
                if age_days >= ML_RETRAIN_DAYS:
                    retrain_needed = True
                    retrain_reason = f"Model age {age_days}d exceeds retrain window {ML_RETRAIN_DAYS}d"
            except Exception:
                age_days = None

        live_precision, sample_size = self._recent_live_precision()
        if sample_size >= ML_LIVE_MIN_EVAL_TRADES and live_precision is not None:
            baseline = max(ML_LIVE_MIN_PRECISION, float(getattr(self.ensemble.meta, "val_precision", 0.0) or 0.0) * 0.7)
            if live_precision < baseline:
                retrain_needed = True
                retrain_reason = (
                    f"Live precision {live_precision:.2f} below floor {baseline:.2f} "
                    f"over {sample_size} trades"
                )

        changed = (
            retrain_needed != self._health.get("retrain_needed")
            or retrain_reason != self._health.get("retrain_reason")
        )
        self._health.update({
            "active": bool(self.ensemble.is_trained),
            "model_age_days": age_days,
            "live_precision": round(live_precision, 4) if live_precision is not None else None,
            "live_sample_size": sample_size,
            "retrain_needed": retrain_needed,
            "retrain_reason": retrain_reason,
            "daily_approvals": self._daily_approvals,
            "daily_budget": SIGNAL_APPROVAL_DAILY_BUDGET,
            "session_approvals": dict(self._session_approvals),
            "regime_approvals": dict(self._regime_approvals),
        })
        if changed and retrain_needed:
            logger.warning(f"[{self.NAME}] Retrain needed | {retrain_reason}")
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self.bus.publish(
                        Topic.ALERT,
                        {
                            "type": "ml_retrain_needed",
                            "text": retrain_reason,
                            "severity": "WARNING",
                        },
                        self.NAME,
                    )
                )
            except RuntimeError:
                logger.debug(
                    f"[{self.NAME}] Retrain alert deferred because no event loop is running"
                )

    def _recent_live_precision(self) -> tuple[Union[float, None], int]:
        csv_files = sorted(Path(JOURNAL_DIR).glob("signals_*.csv"))
        if not csv_files:
            return None, 0
        frames = []
        for path in csv_files[-10:]:
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                continue
        if not frames:
            return None, 0
        df = pd.concat(frames, ignore_index=True)
        if "ml_decision" not in df.columns or "outcome_eod" not in df.columns:
            return None, 0
        filtered = df[
            (
                df["ml_decision"].astype(str).isin(["ML", "ML_SECONDARY"])
            ) & (
                df["outcome_eod"].astype(str).isin(["WIN", "LOSS"])
            )
        ].copy()
        if filtered.empty:
            return None, 0
        if "signal_id" in filtered.columns:
            filtered = filtered.drop_duplicates(subset=["signal_id"], keep="last")
        filtered = filtered.tail(ML_LIVE_PRECISION_WINDOW)
        sample_size = len(filtered)
        if sample_size == 0:
            return None, 0
        precision = float((filtered["outcome_eod"].astype(str) == "WIN").mean())
        return precision, sample_size

    @staticmethod
    def _resolve_ts(raw: Union[str, None]) -> datetime:
        if raw:
            ts = datetime.fromisoformat(str(raw))
            return ts if ts.tzinfo else IST.localize(ts)
        return datetime.now(IST)

    @staticmethod
    def _session_bucket(ts: datetime) -> str:
        hhmm = ts.strftime("%H:%M")
        if hhmm < "13:00":
            return "OPENING"
        if hhmm < "17:00":
            return "MIDDAY"
        return "CLOSING"

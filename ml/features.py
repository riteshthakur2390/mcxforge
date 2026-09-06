"""
ml/features.py — Feature Engineering for SignalForge ML
=========================================================
Single source of truth for all feature extraction.
Used by BOTH:
  - ml/training/trainer.py  (builds training dataset from journal)
  - agents_code/agent3_ml/filter.py  (extracts features at signal time)

If you add a feature here, both training and inference get it automatically.
Never extract features in two places — always import from here.

Feature categories:
  1. Candle structure  (body, wicks)
  2. Price returns     (1/3/5/10/20 candle)
  3. Momentum          (RSI, MACD, Stochastic, ROC)
  4. Trend             (ADX, EMA relationships)
  5. Volatility        (ATR, Bollinger Bands)
  6. Volume            (ratio, trend)
  7. Time              (hour, day, distance from open)
  8. Signal meta       (strategy confidence, vote count)
"""

from typing import Union
import numpy as np
import pandas as pd
from loguru import logger
from config.settings import (
    ML_FEATURE_LOOKBACK,
    ML_FEATURE_MIN_DF_LEN,
    ML_FEATURE_RSI_PERIOD,
    ML_FEATURE_EMA_PERIODS,
    ML_FEATURE_BB_PERIOD,
    ML_FEATURE_BB_STD,
    ML_FEATURE_VOL_MA_PERIOD,
)

# Optional TA import — graceful fallback if not installed
try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("[Features] pandas_ta not available — basic features only")


# ── FEATURE NAMES ─────────────────────────────────────────────────────────────
# Keep this list in sync with _extract() below.
# Order matters for model consistency.

FEATURE_NAMES = [
    # Candle structure
    "body_pct", "upper_wick_pct", "lower_wick_pct", "is_bullish", "candle_range_pct",
    # Returns
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    # Momentum
    "rsi", "rsi_slope", "rsi_vs_50",
    "macd_hist", "macd_signal",
    "stoch_k", "stoch_d",
    "roc_5", "roc_10",
    # Trend
    "adx", "plus_di", "minus_di", "di_diff",
    "vs_ema9", "vs_ema20", "vs_ema50",
    "ema9_slope", "ema20_slope",
    # Volatility
    "atr_pct", "bb_width", "bb_position",
    "atr_ratio",  # NEW: current ATR / 20-bar avg ATR (expanding vs contracting)
    # Volume
    "vol_ratio", "vol_trend",
    "vol_spike",  # NEW: 1 if volume > 2× average (institutional activity)
    # Price structure (NEW)
    "vs_vwap",  # price distance from intraday VWAP %
    "trend_persist",  # consecutive same-direction candle count
    "session_bucket",  # 1=morning 2=mid 3=lunch 4=afternoon (not linear hour)
    # Time
    "hour", "minute", "day_of_week", "mins_from_open",
    # Regime context (NEW — from detailed_regime in regime_details)
    "regime_label_enc",  # 0=RANGING 1=TRENDING 2=HIGH_VOLATILITY
    "regime_confidence",  # 0.0–1.0 how certain the regime classification is
    "regime_atr_ratio",  # ATR expansion/contraction ratio
    # Setup / structure / liquidity context (NEW)
    "setup_type_enc",  # encoded setup type
    "setup_strength",  # setup confidence from setup engine
    "expected_move_pct",  # expected move normalized by close
    "winner_avg_weight",  # weighted vote quality
    "structure_bias_enc",  # -1 bearish / 0 neutral / +1 bullish
    "bos_flag",  # break-of-structure present
    "choch_flag",  # change-of-character present
    "liquidity_event_flag",  # liquidity sweep / structure break present
    "liquidity_nearest_upper_pct",  # distance to nearest upper liquidity
    "liquidity_nearest_lower_pct",  # distance to nearest lower liquidity
    "middle_range_flag",  # avoid entries in middle of range
    "weak_pair_flag",  # historically weak pair marker
    # Signal meta
    "strategy_conf", "votes", "is_call",
]


def extract(
    df:        pd.DataFrame,
    conf:      float,
    votes:     int,
    direction: str = "BUY_CALL",
    lookback:  int = 60,
     **kwargs,  # accepts regime_info/signal_context/strategies_fired
) -> Union[dict, None]:
    """
    Extract all features from a candle DataFrame.

    Args:
        df:        OHLCV DataFrame with DatetimeIndex
        conf:      strategy ensemble confidence (0–1)
        votes:     number of strategies that agreed
        direction: "BUY_CALL" or "BUY_PUT"
        lookback:  number of candles to use

    Returns:
        dict of feature_name → float, or None if not enough data
    """
    if df is None or len(df) < max(lookback, 30):
        return None

    w = df.tail(lookback).copy()
    c = w.iloc[-1]
    f: dict = {}

    # ── 1. CANDLE STRUCTURE ───────────────────────────────────────────────────
    body = abs(float(c["close"]) - float(c["open"]))
    rng  = max(float(c["high"]) - float(c["low"]), 0.01)
    f["body_pct"]       = body / rng
    f["upper_wick_pct"] = (float(c["high"]) - max(float(c["open"]), float(c["close"]))) / rng
    f["lower_wick_pct"] = (min(float(c["open"]), float(c["close"])) - float(c["low"])) / rng
    f["is_bullish"]     = int(float(c["close"]) > float(c["open"]))
    f["candle_range_pct"] = rng / max(float(c["close"]), 1) * 100

    # ── 2. PRICE RETURNS ──────────────────────────────────────────────────────
    close_arr = w["close"].values
    for n in [1, 3, 5, 10, 20]:
        if len(close_arr) > n:
            f[f"ret_{n}"] = (close_arr[-1] - close_arr[-n-1]) / max(close_arr[-n-1], 1) * 100
        else:
            f[f"ret_{n}"] = 0.0

    if TA_AVAILABLE:
        # ── 3. MOMENTUM ───────────────────────────────────────────────────────
        try:
            rsi = ta.rsi(w["close"], 14)
            if rsi is not None and not rsi.dropna().empty:
                f["rsi"]       = float(rsi.iloc[-1])
                f["rsi_slope"] = float(rsi.iloc[-1] - rsi.iloc[-5]) if len(rsi) >= 5 else 0.0
                f["rsi_vs_50"] = float(rsi.iloc[-1]) - 50
            else:
                f["rsi"] = f["rsi_slope"] = f["rsi_vs_50"] = 0.0
        except Exception:
            f["rsi"] = f["rsi_slope"] = f["rsi_vs_50"] = 50.0

        try:
            macd = ta.macd(w["close"])
            if macd is not None:
                f["macd_hist"]   = float(macd["MACDh_12_26_9"].iloc[-1])
                f["macd_signal"] = float(macd["MACDs_12_26_9"].iloc[-1])
            else:
                f["macd_hist"] = f["macd_signal"] = 0.0
        except Exception:
            f["macd_hist"] = f["macd_signal"] = 0.0

        try:
            stoch = ta.stoch(w["high"], w["low"], w["close"])
            if stoch is not None:
                f["stoch_k"] = float(stoch["STOCHk_14_3_3"].iloc[-1])
                f["stoch_d"] = float(stoch["STOCHd_14_3_3"].iloc[-1])
            else:
                f["stoch_k"] = f["stoch_d"] = 50.0
        except Exception:
            f["stoch_k"] = f["stoch_d"] = 50.0

        try:
            f["roc_5"]  = float(ta.roc(w["close"], 5).iloc[-1])  if len(w) >= 5  else 0.0
            f["roc_10"] = float(ta.roc(w["close"], 10).iloc[-1]) if len(w) >= 10 else 0.0
        except Exception:
            f["roc_5"] = f["roc_10"] = 0.0

        # ── 4. TREND ──────────────────────────────────────────────────────────
        try:
            adx_df = ta.adx(w["high"], w["low"], w["close"], 14)
            if adx_df is not None:
                f["adx"]      = float(adx_df["ADX_14"].iloc[-1])
                f["plus_di"]  = float(adx_df["DMP_14"].iloc[-1])
                f["minus_di"] = float(adx_df["DMN_14"].iloc[-1])
                f["di_diff"]  = f["plus_di"] - f["minus_di"]
            else:
                f["adx"] = f["plus_di"] = f["minus_di"] = f["di_diff"] = 0.0
        except Exception:
            f["adx"] = f["plus_di"] = f["minus_di"] = f["di_diff"] = 0.0

        for period, key in [(9, "vs_ema9"), (20, "vs_ema20"), (50, "vs_ema50")]:
            try:
                ema = ta.ema(w["close"], period)
                if ema is not None and not ema.dropna().empty:
                    val = float(ema.iloc[-1])
                    f[key] = (float(c["close"]) - val) / max(val, 1) * 100
                else:
                    f[key] = 0.0
            except Exception:
                f[key] = 0.0

        try:
            ema9  = ta.ema(w["close"], 9)
            ema20 = ta.ema(w["close"], 20)
            f["ema9_slope"]  = float(ema9.iloc[-1]  - ema9.iloc[-5])  if ema9  is not None and len(ema9)  >= 5 else 0.0
            f["ema20_slope"] = float(ema20.iloc[-1] - ema20.iloc[-5]) if ema20 is not None and len(ema20) >= 5 else 0.0
        except Exception:
            f["ema9_slope"] = f["ema20_slope"] = 0.0

        # ── 5. VOLATILITY ─────────────────────────────────────────────────────
        try:
            atr = ta.atr(w["high"], w["low"], w["close"], 14)
            f["atr_pct"] = float(atr.iloc[-1]) / max(float(c["close"]), 1) * 100 if atr is not None else 0.0
        except Exception:
            f["atr_pct"] = 0.0

        try:
            bb = ta.bbands(w["close"], 20, 2.0)
            if bb is not None:
                u = float(bb["BBU_20_2.0"].iloc[-1])
                l = float(bb["BBL_20_2.0"].iloc[-1])
                f["bb_width"]    = (u - l) / max(float(c["close"]), 1) * 100
                f["bb_position"] = (float(c["close"]) - l) / max(u - l, 0.01)
            else:
                f["bb_width"] = f["bb_position"] = 0.0
        except Exception:
            f["bb_width"] = f["bb_position"] = 0.0

    else:
        # Basic fallback without TA-Lib
        for feat in ["rsi","rsi_slope","rsi_vs_50","macd_hist","macd_signal",
                     "stoch_k","stoch_d","roc_5","roc_10","adx","plus_di",
                     "minus_di","di_diff","vs_ema9","vs_ema20","vs_ema50",
                     "ema9_slope","ema20_slope","atr_pct","bb_width","bb_position"]:
            f[feat] = 0.0

    # ── 6. VOLUME ─────────────────────────────────────────────────────────────
    try:
        vol_ma = w["volume"].rolling(20).mean().iloc[-1]
        f["vol_ratio"] = float(c["volume"]) / max(float(vol_ma), 1)
        vol_recent = w["volume"].tail(5).mean()
        vol_prior  = w["volume"].iloc[-10:-5].mean()
        f["vol_trend"] = float(vol_recent) / max(float(vol_prior), 1)
    except Exception:
        f["vol_ratio"] = f["vol_trend"] = 1.0
        
    # ── NEW: ATR ratio (volatility expanding or contracting) ─────────────────
    if TA_AVAILABLE:
        try:
            atr_series = ta.atr(w["high"], w["low"], w["close"], 14)
            if atr_series is not None and len(atr_series.dropna()) >= 20:
                atr_now = float(atr_series.iloc[-1])
                atr_avg20 = float(atr_series.dropna().tail(20).mean())
                f["atr_ratio"] = atr_now / max(atr_avg20, 0.01)
            else:
                f["atr_ratio"] = 1.0
        except Exception:
            f["atr_ratio"] = 1.0
    else:
        f["atr_ratio"] = 1.0

    # ── NEW: VWAP distance ────────────────────────────────────────────────────
    try:
        tp = (w["high"] + w["low"] + w["close"]) / 3
        cum_vol = w["volume"].cumsum().iloc[-1]
        vwap = float((tp * w["volume"]).cumsum().iloc[-1] / max(cum_vol, 1))
        f["vs_vwap"] = (float(c["close"]) - vwap) / max(vwap, 1) * 100
    except Exception:
        f["vs_vwap"] = 0.0

    # ── NEW: Trend persistence count ──────────────────────────────────────────
    try:
        closes = w["close"].values
        direction = 1 if closes[-1] > closes[-2] else -1
        count = 1
        for i in range(2, min(len(closes), 10)):
            d = 1 if closes[-i] > closes[-i - 1] else -1
            if d == direction:
                count += 1
            else:
                break
        f["trend_persist"] = count
    except Exception:
        f["trend_persist"] = 1

    # ── NEW: Session bucket (categorical time) ────────────────────────────────
    try:
        ts = w.index[-1]
        h, m = int(ts.hour), int(ts.minute)
        total = h * 60 + m
        if total < 10 * 60 + 30:
            f["session_bucket"] = 1  # morning ORB
        elif total < 12 * 60:
            f["session_bucket"] = 2  # mid-morning
        elif total < 13 * 60 + 30:
            f["session_bucket"] = 3  # lunch
        else:
            f["session_bucket"] = 4  # afternoon
    except Exception:
        f["session_bucket"] = 2

    # ── 7. TIME ───────────────────────────────────────────────────────────────
    try:
        ts = w.index[-1]
        f["hour"]           = int(ts.hour)
        f["minute"]         = int(ts.minute)
        f["day_of_week"]    = int(ts.dayofweek)
        f["mins_from_open"] = max(0, (int(ts.hour) - 9) * 60 + int(ts.minute) - 15)
    except Exception:
        f["hour"] = f["minute"] = f["day_of_week"] = f["mins_from_open"] = 0

    # ── NEW: Regime context ───────────────────────────────────────────────────
    ri = kwargs.get("regime_info") or {}
    lbl = ri.get("label", "RANGING")
    f["regime_label_enc"] = 1.0 if lbl == "TRENDING" else (2.0 if lbl == "HIGH_VOLATILITY" else 0.0)
    f["regime_confidence"] = float(ri.get("confidence", 0.5))
    f["regime_atr_ratio"] = float(ri.get("atr_ratio", 1.0))

    # ── NEW: Setup / structure / liquidity context ───────────────────────────
    signal_context = kwargs.get("signal_context") or {}
    setup = (signal_context.get("setup") or {})
    weighted_vote = (signal_context.get("weighted_vote") or {})
    market_structure = (
        signal_context.get("market_structure")
        or setup.get("context", {}).get("market_structure")
        or {}
    )
    structure_state = market_structure.get("structure_state", {}) or {}
    liquidity_event = market_structure.get("liquidity_event", {}) or {}
    entry_validation = market_structure.get("entry_validation", {}) or {}

    setup_type = str(setup.get("setup_type", "unknown") or "unknown").lower()
    setup_type_map = {
        "breakout": 1.0,
        "trend_pullback": 2.0,
        "mean_reversion": 3.0,
        "reversal": 4.0,
        "continuation": 5.0,
    }
    f["setup_type_enc"] = setup_type_map.get(setup_type, 0.0)
    f["setup_strength"] = float(setup.get("setup_strength", 0.5) or 0.5)
    expected_move = float(setup.get("expected_move", 0.0) or 0.0)
    f["expected_move_pct"] = expected_move / max(float(c["close"]), 1.0) * 100.0
    f["winner_avg_weight"] = float(weighted_vote.get("winner_avg_weight", 1.0) or 1.0)

    bias = str(structure_state.get("bias", "UNKNOWN")).upper()
    if "BULL" in bias:
        f["structure_bias_enc"] = 1.0
    elif "BEAR" in bias:
        f["structure_bias_enc"] = -1.0
    else:
        f["structure_bias_enc"] = 0.0

    f["bos_flag"] = float(str(structure_state.get("bos", "NONE")).upper() != "NONE")
    f["choch_flag"] = float(str(structure_state.get("choch", "NONE")).upper() != "NONE")
    f["liquidity_event_flag"] = float(
        str(liquidity_event.get("type", "NONE")).upper() != "NONE"
    )
    f["liquidity_nearest_upper_pct"] = float(
        entry_validation.get("nearest_upper_liquidity", 0.0) or 0.0
    ) / max(float(c["close"]), 1.0) * 100.0
    f["liquidity_nearest_lower_pct"] = float(
        entry_validation.get("nearest_lower_liquidity", 0.0) or 0.0
    ) / max(float(c["close"]), 1.0) * 100.0
    f["middle_range_flag"] = float(bool(entry_validation.get("middle_range", False)))

    strategies = kwargs.get("strategies_fired") or signal_context.get("strategies_fired") or []
    strategy_set = {str(item).strip() for item in strategies if str(item).strip()}
    weak_pairs = [
        {"VWAP+EMA", "ADX+PSAR"},
        {"SuperTrend+RSI", "BBSqueeze"},
        {"BBSqueeze", "ADX+PSAR"},
    ]
    f["weak_pair_flag"] = float(any(pair.issubset(strategy_set) for pair in weak_pairs))

    # ── 8. SIGNAL META ────────────────────────────────────────────────────────
    f["strategy_conf"] = float(conf)
    f["votes"]         = int(votes)
    f["is_call"]       = int(direction == "BUY_CALL")

    # ── CLEAN: replace NaN/Inf ────────────────────────────────────────────────
    cleaned = {}
    for k, v in f.items():
        try:
            fv = float(v)
            cleaned[k] = 0.0 if (np.isnan(fv) or np.isinf(fv)) else fv
        except Exception:
            cleaned[k] = 0.0

    return cleaned


def to_dataframe(feature_dict: dict) -> pd.DataFrame:
    """Convert feature dict to DataFrame with correct column order."""
    cols = [c for c in FEATURE_NAMES if c in feature_dict]
    return pd.DataFrame([feature_dict])[cols].fillna(0.0)


def feature_count() -> int:
    return len(FEATURE_NAMES)

import os
from config.settings.utils import *


# ── ML QUALITY ──
ML_QUALITY_MIN_SCORE = 0.55
ML_QUALITY_MIN_PREMIUM = float(os.getenv("ML_QUALITY_MIN_PREMIUM", "50.0"))
ML_QUALITY_DIVERSITY_BONUS = float(os.getenv("ML_QUALITY_DIVERSITY_BONUS", "0.05"))

# Additional ML Quality Settings
ML_MIN_TOTAL_SCORE = 0.58
ML_CHEAP_OPTION_PENALTY_FACTOR = float(os.getenv("ML_CHEAP_OPTION_PENALTY_FACTOR", "0.75"))
ML_MAX_TOTAL_CONFIDENCE = float(os.getenv("ML_MAX_TOTAL_CONFIDENCE", "0.95"))

# ── ML SCORER TIMING SCORES (h0, m0, h1, m1) ──
ML_TIMING_SCORES = { (12, 0, 13, 30): 0.8, 
    (9, 30, 10, 30): 1.0,   # ORB session — fresh momentum
    (10, 30, 12, 0): 0.9,   # mid-morning — good follow-through
    (12, 0, 13, 30): 0.8,   # LUNCH CHOP — Low score to only allow top-tier setups (was 0.0)
    (13, 30, 14, 30): 0.85,  # afternoon re-entry — decent
    (14, 30, 15, 0): 0.75,  # late afternoon — theta decay accelerates
}


ML_TIMING_FALLBACK_SESSION_SCORE = float(os.getenv("ML_TIMING_FALLBACK_SESSION_SCORE", "0.85")) # Before 3 PM, general positive bias
ML_TIMING_FALLBACK_DEFAULT_SCORE = float(os.getenv("ML_TIMING_FALLBACK_DEFAULT_SCORE", "0.80"))   # after 3 PM or before 9 AM

# ── ML FEATURES ──
ML_FEATURE_LOOKBACK = int(os.getenv("ML_FEATURE_LOOKBACK", "60"))
ML_FEATURE_MIN_DF_LEN = int(os.getenv("ML_FEATURE_MIN_DF_LEN", "30"))
ML_FEATURE_RSI_PERIOD = int(os.getenv("ML_FEATURE_RSI_PERIOD", "14"))
ML_FEATURE_EMA_PERIODS = [9, 20, 50]
ML_FEATURE_BB_PERIOD = int(os.getenv("ML_FEATURE_BB_PERIOD", "20"))
ML_FEATURE_BB_STD = float(os.getenv("ML_FEATURE_BB_STD", "2.0"))
ML_FEATURE_VOL_MA_PERIOD = int(os.getenv("ML_FEATURE_VOL_MA_PERIOD", "20"))

# ── ML ENSEMBLE ──
ML_ENSEMBLE_WEIGHTS = {"xgb": 0.40, "lgb": 0.40, "rf": 0.20}
ML_ENSEMBLE_DEFAULT_THRESHOLD = 0.55
ML_CV_SPLITS = int(os.getenv("ML_CV_SPLITS", "5"))

# ── ML FILTER ──
ML_MIN_CONFIDENCE    = float(os.getenv("ML_MIN_CONFIDENCE", "0.22"))   # 0.22 empirical alpha gate for commodities
ML_MIN_MODEL_CONFIDENCE = float(os.getenv("ML_MIN_MODEL_CONFIDENCE", "0.22"))
ML_TREND_PULLBACK_MIN_CONFIDENCE = float(os.getenv("ML_TREND_PULLBACK_MIN_CONFIDENCE", "0.22"))
ML_BREAKOUT_MIN_CONFIDENCE = float(os.getenv("ML_BREAKOUT_MIN_CONFIDENCE", "0.22"))
ML_THRESHOLD_OVERRIDE = float(os.getenv("ML_THRESHOLD_OVERRIDE", "0.22"))

ML_LOOKBACK_CANDLES  = 60
ML_MODELS_DIR        = "ml/saved_models"
ML_RETRAIN_DAYS      = int(os.getenv("ML_RETRAIN_DAYS", "90"))
ML_CALIBRATION_ENABLED = _flag("ML_CALIBRATION_ENABLED", "false")
ML_LIVE_PRECISION_WINDOW = int(os.getenv("ML_LIVE_PRECISION_WINDOW", "25"))
ML_LIVE_MIN_PRECISION = float(os.getenv("ML_LIVE_MIN_PRECISION", "0.45"))
ML_LIVE_MIN_EVAL_TRADES = int(os.getenv("ML_LIVE_MIN_EVAL_TRADES", "12"))
ML_FALLBACK_MIN_CONFIDENCE = float(os.getenv("ML_FALLBACK_MIN_CONFIDENCE", "0.68"))
ML_FALLBACK_ALLOW_RANGING = _flag("ML_FALLBACK_ALLOW_RANGING", "false")
ML_SECONDARY_THRESHOLD = float(os.getenv("ML_SECONDARY_THRESHOLD", "0.45"))
ML_SECONDARY_MIN_RAW_CONFIDENCE = float(
    os.getenv("ML_SECONDARY_MIN_RAW_CONFIDENCE", "0.72")
)
ML_SECONDARY_MIN_VOTES = int(os.getenv("ML_SECONDARY_MIN_VOTES", "2"))
ML_RANK_SOFT_THRESHOLD = 0.32
ML_RANK_HIGH_THRESHOLD = float(os.getenv("ML_RANK_HIGH_THRESHOLD", "0.70"))
ML_RANK_MODEL_WEIGHT = float(os.getenv("ML_RANK_MODEL_WEIGHT", "0.60"))
ML_RANK_STRATEGY_WEIGHT = float(os.getenv("ML_RANK_STRATEGY_WEIGHT", "0.22"))
ML_RANK_VOTE_WEIGHT = float(os.getenv("ML_RANK_VOTE_WEIGHT", "0.10"))
ML_RANK_REGIME_WEIGHT = float(os.getenv("ML_RANK_REGIME_WEIGHT", "0.08"))
ML_SECONDARY_ALLOWED_STRATEGY_PAIRS = tuple(
    pair.strip()
    for pair in os.getenv(
        "ML_SECONDARY_ALLOWED_STRATEGY_PAIRS",
        (
            "SuperTrend+RSI|ADX+PSAR,"
            "SuperTrend+RSI|BBSqueeze,"
            "ADX+PSAR|BBSqueeze,"
            "VWAP+EMA|ADX+PSAR,"
            "ORB|ADX+PSAR"
        ),
    ).split(",")
    if pair.strip()
)
ML_BLOCKED_STRATEGY_PAIRS = ()
ML_FALLBACK_BLOCKED_STRATEGY_PAIRS = ()
SIGNAL_APPROVAL_DAILY_BUDGET = 15
SIGNAL_SESSION_BUDGETS = _parse_budget_map(
    os.getenv("SIGNAL_SESSION_BUDGETS", ""),
    {"OPENING": 4, "MIDDAY": 4, "CLOSING": 6},
)
SIGNAL_REGIME_BUDGETS = _parse_budget_map(os.getenv("SIGNAL_REGIME_BUDGETS", ""), {"TRENDING": 10, "RANGING": 5, "CHOPPY": 3, "HIGH_VOL": 2})

# ── ML FEATURES ──
ML_FEATURE_RETURN_PERIODS = [1, 3, 5, 10, 20]
ML_FEATURE_RSI_SLOPE_PERIOD = 5
ML_FEATURE_EMA_SLOPE_PERIOD = 5
ML_FEATURE_VOL_TREND_PERIODS = (5, 10)
ML_FEATURE_ATR_RATIO_PERIOD = 20
ML_FEATURE_TREND_PERSIST_LIMIT = 10
ML_FEATURE_SESSION_MORNING_LIMIT = 10 * 60 + 30
ML_FEATURE_SESSION_MID_LIMIT = 12 * 60
ML_FEATURE_SESSION_LUNCH_LIMIT = 13 * 60 + 30

# ── PRE-MARKET RISK GATE ──
PREMARKET_MAX_VIX = 28.0  # don't trade if India VIX > 28
PREMARKET_MAX_GAP_PCT = 1.5  # reduce size if gap > 1.5%
PREMARKET_REQUIRE_SGX_DATA = False  # set True when SGX Nifty feed available

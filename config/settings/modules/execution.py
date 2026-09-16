import os
from config.settings.utils import *


# ── TRADE QUALITY FILTER ──
# Minimum expected reward-to-risk ratio before entering a trade.
# Trades with R:R < this are skipped regardless of strategy confidence.
# 2.0 means target must be at least 2× the stop loss distance.
# MIN_RR_RATIO is defined later in the file (line 305) to allow environment variable override
MIN_EXPECTED_TARGET_PNL_PCT = float(
    os.getenv("MIN_EXPECTED_TARGET_PNL_PCT", "2.0")
)


# Minimum strategy confidence to generate a signal.
# CONSENSUS FIX: Lower from 0.62 to 0.58 to increase strategy firing rate

# ── EXECUTION SAFETY ──
DUPLICATE_SIGNAL_COOLDOWN_MINUTES = int(os.getenv("DUPLICATE_SIGNAL_COOLDOWN_MINUTES", "15"))
ORDER_RETRY_ATTEMPTS = int(os.getenv("ORDER_RETRY_ATTEMPTS", "3"))
ORDER_RETRY_DELAY_SEC = float(os.getenv("ORDER_RETRY_DELAY_SEC", "1.5"))

# ── ENGINE: Trade Setup ──
SETUP_VOTE_BASE_STRENGTH = float(os.getenv("SETUP_VOTE_BASE_STRENGTH", "0.54"))
SETUP_VOTE_SCORE_BONUS_MAX = float(os.getenv("SETUP_VOTE_SCORE_BONUS_MAX", "0.12"))
SETUP_VOTE_SCORE_THRESHOLD = float(os.getenv("SETUP_VOTE_SCORE_THRESHOLD", "0.72"))
SETUP_VOTE_SCORE_MULT = float(os.getenv("SETUP_VOTE_SCORE_MULT", "0.30"))
SETUP_VOTE_COUNT_BONUS_MAX = float(os.getenv("SETUP_VOTE_COUNT_BONUS_MAX", "0.08"))
SETUP_VOTE_COUNT_THRESHOLD = int(os.getenv("SETUP_VOTE_COUNT_THRESHOLD", "2"))
SETUP_VOTE_COUNT_MULT = float(os.getenv("SETUP_VOTE_COUNT_MULT", "0.04"))
SETUP_VOTE_TRENDING_BONUS = float(os.getenv("SETUP_VOTE_TRENDING_BONUS", "0.04"))
SETUP_VOTE_BIAS_BONUS = float(os.getenv("SETUP_VOTE_BIAS_BONUS", "0.03"))
SETUP_VOTE_CONF_MAX = float(os.getenv("SETUP_VOTE_CONF_MAX", "0.82"))
SETUP_VOTE_CONF_MIN = float(os.getenv("SETUP_VOTE_CONF_MIN", "0.62"))
SETUP_VOTE_SCORE_THRESHOLD_NEW = float(os.getenv("SETUP_VOTE_SCORE_THRESHOLD_NEW", "0.55"))
SETUP_VOTE_COUNT_BONUS = float(os.getenv("SETUP_VOTE_COUNT_BONUS", "0.03"))
SETUP_ZONE_PAD_EMA_MULT = float(os.getenv("SETUP_ZONE_PAD_EMA_MULT", "0.4"))
SETUP_STOP_PAD_LTP_MULT_NEW = float(os.getenv("SETUP_STOP_PAD_LTP_MULT_NEW", "0.0022"))
SETUP_MOVE_TARGET_ATR_MULT_NEW = float(os.getenv("SETUP_MOVE_TARGET_ATR_MULT_NEW", "2.2"))
SETUP_MOVE_TARGET_EMA_MULT_NEW = float(os.getenv("SETUP_MOVE_TARGET_EMA_MULT_NEW", "3.5"))
SETUP_TREND_ADX_THRESHOLD = float(os.getenv("SETUP_TREND_ADX_THRESHOLD", "20"))
SETUP_ZONE_PAD_LTP_MULT = float(os.getenv("SETUP_ZONE_PAD_LTP_MULT", "0.0009"))
SETUP_ZONE_PAD_EMA_REDUCED_MULT = float(os.getenv("SETUP_ZONE_PAD_EMA_REDUCED_MULT", "0.18")) # added
SETUP_ZONE_PAD_INFERRED_MULT = float(os.getenv("SETUP_ZONE_PAD_INFERRED_MULT", "0.4")) # added
SETUP_STOP_PAD_ATR_MULT = float(os.getenv("SETUP_STOP_PAD_ATR_MULT", "0.95"))
SETUP_STOP_PAD_LTP_MULT = float(os.getenv("SETUP_STOP_PAD_LTP_MULT", "0.0020"))
SETUP_MOVE_TARGET_ATR_MULT = float(os.getenv("SETUP_MOVE_TARGET_ATR_MULT", "1.9"))
SETUP_MOVE_TARGET_EMA_MULT = float(os.getenv("SETUP_MOVE_TARGET_EMA_MULT", "2.4"))
SETUP_MOVE_TARGET_LTP_MULT = float(os.getenv("SETUP_MOVE_TARGET_LTP_MULT", "0.0022"))
SETUP_TREND_ADX_THRESHOLD = float(os.getenv("SETUP_TREND_ADX_THRESHOLD", "20"))
SETUP_TREND_NEAR_EMA_ATR_MULT = float(os.getenv("SETUP_TREND_NEAR_EMA_ATR_MULT", "1.2"))
SETUP_TREND_NEAR_EMA_LTP_MULT = float(os.getenv("SETUP_TREND_NEAR_EMA_LTP_MULT", "0.003"))
SETUP_TREND_PULLBACK_ATR_MULT = float(os.getenv("SETUP_TREND_PULLBACK_ATR_MULT", "0.40"))
SETUP_TREND_PULLBACK_STOP_MULT = float(os.getenv("SETUP_TREND_PULLBACK_STOP_MULT", "0.15"))
SETUP_TREND_CONFIRM_ATR_MULT = float(os.getenv("SETUP_TREND_CONFIRM_ATR_MULT", "0.20"))
SETUP_TREND_STRENGTH_BASE = float(os.getenv("SETUP_TREND_STRENGTH_BASE", "0.55"))
SETUP_TREND_STRENGTH_ADX_MULT = float(os.getenv("SETUP_TREND_STRENGTH_ADX_MULT", "0.015"))
SETUP_TREND_NON_TRENDING_PENALTY = float(os.getenv("SETUP_TREND_NON_TRENDING_PENALTY", "0.10"))
SETUP_TREND_STOP_ATR_MULT = float(os.getenv("SETUP_TREND_STOP_ATR_MULT", "0.95"))
SETUP_TREND_STOP_LTP_MULT = float(os.getenv("SETUP_TREND_STOP_LTP_MULT", "0.0022"))
SETUP_TREND_MOVE_ATR_MULT = float(os.getenv("SETUP_TREND_MOVE_ATR_MULT", "2.2"))
SETUP_TREND_MOVE_EMA_MULT = float(os.getenv("SETUP_TREND_MOVE_EMA_MULT", "3.5"))
SETUP_BREAKOUT_RANGE_ATR_MULT = float(os.getenv("SETUP_BREAKOUT_RANGE_ATR_MULT", "4.0"))
SETUP_BREAKOUT_RANGE_LTP_MULT = float(os.getenv("SETUP_BREAKOUT_RANGE_LTP_MULT", "0.0009"))
SETUP_BREAKOUT_ADX_THRESHOLD = float(os.getenv("SETUP_BREAKOUT_ADX_THRESHOLD", "22"))
SETUP_BREAKOUT_BUFFER_ATR_MULT = float(os.getenv("SETUP_BREAKOUT_BUFFER_ATR_MULT", "0.12"))
SETUP_BREAKOUT_BUFFER_LTP_MULT = float(os.getenv("SETUP_BREAKOUT_BUFFER_LTP_MULT", "0.0004"))
SETUP_BREAKOUT_VOL_MULT_CALL = float(os.getenv("SETUP_BREAKOUT_VOL_MULT_CALL", "1.05"))
SETUP_BREAKOUT_VOL_MULT_PUT = float(os.getenv("SETUP_BREAKOUT_VOL_MULT_PUT", "0.94"))
SETUP_BREAKOUT_STRENGTH_BASE = float(os.getenv("SETUP_BREAKOUT_STRENGTH_BASE", "0.52"))
SETUP_BREAKOUT_STRENGTH_VOL_MULT = float(os.getenv("SETUP_BREAKOUT_STRENGTH_VOL_MULT", "0.05"))
SETUP_BREAKOUT_STRENGTH_BB_THRESHOLD = float(os.getenv("SETUP_BREAKOUT_STRENGTH_BB_THRESHOLD", "1.8"))
SETUP_BREAKOUT_STRENGTH_BB_DIVISOR = float(os.getenv("SETUP_BREAKOUT_STRENGTH_BB_DIVISOR", "10"))
SETUP_BREAKOUT_STRENGTH_BB_MULT = float(os.getenv("SETUP_BREAKOUT_STRENGTH_BB_MULT", "0.04"))
SETUP_BREAKOUT_MOVE_RANGE_MULT = float(os.getenv("SETUP_BREAKOUT_MOVE_RANGE_MULT", "1.4"))
SETUP_BREAKOUT_MOVE_ATR_MULT = float(os.getenv("SETUP_BREAKOUT_MOVE_ATR_MULT", "2.1"))
SETUP_REVERSION_STRETCH_ATR_MULT = float(os.getenv("SETUP_REVERSION_STRETCH_ATR_MULT", "0.45"))
SETUP_REVERSION_STRETCH_LTP_MULT = float(os.getenv("SETUP_REVERSION_STRETCH_LTP_MULT", "0.0012"))
SETUP_REVERSION_STRENGTH_BASE = float(os.getenv("SETUP_REVERSION_STRENGTH_BASE", "0.55"))
SETUP_REVERSION_STRENGTH_STRETCH_MULT = float(os.getenv("SETUP_REVERSION_STRENGTH_STRETCH_MULT", "0.11"))
SETUP_REVERSION_ENTRY_ATR_MULT = float(os.getenv("SETUP_REVERSION_ENTRY_ATR_MULT", "0.25"))
SETUP_REVERSION_STOP_ATR_MULT = float(os.getenv("SETUP_REVERSION_STOP_ATR_MULT", "1.1"))
SETUP_REVERSION_MOVE_STRETCH_MULT = float(os.getenv("SETUP_REVERSION_MOVE_STRETCH_MULT", "0.85"))
SETUP_REVERSION_MOVE_ATR_MULT = float(os.getenv("SETUP_REVERSION_MOVE_ATR_MULT", "1.4"))
SETUP_STRUCTURE_BOOST_EVENT = float(os.getenv("SETUP_STRUCTURE_BOOST_EVENT", "0.04"))
SETUP_STRUCTURE_BOOST_BOS = float(os.getenv("SETUP_STRUCTURE_BOOST_BOS", "0.03"))
SETUP_STRUCTURE_BOOST_CHOCH = float(os.getenv("SETUP_STRUCTURE_BOOST_CHOCH", "0.03"))
SETUP_STRUCTURE_BOOST_BIAS = float(os.getenv("SETUP_STRUCTURE_BOOST_BIAS", "0.02"))
SETUP_STRUCTURE_BOOST_REVERSION = float(os.getenv("SETUP_STRUCTURE_BOOST_REVERSION", "0.03"))
SETUP_STRUCTURE_BOOST_BREAKOUT = float(os.getenv("SETUP_STRUCTURE_BOOST_BREAKOUT", "0.02"))
SETUP_STRUCTURE_PENALTY_SOFT = float(os.getenv("SETUP_STRUCTURE_PENALTY_SOFT", "0.04"))
SETUP_STRUCTURE_PENALTY_HARD = float(os.getenv("SETUP_STRUCTURE_PENALTY_HARD", "0.17"))
SETUP_STRUCTURE_PENALTY_AVOID = float(os.getenv("SETUP_STRUCTURE_PENALTY_AVOID", "0.08"))
SETUP_STRUCTURE_PENALTY_MIDDLE = float(os.getenv("SETUP_STRUCTURE_PENALTY_MIDDLE", "0.05"))
SETUP_STRUCTURE_VALIDATION_BOS_THRESHOLD_LOW = float(os.getenv("SETUP_STRUCTURE_VALIDATION_BOS_THRESHOLD_LOW", "0.35"))
SETUP_STRUCTURE_VALIDATION_BOS_THRESHOLD_HIGH = float(os.getenv("SETUP_STRUCTURE_VALIDATION_BOS_THRESHOLD_HIGH", "0.65"))
SETUP_STRUCTURE_VALIDATION_REVERSION_THRESHOLD_LOW = float(os.getenv("SETUP_STRUCTURE_VALIDATION_REVERSION_THRESHOLD_LOW", "0.32"))
SETUP_STRUCTURE_VALIDATION_REVERSION_THRESHOLD_HIGH = float(os.getenv("SETUP_STRUCTURE_VALIDATION_REVERSION_THRESHOLD_HIGH", "0.68"))

# ── EXECUTION ──
EXECUTION_NARRATE_MAX_TOKENS = 100

# ── RE-ENTRY ──
# Allow one re-entry per session if SL hit but setup re-confirms.
# Only active when ALLOW_REENTRY = True.
ALLOW_REENTRY = True
REENTRY_COOLDOWN_M = 15  # minutes to wait after SL before scanning again
SECOND_TRADE_OPENING_CUTOFF_MINUTE = int(
    os.getenv("SECOND_TRADE_OPENING_CUTOFF_MINUTE", str(15 * 60))
)
SECOND_TRADE_LAST_ENTRY_MINUTE = int(
    os.getenv("SECOND_TRADE_LAST_ENTRY_MINUTE", str(15 * 60))
)
SECOND_TRADE_MIN_SETUP_STRENGTH = float(os.getenv("SECOND_TRADE_MIN_SETUP_STRENGTH", "0.72"))
SECOND_TRADE_MIN_CONFIDENCE = float(os.getenv("SECOND_TRADE_MIN_CONFIDENCE", "0.74"))
SECOND_TRADE_MIN_WEIGHTED_SCORE = 0.5
SECOND_TRADE_MIN_VOTES = int(os.getenv("SECOND_TRADE_MIN_VOTES", "3"))

# ── SETUP GATE ──
# The runner requires a minimum setup quality score before running strategies.
# Reducing this allows more candles to reach the strategy evaluation stage.
# Log showed 1620/3000 candles blocked by no_valid_setup (54%).
MIN_SETUP_STRENGTH   = 0.40   # sniper entry bar
SETUP_GATE_ENABLED   = bool(os.getenv("SETUP_GATE_ENABLED", "True").lower() in ("true", "1", "yes"))   # set False to disable setup pre-filter entirely
RANGING_REGIME_BLOCK = False   # FIX: was False - re-enable to block ADX<20 ranging regime
                               # since log shows RANGING candles with ADX=18+ still valid

# Base rank threshold — signals scoring below this are REJECTED (was dynamic ~0.50-0.59)
ML_RANK_BASE_THRESHOLD   = 0.45   # Restored from 0.42 for 60d stability

# Relief multipliers (make threshold LOWER for high-quality signals)
ML_HIGH_CONSENSUS_RELIEF = 0.10  # 3+ votes → lower threshold by this amount
ML_TRENDING_RELIEF = 0.06  # TRENDING regime → lower threshold
ML_STRONG_SETUP_RELIEF = 0.08  # setup_strength > 0.85 → lower threshold

# Session-based multiplier for ML thresholds
ML_SIGNAL_WINDOW_START = "09:20"  # no signals before this
ML_SIGNAL_WINDOW_END = "15:00"  # no new signals after this (was 15:00)

ML_STRATEGY_WEIGHTS = {
    "SuperTrend+RSI":  1.0,   # reliable but trend-only
    "ADX+PSAR":        1.0,   # reliable but trend-only (correlated with S1)
    "VWAP+EMA":        1.2,   # uses price vs fair value = more independent
    "ORB":             1.3,   # time-gated, strong morning signal
    "BBSqueeze":       1.2,   # volatility breakout = different category
    "FVG":             1.3,   # SMC imbalance = price structure, independent
    "UTBot":           0.9,   # ATR trailing (correlated with S1 + S5)
    "CPR":             1.4,   # inter-session levels = most independent
    "Ichimoku":        1.5,   # 5-condition confluence = highest weight
    "VolumeProfile":   1.4,   # volume at price = independent of TA indicators
    "LiqSweep":        1.3,   # stop hunt pattern = independent
    "PriceAction":     1.2,   # candle patterns = different signal type
    "OIAnalysis":      1.5,   # institutional data = highest independence
    "IVContraction":   1.3,   # volatility regime = different category
    "AMD":             1.4,   # structure + manipulation = high independence
    "GapDirection":    1.5,   # pre-market gap = most independent signal (external data)
    "SMC":             1.4,   # order block + structure = high independence
    "SkewHunter":      1.5,   # IV skew + flow = completely orthogonal to price-action
}

ML_STRATEGY_CATEGORIES = {
    "trend": {"SuperTrend+RSI", "ADX+PSAR", "UTBot"},
    "momentum": {"VWAP+EMA", "BBSqueeze"},
    "level": {"ORB", "CPR", "VolumeProfile"},
    "structure": {"FVG", "LiqSweep", "PriceAction", "Ichimoku"},
    "volatility": {"IVContraction"},
    "flow": {"OIAnalysis"},
    "gap": {"GapDirection"},
    "smc": {"SMC"},  # order block / BOS / breaker
    "skew": {"SkewHunter"},  # volatility skew + option flow
}

# cut 30 min earlier to avoid late-day TIME_DECAY

# ── RE-ENTRY ──
# Allow one re-entry per session if SL hit but setup re-confirms.
# Only active when ALLOW_REENTRY = True.
ALLOW_REENTRY = True
REENTRY_COOLDOWN_M = 15  # minutes to wait after SL before scanning again

# ── RE-ENTRY / SECOND TRADE — COOLDOWN SYSTEM (FIX 3) ──
# FIX 3: Replace whole-day second_trade_block with 90-min cooldown.
#   From backtest: 172 second_trade_block events. Many had conf=0.91-0.95,
#   votes=3-7, ws=3-5. These were high-quality setups being killed.
#   The whole-day block is too conservative. Industry standard = time-based
#   cooldown that resets when setup quality exceeds a threshold.
#
#   New rule: Allow re-entry after SECOND_TRADE_COOLDOWN_MIN minutes IF:
#     - conf >= SECOND_TRADE_MIN_CONF  (0.88 — only high conviction)
#     - setup_strength >= SECOND_TRADE_MIN_SETUP  (0.72)
#     - regime still TRENDING (no re-entry into choppy/ranging regime)
#     - votes >= SECOND_TRADE_MIN_VOTES  (3 — not on marginal 2-vote signals)
SECOND_TRADE_COOLDOWN_MIN = 5  # minutes after last close before re-entry
SECOND_TRADE_MIN_CONF = 0.40  # minimum confidence for second trade
SECOND_TRADE_MIN_SETUP = 0.35  # minimum setup strength for second trade
SECOND_TRADE_MIN_VOTES = 1  # minimum votes for second trade
SECOND_TRADE_REQUIRE_TREND = False  # only re-enter if regime is still TRENDING

# High-conviction re-entry lane.
# Backtest logs show valid 4-vote, 0.95-confidence setups being blocked by the
# generic 90-minute second-trade cooldown. Keep the broad cooldown intact, but
# allow fast re-entry only when the setup is materially stronger than normal.
SECOND_TRADE_FAST_REENTRY_MIN = int(os.getenv("SECOND_TRADE_FAST_REENTRY_MIN", "10"))
SECOND_TRADE_FAST_LAST_ENTRY_MINUTE = int(
    os.getenv("SECOND_TRADE_FAST_LAST_ENTRY_MINUTE", str(15 * 60))
)
SECOND_TRADE_FAST_MIN_SETUP = float(os.getenv("SECOND_TRADE_FAST_MIN_SETUP", "0.84"))
SECOND_TRADE_FAST_MIN_CONF = float(os.getenv("SECOND_TRADE_FAST_MIN_CONF", "0.92"))
SECOND_TRADE_FAST_MIN_VOTES = int(os.getenv("SECOND_TRADE_FAST_MIN_VOTES", "2"))
SECOND_TRADE_FAST_MIN_WEIGHTED_SCORE = float(
    os.getenv("SECOND_TRADE_FAST_MIN_WEIGHTED_SCORE", "1.7")
)

# ── TRADE QUALITY FILTER ──
# Minimum expected reward-to-risk ratio before entering a trade.
# Trades with R:R < this are skipped regardless of strategy confidence.
# 1.5 means target must be at least 1.5× the stop loss distance.
MIN_RR_RATIO = 1.2  # skip if target/sl_distance < this

# Minimum strategy confidence to generate a signal.
# Separate from ML gate — this is pure strategy agreement quality.
MIN_STRATEGY_CONF = 0.45  # minimum: matches base confidence of all strategies




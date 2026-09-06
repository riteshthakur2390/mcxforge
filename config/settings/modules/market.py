import os
from config.settings.utils import *


# ── MARKET STRUCTURE ──
MARKET_STRUCTURE_ATR_PERIOD = 14
MARKET_STRUCTURE_TOLERANCE_ATR_MULTIPLIER = 0.18
MARKET_STRUCTURE_TOLERANCE_CLOSE_MULTIPLIER = 0.0008
MARKET_STRUCTURE_TOLERANCE_MIN = 0.15
MARKET_STRUCTURE_LOOKBACK = 80
MARKET_STRUCTURE_SWING_SPAN = 2
MARKET_STRUCTURE_BOS_TOLERANCE_MULTIPLIER = 0.35
MARKET_STRUCTURE_ZONE_CLUSTER_MAX_SWINGS = 8
MARKET_STRUCTURE_MAX_LIQUIDITY_ZONES = 8
MARKET_STRUCTURE_LIQUIDITY_EVENT_LOOKBACK = 10
MARKET_STRUCTURE_ENTRY_VALIDATION_LOOKBACK = 20
MARKET_STRUCTURE_RANGE_POS_MIDDLE_LOWER = 0.38
MARKET_STRUCTURE_RANGE_POS_MIDDLE_UPPER = 0.62
MARKET_STRUCTURE_DEFAULT_ZONE_DISTANCE = 999999.0
MARKET_STRUCTURE_ZONE_BUFFER_ATR_MULTIPLIER = 0.35
MARKET_STRUCTURE_ZONE_BUFFER_CLOSE_MULTIPLIER = 0.0015
MARKET_STRUCTURE_ZONE_BUFFER_MIN = 0.3

# ── REGIME FILTER ──
# FIX: Relaxed thresholds to reduce over-suppression in choppy markets.
ADX_TREND_THRESHOLD  = 10           # ADX > 17
ADX_CHOP_THRESHOLD   = 11           # ADX < 11
CHOP_INDEX_THRESHOLD = 68.0         # Choppiness > 68.0
VIX_HIGH_THRESHOLD   = 30.0         # VIX > 30 → suppress all

# ── TREND PERSISTENCE ──
# Minimum consecutive same-direction candles before a signal is valid.
# Prevents entering on single-candle fakeouts. 3 = 15 minutes of trend.
TREND_PERSIST_CANDLES = 0 # disabled: strategies self-confirm trend

# ── GAP BIAS ──
# If NIFTY gaps > this % at open, bias first signal toward gap direction.
# 0.3% = ~66 points on NIFTY at 22000. Meaningful gap.
GAP_BIAS_THRESHOLD_PCT = 0.3

# ── TREND PERSISTENCE ──
# Minimum consecutive same-direction candles before a signal is valid.
# Prevents entering on single-candle fakeouts. 3 = 15 minutes of trend.
TREND_PERSIST_CANDLES = 0  # disabled: strategies self-confirm trend

# ── GAP BIAS ──
# If NIFTY gaps > this % at open, bias first signal toward gap direction.
# 0.3% = ~66 points on NIFTY at 22000. Meaningful gap.
GAP_BIAS_THRESHOLD_PCT = 0.3

# ── ENTRY REGIME FILTER (FIX 5) ──
ENTRY_MIN_DET_CONF = 0.60
ENTRY_BLOCK_RANGING = False

# ── MARKET INTELLIGENCE ENGINE (PCR + OI + FII + VIX + Trap Detection) ───────
# PCR is the first plugin in an extensible Market Context Engine. Every
# signal gets a Trade Quality Score (0-100). By default this is purely
# INFORMATIONAL — set > 0 to start suppressing low-quality signals below
# this threshold (spec: "SignalForge should optionally suppress poor-quality
# signals. Threshold must be configurable.")
SIGNAL_SUPPRESS_BELOW_QUALITY = float(os.getenv("SIGNAL_SUPPRESS_BELOW_QUALITY", "0.0"))   # 0 = never suppress; try 70.0 once tuned
MARKET_INTEL_LOG_ALL_SCORES   = _flag("MARKET_INTEL_LOG_ALL_SCORES", "true")   # log every plugin's contribution for explainability


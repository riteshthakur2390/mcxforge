import os
from config.settings.utils import *


# ── SIGNAL SETTINGS ──
# SIGNAL_TIMEFRAME     = os.getenv("SIGNAL_TIMEFRAME", "5minute").strip()
BACKTEST_TIMEFRAME   = os.getenv(
    "BACKTEST_TIMEFRAME",
    os.getenv("BACKTEST_SIGNAL", "5minute"),
).strip()
BACKTEST_SIGNAL      = BACKTEST_TIMEFRAME
LIVE_TIMEFRAME       = os.getenv("LIVE_TIMEFRAME", os.getenv("SIGNAL_TIMEFRAME", "5minute")).strip()
CANDLE_LOOKBACK      = int(os.getenv("CANDLE_LOOKBACK", "120"))

DASHBOARD_TICK_INTERVAL_SEC = int(os.getenv("DASHBOARD_TICK_INTERVAL_SEC", "10"))
ONE_SIGNAL_AT_A_TIME = False
MAX_CONCURRENT_SIGNALS = 3


# ── AUTOMATION READINESS ──
AUTOMATION_MIN_LIVE_TRADES = int(os.getenv("AUTOMATION_MIN_LIVE_TRADES", "20"))
AUTOMATION_MIN_BACKTEST_TRADES = int(os.getenv("AUTOMATION_MIN_BACKTEST_TRADES", "20"))
AUTOMATION_MIN_WIN_RATE_PCT = float(os.getenv("AUTOMATION_MIN_WIN_RATE_PCT", "55.0"))
AUTOMATION_MIN_PROFIT_FACTOR = float(os.getenv("AUTOMATION_MIN_PROFIT_FACTOR", "1.5"))
AUTOMATION_MAX_DRAWDOWN_PCT = float(os.getenv("AUTOMATION_MAX_DRAWDOWN_PCT", "8.0"))
AUTOMATION_MAX_WIN_RATE_GAP_PCT = float(os.getenv("AUTOMATION_MAX_WIN_RATE_GAP_PCT", "12.0"))

# ── EXPIRY DAY ──
# Tuesday = NIFTY weekly expiry. Friday = SENSEX weekly expiry.
# Gamma is highest on expiry; only strong directional momentum should pass.
EXPIRY_AFTERNOON_START = "13:00"

# ── ADDITIONAL ENGINE CONSTANTS ──
SETUP_REVERSION_ZONE_THRESHOLD_LOW = 0.30
SETUP_REVERSION_ZONE_THRESHOLD_HIGH = 0.70
SETUP_REVERSION_STRETCH_RATIO_MAX = 2.0
SETUP_BREAKOUT_SCAN_LOOKBACK = 9
SETUP_REVERSION_CLOSE_LOCATION_THRESHOLD = 0.50
SETUP_REVERSION_MIN_ATR = 0.01

# ── EXPIRY DAY ──
# Tuesday = NIFTY weekly expiry. Gamma is highest, moves are sharper.
# Allow signals from 13:00 on expiry days (afternoon momentum window).
EXPIRY_AFTERNOON_START = "13:00"

# ── TIME-OF-DAY PERFORMANCE SEGMENTATION ──
# Tracks win rate per 30-min bucket per strategy
# Auto-reduces position size in historically weak time windows
ENABLE_TOD_FILTER = False  # enable time-of-day filtering
TOD_MIN_TRADES = 10  # need >= 10 historical trades per bucket to filter
TOD_WEAK_WIN_RATE = 0.35  # bucket with < 35% WR = weak, reduce size 50%
TOD_BUCKETS = [
    "09:15", "09:45", "10:15", "10:45",
    "11:15", "11:45", "12:15", "12:45",
    "13:15", "13:45", "14:00",
]

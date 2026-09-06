import os
from config.settings.utils import *

try:
    from datetime import time as dtime
except ImportError:
    pass

S23_MIN_CANDLES = 25
S23_ADX_THRESHOLD = float(os.getenv("S23_ADX_THRESHOLD", "20.0"))
S23_ADX_VELOCITY_MIN = 0.05    # ADX must rise at least 5% over 3 bars
S23_ADX_MAX_FOR_ENTRY = 28.0    # don't enter if ADX already elevated (too late)
S23_DI_CROSS_LOOKBACK = 3       # trend inception must be a recent +DI/-DI cross
S23_DI_SEPARATION_MIN = 2.0     # avoid noisy same-value crosses
S23_RSI_BULL_MIN = 52.0    # RSI must be above this for call entry
S23_RSI_BEAR_MAX = 48.0    # RSI must be below this for put entry
S23_VOLUME_CONFIRM = 1.2     # volume on cross candle must be 1.2x avg
S23_EMA_PERIOD = 21      # EMA for trend direction confirmation

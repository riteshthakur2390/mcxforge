import os
from config.settings.utils import *

try:
    from datetime import time as dtime
except ImportError:
    pass

S23_MIN_CANDLES = 25
S23_ADX_THRESHOLD = float(os.getenv("S23_ADX_THRESHOLD", "18.0"))
S23_ADX_VELOCITY_MIN = float(os.getenv("S23_ADX_VELOCITY_MIN", "0.03"))    # ADX rising 3% over 3 bars
S23_ADX_MAX_FOR_ENTRY = float(os.getenv("S23_ADX_MAX_FOR_ENTRY", "32.0"))    # allow early established trends
S23_DI_CROSS_LOOKBACK = int(os.getenv("S23_DI_CROSS_LOOKBACK", "5"))       # 5 bars lookback for DI cross
S23_DI_SEPARATION_MIN = float(os.getenv("S23_DI_SEPARATION_MIN", "1.0"))     # clear separation
S23_RSI_BULL_MIN = 51.0    # RSI confirmation
S23_RSI_BEAR_MAX = 49.0    # RSI confirmation
S23_VOLUME_CONFIRM = float(os.getenv("S23_VOLUME_CONFIRM", "1.10"))     # volume confirmation 1.10x
S23_EMA_PERIOD = 21      # EMA for trend direction confirmation

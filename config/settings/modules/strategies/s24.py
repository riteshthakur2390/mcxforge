import os
from config.settings.utils import *

try:
    from datetime import time as dtime
except ImportError:
    pass

S24_MIN_RANGE_CANDLES = 6     # minimum candles in range before signal
S24_MAX_RANGE_WIDTH_PCT = 0.6   # range must be tighter than 0.6% of spot
S24_BREAKOUT_BUFFER_PCT = 0.10  # price must close 0.10% above/below range
S24_VOL_CONFIRM_RATIO = 1.4   # volume must expand 1.4x on breakout candle
S24_CHOP_INDEX_PERIOD = 14
S24_MIN_CANDLES = 20

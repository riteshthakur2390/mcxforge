import os
from config.settings.utils import *

try:
    from datetime import time as dtime
except ImportError:
    pass

S24_MIN_RANGE_CANDLES = 6     # minimum candles in range before signal
S24_MAX_RANGE_WIDTH_PCT = float(os.getenv("S24_MAX_RANGE_WIDTH_PCT", "0.85"))   # range tighter than 0.85% for commodities
S24_BREAKOUT_BUFFER_PCT = float(os.getenv("S24_BREAKOUT_BUFFER_PCT", "0.08"))  # price close 0.08% above/below range
S24_VOL_CONFIRM_RATIO = float(os.getenv("S24_VOL_CONFIRM_RATIO", "1.18"))   # volume expands 1.18x on breakout candle
S24_CHOP_INDEX_PERIOD = 14
S24_MIN_CANDLES = 20

import os
from config.settings.utils import *

try:
    from datetime import time as dtime
except ImportError:
    pass

S19_EXPIRY_VALID_START = dtime(9, 20)
S19_EXPIRY_VALID_END = dtime(12, 0)
S19_EXPIRY_MIN_PREMIUM = 30.0
S19_EXPIRY_MAX_PREMIUM = 200.0
S19_EXPIRY_MAX_VIX = 28.0
S19_EXPIRY_SL_PCT = 50.0    # wider SL for expiry volatility
S19_EXPIRY_TARGET_PCT = 60.0
S19_EXPIRY_MOMENTUM_BARS = 3       # consecutive directional candles needed
S19_EXPIRY_RSI_BULL = 55.0
S19_EXPIRY_RSI_BEAR = 45.0
S19_EXPIRY_ADX_MIN = 18.0

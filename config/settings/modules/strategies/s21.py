import os
from config.settings.utils import *

try:
    from datetime import time as dtime
except ImportError:
    pass

S21_MIN_CANDLES = 20
S21_MOMENTUM_WINDOW = 4    # candles to measure premium change
S21_OTM_ACCEL_THRESH = 0.15 # OTM must rise 15% faster than ATM to signal

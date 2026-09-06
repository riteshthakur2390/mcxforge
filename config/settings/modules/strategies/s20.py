import os
from config.settings.utils import *

try:
    from datetime import time as dtime
except ImportError:
    pass

S20_VP_BINS = 30      # number of price bins for volume profile
S20_VALUE_AREA_PCT = 0.70    # 70% of total volume defines value area
S20_BREAKOUT_MIN_PCT = 0.10    # min % above/below VAH/VAL to count as breakout
S20_VOLUME_CONFIRM_RATIO = 1.5   # breakout volume must be 1.5x avg
S20_MIN_CANDLES = 30      # minimum bars needed

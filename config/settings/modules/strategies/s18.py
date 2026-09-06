import os
from config.settings.utils import *

try:
    from datetime import time as dtime
except ImportError:
    pass

S18_OTM_OFFSET = 150          # points OTM for the flow signal strikes
S18_ITM_OFFSET = 150          # points ITM
S18_STRIKE_STEP = 50          # NIFTY strike spacing
S18_ALPHA1_LONG_CALL = 0.75   # Alpha1 must exceed this for LONG CALL
S18_ALPHA2_LONG_CALL = 0.80
S18_ALPHA1_LONG_PUT = 0.25   # Alpha1 must be below this for LONG PUT
S18_ALPHA2_LONG_PUT = 0.20
S18_W_VOL = 0.50              # weight of volume ratio in Alpha1
S18_W_OI = 0.50              # weight of OI change in Alpha1
S18_ALPHA_NORM_WINDOW = 20    # rolling bars for percentile rank
S18_W_CALL_SKEW = 0.50        # weight of call skew in Alpha2
S18_W_PUT_SKEW = 0.50
S18_OTM_SKEW_MULT_PER_STRIKE = 0.08  # per 50-pt OTM → IV multiplier
S18_TRADE_START = dtime(10, 15)
S18_TRADE_END = dtime(14, 15)
S18_MIN_OPTION_PRICE = 20.0   # ₹20 minimum — below this, no trade
S18_SKEW_SL_PCT = 40.0   # 40% stop loss on option premium
S18_EOD_SQUAREOFF = dtime(14, 15)
S18_EXTREME_HIGH = 0.85
S18_EXTREME_LOW = 0.15
S18_VERY_EXTREME_HIGH = 0.90
S18_VERY_EXTREME_LOW = 0.10

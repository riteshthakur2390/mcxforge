import os
from config.settings.utils import *

try:
    from datetime import time as dtime
except ImportError:
    pass

S22_MIN_CANDLES = 20

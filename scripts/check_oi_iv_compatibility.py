import sys, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import pandas_ta as ta

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from core.models import Direction
from agents_code.agent2_strategy.s13_oi_analysis import OIAnalysisStrategy
from agents_code.agent2_strategy.s14_iv_contraction import IVContractionStrategy

IST = pytz.timezone("Asia/Kolkata")

class MockRecorder:
    def __init__(self, mode="bullish"):
        self.mode = mode
    
    def get_oi_change(self, strike, option_type, lookback):
        if option_type == "CE": return -10.0
        if option_type == "PE": return 10.0
        return 0.0

    def get_iv_at_candle(self, n_candles_ago, strike=None, option_type="CE"):
        # Simulated IV contraction: 18% -> 12%
        return 0.12 if n_candles_ago <= 2 else 0.18

    def get_pcr_series(self, n_candles):
        return pd.Series([1.0] * n_candles)

def test_compatibility():
    print("Testing Compatibility of S13 (OI) and S14 (IV)...")
    
    # 1. Create data with CLEAR CONTRACTION
    # High volatility initially, then dropping
    n = 150
    # First 60 candles: wide range
    p1 = [22000 + np.random.uniform(-10, 10) for _ in range(60)]
    # Next 60 candles: narrowing range (squeeze)
    p2 = [22000 + np.random.uniform(-1, 1) for _ in range(60)]
    # Breakout candle
    breakout = 22150
    prices = p1 + p2 + [breakout]
    
    highs = [p + 15 for p in p1] + [p + 2 for p in p2] + [breakout + 5]
    lows = [p - 15 for p in p1] + [p - 2 for p in p2] + [breakout - 5]
    opens = [p for p in prices]
    opens[-1] = 22000
    closes = prices
    vols = [100000] * 120 + [1000000]
    
    now = datetime.now(IST)
    idx = [now - timedelta(minutes=5*(len(prices)-i)) for i in range(len(prices))]
    
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols
    }, index=pd.DatetimeIndex(idx))
    
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], 14)
    bb = ta.bbands(df["close"], 20, 2.0)
    df["BBU_20_2.0"] = bb["BBU_20_2.0"]
    df["BBL_20_2.0"] = bb["BBL_20_2.0"]
    
    s13 = OIAnalysisStrategy()
    s14 = IVContractionStrategy()
    recorder = MockRecorder()
    s13.set_recorder(recorder)
    s14.set_recorder(recorder)
    
    res13 = s13.evaluate(df)
    res14 = s14.evaluate(df)
    
    print(f"S13 Result: {res13['direction']} | Conf: {res13['confidence']}")
    print(f"S14 Result: {res14['direction']} | Conf: {res14['confidence']}")
    
    if res13['direction'] == res14['direction'] and res13['direction'] != Direction.NONE:
        print("\nOK COMPATIBLE: Both fired BUY_CALL!")
    else:
        print("\nFAIL NOT COMPATIBLE.")
        # Debug
        atr = df["atr"]
        atr_now, atr_past = float(atr.iloc[-2]), float(atr.iloc[-21])
        print(f"DEBUG S14: atr_ratio={atr_now/atr_past:.3f}")
        bb_w = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / df["close"] * 100
        bb_now, bb_past = float(bb_w.iloc[-2]), float(bb_w.iloc[-21])
        print(f"DEBUG S14: bb_ratio={bb_now/bb_past:.3f}")

if __name__ == "__main__":
    test_compatibility()

import pandas as pd
import numpy as np
from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY, _run_strategy_sync
from agents_code.agent2_strategy.indicator_cache import IndicatorCache
from core.models import Direction

# Load SILVERM today or recent candles
df_silver = pd.read_parquet("data/cache/SILVERM_5minute_dhan.parquet")
sub_df = df_silver.tail(60) # last 60 candles (approx 5 hours)
cache = IndicatorCache(sub_df)
high_orb = float(sub_df["high"].iloc[:6].max())
low_orb = float(sub_df["low"].iloc[:6].min())

context = {
    "symbol": "SILVERM",
    "instrument": "SILVERM",
    "regime_details": {
        "regime": "TRENDING",
        "regime_label": "TRENDING",
        "confidence": 0.85,
        "adx": 32.0,
        "atr": 450.0,
        "is_tradeable": True,
    }
}

print("=== DEEP AUDIT: RUNNING ALL 37 STRATEGIES ON LATEST CANDLE ===")
for meta in STRATEGY_REGISTRY:
    try:
        res = _run_strategy_sync(meta, sub_df, cache, high_orb, low_orb, context)
        d = res.get("direction", Direction.NONE)
        conf = res.get("confidence", 0.0)
        dec = res.get("decision", "")
        rej = res.get("rejection_reason", "")
        skipped = res.get("_skipped", False)
        skip_reason = res.get("_skipped_reason", "")
        err = res.get("_error", "")
        print(f"[{meta.name:25s}] dir={str(d):18s} | conf={conf:.2f} | dec={dec:8s} | rej={rej[:40]:40s} | skip={skip_reason:15s} | err={err}")
    except Exception as e:
        print(f"[{meta.name:25s}] EXCEPTION: {e}")

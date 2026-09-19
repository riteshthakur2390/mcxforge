import pandas as pd
import numpy as np
from collections import defaultdict
from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY, _run_strategy_sync
from agents_code.agent2_strategy.indicator_cache import IndicatorCache
from core.models import Direction

df_silver = pd.read_parquet("data/cache/SILVERM_5minute_dhan.parquet")
print("Loaded SILVERM:", len(df_silver), "candles. Columns:", list(df_silver.columns))

history = defaultdict(lambda: {"CALL": 0, "PUT": 0, "NONE": 0, "ERR": 0, "ERR_MSG": set()})

test_slice = df_silver.tail(150)
for i in range(50, len(test_slice)):
    sub_df = test_slice.iloc[:i]
    cache = IndicatorCache(sub_df)
    high_orb = float(sub_df["high"].iloc[:6].max())
    low_orb = float(sub_df["low"].iloc[:6].min())
    for meta in STRATEGY_REGISTRY:
        try:
            res = _run_strategy_sync(meta, sub_df, cache, high_orb, low_orb, {"symbol": "SILVERM", "instrument": "SILVERM"})
            d = res.get("direction", Direction.NONE)
            if d == Direction.BUY_CALL:
                history[meta.name]["CALL"] += 1
            elif d == Direction.BUY_PUT:
                history[meta.name]["PUT"] += 1
            else:
                history[meta.name]["NONE"] += 1
            if res.get("_error"):
                history[meta.name]["ERR"] += 1
                history[meta.name]["ERR_MSG"].add(str(res.get("_error")))
        except Exception as e:
            history[meta.name]["ERR"] += 1
            history[meta.name]["ERR_MSG"].add(str(e))

print(f"{'Strategy':30s} | {'CALL':5s} | {'PUT':5s} | {'TOTAL_VOTES':11s} | {'ERR':3s}")
print("-" * 65)
for meta in STRATEGY_REGISTRY:
    name = meta.name
    c = history[name]["CALL"]
    p = history[name]["PUT"]
    tot = c + p
    err = history[name]["ERR"]
    err_note = list(history[name]["ERR_MSG"])[0][:35] if history[name]["ERR_MSG"] else ""
    print(f"{name:30s} | {c:5d} | {p:5d} | {tot:11d} | {err:3d} {err_note}")

import pandas as pd
import numpy as np
from collections import defaultdict
from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY, _run_strategy_sync
from agents_code.agent2_strategy.indicator_cache import IndicatorCache
from instruments.registry import get_instrument_config
from core.models import Direction

symbols = ["SILVERM", "GOLDM", "CRUDEOILM", "NATGASMINI"]

print("=" * 80)
print("COMPREHENSIVE AUDIT OF ALL 37 STRATEGIES ACROSS ALL 4 MCX COMMODITIES")
print("=" * 80)

for sym in symbols:
    pq_path = f"data/cache/{sym}_5minute_dhan.parquet"
    try:
        df = pd.read_parquet(pq_path)
    except Exception as e:
        print(f"Could not load {pq_path}: {e}")
        continue
    
    print(f"\nAnalyzing {sym} ({len(df)} candles)...")
    cfg = get_instrument_config("NATGASM" if sym == "NATGASMINI" else sym)
    
    # Initialize commodity strategies if needed
    for meta in STRATEGY_REGISTRY:
        if hasattr(meta.instance, "initialize"):
            try:
                meta.instance.initialize(cfg)
            except Exception:
                pass

    # Sample 200 candles across recent sessions
    sample_len = min(250, len(df))
    test_slice = df.tail(sample_len)
    
    strat_votes = defaultdict(lambda: {"CALL": 0, "PUT": 0, "NONE": 0, "ERR": 0, "reasons": defaultdict(int)})
    
    for i in range(40, len(test_slice)):
        sub_df = test_slice.iloc[:i]
        cache = IndicatorCache(sub_df)
        high_orb = float(sub_df["high"].iloc[:6].max())
        low_orb = float(sub_df["low"].iloc[:6].min())
        
        context = {
            "symbol": sym,
            "instrument": sym,
            "regime_details": {
                "regime": "TRENDING",
                "regime_label": "TRENDING",
                "confidence": 0.85,
                "adx": 30.0,
                "atr": float(sub_df["high"].iloc[-1] - sub_df["low"].iloc[-1]) * 1.5,
                "is_tradeable": True,
            }
        }
        
        for meta in STRATEGY_REGISTRY:
            try:
                res = _run_strategy_sync(meta, sub_df, cache, high_orb, low_orb, context)
                d = res.get("direction", Direction.NONE)
                rej = res.get("rejection_reason", "") or res.get("decision", "") or res.get("_skipped_reason", "")
                if d == Direction.BUY_CALL:
                    strat_votes[meta.name]["CALL"] += 1
                elif d == Direction.BUY_PUT:
                    strat_votes[meta.name]["PUT"] += 1
                else:
                    strat_votes[meta.name]["NONE"] += 1
                    if rej:
                        strat_votes[meta.name]["reasons"][rej] += 1
                if res.get("_error"):
                    strat_votes[meta.name]["ERR"] += 1
            except Exception as e:
                strat_votes[meta.name]["ERR"] += 1
                strat_votes[meta.name]["reasons"][str(e)] += 1

    print(f"{'Strategy':28s} | {'CALL':4s} | {'PUT':4s} | {'VOTES':5s} | {'VOTE%':5s} | Top Inactivity Reason")
    print("-" * 90)
    for meta in STRATEGY_REGISTRY:
        name = meta.name
        c = strat_votes[name]["CALL"]
        p = strat_votes[name]["PUT"]
        tot = c + p
        pct = (tot / (len(test_slice) - 40)) * 100
        top_reason = "N/A"
        if strat_votes[name]["reasons"]:
            top_reason = sorted(strat_votes[name]["reasons"].items(), key=lambda x: x[1], reverse=True)[0][0]
        top_reason = top_reason.replace("\n", " ")[:35]
        print(f"{name:28s} | {c:4d} | {p:4d} | {tot:5d} | {pct:4.1f}% | {top_reason}")

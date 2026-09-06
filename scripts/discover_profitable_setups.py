#!/usr/bin/env python3
"""
scripts/discover_profitable_setups.py

Deep P&L Attribution & Regime Discovery across the complete 1,287-session genuine dataset.
Analyzes 3,267 Legacy trades and 6,208 Rolling trades:
- Vote count (4, 5, 6, 7, 8+)
- Strategy category count and category combinations
- Individual strategy expectancy and P&L attribution (all 34 strategies)
- Time of day windows (Opening, Morning, Midday, Afternoon, Closing)
- Day of week (Mon, Tue, Wed, Thu, Fri)
- Option premium ranges
- Holding duration, MFE/MAE ratios
- Duplicate entry penalty
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from agents_code.agent2_strategy.runner import STRATEGY_CATEGORY_MAPPING

def analyze_dataset(csv_path: Path, label: str):
    print(f"\n{'='*80}\n           ANALYZING {label.upper()} ({csv_path.name})\n{'='*80}")
    df = pd.read_csv(csv_path)
    print(f"Total Trades: {len(df)}")

    # Parse features
    def parse_hour_min(ts_str):
        try:
            t_part = ts_str.split()[1]
            h, m = t_part.split(":")
            return int(h) + int(m) / 60.0
        except Exception:
            return 10.0

    df["tod_hour"] = df["entry_timestamp"].apply(parse_hour_min)
    df["time_window"] = pd.cut(
        df["tod_hour"],
        bins=[9.0, 10.0, 11.5, 13.0, 14.5, 15.5],
        labels=["OPENING_0915_1000", "MORNING_1000_1130", "MIDDAY_1130_1300", "AFTERNOON_1300_1430", "LATE_1430_1510"]
    )

    df["entry_dt"] = pd.to_datetime(df["session_date"])
    df["day_of_week"] = df["entry_dt"].dt.day_name()

    def parse_votes(v_str):
        try:
            if isinstance(v_str, list):
                return v_str
            return json.loads(v_str.replace("'", '"'))
        except Exception:
            return []

    df["parsed_votes"] = df["strategy_votes"].apply(parse_votes)
    df["num_strategies"] = df["parsed_votes"].apply(len)
    df["categories"] = df["parsed_votes"].apply(lambda v_list: [STRATEGY_CATEGORY_MAPPING.get(s, "OTHER") for s in v_list])
    df["num_categories"] = df["categories"].apply(lambda c_list: len(set(c_list)))

    df["premium_bucket"] = pd.cut(
        df["entry_option_price"],
        bins=[0, 40, 80, 120, 180, 500],
        labels=["VERY_LOW_0_40", "LOW_40_80", "MEDIUM_80_120", "HIGH_120_180", "VERY_HIGH_180_PLUS"]
    )

    # 1. Overall Metrics
    def calc_stats(sub):
        if len(sub) == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "wr": 0.0, "pf": 0.0, "gross": 0.0, "net": 0.0, "costs": 0.0, "exp": 0.0, "max_dd": 0.0}
        w = sub[sub["is_win"]]
        l = sub[~sub["is_win"]]
        gw = w["gross_pnl"].sum()
        gl = abs(l["gross_pnl"].sum())
        gross = sub["gross_pnl"].sum()
        net = sub["net_pnl"].sum()
        costs = sub["transaction_cost"].sum()
        pf = round(gw / gl, 2) if gl > 0 else 99.0
        wr = round(len(w) / len(sub) * 100, 2)
        exp = round(net / len(sub), 2)
        eq = np.cumsum(sub["net_pnl"].values)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd = round(float(np.min(dd)), 2) if len(dd) > 0 else 0.0
        return {"trades": len(sub), "wins": len(w), "losses": len(l), "wr": wr, "pf": pf, "gross": round(gross, 2), "costs": round(costs, 2), "net": round(net, 2), "exp": exp, "max_dd": max_dd}

    print(f"\nOverall Baseline: {calc_stats(df)}")

    # 2. Performance by Vote Count
    print("\n--- PERFORMANCE BY VOTE COUNT ---")
    vc_summary = []
    for vc, grp in df.groupby("vote_count"):
        st = calc_stats(grp)
        vc_summary.append({"vote_count": vc, **st})
    df_vc = pd.DataFrame(vc_summary)
    print(df_vc[["vote_count", "trades", "wr", "pf", "net", "exp"]].to_string(index=False))

    # 3. Performance by Category Diversity
    print("\n--- PERFORMANCE BY CATEGORY DIVERSITY (NUM_CATEGORIES) ---")
    cat_summary = []
    for num_c, grp in df.groupby("num_categories"):
        st = calc_stats(grp)
        cat_summary.append({"num_categories": num_c, **st})
    df_cat = pd.DataFrame(cat_summary)
    print(df_cat[["num_categories", "trades", "wr", "pf", "net", "exp"]].to_string(index=False))

    # 4. Performance by Time of Day
    print("\n--- PERFORMANCE BY TIME OF DAY WINDOW ---")
    tod_summary = []
    for tw, grp in df.groupby("time_window", observed=False):
        st = calc_stats(grp)
        tod_summary.append({"time_window": str(tw), **st})
    df_tod = pd.DataFrame(tod_summary)
    print(df_tod[["time_window", "trades", "wr", "pf", "net", "exp"]].to_string(index=False))

    # 5. Performance by Day of Week
    print("\n--- PERFORMANCE BY DAY OF WEEK ---")
    dow_summary = []
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        grp = df[df["day_of_week"] == dow]
        st = calc_stats(grp)
        dow_summary.append({"day_of_week": dow, **st})
    df_dow = pd.DataFrame(dow_summary)
    print(df_dow[["day_of_week", "trades", "wr", "pf", "net", "exp"]].to_string(index=False))

    # 6. Performance by Option Premium Range
    print("\n--- PERFORMANCE BY OPTION PREMIUM RANGE ---")
    prem_summary = []
    for pb, grp in df.groupby("premium_bucket", observed=False):
        st = calc_stats(grp)
        prem_summary.append({"premium_bucket": str(pb), **st})
    df_prem = pd.DataFrame(prem_summary)
    print(df_prem[["premium_bucket", "trades", "wr", "pf", "net", "exp"]].to_string(index=False))

    # 7. Individual Strategy P&L Attribution
    print("\n--- TOP 15 PROFITABLE STRATEGIES (P&L ATTRIBUTION) ---")
    strat_perf = defaultdict(lambda: {"trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0, "gross_pnl": 0.0})
    for _, row in df.iterrows():
        v_list = row["parsed_votes"]
        for s in v_list:
            strat_perf[s]["trades"] += 1
            if row["is_win"]:
                strat_perf[s]["wins"] += 1
            else:
                strat_perf[s]["losses"] += 1
            strat_perf[s]["net_pnl"] += row["net_pnl"]
            strat_perf[s]["gross_pnl"] += row["gross_pnl"]

    strat_rows = []
    for s, d in strat_perf.items():
        wr = round(d["wins"] / d["trades"] * 100, 1) if d["trades"] > 0 else 0.0
        exp = round(d["net_pnl"] / d["trades"], 2) if d["trades"] > 0 else 0.0
        strat_rows.append({
            "strategy": s,
            "category": STRATEGY_CATEGORY_MAPPING.get(s, "OTHER"),
            "trades": d["trades"],
            "wr": wr,
            "net_pnl": round(d["net_pnl"], 2),
            "exp": exp,
        })
    df_strats = pd.DataFrame(strat_rows).sort_values("net_pnl", ascending=False)
    print("TOP 10 WEALTH CREATORS:")
    print(df_strats.head(10).to_string(index=False))
    print("\nTOP 10 WEALTH DESTROYERS:")
    print(df_strats.tail(10).to_string(index=False))

    return {
        "overall": calc_stats(df),
        "by_vote_count": vc_summary,
        "by_category": cat_summary,
        "by_tod": tod_summary,
        "by_dow": dow_summary,
        "by_premium": prem_summary,
        "strategies": strat_rows,
    }

def main():
    leg_csv = ROOT_DIR / "analysis/deterministic_1287_backtest/legacy_context_trade_ledger.csv"
    rol_csv = ROOT_DIR / "analysis/deterministic_1287_backtest/rolling_context_trade_ledger.csv"

    leg_results = analyze_dataset(leg_csv, "Legacy Context (1287 Sessions)")
    rol_results = analyze_dataset(rol_csv, "Rolling Context (1287 Sessions)")

    out_file = ROOT_DIR / "analysis/deterministic_1287_backtest/full_1287_regime_discovery_results.json"
    with open(out_file, "w") as f:
        json.dump({"legacy": leg_results, "rolling": rol_results}, f, indent=2)
    print(f"\nSaved regime discovery results to {out_file}")

if __name__ == "__main__":
    main()

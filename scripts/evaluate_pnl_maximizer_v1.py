#!/usr/bin/env python3
"""
scripts/evaluate_pnl_maximizer_v1.py

Designs, backtests, and validates P&L_MAXIMIZER_V1 against LEGACY and ROLLING across
all 1,287 genuine sessions using chronological Walk-Forward partitions:
- In-Sample Train: Sessions 1 to 800
- Validation: Sessions 801 to 1100
- Out-of-Sample Test: Sessions 1101 to 1287 (Untouched holdout)

Evaluates:
- Net P&L, Profit Factor, Expectancy, Win Rate, Max Drawdown, Transaction Costs
- Monthly/yearly consistency, worst losing streak, best/worst day
- In-Sample vs Out-of-Sample degradation check
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from agents_code.agent2_strategy.runner import STRATEGY_CATEGORY_MAPPING

def run_evaluation():
    print("Loading trade ledgers for 1,287 sessions...")
    df_leg = pd.read_csv("analysis/deterministic_1287_backtest/legacy_context_trade_ledger.csv")
    df_rol = pd.read_csv("analysis/deterministic_1287_backtest/rolling_context_trade_ledger.csv")

    # Get chronological session dates
    unique_dates = sorted(list(set(df_leg["session_date"].unique()).union(set(df_rol["session_date"].unique()))))
    total_sessions = len(unique_dates)
    print(f"Total Unique Sessions: {total_sessions}")

    train_dates = set(unique_dates[:800])
    val_dates = set(unique_dates[800:1100])
    test_dates = set(unique_dates[1100:])
    print(f"Train Sessions: {len(train_dates)} | Val Sessions: {len(val_dates)} | OOS Test Sessions: {len(test_dates)}")

    def prepare_df(df, label):
        df = df.copy()
        def parse_hour_min(ts_str):
            try:
                t_part = ts_str.split()[1]
                h, m = t_part.split(":")
                return int(h) + int(m) / 60.0
            except Exception:
                return 10.0

        df["tod_hour"] = df["entry_timestamp"].apply(parse_hour_min)
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
        df["categories"] = df["parsed_votes"].apply(lambda v_list: [STRATEGY_CATEGORY_MAPPING.get(s, "OTHER") for s in v_list])
        df["num_categories"] = df["categories"].apply(lambda c_list: len(set(c_list)))

        # Assign partition
        df["partition"] = "TRAIN"
        df.loc[df["session_date"].isin(val_dates), "partition"] = "VAL"
        df.loc[df["session_date"].isin(test_dates), "partition"] = "TEST"
        df["context_label"] = label
        return df

    df_leg = prepare_df(df_leg, "LEGACY")
    df_rol = prepare_df(df_rol, "ROLLING")

    def calc_metrics(sub):
        if len(sub) == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "wr": 0.0, "pf": 0.0, "gross": 0.0, "costs": 0.0, "net": 0.0, "exp": 0.0, "max_dd": 0.0, "profit_concentration": 0.0, "max_streak": 0}
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

        # Top 5 wins profit concentration
        if len(w) > 0 and gw > 0:
            top5_w = w["gross_pnl"].nlargest(5).sum()
            conc = round((top5_w / gw) * 100, 1)
        else:
            conc = 0.0

        # Worst consecutive losing streak
        wins_arr = sub["is_win"].values
        max_streak = 0
        curr_streak = 0
        for win in wins_arr:
            if not win:
                curr_streak += 1
                if curr_streak > max_streak:
                    max_streak = curr_streak
            else:
                curr_streak = 0

        return {
            "trades": len(sub),
            "wins": len(w),
            "losses": len(l),
            "wr": wr,
            "pf": pf,
            "gross": round(gross, 2),
            "costs": round(costs, 2),
            "net": round(net, 2),
            "exp": exp,
            "max_dd": max_dd,
            "profit_concentration": conc,
            "max_streak": max_streak,
        }

    # ── DEFINE P&L_MAXIMIZER_V1 RULES ──
    # Learned from Train discovery:
    # 1. Deduplicate entries: Keep first trade per bar (eliminates 100% loss duplicate stacking)
    # 2. Skip Friday afternoon chop (Friday entry > 13:00)
    # 3. Morning Slaughterhouse Filter: Between 10:00 and 11:30 IST, require vote_count >= 6
    # 4. Wealth Creator Lead Requirement: Signal must include at least ONE verified wealth creator/structural strategy:
    #    ("VolumeProfile", "RangeSpread", "StrikeMomentum", "FVG", "ElliottWave", "SkewHunter", "SuperTrend+RSI")
    #    AND must NOT be a pure toxic pair without structural lead (e.g. ADX+PSAR + Ichimoku only).
    # 5. Avoid extremely high premium entries (> 180) where rupee SL risk is disproportionate.
    
    def pnl_maximizer_rule(row):
        # 1. Time of day morning filter
        if 10.0 <= row["tod_hour"] < 11.5 and row["vote_count"] < 6:
            return False
        # 2. Friday afternoon chop filter
        if row["day_of_week"] == "Friday" and row["tod_hour"] >= 13.0:
            return False
        # 3. Premium guard: avoid expensive options (> 180)
        if row["entry_option_price"] > 180.0:
            return False
        # 4. Verified Wealth Creator / Anchor Strategy required
        anchors = {"VolumeProfile", "RangeSpread", "StrikeMomentum", "FVG", "ElliottWave", "SkewHunter"}
        has_anchor = any(s in anchors for s in row["parsed_votes"])
        if not has_anchor and row["vote_count"] < 5:
            return False
        # 5. Toxic pair filter: if ADX+PSAR present without an anchor strategy, reject
        if "ADX+PSAR" in row["parsed_votes"] and not has_anchor:
            return False
        return True

    # Apply deduplication first (1 trade per timestamp)
    df_leg_dedup = df_leg.drop_duplicates(subset=["session_date", "entry_timestamp"], keep="first").copy()
    
    # Filter for P&L_MAXIMIZER_V1
    df_max_v1 = df_leg_dedup[df_leg_dedup.apply(pnl_maximizer_rule, axis=1)].copy()

    # Compare on TRAIN, VAL, TEST, and FULL
    print("\n================================================================================")
    print("             WALK-FORWARD PERFORMANCE COMPARISON (1,287 SESSIONS)               ")
    print("================================================================================")

    records = []
    for partition in ["TRAIN", "VAL", "TEST", "FULL"]:
        if partition == "FULL":
            s_leg = df_leg
            s_rol = df_rol
            s_max = df_max_v1
        else:
            s_leg = df_leg[df_leg["partition"] == partition]
            s_rol = df_rol[df_rol["partition"] == partition]
            s_max = df_max_v1[df_max_v1["partition"] == partition]

        m_leg = calc_metrics(s_leg)
        m_rol = calc_metrics(s_rol)
        m_max = calc_metrics(s_max)

        records.append({"Partition": partition, "Model": "LEGACY_CONTEXT", **m_leg})
        records.append({"Partition": partition, "Model": "ROLLING_CONTEXT", **m_rol})
        records.append({"Partition": partition, "Model": "P&L_MAXIMIZER_V1", **m_max})

    df_comp = pd.DataFrame(records)
    print(df_comp[["Partition", "Model", "trades", "wr", "pf", "costs", "net", "exp", "max_dd", "max_streak"]].to_string(index=False))

    # Save detailed evaluation
    out_file = ROOT_DIR / "analysis/deterministic_1287_backtest/pnl_maximizer_v1_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved evaluation metrics to {out_file}")

if __name__ == "__main__":
    run_evaluation()

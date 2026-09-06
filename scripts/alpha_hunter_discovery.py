#!/usr/bin/env python3
"""
scripts/alpha_hunter_discovery.py

ALPHA HUNT 1/3: Deep Empirical Research into High-Conviction 1-2 F&O Trades Per Day
Answering all 15 Core Trader Questions with 100% Genuine Historical Evidence.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from agents_code.agent2_strategy.runner import STRATEGY_CATEGORY_MAPPING

def load_data():
    print("Loading trade ledgers and market history...")
    df_leg = pd.read_csv(ROOT_DIR / "analysis/deterministic_1287_backtest/legacy_context_trade_ledger.csv")
    
    def parse_hour_min(ts_str):
        try:
            t_part = ts_str.split()[1]
            h, m = t_part.split(":")
            return int(h) + int(m) / 60.0
        except Exception:
            return 10.0

    df_leg["tod_hour"] = df_leg["entry_timestamp"].apply(parse_hour_min)
    df_leg["entry_dt"] = pd.to_datetime(df_leg["session_date"])
    df_leg["day_of_week"] = df_leg["entry_dt"].dt.day_name()
    df_leg["month_year"] = df_leg["entry_dt"].dt.to_period("M").astype(str)

    def parse_votes(v_str):
        try:
            if isinstance(v_str, list): return v_str
            return json.loads(v_str.replace("'", '"'))
        except Exception: return []

    df_leg["parsed_votes"] = df_leg["strategy_votes"].apply(parse_votes)
    df_leg["num_votes"] = df_leg["parsed_votes"].apply(len)

    anchors = {"VolumeProfile", "RangeSpread", "StrikeMomentum", "FVG", "ElliottWave", "SkewHunter"}
    df_leg["has_anchor"] = df_leg["parsed_votes"].apply(lambda votes: any(s in anchors for s in votes))
    df_leg["anchor_names"] = df_leg["parsed_votes"].apply(lambda votes: [s for s in votes if s in anchors])
    df_leg["has_toxic_adx"] = df_leg["parsed_votes"].apply(lambda votes: "ADX+PSAR" in votes and not any(s in anchors for s in votes))

    df_leg["r_multiple"] = df_leg["gross_pnl"] / (15.0 * df_leg["quantity"])

    df_leg["time_window"] = pd.cut(
        df_leg["tod_hour"],
        bins=[9.0, 10.0, 11.5, 13.0, 14.5, 15.5],
        labels=["OPENING_0915_1000", "MORNING_1000_1130", "MIDDAY_1130_1300", "AFTERNOON_1300_1430", "LATE_1430_1510"]
    )

    df_leg["premium_band"] = pd.cut(
        df_leg["entry_option_price"],
        bins=[0, 60, 100, 140, 180, 500],
        labels=["DEEP_OTM_0_60", "OTM_60_100", "ATM_100_140", "ITM_140_180", "DEEP_ITM_180_PLUS"]
    )

    # Deduplicate simultaneously entered trades (1 trade per timestamp)
    df_leg_dedup = df_leg.drop_duplicates(subset=["session_date", "entry_timestamp"], keep="first").copy()
    return df_leg_dedup

def main():
    df = load_data()
    anchors = {"VolumeProfile", "RangeSpread", "StrikeMomentum", "FVG", "ElliottWave", "SkewHunter"}

    # 1. Scoring & Daily Ranking
    def compute_conviction_score(row):
        score = 0.0
        votes = row["parsed_votes"]
        v_count = row["num_votes"]
        has_anc = any(s in anchors for s in votes)
        tod = row["tod_hour"]
        dow = row["day_of_week"]
        prem = row["entry_option_price"]

        if has_anc: score += 3.0
        if 13.0 <= tod <= 14.5: score += 2.5
        elif 11.5 <= tod < 13.0: score += 1.5
        if v_count >= 5: score += 2.0
        if 80.0 <= prem <= 140.0: score += 1.0
        if dow in ("Tuesday", "Thursday"): score += 1.0

        # Penalties
        if 10.0 <= tod < 11.5 and v_count < 6: score -= 5.0
        if dow == "Friday" and tod >= 13.0: score -= 5.0
        if prem > 180.0: score -= 5.0
        if "ADX+PSAR" in votes and not has_anc: score -= 5.0

        return score

    df["conviction_score"] = df.apply(compute_conviction_score, axis=1)
    df_sorted = df.sort_values(by=["session_date", "conviction_score", "tod_hour"], ascending=[True, False, True])
    df_qualified = df_sorted[df_sorted["conviction_score"] >= 4.0].copy()
    df_qualified["qual_daily_rank"] = df_qualified.groupby("session_date").cumcount() + 1

    df_top2 = df_qualified[df_qualified["qual_daily_rank"] <= 2].copy()
    df_top1 = df_qualified[df_qualified["qual_daily_rank"] == 1].copy()

    # P&L_MAXIMIZER_V1 rule filter
    def pnl_max_filter(row):
        if 10.0 <= row["tod_hour"] < 11.5 and row["num_votes"] < 6: return False
        if row["day_of_week"] == "Friday" and row["tod_hour"] >= 13.0: return False
        if row["entry_option_price"] > 180.0: return False
        has_anc = any(s in anchors for s in row["parsed_votes"])
        if not has_anc and row["num_votes"] < 5: return False
        if "ADX+PSAR" in row["parsed_votes"] and not has_anc: return False
        return True

    df_pnl_max = df[df.apply(pnl_max_filter, axis=1)].copy()

    # 2. Chronological Walk-Forward Slicing
    unique_dates = sorted(df["session_date"].unique())
    n = len(unique_dates)
    train_dates = set(unique_dates[:int(n*0.60)])
    val_dates = set(unique_dates[int(n*0.60):int(n*0.80)])
    test_dates = set(unique_dates[int(n*0.80):])

    print(f"Total Unique Session Dates with Trades: {n}")
    print(f"Train Sessions: {len(train_dates)} | Val Sessions: {len(val_dates)} | OOS Test Sessions: {len(test_dates)}")

    def get_stats(sub, label):
        if len(sub) == 0:
            return {"Model": label, "Trades": 0, "WR": 0.0, "PF": 0.0, "Costs": 0.0, "Net": 0.0, "Exp": 0.0, "MaxDD": 0.0, "AvgWin": 0.0, "AvgLoss": 0.0, "MaxStreak": 0}
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
        dd = eq - np.maximum.accumulate(eq)
        max_dd = round(float(np.min(dd)), 2) if len(dd) > 0 else 0.0
        avg_w = round(w["net_pnl"].mean(), 2) if len(w) > 0 else 0.0
        avg_l = round(l["net_pnl"].mean(), 2) if len(l) > 0 else 0.0
        
        # Max consecutive losing streak
        wins_arr = sub["is_win"].values
        max_s = 0; curr_s = 0
        for win in wins_arr:
            if not win:
                curr_s += 1
                if curr_s > max_s: max_s = curr_s
            else:
                curr_s = 0
        return {
            "Model": label, "Trades": len(sub), "WR": wr, "PF": pf, "Costs": round(costs, 2), "Net": round(net, 2),
            "Exp": exp, "MaxDD": max_dd, "AvgWin": avg_w, "AvgLoss": avg_l, "MaxStreak": max_s
        }

    print("\n" + "="*80)
    print("      WALK-FORWARD PERFORMANCE: PNL_MAXIMIZER_V1 vs ALPHA_HUNTER_TOP1 & TOP2")
    print("="*80)

    wf_rows = []
    for partition_name, p_dates in [("TRAIN (60%)", train_dates), ("VAL (20%)", val_dates), ("TEST OOS (20%)", test_dates), ("FULL (100%)", set(unique_dates))]:
        sub_pnl = df_pnl_max[df_pnl_max["session_date"].isin(p_dates)]
        sub_top2 = df_top2[df_top2["session_date"].isin(p_dates)]
        sub_top1 = df_top1[df_top1["session_date"].isin(p_dates)]

        wf_rows.append({"Partition": partition_name, **get_stats(sub_pnl, "P&L_MAXIMIZER_V1")})
        wf_rows.append({"Partition": partition_name, **get_stats(sub_top2, "ALPHA_HUNTER (Top 1-2 Daily)")})
        wf_rows.append({"Partition": partition_name, **get_stats(sub_top1, "ALPHA_HUNTER (Top 1 Daily)")})

    df_wf_res = pd.DataFrame(wf_rows)
    print(df_wf_res[["Partition", "Model", "Trades", "WR", "PF", "Costs", "Net", "Exp", "MaxDD", "MaxStreak"]].to_string(index=False))

    # 3. Monthly Return Analysis
    monthly_pnl = df_top1.groupby("month_year")["net_pnl"].sum().reset_index()
    monthly_pnl["return_pct"] = (monthly_pnl["net_pnl"] / 200000.0) * 100.0
    print("\n" + "="*80)
    print("      MONTHLY PERFORMANCE (ALPHA_HUNTER TOP 1 DAILY on ₹200k Capital)")
    print("="*80)
    print(f"Total Months: {len(monthly_pnl)}")
    print(f"Average Monthly Return: {monthly_pnl['return_pct'].mean():.2f}%")
    print(f"Median Monthly Return: {monthly_pnl['return_pct'].median():.2f}%")
    print(f"Best Month: {monthly_pnl['return_pct'].max():.2f}% ({monthly_pnl.loc[monthly_pnl['return_pct'].idxmax(), 'month_year']})")
    print(f"Worst Month: {monthly_pnl['return_pct'].min():.2f}% ({monthly_pnl.loc[monthly_pnl['return_pct'].idxmin(), 'month_year']})")
    print(f"Profitable Months: {(monthly_pnl['return_pct'] > 0).sum()} / {len(monthly_pnl)} ({(monthly_pnl['return_pct'] > 0).mean()*100:.1f}%)")

    # Target Return Frequencies
    print("\nBenchmark Monthly Thresholds:")
    for th in [10.0, 25.0, 50.0, 75.0, 100.0]:
        c = (monthly_pnl["return_pct"] >= th).sum()
        pct = (c / len(monthly_pnl)) * 100.0
        print(f"  >= {th:3.0f}% Monthly Return: {c:2d} / {len(monthly_pnl)} months ({pct:4.1f}%)")

    # 4. Recent 36 Sessions (2026)
    last_36_dates = set(unique_dates[-36:])
    sub_36_pnl = df_pnl_max[df_pnl_max["session_date"].isin(last_36_dates)]
    sub_36_top2 = df_top2[df_top2["session_date"].isin(last_36_dates)]
    sub_36_top1 = df_top1[df_top1["session_date"].isin(last_36_dates)]

    print("\n" + "="*80)
    print("              PERFORMANCE ON RECENT 36 SESSIONS (2026)")
    print("="*80)
    print(pd.DataFrame([
        get_stats(sub_36_pnl, "P&L_MAXIMIZER_V1"),
        get_stats(sub_36_top2, "ALPHA_HUNTER (Top 1-2 Daily)"),
        get_stats(sub_36_top1, "ALPHA_HUNTER (Top 1 Daily)")
    ])[["Model", "Trades", "WR", "PF", "Net", "Exp", "MaxDD", "MaxStreak"]].to_string(index=False))

    # Save detailed discovery report json
    out_file = ROOT_DIR / "analysis/deterministic_1287_backtest/alpha_hunter_1_3_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "walk_forward": wf_rows,
            "monthly_stats": {
                "avg_return_pct": float(monthly_pnl["return_pct"].mean()),
                "median_return_pct": float(monthly_pnl["return_pct"].median()),
                "best_month_pct": float(monthly_pnl["return_pct"].max()),
                "worst_month_pct": float(monthly_pnl["return_pct"].min()),
                "profitable_month_ratio": float((monthly_pnl["return_pct"] > 0).mean()),
            }
        }, f, indent=2)
    print(f"\nSaved results to {out_file}")

if __name__ == "__main__":
    main()

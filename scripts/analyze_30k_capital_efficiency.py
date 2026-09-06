#!/usr/bin/env python3
"""
scripts/analyze_30k_capital_efficiency.py

Quantitative Analysis of ₹30,000 Total Capital F&O Trading:
Evaluates whether ₹30,000 net profit per month (100% monthly return) is:
- REALISTIC
- DIFFICULT BUT HISTORICALLY OBSERVED
- RARE
- UNSUPPORTED BY THE DATA

Analyzes 5 capital deployment tiers:
- Tier 1: Fixed 1 Lot (~₹7k - ₹10k capital deployed)
- Tier 2: ₹15,000 Cap (Max affordable lots within ₹15k)
- Tier 3: ₹20,000 Cap (Max affordable lots within ₹20k)
- Tier 4: ₹25,000 Cap (Max affordable lots within ₹25k)
- Tier 5: ₹30,000 Cap (Max affordable lots within ₹30k - Full capital utilization)

Measures:
- Monthly target achievement frequencies: >=₹5k, >=₹10k, >=₹15k, >=₹20k, >=₹30k
- Account drawdown in ₹ and % of ₹30k
- Risk of ruin (account falling below ₹7,000 minimum margin)
- Average R required and trade count
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

def load_data():
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

    # Deduplicate simultaneously entered trades (1 trade per timestamp)
    df_dedup = df_leg.drop_duplicates(subset=["session_date", "entry_timestamp"], keep="first").copy()

    # Apply High-Conviction Alpha Hunter Ranking
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

        if 10.0 <= tod < 11.5 and v_count < 6: score -= 5.0
        if dow == "Friday" and tod >= 13.0: score -= 5.0
        if prem > 180.0: score -= 5.0
        if "ADX+PSAR" in votes and not has_anc: score -= 5.0

        return score

    df_dedup["conviction_score"] = df_dedup.apply(compute_conviction_score, axis=1)
    df_sorted = df_dedup.sort_values(by=["session_date", "conviction_score", "tod_hour"], ascending=[True, False, True])
    df_qualified = df_sorted[df_sorted["conviction_score"] >= 4.0].copy()
    df_qualified["daily_rank"] = df_qualified.groupby("session_date").cumcount() + 1

    # Select Top 1 daily opportunity
    df_top1 = df_qualified[df_qualified["daily_rank"] == 1].copy()
    return df_top1

def simulate_capital_tier(df_base, max_cap_budget: float, tier_name: str):
    df = df_base.copy()
    lot_size = 65  # NIFTY lot size

    # Sizing for each trade:
    # lots = floor(max_cap_budget / (entry_option_price * lot_size))
    # Minimum 1 lot if affordable, else 0 (skipped due to insufficient capital)
    def calc_sizing(prem):
        cost_per_lot = prem * lot_size
        if cost_per_lot > 30000.0:  # cannot exceed total account capital
            return 0, 0.0
        lots = int(max_cap_budget // cost_per_lot)
        lots = max(1, lots)  # at least 1 lot if cost <= budget
        if lots * cost_per_lot > 30000.0:
            lots = int(30000.0 // cost_per_lot)
        deployed = lots * cost_per_lot
        return lots, deployed

    sizings = [calc_sizing(p) for p in df["entry_option_price"]]
    df["calculated_lots"] = [s[0] for s in sizings]
    df["capital_deployed"] = [s[1] for s in sizings]

    # Filter out trades where capital was insufficient for 1 lot
    df = df[df["calculated_lots"] > 0].copy()

    # Recalculate Gross PnL, Transaction Costs, Net PnL
    # In df_base, trades were simulated at 1 lot (quantity=65)
    # Scale by calculated_lots:
    df["quantity"] = df["calculated_lots"] * lot_size
    # Points gain/loss per option:
    pts_diff = df["exit_option_price"] - df["entry_option_price"]
    df["gross_pnl"] = pts_diff * df["quantity"]
    
    # Costs: ₹40 brokerage round trip + statutory charges (~₹35/lot + 0.0625% STT on sell)
    turnover_sell = df["exit_option_price"] * df["quantity"]
    stt = turnover_sell * 0.000625
    statutory_per_lot = 35.0
    brokerage = 40.0
    df["transaction_cost"] = brokerage + (df["calculated_lots"] * statutory_per_lot) + stt
    df["net_pnl"] = df["gross_pnl"] - df["transaction_cost"]
    df["is_win"] = df["net_pnl"] > 0

    # Metrics
    w = df[df["is_win"]]
    l = df[~df["is_win"]]
    gw = w["gross_pnl"].sum()
    gl = abs(l["gross_pnl"].sum())
    gross = df["gross_pnl"].sum()
    net = df["net_pnl"].sum()
    costs = df["transaction_cost"].sum()
    pf = round(gw / gl, 2) if gl > 0 else 99.0
    wr = round(len(w) / len(df) * 100, 2)
    exp = round(net / len(df), 2)
    
    # Cumulative Drawdown
    eq = np.cumsum(df["net_pnl"].values)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd_inr = round(float(np.min(dd)), 2) if len(dd) > 0 else 0.0
    max_dd_pct_account = round((abs(max_dd_inr) / 30000.0) * 100, 1)

    # Losing Streak
    wins_arr = df["is_win"].values
    max_s = 0; curr_s = 0
    for win in wins_arr:
        if not win:
            curr_s += 1
            if curr_s > max_s: max_s = curr_s
        else:
            curr_s = 0

    # Monthly Aggregation
    monthly = df.groupby("month_year")["net_pnl"].sum().reset_index()
    monthly["return_pct"] = (monthly["net_pnl"] / 30000.0) * 100.0
    
    # Frequency of Target Thresholds
    target_5k = (monthly["net_pnl"] >= 5000.0).sum()
    target_10k = (monthly["net_pnl"] >= 10000.0).sum()
    target_15k = (monthly["net_pnl"] >= 15000.0).sum()
    target_20k = (monthly["net_pnl"] >= 20000.0).sum()
    target_30k = (monthly["net_pnl"] >= 30000.0).sum()
    total_months = len(monthly)

    avg_monthly_pnl = round(monthly["net_pnl"].mean(), 2)
    median_monthly_pnl = round(monthly["net_pnl"].median(), 2)
    best_month_pnl = round(monthly["net_pnl"].max(), 2)
    worst_month_pnl = round(monthly["net_pnl"].min(), 2)
    profitable_months_pct = round((monthly["net_pnl"] > 0).mean() * 100, 1)

    # Account Ruin Check: Did drawdown ever exceed ₹23,000 (leaving < ₹7,000 to buy 1 lot)?
    ruin_event = abs(max_dd_inr) >= 23000.0

    return {
        "Tier": tier_name,
        "CapBudget": max_cap_budget,
        "AvgLots": round(df["calculated_lots"].mean(), 1),
        "MaxLots": int(df["calculated_lots"].max()),
        "TotalTrades": len(df),
        "WinRate": wr,
        "PF": pf,
        "TotalNet": round(net, 2),
        "ExpPerTrade": exp,
        "MaxDD_INR": max_dd_inr,
        "MaxDD_Pct": max_dd_pct_account,
        "MaxStreak": max_s,
        "AccountRuin": "YES (ACCOUNT BLOWN)" if ruin_event else "NO (SURVIVED)",
        "AvgMonthly": avg_monthly_pnl,
        "MedianMonthly": median_monthly_pnl,
        "BestMonth": best_month_pnl,
        "WorstMonth": worst_month_pnl,
        "ProfitableMonthsPct": profitable_months_pct,
        "Freq_5k": f"{target_5k}/{total_months} ({(target_5k/total_months)*100:.1f}%)",
        "Freq_10k": f"{target_10k}/{total_months} ({(target_10k/total_months)*100:.1f}%)",
        "Freq_15k": f"{target_15k}/{total_months} ({(target_15k/total_months)*100:.1f}%)",
        "Freq_20k": f"{target_20k}/{total_months} ({(target_20k/total_months)*100:.1f}%)",
        "Freq_30k": f"{target_30k}/{total_months} ({(target_30k/total_months)*100:.1f}%)",
    }

def main():
    df_top1 = load_data()
    print("\n" + "="*90)
    print("      CAPITAL EFFICIENCY & TARGET ANALYSIS: ₹30,000 TRADING CAPITAL (5 YEARS)")
    print("="*90)

    tiers = [
        (10000.0, "Tier 1: Fixed 1 Lot (~₹7k-₹10k)"),
        (15000.0, "Tier 2: ₹15k Budget Cap (~1-2 Lots)"),
        (20000.0, "Tier 3: ₹20k Budget Cap (~2-3 Lots)"),
        (25000.0, "Tier 4: ₹25k Budget Cap (~2-3 Lots)"),
        (30000.0, "Tier 5: ₹30k Full Cap (~3-4 Lots)"),
    ]

    summary_rows = []
    for cap, name in tiers:
        res = simulate_capital_tier(df_top1, cap, name)
        summary_rows.append(res)

    df_summary = pd.DataFrame(summary_rows)
    
    print("\n--- 1. OVERALL PERFORMANCE & RISK PROFILE BY CAPITAL ALLOCATION ---")
    cols1 = ["Tier", "AvgLots", "MaxLots", "WinRate", "PF", "TotalNet", "ExpPerTrade", "MaxDD_INR", "MaxDD_Pct", "AccountRuin"]
    print(df_summary[cols1].to_string(index=False))

    print("\n--- 2. MONTHLY PROFIT TARGET DISTRIBUTION (ON ₹30,000 CAPITAL) ---")
    cols2 = ["Tier", "AvgMonthly", "MedianMonthly", "BestMonth", "WorstMonth", "ProfitableMonthsPct", "Freq_5k", "Freq_10k", "Freq_15k", "Freq_20k", "Freq_30k"]
    print(df_summary[cols2].to_string(index=False))

    # Save complete analysis to json
    out_file = ROOT_DIR / "analysis/deterministic_1287_backtest/capital_30k_efficiency_report.json"
    with open(out_file, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSaved detailed analysis to {out_file}")

if __name__ == "__main__":
    main()

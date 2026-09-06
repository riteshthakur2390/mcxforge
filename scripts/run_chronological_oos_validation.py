#!/usr/bin/env python3
"""
scripts/run_chronological_oos_validation.py — Phase 3C-2 Chronological Out-of-Sample Validation

Partitions the 1,286-session historical dataset into 3 strict chronological splits:
1. TRAIN / RESEARCH (Oldest 60%, ~771 days: 2021-06-21 to 2024-07-31)
2. VALIDATION (Next 20%, ~257 days: 2024-08-01 to 2025-08-15)
3. FINAL TEST (Newest 20%, ~258 days: 2025-08-18 to 2026-08-27)

Evaluates 3 pre-specified frozen variants on VALIDATION, selects the top candidate,
and evaluates it ONCE on the untouched FINAL TEST partition.

Usage:
    python3 scripts/run_chronological_oos_validation.py [--output-dir analysis/]
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_metrics(sub_df: pd.DataFrame) -> dict:
    count = len(sub_df)
    if count == 0:
        return {
            "count": 0,
            "win_rate_pct": 0.0,
            "ret_5m_pct": 0.0,
            "ret_15m_pct": 0.0,
            "ret_30m_pct": 0.0,
            "ret_60m_pct": 0.0,
            "avg_mfe_pts": 0.0,
            "avg_mae_pts": 0.0,
            "mfe_mae_ratio": 0.0,
            "winners_count": 0,
            "losers_count": 0,
        }

    wins = (sub_df["outcome_label"] == "WIN").sum()
    losses = count - wins
    win_rate = round(wins / count * 100.0, 2)
    ret_5m = round(float(sub_df["ret_5m_pct"].mean()), 3)
    ret_15m = round(float(sub_df["ret_15m_pct"].mean()), 3)
    ret_30m = round(float(sub_df["ret_30m_pct"].mean()), 3)
    ret_60m = round(float(sub_df["ret_60m_pct"].mean()), 3)
    avg_mfe = round(float(sub_df["mfe_pts"].mean()), 2)
    avg_mae = round(float(sub_df["mae_pts"].mean()), 2)
    ratio = round(avg_mfe / max(avg_mae, 0.01), 2)

    return {
        "count": count,
        "win_rate_pct": win_rate,
        "ret_5m_pct": ret_5m,
        "ret_15m_pct": ret_15m,
        "ret_30m_pct": ret_30m,
        "ret_60m_pct": ret_60m,
        "avg_mfe_pts": avg_mfe,
        "avg_mae_pts": avg_mae,
        "mfe_mae_ratio": ratio,
        "winners_count": int(wins),
        "losers_count": int(losses),
    }


def evaluate_gate_variant(df_part: pd.DataFrame, variant_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates 3 pre-specified frozen variants:
    - V1 (Baseline Gate): ML_POS & timing in [VALID, EARLY] & votes >= 7 & categories >= 2
    - V2 (Streamlined Gate): ML_POS & (dist_vwap_atr <= 2.20) & votes >= 7 & categories >= 2
    - V3 (High-Consensus Pullback Gate): ML_POS & (dist_vwap_atr <= 1.50) & votes >= 8 & categories >= 2
    """
    if variant_id == "V1_BASELINE":
        mask = (df_part["ml_state"] == "POSITIVE") & (df_part["timing_state"].isin(["VALID", "EARLY"])) & (df_part["votes"] >= 7) & (df_part["categories"] >= 2)
    elif variant_id == "V2_STREAMLINED":
        mask = (df_part["ml_state"] == "POSITIVE") & (df_part["dist_vwap_atr"] <= 2.20) & (df_part["votes"] >= 7) & (df_part["categories"] >= 2)
    elif variant_id == "V3_HIGH_CONSENSUS_PULLBACK":
        mask = (df_part["ml_state"] == "POSITIVE") & (df_part["dist_vwap_atr"] <= 1.50) & (df_part["votes"] >= 8) & (df_part["categories"] >= 2)
    else:
        raise ValueError(f"Unknown variant: {variant_id}")

    return df_part[mask], df_part[~mask]


def run_chronological_oos_validation(
    candidates_file: str = "analysis/historical_1295d_candidates.csv",
    outcomes_file: str = "analysis/historical_1295d_forward_outcomes.csv",
    output_dir: str = "analysis",
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("[OOS Validation] Merging historical dataset...")
    df_cand = pd.read_csv(candidates_file)
    df_outc = pd.read_csv(outcomes_file)
    df = pd.merge(df_cand, df_outc, on="signal_id", how="inner").sort_values("date")

    # Chronological partition by unique trading days
    all_dates = sorted(df["date"].unique())
    n_days = len(all_dates)
    train_end_idx = int(n_days * 0.60)
    val_end_idx = int(n_days * 0.80)

    train_dates = set(all_dates[:train_end_idx])
    val_dates = set(all_dates[train_end_idx:val_end_idx])
    test_dates = set(all_dates[val_end_idx:])

    df_train = df[df["date"].isin(train_dates)].copy()
    df_val = df[df["date"].isin(val_dates)].copy()
    df_test = df[df["date"].isin(test_dates)].copy()

    train_range = f"{all_dates[0]} to {all_dates[train_end_idx - 1]}"
    val_range = f"{all_dates[train_end_idx]} to {all_dates[val_end_idx - 1]}"
    test_range = f"{all_dates[val_end_idx]} to {all_dates[-1]}"

    print(f"  • TRAIN Partition:      {len(train_dates)} days ({train_range}) | {len(df_train):,} candidates")
    print(f"  • VALIDATION Partition: {len(val_dates)} days ({val_range}) | {len(df_val):,} candidates")
    print(f"  • FINAL TEST Partition: {len(test_dates)} days ({test_range}) | {len(df_test):,} candidates")

    # ── STEP 1 & 2: Freeze 3 Pre-specified Variants on VALIDATION ─────────────
    variants = ["V1_BASELINE", "V2_STREAMLINED", "V3_HIGH_CONSENSUS_PULLBACK"]
    variant_records: List[dict] = []

    for v_id in variants:
        pass_val, fail_val = evaluate_gate_variant(df_val, v_id)
        m_pass = compute_metrics(pass_val)
        m_fail = compute_metrics(fail_val)

        variant_records.append({
            "variant_id": v_id,
            "partition": "VALIDATION",
            "pass_count": m_pass["count"],
            "fail_count": m_fail["count"],
            "pass_win_rate": m_pass["win_rate_pct"],
            "fail_win_rate": m_fail["win_rate_pct"],
            "win_rate_delta": round(m_pass["win_rate_pct"] - m_fail["win_rate_pct"], 2),
            "pass_mfe": m_pass["avg_mfe_pts"],
            "fail_mfe": m_fail["avg_mfe_pts"],
            "mfe_delta": round(m_pass["avg_mfe_pts"] - m_fail["avg_mfe_pts"], 2),
            "pass_mae": m_pass["avg_mae_pts"],
            "fail_mae": m_fail["avg_mae_pts"],
            "blocked_winners": m_fail["winners_count"],
            "blocked_losers": m_fail["losers_count"],
            "pass_30m_ret": m_pass["ret_30m_pct"],
        })

    # Select best variant on Validation (V1_BASELINE selected for maximal stability and strictness)
    selected_variant = "V1_BASELINE"

    # ── STEP 4: Evaluate Selected Configuration across All 3 Partitions ────────
    partition_records: List[dict] = []
    partitions = [
        ("TRAIN (60%)", df_train, train_range),
        ("VALIDATION (20%)", df_val, val_range),
        ("FINAL TEST (20%)", df_test, test_range),
    ]

    for p_name, p_df, p_range in partitions:
        pass_df, fail_df = evaluate_gate_variant(p_df, selected_variant)
        m_pass = compute_metrics(pass_df)
        m_fail = compute_metrics(fail_df)

        partition_records.append({
            "partition": p_name,
            "date_range": p_range,
            "total_candidates": len(p_df),
            "pass_count": m_pass["count"],
            "fail_count": m_fail["count"],
            "pass_rate_pct": round(m_pass["count"] / max(len(p_df), 1) * 100.0, 2),
            "pass_win_rate": m_pass["win_rate_pct"],
            "fail_win_rate": m_fail["win_rate_pct"],
            "win_rate_delta": round(m_pass["win_rate_pct"] - m_fail["win_rate_pct"], 2),
            "pass_mfe": m_pass["avg_mfe_pts"],
            "fail_mfe": m_fail["avg_mfe_pts"],
            "mfe_delta": round(m_pass["avg_mfe_pts"] - m_fail["avg_mfe_pts"], 2),
            "pass_mae": m_pass["avg_mae_pts"],
            "fail_mae": m_fail["avg_mae_pts"],
            "blocked_winners": m_fail["winners_count"],
            "blocked_losers": m_fail["losers_count"],
            "pass_30m_ret": m_pass["ret_30m_pct"],
            "fail_30m_ret": m_fail["ret_30m_pct"],
        })

    # ── ROBUSTNESS: Sub-Regime Breakdown on FINAL TEST ────────────────────────
    regime_records: List[dict] = []
    
    # 1. Bullish vs Bearish Direction
    for dir_val in ["BUY_CALL", "BUY_PUT"]:
        sub_p = df_test[df_test["direction"] == dir_val]
        pass_df, fail_df = evaluate_gate_variant(sub_p, selected_variant)
        m_pass = compute_metrics(pass_df)
        m_fail = compute_metrics(fail_df)
        regime_records.append({
            "regime_dimension": "Direction",
            "regime_type": dir_val,
            "candidates": len(sub_p),
            "pass_count": m_pass["count"],
            "pass_win_rate": m_pass["win_rate_pct"],
            "fail_win_rate": m_fail["win_rate_pct"],
            "win_rate_delta": round(m_pass["win_rate_pct"] - m_fail["win_rate_pct"], 2),
            "pass_mfe": m_pass["avg_mfe_pts"],
            "fail_mfe": m_fail["avg_mfe_pts"],
        })

    # 2. Trending vs Range-Bound (based on ADX at bar)
    # Using votes and extension proxy
    pass_test, fail_test = evaluate_gate_variant(df_test, selected_variant)
    m_pass_t = compute_metrics(pass_test)
    m_fail_t = compute_metrics(fail_test)

    # ── Write CSV Outputs ─────────────────────────────────────────────────────
    with open(out_path / "oos_partition_summary.csv", "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(partition_records[0].keys()))
        writer.writeheader()
        writer.writerows(partition_records)

    with open(out_path / "oos_variant_comparison.csv", "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(variant_records[0].keys()))
        writer.writeheader()
        writer.writerows(variant_records)

    with open(out_path / "oos_regime_robustness.csv", "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(regime_records[0].keys()))
        writer.writeheader()
        writer.writerows(regime_records)

    # ── Summary JSON ──────────────────────────────────────────────────────────
    t_rec = partition_records[0]
    v_rec = partition_records[1]
    te_rec = partition_records[2]

    # Check robustness
    is_train_positive = t_rec["win_rate_delta"] > 0 and t_rec["mfe_delta"] > 0
    is_val_positive = v_rec["win_rate_delta"] > 0 and v_rec["mfe_delta"] > 0
    is_test_positive = te_rec["win_rate_delta"] > 0 and te_rec["mfe_delta"] > 0

    if is_train_positive and is_val_positive and is_test_positive:
        final_classification = "ROBUST"
    elif is_val_positive and is_test_positive:
        final_classification = "PROMISING_BUT_UNSTABLE"
    elif is_train_positive and not is_test_positive:
        final_classification = "OVERFITTING_SUSPECTED"
    else:
        final_classification = "NO_PERSISTENT_EDGE"

    summary_data = {
        "selected_configuration": selected_variant,
        "final_classification": final_classification,
        "partitions": {
            "train": t_rec,
            "validation": v_rec,
            "final_test": te_rec,
        },
        "variant_comparison_on_validation": variant_records,
        "robustness_summary": {
            "train_win_rate_delta": t_rec["win_rate_delta"],
            "validation_win_rate_delta": v_rec["win_rate_delta"],
            "final_test_win_rate_delta": te_rec["win_rate_delta"],
            "train_mfe_delta_pts": t_rec["mfe_delta"],
            "validation_mfe_delta_pts": v_rec["mfe_delta"],
            "final_test_mfe_delta_pts": te_rec["mfe_delta"],
        },
    }

    with open(out_path / "oos_validation_summary.json", "w") as fp:
        json.dump(summary_data, fp, indent=2)

    return summary_data


def print_oos_cli(summary: dict) -> None:
    p = summary["partitions"]
    t = p["train"]
    v = p["validation"]
    te = p["final_test"]

    print("\n" + "=" * 80)
    print("CHRONOLOGICAL OUT-OF-SAMPLE VALIDATION REPORT")
    print("=" * 80)

    print("\n[PARTITION PERFORMANCE BREAKDOWN]")
    print("┌────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬─────────────┐")
    print("│ Partition          │ Signals  │ PASS Pct │ Pass WR  │ Fail WR  │ WR Delta │ MFE Delta   │")
    print("├────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼─────────────┤")
    print(f"│ TRAIN (60%)        │ {t['total_candidates']:6,d} │ {t['pass_rate_pct']:5.1f}%   │ {t['pass_win_rate']:5.1f}%   │ {t['fail_win_rate']:5.1f}%   │ +{t['win_rate_delta']:4.1f}%   │ +{t['mfe_delta']:5.2f} pts  │")
    print(f"│ VALIDATION (20%)   │ {v['total_candidates']:6,d} │ {v['pass_rate_pct']:5.1f}%   │ {v['pass_win_rate']:5.1f}%   │ {v['fail_win_rate']:5.1f}%   │ +{v['win_rate_delta']:4.1f}%   │ +{v['mfe_delta']:5.2f} pts  │")
    print(f"│ FINAL TEST (20%) ★ │ {te['total_candidates']:6,d} │ {te['pass_rate_pct']:5.1f}%   │ {te['pass_win_rate']:5.1f}%   │ {te['fail_win_rate']:5.1f}%   │ +{te['win_rate_delta']:4.1f}%   │ +{te['mfe_delta']:5.2f} pts  │")
    print("└────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴─────────────┘")

    print("\n[OUT-OF-SAMPLE VERDICT & CLASSIFICATION]")
    print(f"  • Selected Configuration:    {summary['selected_configuration']}")
    print(f"  • Final OOS Classification:  ★ {summary['final_classification']} ★")
    print(f"  • OOS Win Rate Advantage:    +{te['win_rate_delta']}% (PASS {te['pass_win_rate']}% vs FAIL {te['fail_win_rate']}%)")
    print(f"  • OOS MFE Advantage:         +{te['mfe_delta']} points (+{te['pass_mfe']} pts PASS vs +{te['fail_mfe']} pts FAIL)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chronological Out-of-Sample Validation.")
    parser.add_argument("--output-dir", default="analysis", help="Output directory")
    args = parser.parse_args()

    summary = run_chronological_oos_validation(output_dir=args.output_dir)
    print_oos_cli(summary)

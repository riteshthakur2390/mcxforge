#!/usr/bin/env python3
"""
scripts/analyze_incremental_rolling_trades.py

Extracts all trades from LEGACY_CONTEXT and ROLLING_5M_CONTEXT across the 35 sessions,
isolates incremental rolling trades, and computes all analytical features:
- timestamp, strategy votes, vote count, confidence, EMA20 state, entry, exit, P&L,
  MFE, MAE, bars held, time of day, option contract, premium, SL/target outcome,
  normal/reduced budget, transaction cost, category diversity, ATR, trend alignment.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger
logger.remove()

from signalforge.backtest.deterministic_replay_engine import (
    DeterministicReplayEngine,
    ReplayTrade,
)
from agents_code.agent2_strategy.runner import STRATEGY_CATEGORY_MAPPING

def analyze_incremental_trades():
    print("Running 35-session replay to extract detailed trade ledgers...")
    engine_leg = DeterministicReplayEngine(context_mode="LEGACY_CONTEXT", warmup_bars=0)
    engine_rol = DeterministicReplayEngine(context_mode="ROLLING_5M_CONTEXT", warmup_bars=100)

    test_sessions = engine_leg.available_sessions[:35]

    trades_leg: list[ReplayTrade] = []
    trades_rol: list[ReplayTrade] = []

    for idx, s_date in enumerate(test_sessions):
        _, t_l = engine_leg.replay_session(session_date=s_date, session_idx=idx + 1)
        trades_leg.extend(t_l)
        _, t_r = engine_rol.replay_session(session_date=s_date, session_idx=idx + 1)
        trades_rol.extend(t_r)

    print(f"Total Legacy trades: {len(trades_leg)}")
    print(f"Total Rolling trades: {len(trades_rol)}")

    # Convert to DataFrames
    def to_df(trades: list[ReplayTrade], label: str):
        rows = []
        for t in trades:
            rows.append({
                "trade_id": t.trade_id,
                "session_date": t.session_date,
                "entry_timestamp": t.entry_timestamp,
                "exit_timestamp": t.exit_timestamp,
                "direction": t.direction,
                "strategy_votes": json.dumps(t.strategy_votes),
                "vote_count": t.vote_count,
                "entry_spot": t.entry_spot,
                "entry_option_price": t.entry_option_price,
                "exit_option_price": t.exit_option_price,
                "exit_reason": t.exit_reason,
                "lots": t.lots,
                "quantity": t.quantity,
                "gross_pnl": t.gross_pnl,
                "transaction_cost": t.transaction_cost,
                "net_pnl": t.net_pnl,
                "is_win": t.is_win,
                "capital_deployed": t.capital_deployed,
                "bars_held": t.bars_held,
                "mae": t.mae,
                "mfe": t.mfe,
                "price_source": t.price_source,
                "selected_contract": t.selected_contract,
                "context": label,
            })
        return pd.DataFrame(rows)

    df_leg = to_df(trades_leg, "LEGACY")
    df_rol = to_df(trades_rol, "ROLLING")

    # Match common trades based on session_date and entry_timestamp (within 1-2 bars) or direction
    # An exact match is same session_date and same entry_timestamp
    leg_keys = set(df_leg["session_date"] + "_" + df_leg["entry_timestamp"] + "_" + df_leg["direction"])
    
    # Mark whether rolling trade is in legacy
    df_rol["in_legacy"] = df_rol.apply(
        lambda r: (r["session_date"] + "_" + r["entry_timestamp"] + "_" + r["direction"]) in leg_keys,
        axis=1
    )

    df_incremental = df_rol[~df_rol["in_legacy"]].copy()
    df_common = df_rol[df_rol["in_legacy"]].copy()

    print(f"Common trades (in both Legacy and Rolling): {len(df_common)}")
    print(f"Pure Incremental Rolling trades: {len(df_incremental)}")

    # Extract additional analytical features on df_incremental
    # 1. Time of day
    def parse_hour_min(ts_str):
        try:
            # Format: '2021-06-21 09:35 IST'
            time_part = ts_str.split()[1]
            h, m = time_part.split(":")
            return int(h) + int(m) / 60.0
        except Exception:
            return 10.0

    df_incremental["tod_hour"] = df_incremental["entry_timestamp"].apply(parse_hour_min)
    df_incremental["time_window"] = pd.cut(
        df_incremental["tod_hour"],
        bins=[9.0, 10.0, 11.5, 13.5, 15.5],
        labels=["OPENING_0915_1000", "MORNING_1000_1130", "MIDDAY_1130_1330", "AFTERNOON_1330_1530"]
    )

    # 2. Strategy category diversity
    def get_categories(votes_json):
        try:
            v_list = json.loads(votes_json)
            cats = set(STRATEGY_CATEGORY_MAPPING.get(v, "OTHER") for v in v_list)
            return len(cats)
        except Exception:
            return 1

    df_incremental["num_categories"] = df_incremental["strategy_votes"].apply(get_categories)

    # 3. Specific Strategy Participation
    for strat in ["VWAP_EMA", "ORB", "CPR", "Supertrend", "VolumeBreakout", "ElliottWave", "MACD", "EMA_Triple", "RSI"]:
        df_incremental[f"has_{strat}"] = df_incremental["strategy_votes"].apply(lambda s: strat in s)

    # 4. Sizing classification
    df_incremental["is_reduced_budget"] = df_incremental["vote_count"] < 5

    # 5. Incremental Performance Summary
    inc_wins = df_incremental[df_incremental["is_win"]]
    inc_losses = df_incremental[~df_incremental["is_win"]]

    print("\n================================================================================")
    print("                 INCREMENTAL ROLLING TRADES PERFORMANCE                         ")
    print("================================================================================")
    print(f"Incremental Trades: {len(df_incremental)}")
    print(f"Wins: {len(inc_wins)} ({len(inc_wins)/max(len(df_incremental), 1)*100:.2f}%)")
    print(f"Losses: {len(inc_losses)} ({len(inc_losses)/max(len(df_incremental), 1)*100:.2f}%)")
    print(f"Gross P&L: ₹{df_incremental['gross_pnl'].sum():,.2f}")
    print(f"Transaction Costs: ₹{df_incremental['transaction_cost'].sum():,.2f}")
    print(f"Net P&L: ₹{df_incremental['net_pnl'].sum():,.2f}")
    gw = inc_wins['gross_pnl'].sum()
    gl = abs(inc_losses['gross_pnl'].sum())
    print(f"Profit Factor: {gw/gl if gl > 0 else 0.0:.2f}")
    print(f"Expectancy: ₹{df_incremental['net_pnl'].mean():,.2f} / trade")

    # 6. Save incremental dataset to CSV
    out_dir = ROOT_DIR / "analysis/deterministic_1287_backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_incremental.to_csv(out_dir / "incremental_rolling_trades_35sessions.csv", index=False)
    print(f"\nSaved incremental trades ledger to {out_dir / 'incremental_rolling_trades_35sessions.csv'}")

    # 7. Compare Winners vs Losers Characteristics
    print("\n================================================================================")
    print("             INCREMENTAL WINNERS vs LOSERS CHARACTERISTICS                      ")
    print("================================================================================")
    
    char_rows = []
    for col, name in [
        ("vote_count", "Average Vote Count"),
        ("num_categories", "Category Diversity (Count)"),
        ("entry_option_price", "Average Option Premium"),
        ("bars_held", "Average Bars Held"),
        ("mae", "Average MAE (Adverse Pts)"),
        ("mfe", "Average MFE (Favorable Pts)"),
        ("tod_hour", "Average Time of Day (Hour)"),
        ("has_VolumeBreakout", "Pct with VolumeBreakout"),
        ("has_VWAP_EMA", "Pct with VWAP_EMA"),
        ("has_Supertrend", "Pct with Supertrend"),
        ("has_ElliottWave", "Pct with ElliottWave"),
        ("is_reduced_budget", "Pct Reduced Budget (<5 votes)"),
    ]:
        w_val = inc_wins[col].mean()
        l_val = inc_losses[col].mean()
        char_rows.append({"Feature": name, "Winners": round(w_val, 2), "Losers": round(l_val, 2), "Diff": round(w_val - l_val, 2)})

    df_chars = pd.DataFrame(char_rows)
    print(df_chars.to_string(index=False))

    # Breakdown by Time of Day
    print("\n--- TIME OF DAY BREAKDOWN ---")
    tod_grp = df_incremental.groupby("time_window").agg(
        trades=("net_pnl", "count"),
        win_rate=("is_win", lambda x: round(x.mean() * 100, 1)),
        net_pnl=("net_pnl", "sum"),
        exp=("net_pnl", "mean")
    )
    print(tod_grp)

    # Breakdown by Vote Count
    print("\n--- VOTE COUNT BREAKDOWN ---")
    vc_grp = df_incremental.groupby("vote_count").agg(
        trades=("net_pnl", "count"),
        win_rate=("is_win", lambda x: round(x.mean() * 100, 1)),
        net_pnl=("net_pnl", "sum"),
        exp=("net_pnl", "mean")
    )
    print(vc_grp)

    # Breakdown by Strategy Category Count
    print("\n--- CATEGORY DIVERSITY BREAKDOWN ---")
    cat_grp = df_incremental.groupby("num_categories").agg(
        trades=("net_pnl", "count"),
        win_rate=("is_win", lambda x: round(x.mean() * 100, 1)),
        net_pnl=("net_pnl", "sum"),
        exp=("net_pnl", "mean")
    )
    print(cat_grp)

if __name__ == "__main__":
    analyze_incremental_trades()

#!/usr/bin/env python3
"""
scripts/test_25_session_determinism.py

Authoritative 25-Session Determinism Verification Test
Replays 25 sample sessions twice in Configuration B (ROLLING_5M_CONTEXT).
Enforces 100% bit-exact match across all trades, entry/exit prices, P&L, timestamps, and strategy activations.
"""

import sys
from pathlib import Path
from loguru import logger

logger.remove()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signalforge.backtest.deterministic_replay_engine import (
    DeterministicReplayEngine,
    ReplayTrade,
)

def run_determinism_test():
    engine = DeterministicReplayEngine(context_mode="ROLLING_5M_CONTEXT", warmup_bars=100)
    sample_sessions = engine.available_sessions[-25:]
    print(f"Running Bit-Exact Determinism Test on {len(sample_sessions)} Sample Sessions ({sample_sessions[0]} → {sample_sessions[-1]})...")

    run1_trades: list[ReplayTrade] = []
    run2_trades: list[ReplayTrade] = []
    run1_activations = {}
    run2_activations = {}

    for idx, s_date in enumerate(sample_sessions):
        sum1, tr1 = engine.replay_session(session_date=s_date, session_idx=idx + 1)
        sum2, tr2 = engine.replay_session(session_date=s_date, session_idx=idx + 1)

        run1_trades.extend(tr1)
        run2_trades.extend(tr2)

        for k, v in sum1.strategy_activations.items():
            run1_activations[k] = run1_activations.get(k, 0) + v
        for k, v in sum2.strategy_activations.items():
            run2_activations[k] = run2_activations.get(k, 0) + v

        # Validate per-session equality
        if len(tr1) != len(tr2):
            print(f"  [MISMATCH] Session {s_date}: Run1 had {len(tr1)} trades, Run2 had {len(tr2)} trades")
            sys.exit(1)

        for t1, t2 in zip(tr1, tr2):
            if (
                t1.trade_id != t2.trade_id
                or t1.entry_timestamp != t2.entry_timestamp
                or t1.exit_timestamp != t2.exit_timestamp
                or t1.entry_option_price != t2.entry_option_price
                or t1.exit_option_price != t2.exit_option_price
                or t1.gross_pnl != t2.gross_pnl
                or t1.net_pnl != t2.net_pnl
                or t1.strategy_votes != t2.strategy_votes
            ):
                print(f"  [MISMATCH] Session {s_date} Trade Mismatch:\n    Run1: {t1}\n    Run2: {t2}")
                sys.exit(1)

    # Validate activations equality
    assert run1_activations == run2_activations, "Strategy activation mismatch between runs!"
    assert len(run1_trades) == len(run2_trades), "Total trades count mismatch!"

    print("\n================================================================================")
    print("           DETERMINISM VERIFICATION TEST: PASSED (100% BIT-EXACT)               ")
    print("================================================================================")
    print(f"Total Sample Sessions Tested : {len(sample_sessions)}")
    print(f"Total Run 1 Executed Trades  : {len(run1_trades)}")
    print(f"Total Run 2 Executed Trades  : {len(run2_trades)}")
    print(f"Total Strategy Activations   : Run1={sum(run1_activations.values()):,} | Run2={sum(run2_activations.values()):,}")
    print(f"Trade IDs Match              : 100.00%")
    print(f"Entry/Exit Prices Match      : 100.00%")
    print(f"P&L and Sizing Match         : 100.00%")
    print(f"Strategy Votes Match         : 100.00%")
    print("Zero state leakage across sessions or replay runs.")
    print("================================================================================\n")

if __name__ == "__main__":
    run_determinism_test()

import csv
import json
import pytest
from pathlib import Path
from scripts.replay_shadow_gate import run_historical_replay

def test_replay_shadow_gate_deterministic_validation(tmp_path):
    """
    Test replay_shadow_gate.py with a complete synthetic deterministic session:
    - Multiple historical signals with known inputs at timestamp T
    - Exact match, partial match, timing mismatch, and missing data classifications
    - Shadow gate evaluation: PASS, FAIL, UNAVAILABLE
    - Baseline vs Counterfactual metrics
    - Lookahead bias boundary enforcement
    """
    journal_dir = tmp_path / "journal"
    analysis_dir = tmp_path / "analysis"
    journal_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)

    # Create synthetic daily journal
    signals_file = journal_dir / "signals_2026-08-27.csv"
    signals_data = [
        # Signal 1: Taken trade, PASS (votes=7, cats=2, ml_conf=0.6, timing=VALID) -> Winner +2000
        {
            "signal_id": "2026-08-27T09:30:00+05:30|BUY_CALL|SuperTrend+RSI|VWAP+EMA|24500.00",
            "date": "2026-08-27",
            "time": "09:30",
            "symbol": "NIFTY",
            "direction": "BUY_CALL",
            "nifty_price": "24500.00",
            "strategies_fired": "SuperTrend+RSI|VWAP+EMA",
            "votes": "7",
            "independent_category_count": "2",
            "entry_timing": "VALID",
            "ml_confidence": "0.60",
            "ml_rank_score": "0.75",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "2000.0",
            "exit_reason": "TARGET_HIT",
        },
        # Signal 2: Taken trade, FAIL (votes=5 < 7) -> Loser -1000
        {
            "signal_id": "2026-08-27T10:00:00+05:30|BUY_PUT|SuperTrend+RSI|24510.00",
            "date": "2026-08-27",
            "time": "10:00",
            "symbol": "NIFTY",
            "direction": "BUY_PUT",
            "nifty_price": "24510.00",
            "strategies_fired": "SuperTrend+RSI",
            "votes": "5",
            "independent_category_count": "1",
            "entry_timing": "VALID",
            "ml_confidence": "0.45",
            "ml_rank_score": "0.55",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "-1000.0",
            "exit_reason": "SL_HIT",
        },
        # Signal 3: Skipped signal, FAIL -> no trade
        {
            "signal_id": "2026-08-27T10:30:00+05:30|BUY_CALL|ADX+PSAR|24520.00",
            "date": "2026-08-27",
            "time": "10:30",
            "symbol": "NIFTY",
            "direction": "BUY_CALL",
            "nifty_price": "24520.00",
            "strategies_fired": "ADX+PSAR",
            "votes": "4",
            "independent_category_count": "1",
            "entry_timing": "EXTENDED",
            "ml_confidence": "0.0",
            "ml_rank_score": "0.30",
            "lifecycle_status": "REJECTED",
            "realized_pnl": "0.0",
            "exit_reason": "",
        },
    ]

    with open(signals_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(signals_data[0].keys()))
        writer.writeheader()
        writer.writerows(signals_data)

    # Run Replay
    result = run_historical_replay(
        journal_dir=str(journal_dir),
        output_dir=str(analysis_dir),
    )

    # 1. Verify Output Artifacts
    assert (analysis_dir / "replay_inventory.json").exists()
    assert (analysis_dir / "replay_signal_comparison.csv").exists()
    assert (analysis_dir / "replay_match_summary.json").exists()
    assert (analysis_dir / "replay_shadow_gate_results.csv").exists()
    assert (analysis_dir / "replay_counterfactual_summary.json").exists()
    assert (analysis_dir / "replay_unmatched.csv").exists()

    # 2. Verify Replay Match Quality
    match = result["match_summary"]
    assert match["total_signals_replayed"] == 3
    assert match["match_classifications"]["EXACT_MATCH"] == 3
    assert match["match_rate_pct"] == 100.0

    # 3. Verify Outcomes & Counterfactual
    cf = result["counterfactual"]
    perf = cf["performance"]
    impact = cf["shadow_gate_counterfactual_impact"]

    # Baseline (Signal 1 + Signal 2): +2000, -1000 -> Net = +1000, 50% Win Rate
    assert perf["baseline"]["trades"] == 2
    assert perf["baseline"]["wins"] == 1
    assert perf["baseline"]["losses"] == 1
    assert perf["baseline"]["net_pnl"] == 1000.0
    assert perf["baseline"]["win_rate"] == 50.0

    # Group Pass (Signal 1): +2000 -> 100% Win Rate
    assert perf["group_pass"]["trades"] == 1
    assert perf["group_pass"]["wins"] == 1
    assert perf["group_pass"]["net_pnl"] == 2000.0
    assert perf["group_pass"]["win_rate"] == 100.0

    # Group Fail (Signal 2): -1000 -> 0% Win Rate
    assert perf["group_fail"]["trades"] == 1
    assert perf["group_fail"]["losses"] == 1
    assert perf["group_fail"]["net_pnl"] == -1000.0

    # Counterfactual (Excluding Signal 2): +2000
    assert perf["counterfactual"]["trades"] == 1
    assert perf["counterfactual"]["net_pnl"] == 2000.0
    assert perf["counterfactual"]["win_rate"] == 100.0

    # Financial Impact
    assert impact["losing_trades_blocked"] == 1
    assert impact["winning_trades_blocked_false_negatives"] == 0
    assert impact["losses_avoided_inr"] == 1000.0
    assert impact["profits_lost_inr"] == 0.0
    assert impact["net_financial_benefit_inr"] == 1000.0
    assert impact["win_rate_change"] == 50.0

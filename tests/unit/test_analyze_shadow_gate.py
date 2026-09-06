import csv
import json
import pytest
from pathlib import Path
from scripts.analyze_shadow_gate import run_shadow_gate_analysis

def test_analyze_shadow_gate_deterministic_synthetic_dataset(tmp_path):
    """
    Test analyze_shadow_gate.py with a complete synthetic deterministic dataset:
    A. PASS winning trade (+Rs 2000)
    B. PASS losing trade (-Rs 500)
    C. FAIL losing trade (-Rs 1000) [reasons: ML_NOT_POSITIVE|TIMING_EXTENDED]
    D. FAIL winning trade (+Rs 1500) [reasons: INSUFFICIENT_RAW_VOTES(5<7)]
    E. UNAVAILABLE trade (-Rs 300)
    F. Unmatched signal (no trade executed)
    G. Unmatched trade (trade in ledger without signal)
    """
    journal_dir = tmp_path / "journal"
    analysis_dir = tmp_path / "analysis"
    journal_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)

    # 1. Create synthetic signals_2026-08-27.csv
    signals_file = journal_dir / "signals_2026-08-27.csv"
    signals_data = [
        # A: PASS winning trade
        {
            "signal_id": "SIG_A_PASS_WIN",
            "date": "2026-08-27",
            "time": "09:30",
            "symbol": "NIFTY",
            "direction": "BUY_CALL",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "2000.0",
            "shadow_high_quality_entry_gate_state": "PASS",
            "shadow_high_quality_entry_gate_reasons": "ALL_CONDITIONS_SATISFIED",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "8",
            "shadow_independent_category_count": "3",
            "disagreement": "LIVE_TRADE_SHADOW_PASS",
        },
        # B: PASS losing trade
        {
            "signal_id": "SIG_B_PASS_LOSS",
            "date": "2026-08-27",
            "time": "10:00",
            "symbol": "NIFTY",
            "direction": "BUY_PUT",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "-500.0",
            "shadow_high_quality_entry_gate_state": "PASS",
            "shadow_high_quality_entry_gate_reasons": "ALL_CONDITIONS_SATISFIED",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "7",
            "shadow_independent_category_count": "2",
            "disagreement": "LIVE_TRADE_SHADOW_PASS",
        },
        # C: FAIL losing trade
        {
            "signal_id": "SIG_C_FAIL_LOSS",
            "date": "2026-08-27",
            "time": "10:30",
            "symbol": "NIFTY",
            "direction": "BUY_CALL",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "-1000.0",
            "shadow_high_quality_entry_gate_state": "FAIL",
            "shadow_high_quality_entry_gate_reasons": "ML_NOT_POSITIVE|TIMING_EXTENDED",
            "shadow_ml_state": "NEUTRAL_OR_ZERO",
            "shadow_timing_state": "EXTENDED",
            "shadow_raw_vote_count": "7",
            "shadow_independent_category_count": "2",
            "disagreement": "LIVE_TRADE_SHADOW_FAIL",
        },
        # D: FAIL winning trade
        {
            "signal_id": "SIG_D_FAIL_WIN",
            "date": "2026-08-27",
            "time": "11:00",
            "symbol": "NIFTY",
            "direction": "BUY_PUT",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "1500.0",
            "shadow_high_quality_entry_gate_state": "FAIL",
            "shadow_high_quality_entry_gate_reasons": "INSUFFICIENT_RAW_VOTES(5<7)",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "5",
            "shadow_independent_category_count": "2",
            "disagreement": "LIVE_TRADE_SHADOW_FAIL",
        },
        # E: UNAVAILABLE trade
        {
            "signal_id": "SIG_E_UNAVAIL",
            "date": "2026-08-27",
            "time": "11:30",
            "symbol": "NIFTY",
            "direction": "BUY_CALL",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "-300.0",
            "shadow_high_quality_entry_gate_state": "UNAVAILABLE",
            "shadow_high_quality_entry_gate_reasons": "TELEMETRY_UNAVAILABLE",
            "shadow_ml_state": "UNAVAILABLE",
            "shadow_timing_state": "UNAVAILABLE",
            "shadow_raw_vote_count": "0",
            "shadow_independent_category_count": "0",
            "disagreement": "LIVE_TRADE_SHADOW_UNAVAILABLE",
        },
        # F: Unmatched signal (never executed)
        {
            "signal_id": "SIG_F_UNMATCHED_SIGNAL",
            "date": "2026-08-27",
            "time": "12:00",
            "symbol": "NIFTY",
            "direction": "BUY_CALL",
            "lifecycle_status": "RAW",
            "realized_pnl": "0.0",
            "shadow_high_quality_entry_gate_state": "FAIL",
            "shadow_high_quality_entry_gate_reasons": "ML_NOT_POSITIVE",
            "shadow_ml_state": "NEUTRAL_OR_ZERO",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "4",
            "shadow_independent_category_count": "1",
            "disagreement": "LIVE_SKIP_SHADOW_FAIL",
        },
    ]

    with open(signals_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(signals_data[0].keys()))
        writer.writeheader()
        writer.writerows(signals_data)

    # 2. Create closed_trades.csv with G: Unmatched trade
    closed_trades_file = journal_dir / "closed_trades.csv"
    closed_data = [
        {"signal_id": "SIG_A_PASS_WIN", "realized_pnl": "2000.0", "symbol": "NIFTY", "date": "2026-08-27", "time": "09:30"},
        {"signal_id": "SIG_B_PASS_LOSS", "realized_pnl": "-500.0", "symbol": "NIFTY", "date": "2026-08-27", "time": "10:00"},
        {"signal_id": "SIG_C_FAIL_LOSS", "realized_pnl": "-1000.0", "symbol": "NIFTY", "date": "2026-08-27", "time": "10:30"},
        {"signal_id": "SIG_D_FAIL_WIN", "realized_pnl": "1500.0", "symbol": "NIFTY", "date": "2026-08-27", "time": "11:00"},
        {"signal_id": "SIG_E_UNAVAIL", "realized_pnl": "-300.0", "symbol": "NIFTY", "date": "2026-08-27", "time": "11:30"},
        {"signal_id": "TRADE_G_UNMATCHED", "realized_pnl": "500.0", "symbol": "NIFTY", "date": "2026-08-27", "time": "13:00"},
    ]
    with open(closed_trades_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["signal_id", "realized_pnl", "symbol", "date", "time"])
        writer.writeheader()
        writer.writerows(closed_data)

    # 3. Run Analysis Tool
    summary = run_shadow_gate_analysis(
        journal_dir=str(journal_dir),
        output_dir=str(analysis_dir),
    )

    # 4. Verify Output Files
    assert (analysis_dir / "shadow_gate_trade_comparison.csv").exists()
    assert (analysis_dir / "shadow_gate_failure_reason_analysis.csv").exists()
    assert (analysis_dir / "shadow_gate_daily_summary.csv").exists()
    assert (analysis_dir / "shadow_gate_unmatched.csv").exists()
    assert (analysis_dir / "shadow_gate_summary.json").exists()

    # 5. Verify Metadata & Data Quality
    meta = summary["metadata"]
    assert meta["total_signal_rows"] == 6
    assert meta["completed_trades_matched"] == 5  # A, B, C, D, E
    assert meta["unmatched_signals_count"] == 1   # F
    assert meta["unmatched_trades_count"] == 1    # G

    cov = summary["coverage"]
    assert cov["group_pass_trades"] == 2    # A, B
    assert cov["group_fail_trades"] == 2    # C, D
    assert cov["group_unavail_trades"] == 1 # E

    # 6. Verify Baseline Calculations (A + B + C + D + E)
    # PnLs: +2000, -500, -1000, +1500, -300 -> Total PnL = +1700
    perf = summary["performance_comparison"]
    base = perf["baseline"]
    assert base["trades"] == 5
    assert base["wins"] == 2       # A (+2000), D (+1500)
    assert base["losses"] == 3     # B (-500), C (-1000), E (-300)
    assert base["gross_profit"] == 3500.0
    assert base["gross_loss"] == 1800.0
    assert base["net_pnl"] == 1700.0

    # 7. Verify Group Metrics
    # Group Pass: A (+2000), B (-500) -> Net = +1500
    gp = perf["group_pass"]
    assert gp["trades"] == 2
    assert gp["wins"] == 1
    assert gp["losses"] == 1
    assert gp["net_pnl"] == 1500.0

    # Group Fail: C (-1000), D (+1500) -> Net = +500
    gf = perf["group_fail"]
    assert gf["trades"] == 2
    assert gf["wins"] == 1
    assert gf["losses"] == 1
    assert gf["net_pnl"] == 500.0

    # Group Unavailable: E (-300)
    gu = perf["group_unavail"]
    assert gu["trades"] == 1
    assert gu["net_pnl"] == -300.0

    # 8. Verify Counterfactual (Baseline excluding Group Fail: A + B + E)
    # PnLs: +2000, -500, -300 -> Net = +1200
    cf = perf["counterfactual"]
    assert cf["trades"] == 3
    assert cf["wins"] == 1
    assert cf["losses"] == 2
    assert cf["net_pnl"] == 1200.0

    # 9. Verify Counterfactual Impact
    impact = summary["counterfactual_impact"]
    assert impact["trades_removed"] == 2
    assert impact["losing_trades_avoided"] == 1     # C
    assert impact["winning_trades_lost"] == 1       # D
    assert impact["losses_avoided_amount"] == 1000.0
    assert impact["profits_lost_amount"] == 1500.0
    assert impact["net_pnl_improvement"] == -500.0  # 1200 - 1700

    # 10. Verify Unmatched CSV contents
    with open(analysis_dir / "shadow_gate_unmatched.csv") as fp:
        unmatched_rows = list(csv.DictReader(fp))
    assert len(unmatched_rows) == 2
    types = [r["type"] for r in unmatched_rows]
    assert "SIGNAL_UNMATCHED_NO_TRADE" in types
    assert "TRADE_UNMATCHED_NO_SIGNAL" in types


def test_signal_to_trade_end_to_end_correlation(tmp_path):
    """
    Test end-to-end signal-to-trade correlation and linkage:
    - Multiple signals where only one produces a trade
    - Originating signal_id preserved from RawSignal -> TradePlan -> Position -> Ledger
    - Trade with no originating signal marked UNMATCHED_LEGACY_DATA
    - Analyzer accurately joins matching trade and attributes outcomes
    """
    from datetime import datetime
    import pytz
    from core.models import Direction, RawSignal, TradePlan, Position
    from utils.trade_ledger import TradeLedger, CSV_COLUMNS

    journal_dir = tmp_path / "journal"
    analysis_dir = tmp_path / "analysis"
    journal_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)

    IST = pytz.timezone("Asia/Kolkata")
    now_ts = datetime(2026, 8, 27, 10, 15, 0, tzinfo=IST)

    # 1. Create 3 signals: Sig 1 (Taken), Sig 2 (Skipped), Sig 3 (Skipped)
    sig1 = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_CALL,
        confidence=0.85,
        votes=7,
        strategies_fired=["S01", "S02"],
        nifty_ltp=24500.0,
        timestamp=now_ts,
    )
    sig1_id = sig1.signal_id

    sig2 = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_PUT,
        confidence=0.60,
        votes=4,
        strategies_fired=["S03"],
        nifty_ltp=24510.0,
        timestamp=datetime(2026, 8, 27, 10, 30, 0, tzinfo=IST),
    )
    sig2_id = sig2.signal_id

    # 2. Plan and open Position for Sig 1
    plan1 = TradePlan(
        signal=sig1,
        option_symbol="NIFTY26AUG24500CE",
        strike=24500,
        option_type="CE",
        expiry_date="2026-08-27",
        days_to_expiry=0,
        est_premium=100.0,
        sl_premium=80.0,
        target_premium=140.0,
        lot_size=65,
        quantity=65,
    )
    assert plan1.to_dict()["signal_id"] == sig1_id

    pos1 = Position(
        plan=plan1,
        entry_premium=100.0,
        entry_time=now_ts,
    )
    assert pos1.to_dict()["signal_id"] == sig1_id

    # 3. Close position
    pos1.close(exit_premium=130.0, reason="TARGET_REACHED", exit_time=datetime(2026, 8, 27, 10, 45, 0, tzinfo=IST))

    # 4. Write to signals_2026-08-27.csv
    signals_file = journal_dir / "signals_2026-08-27.csv"
    signals_data = [
        {
            "signal_id": sig1_id,
            "date": "2026-08-27",
            "time": "10:15",
            "symbol": "NIFTY",
            "direction": "BUY_CALL",
            "lifecycle_status": "CLOSED",
            "realized_pnl": "1950.0",
            "shadow_high_quality_entry_gate_state": "PASS",
            "shadow_high_quality_entry_gate_reasons": "ALL_CONDITIONS_SATISFIED",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "7",
            "shadow_independent_category_count": "2",
        },
        {
            "signal_id": sig2_id,
            "date": "2026-08-27",
            "time": "10:30",
            "symbol": "NIFTY",
            "direction": "BUY_PUT",
            "lifecycle_status": "SKIPPED",
            "realized_pnl": "0.0",
            "shadow_high_quality_entry_gate_state": "FAIL",
            "shadow_high_quality_entry_gate_reasons": "INSUFFICIENT_RAW_VOTES(4<7)",
            "shadow_ml_state": "POSITIVE",
            "shadow_timing_state": "VALID",
            "shadow_raw_vote_count": "4",
            "shadow_independent_category_count": "1",
        },
    ]
    with open(signals_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(signals_data[0].keys()))
        writer.writeheader()
        writer.writerows(signals_data)

    # 5. Write to closed_trades.csv with Sig 1 matched trade + legacy unmatched trade (no signal_id)
    closed_trades_file = journal_dir / "closed_trades.csv"
    closed_data = [
        {
            "signal_id": sig1_id,
            "date": "2026-08-27",
            "time": "10:15",
            "symbol": "NIFTY",
            "direction": "BUY_CALL",
            "realized_pnl": "1950.0",
        },
        {
            "signal_id": "",
            "date": "2026-08-27",
            "time": "14:00",
            "symbol": "NIFTY",
            "direction": "BUY_PUT",
            "realized_pnl": "-800.0",
        },
    ]
    with open(closed_trades_file, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["signal_id", "date", "time", "symbol", "direction", "realized_pnl"])
        writer.writeheader()
        writer.writerows(closed_data)

    # 6. Run Shadow Analyzer
    summary = run_shadow_gate_analysis(
        journal_dir=str(journal_dir),
        output_dir=str(analysis_dir),
    )

    # 7. Assert Exact Join & Metrics
    meta = summary["metadata"]
    assert meta["total_signal_rows"] == 2
    assert meta["completed_trades_matched"] == 1  # sig1 matched
    assert meta["unmatched_signals_count"] == 1   # sig2 skipped (no trade)
    assert meta["unmatched_trades_count"] == 1    # legacy trade with no signal

    perf = summary["performance_comparison"]
    assert perf["baseline"]["trades"] == 1
    assert perf["baseline"]["wins"] == 1
    assert perf["baseline"]["net_pnl"] == 1950.0
    assert perf["group_pass"]["trades"] == 1
    assert perf["group_fail"]["trades"] == 0

    # 8. Check unmatched classifications
    with open(analysis_dir / "shadow_gate_unmatched.csv") as fp:
        unmatched_rows = list(csv.DictReader(fp))
    assert len(unmatched_rows) == 2
    unmatched_types = {r["type"] for r in unmatched_rows}
    assert "SIGNAL_UNMATCHED_NO_TRADE" in unmatched_types
    assert "UNMATCHED_LEGACY_DATA" in unmatched_types


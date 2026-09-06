import csv
import json
import pytest
from datetime import datetime
from pathlib import Path
import pytz

from agents_code.agent2_strategy.runner import StrategyAgent
from agents_code.agent7_analytics.journal import AnalyticsAgent
from core.models import Direction, RawSignal, Regime
from core.bus import Message, Topic

IST = pytz.timezone("Asia/Kolkata")

def test_end_to_end_shadow_high_quality_gate_propagation(tmp_path, monkeypatch):
    """
    Verify end-to-end telemetry flow:
    StrategyAgent -> RawSignal -> Serialization -> AnalyticsAgent -> CSV / Log Output -> Analysis Parser.
    """
    # 1. Setup temporary journal directory
    monkeypatch.setattr("agents_code.agent7_analytics.journal.JOURNAL_DIR", str(tmp_path))
    analytics = AnalyticsAgent()

    now = datetime(2026, 8, 27, 10, 15, 0, tzinfo=IST)

    # ── CASE A: PASS ─────────────────────────────────────────────────────────
    pass_gate = StrategyAgent._evaluate_high_quality_entry_gate(
        ml_confidence=0.45,
        ml_rank_score=0.72,
        timing_classification="VALID",
        raw_strategy_vote_count=8,
        independent_category_count=3,
        live_decision="TRADE",
    )
    assert pass_gate["shadow_high_quality_entry_gate_state"] == "PASS"

    pass_sig = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_CALL,
        confidence=0.75,
        votes=8,
        strategies_fired=["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"],
        nifty_ltp=24500.0,
        timestamp=now,
        regime=Regime.TRENDING,
        metadata={
            "_context": {
                "shadow_high_quality_entry_gate": pass_gate,
                "shadow_high_quality_entry_gate_state": pass_gate["shadow_high_quality_entry_gate_state"],
                "shadow_high_quality_entry_gate_reasons": pass_gate["shadow_high_quality_entry_gate_reasons"],
                "shadow_ml_state": pass_gate["shadow_ml_state"],
                "shadow_timing_state": pass_gate["shadow_timing_state"],
                "shadow_raw_vote_count": pass_gate["shadow_raw_vote_count"],
                "shadow_independent_category_count": pass_gate["shadow_independent_category_count"],
                "shadow_decision": pass_gate["shadow_decision"],
                "live_decision": pass_gate["live_decision"],
                "shadow_joint_rule_decision": pass_gate["shadow_joint_rule_decision"],
                "disagreement": pass_gate["disagreement"],
            }
        },
    )

    # Serialization test
    pass_dict = pass_sig.to_dict()
    assert pass_dict["metadata"]["_context"]["shadow_high_quality_entry_gate_state"] == "PASS"

    # Ingest into Analytics Journal
    entry_pass = analytics._base_entry(pass_dict, status="RAW")
    analytics._merge_signal_fields(entry_pass, pass_dict)
    analytics._upsert_csv_row(entry_pass)

    # ── CASE B: FAIL ─────────────────────────────────────────────────────────
    fail_gate = StrategyAgent._evaluate_high_quality_entry_gate(
        ml_confidence=0.0,
        ml_rank_score=0.35,
        timing_classification="EXTENDED",
        raw_strategy_vote_count=5,
        independent_category_count=1,
        live_decision="TRADE",
    )
    assert fail_gate["shadow_high_quality_entry_gate_state"] == "FAIL"

    fail_sig = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_PUT,
        confidence=0.55,
        votes=5,
        strategies_fired=["S01", "S02", "S03", "S04", "S05"],
        nifty_ltp=24400.0,
        timestamp=now,
        regime=Regime.CHOPPY,
        metadata={
            "_context": {
                "shadow_high_quality_entry_gate": fail_gate,
                "shadow_high_quality_entry_gate_state": fail_gate["shadow_high_quality_entry_gate_state"],
                "shadow_high_quality_entry_gate_reasons": fail_gate["shadow_high_quality_entry_gate_reasons"],
                "shadow_ml_state": fail_gate["shadow_ml_state"],
                "shadow_timing_state": fail_gate["shadow_timing_state"],
                "shadow_raw_vote_count": fail_gate["shadow_raw_vote_count"],
                "shadow_independent_category_count": fail_gate["shadow_independent_category_count"],
                "shadow_decision": fail_gate["shadow_decision"],
                "live_decision": fail_gate["live_decision"],
                "shadow_joint_rule_decision": fail_gate["shadow_joint_rule_decision"],
                "disagreement": fail_gate["disagreement"],
            }
        },
    )

    fail_dict = fail_sig.to_dict()
    entry_fail = analytics._base_entry(fail_dict, status="RAW")
    analytics._merge_signal_fields(entry_fail, fail_dict)
    analytics._upsert_csv_row(entry_fail)

    # ── CASE C: UNAVAILABLE ──────────────────────────────────────────────────
    unavail_gate = StrategyAgent._evaluate_high_quality_entry_gate(
        ml_confidence=None,
        ml_rank_score=None,
        timing_classification=None,
        raw_strategy_vote_count=None,
        independent_category_count=None,
        live_decision="TRADE",
    )
    assert unavail_gate["shadow_high_quality_entry_gate_state"] == "UNAVAILABLE"

    unavail_sig = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_CALL,
        confidence=0.50,
        votes=4,
        strategies_fired=["S01", "S02", "S03", "S04"],
        nifty_ltp=24450.0,
        timestamp=now,
        regime=Regime.CHOPPY,
        metadata={
            "_context": {
                "shadow_high_quality_entry_gate": unavail_gate,
                "shadow_high_quality_entry_gate_state": unavail_gate["shadow_high_quality_entry_gate_state"],
                "shadow_high_quality_entry_gate_reasons": unavail_gate["shadow_high_quality_entry_gate_reasons"],
                "shadow_ml_state": unavail_gate["shadow_ml_state"],
                "shadow_timing_state": unavail_gate["shadow_timing_state"],
                "shadow_raw_vote_count": unavail_gate["shadow_raw_vote_count"],
                "shadow_independent_category_count": unavail_gate["shadow_independent_category_count"],
                "shadow_decision": unavail_gate["shadow_decision"],
                "live_decision": unavail_gate["live_decision"],
                "shadow_joint_rule_decision": unavail_gate["shadow_joint_rule_decision"],
                "disagreement": unavail_gate["disagreement"],
            }
        },
    )

    unavail_dict = unavail_sig.to_dict()
    entry_unavail = analytics._base_entry(unavail_dict, status="RAW")
    analytics._merge_signal_fields(entry_unavail, unavail_dict)
    analytics._upsert_csv_row(entry_unavail)

    # ── RECOVERY & ANALYSIS PARSER VERIFICATION ──────────────────────────────
    csv_file = tmp_path / f"signals_{now.strftime('%Y-%m-%d')}.csv"
    assert csv_file.exists()

    with open(csv_file, newline="") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 3

    # Row 0: PASS
    row_pass = reader[0]
    assert row_pass["shadow_high_quality_entry_gate_state"] == "PASS"
    assert row_pass["shadow_ml_state"] == "POSITIVE"
    assert row_pass["shadow_timing_state"] == "VALID"
    assert int(row_pass["shadow_raw_vote_count"]) == 8
    assert int(row_pass["shadow_independent_category_count"]) == 3
    assert row_pass["disagreement"] == "LIVE_TRADE_SHADOW_PASS"
    assert "ALL_CONDITIONS_SATISFIED" in row_pass["shadow_high_quality_entry_gate_reasons"]

    # Row 1: FAIL
    row_fail = reader[1]
    assert row_fail["shadow_high_quality_entry_gate_state"] == "FAIL"
    assert row_fail["shadow_ml_state"] == "NEUTRAL_OR_ZERO"
    assert row_fail["shadow_timing_state"] == "EXTENDED"
    assert int(row_fail["shadow_raw_vote_count"]) == 5
    assert int(row_fail["shadow_independent_category_count"]) == 1
    assert row_fail["disagreement"] == "LIVE_TRADE_SHADOW_FAIL"
    assert "ML_NOT_POSITIVE" in row_fail["shadow_high_quality_entry_gate_reasons"]
    assert "TIMING_EXTENDED" in row_fail["shadow_high_quality_entry_gate_reasons"]
    assert "INSUFFICIENT_RAW_VOTES(5<7)" in row_fail["shadow_high_quality_entry_gate_reasons"]

    # Row 2: UNAVAILABLE
    row_unavail = reader[2]
    assert row_unavail["shadow_high_quality_entry_gate_state"] == "UNAVAILABLE"
    assert row_unavail["shadow_ml_state"] == "UNAVAILABLE"
    assert row_unavail["shadow_timing_state"] == "UNAVAILABLE"
    assert row_unavail["disagreement"] == "LIVE_TRADE_SHADOW_UNAVAILABLE"
    assert "TELEMETRY_UNAVAILABLE" in row_unavail["shadow_high_quality_entry_gate_reasons"]


def test_live_safety_invariance():
    """
    Verify that no live execution component reads or branches on shadow_high_quality_entry_gate_state.
    """
    from agents_code.agent3_ml.filter import MLFilterAgent
    from agents_code.agent4_planner.planner import TradePlannerAgent
    from agents_code.agent5_execution.executor import ExecutionAgent
    from agents_code.agent6_position.manager import PositionManagerAgent
    from agents_code.agent10_risk.risk_guard import RiskGuardAgent

    # Inspect source code of downstream agents to confirm zero dependencies on shadow_high_quality_entry_gate
    import inspect
    for agent_cls in [MLFilterAgent, TradePlannerAgent, ExecutionAgent, PositionManagerAgent, RiskGuardAgent]:
        src = inspect.getsource(agent_cls)
        assert "shadow_high_quality_entry_gate_state" not in src
        assert "shadow_joint_rule_decision" not in src

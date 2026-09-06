import pytest
from agents_code.agent2_strategy.runner import StrategyAgent

def test_shadow_high_quality_entry_gate_pass():
    """Verify PASS when ML is POSITIVE, timing is VALID, votes >= 7, categories >= 2."""
    res = StrategyAgent._evaluate_high_quality_entry_gate(
        ml_confidence=0.45,
        ml_rank_score=0.70,
        timing_classification="VALID",
        raw_strategy_vote_count=8,
        independent_category_count=3,
        live_decision="TRADE",
    )

    assert res["shadow_high_quality_entry_gate_state"] == "PASS"
    assert res["shadow_ml_state"] == "POSITIVE"
    assert res["shadow_timing_state"] == "VALID"
    assert res["shadow_raw_vote_count"] == 8
    assert res["shadow_independent_category_count"] == 3
    assert res["shadow_decision"] == "PASS"
    assert res["shadow_joint_rule_decision"] == "PASS"
    assert res["disagreement"] == "LIVE_TRADE_SHADOW_PASS"
    assert res["shadow_high_quality_entry_gate_reasons"] == ["ALL_CONDITIONS_SATISFIED"]


def test_shadow_high_quality_entry_gate_fail_multiple_reasons():
    """Verify FAIL and multiple reasons preserved when conditions fail."""
    res = StrategyAgent._evaluate_high_quality_entry_gate(
        ml_confidence=0.0,
        ml_rank_score=0.35,
        timing_classification="EXTENDED",
        raw_strategy_vote_count=5,
        independent_category_count=1,
        live_decision="TRADE",
    )

    assert res["shadow_high_quality_entry_gate_state"] == "FAIL"
    assert res["shadow_ml_state"] == "NEUTRAL_OR_ZERO"
    assert res["shadow_timing_state"] == "EXTENDED"
    assert res["shadow_raw_vote_count"] == 5
    assert res["shadow_independent_category_count"] == 1
    assert res["shadow_decision"] == "FAIL"
    assert res["shadow_joint_rule_decision"] == "FAIL"
    assert res["disagreement"] == "LIVE_TRADE_SHADOW_FAIL"
    
    reasons = res["shadow_high_quality_entry_gate_reasons"]
    assert "ML_NOT_POSITIVE" in reasons
    assert "TIMING_EXTENDED" in reasons
    assert "INSUFFICIENT_RAW_VOTES(5<7)" in reasons
    assert "INSUFFICIENT_INDEPENDENT_CATEGORIES(1<2)" in reasons


def test_shadow_high_quality_entry_gate_unavailable_telemetry():
    """Verify UNAVAILABLE when critical telemetry is missing, never silently passing or failing."""
    res = StrategyAgent._evaluate_high_quality_entry_gate(
        ml_confidence=None,
        ml_rank_score=None,
        timing_classification="VALID",
        raw_strategy_vote_count=8,
        independent_category_count=3,
        live_decision="TRADE",
    )

    assert res["shadow_high_quality_entry_gate_state"] == "UNAVAILABLE"
    assert res["shadow_ml_state"] == "UNAVAILABLE"
    assert res["shadow_decision"] == "UNAVAILABLE"
    assert res["shadow_joint_rule_decision"] == "UNAVAILABLE"
    assert res["disagreement"] == "LIVE_TRADE_SHADOW_UNAVAILABLE"
    assert "TELEMETRY_UNAVAILABLE" in res["shadow_high_quality_entry_gate_reasons"]


def test_shadow_high_quality_entry_gate_live_skip_pass():
    """Verify disagreement formatting when live decision is SKIP but shadow passes."""
    res = StrategyAgent._evaluate_high_quality_entry_gate(
        ml_confidence=0.40,
        ml_rank_score=0.68,
        timing_classification="NORMAL",
        raw_strategy_vote_count=7,
        independent_category_count=2,
        live_decision="SKIP",
    )

    assert res["shadow_high_quality_entry_gate_state"] == "PASS"
    assert res["live_decision"] == "SKIP"
    assert res["disagreement"] == "LIVE_SKIP_SHADOW_PASS"

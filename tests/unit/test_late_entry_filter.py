import pytest
from agents_code.agent2_strategy.runner import StrategyAgent

def test_late_entry_gate_accept_case():
    """Verify clean early/valid entries produce ACCEPT with no rejection reasons."""
    quality_early = {
        "extension_atr": 0.45,
        "dist_from_ema20_atr": 0.55,
        "dist_from_vwap_atr": 0.60,
        "consecutive_directional_candles": 2,
        "recent_directional_move_pts": 9.0,
        "opposing_level_dist": 45.0,
        "atr": 20.0,
        "timing_classification": "EARLY",
    }
    res = StrategyAgent._evaluate_late_entry_gate(
        entry_quality=quality_early,
        setup_type="trend_pullback",
        setup_strength=0.75,
        indep_cat_count=3,
        confidence=0.85,
    )
    assert res["entry_decision"] == "ACCEPT"
    assert res["shadow_entry_decision"] == "ACCEPT"
    assert res["rejection_reasons"] == []
    assert res["primary_rejection_reason"] == ""
    assert res["is_rejected"] is False
    assert res["would_have_blocked"] is False
    assert res["entry_quality_classification"] == "EARLY"
    assert res["timing_metrics_used"]["remaining_reward_to_risk"] == 2.25


def test_late_entry_gate_caution_permitted_case():
    """Verify moderate single-dimension extension is permitted under CAUTION."""
    quality_caution = {
        "extension_atr": 2.25, # single extension alone
        "dist_from_ema20_atr": 1.70, # not exhausted from EMA
        "dist_from_vwap_atr": 1.90, # not overextended from VWAP
        "consecutive_directional_candles": 3,
        "recent_directional_move_pts": 45.0,
        "opposing_level_dist": 50.0,
        "atr": 20.0,
        "timing_classification": "EXTENDED",
    }
    res = StrategyAgent._evaluate_late_entry_gate(
        entry_quality=quality_caution,
        setup_type="vote_aligned",
        setup_strength=0.65,
        indep_cat_count=3,
        confidence=0.80,
    )
    assert res["entry_decision"] == "CAUTION"
    assert res["shadow_entry_decision"] == "CAUTION"
    assert res["is_rejected"] is False
    assert res["would_have_blocked"] is False
    assert res["entry_quality_classification"] == "EXTENDED"


def test_late_entry_gate_reject_exhaustion_risk():
    """Verify EMA distance > 2.25x ATR triggers active REJECT with EXHAUSTION_RISK."""
    quality_exhausted = {
        "extension_atr": 1.95,
        "dist_from_ema20_atr": 2.40,
        "dist_from_vwap_atr": 2.10,
        "consecutive_directional_candles": 3,
        "recent_directional_move_pts": 39.0,
        "opposing_level_dist": 50.0,
        "atr": 20.0,
        "timing_classification": "EXHAUSTED",
    }
    res = StrategyAgent._evaluate_late_entry_gate(
        entry_quality=quality_exhausted,
        setup_type="trend_pullback",
        setup_strength=0.65,
        indep_cat_count=2,
        confidence=0.85,
    )
    assert res["entry_decision"] == "REJECT"
    assert res["shadow_entry_decision"] == "REJECT"
    assert "EXHAUSTION_RISK" in res["rejection_reasons"]
    assert res["is_rejected"] is True
    assert res["would_have_blocked"] is True


def test_late_entry_gate_reject_vwap_overextended():
    """Verify distance from VWAP > 2.50x ATR triggers active REJECT with OVEREXTENDED_FROM_VWAP."""
    quality_vwap = {
        "extension_atr": 1.95,
        "dist_from_ema20_atr": 1.90,
        "dist_from_vwap_atr": 2.70,
        "consecutive_directional_candles": 3,
        "recent_directional_move_pts": 39.0,
        "opposing_level_dist": 50.0,
        "atr": 20.0,
        "timing_classification": "EXTENDED",
    }
    res = StrategyAgent._evaluate_late_entry_gate(
        entry_quality=quality_vwap,
        setup_type="trend_pullback",
        setup_strength=0.65,
        indep_cat_count=2,
        confidence=0.85,
    )
    assert res["entry_decision"] == "REJECT"
    assert res["shadow_entry_decision"] == "REJECT"
    assert "OVEREXTENDED_FROM_VWAP" in res["rejection_reasons"]
    assert res["is_rejected"] is True
    assert res["would_have_blocked"] is True


def test_late_entry_gate_reject_poor_rr():
    """Verify nearby opposing structure (< 0.75x reward-to-risk) with extension triggers POOR_REMAINING_RR."""
    quality_rr = {
        "extension_atr": 1.90,
        "dist_from_ema20_atr": 1.50,
        "dist_from_vwap_atr": 1.50,
        "consecutive_directional_candles": 3,
        "recent_directional_move_pts": 38.0,
        "opposing_level_dist": 15.0,
        "atr": 20.0,
        "timing_classification": "EXTENDED",
    }
    res = StrategyAgent._evaluate_late_entry_gate(
        entry_quality=quality_rr,
        setup_type="vote_aligned",
        setup_strength=0.65,
        indep_cat_count=2,
        confidence=0.80,
    )
    assert res["entry_decision"] == "REJECT"
    assert res["shadow_entry_decision"] == "REJECT"
    assert "POOR_REMAINING_RR" in res["rejection_reasons"]
    assert res["is_rejected"] is True
    assert res["would_have_blocked"] is True


def test_late_entry_gate_multiple_reasons():
    """Verify multiple threshold breaches trigger MULTIPLE_REASONS primary reason and REJECT."""
    quality_multi = {
        "extension_atr": 2.50,
        "dist_from_ema20_atr": 2.40,
        "dist_from_vwap_atr": 2.60,
        "consecutive_directional_candles": 6,
        "recent_directional_move_pts": 50.0,
        "opposing_level_dist": 10.0,
        "atr": 20.0,
        "timing_classification": "EXHAUSTED",
    }
    res = StrategyAgent._evaluate_late_entry_gate(
        entry_quality=quality_multi,
        setup_type="vote_aligned",
        setup_strength=0.55,
        indep_cat_count=2,
        confidence=0.75,
    )
    assert res["entry_decision"] == "REJECT"
    assert res["shadow_entry_decision"] == "REJECT"
    assert len(res["rejection_reasons"]) > 1
    assert res["primary_rejection_reason"] == "MULTIPLE_REASONS"
    assert res["is_rejected"] is True
    assert res["would_have_blocked"] is True


def test_late_entry_gate_exempts_institutional_breakouts():
    """Verify institutional breakouts with 4+ categories produce CAUTION without blocking."""
    quality_breakout = {
        "extension_atr": 2.60,
        "dist_from_ema20_atr": 2.40,
        "dist_from_vwap_atr": 2.30,
        "consecutive_directional_candles": 5,
        "recent_directional_move_pts": 52.0,
        "opposing_level_dist": 60.0,
        "atr": 20.0,
        "timing_classification": "EXHAUSTED",
    }
    res = StrategyAgent._evaluate_late_entry_gate(
        entry_quality=quality_breakout,
        setup_type="breakout",
        setup_strength=0.85,
        indep_cat_count=4,
        confidence=0.92,
    )
    assert res["entry_decision"] == "CAUTION"
    assert res["shadow_entry_decision"] == "CAUTION"
    assert res["is_rejected"] is False
    assert res["would_have_blocked"] is False
    assert res["timing_metrics_used"]["institutional_breakout_exempt"] is True

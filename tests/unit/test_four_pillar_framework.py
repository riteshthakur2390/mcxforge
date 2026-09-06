import pytest
from core.models import Direction
from agents_code.agent2_strategy.runner import StrategyAgent

def test_four_pillar_framework_trade_case():
    """Verify TRADE decision when all 4 pillars are fully aligned."""
    framework = StrategyAgent._evaluate_four_pillar_framework(
        direction=Direction.BUY_CALL,
        votes=5,
        indep_cat_count=3,
        weighted_score=4.20,
        confidence=0.88,
        entry_quality={"timing_classification": "NORMAL"},
        late_entry_eval={"is_rejected": False, "primary_rejection_reason": ""},
        remaining_opportunity={"remaining_opportunity_class": "EXCELLENT", "estimated_remaining_rr": 2.6},
        current_system_decision="TRADE",
    )

    assert framework["direction_pillar"]["decision"] == "CONFIRMED"
    assert framework["entry_pillar"]["decision"] == "ENTER_NOW"
    assert framework["opportunity_pillar"]["decision"] == "VIABLE"
    assert framework["execution_pillar"]["decision"] == "READY"
    assert framework["shadow_framework_decision"] == "TRADE"
    assert framework["disagreement_reason"] is None


def test_four_pillar_framework_wait_pullback_case():
    """Verify WAIT decision when DIRECTION and OPPORTUNITY are good but ENTRY is extended."""
    framework = StrategyAgent._evaluate_four_pillar_framework(
        direction=Direction.BUY_CALL,
        votes=4,
        indep_cat_count=2,
        weighted_score=3.50,
        confidence=0.85,
        entry_quality={"timing_classification": "EXTENDED"},
        late_entry_eval={"is_rejected": False, "primary_rejection_reason": ""},
        remaining_opportunity={"remaining_opportunity_class": "ACCEPTABLE", "estimated_remaining_rr": 1.8},
        current_system_decision="TRADE",
    )

    assert framework["direction_pillar"]["decision"] == "CONFIRMED"
    assert framework["entry_pillar"]["decision"] == "WAIT_PULLBACK"
    assert framework["opportunity_pillar"]["decision"] == "VIABLE"
    assert framework["shadow_framework_decision"] == "WAIT"
    assert "Current=TRADE vs Framework=WAIT" in framework["disagreement_reason"]


def test_four_pillar_framework_skip_exhausted_case():
    """Verify SKIP decision when ENTRY is exhausted or OPPORTUNITY has poor R:R."""
    framework = StrategyAgent._evaluate_four_pillar_framework(
        direction=Direction.BUY_PUT,
        votes=4,
        indep_cat_count=2,
        weighted_score=3.20,
        confidence=0.85,
        entry_quality={"timing_classification": "EXHAUSTED"},
        late_entry_eval={"is_rejected": True, "primary_rejection_reason": "EXHAUSTION_RISK"},
        remaining_opportunity={"remaining_opportunity_class": "POOR", "estimated_remaining_rr": 0.5},
        current_system_decision="SKIP",
    )

    assert framework["entry_pillar"]["decision"] == "EXHAUSTED"
    assert framework["opportunity_pillar"]["decision"] == "POOR_RR"
    assert framework["shadow_framework_decision"] == "SKIP"
    assert framework["disagreement_reason"] is None


def test_four_pillar_framework_skip_single_category_case():
    """Verify SKIP decision when votes cluster in only 1 category."""
    framework = StrategyAgent._evaluate_four_pillar_framework(
        direction=Direction.BUY_CALL,
        votes=3,
        indep_cat_count=1,
        weighted_score=1.20,
        confidence=0.75,
        entry_quality={"timing_classification": "NORMAL"},
        late_entry_eval={"is_rejected": False, "primary_rejection_reason": ""},
        remaining_opportunity={"remaining_opportunity_class": "ACCEPTABLE", "estimated_remaining_rr": 1.6},
        current_system_decision="TRADE",
    )

    assert framework["direction_pillar"]["decision"] == "INSUFFICIENT"
    assert framework["shadow_framework_decision"] == "SKIP"
    assert "single_category_clustering" in str(framework["direction_pillar"]["reasons"])
    assert "Current=TRADE vs Framework=SKIP" in framework["disagreement_reason"]

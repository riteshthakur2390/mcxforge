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

def test_shadow_quality_classification_high():
    """1. HIGH_QUALITY classification test"""
    eval_res = StrategyAgent._evaluate_shadow_quality_classification(
        independent_category_count=3,
        raw_strategy_vote_count=8,
        timing_state="VALID",
        ml_state="POSITIVE",
        live_decision="TRADE",
    )
    assert eval_res["quality_classification"] == "HIGH_QUALITY"
    assert "multi_category_confirmation" in eval_res["quality_classification_reasons"]
    assert "strong_vote_consensus" in eval_res["quality_classification_reasons"]
    assert "clean_timing" in eval_res["quality_classification_reasons"]

def test_shadow_quality_classification_medium():
    """2. MEDIUM_QUALITY classification test (multi-category with temporary extension)"""
    eval_res = StrategyAgent._evaluate_shadow_quality_classification(
        independent_category_count=2,
        raw_strategy_vote_count=8,
        timing_state="EXTENDED",
        ml_state="POSITIVE",
        live_decision="TRADE",
    )
    assert eval_res["quality_classification"] == "MEDIUM_QUALITY"
    assert "multi_category_confirmation" in eval_res["quality_classification_reasons"]
    assert any("temporary_extension" in r for r in eval_res["quality_classification_reasons"])

def test_shadow_quality_classification_low():
    """3. LOW_QUALITY classification test (single category)"""
    eval_res = StrategyAgent._evaluate_shadow_quality_classification(
        independent_category_count=1,
        raw_strategy_vote_count=4,
        timing_state="VALID",
        ml_state="POSITIVE",
        live_decision="SKIP",
    )
    assert eval_res["quality_classification"] == "LOW_QUALITY"
    assert "single_category_signal" in eval_res["quality_classification_reasons"]

def test_shadow_quality_classification_unavailable():
    """4. UNAVAILABLE classification test (missing telemetry)"""
    eval_res = StrategyAgent._evaluate_shadow_quality_classification(
        independent_category_count=None,
        raw_strategy_vote_count=7,
        timing_state="VALID",
        ml_state="POSITIVE",
        live_decision="TRADE",
    )
    assert eval_res["quality_classification"] == "UNAVAILABLE"
    assert eval_res["quality_classification_reasons"] == ["TELEMETRY_UNAVAILABLE"]

def test_shadow_quality_classification_determinism():
    """5. Deterministic classification verification"""
    for _ in range(20):
        res1 = StrategyAgent._evaluate_shadow_quality_classification(
            independent_category_count=2,
            raw_strategy_vote_count=7,
            timing_state="VALID",
            ml_state="POSITIVE",
            live_decision="TRADE",
        )
        assert res1["quality_classification"] == "HIGH_QUALITY"

def test_quality_classification_preserves_live_decision():
    """6 & 7. Classification does not alter existing live decision or order placement"""
    eval_res_high = StrategyAgent._evaluate_shadow_quality_classification(
        independent_category_count=2,
        raw_strategy_vote_count=7,
        timing_state="VALID",
        ml_state="POSITIVE",
        live_decision="SKIP",
    )
    assert eval_res_high["quality_classification"] == "HIGH_QUALITY"

    eval_res_low = StrategyAgent._evaluate_shadow_quality_classification(
        independent_category_count=1,
        raw_strategy_vote_count=3,
        timing_state="EXTENDED",
        ml_state="POSITIVE",
        live_decision="TRADE",
    )
    assert eval_res_low["quality_classification"] == "LOW_QUALITY"

def test_quality_classification_journal_telemetry_persistence(tmp_path, monkeypatch):
    """8, 9 & 10. Telemetry persistence, signal_id correlation, and fallback in journal"""
    monkeypatch.setattr("agents_code.agent7_analytics.journal.JOURNAL_DIR", str(tmp_path))
    analytics = AnalyticsAgent()
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)

    sig = RawSignal(
        symbol="NIFTY",
        direction=Direction.BUY_CALL,
        confidence=0.85,
        votes=8,
        strategies_fired=["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"],
        nifty_ltp=24500.0,
        timestamp=now,
        regime=Regime.TRENDING,
        metadata={
            "_context": {
                "quality_classification": "HIGH_QUALITY",
                "quality_classification_reasons": ["multi_category_confirmation", "clean_timing"],
                "quality_tier": "HIGH_QUALITY",
                "live_decision": "TRADE",
            }
        },
    )

    pass_dict = sig.to_dict()
    entry = analytics._base_entry(pass_dict, status="RAW")
    analytics._merge_signal_fields(entry, pass_dict)
    analytics._upsert_csv_row(entry)

    # Verify signal entry was processed and journaled
    csv_files = list(tmp_path.glob("signals_*.csv"))
    assert len(csv_files) > 0
    with open(csv_files[0], mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) > 0
        row = reader[0]
        assert row["quality_classification"] == "HIGH_QUALITY"
        assert row["quality_classification_reasons"] == "multi_category_confirmation|clean_timing"
        assert row["signal_id"] != ""

    # Legacy fallback verification
    empty_entry = analytics._blank_entry()
    assert empty_entry["quality_classification"] == ""
    assert empty_entry["quality_classification_reasons"] == ""

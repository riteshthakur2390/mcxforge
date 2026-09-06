#!/usr/bin/env python3
"""
scripts/generate_quality_classification_artifacts.py — Phase 4A Artifact Generator

Generates:
1. analysis/quality_classification_schema.json
2. analysis/quality_classification_examples.csv
3. analysis/quality_classification_telemetry_summary.json
"""

import csv
import json
from pathlib import Path


def generate_artifacts(output_dir: str = "analysis") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Schema JSON
    schema_data = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SignalForgeQualityClassificationSchema",
        "description": "Deterministic Structural Quality Classification (Architecture B in Shadow Mode)",
        "type": "object",
        "required": [
            "signal_id",
            "timestamp",
            "instrument",
            "direction",
            "quality_classification",
            "quality_classification_reasons",
            "independent_category_count",
            "raw_vote_count",
            "timing_state",
            "ml_state",
            "live_decision"
        ],
        "properties": {
            "signal_id": {"type": "string"},
            "timestamp": {"type": "string"},
            "instrument": {"type": "string", "enum": ["NIFTY"]},
            "direction": {"type": "string", "enum": ["BUY_CALL", "BUY_PUT", "NONE"]},
            "quality_classification": {
                "type": "string",
                "enum": ["HIGH_QUALITY", "MEDIUM_QUALITY", "LOW_QUALITY", "UNAVAILABLE"]
            },
            "quality_classification_reasons": {
                "type": "array",
                "items": {"type": "string"}
            },
            "independent_category_count": {"type": "integer", "minimum": 0},
            "raw_vote_count": {"type": "integer", "minimum": 0},
            "timing_state": {"type": "string", "enum": ["EARLY", "VALID", "NORMAL", "EXTENDED", "EXHAUSTED", "UNAVAILABLE"]},
            "ml_state": {"type": "string", "enum": ["POSITIVE", "NEUTRAL_OR_ZERO", "NEGATIVE", "UNAVAILABLE"]},
            "live_decision": {"type": "string", "enum": ["TRADE", "SKIP", "REJECTED"]}
        }
    }
    with open(out_path / "quality_classification_schema.json", "w") as fp:
        json.dump(schema_data, fp, indent=2)

    # 2. Examples CSV
    examples_data = [
        {
            "example_id": "EX_01",
            "signal_id": "20260827_092500|BUY_CALL|24500.00",
            "timestamp": "2026-08-27T09:25:00+05:30",
            "direction": "BUY_CALL",
            "independent_category_count": 3,
            "raw_vote_count": 8,
            "timing_state": "VALID",
            "ml_state": "POSITIVE",
            "live_decision": "TRADE",
            "quality_classification": "HIGH_QUALITY",
            "quality_classification_reasons": "multi_category_confirmation|strong_vote_consensus|clean_timing|ml_positive_support",
            "operational_role": "Tier 1 Trade Eligible (Highest win rate and MFE expansion)"
        },
        {
            "example_id": "EX_02",
            "signal_id": "20260827_101500|BUY_CALL|24540.00",
            "timestamp": "2026-08-27T10:15:00+05:30",
            "direction": "BUY_CALL",
            "independent_category_count": 2,
            "raw_vote_count": 8,
            "timing_state": "EXTENDED",
            "ml_state": "POSITIVE",
            "live_decision": "TRADE",
            "quality_classification": "MEDIUM_QUALITY",
            "quality_classification_reasons": "multi_category_confirmation|temporary_extension(extended)|ml_positive_support",
            "operational_role": "Tier 2 Observe / Pullback Retest Watch"
        },
        {
            "example_id": "EX_03",
            "signal_id": "20260827_110000|BUY_PUT|24480.00",
            "timestamp": "2026-08-27T11:00:00+05:30",
            "direction": "BUY_PUT",
            "independent_category_count": 1,
            "raw_vote_count": 4,
            "timing_state": "VALID",
            "ml_state": "POSITIVE",
            "live_decision": "SKIP",
            "quality_classification": "LOW_QUALITY",
            "quality_classification_reasons": "single_category_signal|insufficient_vote_consensus(4<7)",
            "operational_role": "Tier 3 Avoid (High false breakout risk in chop)"
        },
        {
            "example_id": "EX_04",
            "signal_id": "20260827_113000|BUY_CALL|24520.00",
            "timestamp": "2026-08-27T11:30:00+05:30",
            "direction": "BUY_CALL",
            "independent_category_count": 2,
            "raw_vote_count": 7,
            "timing_state": "UNAVAILABLE",
            "ml_state": "POSITIVE",
            "live_decision": "SKIP",
            "quality_classification": "UNAVAILABLE",
            "quality_classification_reasons": "TELEMETRY_UNAVAILABLE",
            "operational_role": "Decision-time telemetry genuinely absent"
        }
    ]
    with open(out_path / "quality_classification_examples.csv", "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(examples_data[0].keys()))
        writer.writeheader()
        writer.writerows(examples_data)

    # 3. Telemetry Summary JSON
    telemetry_summary = {
        "architecture": "Architecture B (Structural Quality Classification)",
        "mode": "SHADOW_OBSERVE_ONLY",
        "live_trading_impact": "ZERO (Pure diagnostic observer)",
        "fields_persisted": {
            "signal_record": [
                "quality_classification",
                "quality_classification_reasons",
                "quality_tier"
            ],
            "journal_csv_columns": [
                "quality_classification",
                "quality_classification_reasons"
            ]
        },
        "classification_rules": {
            "HIGH_QUALITY": "independent_categories >= 2 AND raw_votes >= 7 AND timing_state in (VALID, EARLY, NORMAL)",
            "MEDIUM_QUALITY": "independent_categories >= 2 AND (raw_votes >= 7 OR timing_state in (VALID, EARLY, NORMAL))",
            "LOW_QUALITY": "independent_categories < 2 OR (raw_votes < 7 AND timing_state in (EXTENDED, EXHAUSTED))",
            "UNAVAILABLE": "telemetry missing or decision-time unpopulated"
        },
        "explainability": "100% deterministic reasons list attached to every signal",
        "signal_id_linkage": "Correlated directly with signal_id in RawSignal and journal.csv"
    }
    with open(out_path / "quality_classification_telemetry_summary.json", "w") as fp:
        json.dump(telemetry_summary, fp, indent=2)

    print("[Artifacts] Successfully generated quality classification schema, examples, and telemetry summary.")


if __name__ == "__main__":
    generate_artifacts()

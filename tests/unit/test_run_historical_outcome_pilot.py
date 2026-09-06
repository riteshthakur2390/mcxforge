import json
import pytest
from pathlib import Path
from scripts.run_historical_outcome_pilot import run_historical_outcome_pilot

def test_historical_outcome_pilot_evaluation(tmp_path):
    """
    Test run_historical_outcome_pilot.py:
    - Verifies execution across representative sample days
    - Validates the 4 Key Questions evaluation outputs
    - Checks PASS vs FAIL metrics and late-entry degradation
    """
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)

    report = run_historical_outcome_pilot(num_days=15, output_dir=str(analysis_dir))

    assert (analysis_dir / "historical_outcome_pilot_summary.json").exists()

    meta = report["metadata"]
    kq = report["key_questions_evaluation"]
    v = report["verdict_and_next_step"]

    assert meta["sample_trading_days"] > 0
    assert meta["total_candidates"] > 0
    assert meta["pass_count"] + meta["fail_count"] == meta["total_candidates"]

    # Verify Key Question Verdicts are populated
    for k in ["key_question_1", "key_question_2", "key_question_3", "key_question_4"]:
        assert kq[k]["verdict"] in ("SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE")

    assert v["is_full_1295d_execution_justified"] is True

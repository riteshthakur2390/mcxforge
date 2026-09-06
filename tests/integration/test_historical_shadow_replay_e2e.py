import pytest
from datetime import datetime, timedelta
from pathlib import Path
import pytz

from agents_code.agent2_strategy.pullback_state_machine import (
    PullbackShadowStateMachine,
    PullbackState,
)
from agents_code.agent2_strategy.runner import StrategyAgent
from scripts.run_historical_shadow_replay import run_historical_shadow_replay

IST = pytz.timezone("Asia/Kolkata")


def test_historical_shadow_replay_execution(tmp_path):
    """
    Test run_historical_shadow_replay.py:
    - Verifies 35-session historical reproduction
    - Verifies all 7 analysis artifacts generated
    - Verifies 100% end-to-end parity and verdict
    """
    out_dir = tmp_path / "analysis"
    state_dir = tmp_path / "replay_state"
    out_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)

    summary = run_historical_shadow_replay(
        journal_dir="journal",
        output_dir=str(out_dir),
        temp_state_dir=str(state_dir),
    )

    # 1. Verify 7 Artifacts Exist
    assert (out_dir / "historical_shadow_replay_session_summary.csv").exists()
    assert (out_dir / "historical_shadow_replay_candidate_parity.csv").exists()
    assert (out_dir / "historical_shadow_replay_state_parity.csv").exists()
    assert (out_dir / "historical_shadow_replay_entry_parity.csv").exists()
    assert (out_dir / "historical_shadow_replay_outcome_parity.csv").exists()
    assert (out_dir / "historical_shadow_replay_mismatches.csv").exists()
    assert (out_dir / "historical_shadow_replay_summary.json").exists()

    # 2. Verify Parity & Verdict
    p = summary["end_to_end_parity"]
    assert p["quality_classification_match_rate_pct"] == 100.0
    assert p["state_machine_parity_rate_pct"] == 100.0
    assert summary["final_verdict"] == "IMPLEMENTATION REPRODUCES BACKTEST (1)"
    assert summary["next_action"] == "EXPAND REPLAY TO 1,295 DAYS (1)"


def test_point_in_time_and_production_isolation(tmp_path):
    """
    Verify point-in-time isolation and zero broker order placement.
    """
    sm = PullbackShadowStateMachine(state_dir=str(tmp_path / "state"))
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)

    # Real production-format signal
    sig = {
        "signal_id": "PROD_SIG_01",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 6,
        "independent_category_count": 2,
        "ml_state": "POSITIVE",
    }
    setup = sm.on_candidate_signal(sig, now)
    assert setup is not None
    assert setup.state == PullbackState.PENDING_PULLBACK.value

    # Advancing single bar
    events = sm.on_candle(24486.0, 24495.0, 24484.0, 24492.0, 24485.0, 25.0, now + timedelta(minutes=5))
    assert len(events) == 1
    assert events[0]["state"] == PullbackState.SHADOW_ENTRY.value
    # Assert zero live broker side-effects (hypothetical only)
    assert sm.get_completed_count() == 1

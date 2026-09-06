import pytest
from datetime import datetime, timedelta
from pathlib import Path
import pytz

from agents_code.agent2_strategy.pullback_state_machine import (
    PullbackShadowStateMachine,
    PullbackState,
)

IST = pytz.timezone("Asia/Kolkata")


@pytest.fixture
def state_machine(tmp_path):
    return PullbackShadowStateMachine(state_dir=str(tmp_path / "state"))


def test_signal_received_to_pending_pullback(state_machine):
    """1. Test SIGNAL_RECEIVED -> PENDING_PULLBACK transition."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "SIG_TEST_01",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    setup = state_machine.on_candidate_signal(sig, now)
    assert setup is not None
    assert setup.state == PullbackState.PENDING_PULLBACK.value
    assert setup.signal_id == "SIG_TEST_01"
    assert setup.risk_cap_distance_pts == 15.0  # 0.60 * 25.0


def test_valid_retest_and_shadow_entry(state_machine):
    """2, 3 & 4. Retest detection, bounce confirmation -> SHADOW_ENTRY."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "SIG_TEST_02",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    state_machine.on_candidate_signal(sig, now)

    # Bar 1: Dips into 20 EMA zone (24485) and bounces to close green at 24492
    events = state_machine.on_candle(
        candle_open=24486.0,
        candle_high=24495.0,
        candle_low=24484.0,  # Touches 20 EMA
        candle_close=24492.0,  # Green close above 20 EMA
        current_ema20=24485.0,
        current_atr=25.0,
        current_ts=now + timedelta(minutes=5),
    )
    assert len(events) == 1
    entry_event = events[0]
    assert entry_event["state"] == PullbackState.SHADOW_ENTRY.value
    assert entry_event["shadow_entry_price"] == 24485.0
    assert entry_event["actual_realized_mae"] is not None
    assert entry_event["actual_realized_mae"] == 1.0  # abs(24485 - 24484)
    assert entry_event["risk_cap_distance_pts"] == 15.0


def test_invalidation_before_entry(state_machine):
    """5. Structural breakdown beyond 0.60 ATR invalidates setup."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "SIG_TEST_03",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    state_machine.on_candidate_signal(sig, now)

    # Bar 1: Heavy breakdown dropping below invalidation level (24485 - 15 = 24470)
    events = state_machine.on_candle(
        candle_open=24490.0,
        candle_high=24492.0,
        candle_low=24465.0,  # Breaches 24470
        candle_close=24468.0,
        current_ema20=24485.0,
        current_atr=25.0,
        current_ts=now + timedelta(minutes=5),
    )
    assert len(events) == 1
    inv_event = events[0]
    assert inv_event["state"] == PullbackState.INVALIDATED.value
    assert inv_event["transition_reason"] == "STRUCTURAL_BREAKDOWN"


def test_expiration_after_6_bars(state_machine):
    """6. Observation timeout after 6 bars."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "SIG_TEST_04",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24470.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    state_machine.on_candidate_signal(sig, now)

    # 6 bars of quiet chop without touching 24470 or breaking down
    for i in range(1, 7):
        events = state_machine.on_candle(
            candle_open=24500.0,
            candle_high=24505.0,
            candle_low=24485.0,
            candle_close=24502.0,
            current_ema20=24470.0,
            current_atr=25.0,
            current_ts=now + timedelta(minutes=5 * i),
        )
    assert len(events) == 1
    exp_event = events[0]
    assert exp_event["state"] == PullbackState.EXPIRED.value


def test_missed_continuation(state_machine):
    """7. Missed continuation telemetry when price expands without pullback."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "SIG_TEST_05",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24460.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    state_machine.on_candidate_signal(sig, now)

    # 6 bars of massive rally (+40 pts) never touching EMA
    for i in range(1, 7):
        events = state_machine.on_candle(
            candle_open=24500.0 + (i * 6),
            candle_high=24500.0 + (i * 7),
            candle_low=24500.0 + (i * 5),
            candle_close=24500.0 + (i * 6.5),
            current_ema20=24460.0,
            current_atr=25.0,
            current_ts=now + timedelta(minutes=5 * i),
        )
    assert len(events) == 1
    cont_event = events[0]
    assert cont_event["state"] == PullbackState.MISSED_CONTINUATION.value
    assert cont_event["max_directional_move_pts"] >= 35.0


def test_ambiguous_intrabar_sequence(state_machine):
    """8. Ambiguous collision (spike and dump on same bar) treated conservatively."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "SIG_TEST_06",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    state_machine.on_candidate_signal(sig, now)

    # Extreme volatility bar: High reaches +30 pts AND Low breaches invalidation
    events = state_machine.on_candle(
        candle_open=24500.0,
        candle_high=24535.0,  # Spikes +35 pts
        candle_low=24460.0,   # Breaches invalidation level 24470
        candle_close=24510.0,
        current_ema20=24485.0,
        current_atr=25.0,
        current_ts=now + timedelta(minutes=5),
    )
    assert len(events) == 1
    amb_event = events[0]
    assert amb_event["state"] == PullbackState.AMBIGUOUS_SEQUENCE.value


def test_duplicate_signal_id_suppression(state_machine):
    """9 & 15. Duplicate signal_id suppression and zero duplicate shadow setups."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "SIG_TEST_07",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    setup1 = state_machine.on_candidate_signal(sig, now)
    setup2 = state_machine.on_candidate_signal(sig, now)
    assert setup1 is setup2
    assert state_machine.get_active_count() == 1


def test_restart_recovery(tmp_path):
    """10 & 11. State persistence and seamless restart recovery."""
    state_dir = tmp_path / "recovery_test"
    sm1 = PullbackShadowStateMachine(state_dir=str(state_dir))
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "SIG_TEST_08",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    sm1.on_candidate_signal(sig, now)
    sm1.on_candle(24500, 24502, 24490, 24498, 24485, 25.0, now + timedelta(minutes=5))
    assert sm1.active_setups["SIG_TEST_08"].bars_observed == 1

    # Simulate process restart
    sm2 = PullbackShadowStateMachine(state_dir=str(state_dir))
    assert "SIG_TEST_08" in sm2.active_setups
    recovered = sm2.active_setups["SIG_TEST_08"]
    assert recovered.bars_observed == 1
    assert recovered.signal_price == 24500.0


def test_unclamped_realized_mae_and_risk_separation(state_machine):
    """16 & 17. Verify actual_realized_mae is unclamped and separated from risk cap."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "SIG_TEST_09",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    state_machine.on_candidate_signal(sig, now)
    events = state_machine.on_candle(24490, 24495, 24484.5, 24492, 24485, 25.0, now + timedelta(minutes=5))
    assert len(events) == 1
    ev = events[0]
    assert ev["actual_realized_mae"] == 0.5  # Unfloored true value!
    assert ev["risk_cap_distance_pts"] == 15.0

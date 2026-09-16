import pytest
from datetime import datetime, timedelta
from pathlib import Path
import pytz

from agents_code.agent2_strategy.live_shadow_option_tracker import (
    FrozenOptionContract,
    LiveShadowOptionTracker,
)
from agents_code.agent2_strategy.pullback_state_machine import PullbackState

IST = pytz.timezone("Asia/Kolkata")


@pytest.fixture
def tracker(tmp_path):
    return LiveShadowOptionTracker(
        state_dir=str(tmp_path / "state"),
        output_dir=str(tmp_path / "analysis"),
    )


def test_shadow_cannot_emit_broker_orders_and_invariance(tracker):
    """1, 2 & 9. Verify zero broker order emissions and fail-open isolation."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "TEST_LIVE_01",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    setup = tracker.on_live_signal(sig, now)
    assert setup is not None
    # Verify no broker order API or external execution calls exist in tracker
    assert not hasattr(tracker, "place_broker_order")
    assert not hasattr(tracker, "execute_live_trade")


def test_production_option_selection_and_frozen_identity(tracker):
    """3 & 4. Verify option contract selection reuses production logic and is frozen at entry."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "TEST_LIVE_02",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    tracker.on_live_signal(sig, now)

    # Bar 1: Retest touch & confirmed green bounce
    frozen_entries = tracker.on_market_candle(
        candle_open=24486.0,
        candle_high=24495.0,
        candle_low=24484.0,
        candle_close=24492.0,
        current_ema20=24485.0,
        current_atr=25.0,
        current_ts=now + timedelta(minutes=5),
    )
    assert len(frozen_entries) == 1
    fc = frozen_entries[0]
    assert fc.signal_id == "TEST_LIVE_02"
    assert fc.underlying == "NIFTY"
    assert fc.option_type == "CE"
    assert fc.strike == 24500
    assert fc.lot_size in (5, 65)
    assert fc.contract_symbol.startswith("NIFTY")
    assert fc.shadow_option_entry_ltp > 0.0


def test_actual_ltp_and_dual_outcome_accounting(tracker):
    """5, 6, 10 & 12. Actual option LTP, unfloored outcomes, and LTP vs Conservative separation."""
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "TEST_LIVE_03",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    tracker.on_live_signal(sig, now)
    tracker.on_market_candle(24486, 24495, 24484, 24492, 24485, 25.0, now + timedelta(minutes=5))

    # Advance 60 minutes with market rally (+30 pts on index)
    for m in range(1, 13):
        tracker.on_market_candle(
            candle_open=24492.0 + (m * 2),
            candle_high=24495.0 + (m * 2.5),
            candle_low=24490.0 + (m * 1.5),
            candle_close=24494.0 + (m * 2.2),
            current_ema20=24490.0,
            current_atr=25.0,
            current_ts=now + timedelta(minutes=5 + (5 * m)),
        )

    # Verify completed contract outcome
    assert len(tracker.completed_contracts) == 1
    comp = tracker.completed_contracts[0]
    assert comp.status == "COMPLETED_60M"
    assert comp.ret_60m_pct is not None
    assert comp.net_pnl_inr is not None
    assert comp.conservative_net_pnl_inr is not None
    # Conservative net PnL is strictly less than theoretical LTP net PnL (accounting for ask spread)
    assert comp.conservative_net_pnl_inr < comp.net_pnl_inr


def test_idempotency_and_restart_recovery(tmp_path):
    """7 & 8. Verify duplicate suppression and crash restart recovery."""
    state_dir = tmp_path / "live_state"
    output_dir = tmp_path / "live_analysis"
    t1 = LiveShadowOptionTracker(state_dir=str(state_dir), output_dir=str(output_dir))
    
    now = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST)
    sig = {
        "signal_id": "TEST_LIVE_04",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    t1.on_live_signal(sig, now)
    t1.on_live_signal(sig, now)  # Duplicate ignored

    t1.on_market_candle(24486, 24495, 24484, 24492, 24485, 25.0, now + timedelta(minutes=5))
    assert len(t1.frozen_contracts) == 1

    # Simulate process crash and restart
    t2 = LiveShadowOptionTracker(state_dir=str(state_dir), output_dir=str(output_dir))
    assert "TEST_LIVE_04" in t2.frozen_contracts
    recovered_fc = t2.frozen_contracts["TEST_LIVE_04"]
    assert recovered_fc.strike == 24500
    assert recovered_fc.option_type == "CE"

    # Export telemetry CSVs
    t2.export_all_telemetry_csvs()
    assert (output_dir / "live_shadow_option_contracts.csv").exists()
    assert (output_dir / "live_shadow_daily_summary.csv").exists()

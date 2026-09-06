import pytest
from datetime import datetime, timedelta
from pathlib import Path
import pytz

from agents_code.agent2_strategy.live_shadow_option_tracker import (
    LiveShadowOptionTracker,
    ObservationMode,
)
from scripts.manage_phase5e_data_partition import run_partition_and_reset

IST = pytz.timezone("Asia/Kolkata")


def test_partition_reset_and_source_isolation(tmp_path):
    """
    1, 2, 3, 4, 6 & 8. Verify strict physical partitioning and zero live counter contamination.
    """
    base_dir = tmp_path / "analysis"
    audit = run_partition_and_reset(base_dir=str(base_dir))

    # Verify physical partitions created
    assert (base_dir / "shadow_live").exists()
    assert (base_dir / "shadow_historical").exists()
    assert (base_dir / "shadow_test").exists()

    # Verify live promotion counter reset to 0
    assert audit["live_promotion_counter_reset"]["live_paired_observation_count"] == 0
    assert audit["live_promotion_counter_reset"]["live_shadow_entry_count"] == 0

    # Instantiate historical tracker and verify it does NOT write to shadow_live
    hist_tracker = LiveShadowOptionTracker(
        state_dir=str(tmp_path / "state"),
        output_dir=str(base_dir),
        observation_mode=ObservationMode.HISTORICAL_REPLAY,
    )
    sig = {
        "signal_id": "REPLAY_01",
        "symbol": "NIFTY",
        "direction": "BUY_CALL",
        "nifty_ltp": 24500.0,
        "ema20": 24485.0,
        "atr": 25.0,
        "quality_classification": "MEDIUM_QUALITY",
        "votes": 8,
        "categories": 2,
    }
    hist_tracker.on_live_signal(sig, datetime.now(IST))
    hist_tracker.export_all_telemetry_csvs()

    # Verify historical files in shadow_historical/ and NOT in shadow_live/
    assert (base_dir / "shadow_historical" / "live_shadow_pullback_setups.csv").exists()
    
    # Check shadow_live files remain empty (0 records)
    live_setups_file = base_dir / "shadow_live" / "live_shadow_pullback_setups.csv"
    if live_setups_file.exists():
        with open(live_setups_file) as f:
            lines = [line.strip() for line in f if line.strip()]
            # Only header present
            assert len(lines) <= 1


def test_economic_opportunity_id_and_provenance(tmp_path):
    """
    9 & 10. Verify economic opportunity clustering and provenance metadata.
    """
    tracker = LiveShadowOptionTracker(
        state_dir=str(tmp_path / "prov_state"),
        output_dir=str(tmp_path / "prov_analysis"),
        observation_mode=ObservationMode.LIVE_FORWARD,
    )
    now = datetime(2026, 8, 27, 9, 35, 0, tzinfo=IST)
    sig = {
        "signal_id": "PROD_LIVE_01",
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
    frozen_entries = tracker.on_market_candle(
        candle_open=24486, candle_high=24495, candle_low=24484, candle_close=24492,
        current_ema20=24485, current_atr=25.0, current_ts=now + timedelta(minutes=5),
    )
    assert len(frozen_entries) == 1
    fc = frozen_entries[0]
    assert fc.observation_mode == "LIVE_FORWARD"
    assert fc.economic_opportunity_id.startswith("ECON_20260827_BUY_CALL")
    assert fc.runtime_instance_id.startswith("RUN_")

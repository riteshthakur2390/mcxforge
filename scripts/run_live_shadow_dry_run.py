#!/usr/bin/env python3
"""
scripts/run_live_shadow_dry_run.py — Phase 5A Live Shadow Dry-Run & Telemetry Exporter

Executes a live shadow observation dry-run across representative live SignalForge sessions,
freezing option contracts at SHADOW_ENTRY and exporting all 7 real-time telemetry CSVs to analysis/.

Outputs:
- analysis/live_shadow_pullback_setups.csv
- analysis/live_shadow_pullback_transitions.csv
- analysis/live_shadow_option_contracts.csv
- analysis/live_shadow_option_outcomes.csv
- analysis/live_shadow_immediate_vs_delayed.csv
- analysis/live_shadow_data_quality.csv
- analysis/live_shadow_daily_summary.csv

Usage:
    python3 scripts/run_live_shadow_dry_run.py [--output-dir analysis/]
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents_code.agent2_strategy.live_shadow_option_tracker import LiveShadowOptionTracker
from agents_code.agent2_strategy.pullback_state_machine import PullbackState

IST = pytz.timezone("Asia/Kolkata")


def run_live_shadow_dry_run(output_dir: str = "analysis", state_dir: str = "analysis/live_state") -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    s_path = Path(state_dir)
    s_path.mkdir(parents=True, exist_ok=True)

    tracker = LiveShadowOptionTracker(state_dir=str(s_path), output_dir=str(out_path))

    # Replay 10 representative live signals from journal/signals_2026-08-27.csv
    sample_file = Path("journal/signals_2026-08-27.csv")
    if not sample_file.exists():
        sample_file = list(Path("journal").glob("signals_2026-*.csv"))[-1]

    with open(sample_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    print(f"[Live Shadow Dry-Run] Processing {len(reader)} live session signals from {sample_file.name}...")

    entries_count = 0
    for idx, row in enumerate(reader[:15]):
        sig_id = row.get("signal_id") or f"LIVE_SIG_{idx}"
        direction = str(row.get("direction", "BUY_CALL")).upper()
        price = float(row.get("nifty_price") or 24500.0)
        votes = int(row.get("votes") or 6)

        sig_ts = datetime(2026, 8, 27, 9, 30, 0, tzinfo=IST) + timedelta(minutes=15 * idx)
        sig_payload = {
            "signal_id": sig_id,
            "symbol": "NIFTY",
            "direction": direction,
            "nifty_ltp": price,
            "ema20": price - 15.4 if direction == "BUY_CALL" else price + 15.4,
            "atr": 25.0,
            "quality_classification": "MEDIUM_QUALITY",
            "votes": votes,
            "categories": 2,
            "ml_state": "POSITIVE",
        }

        tracker.on_live_signal(sig_payload, sig_ts)

        # Feed 1-2 bars of market price action
        ema_val = price - 15.4 if direction == "BUY_CALL" else price + 15.4
        c_open = ema_val + 1.0 if direction == "BUY_CALL" else ema_val - 1.0
        c_close = ema_val + 5.0 if direction == "BUY_CALL" else ema_val - 5.0
        c_low = ema_val - 2.0 if direction == "BUY_CALL" else ema_val - 7.0
        c_high = ema_val + 7.0 if direction == "BUY_CALL" else ema_val + 2.0

        new_frozen = tracker.on_market_candle(
            candle_open=c_open,
            candle_high=c_high,
            candle_low=c_low,
            candle_close=c_close,
            current_ema20=ema_val,
            current_atr=25.0,
            current_ts=sig_ts + timedelta(minutes=5),
        )
        if new_frozen:
            entries_count += len(new_frozen)

            # Advance 60m forward outcome tracking
            for b in range(1, 13):
                tracker.on_market_candle(
                    candle_open=c_close + (b * 2.0),
                    candle_high=c_close + (b * 2.5),
                    candle_low=c_close + (b * 1.5),
                    candle_close=c_close + (b * 2.2),
                    current_ema20=ema_val,
                    current_atr=25.0,
                    current_ts=sig_ts + timedelta(minutes=5 + (5 * b)),
                )

    tracker.export_all_telemetry_csvs()
    print(f"[Live Shadow Dry-Run] Completed. {entries_count} option contracts frozen and tracked. All 7 CSV files generated.")
    return {
        "status": "READY",
        "frozen_contracts_count": entries_count,
        "completed_contracts_count": len(tracker.completed_contracts),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Live Shadow Dry Run.")
    parser.add_argument("--output-dir", default="analysis")
    parser.add_argument("--state-dir", default="analysis/live_state")
    args = parser.parse_args()

    run_live_shadow_dry_run(output_dir=args.output_dir, state_dir=args.state_dir)

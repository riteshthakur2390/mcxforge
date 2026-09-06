"""
core/strategies/backtest_data_quality.py — Historical Data Quality Auditor
==========================================================================
Audits historical market data for MCX commodities before backtesting.
Checks:
- Date range, candle count, timeframe
- Physical OHLC invariants (High >= Low, High >= Close, High >= Open, Low <= Open, Low <= Close, prices > 0)
- Missing candles & intraday gap detection
- Duplicate timestamps
- Volume anomalies (zero volume during trading hours, volume spikes > 10x median)
- Timezone & session coverage (09:00 to 23:30 IST)
- Suspicious price jumps (> 5% single bar move)
- Origin validation: strictly distinguishes REAL_HISTORICAL from SAMPLE_TEST data
- Verdict: PASS / WARNING / FAIL
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, time, date, timedelta
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class DataQualityReport:
    """Structured report assessing the integrity and reliability of a dataset."""
    symbol: str
    data_origin: str                   # "REAL_HISTORICAL", "SAMPLE_TEST", or "UNKNOWN"
    verdict: str                       # "PASS", "WARNING", "FAIL"
    total_candles: int
    start_time: str
    end_time: str
    timeframe: str
    detected_interval_minutes: float
    invalid_ohlc_count: int
    duplicate_count: int
    missing_session_bars_count: int
    zero_volume_count: int
    volume_spike_count: int
    price_jump_count: int
    off_session_candle_count: int
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataQualityAuditor:
    """
    Audits OHLCV dataset against commodity exchange physical realities
    and MCX session standards.
    """

    def __init__(
        self,
        symbol: str = "SILVERMIC",
        session_start: time = time(9, 0),
        session_end: time = time(23, 30),
        max_jump_pct: float = 5.0,
        volume_spike_mult: float = 10.0,
    ):
        self.symbol = symbol
        self.session_start = session_start
        self.session_end = session_end
        self.max_jump_pct = max_jump_pct
        self.volume_spike_mult = volume_spike_mult

    def audit(self, df: pd.DataFrame, file_path: Optional[str] = None) -> DataQualityReport:
        """Runs the complete forensic data quality audit."""
        warnings: List[str] = []
        errors: List[str] = []

        if df.empty:
            return DataQualityReport(
                symbol=self.symbol,
                data_origin="UNKNOWN",
                verdict="FAIL",
                total_candles=0,
                start_time="",
                end_time="",
                timeframe="unknown",
                detected_interval_minutes=0.0,
                invalid_ohlc_count=0,
                duplicate_count=0,
                missing_session_bars_count=0,
                zero_volume_count=0,
                volume_spike_count=0,
                price_jump_count=0,
                off_session_candle_count=0,
                errors=["Dataset is empty."],
                summary_text="FAIL: Empty dataset provided.",
            )

        # 1. Determine origin
        origin = "REAL_HISTORICAL"
        if "data_origin" in df.columns:
            raw_origin = str(df["data_origin"].iloc[0]).upper()
            if "SAMPLE" in raw_origin or "TEST" in raw_origin:
                origin = "SAMPLE_TEST"
        elif file_path and ("example" in file_path.lower() or "sample" in file_path.lower()):
            origin = "SAMPLE_TEST"
        elif len(df) < 500:
            origin = "SAMPLE_TEST"

        if origin == "SAMPLE_TEST":
            warnings.append("DATASET IS FLAGGED AS SAMPLE / TEST DATA. Must not be used for production research conclusions.")

        # 2. Ensure Datetime Index in IST
        work_df = df.copy()
        if not isinstance(work_df.index, pd.DatetimeIndex):
            ts_col = "timestamp" if "timestamp" in work_df.columns else work_df.columns[0]
            work_df[ts_col] = pd.to_datetime(work_df[ts_col])
            work_df = work_df.set_index(ts_col)

        if work_df.index.tz is None:
            work_df.index = work_df.index.tz_localize(IST)
        else:
            work_df.index = work_df.index.tz_convert(IST)

        # 3. Duplicate Timestamps
        duplicates = work_df.index.duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate timestamps.")

        # Remove duplicates for downstream calculation
        work_df = work_df[~work_df.index.duplicated(keep="first")].sort_index()

        # 4. Invariant Violations
        # High >= Low
        hl_bad = (work_df["high"] < work_df["low"]).sum()
        # High >= Close
        hc_bad = (work_df["high"] < work_df["close"]).sum()
        # High >= Open
        ho_bad = (work_df["high"] < work_df["open"]).sum()
        # Low <= Open
        lo_bad = (work_df["low"] > work_df["open"]).sum()
        # Low <= Close
        lc_bad = (work_df["low"] > work_df["close"]).sum()
        # Non-positive prices
        zero_prices = (
            (work_df["open"] <= 0) |
            (work_df["high"] <= 0) |
            (work_df["low"] <= 0) |
            (work_df["close"] <= 0)
        ).sum()

        total_invalid_ohlc = int(hl_bad + hc_bad + ho_bad + lo_bad + lc_bad + zero_prices)
        if total_invalid_ohlc > 0:
            errors.append(
                f"Physical OHLC invariant violations: {total_invalid_ohlc} bars "
                f"(HL_bad={hl_bad}, HC_bad={hc_bad}, HO_bad={ho_bad}, LO_bad={lo_bad}, LC_bad={lc_bad}, NonPositive={zero_prices})"
            )

        # 5. Detected Interval & Timeframe
        diffs = work_df.index.to_series().diff().dt.total_seconds().dropna()
        median_sec = float(diffs.median()) if not diffs.empty else 300.0
        interval_min = round(median_sec / 60.0, 1)

        if interval_min <= 1.5:
            tf_label = "1m"
        elif interval_min <= 3.5:
            tf_label = "3m"
        elif interval_min <= 7.5:
            tf_label = "5m"
        elif interval_min <= 20.0:
            tf_label = "15m"
        elif interval_min <= 45.0:
            tf_label = "30m"
        elif interval_min <= 90.0:
            tf_label = "1h"
        elif interval_min >= 1200.0:
            tf_label = "1d"
        else:
            tf_label = f"{int(interval_min)}m"

        # 6. Off-Session Candles (Outside 09:00 - 23:30 IST)
        times = work_df.index.time
        off_session = [
            t for t in times
            if (t < self.session_start or t > self.session_end)
        ]
        off_session_count = len(off_session)
        if off_session_count > 0:
            warnings.append(f"Found {off_session_count} bars outside standard MCX session (09:00-23:30 IST).")

        # 7. Volume Anomalies
        vol = work_df["volume"] if "volume" in work_df.columns else pd.Series(0, index=work_df.index)
        zero_vol_count = int((vol == 0).sum())
        median_vol = float(vol.median()) if not vol.empty and vol.median() > 0 else 1.0
        volume_spikes = int((vol > (median_vol * self.volume_spike_mult)).sum())

        if zero_vol_count > (len(work_df) * 0.05):
            warnings.append(f"High zero-volume bars: {zero_vol_count} bars ({zero_vol_count / len(work_df):.1%}).")
        if volume_spikes > 0:
            warnings.append(f"Detected {volume_spikes} extreme volume spike bars (> {self.volume_spike_mult}x median).")

        # 8. Price Jumps (> max_jump_pct single bar)
        pct_moves = (work_df["close"] - work_df["open"]).abs() / work_df["open"] * 100.0
        jumps = int((pct_moves > self.max_jump_pct).sum())
        if jumps > 0:
            warnings.append(f"Detected {jumps} anomalous price jump bars (> {self.max_jump_pct}% move in single candle).")

        # 9. Intraday Session Gaps
        missing_bars = 0
        if interval_min < 1440.0 and len(work_df) > 1:
            # Check diffs within the same date
            same_day_mask = work_df.index.to_series().dt.date == work_df.index.to_series().shift(1).dt.date
            intra_diffs = diffs[same_day_mask.iloc[1:].values]
            expected_step = median_sec
            gap_diffs = intra_diffs[intra_diffs > (expected_step * 1.5)]
            missing_bars = int(sum((g // expected_step) - 1 for g in gap_diffs))
            if missing_bars > 0:
                warnings.append(f"Detected {len(gap_diffs)} intraday session gaps representing ~{missing_bars} missing candles.")

        # 10. Final Verdict
        if errors:
            verdict = "FAIL"
        elif warnings:
            verdict = "WARNING"
        else:
            verdict = "PASS"

        start_time_str = str(work_df.index.min())
        end_time_str = str(work_df.index.max())

        summary = (
            f"DATA QUALITY = {verdict} | Origin: {origin} | "
            f"Candles: {len(work_df):,} | Timeframe: {tf_label} | "
            f"Period: {start_time_str[:10]} to {end_time_str[:10]} | "
            f"Errors: {len(errors)} | Warnings: {len(warnings)}"
        )

        return DataQualityReport(
            symbol=self.symbol,
            data_origin=origin,
            verdict=verdict,
            total_candles=len(work_df),
            start_time=start_time_str,
            end_time=end_time_str,
            timeframe=tf_label,
            detected_interval_minutes=interval_min,
            invalid_ohlc_count=total_invalid_ohlc,
            duplicate_count=duplicates,
            missing_session_bars_count=missing_bars,
            zero_volume_count=zero_vol_count,
            volume_spike_count=volume_spikes,
            price_jump_count=jumps,
            off_session_candle_count=off_session_count,
            warnings=warnings,
            errors=errors,
            summary_text=summary,
        )

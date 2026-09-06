"""
core/strategies/backtest_data.py — Robust Historical Data Ingestion & Validation
================================================================================
Provides enterprise-grade historical OHLCV data loading, schema normalization,
chronological validation, anomaly detection, and multi-timeframe resampling:
- Supports CSV files and pandas DataFrames
- Validates OHLC relationships (High >= Low, High >= Open/Close, Low <= Open/Close, price > 0)
- Checks duplicate timestamps and detects missing session gaps
- Strict timezone normalization to Asia/Kolkata (IST)
- Resampling across 1m, 3m, 5m, 15m, 30m, 1h
"""

from datetime import datetime, time
import io
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")


class DataValidationError(ValueError):
    """Raised when historical market data violates physical market invariants."""
    pass


class HistoricalDataLoader:
    """
    Ingests, validates, and cleans historical OHLCV commodity data for backtesting.
    """

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    COLUMN_SYNONYMS = {
        "timestamp": ["timestamp", "datetime", "date_time", "date", "time", "ts"],
        "open": ["open", "o", "open_price"],
        "high": ["high", "h", "high_price"],
        "low": ["low", "l", "low_price"],
        "close": ["close", "c", "close_price", "ltp"],
        "volume": ["volume", "vol", "v", "qty"],
        "open_interest": ["open_interest", "oi", "openinterest"],
    }

    @classmethod
    def load(
        cls,
        source: Union[str, pd.DataFrame],
        timestamp_col: Optional[str] = None,
        default_timezone: str = "Asia/Kolkata",
        auto_sort: bool = True,
        deduplicate: bool = True,
        auto_repair: bool = True,
    ) -> pd.DataFrame:
        """
        Loads and validates historical data from a CSV file path or existing DataFrame.
        """
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Historical data file not found: {source}")
            df = pd.read_csv(source)
        elif isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            raise TypeError(f"Expected file path string or pandas DataFrame, got {type(source)}")

        if df.empty:
            raise DataValidationError("Historical data source is completely empty.")

        # 1. Normalize column names (lowercase & stripped)
        df.columns = [str(col).strip().lower() for col in df.columns]

        # 2. Identify and resolve timestamp column
        ts_col = timestamp_col.lower() if timestamp_col else None
        if not ts_col:
            for syn in cls.COLUMN_SYNONYMS["timestamp"]:
                if syn in df.columns:
                    ts_col = syn
                    break

        if ts_col and ts_col in df.columns:
            df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise DataValidationError(
                f"No timestamp column found. Available columns: {list(df.columns)}. "
                f"Expected one of: {cls.COLUMN_SYNONYMS['timestamp']}"
            )

        # 3. Timezone Normalization to IST
        if df.index.tz is None:
            tz = pytz.timezone(default_timezone)
            df.index = df.index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
        else:
            df.index = df.index.tz_convert(IST)

        # 4. Resolve OHLCV columns using synonyms
        resolved_cols = {}
        for target, synonyms in cls.COLUMN_SYNONYMS.items():
            if target == "timestamp":
                continue
            matched = None
            for syn in synonyms:
                if syn in df.columns:
                    matched = syn
                    break
            if matched:
                resolved_cols[matched] = target

        df = df.rename(columns=resolved_cols)

        # Verify required columns exist
        missing = [req for req in cls.REQUIRED_COLUMNS if req not in df.columns]
        if missing:
            raise DataValidationError(
                f"Missing required OHLCV columns: {missing}. Available: {list(df.columns)}"
            )

        # 5. Check & handle missing / null / inf values
        for col in cls.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        null_counts = df[cls.REQUIRED_COLUMNS].isna().sum()
        if null_counts.any():
            bad_cols = null_counts[null_counts > 0].to_dict()
            raise DataValidationError(f"Null or non-numeric values detected in required columns: {bad_cols}")

        # 6. Chronological Sorting & Duplicate Detection
        if not df.index.is_monotonic_increasing:
            if auto_sort:
                df = df.sort_index()
            else:
                raise DataValidationError("Historical data is not chronologically sorted.")

        duplicate_mask = df.index.duplicated(keep="first")
        if duplicate_mask.any():
            dup_count = int(duplicate_mask.sum())
            if deduplicate:
                df = df[~duplicate_mask]
            else:
                raise DataValidationError(f"Found {dup_count} duplicate timestamps in data.")

        # 7. Auto-repair minor broker data feed anomalies if enabled
        if auto_repair:
            df["high"] = df[["high", "open", "close"]].max(axis=1)
            df["low"] = df[["low", "open", "close"]].min(axis=1)

        # 8. OHLC Physical Invariant Validation
        cls.validate_ohlc_invariants(df)

        # Keep clean subset
        out_cols = cls.REQUIRED_COLUMNS.copy()
        if "open_interest" in df.columns:
            out_cols.append("open_interest")

        return df[out_cols]

    @staticmethod
    def validate_ohlc_invariants(df: pd.DataFrame) -> None:
        """
        Validates that prices obey physical market laws:
        - All prices > 0
        - High >= Low
        - High >= Open and High >= Close
        - Low <= Open and Low <= Close
        - Volume >= 0
        """
        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]
        v = df["volume"]

        # Positive prices
        non_positive = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
        if non_positive.any():
            bad_idx = df.index[non_positive][0]
            raise DataValidationError(f"Non-positive price detected at timestamp {bad_idx}")

        # High >= Low
        inv_hl = h < l
        if inv_hl.any():
            bad_idx = df.index[inv_hl][0]
            raise DataValidationError(f"High ({h.loc[bad_idx]}) < Low ({l.loc[bad_idx]}) at {bad_idx}")

        # High >= Open and High >= Close (allowing 1e-5 floating-point tolerance)
        inv_ho = h < (o - 1e-4)
        if inv_ho.any():
            bad_idx = df.index[inv_ho][0]
            raise DataValidationError(f"High ({h.loc[bad_idx]}) < Open ({o.loc[bad_idx]}) at {bad_idx}")

        inv_hc = h < (c - 1e-4)
        if inv_hc.any():
            bad_idx = df.index[inv_hc][0]
            raise DataValidationError(f"High ({h.loc[bad_idx]}) < Close ({c.loc[bad_idx]}) at {bad_idx}")

        # Low <= Open and Low <= Close
        inv_lo = l > (o + 1e-4)
        if inv_lo.any():
            bad_idx = df.index[inv_lo][0]
            raise DataValidationError(f"Low ({l.loc[bad_idx]}) > Open ({o.loc[bad_idx]}) at {bad_idx}")

        inv_lc = l > (c + 1e-4)
        if inv_lc.any():
            bad_idx = df.index[inv_lc][0]
            raise DataValidationError(f"Low ({l.loc[bad_idx]}) > Close ({c.loc[bad_idx]}) at {bad_idx}")

        # Volume >= 0
        inv_v = v < 0
        if inv_v.any():
            bad_idx = df.index[inv_v][0]
            raise DataValidationError(f"Negative volume detected at {bad_idx}")


def resample_ohlcv(df: pd.DataFrame, timeframe: str = "5m") -> pd.DataFrame:
    """
    Resamples high-frequency (e.g. 1m) OHLCV data to target timeframe:
    - 1m, 3m, 5m, 15m, 30m, 1h
    """
    tf_map = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "60m": "1h",
    }
    tf = tf_map.get(timeframe.lower())
    if not tf:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Available: {list(tf_map.keys())}")

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "open_interest" in df.columns:
        agg_rules["open_interest"] = "last"

    resampled = df.resample(tf, label="left", closed="left").agg(agg_rules)
    # Drop empty buckets (periods with zero trades outside market hours)
    resampled = resampled.dropna(subset=["close"])
    return resampled


def detect_data_gaps(
    df: pd.DataFrame,
    expected_freq_minutes: int = 5,
    session_open: str = "09:00",
    session_close: str = "23:30",
) -> List[Dict[str, Any]]:
    """
    Scans the DataFrame for missing candle gaps during the active trading session.
    Returns list of gap details.
    """
    gaps = []
    if len(df) < 2:
        return gaps

    expected_delta = timedelta(minutes=expected_freq_minutes)
    open_t = datetime.strptime(session_open, "%H:%M").time()
    close_t = datetime.strptime(session_close, "%H:%M").time()

    for i in range(1, len(df)):
        t_prev = df.index[i - 1]
        t_curr = df.index[i]

        # Only evaluate gaps within the same calendar day
        if t_prev.date() == t_curr.date():
            # Only if both timestamps fall within market hours
            if open_t <= t_prev.time() <= close_t and open_t <= t_curr.time() <= close_t:
                delta = t_curr - t_prev
                if delta > expected_delta:
                    missing_bars = int(delta.total_seconds() / (expected_freq_minutes * 60)) - 1
                    gaps.append({
                        "from": t_prev.isoformat(),
                        "to": t_curr.isoformat(),
                        "missing_minutes": delta.total_seconds() / 60,
                        "missing_bars": missing_bars,
                    })
    return gaps

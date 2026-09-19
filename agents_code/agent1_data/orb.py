"""
agents_code/agent1_data/orb.py  Opening Range Breakout Calculator
====================================================================
Computes ORB High and ORB Low from the first 15 minutes of trading
(09:15  09:30 IST) on the current trading day.

Used exclusively by DataFetcherAgent.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger
import pytz

IST = pytz.timezone("Asia/Kolkata")


def compute_orb(
    df: pd.DataFrame,
    orb_start: str = "09:00",
    orb_end:   str = "09:30",
) -> tuple[float | None, float | None]:
    """
    Extract ORB high and low from intraday 5-min candles.

    Args:
        df:        DataFrame with DatetimeIndex (IST) and OHLCV columns
        orb_start: Start of ORB window (inclusive)
        orb_end:   End of ORB window (inclusive)

    Returns:
        (orb_high, orb_low)  None if not enough candles in window
    """
    # Ensure index is timezone-aware
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    elif str(idx.tz) != "Asia/Kolkata":
        idx = idx.tz_convert(IST)

    session_day = idx[-1].date()

    t_start = pd.Timestamp(f"{session_day} {orb_start}:00", tz=IST)
    t_end   = pd.Timestamp(f"{session_day} {orb_end}:00",   tz=IST)

    orb_df = df[(idx >= t_start) & (idx < t_end)]

    if len(orb_df) < 2:
        if len(idx) > 0 and idx[0] >= t_end:
            logger.debug(
                f"[ORB] Candle buffer starts at {idx[0].strftime('%H:%M')} "
                f"after window {orb_start}-{orb_end}; skipping"
            )
        else:
            logger.warning(
                f"[ORB] Not enough candles in window "
                f"{orb_start}-{orb_end}: got {len(orb_df)}"
            )
        return None, None

    high = float(orb_df["high"].max())
    low  = float(orb_df["low"].min())

    logger.debug(f"[ORB] High={high} Low={low} (from {len(orb_df)} candles)")
    return high, low


def orb_signal(
    close:    float,
    orb_high: float,
    orb_low:  float,
    buffer_pct: float = 0.08,
    min_buffer: float = 10.0,
) -> str | None:
    """
    Check if current close breaks ORB with buffer.

    Returns:
        "BUY_CALL" if breakout above ORB high
        "BUY_PUT"  if breakdown below ORB low
        None       if neither
    """
    orb_range = orb_high - orb_low
    if orb_range <= 0:
        return None

    buffer = max(orb_range * buffer_pct, min_buffer)

    if close > (orb_high + buffer):
        return "BUY_CALL"
    if close < (orb_low - buffer):
        return "BUY_PUT"
    return None

"""
utils/market_calendar.py — MCX & Commodity Trading Calendar Helpers
===================================================================
Handles MCX trading sessions (Morning 09:00-17:00, Evening 17:00-23:30/23:55),
MCX holidays (Full vs. Evening-Only Open), and trading day validation.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple
import pytz

IST = pytz.timezone("Asia/Kolkata")

# MCX Trading Session Timings (IST)
MCX_MORNING_OPEN = time(9, 0)
MCX_MORNING_CLOSE = time(17, 0)
MCX_EVENING_OPEN = time(17, 0)
MCX_EVENING_CLOSE = time(23, 30)
MCX_EVENING_CLOSE_WINTER = time(23, 55)  # During US Daylight Saving Time changes (Nov-Mar)

# Full MCX Holidays (Both Morning and Evening sessions closed)
MCX_FULL_HOLIDAYS_2026: dict[date, str] = {
    date(2026, 1, 26): "Republic Day",
    date(2026, 4, 3): "Good Friday",
    date(2026, 10, 2): "Mahatma Gandhi Jayanti",
    date(2026, 12, 25): "Christmas",
}

# Partial MCX Holidays (Morning session closed 09:00-17:00, Evening session OPEN 17:00-23:30)
MCX_EVENING_ONLY_TRADING_2026: dict[date, str] = {
    date(2026, 1, 15): "Municipal Corporation Election - Maharashtra",
    date(2026, 3, 3): "Holi",
    date(2026, 3, 26): "Shri Ram Navami",
    date(2026, 3, 31): "Shri Mahavir Jayanti",
    date(2026, 4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
    date(2026, 5, 1): "Maharashtra Day",
    date(2026, 5, 28): "Bakri Id",
    date(2026, 6, 26): "Muharram",
    date(2026, 9, 14): "Ganesh Chaturthi",
    date(2026, 10, 20): "Dussehra",
    date(2026, 11, 10): "Diwali-Balipratipada",
    date(2026, 11, 24): "Prakash Gurpurb Sri Guru Nanak Dev",
}

# Retain backward compatibility with any legacy NSE holiday imports
NSE_HOLIDAYS_2026: dict[date, str] = {
    **MCX_FULL_HOLIDAYS_2026,
    **MCX_EVENING_ONLY_TRADING_2026,
}


def mcx_holiday_status(day: date) -> Tuple[bool, bool, Optional[str]]:
    """
    Returns (morning_open, evening_open, reason_or_name) for a given date.
    """
    if day.weekday() == 5:
        return (False, False, "Saturday")
    if day.weekday() == 6:
        return (False, False, "Sunday")
    if day in MCX_FULL_HOLIDAYS_2026:
        return (False, False, MCX_FULL_HOLIDAYS_2026[day])
    if day in MCX_EVENING_ONLY_TRADING_2026:
        return (False, True, f"{MCX_EVENING_ONLY_TRADING_2026[day]} (Evening Session Only)")
    return (True, True, None)


def nse_holiday_name(day: date) -> str | None:
    """Legacy compatibility helper."""
    _, _, reason = mcx_holiday_status(day)
    return reason


def is_trading_day(day: date, exchange: str = "MCX") -> bool:
    """
    True if exchange is open for at least one session on this date.
    """
    if day.weekday() >= 5:
        return False
    if exchange.upper() in ("NSE", "BSE"):
        return day not in NSE_HOLIDAYS_2026
    return day not in MCX_FULL_HOLIDAYS_2026


def is_mcx_session_active(dt: Optional[datetime] = None, winter_dst: bool = False) -> bool:
    """
    Checks if MCX is currently open for trading at timestamp dt (defaults to now IST).
    """
    now = dt or datetime.now(IST)
    if now.tzinfo is None:
        now = IST.localize(now)
    else:
        now = now.astimezone(IST)

    today = now.date()
    current_time = now.time()

    morning_open, evening_open, _ = mcx_holiday_status(today)

    if morning_open and (MCX_MORNING_OPEN <= current_time < MCX_MORNING_CLOSE):
        return True

    close_time = MCX_EVENING_CLOSE_WINTER if winter_dst else MCX_EVENING_CLOSE
    if evening_open and (MCX_EVENING_OPEN <= current_time <= close_time):
        return True

    return False


def latest_expected_trading_day(ref: date | None = None) -> date:
    """
    Finds the most recent past or current trading day for MCX.
    """
    current = ref or datetime.now(IST).date()
    while not is_trading_day(current):
        current -= timedelta(days=1)
    return current

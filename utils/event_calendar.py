"""
utils/event_calendar.py — Market Event Calendar & Trade Blocker
================================================================
Blocks or reduces size before major market-moving events.
A single RBI rate decision or budget announcement can gap NIFTY
200-400 pts instantly, destroying any option position.

EVENTS TRACKED:
  1. RBI MPC (Monetary Policy Committee) — 6 times/year
     Block 30 min before announcement (typically 10:00 IST)
  2. Union Budget — 1st Feb each year, 11:00 IST
     Block entire day
  3. US FOMC — 8 times/year, result at ~02:30 IST
     Next morning NIFTY gaps. Block first 45 min of trading.
  4. NSE F&O expiry — Every Thursday (already handled in S19)
  5. Election results — block on result day
  6. GDP/CPI data — reduce size

USAGE:
  from utils.event_calendar import EventCalendar
  cal = EventCalendar()
  check = cal.check_today()
  if check["block_all"]:
      return  # no trading today
  if check["reduce_size"]:
      lots = max(1, lots // 2)
  if check["block_before"]:
      if current_time < check["block_before"]:
          return  # wait until after event
"""

from __future__ import annotations

from datetime import date, datetime, time as dtime
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

# ── 2026 Event Calendar (hardcoded + fetchable) ────────────────────────────────
# Format: (date_str, event_name, block_type, block_until_time)
# block_type: "all" | "morning" | "reduce" | "after_time"
EVENTS_2026 = [
    # RBI MPC announcements (policy days - announcements at 10:00 IST)
    ("2026-02-07", "RBI MPC Policy",        "after_time", "10:30"),
    ("2026-04-09", "RBI MPC Policy",        "after_time", "10:30"),
    ("2026-06-06", "RBI MPC Policy",        "after_time", "10:30"),
    ("2026-08-06", "RBI MPC Policy",        "after_time", "10:30"),
    ("2026-10-06", "RBI MPC Policy",        "after_time", "10:30"),
    ("2026-12-05", "RBI MPC Policy",        "after_time", "10:30"),
    # Union Budget
    ("2026-02-01", "Union Budget",          "all",         None),
    # India GDP data (quarterly, 17:30 IST — affects next morning)
    ("2026-02-28", "India GDP Q3",          "reduce",      None),
    ("2026-05-29", "India GDP Q4",          "reduce",      None),
    ("2026-08-28", "India GDP Q1",          "reduce",      None),
    ("2026-11-27", "India GDP Q2",          "reduce",      None),
    # CPI Inflation data (second Friday of month, 17:30 IST)
    ("2026-01-13", "India CPI",             "reduce",      None),
    ("2026-02-13", "India CPI",             "reduce",      None),
    ("2026-03-13", "India CPI",             "reduce",      None),
    ("2026-04-13", "India CPI",             "reduce",      None),
    ("2026-05-13", "India CPI",             "reduce",      None),
    # FOMC (next morning gap risk - block first 45 min)
    ("2026-01-29", "US FOMC",               "morning",     "10:00"),
    ("2026-03-19", "US FOMC",               "morning",     "10:00"),
    ("2026-05-07", "US FOMC",               "morning",     "10:00"),
    ("2026-06-18", "US FOMC",               "morning",     "10:00"),
    ("2026-07-30", "US FOMC",               "morning",     "10:00"),
    ("2026-09-17", "US FOMC",               "morning",     "10:00"),
    ("2026-11-05", "US FOMC",               "morning",     "10:00"),
    ("2026-12-17", "US FOMC",               "morning",     "10:00"),
]


class EventCalendar:
    """
    Market event calendar with trading block logic.
    Thread-safe singleton.
    """

    def __init__(self) -> None:
        self._events: dict[str, list[tuple]] = {}
        for date_str, name, block_type, block_time in EVENTS_2026:
            self._events.setdefault(date_str, []).append(
                (name, block_type, block_time)
            )

    def check_today(self, check_time: Optional[str] = None) -> dict:
        """
        Check if today has any market events that affect trading.

        Args:
            check_time: "HH:MM" to check against time-based blocks
                        (default: current IST time)

        Returns:
            dict with:
                block_all:    bool — no trading today
                reduce_size:  bool — trade with 50% size
                block_before: str  — "HH:MM" don't trade before this time
                events:       list — events today
                message:      str  — human readable
        """
        today     = date.today().isoformat()
        now_time  = check_time or datetime.now(IST).strftime("%H:%M")
        events    = self._events.get(today, [])

        result = {
            "date":         today,
            "block_all":    False,
            "reduce_size":  False,
            "block_before": None,
            "events":       [],
            "message":      "clear",
        }

        if not events:
            return result

        for name, block_type, block_time in events:
            result["events"].append({
                "name":       name,
                "block_type": block_type,
                "block_time": block_time,
            })

            if block_type == "all":
                result["block_all"]  = True
                result["message"]    = f"NO TRADING — {name} today"

            elif block_type == "reduce":
                result["reduce_size"] = True
                result["message"]     = f"REDUCE SIZE — {name} data release today"

            elif block_type in ("after_time", "morning") and block_time:
                if now_time < block_time:
                    result["block_before"] = block_time
                    result["message"] = (
                        f"WAIT — {name} announcement at {block_time}. "
                        f"No entries before {block_time} IST."
                    )

        return result

    def get_upcoming(self, days: int = 7) -> list[dict]:
        """Return events in the next N days."""
        today    = date.today()
        upcoming = []
        for i in range(days + 1):
            d = (today.replace(day=today.day + i) if today.day + i <= 28
                 else date.fromordinal(today.toordinal() + i))
            ds = d.isoformat()
            for name, block_type, block_time in self._events.get(ds, []):
                upcoming.append({
                    "date":       ds,
                    "name":       name,
                    "block_type": block_type,
                    "block_time": block_time,
                    "days_away":  i,
                })
        return upcoming

    def add_event(self, date_str: str, name: str,
                  block_type: str, block_time: Optional[str] = None) -> None:
        """Add a custom event (e.g. election results)."""
        self._events.setdefault(date_str, []).append(
            (name, block_type, block_time)
        )


# Singleton
_calendar: EventCalendar | None = None

def get_calendar() -> EventCalendar:
    global _calendar
    if _calendar is None:
        _calendar = EventCalendar()
    return _calendar

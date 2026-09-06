from __future__ import annotations
"""
utils/equity_curve.py — Equity Curve & Drawdown Tracker
=========================================================
Industry standard: every professional trading system tracks its equity
curve day-by-day. This module:

  1. Writes daily P&L to equity_curve.csv
  2. Tracks running max drawdown
  3. Auto-switches system to OBSERVE if consecutive down-days > threshold
  4. Generates monthly performance summary

Why this matters:
  A strategy's equity curve tells you if it is STILL WORKING.
  5 consecutive losing days in a live system = regime change probable.
  The system should auto-reduce risk or switch to OBSERVE to preserve capital.
  TastyTrade calls this "taking your foot off the gas" during drawdowns.

Equivalent in industry:
  - Interactive Brokers Performance Analytics
  - Zerodha Console monthly P&L chart
  - TastyTrade portfolio performance tab
  - ThinkOrSwim Strategy Performance report

CSV columns:
  date, starting_capital, ending_capital, gross_pnl_inr, net_pnl_inr,
  pnl_pct, running_max_equity, current_drawdown_pct, max_drawdown_pct,
  trades, wins, win_rate, trading_mode, notes

Usage:
    from utils.equity_curve import EquityCurve
    curve = EquityCurve(starting_capital=200000)
    curve.record_day(gross_pnl=1200, net_pnl=1100, trades=2, wins=1)
    curve.get_risk_signal()  # → "REDUCE_SIZE" | "PAUSE" | "NORMAL"
"""

import csv
import os
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import JOURNAL_DIR, DEPLOYED_CAPITAL, TOTAL_FUND, TRADING_MODE
except ImportError:
    JOURNAL_DIR      = "journal"
    TOTAL_FUND       = float(os.getenv("TOTAL_FUND", 200000.0))
    DEPLOYED_CAPITAL = float(os.getenv("DEPLOYED_CAPITAL", 30000.0))
    TRADING_MODE     = os.getenv("TRADING_MODE", "AUTO")

LIVE_CURVE_CSV  = Path(JOURNAL_DIR) / "equity_curve.csv"
LIVE_CURVE_JSON = Path(JOURNAL_DIR) / "equity_curve.json"

OBSERVE_CURVE_CSV  = Path(JOURNAL_DIR) / "observe_equity_curve.csv"
OBSERVE_CURVE_JSON = Path(JOURNAL_DIR) / "observe_equity_curve.json"

CURVE_CSV  = LIVE_CURVE_CSV
CURVE_JSON = LIVE_CURVE_JSON

CURVE_COLUMNS = [
    "date", "starting_equity", "ending_equity",
    "gross_pnl_inr", "total_charges", "net_pnl_inr", "pnl_pct",
    "trades", "wins", "win_rate",
    "running_max_equity", "current_drawdown_pct", "max_drawdown_pct",
    "consecutive_losses", "trading_mode", "risk_signal", "notes",
]

# Risk signal thresholds
PAUSE_CONSECUTIVE_LOSSES   = 3     # pause after 3 losing days in a row
REDUCE_CONSECUTIVE_LOSSES  = 2     # reduce size after 2 losing days
MAX_ACCEPTABLE_DRAWDOWN    = 15.0  # % of starting capital → pause trading


class EquityCurve:
    """
    Tracks daily equity curve with automatic risk signal generation.
    Persists to CSV (human readable) and JSON (machine readable).
    """

    def __init__(
        self,
        starting_capital: float = None,
        mode: str = "LIVE",
        csv_path: Optional[Path] = None,
        json_path: Optional[Path] = None,
    ) -> None:
        env_fund = float(os.getenv("TOTAL_FUND", TOTAL_FUND))
        self._starting_capital = starting_capital if starting_capital is not None else env_fund
        self._mode = str(mode or "LIVE").upper()
        if csv_path:
            self._csv_path = Path(csv_path)
        else:
            self._csv_path = OBSERVE_CURVE_CSV if self._mode == "OBSERVE" else LIVE_CURVE_CSV
            
        if json_path:
            self._json_path = Path(json_path)
        else:
            self._json_path = OBSERVE_CURVE_JSON if self._mode == "OBSERVE" else LIVE_CURVE_JSON

        Path(JOURNAL_DIR).mkdir(parents=True, exist_ok=True)
        self._history = self._load()
        self._init_csv()

    @property
    def current_equity(self) -> float:
        if self._history:
            return float(self._history[-1].get("ending_equity", self._starting_capital))
        return float(self._starting_capital)

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def record_day(
        self,
        gross_pnl:   float,
        net_pnl:     float,
        trades:      int,
        wins:        int,
        mode:        str = None,
        notes:       str = "",
        day:         date | str | None = None,
        total_charges: float = None,
    ) -> dict:
        """
        Record one trading day's result.
        Call this at EOD (15:32 IST) automatically via AnalyticsAgent.

        Returns:
            dict with risk_signal: "NORMAL" | "REDUCE_SIZE" | "PAUSE"
        """
        if isinstance(day, date):
            today = day.isoformat()
        elif day:
            today = str(day)
        else:
            today = date.today().isoformat()

        existing_idx = next(
            (idx for idx, row in enumerate(self._history) if str(row.get("date")) == today),
            None,
        )
        history_without_today = [
            row for row in self._history if str(row.get("date")) != today
        ]

        # Calculate equity
        prev_equity = float(history_without_today[-1].get("ending_equity", self._starting_capital) or self._starting_capital) if history_without_today \
                      else float(self._starting_capital)
        end_equity  = float(prev_equity + float(net_pnl))
        pnl_pct     = float(float(net_pnl) / max(prev_equity, 1.0) * 100.0)

        # Running max and drawdown
        all_ending   = [float(r.get("ending_equity", self._starting_capital) or self._starting_capital) for r in history_without_today] + [end_equity]
        running_max  = float(max(all_ending))
        drawdown_pct = float((running_max - end_equity) / max(running_max, 1.0) * 100.0)
        max_dd       = float(max(
            [float(r.get("current_drawdown_pct", 0.0) or 0.0) for r in history_without_today] + [drawdown_pct]
        ))

        # Consecutive loss days
        consec = 0
        for r in reversed(history_without_today):
            pnl_val = float(r.get("net_pnl_inr", r.get("net_pnl", 0.0)) or 0.0)
            if pnl_val < 0:
                consec += 1
            else:
                break
        if net_pnl < 0:
            consec += 1
        else:
            consec = 0

        # Risk signal
        signal = self._risk_signal(consec, drawdown_pct)

        calculated_charges = float(total_charges if total_charges is not None else max(0.0, float(gross_pnl) - float(net_pnl)))

        row = {
            "date":                  today,
            "starting_equity":       round(prev_equity, 2),
            "ending_equity":         round(end_equity, 2),
            "gross_pnl_inr":         round(gross_pnl, 2),
            "total_charges":         round(calculated_charges, 2),
            "net_pnl_inr":           round(net_pnl, 2),
            "pnl_pct":               round(pnl_pct, 3),
            "trades":                trades,
            "wins":                  wins,
            "win_rate":              round(wins / max(trades, 1) * 100, 1),
            "running_max_equity":    round(running_max, 2),
            "current_drawdown_pct":  round(drawdown_pct, 3),
            "max_drawdown_pct":      round(max_dd, 3),
            "consecutive_losses":    consec,
            "trading_mode":          mode or TRADING_MODE,
            "risk_signal":           signal,
            "notes":                 notes,
        }

        if existing_idx is None:
            self._history.append(row)
        else:
            self._history[existing_idx] = row
        self._history.sort(key=lambda item: str(item.get("date", "")))
        self._rewrite_csv()
        self._write_json()

        try:
            from loguru import logger
            emoji = "📈" if net_pnl > 0 else "📉"
            logger.info(
                f"[EquityCurve] {emoji} {today} | "
                f"Net=₹{net_pnl:+,.0f} ({pnl_pct:+.2f}%) | "
                f"Equity=₹{end_equity:,.0f} | "
                f"DD={drawdown_pct:.1f}% | "
                f"ConsecLoss={consec} | "
                f"Signal={signal}"
            )
        except Exception:
            pass

        return row

    def get_risk_signal(self) -> str:
        """
        Current risk signal based on recent performance.
        Called by executor before entering a new trade.

        Returns:
            "NORMAL"      → trade normally
            "REDUCE_SIZE" → use 50% of normal position size
            "PAUSE"       → do not enter new trades (OBSERVE mode)
        """
        if not self._history:
            return "NORMAL"
        consec   = self._history[-1].get("consecutive_losses", 0)
        drawdown = self._history[-1].get("current_drawdown_pct", 0)
        return self._risk_signal(consec, drawdown)

    def get_monthly_summary(self, year: int = None, month: int = None) -> dict:
        """Monthly performance metrics for EOD report."""
        today = date.today()
        y = year  or today.year
        m = month or today.month
        prefix = f"{y}-{m:02d}"

        month_rows = [r for r in self._history if str(r["date"]).startswith(prefix)]
        if not month_rows:
            return {"period": prefix, "no_data": True}

        total_net  = sum(r["net_pnl_inr"] for r in month_rows)
        total_days = len(month_rows)
        win_days   = sum(1 for r in month_rows if r["net_pnl_inr"] > 0)
        max_dd     = max(r["current_drawdown_pct"] for r in month_rows)
        start_eq   = month_rows[0]["starting_equity"]
        end_eq     = month_rows[-1]["ending_equity"]
        month_ret  = (end_eq - start_eq) / max(start_eq, 1) * 100
        best_day   = max(r["net_pnl_inr"] for r in month_rows)
        worst_day  = min(r["net_pnl_inr"] for r in month_rows)

        return {
            "period":        prefix,
            "trading_days":  total_days,
            "win_days":      win_days,
            "day_win_rate":  round(win_days / max(total_days, 1) * 100, 1),
            "net_pnl_inr":   round(total_net, 2),
            "month_return":  round(month_ret, 2),
            "max_drawdown":  round(max_dd, 2),
            "best_day_inr":  round(best_day, 2),
            "worst_day_inr": round(worst_day, 2),
            "start_equity":  round(start_eq, 2),
            "end_equity":    round(end_eq, 2),
        }

    def get_summary(self) -> dict:
        """Returns high-level KPI summary for dashboard Master Journal card."""
        from utils.live_trade_history import get_live_trade_history, get_observe_trade_history
        hist = get_observe_trade_history() if self._mode == "OBSERVE" else get_live_trade_history()
        summary = hist.summary().get("all", {})
        
        env_fund = float(os.getenv("TOTAL_FUND", self._starting_capital or TOTAL_FUND))
        env_deployed = float(os.getenv("DEPLOYED_CAPITAL", DEPLOYED_CAPITAL))
        starting_cap = env_fund
        trades = summary.get("trades", 0)
        wins = summary.get("wins", 0)
        losses = summary.get("losses", 0)
        wr = summary.get("win_rate", 0.0)
        gross_pnl = summary.get("gross_pnl", 0.0)
        charges = summary.get("total_charges", 0.0)
        net_pnl = summary.get("net_pnl", 0.0)
        curr_equity = starting_cap + net_pnl
        account_roi_pct = round((net_pnl / max(starting_cap, 1.0)) * 100.0, 2)

        return {
            "starting_capital": starting_cap,
            "deployed_capital": env_deployed,
            "current_equity": round(curr_equity, 2),
            "account_roi_pct": account_roi_pct,
            "closed_trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wr,
            "gross_pnl": round(gross_pnl, 2),
            "charges_taxes": round(charges, 2),
            "net_pnl": round(net_pnl, 2),
            "mode": self._mode,
        }

    def get_equity_curve_data(self) -> dict:
        self._history = self._load()
        env_fund = float(os.getenv("TOTAL_FUND", self._starting_capital or TOTAL_FUND))
        return {
            "starting_capital": env_fund,
            "current_equity": self.current_equity,
            "history": self._history,
            "mode": self._mode,
        }

    def get_streak_info(self) -> dict:
        """Current win/loss streak and consecutive counts."""
        if not self._history:
            return {"streak": "none", "count": 0}
        last = self._history[-1]
        return {
            "streak":       "win" if last["net_pnl_inr"] > 0 else "loss",
            "count":        last["consecutive_losses"],
            "risk_signal":  last["risk_signal"],
        }

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    @staticmethod
    def _risk_signal(consecutive_losses: int, drawdown_pct: float) -> str:
        if drawdown_pct >= MAX_ACCEPTABLE_DRAWDOWN:
            return "PAUSE"
        # Position sizing is determined per-signal dynamically (FULL / REDUCED / HERO_ZERO)
        # rather than imposing a blanket daily reduction on the entire session.
        return "NORMAL"

    def _load(self) -> list[dict]:
        if not self._json_path.exists():
            return []
        try:
            with open(self._json_path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get("history", [])
                return []
        except Exception:
            return []

    def _init_csv(self) -> None:
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CURVE_COLUMNS)

    def _write_csv(self, row: dict) -> None:
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in CURVE_COLUMNS])

    def _rewrite_csv(self) -> None:
        rows = sorted(self._history, key=lambda row: str(row.get("date", "")))
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CURVE_COLUMNS)
            for row in rows:
                writer.writerow([row.get(c, "") for c in CURVE_COLUMNS])

    def _write_json(self) -> None:
        with open(self._json_path, "w") as f:
            json.dump(self._history, f, indent=2, default=str)


# ── Singletons ────────────────────────────────────────────────────────────────
_live_curve: Optional[EquityCurve] = None
_observe_curve: Optional[EquityCurve] = None


def get_live_equity_curve() -> EquityCurve:
    global _live_curve
    if _live_curve is None:
        _live_curve = EquityCurve(mode="LIVE")
    return _live_curve


def get_observe_equity_curve() -> EquityCurve:
    global _observe_curve
    if _observe_curve is None:
        _observe_curve = EquityCurve(mode="OBSERVE")
    return _observe_curve


def get_equity_curve(mode: Optional[str] = None) -> EquityCurve:
    resolved_mode = str(mode or os.getenv("TRADING_MODE", "AUTO")).strip().upper()
    if resolved_mode == "OBSERVE":
        return get_observe_equity_curve()
    return get_live_equity_curve()


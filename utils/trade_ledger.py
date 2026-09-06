"""
utils/trade_ledger.py — Master Trade Ledger
============================================
Industry standard: every trade written to a persistent ledger CSV AND SQLite DB.
Required for:
  - CA / tax filing (STT, LTCG, speculative income segregation)
  - Broker reconciliation (match our records vs Zerodha/Dhan contract note)
  - SEBI algo audit trail (mandatory for registered algo systems)
  - ML training data (labelled outcomes for retraining)
  - Performance attribution (which strategy → which outcome)

Equivalent: Zerodha Console trade book, Sensibull trade log, AlgoTest journal.

CSV columns match NSE contract note format so you can cross-verify with broker.
SQLite enables fast querying: "show me all AMD strategy trades this month"

Usage:
    from utils.trade_ledger import TradeLedger
    ledger = TradeLedger()
    ledger.record_trade(entry_data, exit_data)
    ledger.export_excel("reports/trades_april.xlsx")  # monthly
"""

import csv
import json
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

# ── Paths ─────────────────────────────────────────────────────────────────────
try:
    from config.settings import JOURNAL_DIR
except ImportError:
    JOURNAL_DIR = "journal"

LEDGER_DIR  = Path(JOURNAL_DIR)
LEDGER_CSV  = LEDGER_DIR / "master_trade_ledger.csv"
LEDGER_DB   = LEDGER_DIR / "signalforge.db"

# ── CSV columns (NSE contract note + SignalForge extras) ──────────────────────
CSV_COLUMNS = [
    # Trade identity
    "trade_id", "signal_id", "date", "entry_time", "exit_time",
    # Instrument
    "symbol", "option_symbol", "direction", "expiry", "strike", "option_type",
    # Execution
    "entry_premium", "exit_premium", "lots", "quantity",
    "entry_nifty", "exit_nifty",
    # P&L (NSE format)
    "gross_pnl_pct", "gross_pnl_inr", "brokerage_est",
    "stt_est", "net_pnl_inr", "net_pnl_pct",
    # Risk
    "sl_premium", "target_premium", "risk_reward",
    "initial_risk_pct", "peak_pnl_pct",
    # Exit
    "exit_reason", "holding_minutes",
    # Signal quality
    "strategies_fired", "votes", "strategy_conf", "ml_conf",
    "ml_decision", "setup_strength", "quality_score",
    # Regime context
    "regime", "adx_at_entry", "vix_at_entry", "det_conf",
    # TSL tracking
    "tsl_activated", "tsl_tier", "tsl_trail_pct",
    # Mode
    "trading_mode", "simulated",
    # Misc
    "notes",
]


def _next_trade_id(db_path: Path) -> str:
    """Generate sequential trade ID: SF-20260514-001"""
    today = date.today().strftime("%Y%m%d")
    conn  = sqlite3.connect(db_path)
    cur   = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE date = ?",
        (date.today().isoformat(),)
    )
    n = cur.fetchone()[0] + 1
    conn.close()
    return f"SF-{today}-{n:03d}"


class TradeLedger:
    """
    Persistent trade record. Thread-safe for single-process use.
    Writes every trade to both CSV (human readable) and SQLite (queryable).
    """

    def __init__(self) -> None:
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._init_csv()

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def record_trade(self, entry: dict, exit_: dict) -> str:
        """
        Write a completed trade to ledger.

        Args:
            entry: dict from ORDER_PLACED / ORDER_DRY_RUN payload
            exit_:  dict from POSITION_CLOSED payload

        Returns:
            trade_id: "SF-20260514-001"
        """
        trade_id = _next_trade_id(LEDGER_DB)
        now      = datetime.now(IST)
        sig      = entry.get("signal", {})

        # ── Calculate P&L ─────────────────────────────────────────────────────
        entry_prem = float(
            entry.get("actual_premium", entry.get("entry_premium", entry.get("est_premium", 0)))
        )
        exit_prem  = float(exit_.get("exit_premium",  0))
        lots       = int(entry.get("lots", 1) or 1)
        qty        = int(entry.get("quantity") or exit_.get("quantity") or (lots * 65))
        gross_inr  = (exit_prem - entry_prem) * qty
        gross_pct  = float(exit_.get("pnl_pct", 0))

        # Comprehensive Indian F&O Brokerage & Statutory Taxes
        from utils.brokerage_calculator import calculate_option_trade_charges
        charges = calculate_option_trade_charges(entry_prem, exit_prem, qty)
        stt       = charges.stt
        brokerage = charges.brokerage
        net_inr   = charges.net_pnl_inr
        net_pct   = charges.net_pnl_pct

        # ── Build record ──────────────────────────────────────────────────────
        option_sym = entry.get("option_symbol", "")
        parts      = _parse_option_symbol(option_sym)

        strats_fired = sig.get("strategies_fired", [])
        sig_id = str(entry.get("signal_id", "") or sig.get("signal_id", "") or exit_.get("signal_id", "") or "")
        record = {
            "trade_id":        trade_id,
            "signal_id":       sig_id,
            "date":            date.today().isoformat(),
            "entry_time":      entry.get("entry_time", now.strftime("%H:%M:%S")),
            "exit_time":       exit_.get("exit_time",  now.strftime("%H:%M:%S")),
            "symbol":          "NIFTY",
            "option_symbol":   option_sym,
            "direction":       sig.get("direction", exit_.get("direction", "")),
            "expiry":          parts.get("expiry", ""),
            "strike":          parts.get("strike", ""),
            "option_type":     parts.get("option_type", ""),
            "entry_premium":   entry_prem,
            "exit_premium":    exit_prem,
            "lots":            lots,
            "quantity":        qty,
            "entry_nifty":     sig.get("nifty_ltp", 0),
            "exit_nifty":      exit_.get("nifty_ltp", 0),
            "gross_pnl_pct":   round(gross_pct, 4),
            "gross_pnl_inr":   round(gross_inr, 2),
            "brokerage_est":   brokerage,
            "stt_est":         stt,
            "net_pnl_inr":     net_inr,
            "net_pnl_pct":     net_pct,
            "sl_premium":      round(float(entry.get("sl_premium", 0)), 2),
            "target_premium":  round(float(entry.get("target_premium", 0)), 2),
            "risk_reward":     round(float(entry.get("risk_reward", 0)), 2),
            "initial_risk_pct":round(float(entry.get("risk_pct", 25.0)), 2),
            "peak_pnl_pct":    round(float(exit_.get("peak_pnl", 0)), 4),
            "exit_reason":     exit_.get("exit_reason", "UNKNOWN"),
            "holding_minutes": int(exit_.get("holding_minutes", exit_.get("candles_held", 0)) * 5),
            "strategies_fired":"|".join(strats_fired) if isinstance(strats_fired, list) else str(strats_fired),
            "votes":           sig.get("votes", 0),
            "strategy_conf":   round(float(sig.get("confidence", 0)), 4),
            "ml_conf":         round(float(entry.get("ml_confidence", 0)), 4),
            "ml_decision":     entry.get("ml_decision", ""),
            "setup_strength":  round(float(sig.get("setup_strength", 0)), 4),
            "quality_score":   round(float(sig.get("quality_score", 0)), 4),
            "regime":          sig.get("regime", ""),
            "adx_at_entry":    round(float(sig.get("adx", 0)), 2),
            "vix_at_entry":    round(float(sig.get("india_vix", 0)), 2),
            "det_conf":        round(float(sig.get("det_conf", 0)), 4),
            "tsl_activated":   str(exit_.get("tsl_activated", False)),
            "tsl_tier":        exit_.get("tsl_tier", ""),
            "tsl_trail_pct":   exit_.get("tsl_trail_pct", 0),
            "trading_mode":    entry.get("mode", "OBSERVE"),
            "simulated":       str(exit_.get("simulated", True)),
            "notes":           "",
        }

        self._write_csv(record)
        self._write_db(record)

        try:
            from loguru import logger
            emoji = "🟢" if gross_pct > 0 else "🔴"
            logger.info(
                f"[TradeLedger] {emoji} {trade_id} | "
                f"{record['exit_reason']} | "
                f"PnL={gross_pct:+.2f}% (₹{gross_inr:+.0f}) | "
                f"Net=₹{net_inr:+.0f} | "
                f"Strategies={record['strategies_fired']}"
            )
        except Exception:
            pass

        return trade_id

    def get_today_summary(self) -> dict:
        """Return today's P&L summary from DB."""
        conn = sqlite3.connect(LEDGER_DB)
        cur  = conn.execute(
            """SELECT COUNT(*), SUM(gross_pnl_inr), SUM(net_pnl_inr),
                      SUM(CASE WHEN gross_pnl_inr > 0 THEN 1 ELSE 0 END)
               FROM trades WHERE date = ?""",
            (date.today().isoformat(),)
        )
        row = cur.fetchone()
        conn.close()
        n, gross, net, wins = row
        return {
            "trades":    n or 0,
            "gross_pnl": round(gross or 0, 2),
            "net_pnl":   round(net or 0, 2),
            "wins":      wins or 0,
            "win_rate":  round((wins or 0) / max(n or 1, 1) * 100, 1),
        }

    def get_month_summary(self, year: int = None, month: int = None) -> dict:
        """Return monthly summary — used for equity curve."""
        today = date.today()
        y = year  or today.year
        m = month or today.month
        prefix = f"{y}-{m:02d}"
        conn = sqlite3.connect(LEDGER_DB)
        cur  = conn.execute(
            """SELECT COUNT(*), SUM(gross_pnl_inr), SUM(net_pnl_inr),
                      AVG(gross_pnl_pct), MAX(peak_pnl_pct),
                      SUM(CASE WHEN gross_pnl_inr > 0 THEN 1 ELSE 0 END)
               FROM trades WHERE date LIKE ?""",
            (f"{prefix}%",)
        )
        row = cur.fetchone()
        conn.close()
        n, gross, net, avg_pct, best, wins = row
        return {
            "period":     prefix,
            "trades":     n or 0,
            "gross_pnl":  round(gross or 0, 2),
            "net_pnl":    round(net or 0, 2),
            "avg_pct":    round(avg_pct or 0, 2),
            "best_trade": round(best or 0, 2),
            "wins":       wins or 0,
            "win_rate":   round((wins or 0) / max(n or 1, 1) * 100, 1),
        }

    def export_excel(self, output_path: str = None) -> str:
        """
        Export ledger to Excel with formatting.
        Requires openpyxl: pip install openpyxl
        """
        if output_path is None:
            output_path = str(LEDGER_DIR / f"trade_report_{date.today().isoformat()}.xlsx")
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, numbers
            from openpyxl.utils import get_column_letter

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Trade Ledger"

            # Header row
            header_fill = PatternFill("solid", fgColor="1F3864")
            header_font = Font(color="FFFFFF", bold=True, size=10)
            for col, header in enumerate(CSV_COLUMNS, 1):
                cell = ws.cell(row=1, column=col, value=header.replace("_", " ").title())
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center")

            # Data rows from DB
            conn = sqlite3.connect(LEDGER_DB)
            rows = conn.execute(f"SELECT {','.join(CSV_COLUMNS)} FROM trades ORDER BY date, entry_time").fetchall()
            conn.close()

            green = PatternFill("solid", fgColor="E2EFDA")
            red   = PatternFill("solid", fgColor="FCE4D6")

            for r, row in enumerate(rows, 2):
                pnl_val = row[CSV_COLUMNS.index("gross_pnl_pct")]
                fill    = green if (pnl_val or 0) > 0 else red
                for col, val in enumerate(row, 1):
                    cell       = ws.cell(row=r, column=col, value=val)
                    cell.fill  = fill
                    cell.alignment = Alignment(horizontal="center")

            # Auto-width
            for col in ws.columns:
                max_len = max(len(str(cell.value or "")) for cell in col)
                ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 2, 30)

            wb.save(output_path)
            return output_path
        except ImportError:
            # Fallback: plain CSV
            import shutil
            shutil.copy(LEDGER_CSV, output_path.replace(".xlsx", ".csv"))
            return output_path.replace(".xlsx", ".csv")

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        conn = sqlite3.connect(LEDGER_DB)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS trades (
                {', '.join(f'{col} TEXT' for col in CSV_COLUMNS)},
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            conn.execute("ALTER TABLE trades ADD COLUMN signal_id TEXT")
        except sqlite3.OperationalError:
            pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_date ON trades(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_exit_reason ON trades(exit_reason)")
        conn.commit()
        conn.close()

    def _init_csv(self) -> None:
        if not LEDGER_CSV.exists():
            with open(LEDGER_CSV, "w", newline="") as f:
                csv.writer(f).writerow(CSV_COLUMNS)

    def _write_csv(self, record: dict) -> None:
        with open(LEDGER_CSV, "a", newline="") as f:
            csv.writer(f).writerow([record.get(c, "") for c in CSV_COLUMNS])

    def _write_db(self, record: dict) -> None:
        conn = sqlite3.connect(LEDGER_DB)
        placeholders = ",".join(["?"] * len(CSV_COLUMNS))
        conn.execute(
            f"INSERT INTO trades ({','.join(CSV_COLUMNS)}) VALUES ({placeholders})",
            [str(record.get(c, "")) for c in CSV_COLUMNS]
        )
        conn.commit()
        conn.close()


    def get_all_trades(self) -> list[dict]:
        from utils.live_trade_history import get_live_trade_history
        return get_live_trade_history().rows()


def _parse_option_symbol(sym: str) -> dict:
    """Parse NIFTY26APR2522500CE → expiry/strike/type."""
    import re
    m = re.match(r"NIFTY(\d{2})([A-Z]{3})(\d{2,4})(\d+)(CE|PE)", sym or "")
    if m:
        return {
            "expiry":      f"20{m.group(3)}-{m.group(2)}-{m.group(1)}",
            "strike":      m.group(4),
            "option_type": m.group(5),
        }
    return {}


# ── Singleton ─────────────────────────────────────────────────────────────────
_ledger: Optional[TradeLedger] = None

def get_ledger() -> TradeLedger:
    global _ledger
    if _ledger is None:
        _ledger = TradeLedger()
    return _ledger

get_trade_ledger = get_ledger


#!/usr/bin/env python3
"""
scripts/trade_reconciliation.py — Broker vs SignalForge Trade Reconciliation
=============================================================================
Every evening: compare SignalForge's trade ledger vs broker contract note.
Flags any mismatches: missed fills, wrong quantity, price slippage.

This is mandatory for:
  - Live trading audit trail
  - SEBI compliance for registered algos
  - Catching order execution failures silently
  - Slippage analysis (are we getting expected fills?)

USAGE:
  python scripts/trade_reconciliation.py
  python scripts/trade_reconciliation.py --broker-csv /path/to/zerodha_tradebook.csv
  python scripts/trade_reconciliation.py --date 2026-05-15

Broker CSV format (Zerodha/Dhan tradebook):
  symbol, trade_date, trade_time, trade_type, quantity, price, order_id

Run via cron at 17:30 IST daily:
  30 17 * * 1-5 cd /opt/signalforge && python scripts/trade_reconciliation.py
"""

import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytz
IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import JOURNAL_DIR, LOGS_DIR, TELEGRAM_ENABLED
except ImportError:
    JOURNAL_DIR      = "journal"
    LOGS_DIR         = "logs"
    TELEGRAM_ENABLED = False

LEDGER_DB = Path(JOURNAL_DIR) / "signalforge.db"

G   = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
W   = "\033[97m"; D = "\033[2m";  RST = "\033[0m"

SLIPPAGE_WARN_PCT = 2.0   # warn if fill price differs > 2% from expected


def load_signalforge_trades(target_date: str) -> list[dict]:
    """Load today's trades from SignalForge ledger."""
    if not LEDGER_DB.exists():
        return []
    conn  = sqlite3.connect(LEDGER_DB)
    cols  = [d[0] for d in conn.execute("PRAGMA table_info(trades)").fetchall()]
    rows  = conn.execute(
        "SELECT * FROM trades WHERE date = ?", (target_date,)
    ).fetchall()
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def load_broker_trades(csv_path: str) -> list[dict]:
    """
    Load broker tradebook CSV.
    Supports Zerodha and Dhan CSV formats.
    """
    if not os.path.exists(csv_path):
        return []
    trades = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append({k.strip().lower(): v.strip() for k, v in row.items()})
    return trades


def reconcile(sf_trades: list[dict], broker_trades: list[dict]) -> dict:
    """
    Compare SignalForge trades vs broker fills.
    Returns reconciliation report.
    """
    matched    = []
    sf_only    = []
    broker_only = []
    mismatches = []

    # Index broker trades by option symbol
    broker_idx: dict[str, list] = {}
    for bt in broker_trades:
        sym = (bt.get("symbol", bt.get("tradingsymbol", ""))
               .replace(" ", "").upper())
        broker_idx.setdefault(sym, []).append(bt)

    for sf in sf_trades:
        sym = str(sf.get("option_symbol", "")).upper()
        matches = broker_idx.get(sym, [])

        if not matches:
            sf_only.append({
                "trade_id":   sf.get("trade_id"),
                "symbol":     sym,
                "direction":  sf.get("direction"),
                "entry_prem": sf.get("entry_premium"),
                "issue":      "NO_BROKER_FILL — order may not have executed",
            })
            continue

        # Find closest match by price
        sf_entry = float(sf.get("entry_premium", 0) or 0)
        sf_exit  = float(sf.get("exit_premium",  0) or 0)
        best     = min(matches, key=lambda b:
                       abs(float(b.get("price", b.get("average_price", 0)) or 0) - sf_entry))

        broker_price = float(best.get("price", best.get("average_price", 0)) or 0)
        slippage_pct = abs(broker_price - sf_entry) / max(sf_entry, 0.01) * 100

        record = {
            "trade_id":      sf.get("trade_id"),
            "symbol":        sym,
            "sf_entry":      sf_entry,
            "broker_fill":   broker_price,
            "slippage_pct":  round(slippage_pct, 3),
            "sf_direction":  sf.get("direction"),
            "sf_exit":       sf_exit,
            "sf_pnl":        sf.get("gross_pnl_pct"),
            "matched":       True,
        }

        if slippage_pct > SLIPPAGE_WARN_PCT:
            record["warning"] = f"HIGH_SLIPPAGE: {slippage_pct:.1f}%"
            mismatches.append(record)
        else:
            matched.append(record)

    # Broker trades not in SignalForge
    sf_symbols = {str(sf.get("option_symbol", "")).upper() for sf in sf_trades}
    for sym, bts in broker_idx.items():
        if sym not in sf_symbols:
            for bt in bts:
                broker_only.append({
                    "symbol":  sym,
                    "price":   bt.get("price", bt.get("average_price")),
                    "qty":     bt.get("quantity", bt.get("qty")),
                    "issue":   "BROKER_TRADE_NOT_IN_SIGNALFORGE — manual trade or duplicate",
                })

    # Slippage summary
    all_slippage = [r["slippage_pct"] for r in matched + mismatches]
    avg_slippage = sum(all_slippage) / max(len(all_slippage), 1)

    return {
        "date":          date.today().isoformat(),
        "sf_trades":     len(sf_trades),
        "broker_trades": len(broker_trades),
        "matched":       len(matched),
        "sf_only":       len(sf_only),
        "broker_only":   len(broker_only),
        "mismatches":    len(mismatches),
        "avg_slippage":  round(avg_slippage, 3),
        "status":        "CLEAN" if not sf_only and not broker_only and not mismatches
                         else "ISSUES_FOUND",
        "details": {
            "matched":     matched,
            "sf_only":     sf_only,
            "broker_only": broker_only,
            "mismatches":  mismatches,
        },
    }


def print_report(report: dict) -> None:
    status_col = G if report["status"] == "CLEAN" else R

    print(f"\n{W}{'═'*60}{RST}")
    print(f"{W}  Trade Reconciliation  |  {report['date']}{RST}")
    print(f"{W}{'═'*60}{RST}\n")

    print(f"  {W}Summary:{RST}")
    print(f"    SignalForge trades:  {report['sf_trades']}")
    print(f"    Broker trades:      {report['broker_trades']}")
    print(f"    Matched:            {G}{report['matched']}{RST}")
    mm_val = report['mismatches']
    sf_val = report['sf_only']
    bo_val = report['broker_only']
    print(f"    Mismatches:         {R+str(mm_val)+RST if mm_val else '0'}")
    print(f"    SF only (no fill):  {R+str(sf_val)+RST if sf_val else '0'}")
    print(f"    Broker only:        {Y+str(bo_val)+RST if bo_val else '0'}")
    print(f"    Avg slippage:       {report['avg_slippage']:.2f}%")
    print(f"\n  {W}Status: {status_col}{report['status']}{RST}\n")

    d = report["details"]

    if d["sf_only"]:
        print(f"  {R}⚠️  Trades NOT filled by broker:{RST}")
        for t in d["sf_only"]:
            print(f"    {t['trade_id']}  {t['symbol']}  "
                  f"expected ₹{t['entry_prem']}  — {t['issue']}")

    if d["mismatches"]:
        print(f"\n  {Y}⚠️  High slippage trades:{RST}")
        for t in d["mismatches"]:
            print(f"    {t['trade_id']}  {t['symbol']}  "
                  f"SF=₹{t['sf_entry']} Broker=₹{t['broker_fill']}  "
                  f"slip={t['slippage_pct']:.1f}%")

    if d["broker_only"]:
        print(f"\n  {Y}ℹ️  Broker trades not in SignalForge (manual?):{RST}")
        for t in d["broker_only"]:
            print(f"    {t['symbol']}  ₹{t['price']}  qty={t['qty']}  — {t['issue']}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",       default=date.today().isoformat())
    parser.add_argument("--broker-csv", default=None,
                        help="Path to broker tradebook CSV")
    args = parser.parse_args()

    print(f"\n{D}Running reconciliation for {args.date}...{RST}")

    sf_trades     = load_signalforge_trades(args.date)
    broker_trades = load_broker_trades(args.broker_csv) if args.broker_csv else []

    if not sf_trades:
        print(f"  No SignalForge trades found for {args.date}")
        return

    if not broker_trades:
        print(f"  {Y}No broker CSV provided — showing SignalForge trades only{RST}")
        print(f"  Trades today: {len(sf_trades)}")
        for t in sf_trades:
            pnl = float(t.get("gross_pnl_pct", 0) or 0)
            col = G if pnl > 0 else R
            print(f"    {t.get('trade_id')}  {t.get('option_symbol')}  "
                  f"{col}{pnl:+.2f}%{RST}  {t.get('exit_reason')}")
        return

    report = reconcile(sf_trades, broker_trades)
    print_report(report)

    # Save report
    out = Path(LOGS_DIR) / f"reconciliation_{args.date}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {out}")

    # Telegram
    if TELEGRAM_ENABLED and report["status"] != "CLEAN":
        try:
            import asyncio
            from utils.telegram_notifier import get_notifier
            msg = (
                f"⚠️ *Reconciliation Issues: {args.date}*\n"
                f"SF only: {report['sf_only']}\n"
                f"Mismatches: {report['mismatches']}\n"
                f"Avg slippage: {report['avg_slippage']:.2f}%"
            )
            asyncio.run(get_notifier().send_text(msg, target="LIVE"))
        except Exception:
            pass


if __name__ == "__main__":
    main()

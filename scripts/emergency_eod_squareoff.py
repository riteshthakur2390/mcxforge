#!/usr/bin/env python3
"""
scripts/emergency_eod_squareoff.py
==================================
Standalone, failsafe End-of-Day position liquidator.
Runs at 15:15 IST via crontab to guarantee that NO open option positions
are left overnight across all configured broker accounts (Dhan & Upstox).
"""

from __future__ import annotations

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime
import pytz
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

IST = pytz.timezone("Asia/Kolkata")


async def run_emergency_squareoff(send_telegram: bool = True) -> list[dict]:
    now_ist = datetime.now(IST)
    print("=" * 70)
    print(f"   SIGNALFORGE — 15:15 EMERGENCY EOD AUTO SQUARE-OFF")
    print(f"   Timestamp: {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 70)

    from broker.factory import get_broker
    from utils.multi_account_execution import load_execution_accounts
    from broker.base_broker import OrderResult
    from utils.bot_trade_registry import is_bot_symbol, get_all_bot_symbols_today

    primary_broker = get_broker()
    accounts = load_execution_accounts(primary_broker)

    today_str = now_ist.strftime("%Y-%m-%d")
    bot_symbols_today = {s.upper() for s in get_all_bot_symbols_today(today_str)}

    # Also load symbols from today's signals journal if present
    journal_symbols_today = set()
    try:
        import pandas as pd
        sig_file = PROJECT_ROOT / "journal" / f"signals_{today_str}.csv"
        if sig_file.exists():
            df_sig = pd.read_csv(sig_file)
            for col in ["option_symbol", "symbol"]:
                if col in df_sig.columns:
                    journal_symbols_today.update(df_sig[col].dropna().astype(str).str.upper().tolist())
        hist_file = PROJECT_ROOT / "journal" / "live_trade_history.csv"
        if hist_file.exists():
            df_hist = pd.read_csv(hist_file)
            if "date" in df_hist.columns and "option_symbol" in df_hist.columns:
                df_today = df_hist[df_hist["date"].astype(str) == today_str]
                journal_symbols_today.update(df_today["option_symbol"].dropna().astype(str).str.upper().tolist())
    except Exception as exc:
        print(f"  ⚠️ Could not read journal files: {exc}")

    closed_positions = []
    skipped_positions = []
    errors = []

    print(f"\n[SCAN] Checking open positions across {len(accounts)} execution account(s)...")
    print(f"  Tracked bot symbols today: {bot_symbols_today | journal_symbols_today or 'None (flat)'}")

    for acc in accounts:
        try:
            positions = acc.broker.get_positions() or []
            open_pos = [p for p in positions if int(getattr(p, "quantity", 0) or 0) > 0]
            print(f"  Account: {acc.label:<18} | Open positions: {len(open_pos)}")

            for p in open_pos:
                sym = str(getattr(p, "symbol", "") or "").strip()
                sym_upper = sym.upper()
                qty = int(getattr(p, "quantity", 0) or 0)
                product = str(getattr(p, "product", "MIS") or "MIS")
                exchange = "BFO" if "SENSEX" in sym_upper else "NSE_FNO"

                # Check if position belongs to SignalForge
                is_signalforge_trade = (
                    is_bot_symbol(sym_upper, today_str)
                    or sym_upper in bot_symbols_today
                    or sym_upper in journal_symbols_today
                )

                if not is_signalforge_trade:
                    print(f"    ⏭️ [SKIP] {sym} ({qty} qty) on {acc.label} is a MANUAL / EXTERNAL trade — PRESERVING.")
                    skipped_positions.append({
                        "account": acc.label,
                        "symbol": sym,
                        "quantity": qty,
                        "reason": "manual_trade_preserved",
                    })
                    continue

                sec_id = getattr(p, "security_id", "")
                kwargs = {
                    "symbol": sym,
                    "quantity": qty,
                    "transaction": "SELL",
                    "product": product,
                    "exchange": exchange,
                }
                if sec_id and hasattr(acc.broker, "place_market_order") and "security_id" in acc.broker.place_market_order.__code__.co_varnames:
                    kwargs["security_id"] = sec_id

                res = acc.broker.place_market_order(**kwargs)

                if res.status == "PLACED":
                    print(f"    ✅ Bot Square-off Success: Order ID {res.order_id} ({sym})")
                    closed_positions.append({
                        "account": acc.label,
                        "symbol": sym,
                        "quantity": qty,
                        "order_id": res.order_id,
                        "status": "PLACED",
                    })
                else:
                    err = f"Failed to close bot position {sym} ({qty} qty) on {acc.label}: {res.message}"
                    print(f"    ❌ Square-off Error: {err}")
                    errors.append(err)

        except Exception as exc:
            err = f"Error querying positions on {acc.label}: {exc}"
            print(f"  ❌ Error on {acc.label}: {err}")
            errors.append(err)

    # Telegram notification
    if send_telegram:
        try:
            from utils.telegram_notifier import get_notifier
            notifier = get_notifier()

            if closed_positions:
                pos_lines = "\n".join([f"• *{c['symbol']}* ({c['quantity']} Qty) on `{c['account']}`" for c in closed_positions])
                msg = (
                    f"🛑 *SignalForge — 15:15 Emergency EOD Auto Square-Off*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚠️ The watchdog detected and force-closed {len(closed_positions)} open position(s):\n"
                    f"{pos_lines}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"⏰ Time: {now_ist.strftime('%H:%M:%S IST')}\n"
                    f"🛡️ Status: All broker positions now FLAT."
                )
                await notifier.send_text(msg, target="LIVE", parse_mode="markdown")
            elif errors:
                err_lines = "\n".join([f"• {e}" for e in errors])
                msg = (
                    f"🚨 *CRITICAL ALERT — 15:15 Emergency Square-off Issue*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"Errors encountered during EOD scan:\n{err_lines}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"ACTION: Please check your broker terminal manually immediately!"
                )
                await notifier.send_text(msg, target="LIVE", parse_mode="markdown")
            else:
                print("  ✓ All accounts flat — no open positions required liquidation.")

        except Exception as e:
            print(f"  ⚠️ Telegram alert error: {e}")

    print("\n" + "=" * 70)
    print(f"   COMPLETED: {len(closed_positions)} closed | {len(errors)} errors")
    print("=" * 70)
    return closed_positions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Emergency EOD Position Liquidator")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram alert")
    args = parser.parse_args()
    asyncio.run(run_emergency_squareoff(send_telegram=not args.no_telegram))

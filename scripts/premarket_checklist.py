#!/usr/bin/env python3
"""
scripts/premarket_checklist.py — MCXForge Pre-Market Risk Gate
==============================================================
Run at 08:50 IST every trading morning (before 09:00 MCX open).
Checks all pre-market risk conditions and sets the day's trading posture.

Schedule in cron:
  50 8 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && ./venv/bin/python scripts/premarket_checklist.py

CHECKS PERFORMED:
  1. Commodity Overnight Gap (> 2.0% = volatile open, reduce size)
  2. Equity curve risk signal (PAUSE / REDUCE_SIZE / NORMAL)
  3. Broker Token & Quote validity check (broker auth & active contract quote)
  4. Active Commodity Contract Resolution (MCX instrument & lot sizing)

OUTPUT:
  - Console report with colour coding
  - logs/premarket_YYYY-MM-DD.json (machine readable)
  - Telegram message with day's posture
  - Sets TRADING_POSTURE env variable for session

POSTURE LEVELS:
  FULL_SIZE   → all systems normal, trade with standard position sizing
  REDUCE_HALF → elevated risk detected, use 50% position size
  OBSERVE     → too risky, paper watch only, zero live orders
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import (
        DEPLOYED_CAPITAL,
        JOURNAL_DIR,
        LOGS_DIR,
        MAX_DAILY_LOSS_PCT,
        PREMARKET_MAX_GAP_PCT,
        TELEGRAM_ENABLED,
    )
except ImportError:
    PREMARKET_MAX_GAP_PCT = 2.0
    TELEGRAM_ENABLED = False
    LOGS_DIR = "logs"
    JOURNAL_DIR = "journal"
    DEPLOYED_CAPITAL = 200000
    MAX_DAILY_LOSS_PCT = 3.0

REPORT_PATH = Path(LOGS_DIR)
REPORT_PATH.mkdir(exist_ok=True)

G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
B = "\033[94m"
W = "\033[97m"
RST = "\033[0m"


def check_gap(symbol: str) -> dict:
    """Check commodity overnight gap from previous close."""
    gap_pct = 0.0
    try:
        from broker.factory import create_broker
        broker = create_broker()
        now = datetime.now()
        start = (now - pytz.timezone("Asia/Kolkata").localize(datetime.now()).replace(tzinfo=None)).strftime("%Y-%m-%d")
        df = broker.get_historical_data(symbol, "5m", days_back=3)
        if df is not None and len(df) >= 2:
            prev_close = float(df["close"].iloc[-2])
            last_close = float(df["close"].iloc[-1])
            if prev_close > 0:
                gap_pct = (last_close - prev_close) / prev_close * 100
    except Exception:
        gap_pct = 0.0

    if abs(gap_pct) > PREMARKET_MAX_GAP_PCT:
        posture = "REDUCE_HALF"
        level = "LARGE_GAP"
    elif abs(gap_pct) > 1.0:
        posture = "FULL_SIZE"
        level = "MODERATE_GAP"
    else:
        posture = "FULL_SIZE"
        level = "FLAT_OPEN"

    return {
        "gap_pct": round(gap_pct, 3),
        "level": level,
        "posture": posture,
        "direction": "UP" if gap_pct > 0.3 else "DOWN" if gap_pct < -0.3 else "FLAT",
    }


def check_equity_curve() -> dict:
    """Check equity curve risk signal from yesterday's performance."""
    try:
        from utils.equity_curve import get_equity_curve

        curve = get_equity_curve()
        signal = curve.get_risk_signal()
        streak = curve.get_streak_info()
        posture = {
            "PAUSE": "OBSERVE",
            "REDUCE_SIZE": "REDUCE_HALF",
            "NORMAL": "FULL_SIZE",
        }.get(signal, "FULL_SIZE")
        return {"signal": signal, "posture": posture, "streak": streak}
    except Exception:
        return {"signal": "NORMAL", "posture": "FULL_SIZE", "streak": {}}


def check_broker_token(symbol: str) -> dict:
    """Verify broker token is still valid and fetch active commodity LTP."""
    try:
        from broker.factory import create_broker

        broker = create_broker()
        ltp = float(broker.get_ltp(symbol) or 0.0)
        valid = ltp > 0
        return {
            "valid": valid,
            "ltp": round(ltp, 2),
            "symbol": symbol,
            "broker": getattr(broker, "broker_name", broker.__class__.__name__),
        }
    except Exception as e:
        return {"valid": False, "symbol": symbol, "error": str(e)[:80]}


def check_commodity_contract(symbol: str) -> dict:
    """Check active commodity instrument and contract details."""
    try:
        from utils.instrument_selector import get_instrument

        cfg = get_instrument(symbol)
        return {
            "symbol": cfg.name,
            "lot_size": cfg.lot_size,
            "tick_size": cfg.tick_size,
            "point_value": cfg.point_value,
            "strike_step": cfg.strike_step,
            "product_type": cfg.product_type,
            "trading_hours": "09:00 - 23:30 IST",
        }
    except Exception as exc:
        return {"symbol": symbol, "error": str(exc)}


def determine_posture(checks: dict) -> str:
    """Combine all checks into final trading posture for the day."""
    postures = [
        checks["gap"]["posture"],
        checks["equity"]["posture"],
    ]
    if "OBSERVE" in postures:
        return "OBSERVE"
    if "REDUCE_HALF" in postures:
        return "REDUCE_HALF"
    if not checks["token"].get("valid", True):
        return "OBSERVE"
    return "FULL_SIZE"


def format_report(checks: dict, posture: str, symbol: str) -> str:
    """Build a formatted console report."""
    now = datetime.now(IST).strftime("%d %b %Y  %H:%M IST")
    gap = checks["gap"]
    eq = checks["equity"]
    tok = checks["token"]
    contract = checks["contract"]

    gap_col = Y if gap["level"] == "LARGE_GAP" else G
    eq_col = R if eq["signal"] == "PAUSE" else Y if eq["signal"] == "REDUCE_SIZE" else G
    tok_col = G if tok.get("valid") else R
    pos_col = G if posture == "FULL_SIZE" else Y if posture == "REDUCE_HALF" else R

    lines = [
        f"\n{B}{'═'*60}{RST}",
        f"{B}  MCXForge — Pre-Market Checklist  |  {now}{RST}",
        f"{B}{'═'*60}{RST}\n",
        f"  {W}Active Commodity:{RST}",
        f"    Symbol:      {Y}{contract.get('symbol', symbol)}{RST}",
        f"    Lot Size:    {contract.get('lot_size', '—')}",
        f"    Hours:       {contract.get('trading_hours', '09:00 - 23:30 IST')}\n",
        f"  {W}Gap Check:{RST}",
        f"    Gap:         {gap_col}{gap['gap_pct']:+.2f}%  ({gap['level']} {gap['direction']}){RST}",
        f"    Posture:     {gap_col}{gap['posture']}{RST}\n",
        f"  {W}Equity Curve:{RST}",
        f"    Signal:      {eq_col}{eq['signal']}{RST}",
        f"    Streak:      {eq['streak'].get('streak','—')} × {eq['streak'].get('count',0)} days",
        f"    Posture:     {eq_col}{eq['posture']}{RST}\n",
        f"  {W}Broker Token & Quote:{RST}",
        f"    Status:      {tok_col}{'✅ VALID' if tok.get('valid') else '❌ INVALID — check broker auth'}{RST}",
    ]
    if tok.get("ltp"):
        lines.append(f"    {symbol} LTP:   ₹{tok['ltp']:,.2f}")
    lines += [
        f"    Broker:      {tok.get('broker','—')}\n",
        f"{'─'*60}",
        f"  {pos_col}{W}TODAY'S POSTURE:  {posture}{RST}",
        f"{'─'*60}",
    ]

    if posture == "OBSERVE":
        lines += [
            f"  {R}⛔ OBSERVE Mode: Paper simulation active.{RST}",
            f"  {R}   Zero live orders placed to broker.{RST}",
        ]
    elif posture == "REDUCE_HALF":
        lines += [
            f"  {Y}⚠️  Trade with 50% of normal position size today.{RST}",
            f"  {Y}   Monitor closely — conditions elevated but tradeable.{RST}",
        ]
    else:
        lines += [
            f"  {G}✅ Normal trading day. Standard position size allowed.{RST}",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print(f"\n{B}Running MCXForge pre-market checks...{RST}")
    active_sym = os.getenv("INSTRUMENT", "SILVERM").strip().upper()

    checks = {
        "gap": check_gap(active_sym),
        "equity": check_equity_curve(),
        "token": check_broker_token(active_sym),
        "contract": check_commodity_contract(active_sym),
    }

    posture = determine_posture(checks)

    # Print report
    print(format_report(checks, posture, active_sym))

    # Save JSON report
    today = date.today().isoformat()
    report = {
        "date": today,
        "time": datetime.now(IST).isoformat(),
        "posture": posture,
        "instrument": active_sym,
        "checks": checks,
    }
    json_path = REPORT_PATH / f"premarket_{today}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved: {json_path}")

    # Write posture to env file for session use
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        content = env_path.read_text()
        updates = {
            "TRADING_POSTURE": posture,
        }
        for key, value in updates.items():
            if re.search(rf"^{key}=", content, flags=re.MULTILINE):
                content = re.sub(rf"^{key}=.*$", f"{key}={value}", content, flags=re.MULTILINE)
            else:
                content += f"\n{key}={value}\n"
        env_path.write_text(content)
        print(f"  .env updated: TRADING_POSTURE={posture}")

    # Send Telegram
    if TELEGRAM_ENABLED:
        try:
            import asyncio
            from utils.telegram_notifier import get_notifier

            notifier = get_notifier()
            payload = {
                "bias": checks["gap"]["direction"],
                "gap_pct": checks["gap"]["gap_pct"],
                "posture": posture,
                "instrument": active_sym,
            }
            asyncio.run(notifier.send_premarket_brief(payload))
            print("  Telegram: morning brief sent ✅")
        except Exception as e:
            print(f"  Telegram: failed ({e})")

    # Exit code signals posture for shell scripts
    exit_codes = {"FULL_SIZE": 0, "REDUCE_HALF": 1, "OBSERVE": 2}
    sys.exit(exit_codes.get(posture, 0))


if __name__ == "__main__":
    main()

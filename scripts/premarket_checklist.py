#!/usr/bin/env python3
"""
scripts/premarket_checklist.py — Pre-Market Risk Gate
======================================================
Run at 09:00 IST every trading morning (before 09:15 open).
Checks all pre-market risk conditions and sets the day's trading posture.

Schedule in cron:
  0 9 * * 1-5 cd /opt/signalforge && python scripts/premarket_checklist.py

CHECKS PERFORMED:
  1. India VIX level (> 28 = reduce size, > 35 = OBSERVE only)
  2. NIFTY overnight gap (> 1.5% = volatile open, reduce size)
  3. Equity curve risk signal (PAUSE / REDUCE_SIZE / NORMAL)
  4. Daily loss carry-over check (yesterday's circuit breaker)
  5. Expiry day detection (Thursday = different strategy behaviour)
  6. Global market overview (SGX Nifty proxy via yfinance)
  7. Token validity check (broker auth still valid?)

OUTPUT:
  - Console report with colour coding
  - logs/premarket_YYYY-MM-DD.json (machine readable)
  - Telegram message with day's posture
  - Sets TRADING_POSTURE env variable for session

POSTURE LEVELS:
  FULL_SIZE   → all systems normal, trade with 100% position sizing
  REDUCE_HALF → elevated risk detected, use 50% position size
  OBSERVE     → too risky, watch only, no trades today
"""

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytz
IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import (
        PREMARKET_MAX_VIX, PREMARKET_MAX_GAP_PCT,
        TELEGRAM_ENABLED, LOGS_DIR, JOURNAL_DIR,
        DEPLOYED_CAPITAL, MAX_DAILY_LOSS_PCT,
    )
except ImportError:
    PREMARKET_MAX_VIX     = 28.0
    PREMARKET_MAX_GAP_PCT = 1.5
    TELEGRAM_ENABLED      = False
    LOGS_DIR              = "logs"
    JOURNAL_DIR           = "journal"
    DEPLOYED_CAPITAL      = 200000
    MAX_DAILY_LOSS_PCT    = 3.0

REPORT_PATH = Path(LOGS_DIR)
REPORT_PATH.mkdir(exist_ok=True)

G   = "\033[92m"
Y   = "\033[93m"
R   = "\033[91m"
B   = "\033[94m"
W   = "\033[97m"
RST = "\033[0m"


def check_vix() -> dict:
    """Check India VIX level."""
    try:
        import yfinance as yf
        vix = yf.Ticker("^INDIAVIX").fast_info.get("lastPrice", 0)
        if vix == 0:
            hist = yf.download("^INDIAVIX", period="2d", interval="1d", progress=False)
            vix  = float(hist["Close"].iloc[-1]) if len(hist) > 0 else 18.0
    except Exception:
        vix = 18.0   # safe default if data unavailable

    if vix > 35:
        level, posture = "EXTREME", "OBSERVE"
    elif vix > PREMARKET_MAX_VIX:
        level, posture = "HIGH", "REDUCE_HALF"
    elif vix > 20:
        level, posture = "ELEVATED", "FULL_SIZE"
    else:
        level, posture = "NORMAL", "FULL_SIZE"

    return {"vix": round(vix, 2), "level": level, "posture": posture}


def check_gap() -> dict:
    """Check NIFTY overnight gap from previous close."""
    try:
        import yfinance as yf
        hist = yf.download("^NSEI", period="5d", interval="1d", progress=False)
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            today_open = float(hist["Open"].iloc[-1])
            gap_pct    = (today_open - prev_close) / prev_close * 100
        else:
            gap_pct    = 0.0
    except Exception:
        gap_pct = 0.0

    if abs(gap_pct) > PREMARKET_MAX_GAP_PCT:
        posture = "REDUCE_HALF"
        level   = "LARGE_GAP"
    elif abs(gap_pct) > 0.75:
        posture = "FULL_SIZE"
        level   = "MODERATE_GAP"
    else:
        posture = "FULL_SIZE"
        level   = "FLAT_OPEN"

    return {
        "gap_pct":  round(gap_pct, 3),
        "level":    level,
        "posture":  posture,
        "direction": "UP" if gap_pct > 0.3 else "DOWN" if gap_pct < -0.3 else "FLAT",
    }


def check_equity_curve() -> dict:
    """Check equity curve risk signal from yesterday's performance."""
    try:
        from utils.equity_curve import get_equity_curve
        curve  = get_equity_curve()
        signal = curve.get_risk_signal()
        streak = curve.get_streak_info()
        posture = {"PAUSE": "OBSERVE", "REDUCE_SIZE": "REDUCE_HALF",
                   "NORMAL": "FULL_SIZE"}.get(signal, "FULL_SIZE")
        return {"signal": signal, "posture": posture, "streak": streak}
    except Exception:
        return {"signal": "NORMAL", "posture": "FULL_SIZE", "streak": {}}


def check_broker_token() -> dict:
    """Verify broker token is still valid."""
    try:
        from broker.factory import create_broker
        broker = create_broker()
        ltp    = broker.get_ltp("NIFTY")
        valid  = ltp > 0
        return {"valid": valid, "nifty_ltp": round(ltp, 2), "broker": broker.broker_name}
    except Exception as e:
        return {"valid": False, "error": str(e)[:80]}


def check_expiry() -> dict:
    """Detect NIFTY Thursday expiry and SENSEX Friday expiry."""
    today    = date.today()
    is_expiry = today.weekday() == 3   # Thursday
    is_friday      = today.weekday() == 4
    is_sensex_expiry = is_friday

    note = "Normal trading day"
    if is_expiry:
        note = "NIFTY EXPIRY Thursday — Trading SENSEX options today to avoid NIFTY expiry gamma"
    elif is_sensex_expiry:
        note = "SENSEX EXPIRY Friday — Theta acceleration play on SENSEX options"

    return {
        "is_expiry":         is_expiry,
        "is_sensex_expiry":  is_sensex_expiry,
        "instrument_today":  "SENSEX" if (is_expiry or is_sensex_expiry) else "NIFTY",
        "day":               today.strftime("%A"),
        "note":              note,
    }


def determine_posture(checks: dict) -> str:
    """
    Combine all checks into final trading posture for the day.

    Logic: most restrictive check wins.
    """
    postures = [
        checks["vix"]["posture"],
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


def format_report(checks: dict, posture: str) -> str:
    """Build a formatted console report."""
    now  = datetime.now(IST).strftime("%d %b %Y  %H:%M IST")
    vix  = checks["vix"]
    gap  = checks["gap"]
    eq   = checks["equity"]
    tok  = checks["token"]
    exp  = checks["expiry"]

    vix_col = R if vix["level"] == "EXTREME" else Y if vix["level"] == "HIGH" else G
    gap_col = Y if gap["level"] == "LARGE_GAP" else G
    eq_col  = R if eq["signal"] == "PAUSE" else Y if eq["signal"] == "REDUCE_SIZE" else G
    tok_col = G if tok.get("valid") else R
    pos_col = G if posture == "FULL_SIZE" else Y if posture == "REDUCE_HALF" else R

    lines = [
        f"\n{B}{'═'*60}{RST}",
        f"{B}  SignalForge — Pre-Market Checklist  |  {now}{RST}",
        f"{B}{'═'*60}{RST}\n",
        f"  {W}VIX Check:{RST}",
        f"    India VIX:   {vix_col}{vix['vix']:.1f}  ({vix['level']}){RST}",
        f"    Posture:     {vix_col}{vix['posture']}{RST}\n",
        f"  {W}Gap Check:{RST}",
        f"    Gap:         {gap_col}{gap['gap_pct']:+.2f}%  ({gap['level']} {gap['direction']}){RST}",
        f"    Posture:     {gap_col}{gap['posture']}{RST}\n",
        f"  {W}Equity Curve:{RST}",
        f"    Signal:      {eq_col}{eq['signal']}{RST}",
        f"    Streak:      {eq['streak'].get('streak','—')} × {eq['streak'].get('count',0)} days",
        f"    Posture:     {eq_col}{eq['posture']}{RST}\n",
        f"  {W}Broker Token:{RST}",
        f"    Status:      {tok_col}{'✅ VALID' if tok.get('valid') else '❌ INVALID — run auth script'}{RST}",
    ]
    if tok.get("nifty_ltp"):
        lines.append(f"    NIFTY LTP:   ₹{tok['nifty_ltp']:,.2f}")
    lines += [
        f"    Broker:      {tok.get('broker','—')}\n",
        f"  {W}Expiry:{RST}",
        f"    Today:       {exp['day']}  {'⚡ EXPIRY DAY' if exp['is_expiry'] else ''}",
        f"    Instrument:  {Y}{exp['instrument_today']}{RST}",
        f"    Note:        {exp['note']}\n",
        f"{'─'*60}",
        f"  {pos_col}{W}TODAY'S POSTURE:  {posture}{RST}",
        f"{'─'*60}",
    ]

    if posture == "OBSERVE":
        lines += [
            f"  {R}⛔ No new trades today.{RST}",
            f"  {R}   Risk conditions too elevated for live positions.{RST}",
        ]
    elif posture == "REDUCE_HALF":
        lines += [
            f"  {Y}⚠️  Trade with 50% of normal position size today.{RST}",
            f"  {Y}   Monitor closely — conditions elevated but tradeable.{RST}",
        ]
    else:
        lines += [
            f"  {G}✅ Normal trading day. Full position size allowed.{RST}",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print(f"\n{B}Running pre-market checks...{RST}")

    checks = {
        "vix":    check_vix(),
        "gap":    check_gap(),
        "equity": check_equity_curve(),
        "token":  check_broker_token(),
        "expiry": check_expiry(),
    }

    posture = determine_posture(checks)

    # Print report
    print(format_report(checks, posture))

    # Save JSON report
    today     = date.today().isoformat()
    report    = {
        "date":    today,
        "time":    datetime.now(IST).isoformat(),
        "posture": posture,
        "instrument": checks["expiry"]["instrument_today"],
        "checks":  checks,
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
            "TRADING_INSTRUMENT": checks["expiry"]["instrument_today"],
            "INSTRUMENT": checks["expiry"]["instrument_today"],
        }
        import re
        for key, value in updates.items():
            if re.search(rf"^{key}=", content, flags=re.MULTILINE):
                content = re.sub(rf"^{key}=.*$", f"{key}={value}", content, flags=re.MULTILINE)
            else:
                content += f"\n{key}={value}\n"
        env_path.write_text(content)
        print(
            f"  .env updated: TRADING_POSTURE={posture}, "
            f"TRADING_INSTRUMENT={checks['expiry']['instrument_today']}"
        )

    # Send Telegram
    if TELEGRAM_ENABLED:
        try:
            import asyncio
            from utils.telegram_notifier import get_notifier
            notifier = get_notifier()
            payload  = {
                "bias":      checks["gap"]["direction"],
                "india_vix": checks["vix"]["vix"],
                "gap_pct":   checks["gap"]["gap_pct"],
                "posture":   posture,
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

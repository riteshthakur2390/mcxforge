#!/usr/bin/env python3
"""
Run a cron command and send a Telegram completion notification.

The wrapped command's stdout/stderr are passed through unchanged so existing
cron log redirection keeps working. The wrapper exits with the same return code
as the command it ran.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import request

import pytz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=True)
except Exception:
    pass


IST = pytz.timezone("Asia/Kolkata")


def _notify(message: str, prefer_backtest: bool = False) -> bool:
    if prefer_backtest:
        token = os.getenv("BACKTEST_TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("BACKTEST_TELEGRAM_CHAT_ID", "").strip()
        token = token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    else:
        token = os.getenv("LIVE_TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("LIVE_TELEGRAM_CHAT_ID", "").strip()
        token = token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("WARN telegram notification skipped: telegram token/chat id missing", file=sys.stderr)
        return False

    payload = json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8")
    req = request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=8) as response:
            body = response.read().decode("utf-8", errors="replace")
            if 200 <= int(response.status) < 300:
                print("Telegram notification sent")
                return True
            print(f"WARN telegram notification HTTP {response.status}: {body[:300]}", file=sys.stderr)
    except Exception as exc:
        print(f"WARN telegram notification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return False


def _format_duration(seconds: float) -> str:
    seconds = max(float(seconds or 0.0), 0.0)
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a cron command and send Telegram completion status.")
    parser.add_argument("--name", required=True, help="Human-readable cron job name")
    parser.add_argument("--cwd", default=str(PROJECT_ROOT), help="Working directory for the command")
    parser.add_argument(
        "--telegram-target",
        choices=("auto", "live", "backtest"),
        default="auto",
        help="Telegram route for completion notification. Default: live unless job/command is a backtest.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute after --")
    args = parser.parse_args()

    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("FAIL missing command after --", file=sys.stderr)
        return 2

    started = time.monotonic()
    started_at = datetime.now(IST)
    result = subprocess.run(command, cwd=args.cwd, env=os.environ.copy())
    duration = time.monotonic() - started

    ok = result.returncode == 0
    status = "OK" if ok else "FAIL"
    message = (
        f"SignalForge Cron {status}\n"
        f"Job: {args.name}\n"
        f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S IST')}\n"
        f"Duration: {_format_duration(duration)}\n"
        f"Exit: {result.returncode}"
    )
    command_text = " ".join(command).lower()
    if args.telegram_target == "backtest":
        prefer_backtest = True
    elif args.telegram_target == "live":
        prefer_backtest = False
    else:
        prefer_backtest = "backtest" in args.name.lower() or "backtest" in command_text
    _notify(message, prefer_backtest=prefer_backtest)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

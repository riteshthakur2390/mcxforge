#!/usr/bin/env python3
"""
Compare settings snapshots across backtest logs.

Usage:
  python3 scripts/compare_backtest_settings.py logs/backtest_20260608_*.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SETTINGS_RE = re.compile(r"\[BACKTEST_SETTINGS\]\s+(\{.*\})")


def _load_snapshot(path: Path) -> dict[str, Any]:
    for line in path.read_text(errors="ignore").splitlines():
        match = SETTINGS_RE.search(line)
        if not match:
            continue
        payload = json.loads(match.group(1))
        settings = payload.get("settings", {}) or {}
        settings["_settings_hash"] = payload.get("settings_hash", "")
        return settings
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="backtest_*.log files")
    args = parser.parse_args()

    rows = []
    for raw in args.logs:
        path = Path(raw)
        settings = _load_snapshot(path)
        if not settings:
            print(f"{path}: no [BACKTEST_SETTINGS] snapshot found")
            continue
        rows.append((path.name, settings))

    if not rows:
        return 1

    keys = sorted({key for _, settings in rows for key in settings})
    print("setting," + ",".join(name for name, _ in rows))
    for key in keys:
        values = [settings.get(key, "") for _, settings in rows]
        if len({json.dumps(value, sort_keys=True, default=str) for value in values}) == 1:
            continue
        print(key + "," + ",".join(json.dumps(value, default=str) for value in values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

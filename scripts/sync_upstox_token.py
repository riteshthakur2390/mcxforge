#!/usr/bin/env python3
"""
scripts/sync_upstox_token.py — Upstox Access Token Cross-Project Synchronizer
=============================================================================
Syncs UPSTOX_ACCESS_TOKEN (and API keys) from SignalForge's .env to all sibling
forge projects (ExpiryForge, SwingForge, ThetaForge).

Usage:
    python scripts/sync_upstox_token.py
    python scripts/sync_upstox_token.py --source-env /path/to/.env --validate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from dotenv import dotenv_values, load_dotenv, set_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCE_ENV = PROJECT_ROOT / ".env"
PROJECTS_DIR = Path("/Users/vishranti/Downloads/projects")

SIBLING_PROJECTS = [
    PROJECTS_DIR / "signalforge",
    PROJECTS_DIR / "expiryforge",
    PROJECTS_DIR / "swingforge",
    PROJECTS_DIR / "thetaforge",
]


def _mask(value: str) -> str:
    if not value or len(value) <= 12:
        return "(short/empty token)"
    return f"{value[:8]}...{value[-4:]}"


def validate_upstox_token(token: str) -> tuple[bool, str]:
    """Test token validity with live Upstox LTP request."""
    if not token or token == "your_upstox_access_token_here":
        return False, "Token is empty or placeholder"

    try:
        resp = requests.get(
            "https://api.upstox.com/v2/market-quote/ltp",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
            },
            params={"instrument_key": "NSE_INDEX|Nifty 50"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            ltp = (
                data.get("data", {})
                .get("NSE_INDEX:Nifty 50", {})
                .get("last_price")
            )
            return True, f"Valid (NIFTY LTP: {ltp})"
        else:
            return False, f"HTTP {resp.status_code}: {resp.text[:120]}"
    except Exception as exc:
        return False, f"Network error: {exc}"


def sync_upstox_token(
    source_env_path: Path = DEFAULT_SOURCE_ENV,
    target_projects: list[Path] | None = None,
    validate: bool = True,
    notify: bool = False,
) -> dict[str, bool]:
    """
    Copy UPSTOX_ACCESS_TOKEN and API keys from source_env_path to target projects.
    """
    if not source_env_path.exists():
        print(f"❌ Source .env not found: {source_env_path}")
        return {}

    source_vals = dotenv_values(source_env_path)
    access_token = source_vals.get("UPSTOX_ACCESS_TOKEN", "").strip()
    api_key = source_vals.get("UPSTOX_API_KEY", "").strip()
    api_secret = source_vals.get("UPSTOX_API_SECRET", "").strip()

    if not access_token or access_token == "your_upstox_access_token_here":
        print(f"❌ UPSTOX_ACCESS_TOKEN in {source_env_path} is missing or placeholder.")
        return {}

    print("=" * 65)
    print("      UPSTOX ACCESS TOKEN CROSS-PROJECT SYNCHRONIZER")
    print("=" * 65)
    print(f"Source Env   : {source_env_path}")
    print(f"Access Token : {_mask(access_token)}")
    if api_key:
        print(f"API Key      : {_mask(api_key)}")

    if validate:
        print("\nVerifying token with live Upstox API probe...")
        is_valid, reason = validate_upstox_token(access_token)
        if is_valid:
            print(f"  ✅ Token Status: {reason}")
        else:
            print(f"  ⚠️ Warning: Token validation check returned: {reason}")

    targets = target_projects if target_projects is not None else SIBLING_PROJECTS
    results: dict[str, bool] = {}

    print(f"\nSynchronizing to {len(targets)} sibling projects:")
    for proj_dir in targets:
        env_file = proj_dir / ".env"
        if not env_file.exists():
            print(f"  ⏭️  Skipped {proj_dir.name} (.env does not exist)")
            results[proj_dir.name] = False
            continue

        try:
            set_key(str(env_file), "UPSTOX_ACCESS_TOKEN", access_token)
            if api_key:
                set_key(str(env_file), "UPSTOX_API_KEY", api_key)
            if api_secret:
                set_key(str(env_file), "UPSTOX_API_SECRET", api_secret)

            print(f"  ✅ Synced -> {proj_dir.name:15s} ({env_file})")
            results[proj_dir.name] = True
        except Exception as exc:
            print(f"  ❌ Failed -> {proj_dir.name:15s}: {exc}")
            results[proj_dir.name] = False

    synced_count = sum(1 for v in results.values() if v)
    print("\n" + "-" * 65)
    print(f"Summary: {synced_count}/{len(targets)} projects successfully updated.")
    print("-" * 65)

    if notify:
        _send_telegram_notification(synced_count, len(targets), access_token)

    return results


def _send_telegram_notification(synced_count: int, total: int, token: str) -> None:
    try:
        from utils.telegram_notifier import get_notifier
        import asyncio

        msg = (
            f"🔐 *Upstox Token Sync Completed*\n\n"
            f"• Source: SignalForge\n"
            f"• Token: `{_mask(token)}`\n"
            f"• Synced Projects: {synced_count}/{total}\n"
            f"• Targets: ExpiryForge, SwingForge, ThetaForge"
        )
        asyncio.run(get_notifier().send_text(msg, target="LIVE", parse_mode="Markdown"))
    except Exception as exc:
        print(f"  (Telegram notice skipped: {exc})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize Upstox token across Forge projects")
    parser.add_argument("--source-env", default=str(DEFAULT_SOURCE_ENV), help="Source .env file path")
    parser.add_argument("--no-validate", action="store_true", help="Skip live Upstox API check")
    parser.add_argument("--notify", action="store_true", help="Send Telegram notification")

    args = parser.parse_args()
    results = sync_upstox_token(
        source_env_path=Path(args.source_env),
        validate=not args.no_validate,
        notify=args.notify,
    )
    return 0 if any(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

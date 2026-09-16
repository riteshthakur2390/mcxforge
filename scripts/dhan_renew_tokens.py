#!/usr/bin/env python3
"""
Renew MCXForge Dhan access tokens before market open.

Validates and renews MCXForge's local .env tokens.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values, load_dotenv, set_key


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_PATH = PROJECT_ROOT / ".env"
DHAN_BASE_URL = "https://api.dhan.co"


@dataclass(frozen=True)
class DhanAccount:
    suffix: str
    client_id_key: str
    token_key: str
    client_id: str
    access_token: str

    @property
    def label(self) -> str:
        return "primary" if not self.suffix else f"account_{self.suffix}"


def _mask(value: str) -> str:
    if len(value) <= 12:
        return "(short token)"
    return f"{value[:8]}...{value[-4:]}"


def _headers(account: DhanAccount, token: str | None = None) -> dict[str, str]:
    return {
        "access-token": token or account.access_token,
        "client-id": account.client_id,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _load_accounts(scan_limit: int) -> list[DhanAccount]:
    from dotenv import dotenv_values
    env_vals = dotenv_values(ENV_PATH)
    accounts: list[DhanAccount] = []
    for idx in range(1, scan_limit + 1):
        suffix = "" if idx == 1 else str(idx)
        client_id_key = "DHAN_CLIENT_ID" if idx == 1 else f"DHAN_CLIENT_ID_{idx}"
        token_key = "DHAN_ACCESS_TOKEN" if idx == 1 else f"DHAN_ACCESS_TOKEN_{idx}"
        client_id = (env_vals.get(client_id_key) or os.getenv(client_id_key, "")).strip()
        token = (env_vals.get(token_key) or os.getenv(token_key, "")).strip()
        if not client_id or not token or any(a.client_id == client_id for a in accounts):
            continue
        accounts.append(DhanAccount(suffix, client_id_key, token_key, client_id, token))
    return accounts


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return {"raw": response.text[:500]}


def _verify(account: DhanAccount, token: str | None = None, timeout: float = 12.0) -> tuple[bool, str]:
    try:
        response = requests.get(f"{DHAN_BASE_URL}/v2/fundlimit", headers=_headers(account, token), timeout=timeout)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if response.status_code == 200:
        return True, "fundlimit OK"
    data = _response_json(response)
    detail = data.get("remarks") or data.get("message") or data.get("errorMessage") or data
    return False, f"HTTP {response.status_code}: {detail}"


def _extract_token(data: dict[str, Any]) -> str:
    candidates = [
        data.get("accessToken"),
        data.get("access_token"),
        data.get("token"),
        data.get("renewedToken"),
        data.get("renewed_token"),
    ]
    nested = data.get("data")
    if isinstance(nested, dict):
        candidates.extend([
            nested.get("accessToken"),
            nested.get("access_token"),
            nested.get("token"),
            nested.get("renewedToken"),
            nested.get("renewed_token"),
        ])
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _renew(account: DhanAccount, timeout: float) -> tuple[bool, str, str]:
    try:
        response = requests.get(
            f"{DHAN_BASE_URL}/v2/RenewToken",
            headers={
                "access-token": account.access_token,
                "dhanClientId": account.client_id,
                "Accept": "application/json",
            },
            timeout=timeout,
        )
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"
    data = _response_json(response)
    if response.status_code not in {200, 201}:
        detail = data.get("remarks") or data.get("message") or data.get("errorMessage") or data
        return False, "", f"HTTP {response.status_code}: {detail}"
    renewed = _extract_token(data)
    if not renewed:
        return False, "", f"renewal response missing token: {data}"
    return True, renewed, "renewed"


def _send_telegram(message: str) -> None:
    try:
        from utils.telegram_notifier import get_notifier
        import asyncio
        asyncio.run(get_notifier().send_text(message, target="LIVE", parse_mode=None))
    except Exception as exc:
        print(f"WARN telegram notification failed: {type(exc).__name__}: {exc}")


SIBLING_PROJECTS = [
    Path("/Users/vishranti/Downloads/projects/thetaforge"),
    Path("/Users/vishranti/Downloads/projects/expiryforge"),
    Path("/Users/vishranti/Downloads/projects/swingforge"),
]


def _sync_to_sibling_projects(token_key: str, client_id_key: str, token_val: str, client_id_val: str) -> list[str]:
    from dotenv import dotenv_values
    synced: list[str] = []
    for proj_dir in SIBLING_PROJECTS:
        env_file = proj_dir / ".env"
        if env_file.exists():
            try:
                set_key(str(env_file), client_id_key, client_id_val)
                set_key(str(env_file), token_key, token_val)
                env_vals = dotenv_values(env_file)
                if "DHAN_ACCESS_TOKEN_2" in env_vals or "DHAN_CLIENT_ID_2" in env_vals:
                    set_key(str(env_file), "DHAN_CLIENT_ID_2", client_id_val)
                    set_key(str(env_file), "DHAN_ACCESS_TOKEN_2", token_val)
                synced.append(proj_dir.name)
            except Exception as exc:
                print(f"  WARN failed to sync to {proj_dir.name}: {exc}")
    return synced


def main() -> int:
    global ENV_PATH

    parser = argparse.ArgumentParser(description="Renew MCXForge Dhan token(s)")
    parser.add_argument("--env", default=str(ENV_PATH))
    parser.add_argument("--scan-limit", type=int, default=int(os.getenv("DHAN_PARALLEL_ACCOUNT_SCAN_LIMIT", "3") or "3"))
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--telegram-always", action="store_true")
    args = parser.parse_args()

    ENV_PATH = Path(args.env).expanduser().resolve()
    if not ENV_PATH.exists():
        print(f"FAIL .env not found: {ENV_PATH}")
        return 2

    accounts = _load_accounts(max(1, args.scan_limit))
    if not accounts:
        print("FAIL no DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN accounts found")
        return 2

    failures: list[str] = []
    changed: list[str] = []
    verified: list[str] = []

    print(f"MCXForge Dhan Token Renewal | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"env: {ENV_PATH}")
    print(f"accounts: {len(accounts)}\n")

    for account in accounts:
        print(f"Account {account.label} | client_id={account.client_id} | token={_mask(account.access_token)}")
        ok, detail = _verify(account, timeout=args.timeout)

        if not ok:
            print(f"  WARN {account.label}: current token pre-verification failed ({detail}) — attempting direct renewal...")

        if args.verify_only:
            if ok:
                verified.append(account.label)
            else:
                failures.append(f"{account.label}: current token invalid: {detail}")
            continue

        renewed_ok, renewed_token, renew_detail = _renew(account, timeout=args.timeout)
        if not renewed_ok:
            msg = f"{account.label}: RenewToken failed ({renew_detail}). Manual Dhan Web token generation required."
            print(f"  FAIL {msg}")
            failures.append(msg)
            continue
        ok, detail = _verify(account, token=renewed_token, timeout=args.timeout)
        if not ok:
            msg = f"{account.label}: renewed token failed verification: {detail}"
            print(f"  FAIL {msg}")
            failures.append(msg)
            continue
        if args.dry_run:
            print(f"  OK renewed token verified, dry-run not writing: {_mask(renewed_token)}")
        else:
            set_key(str(ENV_PATH), account.token_key, renewed_token)
            print(f"  OK renewed token saved to {account.token_key} in {ENV_PATH.name}: {_mask(renewed_token)}")
            # Sync only to DHAN_ACCESS_TOKEN_2 / DHAN_CLIENT_ID_2 if configured in .env
            env_vals = dotenv_values(ENV_PATH)
            if "DHAN_ACCESS_TOKEN_2" in env_vals or "DHAN_CLIENT_ID_2" in env_vals:
                set_key(str(ENV_PATH), "DHAN_CLIENT_ID_2", account.client_id)
                set_key(str(ENV_PATH), "DHAN_ACCESS_TOKEN_2", renewed_token)
                print("  OK synced renewed token to secondary key DHAN_ACCESS_TOKEN_2")
            synced_projects = _sync_to_sibling_projects(account.token_key, account.client_id_key, renewed_token, account.client_id)
            if synced_projects:
                print(f"  OK synced {account.token_key} to sibling projects: {', '.join(synced_projects)}")
        changed.append(account.label)

    print("\nSummary")
    print(f"  verified_only: {len(verified)}")
    print(f"  renewed      : {len(changed)}")
    print(f"  failed       : {len(failures)}")
    for failure in failures:
        print(f"  - {failure}")

    if args.telegram and (failures or args.telegram_always):
        status = "ALERT" if failures else "OK"
        lines = [
            f"MCXForge Dhan Token Renewal {status}",
            f"accounts={len(accounts)} renewed={len(changed)} verified_only={len(verified)} failed={len(failures)}",
        ]
        if failures:
            lines.extend(f"- {failure}" for failure in failures[:8])
        _send_telegram("\n".join(lines))

    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

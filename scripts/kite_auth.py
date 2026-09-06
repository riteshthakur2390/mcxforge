#!/usr/bin/env python3
"""
scripts/kite_auth.py — Daily Authentication Script
====================================================
Handles auth for both Kite and Upstox.
Reads BROKER from .env to decide which flow to run.

Usage:
    python scripts/kite_auth.py          # uses BROKER from .env
    BROKER=upstox python scripts/kite_auth.py
    BROKER=kite   python scripts/kite_auth.py
"""

import os
import sys
import webbrowser
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv, set_key
except ImportError:
    print("❌ python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

# ── Load env ──────────────────────────────────────────────────
ENV_PATH = Path(__file__).parent.parent / ".env"
if not ENV_PATH.exists():
    print(f"❌ .env not found at {ENV_PATH}")
    print("   Run: cp .env.example .env  and fill in your credentials.")
    sys.exit(1)

load_dotenv(ENV_PATH)

BROKER = os.getenv("BROKER", "kite").lower()

import pytz
from datetime import datetime
IST = pytz.timezone("Asia/Kolkata")

print()
print("╔══════════════════════════════════════════════╗")
print(f"║   SignalForge — Daily Auth ({BROKER.upper():^8})         ║")
print(f"║   {datetime.now(IST).strftime('%Y-%m-%d  %H:%M IST')}                      ║")
print("╚══════════════════════════════════════════════╝")
print()


# ════════════════════════════════════════════════════
# KITE AUTH
# ════════════════════════════════════════════════════
def run_kite_auth():
    api_key    = os.getenv("KITE_API_KEY", "")
    api_secret = os.getenv("KITE_API_SECRET", "")

    if not api_key or api_key == "your_api_key_here":
        print("❌ KITE_API_KEY not set in .env")
        sys.exit(1)
    if not api_secret or api_secret == "your_api_secret_here":
        print("❌ KITE_API_SECRET not set in .env")
        sys.exit(1)

    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("❌ kiteconnect not installed. Run: pip install kiteconnect")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    url  = kite.login_url()

    print("Step 1: Opening Kite login in your browser...")
    print(f"        {url}")
    print()
    webbrowser.open(url)

    print("Step 2: Login with Zerodha credentials + TOTP/PIN")
    print()
    print("Step 3: After login, browser redirects to a URL like:")
    print("        http://127.0.0.1:8080/?...&request_token=XXXX&checksum=XXX")
    print()
    print("        ⚠️  Browser will say 'site can't be reached' — that is NORMAL")
    print("        Just copy the request_token value from the URL bar")
    print()

    token = input("Paste request_token here: ").strip()

    # Handle if user pastes full URL
    if "request_token=" in token:
        import urllib.parse as up
        qs    = up.parse_qs(up.urlparse(token).query)
        token = qs.get("request_token", [""])[0]
        print(f"  ✅ Extracted token: {token[:8]}...")

    if not token or len(token) < 10:
        print("❌ Invalid token. Please try again.")
        sys.exit(1)

    print("\nGenerating session...")
    try:
        data = kite.generate_session(token, api_secret=api_secret)
        access_token = data["access_token"]
        print(f"✅ Authenticated as: {data.get('user_name')} ({data.get('user_id')})")
    except Exception as e:
        print(f"❌ Session failed: {e}")
        print("   Common causes: wrong token, already used, or API secret wrong.")
        sys.exit(1)

    set_key(str(ENV_PATH), "KITE_ACCESS_TOKEN", access_token)
    print("✅ KITE_ACCESS_TOKEN saved to .env")
    return access_token


# ════════════════════════════════════════════════════
# UPSTOX AUTH
# ════════════════════════════════════════════════════
def run_upstox_auth():
    api_key    = os.getenv("UPSTOX_API_KEY", "")
    api_secret = os.getenv("UPSTOX_API_SECRET", "")

    if not api_key or api_key == "your_upstox_api_key_here":
        print("❌ UPSTOX_API_KEY not set in .env")
        print("   Get from: https://developer.upstox.com/")
        sys.exit(1)

    redirect_uri = "http://localhost:8080/"
    url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?client_id={api_key}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
    )

    print("Step 1: Opening Upstox login in your browser...")
    print(f"        {url}")
    print()
    webbrowser.open(url)

    print("Step 2: Login with your Upstox credentials")
    print()
    print("Step 3: After login, browser redirects to:")
    print("        http://localhost:8080/?code=XXXXXXXXXXXX")
    print()
    print("        ⚠️  Browser will say 'site can't be reached' — that is NORMAL")
    print("        Copy the 'code' value from the URL bar")
    print()

    auth_code = input("Paste the code from URL here: ").strip()

    # Handle if user pastes full URL
    if "code=" in auth_code:
        import urllib.parse as up
        qs        = up.parse_qs(up.urlparse(auth_code).query)
        auth_code = qs.get("code", [""])[0]
        print(f"  ✅ Extracted code: {auth_code[:8]}...")

    if not auth_code or len(auth_code) < 5:
        print("❌ Invalid code. Please try again.")
        sys.exit(1)

    print("\nExchanging code for access token...")
    try:
        import requests
        resp = requests.post(
            "https://api.upstox.com/v2/login/authorization/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "code":          auth_code,
                "client_id":     api_key,
                "client_secret": api_secret,
                "redirect_uri":  redirect_uri,
                "grant_type":    "authorization_code",
            },
        )
        data  = resp.json()
        token = data.get("access_token", "")
        if not token:
            print(f"❌ Auth failed: {data}")
            sys.exit(1)
        print(f"✅ Authenticated as: {data.get('user_name', 'Upstox User')}")
    except Exception as e:
        print(f"❌ Token exchange failed: {e}")
        sys.exit(1)

    set_key(str(ENV_PATH), "UPSTOX_ACCESS_TOKEN", token)
    print("✅ UPSTOX_ACCESS_TOKEN saved to .env")
    return token


# ════════════════════════════════════════════════════
# RESTART CONTAINER
# ════════════════════════════════════════════════════
def restart_container():
    print("\nChecking if MCXForge container is running...")
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=mcxforge", "--filter", "name=signalforge", "--format", "{{.Names}}"],
        capture_output=True, text=True,
    )
    if "mcxforge" in result.stdout or "signalforge" in result.stdout:
        print("  Container running — restarting with new token...")
        subprocess.run(
            ["docker", "compose", "restart"],
            cwd=str(ENV_PATH.parent),
            capture_output=True,
        )
        print("✅ Container restarted.")
    else:
        print("  Container not running.")
        print("  Start with: docker compose up -d")


# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════
if __name__ == "__main__":
    if BROKER == "upstox":
        run_upstox_auth()
    else:
        run_kite_auth()

    restart_container()

    print()
    print("━" * 50)
    print(f"  Ready for market!  |  Broker: {BROKER.upper()}")
    print(f"  {datetime.now(IST).strftime('%H:%M IST')}")
    print("  Dashboard: http://localhost:5050")
    print("━" * 50)
    print()

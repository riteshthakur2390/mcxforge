#!/usr/bin/env python3
"""
scripts/upstox_auth.py — Daily Upstox Authentication
======================================================
Run this every morning before 09:10 IST when using Upstox.

Usage:
    python scripts/upstox_auth.py

What it does:
    1. Opens Upstox OAuth login URL in your browser
    2. You login with Upstox credentials
    3. Browser redirects to localhost:8080/?code=XXXX
    4. You paste the code (or full URL) here
    5. Script exchanges code for access token
    6. Token saved to .env automatically
    7. Docker container restarted if running

Setup (one time only):
    1. Go to https://developer.upstox.com/
    2. Login → My Apps → Create New App
    3. App Name: SignalForge-Dev
    4. Redirect URL: http://localhost:8080/
    5. Copy API Key and API Secret into .env
"""

import os
import sys
import webbrowser
import subprocess
import requests
from pathlib import Path
from datetime import datetime

# ── Check dotenv installed ─────────────────────────────────────────────────
try:
    from dotenv import load_dotenv, set_key
except ImportError:
    print("\n❌ python-dotenv not installed.")
    print("   Run: pip install python-dotenv")
    sys.exit(1)

import pytz
IST = pytz.timezone("Asia/Kolkata")

# ── Load .env ──────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).parent.parent / ".env"
if not ENV_PATH.exists():
    print(f"\n❌ .env file not found at: {ENV_PATH}")
    print("   Run: cp .env.example .env  then fill in your Upstox credentials.")
    sys.exit(1)

load_dotenv(ENV_PATH)

API_KEY    = os.getenv("UPSTOX_API_KEY", "")
API_SECRET = os.getenv("UPSTOX_API_SECRET", "")

# ── Validate credentials set ───────────────────────────────────────────────
if not API_KEY or API_KEY == "your_upstox_api_key_here":
    print("\n❌ UPSTOX_API_KEY not set in .env")
    print("   Get from: https://developer.upstox.com/ → My Apps → Create App")
    print(f"   Then edit: {ENV_PATH}")
    sys.exit(1)

if not API_SECRET or API_SECRET == "your_upstox_api_secret_here":
    print("\n❌ UPSTOX_API_SECRET not set in .env")
    print(f"   Edit: {ENV_PATH}")
    sys.exit(1)

REDIRECT_URI = "http://localhost:8080/"

# ══════════════════════════════════════════════════════════════════════════
# MAIN AUTH FLOW
# ══════════════════════════════════════════════════════════════════════════

print()
print("╔════════════════════════════════════════════════╗")
print("║   MCXForge — Upstox Daily Auth                 ║")
print(f"║   {datetime.now(IST).strftime('%Y-%m-%d  %H:%M IST')}                        ║")
print("╚════════════════════════════════════════════════╝")
print()

# ── Step 1: Build login URL ────────────────────────────────────────────────
login_url = (
    f"https://api.upstox.com/v2/login/authorization/dialog"
    f"?client_id={API_KEY}"
    f"&redirect_uri={REDIRECT_URI}"
    f"&response_type=code"
)

print("━" * 52)
print("  STEP 1 — Login to Upstox")
print("━" * 52)
print()
print("  Opening login URL in your browser...")
print(f"  {login_url}")
print()
webbrowser.open(login_url)

# ── Step 2: Guide user ────────────────────────────────────────────────────
print("━" * 52)
print("  STEP 2 — Complete Login")
print("━" * 52)
print()
print("  1. Enter your Upstox mobile number + password")
print("  2. Enter OTP sent to your mobile")
print("  3. Enter your 6-digit PIN")
print()

# ── Step 3: Get auth code from redirect URL ───────────────────────────────
print("━" * 52)
print("  STEP 3 — Copy the Code from Redirect URL")
print("━" * 52)
print()
print("  After login, browser redirects to a URL like:")
print()
print("  http://localhost:8080/?code=AbCdEfGhIjKlMnOpQrSt123456")
print("                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print("                              copy this part only")
print()
print("  ⚠️  Browser will say 'site can't be reached' — that is NORMAL")
print("  Just look at the URL bar and copy the code after 'code='")
print()

raw_input = input("  Paste the code (or full URL) here: ").strip()

# ── Extract code if user pasted full URL ──────────────────────────────────
if not raw_input:
    print("\n❌ Nothing entered. Exiting.")
    sys.exit(1)

auth_code = raw_input
if "code=" in raw_input:
    import urllib.parse as up
    try:
        qs        = up.parse_qs(up.urlparse(raw_input).query)
        auth_code = qs.get("code", [""])[0]
        print(f"\n  ✅ Extracted code: {auth_code[:8]}...{auth_code[-4:]}")
    except Exception as parse_err:
        # AUTH LOG: URL parse failed — log what went wrong so user knows why
        print(f"\n  ⚠️  Could not parse URL as a redirect — using input as-is.")
        print(f"     Parse error: {parse_err}")
        print(f"     Input received: {raw_input[:80]}")

if len(auth_code) < 5:
    print("\n❌ Code too short — looks invalid. Please try again.")
    sys.exit(1)

# ── Step 4: Exchange code for access token ────────────────────────────────
print()
print("━" * 52)
print("  STEP 4 — Generating Access Token")
print("━" * 52)
print()
print("  Contacting Upstox API...")

try:
    resp = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept":        "application/json",
        },
        data={
            "code":          auth_code,
            "client_id":     API_KEY,
            "client_secret": API_SECRET,
            "redirect_uri":  REDIRECT_URI,
            "grant_type":    "authorization_code",
        },
        timeout=15,
    )
    data = resp.json()
except requests.exceptions.Timeout:
    # AUTH LOG: token exchange timed out — log URL and timeout value
    print("❌ AUTHENTICATION FAILED — Request timed out.")
    print("   Where:   POST https://api.upstox.com/v2/login/authorization/token")
    print("   Reason:  No response within 15 seconds")
    print("   Fix:     Check your internet connection and try again")
    sys.exit(1)
except Exception as e:
    # AUTH LOG: network/request exception — log type + message for diagnosis
    print(f"❌ AUTHENTICATION FAILED — Network error during token exchange.")
    print(f"   Where:     POST https://api.upstox.com/v2/login/authorization/token")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error:      {e}")
    print(f"   Fix:        Check internet, firewall, or Upstox API status")
    sys.exit(1)

# ── Handle response ───────────────────────────────────────────────────────
if resp.status_code != 200 or "access_token" not in data:
    # AUTH LOG: token exchange rejected by Upstox — log every diagnostic detail
    print(f"❌ AUTHENTICATION FAILED — Upstox rejected the token exchange.")
    print(f"   Where:       POST https://api.upstox.com/v2/login/authorization/token")
    print(f"   HTTP status: {resp.status_code} {resp.reason}")
    print(f"   Response:    {data}")
    # Upstox error fields
    err_code = data.get("error", data.get("status", "unknown"))
    err_msg  = data.get("error_description", data.get("message", ""))
    if err_code != "unknown" or err_msg:
        print(f"   Error code:  {err_code}")
        print(f"   Description: {err_msg}")
    # Missing field detail
    if resp.status_code == 200 and "access_token" not in data:
        print(f"   Specific:    HTTP 200 but 'access_token' field missing in response")
        print(f"   Keys in response: {list(data.keys())}")
    print()
    print("  Common causes:")
    print("  - Auth code already used (each code works only once — get a fresh one)")
    print("  - Auth code expired (valid for ~5 minutes — login again)")
    print("  - Wrong UPSTOX_API_SECRET in .env")
    print("  - Redirect URL mismatch (must be http://localhost:8080/ exactly)")
    print("  - UPSTOX_API_KEY does not match the app that generated this code")
    sys.exit(1)

access_token = data["access_token"]
user_name    = data.get("user_name", "Upstox User")
user_id      = data.get("user_id", "")
token_type   = data.get("token_type", "Bearer")
expires_in   = data.get("expires_in", 86400)

print(f"  ✅ Authenticated as: {user_name}")
if user_id:
    print(f"     User ID:          {user_id}")
print(f"     Token type:       {token_type}")
print(f"     Expires in:       {expires_in // 3600} hours")

# ── Step 5: Save token to .env ─────────────────────────────────────────────
print()
set_key(str(ENV_PATH), "UPSTOX_ACCESS_TOKEN", access_token)
print(f"  ✅ UPSTOX_ACCESS_TOKEN saved to .env")

# Also set BROKER=upstox in .env if not already set
current_broker = os.getenv("BROKER", "")
if current_broker.lower() != "upstox":
    set_key(str(ENV_PATH), "BROKER", "upstox")
    print(f"  ✅ BROKER=upstox set in .env")

# Automatically sync to all sibling Forge projects (ExpiryForge, SwingForge, ThetaForge)
try:
    from scripts.sync_upstox_token import sync_upstox_token
    print("\n  Syncing token to sibling projects (ExpiryForge, SwingForge, ThetaForge)...")
    sync_upstox_token(source_env_path=ENV_PATH, validate=False, notify=True)
except Exception as sync_exc:
    print(f"  ⚠️ Cross-project sync warning: {sync_exc}")

# ── Step 6: Quick API test ─────────────────────────────────────────────────
print()
print("━" * 52)
print("  STEP 5 — Verifying Token with Live API Call")
print("━" * 52)
print()
print("  Testing: fetching NIFTY 50 LTP...")

try:
    test_resp = requests.get(
        "https://api.upstox.com/v2/market-quote/ltp",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept":        "application/json",
        },
        params={"instrument_key": "NSE_INDEX|Nifty 50"},
        timeout=10,
    )
    test_data = test_resp.json()

    if test_resp.status_code == 200 and "data" in test_data:
        nifty_data = test_data["data"]
        # Upstox returns instrument key as the dict key
        key = list(nifty_data.keys())[0] if nifty_data else None
        if key:
            ltp = nifty_data[key].get("last_price", 0)
            print(f"  ✅ NIFTY 50 LTP: ₹{ltp:,.2f}")
        else:
            print("  ✅ Token valid (market may be closed — no LTP)")
    else:
        # AUTH LOG: token saved but LTP verification failed — log full details
        print(f"  ⚠️  LTP verification failed — token was saved but API test failed.")
        print(f"     Where:       GET https://api.upstox.com/v2/market-quote/ltp")
        print(f"     HTTP status: {test_resp.status_code}")
        print(f"     Response:    {test_data}")
        print("     Token is still saved — this may just be outside market hours.")

except Exception as ltp_err:
    # AUTH LOG: LTP verification threw an exception
    print(f"  ⚠️  LTP verification exception — token is still saved.")
    print(f"     Where:      GET https://api.upstox.com/v2/market-quote/ltp")
    print(f"     Error type: {type(ltp_err).__name__}")
    print(f"     Error:      {ltp_err}")
    print("     Not critical — system will work when market opens.")

# ── Step 7: Restart Docker container ──────────────────────────────────────
print()
print("━" * 52)
print("  STEP 6 — Restarting MCXForge Container")
print("━" * 52)
print()

result = subprocess.run(
    ["docker", "ps", "--filter", "name=mcxforge", "--filter", "name=signalforge", "--format", "{{.Names}}"],
    capture_output=True, text=True,
)
if "mcxforge" in result.stdout or "signalforge" in result.stdout:
    print("  Container running — restarting with new token...")
    restart = subprocess.run(
        ["docker", "compose", "restart"],
        cwd=str(ENV_PATH.parent),
        capture_output=True, text=True,
    )
    if restart.returncode == 0:
        print("  ✅ Container restarted successfully.")
    else:
        print(f"  ⚠️  Restart failed: {restart.stderr}")
        print("     Manually run: docker compose restart")
else:
    print("  Container not running.")
    print("  Start with: docker compose up -d")

# ── Done ──────────────────────────────────────────────────────────────────
print()
print("━" * 52)
print(f"  ✅ Upstox auth complete!")
print(f"  Broker: UPSTOX  |  {datetime.now(IST).strftime('%H:%M IST')}")
print(f"  Dashboard: http://localhost:5050")
print("━" * 52)
print()
print("  Daily reminder:")
print("  → Run this script every morning before 09:10 IST")
print("  → Token expires at midnight — needs daily refresh")
print()
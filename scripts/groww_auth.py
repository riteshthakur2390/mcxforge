#!/usr/bin/env python3
"""
scripts/groww_auth.py — Daily Groww Authentication
====================================================
Groww uses API Key + TOTP (no OAuth redirect needed).
Much simpler than Kite or Upstox — no browser redirect required.

Auth flow:
    1. You have API Key + TOTP Secret from Groww Cloud console
    2. Script generates TOTP automatically using pyotp
    3. Calls GrowwAPI.get_access_token(api_key, totp=totp)
    4. Saves access token to .env
    5. Restarts Docker container

Setup (one time only):
    1. Go to https://groww.in/trade-api → Subscribe (Rs 499/month)
    2. Groww Cloud → API Keys → Generate API Key
       - Enter name: SignalForge-Dev
       - Copy API Key and API Secret
    3. Groww Cloud → Generate TOTP Token
       - Copy TOTP Secret (long string) — NOT the 6-digit OTP
       - Add to authenticator app (optional) OR use this script directly
    4. Add all three to your .env file

Usage:
    python scripts/groww_auth.py
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ── Check dependencies ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv, set_key
except ImportError:
    print("\n❌ python-dotenv not installed.")
    print("   Run: pip install python-dotenv")
    sys.exit(1)

try:
    import pyotp
except ImportError:
    print("\n❌ pyotp not installed.")
    print("   Run: pip install pyotp")
    sys.exit(1)

import pytz
IST = pytz.timezone("Asia/Kolkata")

# ── Load .env ──────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).parent.parent / ".env"
if not ENV_PATH.exists():
    print(f"\n❌ .env file not found at: {ENV_PATH}")
    print("   Run: cp .env.example .env  then fill in your Groww credentials.")
    sys.exit(1)

load_dotenv(ENV_PATH)

API_KEY     = os.getenv("GROWW_API_KEY", "")
API_SECRET  = os.getenv("GROWW_API_SECRET", "")
TOTP_SECRET = os.getenv("GROWW_TOTP_SECRET", "")

# ── Validate credentials ───────────────────────────────────────────────────
missing = []
if not API_KEY    or API_KEY    == "your_groww_api_key_here":    missing.append("GROWW_API_KEY")
if not API_SECRET or API_SECRET == "your_groww_api_secret_here": missing.append("GROWW_API_SECRET")
if not TOTP_SECRET or TOTP_SECRET == "your_groww_totp_secret_here": missing.append("GROWW_TOTP_SECRET")

if missing:
    print(f"\n❌ Missing in .env: {', '.join(missing)}")
    print()
    print("  Get from: https://groww.in/trade-api → Groww Cloud → API Keys")
    print("  Steps:")
    print("  1. Subscribe to Trading API (Rs 499+tax/month)")
    print("  2. Generate API Key → copy GROWW_API_KEY + GROWW_API_SECRET")
    print("  3. Generate TOTP Token → copy the TOTP Secret → GROWW_TOTP_SECRET")
    print(f"\n  Edit: {ENV_PATH}")
    sys.exit(1)

# ── Check growwapi installed ───────────────────────────────────────────────
try:
    from growwapi import GrowwAPI
except ImportError:
    print("\n❌ growwapi not installed.")
    print("   Run: pip install growwapi")
    print("   Also add 'growwapi>=1.5.0' to requirements.txt")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# MAIN AUTH FLOW
# ══════════════════════════════════════════════════════════════════════════

print()
print("╔════════════════════════════════════════════════╗")
print("║   MCXForge — Groww Daily Auth                  ║")
print(f"║   {datetime.now(IST).strftime('%Y-%m-%d  %H:%M IST')}                        ║")
print("╚════════════════════════════════════════════════╝")
print()
print("  ℹ️  Groww uses TOTP auth — no browser login needed.")
print()

# ── Step 1: Generate TOTP ─────────────────────────────────────────────────
print("━" * 52)
print("  STEP 1 — Generating TOTP")
print("━" * 52)
print()

try:
    totp_gen = pyotp.TOTP(TOTP_SECRET)
    totp     = totp_gen.now()
    print(f"  ✅ TOTP generated: {totp}")
    print(f"     (Valid for ~30 seconds)")
except Exception as e:
    print(f"  ❌ Failed to generate TOTP: {e}")
    print("     Check your GROWW_TOTP_SECRET — it should be a long base32 string")
    print("     NOT the 6-digit OTP you see in an authenticator app")
    sys.exit(1)

# ── Step 2: Get access token ──────────────────────────────────────────────
print()
print("━" * 52)
print("  STEP 2 — Generating Access Token")
print("━" * 52)
print()
print("  Contacting Groww API...")

try:
    access_token = GrowwAPI.get_access_token(
        api_key = API_KEY,
        totp    = totp,
    )
    if not access_token:
        raise ValueError("Empty access token returned")
    print(f"  ✅ Access token received: {access_token[:12]}...{access_token[-4:]}")
except Exception as e:
    print(f"  ❌ Token generation failed: {e}")
    print()
    print("  Common causes:")
    print("  - TOTP expired (30-second window) — run the script again immediately")
    print("  - Wrong GROWW_API_KEY or GROWW_TOTP_SECRET in .env")
    print("  - API subscription expired")
    sys.exit(1)

# ── Step 3: Verify token with live API call ───────────────────────────────
print()
print("━" * 52)
print("  STEP 3 — Verifying Token with Live API Call")
print("━" * 52)
print()
print("  Testing: fetching NIFTY 50 LTP...")

try:
    groww = GrowwAPI(access_token)
    # Use feed to get NIFTY index value
    from growwapi import GrowwFeed
    feed = GrowwFeed(groww)
    feed.subscribe_index_value(
        [{"exchange": "NSE", "segment": "CASH", "exchange_token": "NIFTY"}]
    )
    import time
    time.sleep(2)   # wait for feed data
    index_data = feed.get_index_value()

    nifty_ltp = 0.0
    if index_data and "NSE" in index_data:
        nifty_ltp = index_data["NSE"].get("CASH", {}).get("NIFTY", {}).get("value", 0)

    if nifty_ltp > 0:
        print(f"  ✅ NIFTY 50 LTP: ₹{nifty_ltp:,.2f}")
    else:
        # Try LTP via REST API as fallback
        import requests
        resp = requests.get(
            "https://api.groww.in/v1/live-data/ohlc",
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-API-VERSION": "1.0",
                "Accept":        "application/json",
            },
            params={"segment": "CASH", "exchange_symbols": "NSE_NIFTY"},
            timeout=10,
        )
        if resp.status_code == 200:
            print(f"  ✅ Token valid (market may be closed — LTP via REST OK)")
        else:
            print(f"  ⚠️  Token saved but LTP test inconclusive (market may be closed)")

except Exception as e:
    print(f"  ⚠️  LTP verification skipped: {e}")
    print("     Token is saved — this is not critical.")

# ── Step 4: Save token to .env ────────────────────────────────────────────
print()
print("━" * 52)
print("  STEP 4 — Saving Token")
print("━" * 52)
print()

set_key(str(ENV_PATH), "GROWW_ACCESS_TOKEN", access_token)
print(f"  ✅ GROWW_ACCESS_TOKEN saved to .env")

# Also set BROKER=groww
current_broker = os.getenv("BROKER", "")
if current_broker.lower() != "groww":
    set_key(str(ENV_PATH), "BROKER", "groww")
    print(f"  ✅ BROKER=groww set in .env")

# ── Step 5: Restart Docker container ──────────────────────────────────────
print()
print("━" * 52)
print("  STEP 5 — Restarting MCXForge Container")
print("━" * 52)
print()

result = subprocess.run(
    ["docker", "ps", "--filter", "name=mcxforge", "--filter", "name=signalforge", "--format", "{{.Names}}"],
    capture_output=True, text=True,
)
if "mcxforge" in result.stdout or "signalforge" in result.stdout:
    print("  Container running — restarting with new token...")
    restart = subprocess.run(
        ["docker-compose", "restart"],
        cwd=str(ENV_PATH.parent),
        capture_output=True, text=True,
    )
    if restart.returncode == 0:
        print("  ✅ Container restarted successfully.")
    else:
        print(f"  ⚠️  Restart failed: {restart.stderr}")
        print("     Manually run: docker-compose restart")
else:
    print("  Container not running.")
    print("  Start with: docker-compose up -d")

# ── Done ──────────────────────────────────────────────────────────────────
print()
print("━" * 52)
print(f"  ✅ Groww auth complete!")
print(f"  Broker: GROWW  |  {datetime.now(IST).strftime('%H:%M IST')}")
print(f"  Dashboard: http://localhost:5050")
print("━" * 52)
print()
print("  Daily reminder:")
print("  → Run this script every morning before 09:10 IST")
print("  → Token expires daily — needs refresh each morning")
print()
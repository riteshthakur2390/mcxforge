#!/usr/bin/env python3
"""
scripts/dhan_auth.py — Dhan HQ Credential Setup & Verification
===============================================================
Use this script to verify a manually generated Dhan token. Dhan tokens should
be renewed before expiry by scripts/dhan_renew_tokens.py.

Run this ONCE when you first set up Dhan, or after regenerating
a new token from the Dhan web dashboard.

What this script does:
    1. Checks your .env has DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN set
    2. Verifies the token is valid with a live API call (fund limit)
    3. Verifies market data works (NIFTY LTP fetch)
    4. Sets BROKER=dhan in .env
    5. Restarts Docker container if running

How to get your credentials:
    1. Go to https://web.dhan.co
    2. Login with your Dhan account
    3. Go to: My Profile → Apps → Create / Manage API App
    4. Note your Client ID (shown at top)
    5. Click "Generate Token" → copy the Access Token
    6. Paste both into your .env file:
           DHAN_CLIENT_ID=your_client_id_here
           DHAN_ACCESS_TOKEN=your_access_token_here
    7. Run this script: python scripts/dhan_auth.py

Token expiry:
    Dhan access tokens must be renewed before they expire. If a token has
    already expired, generate a fresh token from the dashboard, put it in
    .env, and then run scripts/dhan_renew_tokens.py from cron going forward.
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from pathlib import Path
import os
import sys
import requests

from dotenv import load_dotenv, set_key



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
    print("   Run: cp .env.example .env  then fill in your Dhan credentials.")
    sys.exit(1)

load_dotenv(ENV_PATH)

CLIENT_ID    = os.getenv("DHAN_CLIENT_ID", "").strip()
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
DHAN_BASE    = "https://api.dhan.co"

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
print()
print("╔════════════════════════════════════════════════╗")
print("║   SignalForge — Dhan HQ Credential Setup       ║")
print(f"║   {datetime.now(IST).strftime('%Y-%m-%d  %H:%M IST')}                        ║")
print("╚════════════════════════════════════════════════╝")
print()
print("  Dhan token verification")
print("     This script verifies the manually generated token currently in .env.")
print()

# ── Step 1: Validate credentials present ─────────────────────────────────
print("━" * 52)
print("  STEP 1 — Checking .env credentials")
print("━" * 52)
print()

missing = []
if not CLIENT_ID or CLIENT_ID == "your_dhan_client_id_here":
    missing.append("DHAN_CLIENT_ID")
if not ACCESS_TOKEN or ACCESS_TOKEN == "your_dhan_access_token_here":
    missing.append("DHAN_ACCESS_TOKEN")

if missing:
    # AUTH LOG: credentials missing — tell user exactly where to get them
    print("❌ AUTHENTICATION FAILED — Missing credentials in .env")
    print(f"   Missing: {', '.join(missing)}")
    print()
    print("  How to fix:")
    print("  1. Go to https://web.dhan.co → My Profile → Apps")
    print("  2. Create an app or open existing app")
    print("  3. Copy Client ID and generate Access Token")
    print(f"  4. Open your .env file at: {ENV_PATH}")
    print("  5. Set:")
    for m in missing:
        print(f"       {m}=<your value here>")
    print()
    sys.exit(1)

# Mask token for display
token_preview = ACCESS_TOKEN[:8] + "..." + ACCESS_TOKEN[-4:]
print(f"  ✅ DHAN_CLIENT_ID    = {CLIENT_ID}")
print(f"  ✅ DHAN_ACCESS_TOKEN = {token_preview}  (masked)")
print()

# ── Step 2: Verify token via fund limit API ───────────────────────────────
print("━" * 52)
print("  STEP 2 — Verifying Token (Fund Limit API)")
print("━" * 52)
print()
print("  Contacting Dhan API...")

HEADERS = {
    "access-token": ACCESS_TOKEN,
    "client-id":    CLIENT_ID,
    "Content-Type": "application/json",
}

try:
    resp = requests.get(
        f"{DHAN_BASE}/v2/fundlimit",
        headers=HEADERS,
        timeout=15,
    )
    data = resp.json()

except requests.exceptions.Timeout:
    # AUTH LOG: verify call timed out
    print("❌ AUTHENTICATION FAILED — Request timed out.")
    print(f"   Where:  GET {DHAN_BASE}/v2/fundlimit")
    print("   Reason: No response within 15 seconds")
    print("   Fix:    Check your internet connection and try again")
    sys.exit(1)

except Exception as e:
    # AUTH LOG: network exception during verify
    print("❌ AUTHENTICATION FAILED — Network error.")
    print(f"   Where:      GET {DHAN_BASE}/v2/fundlimit")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error:      {e}")
    print("   Fix:        Check internet connection or Dhan API status")
    sys.exit(1)

if resp.status_code != 200:
    # AUTH LOG: token rejected — extract and show all error details
    err_code  = data.get("errorCode", data.get("status", "unknown"))
    err_msg   = data.get("remarks", data.get("message", "no detail"))
    print("❌ AUTHENTICATION FAILED — Dhan rejected the token.")
    print(f"   Where:       GET {DHAN_BASE}/v2/fundlimit")
    print(f"   HTTP status: {resp.status_code}")
    print(f"   Error code:  {err_code}")
    print(f"   Description: {err_msg}")
    print(f"   Full response: {data}")
    print()
    print("  Common causes:")
    print("  - Access token is incorrect (copy it again from Dhan dashboard)")
    print("  - Token was regenerated/revoked — generate a fresh one")
    print("  - Client ID does not match the token")
    print("  - Dhan account is suspended or not activated for API")
    print()
    print(f"  Dashboard: https://web.dhan.co → My Profile → Apps")
    sys.exit(1)

# Parse fund limit response
available_bal  = data.get("availabelBalance",
                 data.get("availableBalance",
                 data.get("net", 0.0)))
used_margin    = data.get("utilizedAmount", 0.0)
total_balance  = data.get("sodLimit", data.get("totalBalance", 0.0))

print(f"  ✅ Token is VALID")
print(f"     Available balance: ₹{float(available_bal):,.2f}")
if used_margin:
    print(f"     Used margin:       ₹{float(used_margin):,.2f}")
print()

# ── Step 3: Verify market data (NIFTY LTP) ───────────────────────────────
print("━" * 52)
print("  STEP 3 — Verifying Market Data (NIFTY LTP)")
print("━" * 52)
print()
print("  Testing: fetching NIFTY 50 LTP...")

try:
    ltp_resp = requests.post(
        f"{DHAN_BASE}/v2/marketfeed/ltp",
        headers=HEADERS,
        json={"NSE_INDEX": ["13"]},    # "13" = NIFTY 50
        timeout=10,
    )
    ltp_data = ltp_resp.json()

    if ltp_resp.status_code == 200:
        seg  = ltp_data.get("data", {}).get("NSE_INDEX", {})
        ltp  = float(seg.get("13", {}).get("last_price", 0.0))
        if ltp > 0:
            print(f"  ✅ NIFTY 50 LTP: ₹{ltp:,.2f}")
        else:
            print("  ✅ Token valid (NIFTY LTP = 0 — market may be closed)")
    else:
        # AUTH LOG: LTP call failed — token saved but data call broken
        err_msg = ltp_data.get("remarks", ltp_data.get("message", "no detail"))
        print(f"  ⚠️  LTP verification failed — token is valid but market data call failed.")
        print(f"     Where:       POST {DHAN_BASE}/v2/marketfeed/ltp")
        print(f"     HTTP status: {ltp_resp.status_code}")
        print(f"     Error:       {err_msg}")
        print("     Token is valid — this may be a market hours issue or API tier restriction.")

except Exception as ltp_err:
    # AUTH LOG: LTP call threw exception
    print(f"  ⚠️  LTP verification exception — token is valid but market data could not be fetched.")
    print(f"     Where:      POST {DHAN_BASE}/v2/marketfeed/ltp")
    print(f"     Error type: {type(ltp_err).__name__}")
    print(f"     Error:      {ltp_err}")
    print("     Not critical — system will work when market opens.")

print()

# ── Step 4: Set BROKER=dhan in .env ───────────────────────────────────────
print("━" * 52)
print("  STEP 4 — Updating .env")
print("━" * 52)
print()

current_broker = os.getenv("BROKER", "")
if current_broker.lower() != "dhan":
    set_key(str(ENV_PATH), "BROKER", "dhan")
    print(f"  ✅ BROKER=dhan set in .env")
else:
    print(f"  ✅ BROKER=dhan already set")
print()

# ── Done ──────────────────────────────────────────────────────────────────
print()
print("━" * 52)
print(f"  ✅ Dhan setup complete!")
print(f"  Broker: DHAN  |  {datetime.now(IST).strftime('%H:%M IST')}")
print(f"  Client ID: {CLIENT_ID}")
print(f"  Dashboard: http://localhost:5050")
print("━" * 52)
print()
print("  Important notes for Dhan:")
print("  -> Renew tokens before expiry with scripts/dhan_renew_tokens.py")
print("  -> Re-run this script if you manually generate a new token")
print("  → In .env: BROKER=dhan  (already set above)")
print()


DHAN_BASE = "https://api.dhan.co"

def main():
    env_path = Path(__file__).parent.parent / ".env"

    if not env_path.exists():
        print(f"\n❌ .env file not found: {env_path}")
        sys.exit(1)

    load_dotenv(env_path)

    client_id = os.getenv("DHAN_CLIENT_ID", "").strip()
    access_token = os.getenv("DHAN_ACCESS_TOKEN", "").strip()

    print("\nSignalForge — Dhan Auth Verification")
    print("━" * 50)

    # STEP 1
    print("\nSTEP 1 — Checking .env")

    missing = []

    if not client_id:
        missing.append("DHAN_CLIENT_ID")

    if not access_token:
        missing.append("DHAN_ACCESS_TOKEN")

    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        sys.exit(1)

    print("✅ Credentials found")

    headers = {
        "access-token": access_token,
        "client-id": client_id,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # STEP 2
    print("\nSTEP 2 — Verifying Profile")

    try:
        profile_resp = requests.get(
            f"{DHAN_BASE}/v2/profile",
            headers=headers,
            timeout=15,
        )

        if profile_resp.status_code != 200:
            print(f"❌ AUTH FAILED ({profile_resp.status_code})")
            print(profile_resp.text)
            sys.exit(1)

        profile = profile_resp.json().get("data", {})

        client_name = profile.get("clientName", "N/A")
        client_status = profile.get("clientStatus", "N/A")

        print("✅ Profile verified")
        print(f"   Client ID : {client_id}")
        print(f"   Name      : {client_name}")
        print(f"   Status    : {client_status}")

    except Exception as e:
        print(f"❌ Profile verification failed: {e}")
        sys.exit(1)

    # STEP 3
    print("\nSTEP 3 — Verifying Fund Limit")

    available_balance = 0.0

    try:
        fund_resp = requests.get(
            f"{DHAN_BASE}/v2/fundlimit",
            headers=headers,
            timeout=15,
        )

        if fund_resp.status_code == 200:
            fund_data = fund_resp.json()

            available_balance = float(
                fund_data.get(
                    "availableBalance",
                    fund_data.get("availabelBalance", 0),
                )
            )

            print("✅ Fund API working")
            print(f"   Available Balance : ₹{available_balance:,.2f}")

        else:
            print(
                f"⚠️ Fund limit API failed ({fund_resp.status_code})"
            )

    except Exception as e:
        print(f"⚠️ Fund limit check failed: {e}")

    # STEP 4
    print("\nSTEP 4 — Verifying Market Data")

    nifty_ltp = None

    try:
        ltp_resp = requests.post(
            f"{DHAN_BASE}/v2/marketfeed/ltp",
            headers=headers,
            json={"NSE_INDEX": ["13"]},
            timeout=15,
        )

        if ltp_resp.status_code == 200:
            data = ltp_resp.json()

            nifty_ltp = (
                data.get("data", {})
                .get("NSE_INDEX", {})
                .get("13", {})
                .get("last_price")
            )

            if nifty_ltp:
                print(f"✅ NIFTY 50 LTP : ₹{float(nifty_ltp):,.2f}")
            else:
                print("⚠️ Market data returned empty")

        else:
            print(
                f"⚠️ Market data API failed ({ltp_resp.status_code})"
            )

    except Exception as e:
        print(f"⚠️ Market data check failed: {e}")

    # STEP 5
    print("\nSTEP 5 — Setting BROKER=dhan")

    try:
        set_key(str(env_path), "BROKER", "dhan")
        print("✅ BROKER=dhan saved to .env")
    except Exception as e:
        print(f"⚠️ Could not update .env: {e}")

    # DONE
    print("\n" + "━" * 50)
    print("✅ Dhan setup complete")
    print("━" * 50)

    print(f"Client ID         : {client_id}")
    print(f"Client Name       : {client_name}")
    print(f"Client Status     : {client_status}")
    print(f"Available Balance : ₹{available_balance:,.2f}")

    if nifty_ltp:
        print(f"NIFTY 50 LTP      : ₹{float(nifty_ltp):,.2f}")

    print("\nBroker            : DHAN")
    print("Token Status      : VALID")
    print("BROKER=dhan saved in .env")
    print("\nDhan tokens do not require daily refresh.")


if __name__ == "__main__":
    main()

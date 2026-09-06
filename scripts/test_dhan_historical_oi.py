import os
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")

def test_historical_oi():
    if not CLIENT_ID or not ACCESS_TOKEN:
        print("Error: Dhan credentials not found in .env")
        return

    # Target an expiry around 10 days ago (June 11, 2026 was a Thursday)
    # We'll try to get data for NIFTY 23500 CE expired on 2026-06-11
    # Note: Dhan security IDs for expired contracts can be tricky.
    # Usually, we fetch by instrument name or specialized lookup.
    
    # Let's try to find a recent active/expired contract to check the structure
    url = "https://api.dhanhq.co/v2/charts/historical"
    
    headers = {
        "access-token": ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    
    # We will try NIFTY Spot first to verify connectivity
    spot_payload = {
        "symbol": "NIFTY",
        "exchangeSegment": "NSE_INDEX",
        "instrumentId": "13", # NIFTY 50 Index
        "expiryCode": 0,
        "fromDate": "2026-06-10",
        "toDate": "2026-06-11",
        "interval": "5"
    }
    
    print(f"Testing connectivity with NIFTY Spot...")
    try:
        response = requests.post(url, headers=headers, json=spot_payload, timeout=10)
        print(f"Spot Response Status: {response.status_code}")
        if response.status_code == 200:
            print("Successfully connected to Dhan API.")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Connectivity error: {e}")
        return

    # Now attempt to check for Option data with OI
    # Note: Dhan's historical API usually returns OHLCV. 
    # OI is often a separate endpoint or only available for 'intraday' charts.
    print("\nChecking for Option Historical Data + OI...")
    
    # We'll try to fetch NIFTY 11th June 23000 CE (if we had the ID)
    # Since IDs for expired contracts are hard to guess, we'll try a common range
    # and look for OI field in the response schema.
    
    option_payload = {
        "symbol": "NIFTY",
        "exchangeSegment": "NSE_FNO",
        "instrumentId": "55331", # Example ID, might not match 10 days ago
        "expiryCode": 0,
        "fromDate": "2026-06-10",
        "toDate": "2026-06-11",
        "interval": "5"
    }
    
    try:
        response = requests.post(url, headers=headers, json=option_payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                print("Sample Data Point:")
                print(json.dumps(data['data'][0], indent=2))
                fields = data['data'][0].keys()
                print(f"\nAvailable Fields: {list(fields)}")
                if 'oi' in fields or 'open_interest' in fields:
                    print("✅ OI is available in historical charts!")
                else:
                    print("❌ OI is NOT available in standard historical charts.")
            else:
                print("No data returned for this instrument (it may have expired and been removed from the chart API).")
        else:
            print(f"Option Fetch failed: {response.status_code} - {response.text}")
            print("\nNote: Dhan often removes expired contracts from their historical chart API immediately after expiry.")
            
    except Exception as e:
        print(f"Error checking option data: {e}")

if __name__ == "__main__":
    test_historical_oi()

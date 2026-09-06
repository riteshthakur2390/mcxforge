"""
scripts/e2e_live_readiness_audit.py
===================================
Comprehensive End-to-End Live Readiness Audit for SignalForge.
Run this script to verify all broker connections, option resolution,
margin gates, and execution routing before market opens.
"""

import sys, os, time
from datetime import datetime, date
from pathlib import Path
import pytz
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

IST = pytz.timezone("Asia/Kolkata")

def run_audit(send_telegram: bool = False):
    print("=" * 70)
    print("   SIGNALFORGE — END-TO-END LIVE READINESS AUDIT")
    print(f"   Timestamp: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 70)

    all_passed = True

    # ── CHECK 1: Broker Factory & Primary Market Data Feed ───────────────────
    print("\n[CHECK 1] Initializing Primary Broker (Dhan Main)...")
    try:
        from broker.factory import get_broker
        primary_broker = get_broker()
        print(f"  ✓ Broker Name: {primary_broker.broker_name}")
        nifty_ltp = primary_broker.get_ltp("NIFTY")
        print(f"  ✓ NIFTY Spot LTP: {nifty_ltp:.2f}")
        vix = primary_broker.get_india_vix()
        print(f"  ✓ India VIX: {vix:.2f}")
        # Allow pre-market 0 LTP before 09:15 IST
        now_time = datetime.now(IST).time()
        if now_time.hour >= 9 and now_time.minute >= 15:
            assert nifty_ltp > 20000, f"NIFTY LTP out of expected range ({nifty_ltp})"
        else:
            print("  ℹ️ Pre-market hours (before 09:15 IST) — market quotes idle until market open")
    except Exception as e:
        print(f"  ❌ CHECK 1 FAILED: {e}")
        all_passed = False

    # ── CHECK 2: Option Chain & Security ID Resolution (Dhan) ───────────────
    print("\n[CHECK 2] Dhan Option Chain & Security ID Lookup...")
    try:
        from utils.option_utils import get_weekly_expiry, build_option_symbol
        today = date.today()
        expiry = get_weekly_expiry(today, "NIFTY")
        test_pe = build_option_symbol("NIFTY", expiry, 24300, "PE")
        test_ce = build_option_symbol("NIFTY", expiry, 24300, "CE")

        pe_sec_id = primary_broker.get_instrument_key(test_pe)
        ce_sec_id = primary_broker.get_instrument_key(test_ce)

        print(f"  ✓ Expiry: {expiry.isoformat()} (Tuesday)")
        print(f"  ✓ {test_pe} -> Security ID: {pe_sec_id}")
        print(f"  ✓ {test_ce} -> Security ID: {ce_sec_id}")
        assert pe_sec_id.isdigit(), f"Expected numeric securityId for {test_pe}, got {pe_sec_id}"
        assert ce_sec_id.isdigit(), f"Expected numeric securityId for {test_ce}, got {ce_sec_id}"
    except Exception as e:
        print(f"  ❌ CHECK 2 FAILED: {e}")
        all_passed = False

    # ── CHECK 3: Upstox Instrument Key Resolution & Master Cache ─────────────
    print("\n[CHECK 3] Upstox Instrument Resolution & Master Cache...")
    try:
        from broker.upstox_broker import UpstoxBroker
        upstox = UpstoxBroker()
        up_pe_key = upstox._resolve_option_symbol_to_key(test_pe)
        up_ce_key = upstox._resolve_option_symbol_to_key(test_ce)

        print(f"  ✓ {test_pe} -> Upstox Key: {up_pe_key}")
        print(f"  ✓ {test_ce} -> Upstox Key: {up_ce_key}")
        assert up_pe_key.startswith("NSE_FO|"), f"Invalid Upstox key format for {test_pe}: {up_pe_key}"
        assert up_ce_key.startswith("NSE_FO|"), f"Invalid Upstox key format for {test_ce}: {up_ce_key}"
    except Exception as e:
        print(f"  ❌ CHECK 3 FAILED: {e}")
        all_passed = False

    # ── CHECK 4: Multi-Account Execution Configuration ──────────────────────
    print("\n[CHECK 4] Multi-Broker Parallel Execution Configuration...")
    try:
        from utils.multi_account_execution import load_execution_accounts
        accounts = load_execution_accounts(primary_broker)
        print(f"  ✓ Configured Execution Accounts ({len(accounts)}):")
        labels = []
        for a in accounts:
            print(f"    - {a.label:<18} | Broker: {a.broker.__class__.__name__:<15} | Primary: {a.primary}")
            labels.append(a.label)
        assert "upstox:main" in labels or any("upstox" in l for l in labels), "Upstox execution account missing"
        dhan_exec = os.getenv("DHAN_ORDER_EXECUTION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        if dhan_exec:
            assert any("dhan" in l for l in labels), "Dhan execution account missing when DHAN_ORDER_EXECUTION_ENABLED=true"
            print("  ✓ Dhan order execution active (Upstox + Dhan parallel routing)")
        else:
            assert not any("dhan" in l for l in labels), "Dhan account present in execution accounts despite DHAN_ORDER_EXECUTION_ENABLED=false"
            print("  ✓ Dhan order execution disabled (Dhan reserved for market data & candle feeds; Upstox sole order executor)")
    except Exception as e:
        print(f"  ❌ CHECK 4 FAILED: {e}")
        all_passed = False

    # ── CHECK 5: Multi-Account Pooled Margin Verification ────────────────────
    print("\n[CHECK 5] Multi-Account Margin Aggregation Check...")
    try:
        import asyncio
        from utils.order_manager import get_order_manager
        om = get_order_manager(primary_broker, execution_accounts=accounts)
        margin = asyncio.run(om.check_margin(premium=120.0, lot_size=65, lots=1))
        print(f"  ✓ Available Margin: ₹{margin.available_inr:,.2f}")
        print(f"  ✓ Required Margin (1 lot @ ₹120): ₹{margin.required_inr:,.2f}")
        print(f"  ✓ Margin Sufficient: {margin.sufficient}")
        print(f"  ✓ Affordable Lots: {margin.lots_affordable}")
        assert margin.sufficient, f"Margin check failed: {margin.reason}"
        assert margin.available_inr > 20000, f"Pooled margin too low: ₹{margin.available_inr}"
    except Exception as e:
        print(f"  ❌ CHECK 5 FAILED: {e}")
        all_passed = False

    # ── CHECK 6: Smart Money Filter Delta Gate ──────────────────────────────
    print("\n[CHECK 6] Smart Money Filter Delta Gate Verification...")
    try:
        from utils.smart_entry_filter import get_smart_entry_filter
        from config.settings.modules.utils_thresholds import SMART_ENTRY_DELTA_MIN_BUY
        print(f"  ✓ SMART_ENTRY_DELTA_MIN_BUY threshold: {SMART_ENTRY_DELTA_MIN_BUY}")
        assert SMART_ENTRY_DELTA_MIN_BUY <= 0.40, f"Delta min threshold too strict ({SMART_ENTRY_DELTA_MIN_BUY})"
        
        # Test evaluation with delta = 0.44 (which was blocked earlier today)
        sef = get_smart_entry_filter()
        check = sef.check_oi_delta_gamma(oi=5000000, delta=0.44, gamma=0.0012)
        print(f"  ✓ Test PE (delta=0.44, OI=5M): allow_buy={check.allow_buy} | edge={check.edge}")
        assert check.allow_buy, f"Delta 0.44 should pass with updated threshold but got: {check.reason}"
    except Exception as e:
        print(f"  ❌ CHECK 6 FAILED: {e}")
        all_passed = False

    # ── CHECK 7: Strategy & ML Pipeline Integrity ────────────────────────────
    print("\n[CHECK 7] ML Model & Strategy Registry Health...")
    try:
        from pathlib import Path
        from ml.model import SignalForgeEnsemble
        ensemble = SignalForgeEnsemble()
        model_p = Path("ml/saved_models/silvermic_5minute.pkl")
        if not model_p.exists():
            model_p = Path("ml/saved_models/nifty_5minute.pkl")
        loaded = ensemble.load(model_p)
        print(f"  ✓ ML Ensemble loaded ({model_p.name}): models={list(ensemble.models.keys())} | features={len(ensemble.feature_cols)}")
        assert loaded, f"Failed to load ML model file {model_p}"
        from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY
        print(f"  ✓ Total Registered Strategies: {len(STRATEGY_REGISTRY)}")
        assert len(STRATEGY_REGISTRY) >= 19, f"Expected at least 19 strategies, found {len(STRATEGY_REGISTRY)}"
    except Exception as e:
        print(f"  ❌ CHECK 7 FAILED: {e}")
        all_passed = False

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    status_text = "✅ ALL 7 AUDIT CHECKS PASSED — READY FOR LIVE TRADING" if all_passed else "❌ AUDIT FAILED — ISSUES DETECTED"
    print(f"   {status_text}")
    print("=" * 70)

    if send_telegram:
        try:
            from utils.telegram_notifier import get_notifier
            import asyncio
            msg = (
                f"🛡️ *SignalForge — 08:55 Pre-Market Readiness Audit*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Status: *{'READY ✅' if all_passed else 'ACTION REQUIRED ❌'}*\n"
                f"Time: {datetime.now(IST).strftime('%H:%M:%S IST')}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• Primary Broker (Dhan): {'OK ✅' if 'primary_broker' in locals() else 'FAIL ❌'}\n"
                f"• Execution Accounts: {'OK ✅' if 'accounts' in locals() else 'FAIL ❌'}\n"
                f"• Upstox Key Resolution: {'OK ✅' if 'up_pe_key' in locals() else 'FAIL ❌'}\n"
                f"• Pooled Margin: ₹{margin.available_inr:,.0f} ({margin.lots_affordable} lots)\n"
                f"• Strategy Registry: {len(STRATEGY_REGISTRY) if 'STRATEGY_REGISTRY' in locals() else 0} active\n"
                f"• ML Model: {'OK ✅' if 'loaded' in locals() and loaded else 'FAIL ❌'}"
            )
            asyncio.run(get_notifier().send_text(msg, target="LIVE", parse_mode="markdown"))
            print("  ✓ Sent Telegram notification to LIVE channel")
        except Exception as e:
            print(f"  ⚠️ Telegram notification error: {e}")

    return all_passed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SignalForge Live Readiness Audit")
    parser.add_argument("--telegram", action="store_true", help="Send summary to Telegram")
    args = parser.parse_args()
    success = run_audit(send_telegram=args.telegram)
    sys.exit(0 if success else 1)

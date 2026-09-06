import time
import pytest
from datetime import date
from broker.dhan_broker import DhanBroker

def test_option_contracts_warm_cache_latency():
    """Verify get_option_contracts serves warm cached data in < 5ms without network calls."""
    broker = DhanBroker()
    expiry = date(2026, 9, 1)
    
    # Inject warm option chain payload
    sample_chain = {
        "24500.000000": {
            "ce": {"last_price": 120.5, "security_id": 55101, "top_bid_price": 120.0, "top_ask_price": 121.0},
            "pe": {"last_price": 95.0, "security_id": 55102, "top_bid_price": 94.5, "top_ask_price": 95.5},
        },
        "24550.000000": {
            "ce": {"last_price": 90.0, "security_id": 55103, "top_bid_price": 89.5, "top_ask_price": 90.5},
            "pe": {"last_price": 125.0, "security_id": 55104, "top_bid_price": 124.5, "top_ask_price": 125.5},
        }
    }
    
    now = time.monotonic()
    broker._option_chain_cache[("NIFTY", expiry.isoformat())] = (now, sample_chain)
    broker._pre_index_security_ids("NIFTY", expiry, sample_chain)
    
    t0 = time.perf_counter()
    contracts = broker.get_option_contracts("NIFTY", expiry, "CE", [24500, 24550])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    
    assert len(contracts) == 2
    assert contracts[0].strike == 24500
    assert contracts[0].last_price == 120.5
    assert elapsed_ms < 5.0 # Must complete in < 5ms (typically < 0.1ms)


def test_get_instrument_key_pre_indexed_cache():
    """Verify get_instrument_key uses fast pre-indexed memory lookup in < 1ms."""
    broker = DhanBroker()
    expiry = date(2026, 9, 1)
    
    sample_chain = {
        "24500.000000": {
            "ce": {"last_price": 120.5, "security_id": 55101},
            "pe": {"last_price": 95.0, "security_id": 55102},
        }
    }
    
    broker._pre_index_security_ids("NIFTY", expiry, sample_chain)
    
    t0 = time.perf_counter()
    sec_id_ce = broker.get_instrument_key("NIFTY26SEP0124500CE")
    elapsed_ce_ms = (time.perf_counter() - t0) * 1000.0
    
    t1 = time.perf_counter()
    sec_id_pe = broker.get_instrument_key("NIFTY26SEP0124500PE")
    elapsed_pe_ms = (time.perf_counter() - t1) * 1000.0
    
    assert sec_id_ce == "55101"
    assert sec_id_pe == "55102"
    assert elapsed_ce_ms < 1.0
    assert elapsed_pe_ms < 1.0


def test_option_chain_cache_tuple_and_string_key_compatibility():
    """Verify cache lookup works with both tuple ('NIFTY', '2026-09-01') and string 'NIFTY_2026-09-01'."""
    broker = DhanBroker()
    expiry = date(2026, 9, 1)
    
    sample_chain = {
        "24500.000000": {
            "ce": {"last_price": 120.5, "security_id": 55101},
            "pe": {"last_price": 95.0, "security_id": 55102},
        }
    }
    
    now = time.monotonic()
    # Cache only as string key
    broker._option_chain_cache["NIFTY_2026-09-01"] = (now, sample_chain)
    
    sec_id = broker.get_instrument_key("NIFTY26SEP0124500CE")
    assert sec_id == "55101"

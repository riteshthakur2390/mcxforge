"""
tests/unit/test_dhan_mcx_broker.py — Dhan Broker MCX Futures Integration Tests
=============================================================================
Verifies:
  1. Default broker factory returns DhanBroker
  2. MCX commodity security ID resolution in DhanBroker
  3. LTP routing to Seg.MCX_COMM
  4. Historical data payload structure for MCX futures (5-year capable)
  5. Order placement routing to MCX_COMM with commodity lot sizing
"""

import pytest
import unittest.mock as mock
import pandas as pd
from broker.dhan_broker import DhanBroker, Seg
from broker.factory import get_broker, get_active_broker_name


def test_dhan_broker_factory_default(monkeypatch):
    monkeypatch.delenv("BROKER", raising=False)
    # Reset singleton
    import broker.factory as bf
    bf._broker_instance = None
    broker = get_broker()
    assert broker.broker_name == "dhan"
    assert get_active_broker_name() == "dhan"


def test_dhan_mcx_security_id_resolution():
    broker = DhanBroker(client_id="TEST_CLIENT", access_token="TEST_TOKEN")
    # Pre-defined commodity futures
    silvermic_sec = broker.get_instrument_key("SILVERMIC", exchange="MCX")
    assert silvermic_sec == "562058"

    # Contract symbol matching
    contract_sec = broker.get_instrument_key("SILVERMIC26NOVFUT", exchange="MCX")
    assert contract_sec == "562058"

    # Other commodities
    assert broker.get_instrument_key("GOLD", exchange="MCX") == "495214"
    assert broker.get_instrument_key("CRUDEOIL", exchange="MCX") == "562060"
    assert broker.get_instrument_key("NATURALGAS", exchange="MCX") == "562061"

    # Numeric security ID pass-through
    assert broker.get_instrument_key("562058") == "562058"


def test_dhan_mcx_ltp_routing(monkeypatch):
    broker = DhanBroker(client_id="TEST_CLIENT", access_token="TEST_TOKEN")
    
    # Mock requests.post
    mock_post = mock.MagicMock()
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "data": {
            Seg.MCX_COMM: {
                "562058": {"last_price": 85450.0}
            }
        }
    }
    mock_post.return_value = mock_resp
    monkeypatch.setattr("requests.post", mock_post)

    ltp = broker.get_ltp("SILVERMIC")
    assert ltp == 85450.0

    # Verify call payload sent MCX_COMM
    called_payload = mock_post.call_args[1]["json"]
    assert Seg.MCX_COMM in called_payload
    assert called_payload[Seg.MCX_COMM] == ["562058"]


def test_dhan_mcx_historical_chunk_payload(monkeypatch):
    broker = DhanBroker(client_id="TEST_CLIENT", access_token="TEST_TOKEN")

    mock_post = mock.MagicMock()
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": {
            "open": [85000.0, 85200.0],
            "high": [85500.0, 85600.0],
            "low": [84900.0, 85100.0],
            "close": [85400.0, 85500.0],
            "volume": [120, 150],
            "timestamp": [1725422400000, 1725422700000]
        }
    }
    mock_post.return_value = mock_resp
    monkeypatch.setattr("requests.post", mock_post)

    # Test daily chunk (used for 5-year historical backtest data)
    df_daily = broker._fetch_historical_chunk("SILVERMIC", "day", "2024-01-01", "2024-01-10")
    assert not df_daily.empty
    assert len(df_daily) == 2

    # Check payload sent to Dhan API
    called_payload = mock_post.call_args[1]["json"]
    assert called_payload["exchangeSegment"] == Seg.MCX_COMM
    assert called_payload["instrument"] == "FUTCOM"
    assert called_payload["securityId"] == "562058"


def test_dhan_mcx_order_placement_and_lot_sizing(monkeypatch):
    broker = DhanBroker(client_id="TEST_CLIENT", access_token="TEST_TOKEN")

    mock_post = mock.MagicMock()
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"orderId": "MCX_1001", "orderStatus": "PLACED"}
    mock_post.return_value = mock_resp
    monkeypatch.setattr(broker._session, "post", mock_post)

    # Placing market order on SILVERMIC (lot size = 1)
    result = broker.place_market_order(
        symbol="SILVERMIC",
        quantity=1,
        transaction="BUY",
        product="MIS",
        exchange="MCX",
    )

    assert result.status == "PLACED"
    assert result.order_id == "MCX_1001"

    called_body = mock_post.call_args[1]["json"]
    assert called_body["exchangeSegment"] == Seg.MCX_COMM
    assert called_body["productType"] == "INTRADAY"
    assert called_body["securityId"] == "562058"
    assert called_body["quantity"] == 1

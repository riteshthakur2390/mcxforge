from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock
import pytest
import pytz

from data.oi_recorder import OIRecorder
from broker.base_broker import OptionContract

IST = pytz.timezone("Asia/Kolkata")


def test_oi_recorder_initialization(tmp_path):
    broker = MagicMock()
    recorder = OIRecorder(broker, oi_dir=str(tmp_path / "oi_history"))
    assert recorder.oi_dir.exists()
    assert recorder.broker == broker


def test_oi_recorder_strike_step():
    broker = MagicMock()
    recorder = OIRecorder(broker)
    step_silver = recorder._get_strike_step("SILVERM")
    assert step_silver in (250, 500, 1000)

    step_crude = recorder._get_strike_step("CRUDEOIL")
    assert step_crude in (50, 100)


def test_oi_recorder_fetch_contract_quotes_calls_broker():
    broker = MagicMock()
    broker.get_option_contracts.return_value = [
        OptionContract(
            symbol="SILVERM26NOVFUT",
            strike=240000,
            option_type="CE",
            expiry_date="2026-11-26",
            last_price=500.0,
            open_interest=1200,
            implied_volatility=0.22,
        )
    ]
    recorder = OIRecorder(broker)
    quotes = recorder._fetch_contract_quotes([240000], date(2026, 11, 26))

    assert (240000, "CE") in quotes
    assert quotes[(240000, "CE")]["oi"] == 1200
    assert quotes[(240000, "CE")]["iv"] == 0.22
    assert broker.get_option_contracts.called


def test_oi_recorder_record_mocked(tmp_path):
    broker = MagicMock()
    broker.get_option_contracts.return_value = [
        OptionContract(
            symbol="SILVERM26NOVFUT",
            strike=240000,
            option_type="CE",
            expiry_date="2026-11-26",
            last_price=500.0,
            open_interest=5000,
            implied_volatility=0.20,
        ),
        OptionContract(
            symbol="SILVERM26NOVFUT",
            strike=240000,
            option_type="PE",
            expiry_date="2026-11-26",
            last_price=450.0,
            open_interest=6000,
            implied_volatility=0.21,
        ),
    ]
    recorder = OIRecorder(broker, oi_dir=str(tmp_path / "oi_history"))
    snapshot = recorder.record(240000.0, datetime.now(IST))
    assert snapshot is not None
    assert snapshot["atm_strike"] == 240000
    assert snapshot["total_ce_oi"] >= 5000
    assert snapshot["total_pe_oi"] >= 6000
    assert snapshot["pcr"] > 0

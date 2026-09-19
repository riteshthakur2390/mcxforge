import pytest
from utils.live_trade_history import LiveTradeHistory
from utils.equity_curve import get_observe_equity_curve
from agents_code.agent3_ml.filter import MLFilterAgent
from instruments.registry import get_instrument_config, INSTRUMENT_CATALOG
from agents_code.agent8_dashboard.app import DashboardAlertAgent


def test_commodity_params_all_four_configured():
    for sym in ["SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"]:
        assert sym in INSTRUMENT_CATALOG
        inst = get_instrument_config(sym)
        assert inst.strike_step > 0
        assert inst.lot_size > 0
        assert inst.tick_size > 0


def test_candidate_model_paths_symbol_specific():
    for sym in ["SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"]:
        paths = MLFilterAgent._candidate_model_paths("5m", sym)
        assert len(paths) >= 2
        # Verify symbol or base symbol is present in the first candidate
        first_candidate = str(paths[0][0]).lower()
        assert sym.lower() in first_candidate or sym.replace("M", "").lower() in first_candidate


def test_all_open_positions_payload():
    agent = DashboardAlertAgent()
    positions = agent._all_open_positions_payload()
    for sym in ["SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"]:
        assert sym in positions
        # Initially in test/flat environment it should be None
        assert positions[sym] is None or isinstance(positions[sym], dict)


def test_equity_curve_commodity_isolation():
    curve_mgr = get_observe_equity_curve()
    all_summary = curve_mgr.get_summary()
    assert all_summary["starting_capital"] == 500000.0

    curves = curve_mgr.commodity_equity_curves()
    for sym in ["ALL", "SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"]:
        assert sym in curves
        curve = curves[sym]
        assert "starting_capital" in curve
        assert "current_equity" in curve
        assert "history" in curve
        if sym == "ALL":
            assert curve["starting_capital"] == 500000.0
        else:
            assert curve["starting_capital"] == 100000.0


def test_live_trade_history_commodity_summaries():
    history = LiveTradeHistory()
    summaries = history.commodity_summaries()
    for sym in ["ALL", "SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"]:
        assert sym in summaries
        entry = summaries[sym]
        assert "summary" in entry
        assert "trades" in entry
        summ = entry["summary"]
        assert "today" in summ
        assert "month" in summ
        assert "all" in summ
        assert "win_rate" in summ["all"]
        assert "net_pnl" in summ["all"]


def test_ml_get_ensemble_for_symbol():
    agent = MLFilterAgent()
    for sym in ["SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"]:
        ens = agent._get_ensemble_for_symbol(sym)
        assert ens is not None
        assert ens.is_trained
        assert len(ens.models) >= 1


def test_today_signal_feed_payload_symbol_filter():
    agent = DashboardAlertAgent()
    payload_all = agent._today_signal_feed_payload(limit=50)
    assert "rows" in payload_all
    assert "taken_rows" in payload_all

    payload_gold = agent._today_signal_feed_payload(limit=50, symbol="GOLDM")
    assert "rows" in payload_gold
    for row in payload_gold["rows"]:
        assert str(row.get("symbol") or "").upper() == "GOLDM"


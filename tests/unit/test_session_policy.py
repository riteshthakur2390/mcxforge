"""
tests/unit/test_session_policy.py — Session-Aware Policy & Decoupling Tests
"""

import pytest
from datetime import datetime, time, date
import pytz

from config.settings.modules.session_policy import (
    get_session_policy,
    is_macro_news_freeze_window,
    is_expiry_option_freeze,
    MORNING_POLICY,
    EVENING_POLICY,
)

IST = pytz.timezone("Asia/Kolkata")


def test_session_policy_detection():
    # Morning: 10:30 IST
    t_morning = IST.localize(datetime(2026, 9, 18, 10, 30))
    p_morning = get_session_policy(t_morning)
    assert p_morning.session_name == "MORNING"
    assert p_morning.min_strategy_votes == 5
    assert p_morning.ml_min_confidence == 0.35
    assert p_morning.max_lots == 1
    assert p_morning.target_exit_mode == "QUICK_SCALP"
    assert "VWAPMeanReversion" in p_morning.preferred_strategies
    assert "DonchianBreakout" in p_morning.discouraged_strategies

    # Evening: 19:45 IST
    t_evening = IST.localize(datetime(2026, 9, 18, 19, 45))
    p_evening = get_session_policy(t_evening)
    assert p_evening.session_name == "EVENING"
    assert p_evening.min_strategy_votes == 4
    assert p_evening.ml_min_confidence == 0.22
    assert p_evening.max_lots == 3
    assert p_evening.target_exit_mode == "RUNNER_LADDER"
    assert "TrendFollowing" in p_evening.preferred_strategies


def test_macro_news_freeze_window():
    # Wednesday 20:00 IST -> EIA Crude Oil Inventories
    wed_2000 = IST.localize(datetime(2026, 9, 16, 20, 0))  # 2026-09-16 is a Wednesday
    is_frozen, reason = is_macro_news_freeze_window(wed_2000, symbol="CRUDEOILM")
    assert is_frozen is True
    assert "EIA Crude Oil Inventories" in reason

    # Wednesday 20:00 IST for Silver -> Major general data release window
    is_frozen_silver, reason_silver = is_macro_news_freeze_window(wed_2000, symbol="SILVERM")
    assert is_frozen_silver is True
    assert "FOMC" in reason_silver or "Major Data" in reason_silver

    # Thursday 20:00 IST -> EIA Natural Gas Storage
    thu_2000 = IST.localize(datetime(2026, 9, 17, 20, 2))  # 2026-09-17 is a Thursday
    is_frozen_gas, reason_gas = is_macro_news_freeze_window(thu_2000, symbol="NATGASM")
    assert is_frozen_gas is True
    assert "Natural Gas" in reason_gas

    # Normal time: 15:30 IST -> No freeze
    norm_time = IST.localize(datetime(2026, 9, 18, 15, 30))
    is_frozen_norm, _ = is_macro_news_freeze_window(norm_time, symbol="SILVERM")
    assert is_frozen_norm is False


def test_expiry_option_freeze():
    exp_date = date(2026, 9, 24)

    # Expiry day at 15:00 IST -> Allowed
    t_allowed = IST.localize(datetime(2026, 9, 24, 15, 0))
    is_frozen, _ = is_expiry_option_freeze(t_allowed, exp_date)
    assert is_frozen is False

    # Expiry day at 21:15 IST -> Frozen (Theta trap protection)
    t_frozen = IST.localize(datetime(2026, 9, 24, 21, 15))
    is_frozen, reason = is_expiry_option_freeze(t_frozen, exp_date)
    assert is_frozen is True
    assert "21:00" in reason

    # Non-expiry day at 21:15 IST -> Allowed
    t_non_exp = IST.localize(datetime(2026, 9, 23, 21, 15))
    is_frozen_non, _ = is_expiry_option_freeze(t_non_exp, exp_date)
    assert is_frozen_non is False

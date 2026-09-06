import pytest
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

from agents_code.agent2_strategy.runner import (
    Direction,
    StrategyAgent,
    STRATEGY_CATEGORY_MAPPING,
)

def test_pnl_maximizer_anchors():
    anchors = {"VolumeProfile", "RangeSpread", "FVG", "ElliottWave"}
    # Verify all anchors exist in category mapping
    for a in anchors:
        assert a in STRATEGY_CATEGORY_MAPPING

def test_pnl_maximizer_logic():
    # Helper to evaluate PNL_MAXIMIZER_V1 logic
    def evaluate_maximizer(strategies, market_ts_str, votes):
        strat_names = set(strategies)
        dt = datetime.strptime(market_ts_str, "%Y-%m-%d %H:%M").replace(tzinfo=IST)
        dow = dt.strftime("%A")
        now_tod = dt.hour + dt.minute / 60.0

        anchors = {"VolumeProfile", "RangeSpread", "FVG", "ElliottWave"}
        has_anchor = bool(strat_names.intersection(anchors))

        if 10.0 <= now_tod < 11.5 and votes < 6:
            return False, "morning_trap_low_votes"
        elif dow == "Friday" and now_tod >= 13.0:
            return False, "friday_afternoon_chop"
        elif not has_anchor and votes < 5:
            return False, "missing_structural_anchor"
        elif "ADX+PSAR" in strat_names and not has_anchor:
            return False, "toxic_pair_no_anchor"
        return True, "PASS"

    # Case 1: Morning trap with 4 votes -> Reject
    passed, reason = evaluate_maximizer(["SuperTrend+RSI", "EMASlope", "ValueArea", "BBSqueeze"], "2021-06-22 10:45", 4)
    assert not passed
    assert reason == "morning_trap_low_votes"

    # Case 2: Friday afternoon chop -> Reject
    passed, reason = evaluate_maximizer(["VolumeProfile", "SuperTrend+RSI", "EMASlope", "ValueArea", "RangeSpread"], "2021-06-25 13:30", 5)
    assert not passed
    assert reason == "friday_afternoon_chop"

    # Case 3: ADX+PSAR without anchor -> Reject
    passed, reason = evaluate_maximizer(["ADX+PSAR", "SuperTrend+RSI", "EMASlope", "ValueArea", "BBSqueeze"], "2021-06-22 12:10", 5)
    assert not passed
    assert reason == "toxic_pair_no_anchor"

    # Case 4: Strong setup with VolumeProfile anchor -> PASS
    passed, reason = evaluate_maximizer(["VolumeProfile", "SuperTrend+RSI", "EMASlope", "BBSqueeze"], "2021-06-22 12:10", 4)
    assert passed
    assert reason == "PASS"

    # Case 5: Morning setup with 6 votes -> PASS
    passed, reason = evaluate_maximizer(["VolumeProfile", "RangeSpread", "SuperTrend+RSI", "EMASlope", "BBSqueeze", "SkewHunter"], "2021-06-22 10:45", 6)
    assert passed
    assert reason == "PASS"

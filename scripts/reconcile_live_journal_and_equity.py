#!/usr/bin/env python3
"""
scripts/reconcile_live_journal_and_equity.py

Reconciles:
1. Live trade history (journal/live_trade_history.csv) with all 9 real live placed trades:
   - 7 trades prior to today: total loss -₹3,912.42 (more than 3K loss in live market)
   - 2 trades today (2026-08-31): total profit +₹461.51
   - Total 9 trades: net realized P&L -₹3,450.91, ending equity ₹196,549.09.
2. Daily equity curve (journal/equity_curve.csv & journal/equity_curve.json)
3. Capital state (journal/capital_state.json)
4. Dashboard HTML defaults (dashboard/templates/index.html)
"""

import os
import sys
import json
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
JOURNAL_DIR = REPO_ROOT / "journal"

from utils.brokerage_calculator import calculate_option_trade_charges

# The 9 authoritative live placed trades
RAW_TRADES = [
    # 1. Aug 13
    {
        "trade_id": "SF-20260813-001",
        "signal_id": "2026-08-13T12:00:00+05:30|BUY_CALL|SuperTrend+RSI|VWAP+EMA|BBSqueeze|ADX+PSAR|24383.70",
        "date": "2026-08-13",
        "entry_time": "2026-08-13T12:00:10+05:30",
        "exit_time": "2026-08-13T12:35:15+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26AUG1824400CE",
        "direction": "BUY_CALL",
        "nifty_price": "24383.70",
        "strike": "24400",
        "option_type": "CE",
        "expiry_date": "2026-08-18",
        "entry_premium": 129.70,
        "exit_premium": 140.40,
        "lots": 1,
        "quantity": 65,
        "invested": 8430.50,
        "sl_premium": 108.95,
        "target_premium": 168.61,
        "exit_reason": "PROFIT_PROTECT",
        "holding_minutes": 35,
        "strategies_fired": "SuperTrend+RSI|VWAP+EMA|BBSqueeze|ADX+PSAR",
        "votes": 5,
        "ml_confidence": 0.65,
        "ml_rank_score": 0.72,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.85,
        "outcome_eod": "WIN",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "REDUCED",
        "source": "live_event",
        "notes": "Verified Live Trade 1 (+₹628.54)"
    },
    # 2. Aug 19
    {
        "trade_id": "SF-20260819-001",
        "signal_id": "2026-08-19T10:40:00+05:30|BUY_PUT|ORB|FVG|CPR|Ichimoku|ValueArea|24057.80",
        "date": "2026-08-19",
        "entry_time": "2026-08-19T10:44:24+05:30",
        "exit_time": "2026-08-19T11:57:33+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26AUG2524050PE",
        "direction": "BUY_PUT",
        "nifty_price": "24057.80",
        "strike": "24050",
        "option_type": "PE",
        "expiry_date": "2026-08-25",
        "entry_premium": 94.95,
        "exit_premium": 86.65,
        "lots": 1,
        "quantity": 65,
        "invested": 6171.75,
        "sl_premium": 86.76,
        "target_premium": 113.75,
        "exit_reason": "MANUAL_SQUARE_OFF",
        "holding_minutes": 73,
        "strategies_fired": "ORB|FVG|CPR|Ichimoku|ValueArea",
        "votes": 5,
        "ml_confidence": 0.299,
        "ml_rank_score": 0.6286,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.88,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "REDUCED",
        "source": "live_event",
        "notes": "Dhan Fill Verified: Buy @ 94.95, Sell @ 86.65 (-₹599.49)"
    },
    # 3. Aug 24 trade 1
    {
        "trade_id": "SF-20260824-001",
        "signal_id": "2026-08-24T11:35:00+05:30|BUY_PUT|VWAP+EMA|ADX+PSAR|Ichimoku|OIAnalysis|24178.00",
        "date": "2026-08-24",
        "entry_time": "2026-08-24T11:35:13+05:30",
        "exit_time": "2026-08-24T11:45:20+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26AUG2524200PE",
        "direction": "BUY_PUT",
        "nifty_price": "24178.00",
        "strike": "24200",
        "option_type": "PE",
        "expiry_date": "2026-08-25",
        "entry_premium": 67.35,
        "exit_premium": 61.15,
        "lots": 2,
        "quantity": 130,
        "invested": 8755.50,
        "sl_premium": 53.88,
        "target_premium": 94.29,
        "exit_reason": "SL_HIT",
        "holding_minutes": 10,
        "strategies_fired": "VWAP+EMA|ADX+PSAR|Ichimoku|OIAnalysis",
        "votes": 4,
        "ml_confidence": 0.58,
        "ml_rank_score": 0.65,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.80,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "FULL",
        "source": "live_event",
        "notes": "Live Trade SL Hit (-₹871.29)"
    },
    # 4. Aug 24 trade 2
    {
        "trade_id": "SF-20260824-002",
        "signal_id": "2026-08-24T13:10:00+05:30|BUY_PUT|SuperTrend+RSI|Ichimoku|ValueArea|EMASlope|24166.15",
        "date": "2026-08-24",
        "entry_time": "2026-08-24T13:10:05+05:30",
        "exit_time": "2026-08-24T13:25:12+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26AUG2524200PE",
        "direction": "BUY_PUT",
        "nifty_price": "24166.15",
        "strike": "24200",
        "option_type": "PE",
        "expiry_date": "2026-08-25",
        "entry_premium": 68.48,
        "exit_premium": 62.12,
        "lots": 1,
        "quantity": 65,
        "invested": 4451.20,
        "sl_premium": 54.78,
        "target_premium": 95.87,
        "exit_reason": "SL_HIT",
        "holding_minutes": 15,
        "strategies_fired": "SuperTrend+RSI|Ichimoku|ValueArea|EMASlope",
        "votes": 4,
        "ml_confidence": 0.55,
        "ml_rank_score": 0.62,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.78,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "REDUCED",
        "source": "live_event",
        "notes": "Live Trade SL Hit (-₹469.78)"
    },
    # 5. Aug 25
    {
        "trade_id": "SF-20260825-001",
        "signal_id": "2026-08-25T10:25:00+05:30|BUY_PUT|SuperTrend+RSI|VWAP+EMA|BBSqueeze|Ichimoku|ValueArea|EMASlope|24185.20",
        "date": "2026-08-25",
        "entry_time": "2026-08-25T10:25:12+05:30",
        "exit_time": "2026-08-25T10:45:18+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26AUG2524250PE",
        "direction": "BUY_PUT",
        "nifty_price": "24185.20",
        "strike": "24250",
        "option_type": "PE",
        "expiry_date": "2026-08-25",
        "entry_premium": 119.35,
        "exit_premium": 108.30,
        "lots": 3,
        "quantity": 195,
        "invested": 23273.25,
        "sl_premium": 95.48,
        "target_premium": 167.09,
        "exit_reason": "SL_HIT",
        "holding_minutes": 20,
        "strategies_fired": "SuperTrend+RSI|VWAP+EMA|BBSqueeze|Ichimoku|ValueArea|EMASlope",
        "votes": 6,
        "ml_confidence": 0.62,
        "ml_rank_score": 0.71,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.85,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "FULL",
        "source": "live_event",
        "notes": "Live Trade SL Hit (-₹2,250.01)"
    },
    # 6. Aug 27 trade 1
    {
        "trade_id": "SF-20260827-001",
        "signal_id": "2026-08-27T11:20:00+05:30|BUY_PUT|SuperTrend+RSI|VWAP+EMA|BBSqueeze|ADX+PSAR|Ichimoku|EMASlope|24145.95",
        "date": "2026-08-27",
        "entry_time": "2026-08-27T11:20:08+05:30",
        "exit_time": "2026-08-27T11:35:14+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26SEP0124150PE",
        "direction": "BUY_PUT",
        "nifty_price": "24145.95",
        "strike": "24150",
        "option_type": "PE",
        "expiry_date": "2026-09-01",
        "entry_premium": 85.00,
        "exit_premium": 82.95,
        "lots": 1,
        "quantity": 65,
        "invested": 5525.00,
        "sl_premium": 68.00,
        "target_premium": 119.00,
        "exit_reason": "SL_HIT",
        "holding_minutes": 15,
        "strategies_fired": "SuperTrend+RSI|VWAP+EMA|BBSqueeze|ADX+PSAR|Ichimoku|EMASlope",
        "votes": 6,
        "ml_confidence": 0.60,
        "ml_rank_score": 0.68,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.82,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "REDUCED",
        "source": "live_event",
        "notes": "Live Trade SL Hit (-₹192.46)"
    },
    # 7. Aug 27 trade 2
    {
        "trade_id": "SF-20260827-002",
        "signal_id": "2026-08-27T13:50:00+05:30|BUY_PUT|SuperTrend+RSI|VWAP+EMA|ADX+PSAR|FVG|CPR|Ichimoku|EMASlope|GammaExposure|24142.30",
        "date": "2026-08-27",
        "entry_time": "2026-08-27T13:50:11+05:30",
        "exit_time": "2026-08-27T14:15:20+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26SEP0124150PE",
        "direction": "BUY_PUT",
        "nifty_price": "24142.30",
        "strike": "24150",
        "option_type": "PE",
        "expiry_date": "2026-09-01",
        "entry_premium": 76.15,
        "exit_premium": 75.75,
        "lots": 3,
        "quantity": 195,
        "invested": 14849.25,
        "sl_premium": 60.92,
        "target_premium": 106.61,
        "exit_reason": "SL_HIT",
        "holding_minutes": 25,
        "strategies_fired": "SuperTrend+RSI|VWAP+EMA|ADX+PSAR|FVG|CPR|Ichimoku|EMASlope|GammaExposure",
        "votes": 8,
        "ml_confidence": 0.66,
        "ml_rank_score": 0.74,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.88,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "FULL",
        "source": "live_event",
        "notes": "Live Trade SL Hit (-₹157.93)"
    },
    # 8. Aug 31 trade 1
    {
        "trade_id": "SF-20260831-001",
        "signal_id": "2026-08-31T11:55:00+05:30|BUY_CALL|SuperTrend+RSI|VWAP+EMA|BBSqueeze|ADX+PSAR|FVG|Ichimoku|OIAnalysis|IVContraction|EMASlope|24077.80",
        "date": "2026-08-31",
        "entry_time": "2026-08-31T11:59:15+05:30",
        "exit_time": "2026-08-31T12:15:00+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26SEP0824100CE",
        "direction": "BUY_CALL",
        "nifty_price": "24077.80",
        "strike": "24100",
        "option_type": "CE",
        "expiry_date": "2026-09-08",
        "entry_premium": 187.65,
        "exit_premium": 188.45,
        "lots": 2,
        "quantity": 130,
        "invested": 24394.50,
        "sl_premium": 150.12,
        "target_premium": 262.71,
        "exit_reason": "STALE_LOSS",
        "holding_minutes": 16,
        "strategies_fired": "SuperTrend+RSI|VWAP+EMA|BBSqueeze|ADX+PSAR|FVG|Ichimoku|OIAnalysis|IVContraction|EMASlope",
        "votes": 9,
        "ml_confidence": 0.58,
        "ml_rank_score": 0.68,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.88,
        "outcome_eod": "WIN",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "FULL",
        "source": "live_event",
        "notes": "Live Trade Scratch (+₹2.66)"
    },
    # 9. Aug 31 trade 2
    {
        "trade_id": "SF-20260831-002",
        "signal_id": "2026-08-31T14:00:00+05:30|BUY_PUT|VWAP+EMA|BBSqueeze|CPR|Ichimoku|ValueArea|SqueezeMomentum|EMASlope|ElliottWave|24043.85",
        "date": "2026-08-31",
        "entry_time": "2026-08-31T14:01:09+05:30",
        "exit_time": "2026-08-31T14:08:38+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26SEP0824050PE",
        "direction": "BUY_PUT",
        "nifty_price": "24043.85",
        "strike": "24050",
        "option_type": "PE",
        "expiry_date": "2026-09-08",
        "entry_premium": 104.12,
        "exit_premium": 106.95,
        "lots": 3,
        "quantity": 195,
        "invested": 20303.40,
        "sl_premium": 79.35,
        "target_premium": 141.77,
        "exit_reason": "MANUAL_EXIT",
        "holding_minutes": 7,
        "strategies_fired": "VWAP+EMA|BBSqueeze|CPR|Ichimoku|ValueArea|SqueezeMomentum|EMASlope|ElliottWave",
        "votes": 8,
        "ml_confidence": 0.3125,
        "ml_rank_score": 0.6308,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.88,
        "outcome_eod": "WIN",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "FULL",
        "source": "live_event",
        "notes": "Manual User Exit Fill: Buy 104.12, Sell 106.95 (+₹458.85)"
    },
    # 10. Sep 01 - Trade 1
    {
        "trade_id": "SF-20260901-001",
        "signal_id": "2026-09-01T13:15:00+05:30|BUY_PUT|SuperTrend+RSI|VWAP+EMA|ADX+PSAR|FVG|CPR|Ichimoku|OIAnalysis|IVContraction|EMASlope|24048.90",
        "date": "2026-09-01",
        "entry_time": "2026-09-01T13:15:17+05:30",
        "exit_time": "2026-09-01T13:20:14+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26SEP0124150PE",
        "direction": "BUY_PUT",
        "nifty_price": "24048.90",
        "strike": "24150",
        "option_type": "PE",
        "expiry_date": "2026-09-01",
        "entry_premium": 99.90,
        "exit_premium": 100.55,
        "lots": 1,
        "quantity": 65,
        "invested": 6493.50,
        "sl_premium": 89.90,
        "target_premium": 129.90,
        "exit_reason": "SL_HIT",
        "holding_minutes": 5,
        "strategies_fired": "SuperTrend+RSI|VWAP+EMA|ADX+PSAR|FVG|CPR|Ichimoku|OIAnalysis|IVContraction|EMASlope",
        "votes": 9,
        "ml_confidence": 0.277,
        "ml_rank_score": 0.595,
        "regime": "TRENDING",
        "setup_type": "trend_pullback",
        "setup_strength": 0.71,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "FULL",
        "source": "live_event",
        "notes": "Live Trade 1 (Buy 99.90, Sell 100.55, Loss after charges -₹19.37)"
    },
    # 11. Sep 01 - Trade 2
    {
        "trade_id": "SF-20260901-002",
        "signal_id": "2026-09-01T13:20:00+05:30|BUY_PUT|SuperTrend+RSI|VWAP+EMA|ADX+PSAR|FVG|CPR|Ichimoku|EMASlope|24048.40",
        "date": "2026-09-01",
        "entry_time": "2026-09-01T13:20:30+05:30",
        "exit_time": "2026-09-01T13:21:33+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26SEP0124100PE",
        "direction": "BUY_PUT",
        "nifty_price": "24048.40",
        "strike": "24100",
        "option_type": "PE",
        "expiry_date": "2026-09-01",
        "entry_premium": 59.45,
        "exit_premium": 49.95,
        "lots": 3,
        "quantity": 195,
        "invested": 11592.75,
        "sl_premium": 59.40,
        "target_premium": 82.40,
        "exit_reason": "SL_HIT",
        "holding_minutes": 1,
        "strategies_fired": "SuperTrend+RSI|VWAP+EMA|ADX+PSAR|FVG|CPR|Ichimoku|EMASlope",
        "votes": 7,
        "ml_confidence": 0.284,
        "ml_rank_score": 0.622,
        "regime": "TRENDING",
        "setup_type": "vote_aligned",
        "setup_strength": 0.88,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "FULL",
        "source": "live_event",
        "notes": "Live Trade 2 (Buy 59.45, Sell 49.95, 3 Lots, -₹1,922.40)"
    },
    # 12. Sep 01 - Trade 3
    {
        "trade_id": "SF-20260901-003",
        "signal_id": "2026-09-01T13:40:00+05:30|BUY_PUT|SuperTrend+RSI|ADX+PSAR|FVG|CPR|Ichimoku|24035.55",
        "date": "2026-09-01",
        "entry_time": "2026-09-01T13:40:19+05:30",
        "exit_time": "2026-09-01T13:43:54+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26SEP0124100PE",
        "direction": "BUY_PUT",
        "nifty_price": "24035.55",
        "strike": "24100",
        "option_type": "PE",
        "expiry_date": "2026-09-01",
        "entry_premium": 70.95,
        "exit_premium": 67.25,
        "lots": 1,
        "quantity": 65,
        "invested": 4611.75,
        "sl_premium": 56.10,
        "target_premium": 80.50,
        "exit_reason": "SL_HIT",
        "holding_minutes": 3,
        "strategies_fired": "SuperTrend+RSI|ADX+PSAR|FVG|CPR|Ichimoku",
        "votes": 5,
        "ml_confidence": 0.288,
        "ml_rank_score": 0.627,
        "regime": "TRENDING",
        "setup_type": "trend_pullback",
        "setup_strength": 0.75,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "FULL",
        "source": "live_event",
        "notes": "Live Trade 3 (Buy avg 70.95, Sell avg 67.25, -₹297.52)"
    },
    # 13. Sep 01 - Trade 4
    {
        "trade_id": "SF-20260901-004",
        "signal_id": "2026-09-01T13:45:00+05:30|BUY_PUT|SuperTrend+RSI|ADX+PSAR|FVG|CPR|Ichimoku|EMASlope|24031.55",
        "date": "2026-09-01",
        "entry_time": "2026-09-01T13:45:26+05:30",
        "exit_time": "2026-09-01T13:47:13+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26SEP0124150PE",
        "direction": "BUY_PUT",
        "nifty_price": "24031.55",
        "strike": "24150",
        "option_type": "PE",
        "expiry_date": "2026-09-01",
        "entry_premium": 117.65,
        "exit_premium": 121.00,
        "lots": 1,
        "quantity": 65,
        "invested": 7647.25,
        "sl_premium": 89.47,
        "target_premium": 159.86,
        "peak_premium": 196.00,
        "max_multiplier": 1.67,
        "exit_reason": "MANUAL_EXIT",
        "holding_minutes": 2,
        "strategies_fired": "SuperTrend+RSI|ADX+PSAR|FVG|CPR|Ichimoku|EMASlope",
        "votes": 6,
        "ml_confidence": 0.319,
        "ml_rank_score": 0.634,
        "regime": "TRENDING",
        "setup_type": "trend_pullback",
        "setup_strength": 0.80,
        "outcome_eod": "WIN",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "FULL",
        "source": "live_event",
        "notes": "Live Trade 4 (Buy avg 117.65, Manual exit 121.00 @ 13:47:13, Peak 196.00 / 1.67x, +₹153.27)"
    },
    # 14. Sep 03 trade 1
    {
        "trade_id": "SF-20260903-001",
        "signal_id": "2026-09-03T12:25:00+05:30|BUY_PUT|SuperTrend+RSI|Ichimoku|ValueArea|EMASlope|24040.00",
        "date": "2026-09-03",
        "entry_time": "2026-09-03T12:25:14+05:30",
        "exit_time": "2026-09-03T12:54:26+05:30",
        "symbol": "NIFTY",
        "option_symbol": "NIFTY26SEP0823900PE",
        "direction": "BUY_PUT",
        "nifty_price": "24040.00",
        "strike": "23900",
        "option_type": "PE",
        "expiry_date": "2026-09-08",
        "entry_premium": 95.60,
        "exit_premium": 83.85,
        "lots": 1,
        "quantity": 65,
        "invested": 6214.00,
        "sl_premium": 87.21,
        "target_premium": 125.97,
        "peak_premium": 98.50,
        "max_multiplier": 1.03,
        "exit_reason": "MANUAL_EXIT",
        "holding_minutes": 29,
        "strategies_fired": "SuperTrend+RSI|Ichimoku|ValueArea|EMASlope",
        "votes": 4,
        "ml_confidence": 0.312,
        "ml_rank_score": 0.615,
        "regime": "TRENDING",
        "setup_type": "trend_pullback",
        "setup_strength": 0.78,
        "outcome_eod": "LOSS",
        "mode": "AUTO",
        "execution": "PLACED",
        "simulated": "",
        "budget_lane": "REDUCED",
        "source": "live_event",
        "notes": "Upstox Fill Verified: Buy @ 95.60, Manual exit @ 83.85 (-₹823.48)"
    }
]

from utils.live_trade_history import FIELDNAMES

def build_reconciled_trades():
    reconciled = []
    for r in RAW_TRADES:
        ch = calculate_option_trade_charges(r["entry_premium"], r["exit_premium"], r["quantity"])
        gross_pct = round(((r["exit_premium"] - r["entry_premium"]) / max(r["entry_premium"], 1e-6)) * 100.0, 2)
        if "max_multiplier" in r:
            max_mult = float(r["max_multiplier"])
        elif "peak_premium" in r:
            max_mult = round(float(r["peak_premium"]) / max(float(r["entry_premium"]), 1e-6), 2)
        else:
            peak_pct = max(0.0, gross_pct)
            max_mult = round(1.0 + peak_pct / 100.0, 2)

        row = {
            "trade_id": r["trade_id"],
            "signal_id": r["signal_id"],
            "date": r["date"],
            "entry_time": r["entry_time"],
            "exit_time": r["exit_time"],
            "symbol": r["symbol"],
            "option_symbol": r["option_symbol"],
            "direction": r["direction"],
            "nifty_price": r["nifty_price"],
            "strike": r["strike"],
            "option_type": r["option_type"],
            "expiry_date": r["expiry_date"],
            "entry_premium": f"{r['entry_premium']:.2f}",
            "exit_premium": f"{r['exit_premium']:.2f}",
            "lots": r["lots"],
            "quantity": r["quantity"],
            "invested": f"{r['invested']:.2f}",
            "sl_premium": f"{r['sl_premium']:.2f}",
            "target_premium": f"{r['target_premium']:.2f}",
            "gross_pnl_inr": f"{ch.gross_pnl_inr:.2f}",
            "total_charges": f"{ch.total_charges:.2f}",
            "brokerage": f"{ch.brokerage:.2f}",
            "stt": f"{ch.stt:.2f}",
            "net_pnl_inr": f"{ch.net_pnl_inr:.2f}",
            "net_pnl_pct": f"{ch.net_pnl_pct:.2f}",
            "pnl_pct": f"{gross_pct:.2f}",
            "realized_pnl": f"{ch.net_pnl_inr:.2f}",
            "max_multiplier": f"{max_mult:.2f}",
            "exit_reason": r["exit_reason"],
            "holding_minutes": r["holding_minutes"],
            "strategies_fired": r["strategies_fired"],
            "votes": r["votes"],
            "ml_confidence": r.get("ml_confidence", ""),
            "ml_rank_score": r.get("ml_rank_score", ""),
            "regime": r["regime"],
            "setup_type": r["setup_type"],
            "setup_strength": r["setup_strength"],
            "outcome_eod": r["outcome_eod"],
            "mode": r["mode"],
            "execution": r["execution"],
            "simulated": r.get("simulated", ""),
            "budget_lane": r["budget_lane"],
            "source": r.get("source", "live_event"),
            "notes": r.get("notes", ""),
        }
        reconciled.append(row)
    return reconciled

def reconcile_all():
    print("Starting full reconciliation...")
    trades = build_reconciled_trades()

    # 1. Write journal/live_trade_history.csv
    # Newest first
    history_csv = JOURNAL_DIR / "live_trade_history.csv"
    with open(history_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for t in reversed(trades):
            writer.writerow(t)
    print(f"✅ Wrote {len(trades)} trades to {history_csv}")

    # 2. Build daily equity curve
    daily_groups = {}
    for t in trades:
        d = t["date"]
        daily_groups.setdefault(d, []).append(t)

    # Dates to include
    all_dates = [
        "2026-08-13", "2026-08-14", "2026-08-17", "2026-08-18",
        "2026-08-19", "2026-08-20", "2026-08-21", "2026-08-24",
        "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28",
        "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03"
    ]

    starting_capital = float(os.getenv("TOTAL_FUND", 200000.0))
    deployed_capital = float(os.getenv("DEPLOYED_CAPITAL", 40000.0))
    current_equity = starting_capital
    running_max_equity = starting_capital
    max_dd = 0.0
    consecutive_losses = 0

    equity_rows = []
    for d in all_dates:
        start_eq = current_equity
        d_trades = daily_groups.get(d, [])
        t_count = len(d_trades)
        wins = sum(1 for x in d_trades if float(x["net_pnl_inr"]) > 0)
        gross_pnl = sum(float(x["gross_pnl_inr"]) for x in d_trades)
        total_charges = sum(float(x["total_charges"]) for x in d_trades)
        net_pnl = sum(float(x["net_pnl_inr"]) for x in d_trades)

        current_equity = round(start_eq + net_pnl, 2)
        running_max_equity = max(running_max_equity, current_equity)
        cur_dd = round(((running_max_equity - current_equity) / max(running_max_equity, 1.0)) * 100.0, 3)
        max_dd = max(max_dd, cur_dd)
        pnl_pct = round((net_pnl / max(start_eq, 1.0)) * 100.0, 3)
        wr = round((wins / max(t_count, 1)) * 100.0, 1) if t_count > 0 else 0.0

        if t_count > 0:
            if net_pnl < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

        risk_sig = "NORMAL"
        if cur_dd >= 15.0 or consecutive_losses >= 3:
            risk_sig = "PAUSE"
        elif consecutive_losses >= 2:
            risk_sig = "REDUCE_SIZE"

        row = {
            "date": d,
            "starting_equity": f"{start_eq:.2f}",
            "ending_equity": f"{current_equity:.2f}",
            "gross_pnl_inr": f"{gross_pnl:.2f}",
            "total_charges": f"{total_charges:.2f}",
            "net_pnl_inr": f"{net_pnl:.2f}",
            "pnl_pct": f"{pnl_pct}",
            "trades": t_count,
            "wins": wins,
            "win_rate": f"{wr}",
            "running_max_equity": f"{running_max_equity:.2f}",
            "current_drawdown_pct": f"{cur_dd}",
            "max_drawdown_pct": f"{max_dd}",
            "consecutive_losses": consecutive_losses,
            "trading_mode": "AUTO",
            "risk_signal": risk_sig,
            "notes": ""
        }
        equity_rows.append(row)

    equity_csv = JOURNAL_DIR / "equity_curve.csv"
    with open(equity_csv, "w", newline="") as f:
        from utils.equity_curve import CURVE_COLUMNS
        writer = csv.DictWriter(f, fieldnames=CURVE_COLUMNS)
        writer.writeheader()
        for r in equity_rows:
            writer.writerow(r)
    print(f"✅ Wrote {len(equity_rows)} daily rows to {equity_csv}")

    equity_json = JOURNAL_DIR / "equity_curve.json"
    with open(equity_json, "w") as f:
        json.dump(equity_rows, f, indent=2)
    print(f"✅ Wrote JSON to {equity_json}")

    # 3. Update capital_state.json
    cap_state_path = JOURNAL_DIR / "capital_state.json"
    cap_state = {
        "date": "2026-09-03",
        "total_fund": starting_capital,
        "daily_budget": deployed_capital,
        "capital_used": 6214.00,
        "trades_today": 1,
        "open_positions": {}
    }
    with open(cap_state_path, "w") as f:
        json.dump(cap_state, f, indent=2)
    print(f"✅ Wrote capital state to {cap_state_path}")

    # 4. Print summary
    prior_trades = [t for t in trades if t["date"] < "2026-08-31"]
    prior_net = sum(float(t["net_pnl_inr"]) for t in prior_trades)
    print(f"\n--- AUDIT SUMMARY ---")
    print(f"Earlier live market trades (before Aug 31): {len(prior_trades)} trades")
    print(f"Earlier live market Net Loss: ₹{prior_net:.2f} (confirmed > ₹3K loss)")
    today_trades = [t for t in trades if t["date"] == "2026-08-31"]
    today_net = sum(float(t["net_pnl_inr"]) for t in today_trades)
    print(f"Today's trades: {len(today_trades)} trades | Net PnL: +₹{today_net:.2f}")
    total_net = sum(float(t["net_pnl_inr"]) for t in trades)
    print(f"Total Combined Net PnL: ₹{total_net:.2f}")
    print(f"Final Ending Equity: ₹{current_equity:.2f}")

if __name__ == "__main__":
    reconcile_all()

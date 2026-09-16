"""
utils/trade_logger.py — Standardized Trade Logging for SignalForge
Used to compare live and backtest performance in a single consolidated format.
"""
import csv
import os
from datetime import datetime
from pathlib import Path
from loguru import logger
import pytz

IST = pytz.timezone("Asia/Kolkata")

# Standard columns for trade comparison
TRADE_LOG_COLUMNS = [
    "trade_date",
    "entry_time",
    "exit_time",
    "signal_id",
    "run_id",
    "mode",  # LIVE, OBSERVE, BACKTEST
    "symbol",
    "option_symbol",
    "direction",
    "strike",
    "entry_price",
    "exit_price",
    "pnl_pct",
    "realized_pnl",
    "exit_reason",
    "strategies_fired",
    "ml_conf",
    "ml_rank_score",
    "regime",
    "is_backtest"
]

DEFAULT_TRADE_LOG_PATH = "journal/closed_trades.csv"

def log_trade(payload: dict, is_backtest: bool = False, run_id: str = "", csv_path: str = DEFAULT_TRADE_LOG_PATH):
    """
    Appends a closed trade to the standardized CSV log.
    Accepts payload from POSITION_CLOSED or AnalyticsAgent entry.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    file_exists = os.path.isfile(csv_path)
    
    try:
        # Extract fields from payload
        # It handles both POSITION_CLOSED payload and AnalyticsAgent internal entry format
        entry_time = payload.get("entry_time", "")
        exit_time = payload.get("exit_time", "")
        
        # Parse date from entry_time if possible
        trade_date = ""
        if entry_time:
            try:
                trade_date = datetime.fromisoformat(entry_time.replace("Z", "+00:00")).strftime("%Y-%m-%d")
            except:
                trade_date = str(payload.get("date", ""))

        row = {
            "trade_date": trade_date,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "signal_id": payload.get("signal_id", ""),
            "run_id": run_id or payload.get("run_id", ""),
            "mode": payload.get("mode", payload.get("execution_mode", "UNKNOWN")),
            "symbol": payload.get("symbol") or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM")),
            "option_symbol": payload.get("option_symbol", ""),
            "direction": payload.get("direction", ""),
            "strike": payload.get("strike", ""),
            "entry_price": payload.get("entry_premium", payload.get("actual_premium", 0)),
            "exit_price": payload.get("exit_premium", 0),
            "pnl_pct": payload.get("pnl_pct", 0),
            "realized_pnl": payload.get("realized_pnl", 0),
            "exit_reason": payload.get("exit_reason", ""),
            "strategies_fired": payload.get("strategies_fired", ""),
            "ml_conf": payload.get("ml_conf", payload.get("ml_confidence", 0)),
            "ml_rank_score": payload.get("ml_rank_score", 0),
            "regime": payload.get("regime", ""),
            "is_backtest": is_backtest
        }
        
        # Clean up lists/dicts to strings
        for k, v in row.items():
            if isinstance(v, (list, tuple)):
                row[k] = "|".join(map(str, v))
            elif isinstance(v, dict):
                row[k] = str(v)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
        logger.debug(f"[TradeLogger] Logged trade to {csv_path} | {row['option_symbol']} | {row['pnl_pct']}%")
        return True
    except Exception as e:
        logger.error(f"[TradeLogger] Error logging trade: {e}")
        return False

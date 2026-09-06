"""
scripts/replay.py — Historical Data Replay
===========================================
Replays past NIFTY 5-min candles through all 9 agents.
Safe to run on weekends with no live Kite connection.
Great for validating strategy changes before going live.

Usage:
    python scripts/replay.py --days 5
    python scripts/replay.py --days 30
    python scripts/replay.py --date 2026-03-01
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import asyncio
import argparse
import sys
import os
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from loguru import logger
import pytz

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.bus import get_bus, reset_bus, Topic
from config.settings import (
    DATA_CACHE_DIR, BACKTEST_TIMEFRAME, JOURNAL_DIR, CANDLE_LOOKBACK
)

IST = pytz.timezone("Asia/Kolkata")


async def replay(df: pd.DataFrame, speed_ms: int = 100) -> dict:
    """
    Feed historical candles through the agent pipeline.
    Returns summary statistics.
    """
    # Reset bus for clean replay
    reset_bus()
    prev_mode = os.environ.get("TRADING_MODE")
    os.environ["TRADING_MODE"] = "BACKTEST"

    try:
        # Import and wire agents fresh using the same safe stack as backtest
        from agents_code.agent9_regime.classifier  import MarketRegimeAgent
        from agents_code.agent2_strategy.runner    import StrategyAgent
        from agents_code.agent2_strategy.market_context_gate import get_market_context_gate
        from agents_code.agent3_ml.filter          import MLFilterAgent
        from agents_code.agent4_planner.planner    import TradePlannerAgent
        from agents_code.agent6_position.manager   import PositionManagerAgent
        from agents_code.agent7_analytics.journal  import AnalyticsAgent
        from agents_code.agent10_risk.risk_guard   import RiskGuardAgent
        from backtesting.engine                    import BacktestExecutionAgent

        regime_agent   = MarketRegimeAgent()
        strategy_agent = StrategyAgent()
        market_context_gate = get_market_context_gate(bus=get_bus())
        ml_agent       = MLFilterAgent(enable_file_watcher=False)
        planner_agent  = TradePlannerAgent(data_agent=None, backtest_mode=True)
        exec_agent     = BacktestExecutionAgent()
        pos_agent      = PositionManagerAgent(data_agent=None)
        analytics      = AnalyticsAgent(backtest_mode=True)
        risk_agent     = RiskGuardAgent()

        strategy_agent.set_backtest_mode(True)

        regime_agent.register()
        strategy_agent.register()
        market_context_gate.register()
        ml_agent.register()
        planner_agent.register()
        exec_agent.register()
        pos_agent.register()
        analytics.register()
        risk_agent.register()

        bus = get_bus()

        # Track replay stats
        stats = {
            "candles":    0,
            "signals":    0,
            "suppressed": 0,
            "approved":   0,
            "rejected":   0,
            "wins":       0,
            "losses":     0,
            "days":       set(),
        }

        async def on_signal(msg):
            stats["signals"] += 1

        async def on_suppressed(msg):
            stats["suppressed"] += 1

        async def on_approved(msg):
            stats["approved"] += 1

        async def on_rejected(msg):
            stats["rejected"] += 1

        async def on_closed(msg):
            pnl = float(msg.payload.get("pnl_pct", 0))
            if pnl > 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

        bus.subscribe(Topic.RAW_SIGNAL,        on_signal)
        bus.subscribe(Topic.SIGNAL_SUPPRESSED, on_suppressed)
        bus.subscribe(Topic.SIGNAL_APPROVED,   on_approved)
        bus.subscribe(Topic.SIGNAL_REJECTED,   on_rejected)
        bus.subscribe(Topic.POSITION_CLOSED,   on_closed)

        # Group by day and replay
        df["_date"] = [i.date() for i in df.index]
        trading_days = sorted(df["_date"].unique())

        print(f"\n{'━'*56}")
        print(f"  SignalForge Replay — {len(trading_days)} trading days")
        print(f"  Candles: {len(df)} | Speed: {speed_ms}ms/candle")
        print(f"{'━'*56}")

        for day in trading_days:
            day_df = df[df["_date"] == day].drop(columns=["_date"])
            stats["days"].add(str(day))

            # Simulate pre-market (use static values for replay)
            # Compute actual gap from previous day's close
            prev_close = 0.0
            prev_days = [d for d in trading_days if d < day]
            if prev_days:
                prev_day_df = df[df["_date"] == prev_days[-1]]
                prev_close = float(prev_day_df["close"].iloc[-1]) if len(prev_day_df) else 0
                first_open = float(day_df["open"].iloc[0]) if len(day_df) else 0
                gap_pct = ((first_open - prev_close) / prev_close * 100) if prev_close else 0
            else:
                gap_pct = 0.0
            await bus.publish(Topic.PREMARKET_BIAS, {
                "bias": "NEUTRAL",
                "india_vix": 22.0,
                "gap_pct": round(gap_pct, 3),
                "prev_close": prev_close,
                "timestamp": f"{day}T09:00:00+05:30",
            }, "replay")

            # Replay ORB if we have enough candles
            orb_data = day_df.between_time("09:15", "09:30")
            orb_high = orb_low = None
            if len(orb_data) >= 2:
                orb_high = float(orb_data["high"].max())
                orb_low  = float(orb_data["low"].min())
                await bus.publish(Topic.ORB_FORMED, {
                    "orb_high":  orb_high,
                    "orb_low":   orb_low,
                    "orb_range": round(orb_high - orb_low, 2),
                }, "replay")

            # Replay candles one by one
            for i in range(20, len(day_df)):
                window = day_df.iloc[:i+1]
                candle = day_df.iloc[i]
                ts_str = str(day_df.index[i])
                stats["candles"] += 1

                # FIX: use cross-day rolling window so S8 CPR (and all strategies)
                # have access to previous day candles. Find global position in df
                # and take last CANDLE_LOOKBACK bars from the full dataset.
                global_idx = df.index.get_loc(day_df.index[i])
                lookback = min(global_idx + 1, CANDLE_LOOKBACK)
                window = df.iloc[global_idx - lookback + 1: global_idx + 1]

                candles_list = [
                    {
                        "datetime": str(idx),
                        "open":   float(row["open"]),
                        "high":   float(row["high"]),
                        "low":    float(row["low"]),
                        "close":  float(row["close"]),
                        "volume": int(row["volume"]),
                    }
                    for idx, row in window.iterrows()
                ]

                await bus.publish(Topic.CANDLES_READY, {
                    "symbol":    "NIFTY",
                    "timeframe": BACKTEST_TIMEFRAME,
                    "candles":   candles_list,
                    "ltp":       float(candle["close"]),
                    "orb_high":  orb_high,
                    "orb_low":   orb_low,
                    "timestamp": ts_str,
                }, "replay")

                await asyncio.sleep(speed_ms / 1000)

            print(f"  {day} ✓  ({len(day_df)} candles)")

        # EOD
        await analytics.generate_eod_report()

        return stats
    finally:
        if prev_mode is None:
            os.environ.pop("TRADING_MODE", None)
        else:
            os.environ["TRADING_MODE"] = prev_mode


def load_data(days: int = 5, target_date: str = None) -> pd.DataFrame:
    """Load cached NIFTY data for replay."""
    # Try different cache files
    cache_files = list(Path(DATA_CACHE_DIR).glob("NIFTY_5minute_*.parquet"))
    if not cache_files:
        raise FileNotFoundError(
            f"No cached data found in {DATA_CACHE_DIR}.\n"
            f"Run: docker exec triggerpoint python scripts/download_history.py"
        )

    # Pick the file that has the most recent data
    best_df = None
    best_max_ts = None
    
    for cf in cache_files:
        try:
            temp_df = pd.read_parquet(cf)
            if temp_df.empty:
                continue
            temp_df.index = pd.to_datetime(temp_df.index)
            max_ts = temp_df.index.max()
            if best_max_ts is None or (max_ts is not None and (best_max_ts is None or max_ts > best_max_ts)):
                best_max_ts = max_ts
                best_df = temp_df
        except Exception:
            continue
            
    if best_df is None:
        raise ValueError(f"Could not load any valid data from {DATA_CACHE_DIR}")

    df = best_df.sort_index()

    if target_date:
        target = pd.Timestamp(target_date, tz=IST)
        df = df[df.index.date == target.date()]
        if df.empty:
            raise ValueError(f"No data for date: {target_date}")
        return df

    cutoff = pd.Timestamp.now(tz=IST) - pd.Timedelta(days=days)
    df = df[df.index >= cutoff]

    if df.empty:
        raise ValueError(f"No data for last {days} days in cache")

    return df


def print_summary(stats: dict) -> None:
    t   = stats["wins"] + stats["losses"]
    wr  = round(stats["wins"] / t * 100, 1) if t else 0
    ml_approved = stats["approved"]
    total_sig   = stats["signals"]

    print(f"\n{'━'*56}")
    print(f"  SignalForge Replay Summary")
    print(f"{'━'*56}")
    print(f"  Days replayed:       {len(stats['days'])}")
    print(f"  Candles processed:   {stats['candles']}")
    print(f"  Signals fired:       {total_sig}")
    print(f"  Suppressed (choppy): {stats['suppressed']}")
    print(f"  ML approved:         {ml_approved}")
    print(f"  ML rejected:         {stats['rejected']}")
    print(f"  ─────────────────────────────────────")
    print(f"  Wins:                {stats['wins']}")
    print(f"  Losses:              {stats['losses']}")
    print(f"  Win Rate:            {wr}%")

    if wr >= 55:
        print(f"\n  ✅ Win rate {wr}% — above 55% threshold")
    elif wr >= 45:
        print(f"\n  ⚠️  Win rate {wr}% — borderline. Review signals.")
    else:
        print(f"\n  🔴 Win rate {wr}% — below 45%. Check strategy settings.")

    print(f"{'━'*56}\n")


async def async_main(args) -> None:
    logger.remove()      # suppress verbose logs during replay
    logger.add(sys.stderr, level="WARNING")

    print(f"\nLoading cached NIFTY data...")
    df = load_data(days=args.days, target_date=args.date)
    print(f"Loaded {len(df)} candles")

    stats = await replay(df, speed_ms=args.speed)
    print_summary(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="SignalForge Historical Replay")
    parser.add_argument("--days",  type=int, default=5,   help="Number of past trading days")
    parser.add_argument("--date",  type=str, default=None,help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--speed", type=int, default=50,  help="ms per candle (lower=faster)")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

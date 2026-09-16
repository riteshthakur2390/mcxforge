"""
scripts/backtest_runner.py — Backtest CLI for dashboard and terminal use
"""

import argparse
import asyncio
import json
import os
import sys

# Suppress macOS libsystem_malloc diagnostic warnings
for _k in ["MallocStackLogging", "MallocStackLoggingNoCompact", "MallocNanoZone"]:
    os.environ.pop(_k, None)

import warnings
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", message=".*XGBoost, please export the model.*")
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)
warnings.filterwarnings(
    "ignore",
    message=r".*sklearn\.utils\.parallel\.delayed.*",
    category=UserWarning,
)
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*find_spec\(\) not found; falling back to find_module\(\).*",
    category=ImportWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*scikit-learn.org/stable/model_persistence.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Trying to unpickle estimator .* from version .* when using version .*",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TRADING_MODE", "BACKTEST")
os.environ.setdefault("TELEGRAM_TARGET", "BACKTEST")


def _telegram_enabled() -> bool:
    value = os.getenv("BACKTEST_TELEGRAM_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _send_backtest_telegram(message: str) -> None:
    if not _telegram_enabled():
        return
    try:
        from utils.telegram_notifier import get_notifier

        coro = get_notifier().send_text(message, target="BACKTEST", parse_mode=None)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)
    except Exception as exc:
        print(f"WARN backtest telegram failed: {type(exc).__name__}: {exc}", file=sys.stderr)


def _build_progress_callback(run_id: str):
    raw_step = os.getenv("BACKTEST_TELEGRAM_PROGRESS_STEP", "25").strip()
    try:
        step = max(5, int(raw_step))
    except ValueError:
        step = 25
    state = {"next_pct": step}

    def _callback(payload: dict) -> None:
        progress = float(payload.get("progress", 0.0) or 0.0)
        if progress + 1e-9 < state["next_pct"]:
            return
        _send_backtest_telegram(
            "MCXForge Backtest PROGRESS\n"
            f"Run: {run_id}\n"
            f"Progress: {progress:.1f}%\n"
            f"Processing: {payload.get('day')} ({payload.get('current')}/{payload.get('total')})"
        )
        while state["next_pct"] <= progress:
            state["next_pct"] += step

    return _callback


def _summarize_closed_trades(journal_entries: list[dict]) -> dict[str, float]:
    closed = [row for row in journal_entries if row.get("lifecycle_status") == "CLOSED"]
    net_pnl = 0.0
    wins = 0
    for row in closed:
        pnl = row.get("pnl_inr", row.get("realized_pnl", 0.0))
        try:
            pnl_value = float(pnl or 0.0)
        except (TypeError, ValueError):
            pnl_value = 0.0
        net_pnl += pnl_value
        if pnl_value > 0:
            wins += 1
    trades = len(closed)
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": (wins / trades * 100.0) if trades else 0.0,
        "net_pnl": net_pnl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCXForge backtest runner")
    parser.add_argument("--date", type=str, default=None, help="Single trading date YYYY-MM-DD")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=None, help="Trailing completed trading sessions to backtest")
    parser.add_argument("--trading-days", type=int, default=None, help="Trailing completed trading sessions to backtest")
    parser.add_argument("--calendar-days", type=int, default=None, help="Trailing calendar-day window to backtest")
    parser.add_argument("--max", action="store_true", help="Backtest all available historical trading days in archive")
    parser.add_argument("--speed", type=int, default=0, help="Replay speed in ms per candle")
    parser.add_argument("--ignore-regime", action="store_true", help="Bypass regime filter during backtest")
    parser.add_argument(
        "--live-parity",
        action="store_true",
        help="Replay like live mode: previous closed candle as indicator input, current open as LTP, and live adaptive gates enabled",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Custom unique run ID for process isolation")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    from backtesting.engine import load_cached_data, run_backtest_sync
    from utils.report_formatter import format_rich_report

    requested_trading_days = None if args.max else (args.trading_days if args.trading_days is not None else args.days)
    if not args.json:
        print(
            f"Loading cached data | {'all_available_days' if args.max else f'trading_days={requested_trading_days}'} "
            f"calendar_days={args.calendar_days} timeframe={os.getenv('BACKTEST_TIMEFRAME', '') or 'settings'} "
            f"mode={'live-parity' if args.live_parity else 'historical-research'}",
            flush=True,
        )
    df = load_cached_data(
        days=args.calendar_days,
        trading_days=requested_trading_days,
        target_date=args.date,
        start_date=args.start_date,
        end_date=args.end_date,
        all_available=args.max,
    )
    if not args.json:
        print(
            f"Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}",
            flush=True,
        )
    from datetime import datetime
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.json:
        print(f"Starting replay | run_id={run_id}", flush=True)
        _send_backtest_telegram(
            "MCXForge Backtest STARTED\n"
            f"Run: {run_id}\n"
            f"Trading days: {requested_trading_days or 'auto'}\n"
            f"Mode: {'live-parity' if args.live_parity else 'historical-research'}\n"
            f"Candles: {len(df)}\n"
            f"Range: {df.index.min()} to {df.index.max()}"
        )

    try:
        result = run_backtest_sync(
            df,
            speed_ms=args.speed,
            ignore_regime=args.ignore_regime,
            run_id=run_id,
            progress_callback=None if args.json else _build_progress_callback(run_id),
            live_parity=args.live_parity,
        )
    except Exception as exc:
        if not args.json:
            _send_backtest_telegram(
                "MCXForge Backtest FAILED\n"
                f"Run: {run_id}\n"
                f"Error: {type(exc).__name__}: {exc}"
            )
        raise

    # Save journal to CSV for backtest and live comparison
    journal_entries = result.get("journal", [])
    results_dir = REPO_ROOT / "backtesting" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / f"backtest_trades_{run_id}.csv"

    # Export only completed trades. Approved-only signal rows are useful for the
    # signal journal, but they make this file look like real trades when no
    # contract was selected or order was created.
    export_rows = [
        e for e in journal_entries
        if e.get("lifecycle_status") == "CLOSED"
    ]

    if export_rows:
        keys = set()
        for row in export_rows:
            keys.update(row.keys())
        fieldnames = sorted(list(keys))

        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for export_row in export_rows:
                row = {
                    k: str(v) if isinstance(v, (dict, list)) else v
                    for k, v in export_row.items()
                }
                writer.writerow(row)
        result["csv_export_path"] = str(csv_path)
    else:
        result["csv_export_path"] = ""
        stale_csv = csv_path
        if stale_csv.exists():
            stale_csv.unlink()
    result["run_id"] = run_id
    if "manifest" in result:
        result["manifest"]["csv_export_path"] = result["csv_export_path"]

    if args.json:
        print(json.dumps(result, default=str))
        return

    summary = result["summary"]
    journal = result.get("journal", [])
    total_days = len(summary.get("days", []))
    
    # Get paper capital from result if available
    risk_config = result.get("risk", {})
    paper_capital = float(risk_config.get("paper_capital", 1000000.0))

    from config.settings import MAX_POSITION_LOTS
    
    print("\n" + "="*40)
    print("MCXForge Rich Performance Report")
    print("="*40)
    print(format_rich_report(
        journal, 
        total_days=total_days, 
        extra_summary=summary,
        paper_capital=paper_capital,
        lots=MAX_POSITION_LOTS
    ))
    print("="*40)
    print(f"Run ID: {run_id} | Log: {result.get('backtest_log_path', 'n/a')}")
    print(f"CSV: {result.get('csv_export_path', 'n/a')}\n")
    trade_summary = _summarize_closed_trades(journal)
    _send_backtest_telegram(
        "MCXForge Backtest COMPLETED\n"
        f"Run: {run_id}\n"
        f"Net P&L: Rs. {trade_summary['net_pnl']:,.0f}\n"
        f"Trades: {trade_summary['trades']} | Win rate: {trade_summary['win_rate']:.1f}%\n"
        f"Log: {result.get('backtest_log_path', 'n/a')}\n"
        f"CSV: {result.get('csv_export_path', 'n/a')}"
    )


if __name__ == "__main__":
    main()

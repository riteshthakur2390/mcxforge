#!/usr/bin/env python3
"""
scripts/run_parallel_commodity_backtest.py
=========================================
High-Performance Parallel Multi-Chunk Commodity Backtester:
1. Discovers historical trading sessions from the 5-minute / daily archive.
2. Divides the dates into N chronological chunks (default: 12 chunks).
3. Executes chunks across 4 parallel worker processes with thread throttling
   (OMP_NUM_THREADS=1, etc.) to prevent CPU context thrashing and RAM paging.
4. Displays live real-time completion status, trade counts, PnL, and remaining ETA.
5. Aggregates all chunk trade ledgers into:
   analysis/backtest_parallel_5y/parallel_master_ledger.csv
6. Computes unified statistical performance metrics and SignalForge-grade markdown report.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
try:
    import urllib3.exceptions
    warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)
except Exception:
    pass

import sys
import os
import argparse
import subprocess
import json
import glob
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def get_all_commodity_trading_dates(data_file: str | None = None, timeframe: str = "5m") -> list[str]:
    """Extracts all unique trading dates available for the multi-year commodity backtest."""
    if data_file:
        csv_path = Path(data_file)
    elif "d" in timeframe.lower():
        csv_path = REPO_ROOT / "data" / "historical" / "SILVERM_dhan_1d.csv"
        if not csv_path.exists():
            csv_path = REPO_ROOT / "data" / "historical" / "SILVERMIC_dhan_1d.csv"
    else:
        csv_path = REPO_ROOT / "data" / "historical" / "SILVERM_dhan_5m.csv"
        if not csv_path.exists():
            csv_path = REPO_ROOT / "data" / "historical" / "SILVERMIC_dhan_5m.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing historical data at {csv_path}")

    df = pd.read_csv(csv_path)
    ts_col = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()][0]
    dates = sorted(pd.to_datetime(df[ts_col]).dt.date.astype(str).unique().tolist())
    logger.info(f"Discovered {len(dates)} unique commodity trading sessions in {csv_path.name} ({dates[0]} to {dates[-1]})")
    return dates


def run_chunk(
    chunk_id: int,
    total_chunks: int,
    start_date: str,
    end_date: str,
    timeframe: str,
    data_file: str | None,
    min_votes: int,
    session_filter: str,
    capital: float,
    max_cap_pct: float,
    telegram_trades: bool,
    output_base: Path,
    min_ml_conf: float | None = None,
) -> dict:
    """Executes a single chunk subprocess with isolated output files and throttled threads."""
    chunk_dir = output_base / f"chunk_{chunk_id:02d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    expected_csv = chunk_dir / "trade_ledger.csv"
    log_file = chunk_dir / "execution.log"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_30day_commodity_backtest.py"),
        "--start-date", start_date,
        "--end-date", end_date,
        "--timeframe", timeframe,
        "--output-dir", str(chunk_dir),
        "--min-votes", str(min_votes),
        "--session", session_filter,
        "--capital", str(capital),
        "--max-cap-pct", str(max_cap_pct),
    ]
    if min_ml_conf is not None:
        cmd.extend(["--min-ml-conf", str(min_ml_conf)])
    if data_file:
        cmd.extend(["--data-file", str(data_file)])
    if telegram_trades:
        cmd.append("--telegram-trades")

    # Configure environment to prevent thread explosion and memory thrashing
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    if not telegram_trades:
        env["BACKTEST_TELEGRAM_ENABLED"] = "false"

    print(f"[Chunk {chunk_id:02d}/{total_chunks:02d}] 🚀 Started: {start_date} to {end_date}", flush=True)
    start_time = datetime.now()

    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=lf, text=True)

    elapsed = (datetime.now() - start_time).total_seconds()

    if proc.returncode != 0:
        print(f"[Chunk {chunk_id:02d}/{total_chunks:02d}] ❌ FAILED in {elapsed:.1f}s (see {log_file.name})", flush=True)
        return {
            "chunk_id": chunk_id,
            "start": start_date,
            "end": end_date,
            "status": "FAILED",
            "csv": None,
            "elapsed": elapsed,
            "trades": 0,
            "net_pnl": 0.0,
        }

    actual_csv = str(expected_csv) if expected_csv.exists() else None
    trade_count = 0
    net_pnl = 0.0
    if actual_csv:
        try:
            df = pd.read_csv(actual_csv)
            df["chunk_id"] = chunk_id
            df["chunk_log_path"] = str(log_file.resolve())
            df.to_csv(actual_csv, index=False)
            trade_count = len(df)
            net_pnl = float(df["net_pnl_inr"].sum()) if "net_pnl_inr" in df.columns else 0.0
        except Exception:
            pass

    print(f"[Chunk {chunk_id:02d}/{total_chunks:02d}] ✅ Finished in {elapsed:.1f}s | {trade_count:3d} trades | Net: ₹{net_pnl:10,.2f}", flush=True)
    return {
        "chunk_id": chunk_id,
        "start": start_date,
        "end": end_date,
        "status": "SUCCESS",
        "csv": actual_csv,
        "elapsed": elapsed,
        "trades": trade_count,
        "net_pnl": net_pnl,
    }


def main():
    parser = argparse.ArgumentParser(description="Parallel Multi-Chunk Commodity Backtester")
    parser.add_argument("--days", type=int, default=None, help="Number of past trading days to backtest (e.g. 30, 60)")
    parser.add_argument("--all", action="store_true", help="Backtest all available historical trading days")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--chunks", type=int, default=12, help="Number of chronological chunks (default: 12)")
    parser.add_argument("--workers", type=int, default=4, help="Max parallel worker processes (default: 4)")
    parser.add_argument("--timeframe", type=str, default="5m", choices=["5m", "15m", "1h", "1d"], help="Candle timeframe (default: 5m)")
    parser.add_argument("--data-file", type=str, default=None, help="Explicit historical CSV path")
    parser.add_argument("--min-votes", type=int, default=5, help="Minimum vote consensus threshold (default: 5)")
    parser.add_argument("--session", type=str, default="EVENING", choices=["EVENING", "ALL", "MORNING"], help="Session filter (default: EVENING)")
    parser.add_argument("--capital", type=float, default=200_000.0, help="Starting capital (default: 200,000)")
    parser.add_argument("--max-cap-pct", type=float, default=20.0, help="Max capital percentage usable per trade (default: 20.0%%)")
    parser.add_argument("--telegram", action="store_true", help="Send master performance report card to Telegram")
    parser.add_argument("--telegram-trades", action="store_true", help="Send individual trade notifications to Telegram")
    parser.add_argument("--min-ml-conf", type=float, default=float(os.getenv("MIN_ML_CONF", "0.32")), help="Filter out trades below ML confidence threshold (default: 0.32)")
    parser.add_argument("--output-dir", type=str, default="analysis/backtest_parallel_5y", help="Output directory")
    args = parser.parse_args()

    output_base = REPO_ROOT / args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 95)
    print("      MCXFORGE HIGH-EFFICIENCY PARALLEL COMMODITY OPTION BACKTEST (4 WORKERS)")
    print("=" * 95)

    all_dates = get_all_commodity_trading_dates(args.data_file, args.timeframe)
    total_available_days = len(all_dates)

    # Filter dates based on user arguments: --days, --all, or explicit date range
    if args.start_date and args.end_date:
        all_dates = [d for d in all_dates if args.start_date <= d <= args.end_date]
        print(f"Filtering date range: {args.start_date} to {args.end_date} ({len(all_dates)} sessions)")
    elif args.days is not None and not args.all:
        req_days = min(args.days, len(all_dates))
        all_dates = all_dates[-req_days:]
        print(f"Selected last {len(all_dates)} trading sessions (requested past {args.days} days)")
    elif args.all or args.days is None:
        print(f"Selected ALL available {len(all_dates)} trading sessions")

    if not all_dates:
        print("❌ Error: No trading dates found for the specified range.")
        return

    total_days = len(all_dates)
    print(f"Total Trading Sessions to Backtest: {total_days} (from {all_dates[0]} to {all_dates[-1]})")

    # Split into N chunks (capped at total_days so chunks are never empty)
    num_chunks = max(1, min(args.chunks, total_days))
    chunk_size = (total_days + num_chunks - 1) // num_chunks
    chunks = []
    for i in range(num_chunks):
        c_dates = all_dates[i * chunk_size : (i + 1) * chunk_size]
        if c_dates:
            chunks.append((i + 1, c_dates[0], c_dates[-1]))

    print(f"Divided into {len(chunks)} chronological slices (~{chunk_size} days each):")
    for cid, s, e in chunks:
        print(f"  Chunk {cid:02d}: {s} -> {e}")

    max_workers = min(args.workers or 4, len(chunks))
    print(f"\nLaunching {len(chunks)} chunks across {max_workers} worker processes (Thread Throttled = 1 thread/worker)...")
    print(f"Safe RAM budget: ~{max_workers * 1.2:.1f} GB / 16.0 GB (Zero swap thrashing)\n")
    overall_start = datetime.now()

    chunk_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_chunk,
                chunk_id=cid,
                total_chunks=len(chunks),
                start_date=s,
                end_date=e,
                timeframe=args.timeframe,
                data_file=args.data_file,
                min_votes=args.min_votes,
                session_filter=args.session,
                capital=args.capital,
                max_cap_pct=args.max_cap_pct,
                telegram_trades=args.telegram_trades,
                output_base=output_base,
                min_ml_conf=args.min_ml_conf,
            ): cid
            for cid, s, e in chunks
        }
        completed_count = 0
        for fut in as_completed(futures):
            res = fut.result()
            chunk_results.append(res)
            completed_count += 1
            progress_pct = (completed_count / len(chunks)) * 100.0
            elapsed_total = (datetime.now() - overall_start).total_seconds()
            avg_per_chunk = elapsed_total / completed_count
            eta_seconds = avg_per_chunk * (len(chunks) - completed_count)
            print(
                f"   [Overall Progress] {completed_count}/{len(chunks)} Chunks Completed ({progress_pct:.1f}%) | "
                f"Elapsed: {elapsed_total/60.0:.1f}m | Est. Remaining: {eta_seconds/60.0:.1f}m",
                flush=True,
            )

    overall_elapsed = (datetime.now() - overall_start).total_seconds()
    print(f"\nAll {len(chunks)} chunks finished execution in {overall_elapsed/60.0:.2f} minutes!")

    # Merge all chunk CSVs
    all_dfs = []
    for res in sorted(chunk_results, key=lambda x: x["chunk_id"]):
        if res["status"] == "SUCCESS" and res["csv"]:
            try:
                cdf = pd.read_csv(res["csv"])
                all_dfs.append(cdf)
            except Exception as e:
                print(f"Warning loading {res['csv']}: {e}")

    valid_dfs = [df for df in all_dfs if not df.empty]
    if not valid_dfs:
        print("❌ Error: No trade results were generated.")
        return

    master_df = pd.concat(valid_dfs, ignore_index=True)
    dedup_cols = [c for c in ["entry_time", "symbol"] if c in master_df.columns]
    if dedup_cols:
        master_df = master_df.drop_duplicates(subset=dedup_cols)
    if "entry_time" in master_df.columns:
        master_df = master_df.sort_values(by="entry_time").reset_index(drop=True)
    master_df["trade_id"] = [f"T{i+1:04d}" for i in range(len(master_df))]

    master_csv = output_base / "parallel_master_ledger.csv"
    master_df["master_ledger_path"] = str(master_csv.resolve())
    master_df.to_csv(master_csv, index=False)
    print(f"\nSaved Unified Master Ledger ({len(master_df)} trades) to:")
    print(f"  {master_csv}")
    print("Execution Logs & Diagnostics:")
    print(f"  Master Report : {output_base / 'backtest_report.md'}")
    print(f"  Chunk Logs Dir: {output_base}/chunk_*/execution.log")

    # Compute Performance Statistics
    pnl_col = "net_pnl_inr" if "net_pnl_inr" in master_df.columns else "realized_pnl"
    gross_col = "gross_pnl_inr" if "gross_pnl_inr" in master_df.columns else None
    charges_col = "fees_inr" if "fees_inr" in master_df.columns else ("total_charges" if "total_charges" in master_df.columns else None)

    net_pnl = float(master_df[pnl_col].sum())
    gross_pnl = float(master_df[gross_col].sum()) if gross_col else 0.0
    charges = float(master_df[charges_col].sum()) if charges_col else 0.0

    wins = master_df[master_df[pnl_col] > 0]
    losses = master_df[master_df[pnl_col] <= 0]
    wr = len(wins) / len(master_df) * 100.0 if len(master_df) > 0 else 0.0
    gw = float(wins[gross_col].sum()) if gross_col else float(wins[pnl_col].sum())
    gl = abs(float(losses[gross_col].sum())) if gross_col else abs(float(losses[pnl_col].sum()))
    pf = gw / gl if gl > 0 else 99.0

    cum = master_df[pnl_col].cumsum()
    peak = cum.cummax()
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0

    print("\n" + "=" * 95)
    print("           FINAL MULTI-CHUNK PARALLEL BACKTEST PERFORMANCE SUMMARY")
    print("=" * 95)
    print(f"Total Completed Trades : {len(master_df)}")
    print(f"Winning Trades         : {len(wins)} ({wr:.2f}%)")
    print(f"Losing Trades          : {len(losses)} ({100.0 - wr:.2f}%)")
    print(f"Gross Realized P&L     : ₹{gross_pnl:,.2f}")
    print(f"Total F&O Charges      : ₹{charges:,.2f}")
    print(f"Net Realized P&L       : ₹{net_pnl:,.2f}")
    print(f"Profit Factor          : {pf:.2f}")
    print(f"Max Drawdown           : ₹{max_dd:,.2f}")
    print(f"Average Win            : ₹{wins[pnl_col].mean():,.2f}" if len(wins) else "Average Win: N/A")
    print(f"Average Loss           : ₹{losses[pnl_col].mean():,.2f}" if len(losses) else "Average Loss: N/A")
    print(f"Expectancy Per Trade   : ₹{master_df[pnl_col].mean():,.2f}")

    if "exit_reason" in master_df.columns:
        print("\n-- EXIT REASON BREAKDOWN --")
        reasons = master_df.groupby("exit_reason").agg(
            trades=("exit_reason", "count"),
            net_pnl=(pnl_col, "sum"),
            avg_pnl=(pnl_col, "mean"),
            win_rate=(pnl_col, lambda x: (x > 0).mean() * 100.0)
        )
        print(reasons.to_string())

    report_md = output_base / "backtest_report.md"
    report_content = f"""# MCXForge Parallel Multi-Chunk Backtest Report — SILVERM Option Buying

> **Execution Model**: **BUY CALL (CE) / BUY PUT (PE) ONLY**
> **Timeframe**: `{args.timeframe}` | **Min Votes**: `{args.min_votes}` | **Session**: `{args.session}`
> **Total Sessions**: `{total_days}` ({all_dates[0]} to {all_dates[-1]}) | **Workers**: `{max_workers}`

## 1. Executive Performance Summary

| Metric | Value | Metric | Value |
|---|---|---|---|
| **Total Trades** | `{len(master_df)}` | **Win Rate** | `{wr:.2f}%` ({len(wins)}W / {len(losses)}L) |
| **Gross PnL** | `₹{gross_pnl:,.2f}` | **Statutory Charges** | `₹{charges:,.2f}` |
| **Net Realized PnL** | **`₹{net_pnl:,.2f}`** | **Profit Factor** | `{pf:.2f}` |
| **Max Drawdown** | `₹{max_dd:,.2f}` | **Expectancy / Trade** | `₹{master_df[pnl_col].mean():,.2f}` |

## 2. Chunk Execution Breakdown

| Chunk | Date Range | Status | Trades | Net PnL (₹) | Elapsed |
|---|---|---|---|---|---|
"""
    for r in sorted(chunk_results, key=lambda x: x["chunk_id"]):
        report_content += f"| **Chunk {r['chunk_id']:02d}** | {r['start']} to {r['end']} | {r['status']} | {r['trades']} | ₹{r['net_pnl']:,.2f} | {r['elapsed']:.1f}s |\n"

    report_content += f"""
## 3. Execution Log Paths & Deep Diagnostic Traceability

* **Unified Master Ledger**: `{master_csv.resolve()}`
* **Master Markdown Report**: `{report_md.resolve()}`
* **Per-Chunk Execution Logs**:
"""
    for r in sorted(chunk_results, key=lambda x: x["chunk_id"]):
        c_log = output_base / f"chunk_{r['chunk_id']:02d}" / "execution.log"
        report_content += f"* **Chunk {r['chunk_id']:02d} Log** (`{r['start']}` to `{r['end']}`): `{c_log.resolve()}` ({r['trades']} trades | {r['status']})\n"

    report_md.write_text(report_content)
    print(f"\nGenerated SignalForge-Grade Markdown Report: {report_md}")
    print("=" * 95 + "\n")

    if args.telegram:
        try:
            import time
            from utils.telegram_notifier import get_notifier
            notifier = get_notifier()
            
            # 1. Dispatch individual trade alerts (Opened + Closed) in exact chronological sequence
            sorted_trades = master_df.sort_values(by="entry_time").to_dict(orient="records") if "entry_time" in master_df.columns else master_df.to_dict(orient="records")
            print(f"📡 Dispatching {len(sorted_trades)} trade alerts (Open + Close) to Telegram...")
            
            for tr in sorted_trades:
                c_symbol = tr.get("contract_symbol") or tr.get("option_symbol") or "SILVERM Option"
                # Trade Opened payload
                open_payload = {
                    "symbol": tr.get("symbol", "SILVERM"),
                    "direction": tr.get("direction", "BUY"),
                    "contract_symbol": c_symbol,
                    "strike": tr.get("strike"),
                    "option_type": tr.get("option_type", "CE"),
                    "expiry_date": tr.get("expiry_date", ""),
                    "entry_price": tr.get("underlying_entry", tr.get("entry_price", 0.0)),
                    "actual_premium": tr.get("actual_premium", tr.get("entry_premium", 0.0)),
                    "stop_loss": tr.get("entry_price", 0.0) * 0.985,
                    "target": tr.get("entry_price", 0.0) * 1.025,
                    "quantity": tr.get("quantity", 5),
                    "lots": tr.get("lots", 1),
                    "margin_used": tr.get("margin_used_inr", 40000.0),
                    "max_margin_budget": args.capital * (args.max_cap_pct / 100.0),
                    "strategy": tr.get("lead_strategy", ""),
                    "strategies": tr.get("strategies_fired", "").split("+") if isinstance(tr.get("strategies_fired"), str) else [],
                    "votes": tr.get("votes", args.min_votes),
                    "time": tr.get("entry_time", ""),
                    "mode": "BACKTEST",
                    "target": "BACKTEST",
                }
                notifier.send_trade_opened_sync(open_payload, target="BACKTEST")
                time.sleep(0.08)

                # Trade Closed payload
                close_payload = {
                    "symbol": tr.get("symbol", "SILVERM"),
                    "direction": tr.get("direction", "BUY"),
                    "contract_symbol": c_symbol,
                    "strike": tr.get("strike"),
                    "option_type": tr.get("option_type", "CE"),
                    "expiry_date": tr.get("expiry_date", ""),
                    "underlying_entry": tr.get("underlying_entry", tr.get("entry_price", 0.0)),
                    "underlying_exit": tr.get("underlying_exit", tr.get("exit_price", 0.0)),
                    "underlying_points": tr.get("underlying_points", tr.get("pnl_points", 0.0)),
                    "actual_premium": tr.get("actual_premium", tr.get("entry_premium", 0.0)),
                    "entry_premium": tr.get("entry_premium", tr.get("actual_premium", 0.0)),
                    "exit_premium": tr.get("exit_premium", 0.0),
                    "contract_points": tr.get("contract_points", 0.0),
                    "entry_price": tr.get("actual_premium", tr.get("entry_premium", 0.0)),
                    "exit_price": tr.get("exit_premium", 0.0),
                    "gross_pnl_inr": tr.get("gross_pnl_inr", 0.0),
                    "charges": tr.get("fees_inr", 0.0),
                    "net_pnl_inr": tr.get("net_pnl_inr", 0.0),
                    "margin_used": tr.get("margin_used_inr", 40000.0),
                    "max_margin_budget": args.capital * (args.max_cap_pct / 100.0),
                    "lots": tr.get("lots", 1),
                    "quantity": tr.get("quantity", 5),
                    "exit_reason": tr.get("exit_reason", "EXIT"),
                    "peak_pnl_pct": tr.get("peak_pnl_pct", 0.0),
                    "entry_efficiency": tr.get("entry_efficiency", 0.5),
                    "hold_minutes": tr.get("hold_minutes", 0),
                    "time": tr.get("exit_time", ""),
                    "mode": "BACKTEST",
                    "target": "BACKTEST",
                }
                notifier.send_trade_closed_sync(close_payload, target="BACKTEST")
                time.sleep(0.08)

            sharpe_est = (net_pnl / max_dd) if max_dd > 0 else 1.5
            summary_card = {
                "symbol": "SILVERM",
                "period": f"{all_dates[0]} to {all_dates[-1]} ({total_days} Days)",
                "bars": len(master_df),
                "trades": len(master_df),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": wr,
                "gross_pnl": gross_pnl,
                "charges": charges,
                "net_pnl": net_pnl,
                "profit_factor": pf,
                "sharpe": sharpe_est,
                "expectancy": master_df[pnl_col].mean() if len(master_df) > 0 else 0.0,
                "capital": args.capital,
                "capital_return_pct": (net_pnl / max(1.0, args.capital)) * 100.0,
                "peak_margin_used": float(master_df["margin_used_inr"].max()) if "margin_used_inr" in master_df.columns else (args.capital * args.max_cap_pct / 100.0),
                "max_margin_budget": args.capital * (args.max_cap_pct / 100.0),
                "max_dd": max_dd,
                "max_dd_pct": (max_dd / max(1.0, args.capital)) * 100.0,
                "mode": "BACKTEST",
                "target": "BACKTEST",
            }
            notifier.send_backtest_summary_sync(summary_card, target="BACKTEST")
            print("✅ Dispatched all trade alerts and master backtest review to Telegram.")
        except Exception as te:
            print(f"Telegram report dispatch skipped: {te}")


if __name__ == "__main__":
    main()

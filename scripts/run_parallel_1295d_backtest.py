#!/usr/bin/env python3
"""
scripts/run_parallel_1295d_backtest.py

High-Performance Parallel Multi-Chunk 1,295-Day Backtester:
1. Discovers all 1,295 historical trading sessions from the archive.
2. Divides the dates into N chronological chunks (default: 12).
3. Executes chunks with worker thread-throttling (OMP_NUM_THREADS=1, etc.) to prevent CPU context thrashing.
4. Defaults to safe worker concurrency (4 workers) to prevent RAM saturation and macOS swap paging.
5. Displays live real-time completion status, trade counts, PnL, and remaining ETA.
6. Aggregates all trade ledgers into: backtesting/results/parallel_1295d_master_ledger.csv.
7. Computes complete statistical performance metrics.
"""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def get_all_trading_dates():
    """Extracts all unique trading dates available for the multi-year backtest."""
    dates_set = set()
    db_path = REPO_ROOT / "data/historical/market_history.sqlite3"
    if db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT date(ts) FROM candles "
                "WHERE symbol='NIFTY' AND interval='5minute' AND date(ts) >= '2021-06-21' AND date(ts) < '2026-08-31' "
                "ORDER BY date(ts) ASC"
            )
            for row in cursor.fetchall():
                if row[0]:
                    dates_set.add(row[0])
            conn.close()
        except Exception:
            pass

    for parquet_file in REPO_ROOT.glob("data/cache/NIFTY_5minute_*.parquet"):
        try:
            df = pd.read_parquet(parquet_file)
            for d in df.index.strftime("%Y-%m-%d"):
                if "2021-06-21" <= d < "2026-08-31":
                    dates_set.add(d)
        except Exception:
            pass

    if len(dates_set) >= 1000:
        return sorted(list(dates_set))

    from backtesting.engine import load_cached_data
    df = load_cached_data(all_available=True)
    all_dates = [d for d in df.index.strftime("%Y-%m-%d").unique() if d < "2026-08-31"]
    return sorted(all_dates)

def run_chunk(chunk_id, total_chunks, start_date, end_date, live_parity=False, ignore_regime=True):
    """Executes a single chunk subprocess with isolated output files and throttled threads."""
    chunk_run_id = f"chunk_{chunk_id:02d}"
    expected_csv = REPO_ROOT / f"backtesting/results/backtest_trades_{chunk_run_id}.csv"
    log_file = REPO_ROOT / f"logs/parallel_chunk_{chunk_id:02d}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/backtest_runner.py"),
        "--start-date", start_date,
        "--end-date", end_date,
        "--run-id", chunk_run_id,
        "--json"
    ]
    if ignore_regime:
        cmd.append("--ignore-regime")
    if live_parity:
        cmd.append("--live-parity")

    # Configure environment to prevent thread explosion and memory thrashing
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    env["BACKTEST_TELEGRAM_ENABLED"] = "false"

    print(f"[Chunk {chunk_id:02d}/{total_chunks:02d}] 🚀 Started: {start_date} to {end_date}", flush=True)
    start_time = datetime.now()
    
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=lf, text=True)
    
    elapsed = (datetime.now() - start_time).total_seconds()

    if proc.returncode != 0:
        print(f"[Chunk {chunk_id:02d}/{total_chunks:02d}] ❌ FAILED in {elapsed:.1f}s (see {log_file.name})", flush=True)
        return {"chunk_id": chunk_id, "start": start_date, "end": end_date, "status": "FAILED", "csv": None, "elapsed": elapsed}

    actual_csv = str(expected_csv) if expected_csv.exists() else None
    trade_count = 0
    net_pnl = 0.0
    if actual_csv:
        try:
            df = pd.read_csv(actual_csv)
            trade_count = len(df)
            net_pnl = df["net_pnl_inr"].sum() if "net_pnl_inr" in df.columns else (df["realized_pnl"].sum() if "realized_pnl" in df.columns else 0.0)
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
        "net_pnl": net_pnl
    }

def main():
    parser = argparse.ArgumentParser(description="Parallel 12-Chunk 1,295-Day Backtester")
    parser.add_argument("--chunks", type=int, default=12, help="Number of chronological chunks (default: 12)")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel worker processes (default: 4, safe for RAM)")
    parser.add_argument("--live-parity", action="store_true", help="Enable live parity mode")
    parser.add_argument("--use-regime", action="store_true", help="Enable regime filter (default: False, matching benchmark)")
    args = parser.parse_args()

    print("="*95)
    print("        SIGNALFORGE HIGH-EFFICIENCY PARALLEL 1,295-DAY HISTORICAL BACKTEST")
    print("="*95)

    all_dates = get_all_trading_dates()
    total_days = len(all_dates)
    print(f"Total Unique Trading Sessions Found: {total_days} (from {all_dates[0]} to {all_dates[-1]})")

    # Split into N chunks
    num_chunks = args.chunks
    chunk_size = (total_days + num_chunks - 1) // num_chunks
    chunks = []
    for i in range(num_chunks):
        c_dates = all_dates[i * chunk_size : (i + 1) * chunk_size]
        if c_dates:
            chunks.append((i + 1, c_dates[0], c_dates[-1]))

    print(f"Divided into {len(chunks)} chronological slices (~{chunk_size} days each):")
    for cid, s, e in chunks:
        print(f"  Chunk {cid:02d}: {s} -> {e}")

    # Safe worker allocation: default 4 workers to prevent RAM thrashing and UI lockups
    cpu_cores = os.cpu_count() or 4
    max_workers = args.workers or min(4, max(1, cpu_cores // 2))
    print(f"\nLaunching {len(chunks)} chunks across {max_workers} worker processes (Thread Throttled = 1 thread/worker)...")
    print(f"Safe RAM budget: ~{max_workers * 1.2:.1f} GB / 16.0 GB (Zero disk swap paging)\n")
    overall_start = datetime.now()

    chunk_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_chunk, cid, len(chunks), s, e, args.live_parity, not args.use_regime): cid
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
            print(f"   [Overall Progress] {completed_count}/{len(chunks)} Chunks Completed ({progress_pct:.1f}%) | Elapsed: {elapsed_total/60.0:.1f}m | Est. Remaining: {eta_seconds/60.0:.1f}m", flush=True)

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

    if not all_dfs:
        print("Error: No trade results were generated.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)
    # Deduplicate trades by signal_id if present
    if "signal_id" in master_df.columns:
        master_df = master_df.drop_duplicates(subset=["signal_id"])

    master_csv = REPO_ROOT / "backtesting/results/parallel_1295d_master_ledger.csv"
    master_df.to_csv(master_csv, index=False)
    print(f"\nSaved Unified Master Ledger ({len(master_df)} trades) to:")
    print(f"  {master_csv}")

    # Compute Performance Statistics
    pnl_col = "net_pnl_inr" if "net_pnl_inr" in master_df.columns else "realized_pnl"
    gross_col = "gross_pnl_inr" if "gross_pnl_inr" in master_df.columns else None
    charges_col = "total_charges" if "total_charges" in master_df.columns else None

    net_pnl = master_df[pnl_col].sum()
    gross_pnl = master_df[gross_col].sum() if gross_col else 0.0
    charges = master_df[charges_col].sum() if charges_col else 0.0

    wins = master_df[master_df[pnl_col] > 0]
    losses = master_df[master_df[pnl_col] <= 0]
    wr = len(wins) / len(master_df) * 100.0 if len(master_df) > 0 else 0.0
    gw = wins[gross_col].sum() if gross_col else wins[pnl_col].sum()
    gl = abs(losses[gross_col].sum()) if gross_col else abs(losses[pnl_col].sum())
    pf = gw / gl if gl > 0 else 99.0

    print("\n" + "="*95)
    print("                    FINAL 1,295-DAY BACKTEST PERFORMANCE SUMMARY")
    print("="*95)
    print(f"Total Completed Trades : {len(master_df)}")
    print(f"Winning Trades         : {len(wins)} ({wr:.2f}%)")
    print(f"Losing Trades          : {len(losses)} ({100.0 - wr:.2f}%)")
    print(f"Gross Realized P&L     : ₹{gross_pnl:,.2f}")
    print(f"Total F&O Charges      : ₹{charges:,.2f}")
    print(f"Net Realized P&L       : ₹{net_pnl:,.2f}")
    print(f"Profit Factor          : {pf:.2f}")
    print(f"Average Win            : ₹{wins[pnl_col].mean():,.2f}")
    print(f"Average Loss           : ₹{losses[pnl_col].mean():,.2f}")
    print(f"Payoff Ratio           : {abs(wins[pnl_col].mean() / losses[pnl_col].mean()):.2f}x")
    print(f"Expectancy Per Trade   : ₹{master_df[pnl_col].mean():,.2f}")

    # Exit reason breakdown
    if "exit_reason" in master_df.columns:
        print("\n-- EXIT REASON BREAKDOWN --")
        exit_sum = master_df.groupby("exit_reason").agg(
            trades=(pnl_col, "count"),
            net_pnl=(pnl_col, "sum"),
            avg_pnl=(pnl_col, "mean"),
            win_rate=(pnl_col, lambda s: (s > 0).mean() * 100.0)
        ).reset_index()
        print(exit_sum.to_string(index=False))

    print("\n" + "="*95)

if __name__ == "__main__":
    main()

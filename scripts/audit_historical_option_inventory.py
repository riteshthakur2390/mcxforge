#!/usr/bin/env python3
"""
scripts/audit_historical_option_inventory.py

Comprehensive inventory of all historical option datasets in the repository.
Audits SQLite databases, Parquet cache files, Dhan/Upstox datasets, and OI history.
"""

import os
import glob
import sqlite3
import json
from pathlib import Path
import pandas as pd

def audit_inventory():
    results = {
        "parquet_files": [],
        "sqlite_databases": [],
        "oi_history_files": [],
        "coverage_summary": {}
    }

    print("================================================================================")
    print("        COMPREHENSIVE PROJECT OPTION DATA INVENTORY & AUDIT                    ")
    print("================================================================================")

    # 1. PARQUET OPTION FILES
    print("\n[1] PARQUET OPTION FILES IN CACHE:")
    for p in sorted(glob.glob("data/cache/*options*.parquet")):
        try:
            df = pd.read_parquet(p)
            ts_col = "ts" if "ts" in df.columns else ("datetime" if "datetime" in df.columns else df.columns[0])
            strikes = df["strike"].nunique() if "strike" in df.columns else 0
            expiries = df["expiry"].nunique() if "expiry" in df.columns else 0
            opt_types = list(df["option_type"].unique()) if "option_type" in df.columns else []
            min_ts = str(df[ts_col].min())
            max_ts = str(df[ts_col].max())
            
            # Dates
            dates = pd.to_datetime(df[ts_col]).dt.strftime("%Y-%m-%d").unique()
            
            p_info = {
                "file": p,
                "size_bytes": os.path.getsize(p),
                "rows": len(df),
                "columns": list(df.columns),
                "date_range": f"{min_ts} → {max_ts}",
                "unique_sessions": len(dates),
                "unique_strikes": strikes,
                "unique_expiries": expiries,
                "option_types": opt_types,
            }
            results["parquet_files"].append(p_info)
            print(f"  • File: {p}")
            print(f"    Rows: {len(df):,} | Unique Sessions: {len(dates)} | Date Range: {min_ts[:10]} → {max_ts[:10]}")
            print(f"    Strikes: {strikes} | Expiries: {expiries} | Option Types: {opt_types}")
        except Exception as exc:
            print(f"  • File: {p} -> Error: {exc}")

    # 2. SQLITE DATABASES
    print("\n[2] SQLITE DATABASES:")
    for db in ["data/cache/historical_options.db", "data/market_data.db", "data/historical.db", "data/historical/market_history.sqlite3"]:
        if os.path.exists(db):
            try:
                conn = sqlite3.connect(db)
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [r[0] for r in cur.fetchall()]
                db_info = {
                    "database": db,
                    "size_bytes": os.path.getsize(db),
                    "tables": {}
                }
                print(f"  • Database: {db} ({os.path.getsize(db)/(1024*1024):.2f} MB)")
                for t in tables:
                    cur.execute(f"SELECT COUNT(*) FROM {t}")
                    cnt = cur.fetchone()[0]
                    # Check if table has option data
                    cur.execute(f"PRAGMA table_info({t})")
                    cols = [c[1] for c in cur.fetchall()]
                    
                    t_details = {"rows": cnt, "columns": cols}
                    if "strike" in cols and "ts" in cols and cnt > 0:
                        cur.execute(f"SELECT MIN(ts), MAX(ts), COUNT(DISTINCT strike), COUNT(DISTINCT expiry) FROM {t}")
                        min_t, max_t, num_s, num_e = cur.fetchone()
                        t_details["date_range"] = f"{min_t} → {max_t}"
                        t_details["unique_strikes"] = num_s
                        t_details["unique_expiries"] = num_e
                        print(f"    Table '{t}': {cnt:,} rows | Range: {min_t[:10]} → {max_t[:10]} | Strikes: {num_s} | Expiries: {num_e}")
                    else:
                        print(f"    Table '{t}': {cnt:,} rows")
                    db_info["tables"][t] = t_details
                results["sqlite_databases"].append(db_info)
            except Exception as exc:
                print(f"  • Database: {db} -> Error: {exc}")

    # 3. OI HISTORY PARQUET FILES
    oi_files = sorted(glob.glob("data/oi_history/*.parquet"))
    print(f"\n[3] OI HISTORY DAILY PARQUET FILES: {len(oi_files)} files found")
    if oi_files:
        print(f"    Date Coverage: {Path(oi_files[0]).stem} → {Path(oi_files[-1]).stem}")
        results["oi_history_files"] = {
            "count": len(oi_files),
            "earliest": Path(oi_files[0]).stem,
            "latest": Path(oi_files[-1]).stem,
        }

    # 4. OVERALL HISTORICAL COVERAGE SUMMARY (2021-06-21 to 2026-08-13)
    print("\n[4] OVERALL COVERAGE ACROSS 1,287 SESSIONS (2021-06-21 → 2026-08-13):")
    # Query market_history.sqlite3 distinct dates for options
    conn = sqlite3.connect("data/historical/market_history.sqlite3")
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT substr(ts, 1, 10) FROM option_candles WHERE symbol='NIFTY' AND interval='1minute'")
    opt_dates = set([r[0] for r in cur.fetchall()])
    
    cur.execute("SELECT DISTINCT substr(ts, 1, 10) FROM candles WHERE symbol='NIFTY' AND interval='5minute'")
    all_dates = sorted([r[0] for r in cur.fetchall()])

    covered_sessions = sorted(list(opt_dates.intersection(set(all_dates))))
    missing_sessions = sorted(list(set(all_dates) - opt_dates))

    print(f"  • Total Trading Sessions in Dataset : {len(all_dates):,}")
    print(f"  • Sessions With Real 1m Option Data  : {len(covered_sessions)} ({len(covered_sessions)/len(all_dates)*100:.2f}%)")
    print(f"    -> Date Range of Real Option Data  : {covered_sessions[0]} → {covered_sessions[-1]}")
    print(f"  • Sessions Requiring Proxy Fallback  : {len(missing_sessions):,} ({len(missing_sessions)/len(all_dates)*100:.2f}%)")
    print(f"    -> Missing Date Range              : {missing_sessions[0]} → {missing_sessions[-1]}")

    results["coverage_summary"] = {
        "total_sessions": len(all_dates),
        "sessions_with_real_options": len(covered_sessions),
        "real_options_date_range": f"{covered_sessions[0]} → {covered_sessions[-1]}" if covered_sessions else "None",
        "sessions_missing_real_options": len(missing_sessions),
        "missing_options_date_range": f"{missing_sessions[0]} → {missing_sessions[-1]}" if missing_sessions else "None",
    }

    out_path = Path("analysis/deterministic_1287_backtest/historical_option_inventory.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed inventory to {out_path}\n")

if __name__ == "__main__":
    audit_inventory()

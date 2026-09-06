"""
scripts/seed_pilot_historical_data.py

Seeds deterministic synthetic 5-minute candles into:
1. data/historical/market_history.sqlite3 (for SQLite-based backtests and outcome pilots)
2. data/cache/NIFTY_5minute_dhan.parquet (for parquet cache replay pilots)

Covers the 24 representative pilot dates across Bullish, Bearish, Range, and High-Volatility regimes.
"""

import sqlite3
from datetime import datetime, time, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytz

IST = pytz.timezone("Asia/Kolkata")

PILOT_DATES = [
    # 1. Trending Bullish Sessions
    {"date": "2026-02-05", "trend": 1.5, "vol": 10.0},
    {"date": "2026-02-16", "trend": 2.0, "vol": 12.0},
    {"date": "2026-03-02", "trend": 1.8, "vol": 11.0},
    {"date": "2026-04-28", "trend": 1.4, "vol": 9.0},
    {"date": "2026-06-11", "trend": 2.2, "vol": 14.0},
    {"date": "2026-08-11", "trend": 1.6, "vol": 10.0},

    # 2. Trending Bearish Sessions
    {"date": "2026-02-12", "trend": -1.5, "vol": 10.0},
    {"date": "2026-03-19", "trend": -2.0, "vol": 12.0},
    {"date": "2026-04-16", "trend": -1.8, "vol": 11.0},
    {"date": "2026-05-18", "trend": -1.4, "vol": 9.0},
    {"date": "2026-06-24", "trend": -2.2, "vol": 14.0},
    {"date": "2026-08-24", "trend": -1.6, "vol": 10.0},

    # 3. Range-Bound / Chop Sessions
    {"date": "2026-02-20", "trend": 0.0, "vol": 5.0},
    {"date": "2026-03-10", "trend": 0.0, "vol": 6.0},
    {"date": "2026-04-21", "trend": 0.0, "vol": 4.0},
    {"date": "2026-05-08", "trend": 0.0, "vol": 5.0},
    {"date": "2026-07-09", "trend": 0.0, "vol": 4.5},
    {"date": "2026-08-21", "trend": 0.0, "vol": 5.5},

    # 4. High-Volatility Sessions
    {"date": "2026-02-26", "trend": 0.5, "vol": 25.0},
    {"date": "2026-03-06", "trend": -0.5, "vol": 28.0},
    {"date": "2026-04-23", "trend": 0.8, "vol": 30.0},
    {"date": "2026-06-08", "trend": -0.8, "vol": 26.0},
    {"date": "2026-07-17", "trend": 0.3, "vol": 32.0},
    {"date": "2026-08-19", "trend": -0.3, "vol": 29.0},
]


def generate_candles():
    sqlite_rows = []
    parquet_records = []
    now_iso = datetime.now(IST).isoformat()

    base_price = 24000.0

    for item in PILOT_DATES:
        d_str = item["date"]
        trend = item["trend"]
        vol = item["vol"]
        d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()

        # Seed per date for reproducibility
        np.random.seed(int(d_str.replace("-", "")) % 100000)

        # 75 bars from 09:15 to 15:25
        current_price = base_price + np.random.normal(0, 100)
        dt = datetime.combine(d_obj, time(9, 15))

        for bar_idx in range(75):
            bar_ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            time_str = dt.strftime("%H:%M")

            # Deterministic price action
            delta = trend * 2.0 + np.random.normal(0, vol)
            o = round(current_price, 2)
            c = round(current_price + delta, 2)
            spread = max(abs(delta) + 2.0, np.random.uniform(3, 8))
            h = round(max(o, c) + np.random.uniform(0.5, spread), 2)
            l = round(min(o, c) - np.random.uniform(0.5, spread), 2)
            volume = int(np.random.uniform(20000, 150000))
            current_price = c

            sqlite_rows.append((
                "NIFTY", "5minute", "dhan", bar_ts_str,
                o, h, l, c, volume, 0, "historical_seed", now_iso
            ))

            parquet_records.append({
                "timestamp": pd.Timestamp(dt),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": volume,
                "time": time_str,
            })

            dt += timedelta(minutes=5)

    # 1. Populate SQLite
    db_path = Path("data/historical/market_history.sqlite3")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            broker TEXT NOT NULL,
            ts TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            oi INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (symbol, interval, broker, ts)
        )
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_candles_lookup
        ON candles (symbol, interval, ts)
    """)
    cur.executemany("""
        INSERT OR REPLACE INTO candles (
            symbol, interval, broker, ts, open, high, low, close, volume, oi, source, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, sqlite_rows)
    conn.commit()
    conn.close()
    print(f"Seeded {len(sqlite_rows)} candles into {db_path}.")

    # 2. Populate Parquet cache
    cache_path = Path("data/cache/NIFTY_5minute_dhan.parquet")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(parquet_records)
    df.set_index("timestamp", inplace=True)
    df.to_parquet(cache_path)
    print(f"Seeded {len(df)} candles into {cache_path}.")


if __name__ == "__main__":
    generate_candles()

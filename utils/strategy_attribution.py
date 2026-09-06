"""
utils/strategy_attribution.py — Strategy Performance Attribution
================================================================
Answers: which registered strategies are actually generating alpha?

Every professional algo system tracks per-strategy performance.
This module reads the trade ledger and computes per-strategy metrics.

OUTPUTS:
  - Win rate per strategy
  - Average PnL contribution per strategy
  - How often each strategy is the "lead voter" in a winning trade
  - Strategy correlation matrix (do S1+S7 always fire together?)
  - Recommendations: which strategies to tune / remove / boost

Usage:
  python scripts/strategy_attribution.py             # last 30 days
  python scripts/strategy_attribution.py --days 90   # last 90 days
  python scripts/strategy_attribution.py --month 2026-04
"""

import csv
import json
import sqlite3
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from itertools import combinations

import numpy as np

try:
    from config.settings import JOURNAL_DIR
except ImportError:
    JOURNAL_DIR = "journal"

LEDGER_DB = Path(JOURNAL_DIR) / "signalforge.db"
LEDGER_CSV = Path(JOURNAL_DIR) / "master_trade_ledger.csv"


def load_trades(days: int = 30) -> list[dict]:
    """Load trades from ledger for the last N days."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    trades = []

    if LEDGER_DB.exists():
        conn = sqlite3.connect(LEDGER_DB)
        rows = conn.execute(
            "SELECT * FROM trades WHERE date >= ?", (cutoff,)
        ).fetchall()
        cols = [d[0] for d in conn.execute("PRAGMA table_info(trades)").fetchall()]
        conn.close()
        for row in rows:
            trades.append(dict(zip(cols, row)))
    elif LEDGER_CSV.exists():
        with open(LEDGER_CSV) as f:
            for row in csv.DictReader(f):
                if row.get("date", "") >= cutoff:
                    trades.append(row)

    return trades


def compute_attribution(trades: list[dict]) -> dict:
    """Compute per-strategy performance metrics."""
    strat_stats: dict[str, dict] = defaultdict(lambda: {
        "votes": 0, "lead_votes": 0, "trades_contributed": 0,
        "wins": 0, "losses": 0, "pnl_sum": 0.0, "pnl_list": [],
        "as_only_voter": 0,
    })

    all_combos: list[tuple] = []

    for trade in trades:
        strats_raw = trade.get("strategies_fired", "")
        if not strats_raw:
            continue
        strats = [s.strip() for s in str(strats_raw).split("|") if s.strip()]
        pnl    = float(trade.get("gross_pnl_pct", 0))
        win    = pnl > 0

        all_combos.extend(combinations(sorted(strats), 2))

        for i, s in enumerate(strats):
            strat_stats[s]["votes"]               += 1
            strat_stats[s]["trades_contributed"]  += 1
            strat_stats[s]["pnl_sum"]             += pnl
            strat_stats[s]["pnl_list"].append(pnl)
            if win:
                strat_stats[s]["wins"]  += 1
            else:
                strat_stats[s]["losses"] += 1
            if i == 0:
                strat_stats[s]["lead_votes"] += 1
            if len(strats) == 1:
                strat_stats[s]["as_only_voter"] += 1

    # Build summary per strategy
    summary = {}
    for name, stats in strat_stats.items():
        n    = stats["trades_contributed"]
        wins = stats["wins"]
        pnls = stats["pnl_list"]
        summary[name] = {
            "trades":       n,
            "win_rate":     round(wins / max(n, 1) * 100, 1),
            "avg_pnl":      round(stats["pnl_sum"] / max(n, 1), 2),
            "total_pnl":    round(stats["pnl_sum"], 2),
            "best":         round(max(pnls), 2) if pnls else 0,
            "worst":        round(min(pnls), 2) if pnls else 0,
            "lead_votes":   stats["lead_votes"],
            "solo_votes":   stats["as_only_voter"],
            "contribution_rank": 0,
        }

    # Rank by total PnL contribution
    ranked = sorted(summary.items(), key=lambda x: -x[1]["total_pnl"])
    for rank, (name, _) in enumerate(ranked, 1):
        summary[name]["contribution_rank"] = rank

    # Correlation matrix
    pair_counts: dict[tuple, int] = defaultdict(int)
    for combo in all_combos:
        pair_counts[combo] += 1

    # Normalise by total trades
    n_trades = len(trades)
    correlation = {}
    for (a, b), cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:20]:
        correlation[f"{a}+{b}"] = {
            "count":      cnt,
            "pct":        round(cnt / max(n_trades, 1) * 100, 1),
            "assessment": "HIGH CORRELATION — adds little diversity" if cnt > n_trades * 0.3
                          else "normal",
        }

    return {
        "period_days":   30,
        "total_trades":  n_trades,
        "strategies":    summary,
        "top_pairs":     correlation,
    }


def print_report(attr: dict) -> None:
    G   = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
    W   = "\033[97m"; D = "\033[2m";  RST = "\033[0m"

    print(f"\n{W}{'═'*70}{RST}")
    print(f"{W}  Strategy Attribution Report  |  "
          f"Last {attr['period_days']} days  |  "
          f"{attr['total_trades']} trades{RST}")
    print(f"{W}{'═'*70}{RST}\n")

    print(f"  {'Rank':<5} {'Strategy':<20} {'Trades':>7} {'WR':>7} "
          f"{'Avg PnL':>8} {'Total':>9} {'Best':>8} {'Worst':>8}")
    print(f"  {'─'*5} {'─'*20} {'─'*7} {'─'*7} {'─'*8} {'─'*9} {'─'*8} {'─'*8}")

    for name, stats in sorted(attr["strategies"].items(),
                               key=lambda x: x[1]["contribution_rank"]):
        wr   = stats["win_rate"]
        avg  = stats["avg_pnl"]
        col  = G if avg > 5 else Y if avg > 0 else R
        print(f"  {stats['contribution_rank']:<5} {name:<20} "
              f"{stats['trades']:>7} "
              f"{col}{wr:>6.1f}%{RST} "
              f"{col}{avg:>+8.2f}%{RST} "
              f"{col}{stats['total_pnl']:>+9.2f}%{RST} "
              f"{stats['best']:>+8.2f}% "
              f"{stats['worst']:>+8.2f}%")

    print(f"\n{W}  Top Strategy Pairs (co-vote frequency):{RST}")
    for pair, info in list(attr["top_pairs"].items())[:8]:
        col = Y if "HIGH" in info["assessment"] else D
        print(f"    {col}{pair:<35}{RST}  {info['count']:>4}× "
              f"({info['pct']:.1f}%)  {D}{info['assessment']}{RST}")

    # Recommendations
    print(f"\n{W}  Recommendations:{RST}")
    for name, stats in attr["strategies"].items():
        if stats["trades"] >= 5 and stats["win_rate"] < 35:
            print(f"  {R}  ⚠ {name}: win_rate={stats['win_rate']}% — consider disabling{RST}")
        elif stats["trades"] >= 5 and stats["avg_pnl"] > 15:
            print(f"  {G}  ✓ {name}: strong contributor (+{stats['avg_pnl']:.1f}% avg) — keep{RST}")
        elif stats["trades"] < 3:
            print(f"  {Y}  ? {name}: only {stats['trades']} trades — needs more data{RST}")
    print()


def save_report(attr: dict) -> str:
    out = Path(JOURNAL_DIR) / f"attribution_{date.today().isoformat()}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(attr, f, indent=2)
    return str(out)


if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",  type=int, default=30)
    parser.add_argument("--month", type=str, default=None)
    args = parser.parse_args()

    trades = load_trades(args.days)
    if not trades:
        print("No trades found in ledger for the specified period.")
        sys.exit(0)

    attr = compute_attribution(trades)
    attr["period_days"] = args.days
    print_report(attr)
    path = save_report(attr)
    print(f"  Report saved: {path}")

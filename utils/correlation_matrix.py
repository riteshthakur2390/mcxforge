"""
utils/correlation_matrix.py — Strategy Correlation & Independence Analysis
==========================================================================
Detects which strategies always fire together (redundant) vs
which fire independently (true diversity = better signal quality).

WHY THIS MATTERS:
  MIN_STRATEGY_VOTES = 2 assumes diversity. But if S1+S2 always
  co-vote (correlation = 0.95), getting 2 votes from them is NOT
  better than 1 vote — they are measuring the same thing.

  This module:
  1. Builds a co-occurrence matrix from trade history
  2. Flags highly correlated pairs (> 0.7)
  3. Recommends which pairs to treat as a single vote
  4. Identifies strategies that add genuine independence

OUTPUTS:
  - Correlation heatmap data (for dashboard)
  - Ranked list: most → least correlated pairs
  - Diversity score per strategy (0=always fires with others, 1=independent)
  - Recommendation: optimal MIN_STRATEGY_VOTES given current correlations

Usage:
  python utils/correlation_matrix.py
  from utils.correlation_matrix import StrategyCorrelation
  corr = StrategyCorrelation()
  report = corr.compute(days=90)
  corr.print_report(report)
"""

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from itertools import combinations

import numpy as np

try:
    from config.settings import JOURNAL_DIR
    from agents_code.agent2_strategy.runner import ALL_STRATEGY_NAMES
except ImportError:
    JOURNAL_DIR = "journal"
    ALL_STRATEGY_NAMES = [
        "SuperTrend+RSI", "VWAP+EMA", "ORB", "BBSqueeze", "ADX+PSAR",
        "FVG", "UTBot", "CPR", "Ichimoku", "VolumeProfile", "LiqSweep",
        "PriceAction", "OIAnalysis", "IVContraction", "AMD",
        "GapDirection", "SMC", "SkewHunter", "ExpiryWeek",
    ]

LEDGER_DB = Path(JOURNAL_DIR) / "signalforge.db"

HIGH_CORR_THRESHOLD = 0.70   # pairs above this = redundant
LOW_CORR_THRESHOLD  = 0.25   # pairs below this = truly independent

G   = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
W   = "\033[97m"; D = "\033[2m";  RST = "\033[0m"


class StrategyCorrelation:

    def compute(self, days: int = 90) -> dict:
        """
        Compute full correlation analysis from trade ledger.

        Returns dict with:
          matrix:        NxN co-occurrence matrix
          pairs:         sorted list of (strat_a, strat_b, correlation, assessment)
          diversity:     per-strategy independence score
          recommendations: list of actionable findings
          optimal_min_votes: suggested MIN_STRATEGY_VOTES
        """
        trades    = self._load_trades(days)
        strategies = sorted(set(ALL_STRATEGY_NAMES))
        n         = len(strategies)
        idx       = {s: i for i, s in enumerate(strategies)}

        # Build occurrence vectors
        fire_count   = defaultdict(int)
        co_fire      = defaultdict(lambda: defaultdict(int))
        total_candles = len(trades) or 1

        for trade in trades:
            strats_raw = trade.get("strategies_fired", "")
            fired = [s.strip() for s in str(strats_raw).split("|") if s.strip()]
            for s in fired:
                fire_count[s] += 1
            for a, b in combinations(sorted(fired), 2):
                co_fire[a][b] += 1
                co_fire[b][a] += 1

        # Compute Jaccard similarity for each pair
        pairs = []
        matrix = np.zeros((n, n))

        for a, b in combinations(strategies, 2):
            fa   = fire_count.get(a, 0)
            fb   = fire_count.get(b, 0)
            fab  = co_fire[a].get(b, 0)
            union = fa + fb - fab
            jaccard = fab / max(union, 1)
            matrix[idx[a]][idx[b]] = jaccard
            matrix[idx[b]][idx[a]] = jaccard

            if fa > 0 or fb > 0:
                assessment = (
                    "REDUNDANT — treat as 1 vote"  if jaccard > HIGH_CORR_THRESHOLD
                    else "CORRELATED — minor overlap"  if jaccard > 0.40
                    else "INDEPENDENT ✓"              if jaccard < LOW_CORR_THRESHOLD
                    else "NORMAL"
                )
                pairs.append({
                    "a":           a,
                    "b":           b,
                    "correlation": round(jaccard, 4),
                    "co_fires":    fab,
                    "a_fires":     fa,
                    "b_fires":     fb,
                    "assessment":  assessment,
                })

        pairs.sort(key=lambda x: -x["correlation"])

        # Diversity score per strategy
        # = 1 - (avg correlation with all other strategies)
        diversity = {}
        for s in strategies:
            if fire_count.get(s, 0) == 0:
                diversity[s] = {"score": 1.0, "fires": 0, "status": "inactive"}
                continue
            corrs = [
                co_fire[s].get(o, 0) / max(fire_count[s], 1)
                for o in strategies if o != s
            ]
            avg_corr = sum(corrs) / max(len(corrs), 1)
            score    = round(1.0 - avg_corr, 4)
            diversity[s] = {
                "score":   score,
                "fires":   fire_count[s],
                "status":  "independent" if score > 0.75
                           else "moderate"  if score > 0.50
                           else "redundant",
            }

        # Recommendations
        recommendations = []
        redundant_pairs = [p for p in pairs if p["correlation"] > HIGH_CORR_THRESHOLD]
        if redundant_pairs:
            for p in redundant_pairs[:3]:
                recommendations.append(
                    f"{p['a']} + {p['b']} co-fire {p['correlation']:.0%} of the time. "
                    f"Consider merging their signal weight or removing the weaker one."
                )

        inactive = [s for s, d in diversity.items() if d["fires"] == 0]
        if inactive:
            recommendations.append(
                f"Inactive strategies (0 fires in {days}d): {', '.join(inactive[:5])}. "
                f"Check min_candles, time gates, or entry conditions."
            )

        # Optimal MIN_STRATEGY_VOTES
        # If top pair is 0.9 correlated → need 3 votes minimum for real diversity
        max_corr = pairs[0]["correlation"] if pairs else 0
        optimal_votes = 3 if max_corr > HIGH_CORR_THRESHOLD else 2

        return {
            "days":               days,
            "trades_analysed":    len(trades),
            "strategies":         strategies,
            "matrix":             matrix.tolist(),
            "pairs":              pairs[:30],
            "diversity":          diversity,
            "recommendations":    recommendations,
            "optimal_min_votes":  optimal_votes,
            "max_pair_corr":      round(max_corr, 4),
        }

    def print_report(self, report: dict) -> None:
        print(f"\n{W}{'═'*65}{RST}")
        print(f"{W}  Strategy Correlation Analysis  |  "
              f"Last {report['days']} days  |  "
              f"{report['trades_analysed']} trades{RST}")
        print(f"{W}{'═'*65}{RST}\n")

        print(f"  {W}Top correlated pairs:{RST}")
        print(f"  {'Strategy A':<22} {'Strategy B':<22} {'Corr':>6}  Assessment")
        print(f"  {'─'*22} {'─'*22} {'─'*6}  {'─'*35}")

        for p in report["pairs"][:12]:
            corr = p["correlation"]
            col  = (R if corr > HIGH_CORR_THRESHOLD
                    else Y if corr > 0.40
                    else G)
            print(f"  {p['a']:<22} {p['b']:<22} "
                  f"{col}{corr:>6.3f}{RST}  {D}{p['assessment']}{RST}")

        print(f"\n  {W}Strategy independence scores:{RST}")
        for s, d in sorted(report["diversity"].items(),
                           key=lambda x: x[1]["score"], reverse=True)[:10]:
            sc  = d["score"]
            col = G if sc > 0.75 else Y if sc > 0.50 else R
            bar = "█" * int(sc * 20)
            print(f"  {s:<22}  {col}{sc:.3f}{RST}  "
                  f"{D}{bar:<20}{RST}  "
                  f"{d['fires']:>4} fires  {d['status']}")

        print(f"\n  {W}Recommendations:{RST}")
        for rec in report["recommendations"]:
            print(f"  {Y}▸{RST} {rec[:80]}")

        print(f"\n  {W}Optimal MIN_STRATEGY_VOTES: "
              f"{G}{report['optimal_min_votes']}{RST}  "
              f"(max pair correlation: {report['max_pair_corr']:.3f})\n")

    def save_report(self, report: dict) -> str:
        # Remove numpy matrix for JSON serialization
        save = {k: v for k, v in report.items() if k != "matrix"}
        out  = Path(JOURNAL_DIR) / f"correlation_{date.today().isoformat()}.json"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump(save, f, indent=2)
        return str(out)

    def _load_trades(self, days: int) -> list[dict]:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        if not LEDGER_DB.exists():
            return []
        conn  = sqlite3.connect(LEDGER_DB)
        cols  = [d[0] for d in conn.execute("PRAGMA table_info(trades)").fetchall()]
        rows  = conn.execute(
            "SELECT * FROM trades WHERE date >= ?", (cutoff,)
        ).fetchall()
        conn.close()
        return [dict(zip(cols, r)) for r in rows]


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(Path(__file__).parent.parent))
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()

    corr = StrategyCorrelation()
    report = corr.compute(args.days)
    corr.print_report(report)
    path = corr.save_report(report)
    print(f"  Saved: {path}")

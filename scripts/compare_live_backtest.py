"""
Compare live and backtest closed-trade performance using the shared ledger schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from utils.performance_review import build_live_vs_backtest_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare live vs backtest MCXForge performance")
    parser.add_argument("--live-pattern", action="append", default=[])
    parser.add_argument("--backtest-pattern", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_live_vs_backtest_report(
        live_patterns=args.live_pattern or None,
        backtest_patterns=args.backtest_pattern or None,
    )

    if args.json:
        print(json.dumps(report, default=str))
        return

    live = report["live"]
    backtest = report["backtest"]
    gaps = report["gaps"]
    readiness = report["automation_readiness"]

    print("MCXForge Live vs Backtest")
    print(f"Generated: {report['generated_at']}")
    print()
    print(
        f"Live      | trades={live['trades']} wr={live['win_rate_pct']:.2f}% "
        f"pf={live['profit_factor']:.2f} inr={live['net_realized_pnl']:.0f} "
        f"dd={live['max_drawdown_pct']:.2f}"
    )
    print(
        f"Backtest  | trades={backtest['trades']} wr={backtest['win_rate_pct']:.2f}% "
        f"pf={backtest['profit_factor']:.2f} inr={backtest['net_realized_pnl']:.0f} "
        f"dd={backtest['max_drawdown_pct']:.2f}"
    )
    print(
        f"Gaps      | trade_count={gaps['trade_count_gap']} wr={gaps['win_rate_gap_pct']:+.2f}% "
        f"pf={gaps['profit_factor_gap']:+.3f} inr={gaps['net_realized_pnl_gap']:+.0f}"
    )
    print()
    print(f"Automation ready: {'YES' if readiness['ready'] else 'NO'}")
    for check in readiness["checks"]:
        print(f" - [{'PASS' if check['passed'] else 'FAIL'}] {check['rule']}")


if __name__ == "__main__":
    main()

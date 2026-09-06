#!/usr/bin/env python3
"""
Parse a StrategyAgent backtest log + optional trades CSV for per-strategy contribution.

Usage:
  python3 scripts/strategy_signal_audit.py logs/backtest_YYYYMMDD_HHMMSS.log \\
      [--csv backtesting/results/backtest_trades_YYYYMMDD_HHMMSS.csv]

Evidence extracted:
  - [VOTES] lines: directional vote counts (CALL/PUT lists)
  - [RAW] RAW_SIGNAL lines: strategies that formed a min-vote raw signal
  - ml_filtering_ranking passed/filtered counts (global)
  - CSV strategy_combo: realized PnL attribution (sum split equally not applied — full sum to each combo member)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict

# Must match STRATEGY_REGISTRY order/names in agents_code/agent2_strategy/runner.py
REGISTRY_NAMES = [
    "SuperTrend+RSI",
    "VWAP+EMA",
    "ORB",
    "BBSqueeze",
    "ADX+PSAR",
    "FVG",
    "UTBot",
    "CPR",
    "Ichimoku",
    "VolumeProfile",
    "LiqSweep",
    "PriceAction",
    "OIAnalysis",
    "IVContraction",
    "AMD",
    "GapDirection",
    "SkewHunter",
    "SMC",
    "ExpiryWeek",
]


def _qnames(s: str) -> list[str]:
    return re.findall(r"'([^']+)'", s or "")


def audit_log(path: str) -> dict:
    vote_re = re.compile(
        r"\[VOTES\]\s+CALL=\d+\[(?P<call>[^\]]*)\].*?PUT=\d+\[(?P<put>[^\]]*)\].*?eligible=(\d+)"
    )
    raw_re = re.compile(r"\[RAW\] RAW_SIGNAL.*strategies=\[(?P<lst>[^\]]*)\]")
    raw2_re = re.compile(r"\[RAW\] (BUY_CALL|BUY_PUT).*?\|\s*\[(?P<lst>[^\]]+)\]\s*\|")

    votes_call: Counter[str] = Counter()
    votes_put: Counter[str] = Counter()
    eligible_hist: Counter[int] = Counter()
    raw_by: Counter[str] = Counter()
    raw_n = 0
    ml_pass = 0
    ml_fail: Counter[str] = Counter()

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = vote_re.search(line)
            if m:
                eligible_hist[int(m.group(3))] += 1
                for n in _qnames(m.group("call")):
                    votes_call[n] += 1
                for n in _qnames(m.group("put")):
                    votes_put[n] += 1
            if raw_re.search(line):
                raw_n += 1
                m = raw_re.search(line)
                if m:
                    for n in _qnames(m.group("lst")):
                        raw_by[n] += 1
                continue
            m2 = raw2_re.search(line)
            if m2:
                for n in _qnames(m2.group("lst")):
                    raw_by[n] += 1
            if "ml_filtering_ranking | status=passed" in line:
                ml_pass += 1
            if "ml_filtering_ranking | status=filtered" in line:
                mr = re.search(r"reason=([^|]+)", line)
                ml_fail[mr.group(1).strip() if mr else "unknown"] += 1

    votes_any: Counter[str] = Counter(votes_call)
    votes_any.update(votes_put)
    return {
        "vote_lines": sum(eligible_hist.values()),
        "eligible_hist": dict(sorted(eligible_hist.items())),
        "votes_call": dict(votes_call),
        "votes_put": dict(votes_put),
        "votes_any": dict(votes_any),
        "raw_signal_lines": raw_n,
        "raw_combo_hits": dict(raw_by),
        "ml_passed": ml_pass,
        "ml_filtered_top": ml_fail.most_common(12),
    }


def audit_csv(path: str) -> dict[str, dict]:
    pnl_by: defaultdict[str, float] = defaultdict(float)
    n_by: Counter[str] = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            combo = (row.get("strategy_combo") or "").strip()
            pnl = float(row.get("realized_pnl") or 0.0)
            for part in combo.split("|"):
                p = part.strip()
                if p:
                    pnl_by[p] += pnl
                    n_by[p] += 1
    return {k: {"trades": n_by[k], "pnl_rs": round(pnl_by[k], 2)} for k in sorted(n_by)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Strategy vote / raw / trade audit from log+CSV")
    ap.add_argument("log", help="Path to backtest_*.log")
    ap.add_argument("--csv", help="Optional backtest_trades_*.csv for PnL attribution")
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args()

    L = audit_log(args.log)
    C = audit_csv(args.csv) if args.csv else {}

    rows = []
    for name in REGISTRY_NAMES:
        va = int(L["votes_any"].get(name, 0))
        raw_h = int(L["raw_combo_hits"].get(name, 0))
        tr = C.get(name, {}).get("trades", 0) if C else 0
        pnl = C.get(name, {}).get("pnl_rs", 0.0) if C else 0.0
        useful = "ACTIVE" if va > 0 else "SILENT"
        rows.append(
            {
                "strategy": name,
                "votes_any": va,
                "raw_combo_hits": raw_h,
                "trades": tr,
                "pnl_rs": pnl,
                "useful": useful,
            }
        )

    out = {"log": args.log, "csv": args.csv or "", "summary": L, "per_strategy": rows}
    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"Log: {args.log}")
    print(f"Vote lines: {L['vote_lines']} | eligible buckets: {L['eligible_hist']}")
    print(f"RAW_SIGNAL (parsed): {L['raw_signal_lines']} | ML passed: {L['ml_passed']}")
    print("(PnL column sums realized_pnl for every trade listing that strategy in strategy_combo — multi-strat trades count in each row.)")
    print("Top ML reject reasons:", L["ml_filtered_top"])
    print()
    hdr = f"{'strategy':16} | {'votes':>5} | {'raw':>4} | {'trades':>6} | {'pnl_rs':>10} | useful"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['strategy']:16} | {r['votes_any']:5} | {r['raw_combo_hits']:4} | "
            f"{r['trades']:6} | {r['pnl_rs']:10} | {r['useful']}"
        )


if __name__ == "__main__":
    main()

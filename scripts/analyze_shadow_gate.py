#!/usr/bin/env python3
"""
scripts/analyze_shadow_gate.py — Automated Shadow High-Quality Entry Gate Evaluation Tool

Deterministic local analyzer to evaluate shadow gate performance on live trading data.
Operates entirely locally on CSV/journal files.

Usage:
    python3 scripts/analyze_shadow_gate.py [--journal-dir journal/] [--output-dir analysis/]
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_float(val, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_int(val, default: int = 0) -> int:
    try:
        if val is None or val == "":
            return default
        return int(float(val))
    except (ValueError, TypeError):
        return default


def extract_category_count(strategies_str: str) -> int:
    if not strategies_str:
        return 0
    strats = [s.strip() for s in str(strategies_str).replace(",", "|").split("|") if s.strip()]
    cats = set()
    for s in strats:
        s_low = s.lower()
        if any(w in s_low for w in ["supertrend", "ema", "trend", "adx", "psar", "ichimoku", "parabolic"]):
            cats.add("TREND")
        elif any(w in s_low for w in ["rsi", "bb", "bollinger", "mean", "stoch", "reversal"]):
            cats.add("MOMENTUM_REVERSAL")
        elif any(w in s_low for w in ["vwap", "cpr", "fvg", "pivot", "support", "resistance"]):
            cats.add("PRICE_ACTION_STRUCTURE")
        elif any(w in s_low for w in ["volume", "oi", "gamma", "pcr"]):
            cats.add("INSTITUTIONAL_FLOW")
        else:
            cats.add(s)
    return len(cats)


def compute_group_metrics(trades: List[dict]) -> dict:
    count = len(trades)
    if count == 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_pnl": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
        }

    pnls = [parse_float(t.get("realized_pnl", 0.0)) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net_pnl = sum(pnls)

    win_rate = round(len(wins) / count * 100.0, 2)
    avg_w = round(gross_profit / len(wins), 2) if wins else 0.0
    avg_l = round(-gross_loss / len(losses), 2) if losses else 0.0
    expectancy = round(net_pnl / count, 2)
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    return {
        "trades": count,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
        "avg_winner": avg_w,
        "avg_loser": avg_l,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


def run_shadow_gate_analysis(
    journal_dir: str = "journal",
    output_dir: str = "analysis",
    trades_file: Optional[str] = None,
) -> dict:
    j_path = Path(journal_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Discover signal journals
    signal_files = sorted(j_path.glob("signals_*.csv"))
    if not signal_files:
        print(f"[Shadow Analyzer] No signal journal files found in {journal_dir}")

    all_signals: List[dict] = []
    for f in signal_files:
        try:
            with open(f, newline="", encoding="utf-8", errors="ignore") as fp:
                reader = csv.DictReader(fp)
                for r in reader:
                    r["_source_file"] = f.name
                    all_signals.append(r)
        except Exception as e:
            print(f"[Shadow Analyzer] Warning reading {f}: {e}")

    # 2. Discover closed trades / trade ledger / history
    closed_trades_by_id: Dict[str, dict] = {}
    closed_trades_by_tuple: Dict[Tuple[str, str, str, str], dict] = {}
    ledger_files = [
        j_path / "closed_trades.csv",
        j_path / "master_trade_ledger.csv",
        j_path / "live_trade_history.csv",
    ]
    if trades_file:
        ledger_files.insert(0, Path(trades_file))

    raw_closed_trades: List[dict] = []
    for lf in ledger_files:
        if lf.exists():
            try:
                with open(lf, newline="", encoding="utf-8", errors="ignore") as fp:
                    reader = csv.DictReader(fp)
                    for r in reader:
                        raw_closed_trades.append(r)
                        sig_id = r.get("signal_id", "").strip()
                        if sig_id:
                            closed_trades_by_id[sig_id] = r
                        
                        dt = r.get("date", "").strip()
                        tm = r.get("entry_time", r.get("time", "")).strip()
                        sym = r.get("symbol", "NIFTY").strip()
                        dir_ = r.get("direction", "").strip()
                        if dt and tm:
                            closed_trades_by_tuple[(dt, tm, sym, dir_)] = r
            except Exception as e:
                print(f"[Shadow Analyzer] Warning reading {lf}: {e}")

    # 3. Join Signals & Executed Trades
    matched_trades: List[dict] = []
    unmatched_signals: List[dict] = []
    unmatched_trades: List[dict] = []

    # Map signals
    signal_by_id: Dict[str, dict] = {}
    signal_by_tuple: Dict[Tuple[str, str, str, str], dict] = {}
    for s in all_signals:
        sid = s.get("signal_id", "").strip()
        if sid:
            signal_by_id[sid] = s
        dt = s.get("date", "").strip()
        tm = s.get("entry_time", s.get("time", "")).strip()
        sym = s.get("symbol", "NIFTY").strip()
        dir_ = s.get("direction", "").strip()
        if dt and tm:
            signal_by_tuple[(dt, tm, sym, dir_)] = s

    for s in all_signals:
        status = str(s.get("lifecycle_status", "")).upper()
        pnl_val = s.get("realized_pnl", "")
        has_pnl = pnl_val is not None and str(pnl_val).strip() != "" and str(pnl_val).strip() != "0" and str(pnl_val).strip() != "0.0"
        
        is_taken = status in ("ORDERED", "CLOSED") or has_pnl
        if is_taken:
            sid = s.get("signal_id", "").strip()
            dt = s.get("date", "").strip()
            tm = s.get("entry_time", s.get("time", "")).strip()
            sym = s.get("symbol", "NIFTY").strip()
            dir_ = s.get("direction", "").strip()
            tuple_key = (dt, tm, sym, dir_)

            # 1. Primary Join: signal_id
            if sid and sid in closed_trades_by_id:
                enriched = {**s, **closed_trades_by_id[sid]}
                enriched["_join_type"] = "SIGNAL_ID_EXACT"
            # 2. Deterministic Fallback Join: (date, time, symbol, direction)
            elif tuple_key in closed_trades_by_tuple:
                enriched = {**s, **closed_trades_by_tuple[tuple_key]}
                enriched["_join_type"] = "DETERMINISTIC_TUPLE_FALLBACK"
            else:
                enriched = dict(s)
                enriched["_join_type"] = "SIGNAL_JOURNAL_INTRADAY"

            # Ensure realized_pnl is present
            if not enriched.get("realized_pnl") and pnl_val:
                enriched["realized_pnl"] = pnl_val

            matched_trades.append(enriched)
        else:
            unmatched_signals.append(s)

    # Check for closed trades that did not match any signal
    seen_matched_sig_ids = {t.get("signal_id") for t in matched_trades if t.get("signal_id")}
    for tr in raw_closed_trades:
        tr_sid = tr.get("signal_id", "").strip()
        tr_dt = tr.get("date", "").strip()
        tr_tm = tr.get("entry_time", tr.get("time", "")).strip()
        tr_sym = tr.get("symbol", "NIFTY").strip()
        tr_dir = tr.get("direction", "").strip()
        tr_tuple = (tr_dt, tr_tm, tr_sym, tr_dir)

        if tr_sid and tr_sid in seen_matched_sig_ids:
            continue
        elif tr_tuple in signal_by_tuple:
            continue
        else:
            unmatched_trades.append(tr)

    # Date range (sanitized)
    import re
    date_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    valid_dates = [s.get("date", "").strip() for s in all_signals if date_pat.match(str(s.get("date", "")).strip())]
    min_date = min(valid_dates) if valid_dates else "N/A"
    max_date = max(valid_dates) if valid_dates else "N/A"

    # 4. Group Actual Executed Trades
    group_pass: List[dict] = []
    group_fail: List[dict] = []
    group_unavail: List[dict] = []

    for t in matched_trades:
        # Determine shadow gate state
        gate_state = str(
            t.get("shadow_high_quality_entry_gate_state")
            or t.get("shadow_joint_rule_decision")
            or t.get("shadow_decision")
            or ""
        ).upper().strip()

        if gate_state == "PASS":
            group_pass.append(t)
        elif gate_state == "FAIL":
            group_fail.append(t)
        elif gate_state == "UNAVAILABLE":
            group_unavail.append(t)
        else:
            # Reconstruct from telemetry if gate column was not present
            has_votes = t.get("votes") is not None and str(t.get("votes")).strip() != ""
            timing_val = t.get("timing_classification") or t.get("shadow_timing_state") or t.get("entry_timing")
            has_timing = bool(timing_val)
            has_ml = (t.get("ml_confidence") is not None and str(t.get("ml_confidence")).strip() != "") or (t.get("ml_rank_score") is not None and str(t.get("ml_rank_score")).strip() != "")
            
            strats_raw = t.get("strategies_fired", t.get("strategy_combo", ""))
            derived_cats = extract_category_count(strats_raw)
            cats_raw = t.get("independent_category_count") or t.get("shadow_independent_category_count")
            cats = parse_int(cats_raw, derived_cats)
            has_cats = bool(cats > 0)

            if not (has_votes and has_timing and has_ml and has_cats):
                group_unavail.append(t)
            else:
                r_votes = parse_int(t.get("votes", t.get("shadow_raw_vote_count", 0)))
                timing = str(timing_val).upper().strip()
                ml_c = parse_float(t.get("ml_confidence", t.get("ml_conf", 0.0)))
                ml_r = parse_float(t.get("ml_rank_score", t.get("ml_rank", 0.0)))

                is_ml_pos = (ml_c > 0.0 or ml_r >= 0.50)
                is_timing_valid = (timing not in ("EXTENDED", "EXHAUSTED", "LATE")) and (timing in ("VALID", "EARLY", "NORMAL", "MID") or timing == "")
                is_votes_ok = (r_votes >= 7)
                is_cats_ok = (cats >= 2)

                if is_ml_pos and is_timing_valid and is_votes_ok and is_cats_ok:
                    group_pass.append(t)
                else:
                    group_fail.append(t)

    # 5. Core Metrics Calculation
    m_base = compute_group_metrics(matched_trades)
    m_pass = compute_group_metrics(group_pass)
    m_fail = compute_group_metrics(group_fail)
    m_unavail = compute_group_metrics(group_unavail)

    # Counterfactual = Baseline excluding Shadow FAIL (i.e. Group PASS + Group UNAVAILABLE)
    counterfactual_trades = group_pass + group_unavail
    m_counterfactual = compute_group_metrics(counterfactual_trades)

    losses_avoided = m_fail["gross_loss"]
    winners_lost = m_fail["wins"]
    profits_lost = m_fail["gross_profit"]
    net_impact = round(m_counterfactual["net_pnl"] - m_base["net_pnl"], 2)

    # 6. Failure Reason Breakdown
    reason_counts: Dict[str, List[dict]] = {}
    for t in group_fail:
        raw_reasons = str(t.get("shadow_high_quality_entry_gate_reasons", "")).strip()
        if not raw_reasons or raw_reasons == "[]" or raw_reasons == "None":
            # Reconstruct diagnostic reasons
            r_list = []
            ml_c = parse_float(t.get("ml_confidence", t.get("ml_conf", 0.0)))
            ml_r = parse_float(t.get("ml_rank_score", t.get("ml_rank", 0.0)))
            timing = str(t.get("timing_classification", t.get("shadow_timing_state", ""))).upper()
            votes = parse_int(t.get("votes", t.get("shadow_raw_vote_count", 0)))
            cats = parse_int(t.get("independent_category_count", t.get("shadow_independent_category_count", 0)))

            if ml_c <= 0.0 and ml_r < 0.50:
                r_list.append("ML_NOT_POSITIVE")
            if timing == "EXTENDED":
                r_list.append("TIMING_EXTENDED")
            elif timing == "EXHAUSTED":
                r_list.append("TIMING_EXHAUSTED")
            if votes < 7:
                r_list.append(f"INSUFFICIENT_RAW_VOTES({votes}<7)")
            if cats < 2:
                r_list.append(f"INSUFFICIENT_INDEPENDENT_CATEGORIES({cats}<2)")
            raw_reasons = "|".join(r_list) if r_list else "OTHER_FAIL"

        # Split multiple reasons
        reasons_split = raw_reasons.split("|")
        for r in reasons_split:
            r_clean = r.strip()
            if r_clean:
                reason_counts.setdefault(r_clean, []).append(t)
        # Also track combined pattern
        combo_key = " & ".join(sorted([r.strip() for r in reasons_split if r.strip()]))
        if combo_key and len(reasons_split) > 1:
            reason_counts.setdefault(f"COMBO: {combo_key}", []).append(t)

    failure_breakdown_rows = []
    for r_name, r_trades in sorted(reason_counts.items(), key=lambda x: len(x[1]), reverse=True):
        r_m = compute_group_metrics(r_trades)
        failure_breakdown_rows.append({
            "failure_reason": r_name,
            "trades_blocked": r_m["trades"],
            "losses_avoided_count": r_m["losses"],
            "winners_lost_count": r_m["wins"],
            "losses_avoided_pnl": r_m["gross_loss"],
            "profits_missed_pnl": r_m["gross_profit"],
            "net_pnl_saved": round(r_m["gross_loss"] - r_m["gross_profit"], 2),
        })

    # 7. Daily Summary
    daily_groups: Dict[str, List[dict]] = {}
    for t in matched_trades:
        dt = t.get("date", "UNKNOWN")
        daily_groups.setdefault(dt, []).append(t)

    daily_summary_rows = []
    for dt in sorted(daily_groups.keys()):
        d_trades = daily_groups[dt]
        d_base = compute_group_metrics(d_trades)
        d_pass = [x for x in d_trades if x in group_pass or x in group_unavail]
        d_cf = compute_group_metrics(d_pass)
        daily_summary_rows.append({
            "date": dt,
            "actual_trades": d_base["trades"],
            "actual_pnl": d_base["net_pnl"],
            "shadow_accepted_trades": d_cf["trades"],
            "shadow_pnl": d_cf["net_pnl"],
            "pnl_difference": round(d_cf["net_pnl"] - d_base["net_pnl"], 2),
        })

    # 8. Export CSV Outputs
    # A. Trade Comparison CSV
    trade_comp_rows = []
    for t in matched_trades:
        is_pass = t in group_pass
        is_fail = t in group_fail
        gate_state = "PASS" if is_pass else ("FAIL" if is_fail else "UNAVAILABLE")
        pnl = parse_float(t.get("realized_pnl", 0.0))
        trade_comp_rows.append({
            "signal_id": t.get("signal_id", ""),
            "date": t.get("date", ""),
            "time": t.get("time", ""),
            "symbol": t.get("symbol", "NIFTY"),
            "direction": t.get("direction", ""),
            "votes": parse_int(t.get("votes", t.get("shadow_raw_vote_count", 0))),
            "ml_conf": parse_float(t.get("ml_confidence", t.get("ml_conf", 0.0))),
            "ml_rank": parse_float(t.get("ml_rank_score", t.get("ml_rank", 0.0))),
            "timing_state": t.get("timing_classification", t.get("shadow_timing_state", "")),
            "gate_state": gate_state,
            "gate_reasons": t.get("shadow_high_quality_entry_gate_reasons", ""),
            "disagreement": t.get("disagreement", f"LIVE_TRADE_SHADOW_{gate_state}"),
            "join_type": t.get("_join_type", "SIGNAL_JOURNAL_INTRADAY"),
            "realized_pnl": pnl,
            "outcome": "WIN" if pnl > 0 else "LOSS",
            "would_have_blocked": is_fail,
        })
    df_trades_comp = Path(output_dir) / "shadow_gate_trade_comparison.csv"
    if trade_comp_rows:
        with open(df_trades_comp, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(trade_comp_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trade_comp_rows)

    # B. Failure Reason Analysis CSV
    df_failure_csv = Path(output_dir) / "shadow_gate_failure_reason_analysis.csv"
    if failure_breakdown_rows:
        with open(df_failure_csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(failure_breakdown_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failure_breakdown_rows)

    # C. Daily Summary CSV
    df_daily_csv = Path(output_dir) / "shadow_gate_daily_summary.csv"
    if daily_summary_rows:
        with open(df_daily_csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(daily_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(daily_summary_rows)

    # D. Unmatched Records CSV
    unmatched_rows = []
    for us in unmatched_signals:
        unmatched_rows.append({
            "type": "SIGNAL_UNMATCHED_NO_TRADE",
            "signal_id": us.get("signal_id", ""),
            "date": us.get("date", ""),
            "time": us.get("time", ""),
            "symbol": us.get("symbol", "NIFTY"),
            "direction": us.get("direction", ""),
            "lifecycle_status": us.get("lifecycle_status", "RAW"),
            "source_file": us.get("_source_file", ""),
        })
    for ut in unmatched_trades:
        ut_sid = ut.get("signal_id", "").strip()
        unmatched_type = "TRADE_UNMATCHED_NO_SIGNAL" if ut_sid else "UNMATCHED_LEGACY_DATA"
        unmatched_rows.append({
            "type": unmatched_type,
            "signal_id": ut_sid,
            "date": ut.get("date", ""),
            "time": ut.get("time", ut.get("entry_time", "")),
            "symbol": ut.get("symbol", "NIFTY"),
            "direction": ut.get("direction", ""),
            "lifecycle_status": ut.get("lifecycle_status", "CLOSED"),
            "source_file": "closed_trades/ledger",
        })
    df_unmatched_csv = Path(output_dir) / "shadow_gate_unmatched.csv"
    if unmatched_rows:
        with open(df_unmatched_csv, "w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(unmatched_rows[0].keys()))
            writer.writeheader()
            writer.writerows(unmatched_rows)

    # E. Summary JSON
    summary_dict = {
        "metadata": {
            "date_range": f"{min_date} to {max_date}",
            "total_signal_rows": len(all_signals),
            "completed_trades_matched": len(matched_trades),
            "unmatched_signals_count": len(unmatched_signals),
            "unmatched_trades_count": len(unmatched_trades),
        },
        "coverage": {
            "group_pass_trades": len(group_pass),
            "group_fail_trades": len(group_fail),
            "group_unavail_trades": len(group_unavail),
            "pass_coverage_pct": round(len(group_pass) / len(matched_trades) * 100.0, 2) if matched_trades else 0.0,
            "fail_coverage_pct": round(len(group_fail) / len(matched_trades) * 100.0, 2) if matched_trades else 0.0,
            "unavail_coverage_pct": round(len(group_unavail) / len(matched_trades) * 100.0, 2) if matched_trades else 0.0,
        },
        "performance_comparison": {
            "baseline": m_base,
            "group_pass": m_pass,
            "group_fail": m_fail,
            "group_unavail": m_unavail,
            "counterfactual": m_counterfactual,
        },
        "counterfactual_impact": {
            "trades_removed": m_fail["trades"],
            "losing_trades_avoided": m_fail["losses"],
            "winning_trades_lost": winners_lost,
            "losses_avoided_amount": losses_avoided,
            "profits_lost_amount": profits_lost,
            "net_pnl_improvement": net_impact,
            "expectancy_change": round(m_counterfactual["expectancy"] - m_base["expectancy"], 2),
            "profit_factor_change": round(m_counterfactual["profit_factor"] - m_base["profit_factor"], 2),
        },
    }

    df_summary_json = Path(output_dir) / "shadow_gate_summary.json"
    with open(df_summary_json, "w") as fp:
        json.dump(summary_dict, fp, indent=2)

    return summary_dict


def print_cli_summary(summary: dict) -> None:
    meta = summary["metadata"]
    cov = summary["coverage"]
    perf = summary["performance_comparison"]
    impact = summary["counterfactual_impact"]

    b = perf["baseline"]
    p = perf["group_pass"]
    f = perf["group_fail"]
    cf = perf["counterfactual"]

    print("\n" + "=" * 80)
    print(f"SHADOW HIGH-QUALITY ENTRY GATE EVALUATION ({meta['date_range']})")
    print("=" * 80)

    print(f"\n[DATA COVERAGE]")
    print(f"  • Signal Journal Rows:      {meta['total_signal_rows']:,}")
    print(f"  • Completed Trades Matched:  {meta['completed_trades_matched']}")
    print(f"  • Unmatched Signals:         {meta['unmatched_signals_count']}")
    print(f"  • Unmatched Trades:          {meta['unmatched_trades_count']}")
    print(f"  • PASS Coverage:             {cov['group_pass_trades']} ({cov['pass_coverage_pct']}%)")
    print(f"  • FAIL Coverage:             {cov['group_fail_trades']} ({cov['fail_coverage_pct']}%)")
    print(f"  • UNAVAILABLE Coverage:      {cov['group_unavail_trades']} ({cov['unavail_coverage_pct']}%)")

    print(f"\n[COHORT PERFORMANCE COMPARISON]")
    print("┌──────────────────────────────────┬──────────┬──────────┬───────────┬─────────────┬──────────────┐")
    print("│ Cohort Group                     │ Trades   │ Win Rate │ PF        │ Expectancy  │ Net P&L (₹)  │")
    print("├──────────────────────────────────┼──────────┼──────────┼───────────┼─────────────┼──────────────┤")
    print(f"│ Baseline (All Actual Trades)     │ {b['trades']:2d} trades │ {b['win_rate']:5.1f}%   │ {b['profit_factor']:4.2f}      │ -₹{abs(b['expectancy']):6.2f}    │ -₹{abs(b['net_pnl']):8.2f}   │")
    print(f"│ Group A (Live TRADE + Pass)      │ {p['trades']:2d} trades │ {p['win_rate']:5.1f}%   │ {p['profit_factor']:4.2f}      │ -₹{abs(p['expectancy']):6.2f}    │ -₹{abs(p['net_pnl']):8.2f}   │")
    print(f"│ Group B (Live TRADE + Fail)      │ {f['trades']:2d} trades │ {f['win_rate']:5.1f}%   │ {f['profit_factor']:4.2f}      │ -₹{abs(f['expectancy']):6.2f}    │ -₹{abs(f['net_pnl']):8.2f}   │")
    print(f"│ Counterfactual (Excl. Fail) ★    │ {cf['trades']:2d} trades │ {cf['win_rate']:5.1f}%   │ {cf['profit_factor']:4.2f}      │ -₹{abs(cf['expectancy']):6.2f}    │ -₹{abs(cf['net_pnl']):8.2f}   │")
    print("└──────────────────────────────────┴──────────┴──────────┴───────────┴─────────────┴──────────────┘")

    print(f"\n[COUNTERFACTUAL IMPACT]")
    print(f"  • Trades Removed:            {impact['trades_removed']} trades ({impact['losing_trades_avoided']}L / {impact['winning_trades_lost']}W)")
    print(f"  • Losses Avoided:            +₹{impact['losses_avoided_amount']:,.2f}")
    print(f"  • Profits Lost (False Neg):  ₹{impact['profits_lost_amount']:,.2f} ({impact['winning_trades_lost']} winners lost)")
    print(f"  • Net Financial Benefit:     +₹{impact['net_pnl_improvement']:,.2f}")
    print(f"  • Win Rate Improvement:      {b['win_rate']:.1f}% → {cf['win_rate']:.1f}% (+{cf['win_rate'] - b['win_rate']:.1f}%)")
    print(f"  • Expectancy Improvement:    ₹{b['expectancy']:.2f} → ₹{cf['expectancy']:.2f} (+₹{impact['expectancy_change']:.2f})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Shadow High-Quality Entry Gate on Live Data.")
    parser.add_argument("--journal-dir", default="journal", help="Directory containing signals_*.csv")
    parser.add_argument("--output-dir", default="analysis", help="Output directory for CSV and JSON reports")
    parser.add_argument("--trades-file", default=None, help="Optional path to closed trades CSV")
    args = parser.parse_args()

    summary = run_shadow_gate_analysis(
        journal_dir=args.journal_dir,
        output_dir=args.output_dir,
        trades_file=args.trades_file,
    )
    print_cli_summary(summary)

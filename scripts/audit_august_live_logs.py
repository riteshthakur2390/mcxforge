#!/usr/bin/env python3
"""
scripts/audit_august_live_logs.py

Comprehensive Parser of August 2026 Live Logs (logs/signalforge_2026-08-*.log):
1. Day-by-day table of all signals generated
2. Decision under the new system (P&L_MAXIMIZER_V1 & ALPHA_HUNTER):
   - TAKE or REJECT
   - Exact rule reason for rejection or acceptance
3. Deep-dive into the 5 live practice trades taken by the system
"""

import sys
import glob
import re
import ast
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

ANCHORS = {"VolumeProfile", "RangeSpread", "StrikeMomentum", "FVG", "ElliottWave", "SkewHunter"}

def parse_logs():
    log_files = sorted(glob.glob(str(ROOT_DIR / "logs/signalforge_2026-08-*.log")))
    
    # Precise regex matching line 2615 in runner.py
    sig_re = re.compile(
        r"\[RAW\] RAW_SIGNAL \| market_ts=([\d\-]+) (\d{2}:\d{2}) IST \| (BUY_CALL|BUY_PUT) \| conf=([\d\.]+) \| votes=(\d+)/\d+ eligible \| w=[\d\.]+ ws=[\d\.]+ \| strategies=(.*)"
    )

    daily_signals = defaultdict(list)

    for lf in log_files:
        with open(lf, errors="ignore") as f:
            for line in f:
                if "[RAW] RAW_SIGNAL" in line:
                    m = sig_re.search(line)
                    if m:
                        date_str = m.group(1)
                        time_str = m.group(2)
                        direction = m.group(3)
                        conf = float(m.group(4))
                        votes = int(m.group(5))
                        strats_str = m.group(6).strip()
                        try:
                            strats = ast.literal_eval(strats_str)
                        except Exception:
                            strats = [s.strip(" '\"[]") for s in strats_str.split(",") if s.strip(" '\"[]")]

                        daily_signals[date_str].append({
                            "date": date_str,
                            "time": time_str,
                            "direction": direction,
                            "votes": votes,
                            "strats": strats,
                            "conf": conf,
                        })

    return daily_signals

def evaluate_signal_under_new_system(sig):
    t_str = sig["time"]
    try:
        h, m = t_str.split(":")
        tod = int(h) + int(m) / 60.0
    except Exception:
        tod = 10.0

    votes = sig["votes"]
    strats = sig["strats"]
    direction = sig["direction"]
    dow = pd.to_datetime(sig["date"]).day_name()

    has_anchor = any(s in ANCHORS for s in strats)
    has_toxic_adx = "ADX+PSAR" in strats and not has_anchor

    # P&L_MAXIMIZER_V1 Gate Rules:
    # 1. Morning Chop Gate (10:00 to 11:30) requires >= 6 votes
    # 2. Friday afternoon (>= 13:00) cutoff
    # 3. Toxic ADX+PSAR without anchor rejected
    # 4. If no anchor present, requires >= 5 votes

    if 10.0 <= tod < 11.5 and votes < 6:
        return "REJECT", f"Morning Chop Gate ({votes} < 6 votes)"
    if dow == "Friday" and tod >= 13.0:
        return "REJECT", "Friday Afternoon Theta Cutoff (>= 13:00)"
    if has_toxic_adx:
        return "REJECT", "Toxic ADX+PSAR (Lagging momentum without anchor)"
    if not has_anchor and votes < 5:
        return "REJECT", f"Missing Structural Anchor ({votes} < 5 votes)"

    # Passed!
    anchor_name = [s for s in strats if s in ANCHORS]
    anchor_str = anchor_name[0] if anchor_name else "Strong Consensus"
    return "TAKE", f"Approved by Anchor ({anchor_str}) | Votes={votes}"

def main():
    daily_signals = parse_logs()
    
    print("="*95)
    print("           AUGUST 2026 LIVE LOGS AUDIT & NEW SYSTEM DECISION TABLE")
    print("="*95)

    all_rows = []
    
    for d in sorted(daily_signals.keys()):
        sigs = daily_signals[d]
        if not sigs:
            continue

        # Deduplicate signals in same 5m bucket
        seen_times = set()
        unique_sigs = []
        for s in sigs:
            if (s["time"], s["direction"]) not in seen_times:
                seen_times.add((s["time"], s["direction"]))
                unique_sigs.append(s)

        for s in unique_sigs:
            dec, reason = evaluate_signal_under_new_system(s)
            strats_display = ", ".join(s["strats"][:3]) if s["strats"] else f"{s['votes']} votes"
            all_rows.append({
                "Date": s["date"],
                "Time": s["time"],
                "Direction": s["direction"],
                "Votes": s["votes"],
                "Leading Strategies": strats_display,
                "New System Decision": dec,
                "Reason / Rule": reason
            })

    df_res = pd.DataFrame(all_rows)
    
    dates = df_res["Date"].unique()
    print(f"Total Trading Sessions with Signals: {len(dates)}")
    print(f"Total Candidate Signals Audited: {len(df_res)}")
    print(f"Total Signals TAKEN by New System: {(df_res['New System Decision'] == 'TAKE').sum()} ({(df_res['New System Decision'] == 'TAKE').mean()*100:.1f}%)")
    print(f"Total Signals REJECTED by New System: {(df_res['New System Decision'] == 'REJECT').sum()} ({(df_res['New System Decision'] == 'REJECT').mean()*100:.1f}%)\n")

    # Display grouped summary by date
    for d in dates:
        sub = df_res[df_res["Date"] == d]
        taken = (sub['New System Decision'] == 'TAKE').sum()
        rej = (sub['New System Decision'] == 'REJECT').sum()
        print(f"\n==========================================================================================")
        print(f"SESSION: {d} ({pd.to_datetime(d).day_name()}) | Total Signals: {len(sub)} | Taken: {taken} | Rejected: {rej}")
        print(f"==========================================================================================")
        print(sub[["Time", "Direction", "Votes", "Leading Strategies", "New System Decision", "Reason / Rule"]].to_string(index=False))

    # Save to CSV
    out_csv = ROOT_DIR / "analysis/live_state/august_2026_live_logs_audit_table.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved complete audit table to {out_csv}")

if __name__ == "__main__":
    main()

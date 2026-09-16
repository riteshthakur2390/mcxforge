#!/usr/bin/env python3
"""
scripts/eod_diagnostic.py — End-of-Day Strategy Diagnostic Report
==================================================================
Run at end of every trading day (or schedule via cron at 23:35 IST post MCX close).

    python scripts/eod_diagnostic.py
    python scripts/eod_diagnostic.py --date 2026-04-14
    python scripts/eod_diagnostic.py --log logs/mcxforge_20260414.log

WHAT IT ANSWERS:
  1. Which registered strategies fired today? Which were completely silent?
  2. Why was each strategy silent? (not enough candles? wrong regime? low volume?)
  3. Every vote: which strategy voted at what time, in which direction
  4. Every signal that reached voting: how many votes, who voted, what blocked it
  5. Every trade taken: entry time, direction, strategies behind it, outcome
  6. Root cause if NO trades were taken today
  7. Pipeline funnel: candles → regime → strategies → voting → ML → trade
  8. Strategy health score: how often each strategy has been contributing

OUTPUT:
  - Console: colour-coded summary
  - logs/eod_report_YYYY-MM-DD.txt — plain text for review
  - journal/eod_YYYY-MM-DD.json   — machine-readable for trend tracking

USAGE (cron):
  35 23 * * 1-5 cd /Users/vishranti/Downloads/projects/mcxforge && ./venv/bin/python scripts/eod_diagnostic.py --no-color >> logs/eod_cron.log 2>&1
"""

from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path
from datetime import date, datetime
from collections import defaultdict
from typing import Any

import pytz

IST = pytz.timezone("Asia/Kolkata")

# ── All strategy names (must match runner.py STRATEGY_REGISTRY) ───────────────
ALL_STRATEGIES = [
    "SuperTrend+RSI",   # S1
    "VWAP+EMA",         # S2
    "ORB",              # S3
    "BBSqueeze",        # S4
    "ADX+PSAR",         # S5
    "FVG",              # S6
    "UTBot",            # S7
    "CPR",              # S8
    "Ichimoku",         # S9  (needs 78 candles — rarely fires early)
    "VolumeProfile",    # S10
    "LiqSweep",         # S11
    "PriceAction",      # S12
    "OIAnalysis",       # S13 (live broker only)
    "IVContraction",    # S14
    "AMD",              # S15 (needs 60 candles)
    "GapDirection",     # S16 (09:15–10:00 only)
    "SkewHunter",       # S18
    "SMC",              # S17 (needs 50 candles)
    "ExpiryWeek",       # S19
]


def _registered_strategy_names() -> list[str]:
    try:
        from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY

        names = [str(meta.name) for meta in STRATEGY_REGISTRY]
        return names or ALL_STRATEGIES
    except Exception:
        return ALL_STRATEGIES


def _registered_strategy_min_candles() -> dict[str, int]:
    mins = dict(STRATEGY_MIN_CANDLES)
    try:
        from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY

        for meta in STRATEGY_REGISTRY:
            value = getattr(meta, "min_candles", None)
            if value is not None:
                mins[str(meta.name)] = int(value)
    except Exception:
        pass
    return mins

# Minimum candles each strategy needs (from STRATEGY_REGISTRY)
STRATEGY_MIN_CANDLES = {
    "SuperTrend+RSI": 30, "VWAP+EMA": 26, "ORB": 20,
    "BBSqueeze": 30, "ADX+PSAR": 20, "FVG": 25,
    "UTBot": 30, "CPR": 20, "Ichimoku": 60,
    "VolumeProfile": 50, "LiqSweep": 32, "PriceAction": 30,
    "OIAnalysis": 15, "IVContraction": 35, "AMD": 40,
    "GapDirection": 20, "SkewHunter": 30, "SMC": 50,
    "ExpiryWeek": 20,
}

# ── ANSI colours (stripped in file output) ────────────────────────────────────
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
R  = "\033[91m"   # red
B  = "\033[94m"   # blue
W  = "\033[97m"   # white bold
DIM = "\033[2m"
RST = "\033[0m"


def strip_ansi(s: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", s)


# ═════════════════════════════════════════════════════════════════════════════
# LOG PARSER
# ═════════════════════════════════════════════════════════════════════════════

class LogParser:
    """
    Parses an MCXForge log file and extracts all diagnostic events.
    Handles both live and backtest logs.
    """

    def __init__(self, log_path: Path, target_date: str):
        self.log_path    = log_path
        self.target_date = target_date   # YYYY-MM-DD
        self._single_day_log = (
            target_date in log_path.name
            or target_date.replace("-", "") in log_path.name
        )

        # Parsed structures
        self.candle_count      = 0
        self.regime_events:    list[dict] = []   # every MARKET_REGIME publish
        self.vote_events:      list[dict] = []   # every candle with >=1 vote
        self.no_vote_events:   list[dict] = []   # candles with zero votes
        self.raw_signals:      list[dict] = []   # every RAW_SIGNAL published
        self.ml_approved:      list[dict] = []
        self.ml_rejected:      list[dict] = []
        self.ml_bypass:        list[dict] = []
        self.trades:           list[dict] = []   # POSITION_CLOSED entries
        self.suppressed:       list[dict] = []   # SIGNAL_SUPPRESSED
        self.cooldown_blocks:  list[dict] = []
        self.insufficient_vote_events: list[dict] = []

        self._parse()

    # ── MAIN PARSE ─────────────────────────────────────────────────────────

    def _parse(self) -> None:
        if not self.log_path.exists():
            return

        with open(self.log_path, errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                # Live logs are already date-partitioned and many current log
                # lines start with HH:MM only. Do not drop those lines just
                # because the date is not repeated in every row.
                if not self._single_day_log and self.target_date not in line:
                    continue

                self._parse_line(line)

    def _parse_line(self, line: str) -> None:
        ts = self._extract_time(line)

        # ── Candle count (CANDLES_READY or regime fired) ──────────────────
        if "MARKET_REGIME" in line or "regime=" in line:
            regime = "UNKNOWN"
            for r in ("TRENDING", "CHOPPY", "HIGH_VOL", "RANGING"):
                if r in line:
                    regime = r
                    break
            adx = self._re_float(line, r"ADX[=:]([\d.]+)")
            chop = self._re_float(line, r"[Cc]hop[=:]([\d.]+)")
            vix  = self._re_float(line, r"[Vv]ix[=:]([\d.]+)")
            if regime != "UNKNOWN":
                self.regime_events.append({
                    "time": ts, "regime": regime,
                    "adx": adx, "chop": chop, "vix": vix,
                })
                self.candle_count += 1

        # ── Vote events (🗳️ line) ─────────────────────────────────────────
        if "🗳️" in line or "[VOTES]" in line:
            call_strats = self._extract_list(line, r"CALL=\d+\[([^\]]*)\]")
            put_strats  = self._extract_list(line, r"PUT=\d+\[([^\]]*)\]")
            eligible    = self._re_int(line, r"eligible=(\d+)")
            self.vote_events.append({
                "time": ts,
                "call_votes": call_strats,
                "put_votes":  put_strats,
                "total_votes": len(call_strats) + len(put_strats),
                "eligible": eligible,
            })

        # ── No votes ──────────────────────────────────────────────────────
        if "⬜ no votes" in line or "no votes" in line.lower():
            eligible = self._re_int(line, r"eligible=(\d+)")
            ltp = self._re_float(line, r"ltp=([\d.]+)")
            self.no_vote_events.append({
                "time": ts, "eligible": eligible, "ltp": ltp
            })

        # ── Insufficient votes ────────────────────────────────────────────
        if "votes insufficient" in line or "🔶" in line:
            call_strats = self._extract_list(line, r"CALL=\d+\[([^\]]*)\]")
            put_strats  = self._extract_list(line, r"PUT=\d+\[([^\]]*)\]")
            need = self._re_int(line, r"need=(\d+)")
            self.insufficient_vote_events.append({
                "time": ts,
                "call_votes": call_strats,
                "put_votes":  put_strats,
                "need": need,
            })

        # ── RAW_SIGNAL (🎯) ───────────────────────────────────────────────
        if "🎯" in line or ("[RAW] RAW_SIGNAL" in line and "[StrategyAgent]" in line):
            direction = "BUY_CALL" if "BUY_CALL" in line else "BUY_PUT" if "BUY_PUT" in line else "UNKNOWN"
            conf      = self._re_float(line, r"conf=([\d.]+)")
            votes     = self._re_int(line, r"votes=(\d+)")
            quality   = self._re_float(line, r"quality=([\d.]+)")
            strats    = self._extract_list(line, r"strategies=\[([^\]]*)\]")
            self.raw_signals.append({
                "time": ts, "direction": direction,
                "confidence": conf, "votes": votes,
                "quality": quality, "strategies": strats,
            })

        # ── ML decisions ─────────────────────────────────────────────────
        if "BYPASS" in line and ("MLFilter" in line or "ML gate" in line.lower()):
            self.ml_bypass.append({"time": ts, "line": line[-120:]})

        if ("✅ APPROVED" in line or "[OK] ML_RANK" in line) and "MLFilter" in line:
            rank = self._re_float(line, r"rank[=\s]([\d.]+)")
            req  = self._re_float(line, r"(?:required|thr)[=\s]([\d.]+)")
            tags = self._re_str(line, r"\(([^)]+)\)$")
            self.ml_approved.append({
                "time": ts, "rank": rank, "required": req, "tags": tags
            })

        if ("❌ REJECTED" in line or "[X] REJECTED" in line) and ("MLFilter" in line or "ML gate" in line):
            rank = self._re_float(line, r"rank[=\s]([\d.]+)")
            req  = self._re_float(line, r"(?:required|thr)[=\s]([\d.]+)")
            tags = self._re_str(line, r"\(([^)]+)\)$")
            self.ml_rejected.append({
                "time": ts, "rank": rank, "required": req, "tags": tags
            })

        # ── Trades (position closed) ──────────────────────────────────────
        if "POSITION_CLOSED" in line or "pnl=" in line.lower() and "closed" in line.lower():
            pnl     = self._re_float(line, r"pnl[=:]([+-]?[\d.]+)")
            exit_r  = self._re_str(line, r"exit[_=]?reason[=:]([A-Z_]+)")
            held    = self._re_float(line, r"held[=:](\d+)")
            if pnl is not None:
                self.trades.append({
                    "time": ts, "pnl": pnl,
                    "exit_reason": exit_r or "UNKNOWN",
                    "held_min": held,
                })

        # ── Suppressed / Cooldown ────────────────────────────────────────
        if "no_valid_setup" in line or "Suppressed" in line:
            reason = self._re_str(line, r"reason[=:]?([A-Za-z_]+)")
            self.suppressed.append({"time": ts, "reason": reason})

        if "Cooldown" in line or "cooldown" in line:
            elapsed = self._re_int(line, r"(\d+)m ago")
            self.cooldown_blocks.append({"time": ts, "elapsed_min": elapsed})

    # ── HELPERS ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_time(line: str) -> str:
        m = re.search(r"(\d{2}:\d{2}(?::\d{2})?)", line)
        return m.group(1)[:5] if m else "--:--"

    @staticmethod
    def _re_float(line: str, pattern: str) -> float | None:
        m = re.search(pattern, line)
        try:
            return float(m.group(1)) if m else None
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _re_int(line: str, pattern: str) -> int | None:
        m = re.search(pattern, line)
        try:
            return int(m.group(1)) if m else None
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _re_str(line: str, pattern: str) -> str | None:
        m = re.search(pattern, line)
        return m.group(1).strip() if m else None

    @staticmethod
    def _extract_list(line: str, pattern: str) -> list[str]:
        """Extract strategy names from a bracketed list in the log line."""
        m = re.search(pattern, line)
        if not m:
            return []
        raw = m.group(1)
        parts = re.findall(r"[A-Za-z][A-Za-z0-9+_]+", raw)
        return [p for p in parts if len(p) >= 3]


# ═════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

class EODReport:
    """Builds a full end-of-day diagnostic report from parsed log data."""

    def __init__(self, parser: LogParser, target_date: str):
        self.p    = parser
        self.date = target_date
        self.lines: list[str] = []   # output buffer

    def generate(self) -> str:
        self._header()
        self._market_conditions()
        self._pipeline_funnel()
        self._strategy_health()
        self._vote_timeline()
        self._signal_deep_dive()
        self._trade_summary()
        self._root_cause()
        self._recommendations()
        self._footer()
        return "\n".join(self.lines)

    # ── SECTIONS ─────────────────────────────────────────────────────────

    def _header(self) -> None:
        now = datetime.now(IST).strftime("%H:%M IST")
        self._box(f"MCXForge — EOD Diagnostic Report  |  {self.date}  |  Generated {now}")
        self._line()

    def _market_conditions(self) -> None:
        self._h1("1. MARKET CONDITIONS TODAY")

        if not self.p.regime_events:
            self._warn("No regime events found in log. System may not have run today.")
            return

        # Regime distribution
        regimes   = [e["regime"] for e in self.p.regime_events]
        reg_count = defaultdict(int)
        for r in regimes:
            reg_count[r] += 1

        total = len(regimes)
        self._kv("Total candles processed", str(total))
        self._line()

        for regime, cnt in sorted(reg_count.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            bar = "█" * int(pct / 5)
            col = G if regime == "TRENDING" else Y if regime == "RANGING" else R
            self._out(f"  {col}{regime:<15}{RST}  {cnt:>4} candles  ({pct:5.1f}%)  {DIM}{bar}{RST}")

        # ADX / VIX summary
        adx_vals = [e["adx"] for e in self.p.regime_events if e.get("adx")]
        vix_vals = [e["vix"] for e in self.p.regime_events if e.get("vix")]
        chop_vals= [e["chop"] for e in self.p.regime_events if e.get("chop")]

        self._line()
        if adx_vals:
            self._kv("ADX range today", f"{min(adx_vals):.1f} – {max(adx_vals):.1f}  (avg {sum(adx_vals)/len(adx_vals):.1f})")
        if chop_vals:
            self._kv("Chop Index range", f"{min(chop_vals):.1f} – {max(chop_vals):.1f}")
        if vix_vals:
            self._kv("India VIX range", f"{min(vix_vals):.1f} – {max(vix_vals):.1f}")
        self._line()

    def _pipeline_funnel(self) -> None:
        self._h1("2. SIGNAL PIPELINE FUNNEL")
        self._out(f"  Shows exactly where signals were lost at each stage.\n")

        candles     = self.p.candle_count or len(self.p.regime_events) or 1
        suppressed  = len(self.p.suppressed)
        reached_strat = candles - suppressed
        no_votes    = len(self.p.no_vote_events)
        insuf       = len(self.p.insufficient_vote_events)
        cooldown    = len(self.p.cooldown_blocks)
        raw_sigs    = len(self.p.raw_signals)
        ml_rejected = len(self.p.ml_rejected)
        ml_approved = len(self.p.ml_approved)
        ml_bypass   = len(self.p.ml_bypass)
        trades      = len(self.p.trades)

        def bar(n, total, width=30):
            filled = int(n / max(total, 1) * width)
            return "█" * filled + "░" * (width - filled)

        self._out(f"  {'Stage':<35} {'Count':>6}  {'%':>6}  Chart")
        self._out(f"  {'─'*35} {'─'*6}  {'─'*6}  {'─'*30}")

        stages = [
            (candles,       candles,  "Total candles in signal window", G),
            (suppressed,    candles,  "  ↳ Blocked: setup gate / regime", R),
            (reached_strat, candles,  "  ↳ Reached strategy evaluation", Y),
            (no_votes,      reached_strat, "    ↳ Zero strategy votes", R),
            (insuf,         reached_strat, "    ↳ Votes < MIN_VOTES (2)", Y),
            (cooldown,      reached_strat, "    ↳ Blocked by cooldown", Y),
            (raw_sigs,      candles,  "  ↳ RAW_SIGNAL fired (voting passed)", G),
            (ml_rejected,   raw_sigs, "    ↳ ML rejected", R),
            (ml_approved + ml_bypass, raw_sigs, "    ↳ ML approved / bypass", G),
            (trades,        raw_sigs, "  ↳ TRADES TAKEN", G if trades > 0 else R),
        ]

        for count, total_n, label, col in stages:
            if count is None:
                count = 0
            pct  = count / max(total_n, 1) * 100
            b    = bar(count, total_n)
            self._out(f"  {col}{label:<35}{RST} {count:>6}  {pct:5.1f}%  {DIM}{b}{RST}")

        self._line()

    def _strategy_health(self) -> None:
        strategy_names = _registered_strategy_names()
        min_candles = _registered_strategy_min_candles()
        self._h1(f"3. STRATEGY HEALTH — ALL {len(strategy_names)} STRATEGIES")
        self._out("  Shows which strategies voted today, how many times, and their hit rate.\n")

        # Count votes per strategy
        strat_votes:   dict[str, int]  = defaultdict(int)
        strat_calls:   dict[str, int]  = defaultdict(int)
        strat_puts:    dict[str, int]  = defaultdict(int)
        strat_in_sigs: dict[str, int]  = defaultdict(int)

        for ev in self.p.vote_events:
            for s in ev["call_votes"]:
                strat_votes[s] += 1
                strat_calls[s] += 1
            for s in ev["put_votes"]:
                strat_votes[s] += 1
                strat_puts[s]  += 1

        for sig in self.p.raw_signals:
            for s in sig.get("strategies", []):
                strat_in_sigs[s] += 1

        self._out(f"  {'Strategy':<18} {'Status':<10} {'Votes':>6} {'Calls':>6} {'Puts':>6} {'In Signals':>11}  Min Candles")
        self._out(f"  {'─'*18} {'─'*10} {'─'*6} {'─'*6} {'─'*6} {'─'*11}  {'─'*12}")

        silent_count   = 0
        active_count   = 0
        expected_silent = []

        for s in strategy_names:
            votes     = strat_votes.get(s, 0)
            calls     = strat_calls.get(s, 0)
            puts      = strat_puts.get(s, 0)
            in_sigs   = strat_in_sigs.get(s, 0)
            min_c     = min_candles.get(s, "?")

            # Expected silent reasons
            reason = ""
            if s == "OIAnalysis" and votes == 0:
                reason = "(live broker only)"
                col    = DIM
                status = "LIVE ONLY"
            elif s == "GapDirection":
                reason = "(09:15-10:00 only)"
                col    = DIM
                status = "TIME GATED"
            elif min_c and isinstance(min_c, int) and min_c >= 50:
                reason = f"(needs {min_c} candles)"
                col    = Y if votes == 0 else G
                status = "WARMING UP" if votes == 0 else f"ACTIVE {votes}v"
            else:
                col    = G if votes > 0 else R
                status = f"ACTIVE {votes}v" if votes > 0 else "SILENT ❌"

            if votes == 0 and s not in ("OIAnalysis",):
                silent_count += 1
            elif votes > 0:
                active_count += 1

            self._out(
                f"  {col}{s:<18}{RST} {status:<10} {votes:>6} {calls:>6} {puts:>6} {in_sigs:>11}  {min_c} {DIM}{reason}{RST}"
            )

        self._line()
        total_strategies = len(strategy_names)
        self._kv("Active strategies", f"{active_count}/{total_strategies}")
        self._kv("Silent strategies", f"{silent_count}/{total_strategies}  ← investigate if > 6")
        self._line()

    def _vote_timeline(self) -> None:
        self._h1("4. VOTE TIMELINE — EVERY VOTING EVENT TODAY")
        self._out("  Every candle where at least one strategy voted.\n")

        if not self.p.vote_events:
            self._warn("  No vote events today. All strategies were silent.")
            if self.p.no_vote_events:
                self._out(f"  {len(self.p.no_vote_events)} candles had zero votes.")
            self._line()
            return

        self._out(f"  {'Time':>6}  {'Dir':>10}  {'Votes':>6}  Strategies")
        self._out(f"  {'─'*6}  {'─'*10}  {'─'*6}  {'─'*50}")

        for ev in sorted(self.p.vote_events, key=lambda e: e["time"]):
            calls = ev["call_votes"]
            puts  = ev["put_votes"]
            total = len(calls) + len(puts)

            if len(calls) > len(puts):
                direction = "BUY_CALL"
                col = G
                strats = calls
            elif len(puts) > len(calls):
                direction = "BUY_PUT"
                col = R
                strats = puts
            else:
                direction = "SPLIT"
                col = Y
                strats = calls + puts

            strat_str = ", ".join(strats[:5])
            if len(strats) > 5:
                strat_str += f" +{len(strats)-5}"
            self._out(f"  {ev['time']:>6}  {col}{direction:<10}{RST}  {total:>6}  {strat_str}")

        self._line()
        self._kv("Total candles with votes", str(len(self.p.vote_events)))
        self._kv("Total candles zero votes", str(len(self.p.no_vote_events)))
        if self.p.insufficient_vote_events:
            self._out(f"\n  {Y}Insufficient votes (blocked by MIN_VOTES=2):{RST}")
            for ev in self.p.insufficient_vote_events[:10]:
                strats = ev["call_votes"] + ev["put_votes"]
                self._out(f"    {ev['time']}  votes={len(strats)}  [{', '.join(strats)}]")
        self._line()

    def _signal_deep_dive(self) -> None:
        self._h1("5. SIGNAL DEEP DIVE — EVERY RAW_SIGNAL TODAY")
        self._out("  Every signal that cleared the voting threshold, with full ML decision.\n")

        if not self.p.raw_signals:
            self._warn("  No signals cleared voting today.")
            self._line()
            return

        for i, sig in enumerate(self.p.raw_signals, 1):
            direction = sig["direction"]
            col = G if direction == "BUY_CALL" else R
            self._out(
                f"  {B}Signal #{i}{RST}  {col}{direction}{RST}  "
                f"{sig['time']}  conf={sig.get('confidence') or '—'}  "
                f"votes={sig.get('votes') or '—'}  quality={sig.get('quality') or '—'}"
            )
            strats = sig.get("strategies", [])
            if strats:
                self._out(f"    Voted by: {', '.join(strats)}")

        self._line()

        # ML decisions
        self._out(f"  {'ML Outcome':<20} Count")
        self._out(f"  {'─'*20} {'─'*5}")
        self._out(f"  {G}Approved{RST}               {len(self.p.ml_approved):>5}")
        self._out(f"  {DIM}Bypass (untrained){RST}    {len(self.p.ml_bypass):>5}")
        self._out(f"  {R}Rejected{RST}               {len(self.p.ml_rejected):>5}")

        if self.p.ml_rejected:
            # Rejection tag frequency
            tag_count: dict[str, int] = defaultdict(int)
            for rej in self.p.ml_rejected:
                for tag in (rej.get("tags") or "").split(","):
                    t = tag.strip()
                    if t:
                        tag_count[t] += 1
            self._out(f"\n  {R}ML rejection reasons:{RST}")
            for tag, cnt in sorted(tag_count.items(), key=lambda x: -x[1])[:8]:
                self._out(f"    {tag:<35} × {cnt}")

        self._line()

    def _trade_summary(self) -> None:
        self._h1("6. TRADE SUMMARY")

        if not self.p.trades:
            self._warn("  No trades taken today.")
            self._line()
            return

        wins   = [t for t in self.p.trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in self.p.trades if (t.get("pnl") or 0) <= 0]
        total_pnl = sum(t.get("pnl") or 0 for t in self.p.trades)

        self._kv("Total trades", str(len(self.p.trades)))
        self._kv("Wins / Losses", f"{len(wins)} / {len(losses)}")
        self._kv("Win rate", f"{len(wins)/max(len(self.p.trades),1)*100:.0f}%")
        self._kv("Total PnL", f"{total_pnl:+.2f}%")
        self._line()

        self._out(f"  {'Time':>6}  {'PnL':>8}  {'Exit Reason':<18}  {'Held (min)'}")
        self._out(f"  {'─'*6}  {'─'*8}  {'─'*18}  {'─'*10}")
        for t in self.p.trades:
            pnl   = t.get("pnl") or 0
            col   = G if pnl > 0 else R
            held  = t.get("held_min") or "—"
            exit_ = t.get("exit_reason") or "—"
            self._out(f"  {t['time']:>6}  {col}{pnl:>+7.2f}%{RST}  {exit_:<18}  {held}")

        # Exit reason analysis
        exit_count: dict[str, int] = defaultdict(int)
        for t in self.p.trades:
            exit_count[t.get("exit_reason") or "UNKNOWN"] += 1
        self._line()
        self._out("  Exit reason breakdown:")
        for reason, cnt in sorted(exit_count.items(), key=lambda x: -x[1]):
            col = G if "TARGET" in reason else R if "SL" in reason or "DECAY" in reason else Y
            self._out(f"    {col}{reason:<20}{RST} × {cnt}")

        self._line()

    def _root_cause(self) -> None:
        self._h1("7. ROOT CAUSE ANALYSIS — WHY DID THIS DAY LOOK THIS WAY?")

        causes   = []
        warnings = []
        ok       = []

        candles     = max(self.p.candle_count, 1)
        raw_sigs    = len(self.p.raw_signals)
        trades      = len(self.p.trades)
        suppressed  = len(self.p.suppressed)
        ml_rejected = len(self.p.ml_rejected)
        no_votes    = len(self.p.no_vote_events)
        insuf       = len(self.p.insufficient_vote_events)

        # Regime suppression
        regime_block_pct = suppressed / candles * 100
        if regime_block_pct > 60:
            causes.append(
                f"REGIME GATE blocked {regime_block_pct:.0f}% of candles. "
                f"Market was mostly RANGING/CHOPPY today. "
                f"Strategies never even ran on those candles."
            )
        elif regime_block_pct > 40:
            warnings.append(
                f"Regime gate blocked {regime_block_pct:.0f}% of candles. "
                f"Borderline choppy day."
            )
        else:
            ok.append(f"Regime gate only blocked {regime_block_pct:.0f}% — market was tradeable.")

        # No votes
        if no_votes > 100:
            causes.append(
                f"{no_votes} candles had ZERO strategy votes despite passing regime gate. "
                f"Strategies are too strict or market structure doesn't match their logic."
            )

        # Insufficient votes (split signals)
        if insuf > 20:
            warnings.append(
                f"{insuf} candles had votes but below MIN_VOTES=2 threshold. "
                f"Strategies are voting in opposite directions — market is indecisive."
            )

        # ML rejection rate
        if raw_sigs > 0:
            ml_rej_pct = ml_rejected / raw_sigs * 100
            if ml_rej_pct > 80:
                causes.append(
                    f"ML filter rejected {ml_rej_pct:.0f}% of signals ({ml_rejected}/{raw_sigs}). "
                    f"ML threshold may be too aggressive. Check ML_RANK_TIER settings."
                )
            elif ml_rej_pct > 50:
                warnings.append(f"ML filter rejected {ml_rej_pct:.0f}% of signals.")
            else:
                ok.append(f"ML rejection rate {ml_rej_pct:.0f}% — normal range.")

        # No signals at all
        if raw_sigs == 0 and candles > 50:
            causes.append(
                "NO signals reached voting all day. This means ALL candles either: "
                "(a) were suppressed by regime gate, or (b) strategies returned zero votes, "
                "or (c) votes were split. System ran but found nothing worth trading."
            )

        # No trades despite signals
        if raw_sigs > 0 and trades == 0:
            if ml_rejected == raw_sigs:
                causes.append(
                    f"All {raw_sigs} signals were REJECTED by ML filter. "
                    f"No trades taken. Reduce ML_RANK_TIER_LOW in settings.py."
                )
            else:
                warnings.append(
                    f"{raw_sigs} signals fired but {trades} trades taken. "
                    f"Check planner/execution logs for RR or margin failures."
                )

        # Good day
        if trades >= 3 and not causes:
            ok.append(f"Active trading day: {trades} trades taken.")

        # Print
        if causes:
            self._out(f"  {R}ROOT CAUSES (action required):{RST}")
            for i, c in enumerate(causes, 1):
                self._out(f"  {R}{i}.{RST} {c}")
            self._line()

        if warnings:
            self._out(f"  {Y}WARNINGS (monitor):{RST}")
            for w in warnings:
                self._out(f"  {Y}▸{RST} {w}")
            self._line()

        if ok:
            self._out(f"  {G}HEALTHY:{RST}")
            for o in ok:
                self._out(f"  {G}✓{RST} {o}")
            self._line()

    def _recommendations(self) -> None:
        self._h1("8. TOMORROW'S CHECKLIST")

        trades      = len(self.p.trades)
        raw_sigs    = len(self.p.raw_signals)
        ml_rejected = len(self.p.ml_rejected)
        suppressed  = len(self.p.suppressed)
        candles     = max(self.p.candle_count, 1)

        items = []

        if suppressed / candles > 0.5:
            items.append("Consider: is RANGING_REGIME_BLOCK=True? If yes, change to False in settings.py")
        if ml_rejected > raw_sigs * 0.7:
            items.append("Lower ML_RANK_TIER_LOW from 0.36 → 0.32 if win rate is stable")
        if trades == 0 and candles > 50:
            items.append("Run: python scripts/weekly_analysis.py to check multi-day funnel")
        if trades > 5:
            items.append("Good volume day — review all trades in journal/ for pattern")

        # Always-on checks
        items += [
            "Check Upstox/Dhan token: python scripts/upstox_auth.py (daily) or dhan_auth.py (if token error in log)",
            "Verify India VIX in logs — if VIX > 25 consider reducing position size",
            "Check if OIAnalysis strategy is enabled (requires live broker)",
        ]

        for i, item in enumerate(items, 1):
            self._out(f"  {B}{i:2d}.{RST} {item}")

        self._line()

    def _footer(self) -> None:
        self._box(
            f"Report complete — {self.date}  |  "
            f"Signals: {len(self.p.raw_signals)}  "
            f"Trades: {len(self.p.trades)}  "
            f"Candles: {self.p.candle_count}"
        )

    # ── FORMATTING HELPERS ────────────────────────────────────────────────

    def _out(self, s: str = "") -> None:
        self.lines.append(s)

    def _line(self) -> None:
        self.lines.append("")

    def _h1(self, title: str) -> None:
        self.lines.append(f"\n{W}{'─'*70}{RST}")
        self.lines.append(f"{W}  {title}{RST}")
        self.lines.append(f"{W}{'─'*70}{RST}\n")

    def _kv(self, key: str, value: str) -> None:
        self.lines.append(f"  {DIM}{key:<30}{RST} {value}")

    def _warn(self, msg: str) -> None:
        self.lines.append(f"  {Y}⚠  {msg}{RST}")

    def _box(self, msg: str) -> None:
        width = max(len(msg) + 4, 72)
        self.lines.append(f"{B}{'═'*width}{RST}")
        self.lines.append(f"{B}  {msg}{RST}")
        self.lines.append(f"{B}{'═'*width}{RST}")


# ═════════════════════════════════════════════════════════════════════════════
# JSON EXPORT
# ═════════════════════════════════════════════════════════════════════════════

def build_json_summary(parser: LogParser, target_date: str) -> dict:
    """Machine-readable summary for trend tracking over multiple days."""
    strat_votes: dict[str, int] = defaultdict(int)
    strategy_names = _registered_strategy_names()
    for ev in parser.vote_events:
        for s in ev["call_votes"] + ev["put_votes"]:
            strat_votes[s] += 1

    return {
        "date":             target_date,
        "generated_at":     datetime.now(IST).isoformat(),
        "candles":          parser.candle_count,
        "suppressed":       len(parser.suppressed),
        "vote_events":      len(parser.vote_events),
        "no_vote_events":   len(parser.no_vote_events),
        "insufficient_vote_events": len(parser.insufficient_vote_events),
        "raw_signals":      len(parser.raw_signals),
        "ml_approved":      len(parser.ml_approved),
        "ml_bypass":        len(parser.ml_bypass),
        "ml_rejected":      len(parser.ml_rejected),
        "trades":           len(parser.trades),
        "total_pnl":        round(sum(t.get("pnl") or 0 for t in parser.trades), 4),
        "win_rate":         round(
            len([t for t in parser.trades if (t.get("pnl") or 0) > 0]) /
            max(len(parser.trades), 1), 4
        ),
        "strategy_votes":   dict(strat_votes),
        "strategy_health":  {
            s: {
                "votes":      strat_votes.get(s, 0),
                "status":     "active" if strat_votes.get(s, 0) > 0 else "silent",
            }
            for s in strategy_names
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def find_log_for_date(target_date: str, log_dir: Path) -> Path | None:
    """Auto-locate log file for a given date."""
    date_compact = target_date.replace("-", "")
    candidates = [
        log_dir / f"mcxforge_{target_date}.log",
        log_dir / f"mcxforge_{date_compact}.log",
        log_dir / f"signalforge_{target_date}.log",
        log_dir / f"signalforge_{date_compact}.log",
        log_dir / f"backtest_{target_date}.log",
        log_dir / f"backtest_{date_compact}.log",
    ]
    # Also try glob
    for p in log_dir.glob(f"*{target_date}*.log"):
        candidates.append(p)
    for p in log_dir.glob(f"*{date_compact}*.log"):
        candidates.append(p)
    for p in candidates:
        if p.exists():
            return p
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCXForge EOD Diagnostic Report"
    )
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="Date to analyse (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--log", "-l",
        default=None,
        help="Explicit log file path. If not given, auto-discovered from logs/.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output (for file redirect).",
    )
    args = parser.parse_args()

    target_date = args.date or datetime.now(IST).date().isoformat()
    base_dir    = Path(__file__).parent.parent
    log_dir     = base_dir / "logs"
    journal_dir = base_dir / "journal"
    journal_dir.mkdir(exist_ok=True)

    # Locate log
    if args.log:
        log_path = Path(args.log)
    else:
        log_path = find_log_for_date(target_date, log_dir)
        if log_path is None:
            # Fall back: most recently modified .log file
            logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            log_path = logs[0] if logs else log_dir / "mcxforge.log"

    print(f"\nAnalysing: {log_path}")
    print(f"Date:      {target_date}\n")

    # Parse
    log_parser = LogParser(log_path, target_date)

    # Generate report
    report_gen = EODReport(log_parser, target_date)
    report_text = report_gen.generate()

    # Print to console
    console_text = strip_ansi(report_text) if args.no_color else report_text
    print(console_text)

    # Save plain-text file (no ANSI)
    txt_path = log_dir / f"eod_report_{target_date}.txt"
    with open(txt_path, "w") as f:
        f.write(strip_ansi(report_text))
    print(f"\n📄 Report saved: {txt_path}")

    # Save JSON summary
    json_data = build_json_summary(log_parser, target_date)
    json_path = journal_dir / f"eod_{target_date}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"📊 JSON saved:   {json_path}")

    # ── Automatically accumulate today's completed session for all 4 commodities into local CSV & SQLite
    try:
        from scripts.sync_commodity_historical_data import sync_all_commodities_historical
        print("\n📥 Incrementally accumulating latest completed candles for all 4 commodities (SILVERM, GOLDM, CRUDEOIL, NATGAS)...")
        sync_all_commodities_historical(allow_market_hours=True)
    except Exception as e:
        print(f"⚠️ EOD multi-commodity candle sync warning: {e}")


if __name__ == "__main__":
    main()

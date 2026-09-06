"""
utils/performance_analytics.py — Monte Carlo, Sharpe, Sortino, Calmar
======================================================================
Industry-standard performance metrics missing from SignalForge.

1. MONTE CARLO SIMULATION
   Run 10,000 random orderings of trade history.
   Computes 95th percentile max drawdown — worst realistic scenario.
   AlgoTest, QuantConnect, and Interactive Brokers all have this.
   Essential for position sizing: "how bad can it really get?"

2. SHARPE RATIO
   return / volatility of returns.
   Industry benchmark: > 1.0 = acceptable, > 2.0 = excellent.

3. SORTINO RATIO
   return / downside volatility only.
   More relevant for option buyers (asymmetric returns).
   Better than Sharpe for strategies with positive skew.

4. CALMAR RATIO
   annualised return / max drawdown.
   > 1.0 = return exceeds worst drawdown = sustainable.
   < 0.5 = strategy loses more in bad periods than it earns.

5. PARAMETER SENSITIVITY HEATMAP
   Grid search SL_PCT × TARGET_PCT combinations.
   Shows which parameter zone is stable vs overfitted.
   Prevents tuning to a specific value that works by chance.

Usage:
    from utils.performance_analytics import PerformanceAnalytics
    pa = PerformanceAnalytics()
    report = pa.full_report(trades)
    pa.print_report(report)
    mc = pa.monte_carlo(trades, simulations=10000)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    simulations:      int
    median_final_pnl: float
    pct_5:            float    # 5th percentile (bad case)
    pct_25:           float    # 25th percentile
    pct_75:           float    # 75th percentile
    pct_95:           float    # 95th percentile (good case)
    max_dd_median:    float    # median max drawdown
    max_dd_95pct:     float    # 95th percentile max drawdown (worst realistic)
    prob_profitable:  float    # % of simulations ending profitable
    ruin_probability: float    # % ending below -30% (ruin threshold)
    note:             str


@dataclass
class PerformanceReport:
    trades:           int
    win_rate:         float
    avg_win:          float
    avg_loss:         float
    profit_factor:    float
    total_pnl:        float
    max_drawdown:     float
    sharpe:           float
    sortino:          float
    calmar:           float
    expectancy:       float    # per trade
    avg_holding_min:  float
    best_trade:       float
    worst_trade:      float
    recovery_factor:  float    # total_pnl / max_drawdown
    grade:            str      # A/B/C/D based on combined metrics


# ═════════════════════════════════════════════════════════════════════════════
# MONTE CARLO
# ═════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(
    pnl_list:     list[float],
    simulations:  int   = 10_000,
    ruin_thresh:  float = -30.0,
    seed:         int   = 42,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation by randomly reordering trade history.

    Each simulation reshuffles the order of actual trade P&Ls
    (bootstrap resampling). Computes equity curve and max drawdown
    for each simulation. Returns distribution of outcomes.

    Args:
        pnl_list:    list of trade P&L percentages
        simulations: number of random orderings to run
        ruin_thresh: % loss considered "ruin" (default -30%)
    """
    if len(pnl_list) < 5:
        return MonteCarloResult(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            "Insufficient trades for Monte Carlo (need >= 5)"
        )

    random.seed(seed)
    n = len(pnl_list)

    final_pnls  = []
    max_dds     = []
    profitable  = 0
    ruined      = 0

    for _ in range(simulations):
        # Random resample with replacement (bootstrap)
        sample     = random.choices(pnl_list, k=n)
        equity     = 0.0
        peak       = 0.0
        max_dd     = 0.0

        for pnl in sample:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

        final_pnls.append(equity)
        max_dds.append(max_dd)
        if equity > 0:
            profitable += 1
        if equity < ruin_thresh:
            ruined += 1

    final_pnls.sort()
    max_dds.sort()

    return MonteCarloResult(
        simulations      = simulations,
        median_final_pnl = round(final_pnls[simulations // 2], 2),
        pct_5            = round(final_pnls[int(simulations * 0.05)], 2),
        pct_25           = round(final_pnls[int(simulations * 0.25)], 2),
        pct_75           = round(final_pnls[int(simulations * 0.75)], 2),
        pct_95           = round(final_pnls[int(simulations * 0.95)], 2),
        max_dd_median    = round(max_dds[simulations // 2], 2),
        max_dd_95pct     = round(max_dds[int(simulations * 0.95)], 2),
        prob_profitable  = round(profitable / simulations * 100, 1),
        ruin_probability = round(ruined / simulations * 100, 2),
        note             = f"{simulations:,} simulations on {n} trades",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE RATIOS
# ═════════════════════════════════════════════════════════════════════════════

def compute_sharpe(
    pnl_list:      list[float],
    risk_free_rate: float = 6.5,    # India 10yr bond yield %
    trades_per_year: int  = 240,    # estimated annual trades
) -> float:
    """
    Annualised Sharpe Ratio = (mean_return - risk_free) / std_return × sqrt(N)
    > 2.0 = excellent, > 1.0 = acceptable, < 0 = worse than risk-free
    """
    if len(pnl_list) < 3:
        return 0.0
    mean   = sum(pnl_list) / len(pnl_list)
    rf_per = risk_free_rate / trades_per_year
    std    = _std(pnl_list)
    if std == 0:
        return 0.0
    return round((mean - rf_per) / std * math.sqrt(trades_per_year), 3)


def compute_sortino(
    pnl_list:       list[float],
    risk_free_rate:  float = 6.5,
    trades_per_year: int   = 240,
) -> float:
    """
    Sortino uses only DOWNSIDE deviation (losses), not total std.
    Better metric for option buyers with asymmetric returns.
    """
    if len(pnl_list) < 3:
        return 0.0
    mean    = sum(pnl_list) / len(pnl_list)
    rf_per  = risk_free_rate / trades_per_year
    losses  = [min(p - rf_per, 0) for p in pnl_list]
    downdev = math.sqrt(sum(l**2 for l in losses) / max(len(losses), 1))
    if downdev == 0:
        return 0.0
    return round((mean - rf_per) / downdev * math.sqrt(trades_per_year), 3)


def compute_calmar(
    pnl_list:       list[float],
    trades_per_year: int = 240,
) -> float:
    """
    Calmar = annualised return / max drawdown.
    > 1.0 = sustainable, < 0.5 = draws more than it earns in bad periods.
    """
    if not pnl_list:
        return 0.0
    annual_ret = sum(pnl_list) / len(pnl_list) * trades_per_year
    max_dd     = _max_drawdown(pnl_list)
    if max_dd == 0:
        return 0.0
    return round(annual_ret / max_dd, 3)


def compute_profit_factor(pnl_list: list[float]) -> float:
    """Gross profit / gross loss. > 1.5 = good, > 2.0 = excellent."""
    wins   = sum(p for p in pnl_list if p > 0)
    losses = abs(sum(p for p in pnl_list if p < 0))
    return round(wins / max(losses, 0.01), 3)


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETER SENSITIVITY
# ═════════════════════════════════════════════════════════════════════════════

def parameter_sensitivity(
    trades:     list[dict],
    sl_range:   list[float]  = None,
    tgt_range:  list[float]  = None,
) -> dict:
    """
    Grid search SL_PCT × TARGET_PCT to find stable parameter zones.
    Uses actual trade entry/exit data to simulate different SL/target combos.

    Returns heatmap data: {(sl, tgt): {pnl, win_rate, sharpe}}
    """
    sl_vals  = sl_range  or [15, 20, 25, 30, 35]
    tgt_vals = tgt_range or [30, 40, 50, 60, 70, 80]

    results = {}
    for sl in sl_vals:
        for tgt in tgt_vals:
            rr   = tgt / sl
            sim_pnls = []
            for t in trades:
                peak = float(t.get("peak_pnl_pct", t.get("peak_pnl", 0)) or 0)
                # Simulate: would this trade have hit SL or target first?
                if peak >= tgt:
                    sim_pnls.append(tgt)       # target hit
                elif float(t.get("gross_pnl_pct", 0) or 0) < -sl:
                    sim_pnls.append(-sl)        # SL hit
                else:
                    sim_pnls.append(float(t.get("gross_pnl_pct", 0) or 0))

            if sim_pnls:
                wins = [p for p in sim_pnls if p > 0]
                results[(sl, tgt)] = {
                    "sl":         sl,
                    "target":     tgt,
                    "rr":         round(rr, 2),
                    "total_pnl":  round(sum(sim_pnls), 2),
                    "win_rate":   round(len(wins) / len(sim_pnls) * 100, 1),
                    "sharpe":     compute_sharpe(sim_pnls),
                    "pf":         compute_profit_factor(sim_pnls),
                }

    # Find optimal combo
    best = max(results.items(), key=lambda x: x[1]["sharpe"]) if results else None

    return {
        "grid":         {f"SL{k[0]}_TGT{k[1]}": v for k, v in results.items()},
        "best_combo":   {"sl": best[0][0], "target": best[0][1], **best[1]} if best else {},
        "current_combo": results.get((25, 70), {}),   # current SignalForge settings
    }


# ═════════════════════════════════════════════════════════════════════════════
# FULL REPORT
# ═════════════════════════════════════════════════════════════════════════════

class PerformanceAnalytics:

    def full_report(self, trades: list[dict]) -> PerformanceReport:
        """Generate complete performance report from trade list."""
        if not trades:
            return PerformanceReport(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"N/A")

        pnls    = [float(t.get("gross_pnl_pct", t.get("pnl_pct", 0)) or 0) for t in trades]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p <= 0]
        held    = [float(t.get("holding_minutes", 0) or 0) for t in trades]

        total   = sum(pnls)
        max_dd  = _max_drawdown(pnls)
        sharpe  = compute_sharpe(pnls)
        sortino = compute_sortino(pnls)
        calmar  = compute_calmar(pnls)
        pf      = compute_profit_factor(pnls)
        exp     = total / len(pnls)

        # Grade
        if sharpe > 2.0 and pf > 2.0 and calmar > 1.5:
            grade = "A"
        elif sharpe > 1.5 and pf > 1.5:
            grade = "B"
        elif sharpe > 1.0 and pf > 1.2:
            grade = "C"
        else:
            grade = "D"

        return PerformanceReport(
            trades          = len(trades),
            win_rate        = round(len(wins)/len(pnls)*100, 1),
            avg_win         = round(sum(wins)/max(len(wins),1), 2),
            avg_loss        = round(sum(losses)/max(len(losses),1), 2),
            profit_factor   = pf,
            total_pnl       = round(total, 2),
            max_drawdown    = round(max_dd, 2),
            sharpe          = sharpe,
            sortino         = sortino,
            calmar          = calmar,
            expectancy      = round(exp, 2),
            avg_holding_min = round(sum(held)/max(len(held),1), 1),
            best_trade      = round(max(pnls), 2),
            worst_trade     = round(min(pnls), 2),
            recovery_factor = round(total/max(max_dd, 0.01), 2),
            grade           = grade,
        )

    def monte_carlo(self, trades: list[dict], simulations: int = 10_000) -> MonteCarloResult:
        pnls = [float(t.get("gross_pnl_pct", t.get("pnl_pct", 0)) or 0) for t in trades]
        return run_monte_carlo(pnls, simulations)

    def sensitivity(self, trades: list[dict]) -> dict:
        return parameter_sensitivity(trades)

    def print_report(self, r: PerformanceReport) -> None:
        G   = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
        W   = "\033[97m"; D = "\033[2m";  RST = "\033[0m"
        grade_col = {
            "A": G, "B": "\033[96m", "C": Y, "D": R
        }.get(r.grade, W)

        print(f"\n{W}{'═'*55}{RST}")
        print(f"{W}  Performance Analytics Report  Grade: {grade_col}{r.grade}{RST}")
        print(f"{W}{'═'*55}{RST}\n")

        rows = [
            ("Trades",          str(r.trades)),
            ("Win Rate",        f"{r.win_rate:.1f}%",),
            ("Avg Win",         f"+{r.avg_win:.2f}%"),
            ("Avg Loss",        f"{r.avg_loss:.2f}%"),
            ("Profit Factor",   f"{r.profit_factor:.2f}"),
            ("Total PnL",       f"{r.total_pnl:+.2f}%"),
            ("Max Drawdown",    f"-{r.max_drawdown:.2f}%"),
            ("Sharpe Ratio",    f"{r.sharpe:.3f}  (>1.0=ok, >2.0=excellent)"),
            ("Sortino Ratio",   f"{r.sortino:.3f}  (>2.0 = strong for options)"),
            ("Calmar Ratio",    f"{r.calmar:.3f}  (>1.0 = sustainable)"),
            ("Expectancy",      f"{r.expectancy:+.2f}% per trade"),
            ("Avg Hold",        f"{r.avg_holding_min:.0f} min"),
            ("Recovery Factor", f"{r.recovery_factor:.2f}"),
        ]

        for label, value in rows:
            col = G if any(x in value for x in ['+', 'excellent', 'strong', 'sustainable']) else \
                  R if '-' in value and 'Drawdown' in label else W
            print(f"  {D}{label:<20}{RST} {value}")

        print()

    def print_monte_carlo(self, mc: MonteCarloResult) -> None:
        G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; W = "\033[97m"; RST = "\033[0m"
        print(f"\n{W}  Monte Carlo ({mc.simulations:,} simulations){RST}")
        print(f"  {'─'*45}")
        print(f"  Median outcome:        {mc.median_final_pnl:+.1f}%")
        print(f"  Best 25% scenarios:    {mc.pct_75:+.1f}% → {mc.pct_95:+.1f}%")
        print(f"  Worst 25% scenarios:   {mc.pct_5:+.1f}% → {mc.pct_25:+.1f}%")
        print(f"  Max DD (median):       -{mc.max_dd_median:.1f}%")
        dd_col = R if mc.max_dd_95pct > 20 else Y if mc.max_dd_95pct > 10 else G
        print(f"  Max DD (95th pct):     {dd_col}-{mc.max_dd_95pct:.1f}%{RST}  ← worst realistic case")
        prob_col = G if mc.prob_profitable > 65 else Y if mc.prob_profitable > 50 else R
        print(f"  Probability profitable:{prob_col} {mc.prob_profitable:.1f}%{RST}")
        ruin_col = R if mc.ruin_probability > 5 else Y if mc.ruin_probability > 1 else G
        print(f"  Ruin probability:      {ruin_col}{mc.ruin_probability:.2f}%{RST}")
        print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var  = sum((x - mean)**2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def _max_drawdown(pnl_list: list[float]) -> float:
    """Maximum drawdown from equity curve."""
    equity  = 0.0
    peak    = 0.0
    max_dd  = 0.0
    for pnl in pnl_list:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd

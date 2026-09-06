"""
core/analytics/performance.py — Four-Month Validation & Edge Attribution Analytics

Analyzes 4-month observation telemetry to answer:
1. Which strategy actually has a positive statistical edge?
2. Under which market regime does each strategy work?
3. At what session time (Morning 09:00–17:00 vs Evening 17:00–23:30)?
4. What is the realized drawdown and consecutive loss profile?
5. How much does execution slippage and transaction cost degrade the theoretical edge?
6. Long vs Short edge breakdown.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class EdgeReport:
    total_signals: int
    executed_trades: int
    rejected_signals: int
    win_rate_pct: float
    profit_factor: float
    expectancy_points: float
    expectancy_inr: float
    avg_win_points: float
    avg_loss_points: float
    max_drawdown_inr: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_mfe_points: float
    avg_mae_points: float
    mfe_mae_ratio: float
    avg_slippage_points: float
    total_slippage_inr: float
    total_costs_inr: float
    net_pnl_inr: float
    by_strategy: Dict[str, dict]
    by_regime: Dict[str, dict]
    by_session_phase: Dict[str, dict]
    by_direction: Dict[str, dict]


class PerformanceAnalyticsEngine:
    """
    Computes rigorous statistical performance metrics from the 4-month observation journal.
    """

    @staticmethod
    def generate_report(df_journal: pd.DataFrame) -> EdgeReport:
        """Analyzes signal journal DataFrame."""
        if df_journal is None or df_journal.empty:
            return EdgeReport(
                total_signals=0, executed_trades=0, rejected_signals=0,
                win_rate_pct=0.0, profit_factor=0.0, expectancy_points=0.0,
                expectancy_inr=0.0, avg_win_points=0.0, avg_loss_points=0.0,
                max_drawdown_inr=0.0, max_consecutive_wins=0, max_consecutive_losses=0,
                avg_mfe_points=0.0, avg_mae_points=0.0, mfe_mae_ratio=0.0,
                avg_slippage_points=0.0, total_slippage_inr=0.0, total_costs_inr=0.0,
                net_pnl_inr=0.0, by_strategy={}, by_regime={}, by_session_phase={}, by_direction={},
            )

        df = df_journal.copy()
        total_signals = len(df)

        # Separate executed vs rejected
        executed = df[df["rejection_reason"].fillna("").str.strip() == ""]
        rejected = df[df["rejection_reason"].fillna("").str.strip() != ""]

        exec_count = len(executed)
        rej_count = len(rejected)

        if exec_count == 0:
            return EdgeReport(
                total_signals=total_signals, executed_trades=0, rejected_signals=rej_count,
                win_rate_pct=0.0, profit_factor=0.0, expectancy_points=0.0,
                expectancy_inr=0.0, avg_win_points=0.0, avg_loss_points=0.0,
                max_drawdown_inr=0.0, max_consecutive_wins=0, max_consecutive_losses=0,
                avg_mfe_points=0.0, avg_mae_points=0.0, mfe_mae_ratio=0.0,
                avg_slippage_points=0.0, total_slippage_inr=0.0, total_costs_inr=0.0,
                net_pnl_inr=0.0, by_strategy={}, by_regime={}, by_session_phase={}, by_direction={},
            )

        # Core P&L Series
        pnl_pts = executed["pnl_points"].astype(float)
        pnl_inr = executed["pnl_inr"].astype(float)
        wins = pnl_pts[pnl_pts > 0]
        losses = pnl_pts[pnl_pts < 0]

        win_rate = round((len(wins) / exec_count) * 100.0, 2)
        avg_win = round(float(wins.mean()), 2) if not wins.empty else 0.0
        avg_loss = round(abs(float(losses.mean())), 2) if not losses.empty else 0.0

        gross_profits = float(pnl_inr[pnl_inr > 0].sum())
        gross_losses = abs(float(pnl_inr[pnl_inr < 0].sum()))
        profit_factor = round(gross_profits / max(gross_losses, 1e-6), 2)

        # Expectancy: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        p_win = len(wins) / exec_count
        p_loss = len(losses) / exec_count
        expectancy_pts = round((p_win * avg_win) - (p_loss * avg_loss), 2)
        expectancy_inr = round(float(pnl_inr.mean()), 2)

        # Drawdown calculation
        cum_pnl = pnl_inr.cumsum()
        peak = cum_pnl.cummax()
        dd = peak - cum_pnl
        max_dd = round(float(dd.max()), 2) if not dd.empty else 0.0

        # Consecutive Wins and Losses
        is_win_series = (pnl_pts > 0).astype(int)
        win_streaks = (is_win_series != is_win_series.shift()).cumsum()
        max_consec_wins = int(is_win_series.groupby(win_streaks).sum().max()) if not wins.empty else 0

        is_loss_series = (pnl_pts < 0).astype(int)
        loss_streaks = (is_loss_series != is_loss_series.shift()).cumsum()
        max_consec_losses = int(is_loss_series.groupby(loss_streaks).sum().max()) if not losses.empty else 0

        # MFE / MAE
        mfe = executed.get("mfe_points", pd.Series(0.0, index=executed.index)).astype(float)
        mae = executed.get("mae_points", pd.Series(0.0, index=executed.index)).astype(float)
        avg_mfe = round(float(mfe.mean()), 2)
        avg_mae = round(float(mae.mean()), 2)
        mfe_mae_ratio = round(avg_mfe / max(avg_mae, 1.0), 2)

        # Slippage & Costs
        slip_pts = executed.get("slippage_pts", pd.Series(0.0, index=executed.index)).astype(float)
        costs = executed.get("costs", pd.Series(0.0, index=executed.index)).astype(float)
        avg_slip = round(float(slip_pts.mean()), 2)
        tot_slip_inr = round(float(slip_pts.sum()) * 1.0, 2) # for SILVERMIC lot=1
        tot_costs = round(float(costs.sum()), 2)
        net_pnl = round(float(pnl_inr.sum()) - tot_costs, 2)

        # ── Group Attribution ─────────────────────────────────────────────────
        def analyze_subgroup(group_col: str) -> Dict[str, dict]:
            res = {}
            if group_col not in executed.columns:
                return res
            for val, grp in executed.groupby(group_col):
                g_pts = grp["pnl_points"].astype(float)
                g_inr = grp["pnl_inr"].astype(float)
                g_wins = g_pts[g_pts > 0]
                res[str(val)] = {
                    "trades": len(grp),
                    "win_rate_pct": round((len(g_wins) / len(grp)) * 100.0, 1),
                    "net_pnl_inr": round(float(g_inr.sum()), 2),
                    "avg_pnl_pts": round(float(g_pts.mean()), 2),
                    "profit_factor": round(float(g_inr[g_inr > 0].sum()) / max(abs(float(g_inr[g_inr < 0].sum())), 1e-6), 2),
                }
            return res

        by_strategy = analyze_subgroup("strategies_fired")
        by_regime = analyze_subgroup("regime")
        by_direction = analyze_subgroup("direction")

        # Session time-of-day phase (Morning vs European/US overlap)
        if "time" in executed.columns:
            executed["session_phase"] = np.where(executed["time"] < "17:00", "MORNING_SESSION", "EVENING_US_OVERLAP")
            by_phase = analyze_subgroup("session_phase")
        else:
            by_phase = {}

        return EdgeReport(
            total_signals=total_signals,
            executed_trades=exec_count,
            rejected_signals=rej_count,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            expectancy_points=expectancy_pts,
            expectancy_inr=expectancy_inr,
            avg_win_points=avg_win,
            avg_loss_points=avg_loss,
            max_drawdown_inr=max_dd,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            avg_mfe_points=avg_mfe,
            avg_mae_points=avg_mae,
            mfe_mae_ratio=mfe_mae_ratio,
            avg_slippage_points=avg_slip,
            total_slippage_inr=tot_slip_inr,
            total_costs_inr=tot_costs,
            net_pnl_inr=net_pnl,
            by_strategy=by_strategy,
            by_regime=by_regime,
            by_session_phase=by_phase,
            by_direction=by_direction,
        )

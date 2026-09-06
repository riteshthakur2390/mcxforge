"""
utils/commodity_report_formatter.py — SignalForge-Grade Rich Performance Reporting for MCX
========================================================================================
Adapts SignalForge's report_formatter.py into an institutional commodity performance
review system supporting both ANSI rich terminal rendering and persistent Markdown exports:
- Executive Summary & Win Rate / Profit Factor / Sharpe / Expectancy
- Real Statutory Transaction Costs (MCX Exchange, CTT, SEBI, Stamp duty, GST, Brokerage)
- SPAN Margin & Capital Requirements
- Multi-Strategy Contribution Matrix (identifies which strategies drive profitability)
- MCX Session Breakdown (Morning vs Afternoon vs Evening)
- Detailed Trade-by-Trade Audit Journal
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from core.strategies.ensemble import EnsembleRunResult, EnsembleTradeRecord

# ANSI Colors for terminal
C_BOLD = "\033[1m"
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_MAGENTA = "\033[95m"
C_RESET = "\033[0m"


def format_commodity_rich_report(result: EnsembleRunResult, use_colors: bool = True) -> str:
    """Formats ensemble backtest result into a SignalForge ANSI rich terminal report."""
    def col(text: Any, color: str) -> str:
        return f"{color}{text}{C_RESET}" if use_colors else str(text)

    m = result.metrics
    trades = result.trades

    lines: List[str] = []
    header = f"MCXFORGE {result.symbol} — PERFORMANCE REVIEW ({result.timeframe})"
    lines.append(col("=" * len(header), C_CYAN))
    lines.append(col(header, C_BOLD + C_CYAN))
    lines.append(col("=" * len(header), C_CYAN))

    if trades:
        d_min = trades[0].entry_time.strftime("%Y-%m-%d")
        d_max = trades[-1].exit_time.strftime("%Y-%m-%d")
        lines.append(f"Period: {col(f'{d_min} to {d_max} ({result.total_days} trading days)', C_YELLOW)}")
    else:
        lines.append(f"Period: {col('No Trades Recorded', C_RED)}")

    lines.append("")

    # ── 1. SUMMARY ─────────────────────────────────────────────────────────────
    lines.append(col("── SUMMARY ──────────────────────────────────────────────────────────", C_BOLD))
    trades_per_day = (m.total_trades / result.total_days) if result.total_days > 0 else 0.0
    lines.append(f"Total trading days : {result.total_days} | Trades/day: {trades_per_day:.2f}")
    lines.append(f"Total trades       : {m.total_trades}")
    lines.append(f"Wins               : {col(m.winning_trades, C_GREEN)}")
    lines.append(f"Losses             : {col(m.losing_trades, C_RED)}")
    lines.append(f"Win rate           : {col(f'{m.win_rate_pct:.1f}%', C_BOLD)}")
    pf_col = C_GREEN if m.profit_factor >= 1.5 else (C_YELLOW if m.profit_factor >= 1.0 else C_RED)
    lines.append(f"Profit factor      : {col(f'{m.profit_factor:.2f}', C_BOLD + pf_col)}")
    lines.append(f"Sharpe ratio       : {m.sharpe_ratio:.2f} | Sortino: {m.sortino_ratio:.2f} | Calmar: {m.calmar_ratio:.2f}")
    lines.append(f"Expectancy/trade   : Rs. {m.expectancy_inr:,.2f}")
    lines.append(f"Gross P&L          : Rs. {m.gross_pnl_inr:,.2f}")
    lines.append(f"Statutory fees     : Rs. {m.total_costs_inr:,.2f} (MCX CTT, Brokerage, SEBI, GST, Stamp)")
    lines.append(f"Slippage cost      : Rs. {m.total_slippage_inr:,.2f}")

    net_col = C_GREEN if m.net_pnl_inr > 0 else C_RED
    cap_ret = (m.net_pnl_inr / max(result.initial_capital, 1.0)) * 100.0
    lines.append(f"Net Realized P&L   : {col(f'Rs. {m.net_pnl_inr:,.2f} ({cap_ret:+.2f}%)', C_BOLD + net_col)}")
    lines.append("")

    # ── 2. TRADING STATS ───────────────────────────────────────────────────────
    lines.append(col("── TRADING STATS ────────────────────────────────────────────────────", C_BOLD))
    lines.append(f"Avg winning trade  : Rs. {m.avg_win_inr:,.2f}")
    lines.append(f"Avg losing trade   : Rs. {m.avg_loss_inr:,.2f}")
    lines.append(f"Win/Loss ratio     : {m.win_loss_ratio:.2f}")
    lines.append(f"Max drawdown       : {col(f'Rs. {m.max_drawdown_inr:,.2f} ({m.max_drawdown_pct:.2f}%)', C_RED)}")
    lines.append(f"Avg MFE / MAE      : {m.avg_mfe_points:.1f} pts / {m.avg_mae_points:.1f} pts (Ratio: {m.mfe_mae_ratio:.2f})")
    lines.append(f"Avg holding time   : {m.avg_holding_time_minutes:.1f} min")


    target_exits = len([t for t in trades if "TARGET" in t.exit_reason])
    sl_exits = len([t for t in trades if "SL" in t.exit_reason])
    eod_exits = len([t for t in trades if "EOD" in t.exit_reason])
    lines.append(f"Exit breakdown     : TARGET: {target_exits} | STOP-LOSS: {sl_exits} | EOD 23:15: {eod_exits}")
    lines.append("")

    # ── 3. CAPITAL & MARGIN ────────────────────────────────────────────────────
    lines.append(col("── CAPITAL & MARGIN REQUIREMENT ─────────────────────────────────────", C_BOLD))
    lines.append(f"Initial capital    : Rs. {result.initial_capital:,.2f}")
    lines.append(f"Ending capital     : Rs. {result.ending_capital:,.2f}")
    lines.append(f"Peak margin seen   : Rs. {result.peak_margin_seen:,.2f} (SPAN + Exposure)")
    recommended_cap = result.peak_margin_seen + m.max_drawdown_inr
    lines.append(f"Recommended cap    : {col(f'Rs. {recommended_cap:,.2f}', C_YELLOW)} (Peak Margin + Max DD Buffer)")
    lines.append("")

    # ── 4. STRATEGY CONTRIBUTION MATRIX ────────────────────────────────────────
    lines.append(col("── STRATEGY CONTRIBUTION MATRIX ───────────────────────────────────────────────────────────────────", C_BOLD))
    st_header = f"{'Strategy Name':<25} | {'Trades':>6} | {'Win%':>6} | {'Gross P&L':>12} | {'Fees':>10} | {'Net P&L':>12} | {'PF':>5} | {'Contrib%':>8}"
    lines.append(col(st_header, C_BOLD))
    lines.append("-" * len(st_header))

    for s_name, stats in result.strategy_contributions.items():
        if stats["trades"] == 0:
            continue
        pnl = stats["net_pnl_inr"]
        p_col = C_GREEN if pnl > 0 else C_RED
        pnl_str = col(f"Rs.{pnl:>9,.0f}", p_col)
        row = (
            f"{s_name:<25} | {stats['trades']:>6} | {stats['win_rate_pct']:>5.1f}% | "
            f"Rs.{stats['gross_pnl_inr']:>9,.0f} | Rs.{stats['fees_inr']:>7,.0f} | "
            f"{pnl_str} | {stats['profit_factor']:>5.2f} | {stats['contribution_pct']:>7.1f}%"
        )
        lines.append(row)
    lines.append("")

    # ── 5. MCX SESSION BREAKDOWN ───────────────────────────────────────────────
    lines.append(col("── MCX SESSION BREAKDOWN ──────────────────────────────────────────────────────────", C_BOLD))
    sess_header = f"{'Session Window':<20} | {'Trades':>6} | {'Win%':>6} | {'Gross P&L':>12} | {'Fees':>10} | {'Net P&L':>12} | {'PF':>5}"
    lines.append(col(sess_header, C_BOLD))
    lines.append("-" * len(sess_header))

    for s_name, stats in result.session_breakdown.items():
        pnl = stats["net_pnl_inr"]
        p_col = C_GREEN if pnl > 0 else C_RED
        pnl_str = col(f"Rs.{pnl:>9,.0f}", p_col)
        display_name = {
            "MORNING": "Morning (09-13)",
            "AFTERNOON": "Afternoon (13-17)",
            "EVENING": "Evening (17-23:30)",
        }.get(s_name, s_name)
        row = (
            f"{display_name:<20} | {stats['trades']:>6} | {stats['win_rate_pct']:>5.1f}% | "
            f"Rs.{stats['gross_pnl_inr']:>9,.0f} | Rs.{stats['fees_inr']:>7,.0f} | "
            f"{pnl_str} | {stats['profit_factor']:>5.2f}"
        )
        lines.append(row)
    lines.append("")

    # ── 6. TRADE-WISE JOURNAL (Sample recent 15 trades) ───────────────────────
    lines.append(col("── TRADE-WISE JOURNAL (Recent Sample) ──────────────────────────────────────────────────────────────────────────────────────────", C_BOLD))
    j_header = (
        f"{'ID':<5} | {'Date':<10} | {'Entry':<8} | {'Exit':<8} | {'Dir':<4} | "
        f"{'EntryPx':>7} | {'ExitPx':>7} | {'PnL Pts':>7} | {'Net P&L':>10} | "
        f"{'Session':<9} | {'Votes':>5} | {'Strategies Fired':<30} | {'Exit Reason':<14}"
    )
    lines.append(col(j_header, C_BOLD))
    lines.append("-" * len(j_header))

    sample_trades = trades[-25:] if len(trades) > 25 else trades
    for t in sample_trades:
        p_col = C_GREEN if t.net_pnl_inr > 0 else C_RED
        pnl_str = col(f"Rs.{t.net_pnl_inr:>7,.0f}", p_col)
        strats_str = "+".join(t.strategies_fired)
        if len(strats_str) > 30:
            strats_str = strats_str[:28] + "…"

        row = (
            f"{t.trade_id:<5} | {t.entry_time.strftime('%Y-%m-%d'):<10} | "
            f"{t.entry_time.strftime('%H:%M'):<8} | {t.exit_time.strftime('%H:%M'):<8} | "
            f"{t.direction:<4} | {t.entry_price:>7.0f} | {t.exit_price:>7.0f} | "
            f"{t.pnl_points:>+7.1f} | {pnl_str} | {t.session:<9} | {t.votes:>5} | "
            f"{strats_str:<30} | {t.exit_reason[:14]:<14}"
        )
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def format_commodity_markdown_report(result: EnsembleRunResult) -> str:
    """Generates a GitHub-flavored Markdown report of the ensemble backtest result."""
    m = result.metrics
    trades = result.trades

    md: List[str] = []
    md.append(f"# MCXForge Multi-Strategy Ensemble Performance Report: {result.symbol}")
    md.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST  ")
    md.append(f"**Instrument**: {result.symbol} | **Timeframe**: `{result.timeframe}` | **Total Trading Days**: {result.total_days}  ")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 1. Executive Summary")
    md.append("")
    md.append("| Metric | Value | Metric | Value |")
    md.append("|---|---|---|---|")
    md.append(f"| **Total Trades** | {m.total_trades} | **Win Rate** | **{m.win_rate_pct:.1f}%** |")
    md.append(f"| **Winning Trades** | {m.winning_trades} | **Losing Trades** | {m.losing_trades} |")
    md.append(f"| **Gross P&L** | ₹{m.gross_pnl_inr:,.2f} | **Statutory Fees & Taxes** | ₹{m.total_costs_inr:,.2f} |")
    md.append(f"| **Net Realized P&L** | **₹{m.net_pnl_inr:,.2f}** | **Profit Factor** | **{m.profit_factor:.2f}** |")
    md.append(f"| **Expectancy per Trade** | ₹{m.expectancy_inr:,.2f} | **Max Drawdown** | ₹{m.max_drawdown_inr:,.2f} ({m.max_drawdown_pct:.2f}%) |")
    md.append(f"| **Sharpe Ratio** | {m.sharpe_ratio:.2f} | **Sortino Ratio** | {m.sortino_ratio:.2f} |")
    md.append(f"| **Avg Holding Time** | {m.avg_holding_time_minutes:.1f} min | **MFE / MAE Ratio** | {m.mfe_mae_ratio:.2f} |")
    md.append(f"| **Peak SPAN Margin** | ₹{result.peak_margin_seen:,.2f} | **Recommended Capital** | ₹{result.peak_margin_seen + m.max_drawdown_inr:,.2f} |")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 2. Multi-Strategy Contribution Matrix")
    md.append("Measures how each participating strategy contributed to overall ensemble trades and net P&L:")
    md.append("")
    md.append("| Strategy Name | Trades | Win Rate | Gross P&L (₹) | Fees (₹) | Net P&L (₹) | Profit Factor | P&L Contribution |")
    md.append("|---|---|---|---|---|---|---|---|")

    for s_name, stats in result.strategy_contributions.items():
        if stats["trades"] == 0:
            continue
        md.append(
            f"| **{s_name}** | {stats['trades']} | {stats['win_rate_pct']:.1f}% | "
            f"₹{stats['gross_pnl_inr']:,.2f} | ₹{stats['fees_inr']:,.2f} | "
            f"**₹{stats['net_pnl_inr']:,.2f}** | {stats['profit_factor']:.2f} | {stats['contribution_pct']:+.1f}% |"
        )
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 3. MCX Session Breakdown")
    md.append("Breakdown by MCX trading session window to isolate session-specific edge:")
    md.append("")
    md.append("| Session Window | Hours (IST) | Trades | Win Rate | Gross P&L (₹) | Fees (₹) | Net P&L (₹) | Profit Factor |")
    md.append("|---|---|---|---|---|---|---|---|")

    session_hours = {
        "MORNING": "09:00 – 13:00",
        "AFTERNOON": "13:00 – 17:00",
        "EVENING": "17:00 – 23:30",
    }
    for s_name, stats in result.session_breakdown.items():
        hrs = session_hours.get(s_name, "-")
        md.append(
            f"| **{s_name}** | {hrs} | {stats['trades']} | {stats['win_rate_pct']:.1f}% | "
            f"₹{stats['gross_pnl_inr']:,.2f} | ₹{stats['fees_inr']:,.2f} | "
            f"**₹{stats['net_pnl_inr']:,.2f}** | {stats['profit_factor']:.2f} |"
        )
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 4. Trade-Wise Audit Journal (Recent Excerpt)")
    md.append("")
    md.append("| ID | Date | Times | Dir | Entry | Exit | Pts | Net P&L (₹) | Session | Votes | Strategies Fired | Exit Reason |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    sample_trades = trades[-30:] if len(trades) > 30 else trades
    for t in sample_trades:
        strats = "+".join(t.strategies_fired)
        if len(strats) > 35:
            strats = strats[:33] + "…"
        md.append(
            f"| {t.trade_id} | {t.entry_time.strftime('%Y-%m-%d')} | "
            f"{t.entry_time.strftime('%H:%M')}–{t.exit_time.strftime('%H:%M')} | {t.direction} | "
            f"{t.entry_price:.1f} | {t.exit_price:.1f} | {t.pnl_points:+.1f} | "
            f"**₹{t.net_pnl_inr:+,.0f}** | {t.session} | {t.votes} | `{strats}` | {t.exit_reason} |"
        )
    md.append("")
    return "\n".join(md)

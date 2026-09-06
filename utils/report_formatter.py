"""
utils/report_formatter.py — Formats backtest and EOD results into a rich text report.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import math
from config.settings import NIFTY_LOT_SIZE, TRADING_MODE, APP_NAME, VERSION, MAX_POSITION_LOTS

# ANSI Colors for terminal
C_BOLD = "\033[1m"
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_RESET = "\033[0m"

def format_rich_report(
    journal_rows: List[Dict[str, Any]], 
    paper_capital: float = 1000000.0, 
    total_days: int = 0, 
    lots: int = MAX_POSITION_LOTS, 
    use_colors: bool = True,
    extra_summary: Optional[Dict[str, Any]] = None
) -> str:
    if not journal_rows:
        return "No trades recorded."

    def col(text, color):
        return f"{color}{text}{C_RESET}" if use_colors else str(text)

    df_all = pd.DataFrame(journal_rows).copy()
    if "entry_time" in df_all.columns:
        df_all["entry_time"] = pd.to_datetime(df_all["entry_time"], errors="coerce")
    
    # Filter for closed trades
    if "lifecycle_status" in df_all.columns:
        df = df_all[df_all["lifecycle_status"].str.upper() == "CLOSED"].copy()
    else:
        df = df_all.copy()
    if df.empty:
        return "No closed trades recorded."

    # Convert numeric columns safely
    def to_num(col_name):
        series = df.get(col_name)
        if series is None:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(series, errors="coerce").fillna(0.0)

    # Normalize column names between SignalForge and MCXForge
    if "realized_pnl" not in df.columns and "net_pnl_inr" in df.columns:
        df["realized_pnl"] = df["net_pnl_inr"]
    if "actual_premium" not in df.columns and "entry_price" in df.columns:
        df["actual_premium"] = df["entry_price"]
    if "exit_premium" not in df.columns and "exit_price" in df.columns:
        df["exit_premium"] = df["exit_price"]
    if "margin_used_inr" in df.columns:
        df["total_invested"] = df["margin_used_inr"]

    df["realized_pnl"] = to_num("realized_pnl")
    df["pnl_pct"] = to_num("pnl_pct")
    df["exit_time"] = pd.to_datetime(df.get("exit_time"), errors="coerce")
    df["actual_premium"] = to_num("actual_premium")
    df["exit_premium"] = to_num("exit_premium")
    
    # Basic Stats
    total_trades = len(df)
    wins_df = df[df["realized_pnl"] > 0]
    losses_df = df[df["realized_pnl"] <= 0]
    
    wins = len(wins_df)
    losses = len(losses_df)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    gross_profit = wins_df["realized_pnl"].sum()
    gross_loss = abs(losses_df["realized_pnl"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    
    net_pnl = df["realized_pnl"].sum()
    expectancy = (net_pnl / total_trades) if total_trades > 0 else 0.0

    # Calculate Total Invested (Capital allocated for trading)
    max_margin_recorded = float(to_num("margin_used_inr").max()) if "margin_used_inr" in df.columns else 0.0
    if max_margin_recorded > 0 or "margin_req" in df.columns:
        # Commodity futures margin trading:
        # Trade capital allocation is max 15% of paper_capital (Rs. 30,000 for 2L capital).
        # We must NOT sum sequential margin over hundreds of trades (which yields turnover in crores).
        per_trade_cap = paper_capital * 0.15 if paper_capital > 0 else 30000.0
        total_invested_value = min(max_margin_recorded, per_trade_cap) if max_margin_recorded > 0 else per_trade_cap
    elif extra_summary and "total_invested" in extra_summary:
        total_invested_value = float(extra_summary["total_invested"])
    else:
        total_invested = to_num("total_invested")
        if float(total_invested.sum()) <= 0:
            quantity = to_num("quantity")
            if float(quantity.sum()) > 0:
                total_invested = df["actual_premium"] * quantity
            else:
                lot_size = to_num("lot_size").replace(0, NIFTY_LOT_SIZE)
                row_lots = to_num("lots").replace(0, lots)
                total_invested = df["actual_premium"] * lot_size * row_lots
        total_invested_value = float(total_invested.sum())

    invested_return_pct = (
        net_pnl / total_invested_value * 100.0
        if total_invested_value > 0 else 0.0
    )
    
    avg_win = (gross_profit / wins) if wins > 0 else 0.0
    avg_loss = (gross_loss / losses) if losses > 0 else 0.0
    
    # Days Calculation
    all_trade_days = df["entry_time"].dt.date.unique()
    days_with_trade = len(all_trade_days)
    
    if total_days <= 0:
        if not df_all.empty and "entry_time" in df_all.columns:
            valid_times = df_all["entry_time"].dropna()
            if not valid_times.empty:
                total_days = (valid_times.max().date() - valid_times.min().date()).days + 1
            else:
                total_days = days_with_trade
        else:
            total_days = days_with_trade
    
    # Max Drawdown
    daily_pnl = df.groupby(df["exit_time"].dt.date)["realized_pnl"].sum().sort_index()
    equity_curve = daily_pnl.cumsum() + paper_capital
    rolling_peak = equity_curve.cummax()
    drawdown = rolling_peak - equity_curve
    max_drawdown = drawdown.max()
    max_drawdown_pct = (max_drawdown / paper_capital * 100) if paper_capital > 0 else 0.0
    
    # Exit Breakdown (Corrected)
    tp_count = 0
    sl_count = 0
    eod_count = 0
    for _, row in df.iterrows():
        reason = str(row["exit_reason"]).upper()
        pnl = float(row.get("realized_pnl", 0))
        # Positive exits
        if reason in ("TARGET", "TP", "PROFIT_PROTECT", "PROFIT_PRO"):
            tp_count += 1
        # Hard stops
        elif reason in ("STOPLOSS", "SL", "SL_HIT"):
            sl_count += 1
        # Managed exits (Trailing or Time-based) - check actual PnL
        elif reason in ("TRAILING_SL", "TRAILING_S", "TIME_DECAY"):
            if pnl > 0: tp_count += 1
            else: sl_count += 1
        # Day close
        elif reason in ("EOD", "FORCE_CLOSE"):
            eod_count += 1
        # Uncategorized
        else:
            if pnl > 0: tp_count += 1
            else: sl_count += 1
    
    # Capital Requirement
    if "margin_used_inr" in df.columns and float(df["margin_used_inr"].sum()) > 0:
        df["margin_req"] = df["margin_used_inr"]
        max_margin = float(df["margin_req"].max())
        qty = lots
    else:
        spread_width = 500
        qty = lots * NIFTY_LOT_SIZE
        span_buffer = 1.75
        
        def calc_margin(premium):
            max_loss_unit = max(0, spread_width - premium)
            return max_loss_unit * qty * span_buffer

        df["margin_req"] = df["actual_premium"].apply(calc_margin)
        max_margin = float(df["margin_req"].max())
    
    recommended_capital = max(paper_capital, max_margin + max_drawdown)
    roi_annual = (net_pnl / recommended_capital) * (365 / total_days) * 100 if recommended_capital > 0 and total_days > 0 else 0.0

    # Build Report String
    report = []
    header = f"SignalForge {VERSION} - PERFORMANCE REVIEW"
    report.append(col("=" * len(header), C_CYAN))
    report.append(col(header, C_BOLD + C_CYAN))
    report.append(col("=" * len(header), C_CYAN))
    
    report.append(f"Mode: {col(TRADING_MODE, C_YELLOW)}")
    if extra_summary and "days" in extra_summary:
        dates = extra_summary["days"]
        if len(dates) > 5:
            report.append(f"Period: {dates[0]} to {dates[-1]} ({len(dates)} days)")
        else:
            report.append(f"Days: {', '.join(dates)}")
    else:
        report.append(f"Period: {df['entry_time'].min().date()} to {df['exit_time'].max().date()}")
    
    if extra_summary and "candles" in extra_summary:
        report.append(f"Candles: {extra_summary['candles']}")
    
    report.append("")

    # Section: Summary
    report.append(col("── SUMMARY ──────────────────────────────────────────", C_BOLD))
    report.append(f"Total days       : {total_days}")
    trades_per_day = total_trades / total_days if total_days > 0 else 0
    report.append(f"Days with trade  : {days_with_trade} (no breakout: {max(0, total_days - days_with_trade)}) | Trades/day: {trades_per_day:.2f}")
    
    if extra_summary:
        sig = extra_summary.get("signals", 0)
        app = extra_summary.get("approved", 0)
        sup = extra_summary.get("suppressed", 0)
        report.append(f"Signals          : {sig} | Approved: {app} | Suppressed: {sup}")

    report.append(f"Wins             : {col(wins, C_GREEN)}")
    report.append(f"Losses           : {col(losses, C_RED)}")
    report.append(f"Win rate         : {col(f'{win_rate:.1f}%', C_BOLD)}")
    
    pf = extra_summary.get("profit_factor", profit_factor) if extra_summary else profit_factor
    report.append(f"Profit factor    : {col(f'{pf:.2f}', C_BOLD if pf > 1.5 else C_RESET)}")
    
    if extra_summary and "sharpe_ratio" in extra_summary:
        report.append(f"Sharpe Ratio     : {extra_summary['sharpe_ratio']:.3f}")

    report.append(f"Expectancy/trade : Rs. {expectancy:,.0f}")
    report.append(f"Total invested   : Rs. {total_invested_value:,.0f}")
    
    if extra_summary and "capital_return_pct" in extra_summary:
        cap_ret = extra_summary["capital_return_pct"]
        report.append(f"Capital return   : {col(f'{cap_ret:.2f}%', C_BOLD + C_CYAN)}")

    report.append(
        f"Net P&L          : {col(f'Rs. {net_pnl:,.0f} ({invested_return_pct:.2f}%)', C_BOLD + (C_GREEN if net_pnl > 0 else C_RED))}"
    )
    report.append("")

    # Section: Stats
    report.append(col("── TRADING STATS ────────────────────────────────────", C_BOLD))
    report.append(f"Avg winning trade: Rs. {avg_win:,.0f}")
    report.append(f"Avg losing trade : Rs. {avg_loss:,.0f}")
    report.append(f"Max drawdown     : {col(f'Rs. {max_drawdown:,.0f}', C_RED)}")
    report.append(f"Exit breakdown   : POSITIVE: {tp_count} | NEGATIVE: {sl_count} | EOD: {eod_count}")
    report.append("")

    # Section: Capital
    report.append(col("── CAPITAL REQUIREMENT ──────────────────────────────", C_BOLD))
    report.append(f"Lots             : {lots} ({qty} units)")
    display_max_margin = min(max_margin, paper_capital * 0.15) if paper_capital > 0 else min(max_margin, 30000.0)
    report.append(f"Max margin seen  : Rs. {display_max_margin:,.0f} (Max 15% Budget: Rs. 30,000)")
    report.append(f"Recommended Cap  : {col(f'Rs. {recommended_capital:,.0f}', C_YELLOW)} (Margin + DD Buffer)")
    report.append(f"Annual ROI       : {col(f'{roi_annual:.1f}%', C_BOLD + C_CYAN)}")
    report.append("")

    # Section: Trade-wise
    report.append(col("── TRADE-WISE BREAKDOWN ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────", C_BOLD))
    table_header = (
        f"{'Date':<10} | {'Symbol':<20} | {'Entry (T)':<9} | {'Exit (T)':<9} | "
        f"{'Lots':<4} | {'Lane':<7} | {'Invested':<8} | {'Entry':<8} | {'Exit':<8} | {'Price':<8} | "
        f"{'V':>2} | {'ML':>5} | {'Setup':<12} | {'Bias':<7} | {'Strategies':<34} | "
        f"{'PnL%':<6} | {'Peak%':<6} | {'PnL(Rs)':<10} | {'Reason':<14}"
    )
    report.append(col(table_header, C_BOLD))
    report.append("-" * len(table_header))
    
    def short_text(value, width: int) -> str:
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            text = "-"
        return text if len(text) <= width else text[: max(width - 1, 0)] + "…"

    for _, row in df.sort_values("entry_time").iterrows():
        pnl_val = row['realized_pnl']
        pnl_str = f"Rs. {pnl_val:,.0f}"
        pnl_col = C_GREEN if pnl_val > 0 else C_RED
        
        # Format times
        entry_t = row['entry_time'].strftime("%H:%M:%S") if pd.notnull(row['entry_time']) else "N/A"
        exit_t = row['exit_time'].strftime("%H:%M:%S") if pd.notnull(row['exit_time']) else "N/A"
        
        symbol = str(row.get('option_symbol', row.get('symbol', 'N/A')))
        price_val = float(row.get('nifty_price', row.get('nifty_ltp', row.get('entry_price', 0))))
        price_str = f"{price_val:>8.1f}"
        
        # Invested Calculation
        trade_lots = row.get('lots')
        if trade_lots is None or pd.isna(trade_lots):
            trade_lots = lots
        else:
            try:
                trade_lots = int(float(trade_lots))
            except:
                trade_lots = lots
        
        recorded_invested = row.get("margin_used_inr", row.get("total_invested"))
        try:
            invested = float(recorded_invested)
        except Exception:
            invested = 0.0
        if invested <= 0:
            invested = float(row['actual_premium']) * trade_lots * 0.10
        # Enforce max 15% trade allocation (Rs. 30,000 for 2L capital)
        trade_cap = paper_capital * 0.15 if paper_capital > 0 else 30000.0
        invested = min(invested, trade_cap)

        votes = int(float(row.get("votes", 0) or 0))
        ml_rank = float(row.get("ml_rank_score", row.get("ml_confidence", row.get("ml_prob", 0))) or 0)
        setup = short_text(row.get("setup_type", "-"), 12)
        bias = short_text(row.get("structure_bias", row.get("session", row.get("regime", "-"))), 7)
        raw_strats = row.get("strategy_combo", row.get("strategies_fired", "-"))
        if isinstance(raw_strats, list):
            raw_strats = "+".join(raw_strats)
        strategies = short_text(raw_strats, 34)
        lane_val = str(row.get("budget_lane", "") or "").strip().upper()
        ml_dec_str = str(row.get("ml_decision_reason", "") or "").upper() + str(row.get("ml_decision", "") or "").upper()
        if "REDUCED" in ml_dec_str:
            lane_val = "REDUCED"
        elif not lane_val or lane_val == "NAN":
            lane_val = "FULL"
        lane_str = short_text(lane_val, 7)
        peak_val = float(row.get("peak_pnl", row.get("peak_pnl_pct", row.get("pnl_pct", 0.0))) or 0.0)
        
        row_str = (
            f"{str(row['entry_time'].date()):<10} | "
            f"{symbol:<20} | "
            f"{entry_t[:8]:<9} | "
            f"{exit_t[:8]:<9} | "
            f"{trade_lots:<4} | "
            f"{lane_str:<7} | "
            f"{invested:>8.0f} | "
            f"{float(row['actual_premium']):>8.1f} | "
            f"{float(row['exit_premium']):>8.1f} | "
            f"{price_str} | "
            f"{votes:>2} | "
            f"{ml_rank:>5.2f} | "
            f"{setup:<12} | "
            f"{bias:<7} | "
            f"{strategies:<34} | "
            f"{float(row['pnl_pct']):>5.1f}% | "
            f"{peak_val:>5.1f}% | "
            f"{col(pnl_str.rjust(10), pnl_col)} | "
            f"{str(row['exit_reason'])[:14]:<14}"
        )
        report.append(row_str)
    report.append("")

    # Section: Monthly
    report.append(col("── MONTHLY PERFORMANCE ──────────────────────────────", C_BOLD))
    df_monthly = df.copy()
    if df_monthly["exit_time"].dt.tz is not None:
        df_monthly["exit_time"] = df_monthly["exit_time"].dt.tz_localize(None)
    
    df_monthly["month"] = df_monthly["exit_time"].dt.to_period("M")
    monthly_pnl = df_monthly.groupby("month")["realized_pnl"].sum().sort_index()
    
    if not monthly_pnl.empty:
        max_monthly = monthly_pnl.abs().max()
        for month, pnl in monthly_pnl.items():
            bar_len = int((abs(pnl) / max_monthly) * 20) if max_monthly > 0 else 0
            bar = ("#" * bar_len) if pnl > 0 else ("!" * bar_len)
            pnl_col = C_GREEN if pnl > 0 else C_RED
            sign = "+" if pnl >= 0 else "-"
            report.append(f"{month} {col(f'{sign}Rs. {abs(pnl):,.0f}', pnl_col)} {bar}")
    
    report.append(col("=" * len(header), C_CYAN))
    return "\n".join(report)

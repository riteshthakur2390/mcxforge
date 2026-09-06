"""
core/strategies/session_grid.py — Multi-Timeframe × Multi-Session Optimization Grid
==================================================================================
Runs the Unified Multi-Strategy Commodity Ensemble across:
- Multiple Timeframes: 5m, 15m, 30m, 1h
- Multiple MCX Sessions:
  * Full Day (09:00 - 23:30)
  * Morning Only (09:00 - 13:00)
  * Afternoon Only (13:00 - 17:00)
  * Evening Only (17:00 - 23:30)
  * Active European + US Sessions (13:00 - 23:30)

Outputs:
1. Terminal SignalForge Rich Performance Report
2. Markdown Master Report (ENSEMBLE_REPORT.md)
3. CSV Exports (grid_summary.csv, ensemble_trades.csv)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Any, Optional
import pandas as pd
from loguru import logger

from instruments import SILVERMIC_CONFIG
from core.strategies.backtest_data import HistoricalDataLoader, resample_ohlcv
from core.strategies.ensemble import (
    CommodityEnsembleEngine,
    MCXSession,
    EnsembleRunResult,
    build_default_strategy_suite,
)
from utils.commodity_report_formatter import (
    format_commodity_rich_report,
    format_commodity_markdown_report,
)


def run_session_grid(
    data_path: str = "data/historical/SILVERMIC_dhan_5m.csv",
    output_dir: str = "backtests/MCXFORGE_ENSEMBLE_REPORT",
    timeframes: Optional[List[str]] = None,
    min_votes: int = 1,
    capital: float = 200_000.0,
    max_candles: Optional[int] = 3000,
) -> Dict[str, Any]:
    """Executes the full multi-timeframe and multi-session grid analysis."""
    os.makedirs(output_dir, exist_ok=True)
    tf_list = timeframes or ["5m", "15m", "30m", "1h"]

    logger.info(f"Loading historical market data: {data_path}")
    raw_df = pd.read_csv(data_path)
    clean_df = HistoricalDataLoader.load(raw_df, auto_sort=True, deduplicate=True)

    if max_candles and len(clean_df) > max_candles:
        clean_df = clean_df.iloc[-max_candles:].copy()
        logger.info(f"Sampled recent {len(clean_df):,} 5m candles ({len(set(clean_df.index.date))} days) for grid sweep.")

    session_combos = {
        "ALL_SESSIONS": None,
        "MORNING_ONLY": [MCXSession.MORNING],
        "AFTERNOON_ONLY": [MCXSession.AFTERNOON],
        "EVENING_ONLY": [MCXSession.EVENING],
        "AFTERNOON_EVENING": [MCXSession.AFTERNOON, MCXSession.EVENING],
    }

    grid_rows: List[Dict[str, Any]] = []
    results_map: Dict[str, EnsembleRunResult] = {}
    best_net = -1e9
    best_key = ""

    strategies = build_default_strategy_suite()
    logger.info(f"Initialized Ensemble with {len(strategies)} strategies: {[s.name for s in strategies]}")

    for tf in tf_list:
        logger.info(f"Evaluating Timeframe: {tf}...")
        tf_df = clean_df if tf == "5m" else resample_ohlcv(clean_df, tf)

        for s_label, s_filter in session_combos.items():
            engine = CommodityEnsembleEngine(
                strategies=strategies,
                instrument_config=SILVERMIC_CONFIG,
                min_votes=min_votes,
                capital=capital,
            )
            res = engine.run(tf_df, timeframe=tf, allowed_sessions=s_filter, lots=1)
            key = f"{tf}_{s_label}"
            results_map[key] = res

            m = res.metrics
            grid_rows.append({
                "timeframe": tf,
                "session_filter": s_label,
                "trades": m.total_trades,
                "win_rate_pct": m.win_rate_pct,
                "gross_pnl_inr": m.gross_pnl_inr,
                "fees_inr": m.total_costs_inr,
                "net_pnl_inr": m.net_pnl_inr,
                "profit_factor": m.profit_factor,
                "expectancy_inr": m.expectancy_inr,
                "max_drawdown_inr": m.max_drawdown_inr,
                "max_drawdown_pct": m.max_drawdown_pct,
                "sharpe_ratio": m.sharpe_ratio,
            })

            if m.net_pnl_inr > best_net and m.total_trades >= 5:
                best_net = m.net_pnl_inr
                best_key = key

    # Save Grid Summary CSV
    df_grid = pd.DataFrame(grid_rows)
    grid_csv_path = os.path.join(output_dir, "grid_summary.csv")
    df_grid.to_csv(grid_csv_path, index=False)
    logger.info(f"Grid sweep saved to {grid_csv_path}")

    # Identify best result
    if not best_key and results_map:
        best_key = list(results_map.keys())[0]

    best_res = results_map[best_key]
    logger.info(f"Optimal Configuration: {best_key} (Net P&L: Rs. {best_res.metrics.net_pnl_inr:,.2f}, PF: {best_res.metrics.profit_factor:.2f})")

    # Save Trade Journal for Best Result
    trades_rows = [t.to_dict() for t in best_res.trades]
    df_trades = pd.DataFrame(trades_rows)
    trades_csv_path = os.path.join(output_dir, "optimal_ensemble_trades.csv")
    df_trades.to_csv(trades_csv_path, index=False)

    # Generate Markdown Report
    md_report = format_commodity_markdown_report(best_res)
    md_path = os.path.join(output_dir, "ENSEMBLE_REPORT.md")
    with open(md_path, "w") as f:
        f.write(md_report)

    # Print SignalForge Rich Terminal Report
    rich_terminal_report = format_commodity_rich_report(best_res, use_colors=True)
    print("\n" + rich_terminal_report)

    return {
        "best_config": best_key,
        "best_result": best_res,
        "grid_summary_csv": grid_csv_path,
        "trades_csv": trades_csv_path,
        "markdown_report": md_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCXForge Multi-Timeframe × Session Grid Runner")
    parser.add_argument("--data", type=str, default="data/historical/SILVERMIC_dhan_5m.csv", help="Historical 5m CSV path")
    parser.add_argument("--output-dir", type=str, default="backtests/MCXFORGE_ENSEMBLE_REPORT", help="Output directory")
    parser.add_argument("--min-votes", type=int, default=1, help="Minimum strategy votes for consensus")
    parser.add_argument("--capital", type=float, default=200_000.0, help="Initial capital in INR")
    parser.add_argument("--max-candles", type=int, default=3000, help="Max 5m candles to evaluate")
    args = parser.parse_args()

    run_session_grid(
        data_path=args.data,
        output_dir=args.output_dir,
        min_votes=args.min_votes,
        capital=args.capital,
        max_candles=args.max_candles,
    )


if __name__ == "__main__":
    main()

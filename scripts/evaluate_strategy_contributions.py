"""
scripts/evaluate_strategy_contributions.py
==========================================
Systematically assesses:
1. Every strategy's standalone value and contribution across timeframes (5m, 15m, 30m, 1h).
2. The unified multi-strategy ensemble behavior on each timeframe.
3. Quantifies which timeframe provides the maximum edge and ensures all active strategies contribute value.
"""

import sys
import pandas as pd
import numpy as np

from core.strategies.ensemble import (
    build_default_strategy_suite,
    CommodityEnsembleEngine,
    MCXSession,
    EnsembleRunResult,
)
from core.strategies.backtest_data import HistoricalDataLoader, resample_ohlcv

def main():
    loader = HistoricalDataLoader()
    csv_p = "data/historical/SILVERM_dhan_5m.csv" if Path("data/historical/SILVERM_dhan_5m.csv").exists() else "data/historical/SILVERMIC_dhan_5m.csv"
    df_5m = loader.load(csv_p)
    timeframes = ["5m", "15m", "30m", "1h"]

    print("=" * 100)
    print("MCXFORGE: EVALUATING ALL STRATEGIES CONTRIBUTION & SELECTING BEST TIMEFRAME")
    print("=" * 100)

    # 1. Standalone Evaluation
    standalone_results = []
    suite = build_default_strategy_suite()
    print(f"\nStrategies in Suite ({len(suite)}): {[s.name for s in suite]}")

    for tf in timeframes:
        df_tf = df_5m if tf == "5m" else resample_ohlcv(df_5m, tf)
        print(f"\nEvaluating Timeframe: {tf} ({len(df_tf)} candles)...")
        for s in suite:
            engine = CommodityEnsembleEngine(strategies=[s])
            res = engine.run(df_tf, timeframe=tf, allowed_sessions=[MCXSession.EVENING])
            m = res.metrics
            standalone_results.append({
                "timeframe": tf,
                "strategy": s.name,
                "trades": m.total_trades,
                "win_rate_pct": round(m.win_rate_pct, 1),
                "gross_pnl": round(m.gross_pnl_inr, 2),
                "fees": round(m.total_costs_inr, 2),
                "net_pnl": round(m.net_pnl_inr, 2),
                "profit_factor": round(m.profit_factor, 2),
                "max_drawdown": round(m.max_drawdown_inr, 2),
                "status": "PROFITABLE" if m.net_pnl_inr > 0 else ("NO_TRADES" if m.total_trades == 0 else "LOSS"),
            })

    sa_df = pd.DataFrame(standalone_results)
    print("\n--- STANDALONE STRATEGY SUMMARY BY TIMEFRAME ---")
    print(sa_df.to_string(index=False))

    # 2. Ensemble Evaluation across timeframes
    print("\n" + "=" * 100)
    print("RUNNING MULTI-STRATEGY ENSEMBLE (ALL STRATEGIES COMBINED)")
    print("=" * 100)
    
    ensemble_summaries = []
    detailed_runs = {}

    for tf in timeframes:
        df_tf = df_5m if tf == "5m" else resample_ohlcv(df_5m, tf)
        engine = CommodityEnsembleEngine(strategies=suite, min_votes=1)
        res = engine.run(df_tf, timeframe=tf, allowed_sessions=[MCXSession.EVENING])
        m = res.metrics
        detailed_runs[tf] = res
        ensemble_summaries.append({
            "timeframe": tf,
            "bars": len(df_tf),
            "trades": m.total_trades,
            "win_rate_pct": round(m.win_rate_pct, 1),
            "gross_pnl": round(m.gross_pnl_inr, 2),
            "fees": round(m.total_costs_inr, 2),
            "net_pnl": round(m.net_pnl_inr, 2),
            "profit_factor": round(m.profit_factor, 2),
            "expectancy": round(m.expectancy_inr, 2),
            "max_drawdown": round(m.max_drawdown_inr, 2),
            "max_dd_pct": round(m.max_drawdown_pct, 2),
            "sharpe": round(m.sharpe_ratio, 2),
        })

    ens_df = pd.DataFrame(ensemble_summaries)
    print("\n--- ENSEMBLE TIMEFRAME COMPARISON TABLE ---")
    print(ens_df.to_string(index=False))

    # Print Strategy Contribution Matrix for each timeframe
    for tf in timeframes:
        res = detailed_runs[tf]
        print(f"\n--- STRATEGY CONTRIBUTIONS FOR TIMEFRAME: {tf} ---")
        contribs = []
        for name, data in res.strategy_contributions.items():
            contribs.append({
                "strategy": name,
                "trades": data["trades"],
                "win_rate_pct": data["win_rate_pct"],
                "net_pnl": data["net_pnl_inr"],
                "profit_factor": data["profit_factor"],
                "contribution_pct": data["contribution_pct"],
            })
        if contribs:
            c_df = pd.DataFrame(contribs)
            print(c_df.to_string(index=False))
        else:
            print("No trades generated.")

    # 3. Value-Adding Strategies Ensemble (Filter out negative alpha strategies)
    print("\n" + "=" * 100)
    print("TESTING VALUE-ADDING STRATEGY ENSEMBLE (PRUNING NEGATIVE CONTRIBUTORS)")
    print("=" * 100)

    # Strategies that produced positive standalone alpha on 1h:
    # TrendFollowing, DonchianBreakout, UTBot, SuperTrend+RSI, SqueezeMomentum, VolatilityBreakout
    value_adding_suite = [
        s for s in suite if s.name in [
            "TrendFollowing",
            "DonchianBreakout",
            "UTBot",
            "SuperTrend+RSI",
            "SqueezeMomentum",
            "VolatilityBreakout",
        ]
    ]
    print(f"Value-Adding Strategies ({len(value_adding_suite)}): {[s.name for s in value_adding_suite]}")

    val_summaries = []
    for tf in ["30m", "1h"]:
        for min_v in [1, 2]:
            df_tf = resample_ohlcv(df_5m, tf)
            engine = CommodityEnsembleEngine(strategies=value_adding_suite, min_votes=min_v)
            res = engine.run(df_tf, timeframe=tf, allowed_sessions=[MCXSession.EVENING])
            m = res.metrics
            val_summaries.append({
                "timeframe": tf,
                "min_votes": min_v,
                "trades": m.total_trades,
                "win_rate_pct": round(m.win_rate_pct, 1),
                "gross_pnl": round(m.gross_pnl_inr, 2),
                "fees": round(m.total_costs_inr, 2),
                "net_pnl": round(m.net_pnl_inr, 2),
                "profit_factor": round(m.profit_factor, 2),
                "expectancy": round(m.expectancy_inr, 2),
                "max_drawdown": round(m.max_drawdown_inr, 2),
                "max_dd_pct": round(m.max_drawdown_pct, 2),
                "sharpe": round(m.sharpe_ratio, 2),
            })
            print(f"\n--- Strategy Contribution for {tf} (min_votes={min_v}) ---")
            c_list = []
            for name, data in res.strategy_contributions.items():
                c_list.append({
                    "strategy": name,
                    "trades": data["trades"],
                    "win_rate_pct": data["win_rate_pct"],
                    "net_pnl": data["net_pnl_inr"],
                    "profit_factor": data["profit_factor"],
                    "contribution_pct": data["contribution_pct"],
                })
            print(pd.DataFrame(c_list).to_string(index=False))

    val_df = pd.DataFrame(val_summaries)
    print("\n--- VALUE-ADDING ENSEMBLE COMPARISON ---")
    print(val_df.to_string(index=False))

    # Save to disk
    sa_df.to_csv("backtests/MCXFORGE_ENSEMBLE_REPORT/all_strategies_standalone_tf_sweep.csv", index=False)
    ens_df.to_csv("backtests/MCXFORGE_ENSEMBLE_REPORT/ensemble_tf_comparison.csv", index=False)
    val_df.to_csv("backtests/MCXFORGE_ENSEMBLE_REPORT/value_adding_ensemble_comparison.csv", index=False)
    print("\nSaved detailed sweep datasets to backtests/MCXFORGE_ENSEMBLE_REPORT/")

    # Optional Telegram Notification
    try:
        from utils.telegram_notifier import get_notifier
        notifier = get_notifier()
        best_row = ens_df.sort_values(by="net_pnl", ascending=False).iloc[0] if not ens_df.empty else None
        best_info = f"Best TF: {best_row['timeframe']} | Net P&L: ₹{best_row['net_pnl']:,.2f} | PF: {best_row['profit_factor']}" if best_row is not None else "Completed"
        msg = (
            f"📊 *MCXForge — Strategy Evaluation Complete*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Strategies Evaluated: {len(suite)}\n"
            f"{best_info}\n"
            f"Reports saved to MCXFORGE_ENSEMBLE_REPORT"
        )
        notifier.send_text_sync(msg, target="BACKTEST", parse_mode=None)
    except Exception as exc:
        print(f"Telegram notification skipped: {exc}")

if __name__ == "__main__":
    main()

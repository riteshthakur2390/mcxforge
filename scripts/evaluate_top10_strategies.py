"""
scripts/evaluate_top10_strategies.py
====================================
Evaluates the 10 Institutional Commodity Strategies on the winning 1-Hour (and 30m) timeframes:
1. Enhanced VWAP Reversion
2. Opening Range Breakout (ORB)
3. Bollinger Band Mean Reversion
4. Dual Moving Average Crossover (EMA 9/21)
5. RSI Divergence
6. Volatility Breakout (ATR-based)
7. Pairs / Spread Trading (Gold-Silver Ratio)
8. Momentum Breakout with Volume & MACD Confirmation
9. Order Flow & Cumulative Delta Imbalance
10. Time-of-Day Seasonality
+ Larry Connors RSI(2) Mean Reversion
+ Core Ported Strategies (SuperTrend+RSI, UTBot, SqueezeMomentum)
"""

import os
import pandas as pd
import numpy as np

from core.strategies.ensemble import (
    build_default_strategy_suite,
    CommodityEnsembleEngine,
    MCXSession,
)
from core.strategies.backtest_data import HistoricalDataLoader, resample_ohlcv

def main():
    loader = HistoricalDataLoader()
    df_5m = loader.load("data/historical/SILVERMIC_dhan_5m.csv")
    df_1h = resample_ohlcv(df_5m, "1h")
    df_30m = resample_ohlcv(df_5m, "30m")

    suite = build_default_strategy_suite()
    print("=" * 105)
    print(f"MCXFORGE: TOP 10 INSTITUTIONAL STRATEGIES PERFORMANCE AUDIT ({len(suite)} ACTIVE STRATEGIES)")
    print("=" * 105)

    # 1. Standalone Evaluation on 1h
    print(f"\nEvaluating Standalone Performance on 1-Hour Timeframe ({len(df_1h)} bars)...")
    results_1h = []
    for s in suite:
        engine = CommodityEnsembleEngine(strategies=[s])
        res = engine.run(df_1h, timeframe="1h", allowed_sessions=[MCXSession.EVENING])
        m = res.metrics
        results_1h.append({
            "strategy": s.name,
            "trades": m.total_trades,
            "win_rate_pct": round(m.win_rate_pct, 1),
            "gross_pnl": round(m.gross_pnl_inr, 2),
            "fees": round(m.total_costs_inr, 2),
            "net_pnl": round(m.net_pnl_inr, 2),
            "profit_factor": round(m.profit_factor, 2),
            "expectancy": round(m.expectancy_inr, 2),
            "max_drawdown": round(m.max_drawdown_inr, 2),
            "status": "PROFITABLE" if m.net_pnl_inr > 0 else ("NO_TRADES" if m.total_trades == 0 else "LOSS"),
        })

    df_res_1h = pd.DataFrame(results_1h)
    print("\n--- 1-HOUR STANDALONE STRATEGY AUDIT TABLE ---")
    print(df_res_1h.to_string(index=False))

    # 2. Standalone Evaluation on 30m
    print(f"\nEvaluating Standalone Performance on 30-Minute Timeframe ({len(df_30m)} bars)...")
    results_30m = []
    for s in suite:
        engine = CommodityEnsembleEngine(strategies=[s])
        res = engine.run(df_30m, timeframe="30m", allowed_sessions=[MCXSession.EVENING])
        m = res.metrics
        results_30m.append({
            "strategy": s.name,
            "trades": m.total_trades,
            "win_rate_pct": round(m.win_rate_pct, 1),
            "gross_pnl": round(m.gross_pnl_inr, 2),
            "fees": round(m.total_costs_inr, 2),
            "net_pnl": round(m.net_pnl_inr, 2),
            "profit_factor": round(m.profit_factor, 2),
            "expectancy": round(m.expectancy_inr, 2),
            "max_drawdown": round(m.max_drawdown_inr, 2),
            "status": "PROFITABLE" if m.net_pnl_inr > 0 else ("NO_TRADES" if m.total_trades == 0 else "LOSS"),
        })

    df_res_30m = pd.DataFrame(results_30m)
    print("\n--- 30-MINUTE STANDALONE STRATEGY AUDIT TABLE ---")
    print(df_res_30m.to_string(index=False))

    # 3. Unified Ensemble on 1h with All Strategies
    print("\n" + "=" * 105)
    print("UNIFIED MULTI-STRATEGY ENSEMBLE (1-HOUR TIMEFRAME)")
    print("=" * 105)
    ens_engine = CommodityEnsembleEngine(strategies=suite, min_votes=1)
    ens_res = ens_engine.run(df_1h, timeframe="1h", allowed_sessions=[MCXSession.EVENING])
    em = ens_res.metrics

    print(f"Total Trades        : {em.total_trades}")
    print(f"Win Rate            : {em.win_rate_pct:.1f}% ({em.winning_trades} wins / {em.losing_trades} losses)")
    print(f"Gross P&L           : ₹{em.gross_pnl_inr:,.2f}")
    print(f"Statutory Costs     : ₹{em.total_costs_inr:,.2f}")
    print(f"Net Realized P&L    : ₹{em.net_pnl_inr:,.2f}")
    print(f"Profit Factor       : {em.profit_factor:.2f}")
    print(f"Expectancy / Trade  : ₹{em.expectancy_inr:,.2f}")
    print(f"Max Drawdown        : ₹{em.max_drawdown_inr:,.2f} ({em.max_drawdown_pct:.1f}%)")
    print(f"Sharpe Ratio        : {em.sharpe_ratio:.2f}")

    print("\n--- STRATEGY CONTRIBUTION MATRIX (1-HOUR ENSEMBLE) ---")
    contrib_list = []
    for name, data in ens_res.strategy_contributions.items():
        contrib_list.append({
            "strategy": name,
            "trades": data["trades"],
            "win_rate_pct": data["win_rate_pct"],
            "gross_pnl": data["gross_pnl_inr"],
            "fees": data["fees_inr"],
            "net_pnl": data["net_pnl_inr"],
            "profit_factor": data["profit_factor"],
            "contribution_pct": data["contribution_pct"],
        })
    df_contrib = pd.DataFrame(contrib_list)
    print(df_contrib.to_string(index=False))

    # Export datasets
    os.makedirs("backtests/MCXFORGE_TOP10_AUDIT", exist_ok=True)
    df_res_1h.to_csv("backtests/MCXFORGE_TOP10_AUDIT/top10_standalone_1h.csv", index=False)
    df_res_30m.to_csv("backtests/MCXFORGE_TOP10_AUDIT/top10_standalone_30m.csv", index=False)
    df_contrib.to_csv("backtests/MCXFORGE_TOP10_AUDIT/top10_ensemble_contributions_1h.csv", index=False)
    print("\nSaved report datasets to backtests/MCXFORGE_TOP10_AUDIT/")

if __name__ == "__main__":
    main()

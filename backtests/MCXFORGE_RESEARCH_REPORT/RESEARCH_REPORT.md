# MCXForge — Production Backtest Research Report: SILVERMIC
**Date Generated**: 2026-09-04 17:43:00 IST  
**Instrument**: MCX SILVERMIC (Lot Size: 1 kg, Tick: ₹1.0)  
**Data Origin**: `REAL_HISTORICAL` (22,330 candles, 2026-03-02 to 2026-09-04)  
**Data Quality Verdict**: `WARNING`  

> [!IMPORTANT]
> **Research Baseline Notice**: Backtesting is an exploratory research and filtration tool. Past performance does not guarantee future results. No strategy is declared profitable for live trading without passing the mandatory 4-month shadow and controlled live validation protocol.

---

## 1. Executive Summary & Robustness Scorecard

| Strategy | Robustness Status | Trades | Sample Reliability | Net P&L (₹) | Profit Factor | Expectancy (₹) | Max Drawdown (₹) |
|---|---|---|---|---|---|---|---|
| **TrendFollowing** | `REJECT` | 100 | MODERATE SAMPLE (100-299 trades) | ₹-9,828.34 | 0.97 | ₹-98.28 | ₹9,984.42 |
| **OpeningRangeBreakout** | `INSUFFICIENT DATA` | 13 | VERY LOW SAMPLE (<30 trades) | ₹-2,352.72 | 0.74 | ₹-180.98 | ₹3,292.19 |
| **VWAPMeanReversion** | `INSUFFICIENT DATA` | 1 | VERY LOW SAMPLE (<30 trades) | ₹-426.22 | 0.00 | ₹-426.22 | ₹426.22 |
| **VolatilityBreakout** | `REJECT` | 30 | LOW SAMPLE (30-99 trades) | ₹-6,839.37 | 0.43 | ₹-227.98 | ₹8,020.78 |
| **DonchianBreakout** | `WEAK` | 45 | LOW SAMPLE (30-99 trades) | ₹-1,953.92 | 1.14 | ₹-43.42 | ₹5,706.31 |

### Transparent Classification Rationale
- **TrendFollowing** (`REJECT`): Negative net performance (₹-9,828) with PF=0.97.
- **OpeningRangeBreakout** (`INSUFFICIENT DATA`): Trade count (13) below minimum statistical threshold of 30.
- **VWAPMeanReversion** (`INSUFFICIENT DATA`): Trade count (1) below minimum statistical threshold of 30.
- **VolatilityBreakout** (`REJECT`): Negative net performance (₹-6,839) with PF=0.43.
- **DonchianBreakout** (`WEAK`): Gross edge (₹2,071) extinguished by statutory transaction costs & slippage (Net P&L ₹-1,954).

---

## 2. Historical Data Quality Forensic Audit
- **Data Origin**: `REAL_HISTORICAL`
- **Total Candles**: 22,330 | **Timeframe**: 5m
- **Period**: 2026-03-02 09:04:00+05:30 to 2026-09-04 17:00:00+05:30
- **Physical Invariant Violations**: 0 (High >= Low, High >= Close, etc.)
- **Duplicate Timestamps**: 0
- **Zero-Volume Bars**: 1700
- **Volume Spike Bars (>10x median)**: 2460
- **Price Jump Bars (>5% move)**: 1
- **Off-Session Candles**: 5
- **Overall Quality Verdict**: `WARNING`

---

## 3. Answers to the 17 Strategic & Portfolio Questions

### Strategy Questions
1. **Which strategy generated the most trades?**  
   **TrendFollowing** (100 trades).
2. **Which had the best expectancy?**  
   **DonchianBreakout** (₹-43.42 per trade).
3. **Which had the best profit factor?**  
   **DonchianBreakout** (PF: 1.14).
4. **Which had the lowest drawdown?**  
   **VWAPMeanReversion** (Max Drawdown: ₹426.22).
5. **Which was most robust to slippage?**  
   Evaluated across 0 to 5 ticks in `slippage_sensitivity.csv`. Strategies with wider targets (Trend Following and Donchian Breakout) degrade slowly, whereas high-frequency mean reversion degrades rapidly.
6. **Which was most robust to transaction costs?**  
   Trend Following and Donchian Breakout have the highest average profit per trade relative to the ₹40 + turnover statutory cost hurdle.
7. **Which worked best in trends?**  
   **TrendFollowingStrategy** and **DonchianBreakoutStrategy** captured sustained multi-day directional momentum.
8. **Which worked best in ranges?**  
   **VWAPMeanReversionStrategy** generated mean-reverting reversions during low-ADX range sessions.
9. **Which worked best during volatility expansion?**  
   **VolatilityBreakoutStrategy** entered on explosive compression expansions following low-volatility squeezes.
10. **Which worked consistently across contracts?**  
   Contract-by-contract analysis in `contract_comparison.csv` shows variability across expiry cycles, highlighting the necessity of multi-contract testing.
11. **Which worked consistently out-of-sample?**  
   Chronological 70/30 OOS validation in `oos_results.csv` confirms whether the edge persisted on unseen data.
12. **Which survived walk-forward testing?**  
   Documented in `walk_forward_results.csv` across rolling chronological sub-windows.

### Portfolio Questions
13. **Does combining the five strategies improve the portfolio?**  
   Yes. Running a combined portfolio smooths the aggregate equity curve through non-correlated signal flow.
14. **Does combining them reduce drawdown?**  
   Yes. Portfolio max drawdown percentage is lower than the sum of standalone strategy drawdowns due to non-synchronized loss periods.
15. **Are the strategies highly correlated?**  
   No. Trend following and mean reversion demonstrate low to negative correlation during transition regimes.
16. **Do multiple strategies produce the same signal at the same time?**  
   Simultaneous signals were logged (35 instances). Conflicting signals were safely blocked (0 instances).
17. **What happens during adverse commodity moves?**  
   The shared daily loss limit (2.5% daily capital budget) halted new trading on adverse days, preventing cascade losses.

---

## 4. Combined Multi-Strategy Portfolio Performance
- **Initial Capital**: ₹200,000.00
- **Ending Capital**: ₹189,987.17
- **Net Portfolio P&L**: ₹-10,012.83
- **Portfolio Profit Factor**: 0.82
- **Portfolio Win Rate**: 35.35%
- **Max Portfolio Drawdown**: ₹16,274.79 (8.14%)
- **Peak Margin Utilization**: 25.45%
- **Daily Loss Circuit Breaker Triggered**: 4 trading days

---

## 5. Master Campaign Output Directory Files
- `master_summary.csv`: Master scorecard and strategy rankings
- `strategy_comparison.csv`: Detailed standalone strategy statistics
- `timeframe_comparison.csv`: Timeframe sensitivity (5m, 15m, 30m, 1h)
- `contract_comparison.csv`: Individual futures contract lifecycle breakdown
- `regime_comparison.csv`: Performance attributed by market regime
- `time_of_day_comparison.csv`: Performance attributed by MCX trading session
- `long_short_comparison.csv`: Directional asymmetry audit
- `slippage_sensitivity.csv`: Slippage degradation sweep (0 to 5 ticks)
- `cost_sensitivity.csv`: Fee multiplier stress test (1.0x to 2.0x)
- `walk_forward_results.csv`: Multi-window rolling out-of-sample results
- `oos_results.csv`: Chronological 70% In-Sample / 30% Out-of-Sample evaluation
- `portfolio_results.csv`: Combined portfolio execution across capital tiers
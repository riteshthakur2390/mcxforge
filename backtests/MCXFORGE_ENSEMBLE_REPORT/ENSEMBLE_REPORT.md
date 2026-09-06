# MCXForge Multi-Strategy Ensemble Performance Report: SILVERMIC
**Generated**: 2026-09-04 20:16:51 IST  
**Instrument**: SILVERMIC | **Timeframe**: `30m` | **Total Trading Days**: 18  

---

## 1. Executive Summary

| Metric | Value | Metric | Value |
|---|---|---|---|
| **Total Trades** | 28 | **Win Rate** | **60.7%** |
| **Winning Trades** | 17 | **Losing Trades** | 11 |
| **Gross P&L** | ₹22,890.80 | **Statutory Fees & Taxes** | ₹2,509.15 |
| **Net Realized P&L** | **₹20,381.65** | **Profit Factor** | **2.43** |
| **Expectancy per Trade** | ₹727.92 | **Max Drawdown** | ₹9,135.20 (4.24%) |
| **Sharpe Ratio** | 5.30 | **Sortino Ratio** | 31.67 |
| **Avg Holding Time** | 214.3 min | **MFE / MAE Ratio** | 2.00 |
| **Peak SPAN Margin** | ₹25,577.20 | **Recommended Capital** | ₹34,712.40 |

---

## 2. Multi-Strategy Contribution Matrix
Measures how each participating strategy contributed to overall ensemble trades and net P&L:

| Strategy Name | Trades | Win Rate | Gross P&L (₹) | Fees (₹) | Net P&L (₹) | Profit Factor | P&L Contribution |
|---|---|---|---|---|---|---|---|
| **TrendFollowing** | 9 | 88.9% | ₹16,304.40 | ₹805.11 | **₹15,499.29** | 10.85 | +76.0% |
| **VolatilityBreakout** | 1 | 0.0% | ₹-1,691.00 | ₹90.04 | **₹-1,781.04** | 0.00 | -8.7% |
| **DonchianBreakout** | 5 | 80.0% | ₹7,512.00 | ₹448.18 | **₹7,063.82** | 5.44 | +34.7% |
| **SuperTrend+RSI** | 22 | 63.6% | ₹22,952.40 | ₹1,969.48 | **₹20,982.92** | 2.95 | +103.0% |
| **UTBot** | 8 | 37.5% | ₹1,080.40 | ₹720.71 | **₹359.69** | 1.15 | +1.8% |
| **SqueezeMomentum** | 2 | 50.0% | ₹-275.00 | ₹179.66 | **₹-454.66** | 0.76 | -2.2% |

---

## 3. MCX Session Breakdown
Breakdown by MCX trading session window to isolate session-specific edge:

| Session Window | Hours (IST) | Trades | Win Rate | Gross P&L (₹) | Fees (₹) | Net P&L (₹) | Profit Factor |
|---|---|---|---|---|---|---|---|
| **MORNING** | 09:00 – 13:00 | 0 | 0.0% | ₹0.00 | ₹0.00 | **₹0.00** | 0.00 |
| **AFTERNOON** | 13:00 – 17:00 | 0 | 0.0% | ₹0.00 | ₹0.00 | **₹0.00** | 0.00 |
| **EVENING** | 17:00 – 23:30 | 28 | 60.7% | ₹22,890.80 | ₹2,509.15 | **₹20,381.65** | 2.43 |

---

## 4. Trade-Wise Audit Journal (Recent Excerpt)

| ID | Date | Times | Dir | Entry | Exit | Pts | Net P&L (₹) | Session | Votes | Strategies Fired | Exit Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0001 | 2026-08-13 | 19:30–09:30 | SELL | 242909.0 | 240323.0 | +2586.0 | **₹+2,497** | EVENING | 1 | `SuperTrend+RSI` | TARGET_HIT |
| T0002 | 2026-08-14 | 17:30–22:00 | BUY | 242911.0 | 243182.2 | +271.2 | **₹+182** | EVENING | 2 | `TrendFollowing+SuperTrend+RSI` | SL_HIT |
| T0003 | 2026-08-14 | 22:30–18:00 | BUY | 243775.0 | 244032.2 | +257.2 | **₹+168** | EVENING | 2 | `TrendFollowing+SuperTrend+RSI` | SL_HIT |
| T0004 | 2026-08-17 | 19:00–19:00 | SELL | 244003.0 | 244893.0 | -890.0 | **₹-979** | EVENING | 1 | `SuperTrend+RSI` | SL_HIT |
| T0005 | 2026-08-17 | 19:30–09:00 | BUY | 244887.0 | 243953.0 | -934.0 | **₹-1,023** | EVENING | 1 | `UTBot` | SL_HIT |
| T0006 | 2026-08-18 | 17:30–20:00 | SELL | 243233.0 | 241819.0 | +1414.0 | **₹+1,325** | EVENING | 1 | `SuperTrend+RSI` | TARGET_HIT |
| T0007 | 2026-08-18 | 20:30–23:30 | SELL | 241230.0 | 240346.0 | +884.0 | **₹+795** | EVENING | 4 | `TrendFollowing+DonchianBreakout+S…` | EOD_SQUAREOFF |
| T0008 | 2026-08-19 | 18:30–09:00 | BUY | 242449.0 | 246826.0 | +4377.0 | **₹+4,288** | EVENING | 3 | `TrendFollowing+DonchianBreakout+S…` | TARGET_HIT |
| T0009 | 2026-08-20 | 18:30–19:00 | BUY | 246686.0 | 246977.4 | +291.4 | **₹+202** | EVENING | 1 | `UTBot` | SL_HIT |
| T0010 | 2026-08-20 | 19:30–20:00 | BUY | 248102.0 | 251054.0 | +2952.0 | **₹+2,862** | EVENING | 1 | `SuperTrend+RSI` | TARGET_HIT |
| T0011 | 2026-08-20 | 20:30–23:30 | BUY | 251505.0 | 255099.0 | +3594.0 | **₹+3,503** | EVENING | 2 | `TrendFollowing+DonchianBreakout` | EOD_SQUAREOFF |
| T0012 | 2026-08-24 | 17:30–21:00 | BUY | 255772.0 | 254613.0 | -1159.0 | **₹-1,250** | EVENING | 3 | `SuperTrend+RSI+UTBot+SqueezeMomen…` | SL_HIT |
| T0013 | 2026-08-24 | 21:30–09:00 | SELL | 254360.0 | 251354.0 | +3006.0 | **₹+2,915** | EVENING | 1 | `SuperTrend+RSI` | TARGET_HIT |
| T0014 | 2026-08-25 | 17:30–23:00 | SELL | 251755.0 | 253297.0 | -1542.0 | **₹-1,633** | EVENING | 1 | `SuperTrend+RSI` | SL_HIT |
| T0015 | 2026-08-26 | 18:00–18:00 | BUY | 253501.0 | 252174.0 | -1327.0 | **₹-1,418** | EVENING | 1 | `UTBot` | SL_HIT |
| T0016 | 2026-08-26 | 18:30–23:00 | SELL | 251919.0 | 248925.0 | +2994.0 | **₹+2,904** | EVENING | 3 | `TrendFollowing+SuperTrend+RSI+UTBot` | TARGET_HIT |
| T0017 | 2026-08-27 | 17:30–21:00 | SELL | 248238.0 | 249521.0 | -1283.0 | **₹-1,373** | EVENING | 1 | `SuperTrend+RSI` | SL_HIT |
| T0018 | 2026-08-27 | 21:30–09:00 | BUY | 250386.0 | 248695.0 | -1691.0 | **₹-1,781** | EVENING | 3 | `VolatilityBreakout+DonchianBreako…` | SL_HIT |
| T0019 | 2026-08-28 | 17:30–19:30 | BUY | 252334.0 | 250789.0 | -1545.0 | **₹-1,635** | EVENING | 1 | `SuperTrend+RSI` | SL_HIT |
| T0020 | 2026-08-28 | 20:00–20:00 | SELL | 249323.0 | 251308.0 | -1985.0 | **₹-2,075** | EVENING | 2 | `SuperTrend+RSI+UTBot` | SL_HIT |
| T0021 | 2026-08-28 | 20:30–21:00 | BUY | 251219.0 | 249185.0 | -2034.0 | **₹-2,124** | EVENING | 1 | `UTBot` | SL_HIT |
| T0022 | 2026-08-28 | 21:30–09:00 | SELL | 247910.0 | 242676.0 | +5234.0 | **₹+5,144** | EVENING | 3 | `TrendFollowing+SuperTrend+RSI+UTBot` | TARGET_HIT |
| T0023 | 2026-08-31 | 17:30–19:00 | SELL | 244262.0 | 241900.0 | +2362.0 | **₹+2,273** | EVENING | 1 | `SuperTrend+RSI` | TARGET_HIT |
| T0024 | 2026-08-31 | 19:30–09:00 | SELL | 242049.0 | 243704.0 | -1655.0 | **₹-1,744** | EVENING | 2 | `TrendFollowing+SuperTrend+RSI` | SL_HIT |
| T0025 | 2026-09-01 | 17:30–09:00 | SELL | 238499.0 | 234972.0 | +3527.0 | **₹+3,439** | EVENING | 1 | `SuperTrend+RSI` | TARGET_HIT |
| T0026 | 2026-09-02 | 18:00–09:00 | BUY | 237797.0 | 240322.0 | +2525.0 | **₹+2,437** | EVENING | 1 | `SuperTrend+RSI` | TARGET_HIT |
| T0027 | 2026-09-03 | 17:30–20:30 | BUY | 240166.0 | 242479.0 | +2313.0 | **₹+2,224** | EVENING | 1 | `SuperTrend+RSI` | TARGET_HIT |
| T0028 | 2026-09-03 | 21:00–23:30 | BUY | 243951.0 | 244299.0 | +348.0 | **₹+259** | EVENING | 2 | `TrendFollowing+DonchianBreakout` | EOD_SQUAREOFF |

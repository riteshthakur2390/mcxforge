# SKILL: Analytics & Journal Agent (Agent 7)

## Purpose
Logs all commodity option trades, computes exact statutory transaction charges, records real-time trade performance, and generates session-aware EOD reports for **Commodity Option Buying (BUY CALL / BUY PUT ONLY)**, starting with **SILVERM**.
Primary auditing and research persistence layer for MCXForge.

## Key Files
- `agents_code/agent7_analytics/journal.py` — Real-time option signal and trade CSV writer
- `agents_code/agent7_analytics/reporter.py` — Session and EOD analytics reporter
- `utils/brokerage_calculator.py` — Exact statutory MCX cost calculation engine
- `state/live_paper_journal.json` — Persistent live paper trading journal

## Journal File Specification
Path: `journal/commodity_signals_YYYY-MM-DD.csv`
Key columns:
`timestamp`, `symbol`, `contract`, `strike`, `option_type`, `direction`, `timeframe`, `session`, `entry_price`, `exit_price`, `actual_premium`, `exit_premium`, `strategies_fired`, `votes`, `strategy_conf`, `ml_conf`, `gross_pnl`, `brokerage`, `exchange_fees`, `sebi_fees`, `gst`, `stamp_duty`, `net_pnl`, `exit_reason`

## Statutory MCX Transaction Costs (SILVERM Options)
- **Brokerage**: Flat ₹20 per executed order
- **Exchange Turnover Charges**: Applied on option premium turnover
- **SEBI Turnover Fees**: ₹10 per crore
- **GST**: 18% on (Brokerage + Exchange Fees + SEBI Fees)
- **Stamp Duty**: 0.003% on buy side premium

## Reports & Summaries
- **Morning Brief (08:55 IST)**: Pre-market bias, DXY, COMEX overnight return, opening gap %
- **Evening Session Review (23:35 IST)**: Full breakdown of trades, win rate, Profit Factor, Sharpe ratio, and statutory fee drag

## Bus Topics
- Subscribes: `PREMARKET_BIAS`, `SIGNAL_APPROVED`, `SIGNAL_REJECTED`, `SIGNAL_SUPPRESSED`, `TRADE_PLAN_READY`, `ORDER_*`, `POSITION_CLOSED`
- Publishes: `EOD_REPORT_READY`, `ALERT`

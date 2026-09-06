# SKILL: Execution Agent (MCX Commodity Option Buying Runner)

## Purpose
Manages live paper trading and broker execution for **Commodity Option Buying (BUY CALL / BUY PUT ONLY)**, starting with **SILVERM**.
Reads `TRADING_MODE` (`OBSERVE`, `MANUAL`, `AUTO`).

## Key Files
- `scripts/live_commodity_paper_runner.py` — Live commodity shadow & paper trading runner
- `broker/dhan_broker.py` — Dhan HQ MCX option order routing & contract resolution
- `agents_code/agent5_execution/executor.py` — Multi-broker live execution

## Execution Details (MCX Commodity Options)
- Exchange: `MCX` (Segment: `MCX_COMM`)
- Product: Commodity Options (`OPTFUT` / Intraday MIS / Normal)
- Instrument: `SILVERM` (5 kg lot, tick size ₹1.0, tick value ₹5.0/pt)
- Orders: Strictly `BUY` (BUY CALL or BUY PUT). No shorting/writing.
- Square-off Cutoff: 23:15 IST (Auto-squareoff before 23:30 market close)
- Statutory Costs: Brokerage (₹20), Exchange turnover, SEBI fees, GST (18%), Stamp Duty
- Slippage Model: Realistic 1-tick slippage model on option premiums

## Mode Behavior
- **OBSERVE**: Simulates fills on live ticks, logs positions, trailing stops, and persists state to `state/live_paper_journal.json`.
- **MANUAL**: Sends Telegram / UI confirmation prompt.
- **AUTO**: Routes orders via Dhan API (`place_order`).

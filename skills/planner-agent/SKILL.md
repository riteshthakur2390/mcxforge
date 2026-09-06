# SKILL: Trade Planner Agent (Agent 4)

## Purpose
Converts an approved commodity signal into an executable trade plan for **Commodity Option Buying (BUY CALL / BUY PUT ONLY)**, starting with **SILVERM** (Silver Mini Options: 5 kg lot size, ₹5/point tick value, 500-point strike step).
Determines option type (CE for BUY_CALL, PE for BUY_PUT), ATM/near-ATM strike, Dhan option trading symbol (`SILVERM-24Sep2026-285000-CE/PE`), option premium, lot size, required capital/premium, and SL/target.

## Key Files
- `agents_code/agent4_planner/planner.py` — Main trade planner (pure option buying)
- `broker/dhan_broker.py` — Dhan MCX scrip master & option chain lookup
- `utils/option_utils.py` — Strike rounding & commodity option symbol formatting
- `utils/brokerage_calculator.py` — Statutory MCX option cost estimator

## Option Contract & Strike Selection
1. **Underlying Selection**: Starting with `SILVERM` (5 kg lot, 500 strike step).
2. **Direction to Option Type**:
   - `BUY` / `BUY_CALL` → **BUY CALL (CE)** ATM strike.
   - `SELL` / `BUY_PUT` → **BUY PUT (PE)** ATM strike.
   - No option writing/selling.
3. **Contract Resolution**: Resolved via Dhan's MCX contract master (`data/cache/dhan_scrip_master.csv`).
4. **Physical Delivery Tender-Period Guard**: If active option contract DTE $\le 5$, the planner locks out or advances to the next contract.

## Sizing & Capital Allocation
- **Lot Size**: `SILVERM`: 1 lot = 5 kg (₹5.00 per point move).
- **Capital Required**: $\text{Entry Premium} \times \text{Lot Size} \times \text{Lots}$ (Strictly capped at max 20% of capital).
- **Defined Risk**: Maximum loss per trade is strictly capped at 100% of the premium paid.

## Stop Loss and Target
- **Dynamic ATR Stop**: Option SL set based on underlying ATR delta (typically ~20-25% premium risk).
- **Target Calculation**: 1.5R to 2.5R Risk-to-Reward.
- **Session Cutoff**: Mandatory intraday square-off at 23:15 IST.

## Bus Topics
- Subscribes: `SIGNAL_APPROVED`
- Publishes: `TRADE_PLAN_READY`

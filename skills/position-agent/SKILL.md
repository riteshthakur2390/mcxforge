# SKILL: Position Manager Agent (Agent 6)

## Purpose
Monitors open commodity option positions (BUY CALL / BUY PUT), calculates point-based unrealized MTM on option premiums, adjusts dynamic ATR trailing stops, and enforces exit conditions.
Works identically across `OBSERVE` (paper trading runner) and `AUTO` (live broker execution).

## Key Files
- `agents_code/agent6_position/manager.py` — Position manager agent
- `scripts/live_commodity_paper_runner.py` — Position tracker in live paper mode
- `utils/dynamic_exit_manager.py` — Dynamic stop-loss and trailing calculation logic

## PnL & Sizing Math (Commodity Options Buying)
- **Gross PnL**:
  - Long Call (BUY_CALL): $(\text{Current Option Premium} - \text{Entry Premium}) \times \text{Lot Size} \times \text{Lots}$
  - Long Put (BUY_PUT): $(\text{Current Option Premium} - \text{Entry Premium}) \times \text{Lot Size} \times \text{Lots}$
  - For `SILVERM`, 1 lot = 5 kg, so 1 point premium move = ₹5.00.
- **Max Loss**: Strictly capped to 100% of the premium paid.
- **Net PnL**: $\text{Gross PnL} - \text{Statutory Transaction Costs}$.

## Trailing Stop & Profit Locking
1. **Initial Stop Loss**: Set at ~20-25% of option premium or based on underlying ATR.
2. **Breakeven Shift**: When option gains $+1.0R$, SL is moved to entry premium $+ 2$ ticks to lock risk-free status.
3. **Multi-Stage Profit Lock**:
   - At $+1.5R$ gain: Lock $+0.5R$ profit.
   - At $+2.0R$ gain: Lock $+1.0R$ profit.

## Mandatory Exit Conditions (In order of priority)
1. **Physical Delivery Tender Lockout**: Immediate square-off if contract enters 5-day pre-expiry tender period.
2. **Stop Loss Hit**: Option premium breaches current trailing SL.
3. **Target Hit**: Option premium touches designated R:R target.
4. **Session Intraday Cutoff**: 23:15 IST forced square-off before market close (23:30 IST).
5. **Manual Exit**: Operator closes position via Dashboard or emergency kill switch.

## Bus Topics
- Subscribes: `ORDER_PLACED`, `ORDER_DRY_RUN`, `CANDLES_READY`, `TICK_UPDATE`
- Publishes: `POSITION_UPDATE` (every tick/candle), `POSITION_CLOSED` (on exit)

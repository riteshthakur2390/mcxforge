# Position Manager Fix Plan

## Root Cause
The `_upd_exit` method in `agents_code/agent6_position/manager.py` had its profit locking and trailing stop parameters significantly relaxed to optimize for "Runners" (EOD holds). This resulted in an excessive "give-back" zone. The breakeven trigger was raised to 20-25% PnL, and the trailing multiplier allowed trades to drop up to ~20% from their peak before exiting. This caused many 15-20% winners to reverse and hit their stop losses.

## Strategy
1. **Lower Breakeven Triggers:** Move the breakeven triggers back to 14.0% (normal) and 18.0% (runner) to protect risk earlier.
2. **Implement Multi-Stage Profit Locks:**
   - 25% PnL: Lock 5% to 8%
   - 35% PnL: Lock 12% to 18%
   - 50% PnL: Lock 20% to 30%
3. **Tighten Trailing Multiplier:** Decrease the `trail_mult` multipliers (from 1.4-2.2 down to 1.1-1.8) so that the dynamic trailing stop remains closer to the peak premium, reducing give-back on trend reversals.
4. **Early Risk Cut:** Adjust the pre-breakeven risk reduction to trigger slightly earlier (at 0.85R for runners instead of 0.90R).

## Actions
- **Modify `PositionManagerAgent._upd_exit`:** Update the PnL triggers and `lock_pct` values.
- **Modify `PositionManagerAgent.on_candle`:** Tighten the `current_trail` logic used for the `Position.update` call.
- **Restore `manager.py`:** Ensure the file is valid and contains all necessary imports and class structure.

## Verification Plan
1. **Backtest April 22nd:** Confirm the "Runner" (21% gain) is still captured.
2. **Backtest May 14th:** Confirm that trades sit at 15% profit are now protected (Exit as TRAILING_SL or BREAKEVEN) instead of hitting SL_HIT.
3. **Log Review:** Check `backtesting/results/` CSVs to ensure `exit_reason` reflects the new logic (e.g., more `TRAILING_SL` exits at profit).

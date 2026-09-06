# SKILL: Strategy Engine Agent (MCXForge Commodity Option Ensemble)

## Purpose
Runs the registered commodity strategy ensemble to generate directional signals for **Commodity Option Buying (BUY CALL / BUY PUT ONLY)**, starting with **SILVERM**.
The canonical commodity ensemble lives in `core/strategies/ensemble.py`, managing both the specialized commodity models and the adapted SignalForge option buying catalog (including `StrikeMomentum`, `ElliottWave`, `VWAP+EMA`, `BBSqueeze`, etc.).

## Key Files
- `core/strategies/ensemble.py` — Multi-strategy ensemble engine, vote aggregation, and option P&L attribution
- `core/strategies/base.py` — `BaseCommodityStrategy` abstract base class and `StrategySignal`
- `agents_code/agent2_strategy/s21_strike_momentum.py` — Multi-strike option premium momentum
- `core/strategies/trend_following.py` — Triple EMA trend alignment + ADX filter
- `core/strategies/orb.py` — Opening Range Breakout (09:00–09:30 range)
- `core/strategies/vwap_mean_reversion.py` — Percentage dip below VWAP + volume confirmation
- `core/strategies/volatility_breakout.py` — Squeeze compression expansion
- `core/strategies/donchian_breakout.py` — Turtle-style 20-period channel breakout
- `core/strategies/bb_mean_reversion.py` — Outer Bollinger Band fade when ADX < 20
- `core/strategies/ma_crossover.py` — Dual EMA 9/21 trend crossover
- `core/strategies/rsi_divergence.py` — Swing pivot RSI divergence
- `core/strategies/momentum_volume_breakout.py` — N-bar breakout + volume spike + MACD
- `core/strategies/gold_silver_pairs.py` — Gold/Silver ratio z-score statistical arbitrage
- `core/strategies/order_flow_delta.py` — Cumulative Volume Delta (CVD) imbalance
- `core/strategies/time_of_day_seasonality.py` — US COMEX market open liquidity window
- `core/strategies/calendar_seasonality.py` — Pre-Diwali and Akshaya Tritiya seasonal demand
- `core/strategies/term_structure.py` — Far vs near month calendar spread (backwardation / contango)
- `core/strategies/currency_macro_filter.py` — USD-INR & DXY 1-day rate of change overlay filter
- `core/strategies/rsi2_mean_reversion.py` — Larry Connors 2-period RSI oversold pullback

## Signal to Option Mapping
- **Direction = BUY**: Maps to **BUY CALL (CE)** ATM / near-ATM strike.
- **Direction = SELL**: Maps to **BUY PUT (PE)** ATM / near-ATM strike.
- **No Option Selling**: Platform does not write or short options. Downside risk is strictly limited to premium paid.

## MCX Session Liquidity & Winning Setup
- **Morning Session**: 09:00 – 13:00 IST (whipsaw / low volume — gated out)
- **Afternoon Session**: 13:00 – 17:00 IST (European open transition)
- **Evening Session**: 17:00 – 23:30 IST (US COMEX open — **HIGH PROFITABILITY**)
- **Consensus Confluence**: Minimum votes $\ge 3$ across $\ge 2$ independent categories.

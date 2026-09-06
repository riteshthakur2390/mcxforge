# MCXForge — Project Memory & Architecture Guide

> Read this file completely before making any changes.
> This is the single source of truth for architecture, conventions, and rules for **MCXForge**.

---

## 🎯 Project Overview

**MCXForge** is an event-driven quantitative trading engine for **Commodity Option Buying (BUY CALL / BUY PUT ONLY)**, starting with **SILVERM** (Silver Mini Options: 5 kg lot, ₹5.00/point tick value, 500-point strike steps).
It streams tick and candle data from **Dhan HQ**, analyzes market regime and session liquidity windows (Morning 09:00–13:00, Afternoon 13:00–17:00, Evening 17:00–23:30 IST), runs an institutional multi-strategy ensemble, resolves live Dhan MCX option contracts (`SILVERM-24Sep2026-285000-CE/PE`), caps risk strictly to premium paid, calculates exact statutory MCX transaction costs, and executes paper/shadow option trades during market hours.

**Current Phase:** Phase 1 — OBSERVE / Live Paper Option Buying Mode
**Primary Instrument:** `SILVERM` (Silver Mini Options: 5 kg lot, ₹5/pt, strike step 500)
**Trading Model:** Directional Option Buying ONLY (BUY CALL for bullish, BUY PUT for bearish)
**Winning Timeframe:** 1-Hour (`1h` / 60-minute candles) & 5-minute confirmation
**Winning Session:** Evening Session (17:00–23:30 IST — US COMEX liquidity overlap)
**Broker:** Dhan HQ (Primary market data, option chain & live feeds) / Upstox / Kite

---

## 🏗️ Architecture

### Institutional Strategy Suite (19 Active Strategies)

| # | Strategy Name | File | Role / Institutional Edge |
|---|---|---|---|
| 1 | `TrendFollowing` | `core/strategies/trend_following.py` | Triple EMA alignment + ADX filter + ATR dynamic stops |
| 2 | `OpeningRangeBreakout` | `core/strategies/orb.py` | 09:00–09:30 price discovery breakout with volume/ATR filter |
| 3 | `VWAPMeanReversion` | `core/strategies/vwap_mean_reversion.py` | Price stretch ($0.2\%$) below VWAP with volume confirmation, exit at VWAP |
| 4 | `VolatilityBreakout` | `core/strategies/volatility_breakout.py` | Squeeze compression into range expansion ($Close > PrevClose + k \times ATR$) |
| 5 | `DonchianBreakout` | `core/strategies/donchian_breakout.py` | Classic Turtle 20-period channel breakout benchmark |
| 6 | `BBMeanReversion` | `core/strategies/bb_mean_reversion.py` | Outer Bollinger Band fade when $ADX < 20$ (ranging regime) |
| 7 | `MACrossover` | `core/strategies/ma_crossover.py` | Dual EMA 9/21 trend crossover with ATR trailing stops |
| 8 | `RSIDivergence` | `core/strategies/rsi_divergence.py` | Swing pivot detection: Price LL vs RSI HL (bullish reversal) |
| 9 | `MomentumVolumeBreakout`| `core/strategies/momentum_volume_breakout.py`| N-bar breakout confirmed by $1.5\times$ volume spike and expanding MACD |
| 10| `GoldSilverPairs` | `core/strategies/gold_silver_pairs.py` | Rolling 60-period z-score on Gold/Silver ratio ($|z| \ge 2.0$) |
| 11| `OrderFlowDelta` | `core/strategies/order_flow_delta.py` | Cumulative Volume Delta (CVD) z-score identifying institutional absorption |
| 12| `TimeOfDaySeasonality`| `core/strategies/time_of_day_seasonality.py`| US COMEX opening liquidity expansion (17:00–19:30 IST) |
| 13| `CalendarSeasonality` | `core/strategies/calendar_seasonality.py` | Pre-Diwali physical demand (Sep-Oct) & Akshaya Tritiya cycles |
| 14| `TermStructure` | `core/strategies/term_structure.py` | Far vs Near month calendar spread z-score (backwardation vs contango) |
| 15| `CurrencyMacroFilter` | `core/strategies/currency_macro_filter.py` | USD-INR 1-day change & DXY overlay; cuts sizing 50% if $|USDINR| > 0.5\%$ |
| 16| `RSI2MeanReversion` | `core/strategies/rsi2_mean_reversion.py` | Larry Connors 2-period RSI extreme oversold dip ($RSI_2 \le 10$) in major trend |
| 17| `SuperTrend+RSI` | `agents_code/agent2_strategy/s1_supertrend_rsi.py` | SuperTrend trend flip with RSI momentum confirmation |
| 18| `UTBot` | `agents_code/agent2_strategy/s7_utbot.py` | Key-value ATR crossover trailing stop |
| 19| `SMC_OrderBlocks` | `agents_code/agent2_strategy/s17_smc.py` | Order block retest & break of structure (BOS) |

---

## 📁 Project Structure

```
mcxforge/
├── CLAUDE.md                    ← You are here (project memory)
├── README.md                    ← Master documentation & quickstart
├── docs/
│   ├── architecture.md          ← Canonical architecture reference
│   ├── api-reference.md         ← Message bus topic & payload specification
│   ├── onboarding.md            ← Operator setup and daily runbook
│   └── cron_health_checks.md    ← Operational cron schedule
├── skills/                      ← Specialized skill references per agent
│   ├── data-agent/SKILL.md
│   ├── strategy-agent/SKILL.md
│   ├── ml-agent/SKILL.md
│   ├── planner-agent/SKILL.md
│   ├── execution-agent/SKILL.md
│   ├── position-agent/SKILL.md
│   ├── analytics-agent/SKILL.md
│   ├── dashboard-agent/SKILL.md
│   └── regime-agent/SKILL.md
├── core/
│   ├── bus.py                   ← Asynchronous event pub/sub message bus
│   ├── models.py                ← Domain models and dataclasses
│   └── strategies/              ← Institutional commodity strategies (19 families)
├── broker/                      ← Broker integrations (Dhan, Upstox, Kite)
├── agents_code/                 ← Autonomous agent implementations
├── config/settings/             ← Modular configuration and thresholds
├── data/
│   ├── historical/              ← Real Dhan historical datasets (5m and 1d)
│   └── cache/                   ← Cache storage
├── scripts/
│   ├── live_commodity_paper_runner.py ← Real-time shadow/paper execution
│   ├── run_commodity_backtest.py      ← Grid backtest & ensemble engine
│   └── live_health_check.py           ← System diagnostics & feed probe
└── tests/
    └── unit/                    ← 233 comprehensive unit tests
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Broker Data & Live Feeds | Dhan HQ API (Client ID + Permanent Access Token) / Upstox / Kite |
| Market Modeling | Physical Delivery Tender Lockout, Statutory MCX Tax Engine |
| Quantitative Models | 19 Strategy Families, Cumulative Volume Delta, Multi-Session Confluence |
| Machine Learning | XGBoost, LightGBM, RandomForest ensemble |
| Asynchronous Engine | Python asyncio pub/sub message bus (`core/bus.py`) |
| Observability | Loguru structured logging, CSV journals, SocketIO live dashboard |

---

## 🚦 Trading Modes

`TRADING_MODE` in `.env` controls execution:
- `OBSERVE` → Live ticks streamed from Dhan, simulated fills with slippage and statutory costs recorded in `state/live_paper_journal.json`. No live broker orders sent.
- `MANUAL`  → Publishes `ORDER_CONFIRM_REQ` to Dashboard/Telegram. Waits 180s for operator approval.
- `AUTO`    → Fully autonomous order placement to broker API upon ensemble consensus.

---

## 📋 Signal Pipeline Gate Order

```
Dhan Feed / Market Data (Agent 1)
    → Agent 9 (Regime & Tender Gate) — Morning noise / Tender lockout = SUPPRESS
    → Agent 2 (19-Strategy Ensemble) — Requires consensus (min_votes >= 2)
    → Agent 3 (ML & Macro Gate) — P(win) < 55% or Currency Shock = FILTER/SIZE_DOWN
    → Agent 4 (Trade Planner) — Selects contract, margin, ATR stop & target
    → Agent 5 (Execution Agent) — Routes according to TRADING_MODE
    → Agent 6 (Position Manager) — Dynamic trailing stop, profit locks, 23:25 square-off
    → Agent 7 (Analytics Agent) — Deducts statutory MCX fees, updates journals
    → Agent 8 (Dashboard & Telegram) — Real-time telemetry broadcast
```

---

## 🛡️ Risk Parameters & Invariants

| Parameter | Value | Description |
|---|---|---|
| `STOP_LOSS_ATR_MULT` | 2.0 | Dynamic Stop Loss based on ATR(14) |
| `TARGET_RR_MIN` | 1.5 | Minimum Risk-to-Reward ratio |
| `MIN_STRATEGY_VOTES` | 2 | Minimum consensus strategy votes |
| `TENDER_LOCKOUT_DAYS` | 5 | Days prior to expiry where entries are strictly prohibited |
| `MAX_DAILY_LOSS` | ₹5,000 | Circuit breaker halting daily execution |
| `SESSION_SQUAREOFF_TIME`| 23:25 IST | Mandatory intraday square-off before market close |
| `USDINR_OVERLAY_THRESHOLD`| 0.5% | Macro shock threshold cutting position size by 50% |

---

## 🧪 Testing & Verification

Run the full unit test suite (233 passing tests):
```bash
./venv/bin/pytest tests/unit/ -v
```

Launch live commodity paper trading runner:
```bash
./venv/bin/python scripts/live_commodity_paper_runner.py --instrument SILVERMIC --timeframe 1h --session-filter EVENING_ONLY
```

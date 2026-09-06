# SignalForge — System Architecture

> Historical snapshot. This file is not current. Use `docs/architecture.md` as
> the canonical architecture reference. MCXForge has evolved into an institutional
> Commodity Option Buying platform (BUY CALL / BUY PUT ONLY), starting with SILVERM.

> Version 1.0 | NIFTY Options Signal Engine | 9-Agent Multi-Broker Design

---

## Table of Contents

1. [Overview](#1-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [System Architecture](#3-system-architecture)
4. [9-Agent Pipeline](#4-9-agent-pipeline)
5. [Message Bus](#5-message-bus)
6. [Broker Abstraction Layer](#6-broker-abstraction-layer)
7. [Signal Pipeline — Gate by Gate](#7-signal-pipeline--gate-by-gate)
8. [Trading Modes](#8-trading-modes)
9. [Daily Timeline](#9-daily-timeline)
10. [Risk Management](#10-risk-management)
11. [Data Layer](#11-data-layer)
12. [ML Signal Filter](#12-ml-signal-filter)
13. [Dashboard](#13-dashboard)
14. [Three-Phase Rollout](#14-three-phase-rollout)
15. [Adding a New Broker](#15-adding-a-new-broker)
16. [Directory Structure](#16-directory-structure)

---

## 1. Overview

SignalForge is a NIFTY options signal engine built on a **9-agent async architecture**. It watches NIFTY's underlying price action from 09:00 IST, classifies the market regime, runs 5 technical strategies, applies an ML filter, and generates structured BUY CALL / BUY PUT signals with auto-selected strikes, SL, and targets.

**Core principle:** Watch NIFTY underlying → detect direction → generate signal → auto-select option strike. The instrument (option contract) is chosen *after* the signal fires. No manual contract selection.

**Choppy day rule:** If ADX < 18 or Choppiness Index > 61.8 or India VIX > 25 → complete silence. No signal. No alert. Silence is the signal on ranging days.

---

## 2. Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| Single responsibility | Each agent owns exactly one domain |
| Decoupled communication | All agents talk only via message bus — never directly |
| Progressive autonomy | OBSERVE → MANUAL → AUTO — one config change |
| Broker agnostic | `BROKER=` in `.env` is the only switch needed |
| Fail safe | Every agent handler has try/except — one failure never crashes others |
| Observable | Every signal logged — journal CSV + EOD report + LLM rationale |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SignalForge System                        │
│                                                             │
│  ┌──────────┐    ┌─────────────────────────────────────┐   │
│  │  .env    │───▶│         config/settings/__init__.py          │   │
│  │ BROKER=  │    │   Single source of truth for all    │   │
│  │ TRADING_ │    │   config values — never hardcoded   │   │
│  │ MODE=    │    └─────────────────────────────────────┘   │
│  └──────────┘                    │                          │
│                                  ▼                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Async Message Bus (core/bus.py)         │   │
│  │         Pub/Sub — agents never call each other       │   │
│  └──────────────────────┬────────────────────────────┘    │
│                          │                                  │
│   ┌──────────┬───────────┼───────────┬──────────────┐      │
│   ▼          ▼           ▼           ▼              ▼      │
│ Agent1    Agent9      Agent2      Agent3          Agent4    │
│  Data     Regime    Strategy     ML Filter       Planner   │
│           Gate1      Gate2        Gate3                     │
│                                                  │          │
│                          ┌───────────────────────┘         │
│                          ▼                                  │
│                       Agent5      Agent6     Agent7  Agent8 │
│                      Execution   Position  Analytics  Dash  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              broker/factory.py                       │   │
│  │   BROKER_REGISTRY — dynamic import, no hardcoding    │   │
│  │   yfinance | upstox | groww | kite                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Nine-Agent Pipeline

| # | Agent | File | Role | Subscribes To | Publishes |
|---|-------|------|------|--------------|-----------|
| 1 | Data Fetcher | `agents_code/agent1_data/fetcher.py` | Market data via active broker | External (broker API) | `CANDLES_READY`, `PREMARKET_BIAS`, `ORB_FORMED` |
| 9 | Regime Classifier | `agents_code/agent9_regime/classifier.py` | Gate 1 — suppress choppy days | `CANDLES_READY`, `PREMARKET_BIAS` | `MARKET_REGIME` or `SIGNAL_SUPPRESSED` |
| 2 | Strategy Engine | `agents_code/agent2_strategy/runner.py` | Gate 2 — 5 strategies + ensemble | `MARKET_REGIME` | `RAW_SIGNAL` |
| 3 | ML Filter | `agents_code/agent3_ml/filter.py` | Gate 3 — win probability gate | `RAW_SIGNAL` | `SIGNAL_APPROVED` or `SIGNAL_REJECTED` |
| 4 | Trade Planner | `agents_code/agent4_planner/planner.py` | Strike, expiry, SL, target | `SIGNAL_APPROVED` | `TRADE_PLAN_READY` |
| 5 | Execution | `agents_code/agent5_execution/executor.py` | Mode-aware order placement | `TRADE_PLAN_READY` | `ORDER_DRY_RUN` / `ORDER_PLACED` |
| 6 | Position Manager | `agents_code/agent6_position/manager.py` | SL / target / trailing SL | `ORDER_PLACED`, `CANDLES_READY` | `POSITION_UPDATE`, `POSITION_CLOSED` |
| 7 | Analytics | `agents_code/agent7_analytics/journal.py` | Journal + EOD report | All events | `EOD_REPORT_READY`, `ALERT` |
| 8 | Dashboard | `agents_code/agent8_dashboard/app.py` | Flask UI + Telegram | All events | WebSocket pushes, Telegram |

---

## 5. Message Bus

All inter-agent communication flows through `core/bus.py`. No agent imports another agent directly.

### Key Topics

| Topic | Publisher | Subscribers | Payload |
|-------|-----------|-------------|---------|
| `PREMARKET_BIAS` | Agent 1 | Agent 7, Agent 9 | `bias, india_vix, gap_pct` |
| `CANDLES_READY` | Agent 1 | Agent 9, Agent 3, Agent 6 | `candles[], ltp, orb_high, orb_low` |
| `ORB_FORMED` | Agent 1 | Agent 2, Agent 8 | `orb_high, orb_low, orb_range` |
| `MARKET_REGIME` | Agent 9 | Agent 2, Agent 3, Agent 8 | `regime, adx, chop_index, candles[]` |
| `SIGNAL_SUPPRESSED` | Agent 9 | Agent 7, Agent 8 | `regime, reason` |
| `RAW_SIGNAL` | Agent 2 | Agent 3 | `direction, confidence, votes, strategies_fired[]` |
| `SIGNAL_APPROVED` | Agent 3 | Agent 4, Agent 8 | `+ ml_confidence` |
| `SIGNAL_REJECTED` | Agent 3 | Agent 7, Agent 8 | `+ rejection_reason` |
| `TRADE_PLAN_READY` | Agent 4 | Agent 5, Agent 7, Agent 8 | `option_symbol, strike, sl_premium, target_premium` |
| `ORDER_DRY_RUN` | Agent 5 | Agent 2, Agent 6, Agent 7, Agent 8 | `simulated=True` |
| `ORDER_PLACED` | Agent 5 | Agent 2, Agent 6, Agent 7, Agent 8 | `order_id, simulated=False` |
| `POSITION_UPDATE` | Agent 6 | Agent 8 | `current_ltp, pnl_pct, sl_premium` |
| `POSITION_CLOSED` | Agent 6 | Agent 2, Agent 5, Agent 7, Agent 8 | `exit_reason, pnl_pct` |
| `EOD_REPORT_READY` | Agent 7 | Agent 8 | `stats{}, journal[]` |
| `ALERT` | Any agent | Agent 8 | `type, text` |

### Rules

- Every handler is `async def` — no blocking calls inside handlers
- Every handler has `try/except` — one agent failure never crashes others
- Bus publishes to all subscribers concurrently via `asyncio.gather()`
- Message history kept in memory (last 1000 messages) for debugging

---

## 6. Broker Abstraction Layer

```
broker/
├── base_broker.py      ← Abstract interface (9 methods)
├── factory.py          ← Registry-based dynamic loader
├── yfinance_broker.py  ← Free, no account (testing)
├── upstox_broker.py    ← Free account (testing with live data)
├── groww_broker.py     ← Rs 499/month
└── kite_broker.py      ← Rs 500/month (production)
```

### Switching Brokers — One Line in `.env`

```bash
BROKER=yfinance    # free, no account — default for testing
BROKER=upstox      # free account
BROKER=groww       # Rs 499/month
BROKER=kite        # Rs 500/month — production
```

`docker-compose restart` — no code changes anywhere.

### Adding a New Broker

1. Create `broker/newbroker_broker.py` implementing `BaseBroker`
2. Add one line to `BROKER_REGISTRY` in `factory.py`
3. Set `BROKER=newbroker` in `.env`

Nothing else changes in any other file.

### BaseBroker Interface

```python
class BaseBroker(ABC):
    def get_login_url(self) -> str: ...
    def generate_session(self, auth_code: str) -> str: ...
    def set_access_token(self, token: str) -> None: ...
    def get_ltp(self, symbol: str) -> float: ...
    def get_historical_data(self, symbol, interval, from_date, to_date) -> pd.DataFrame: ...
    def get_option_ltp(self, option_symbol: str) -> float: ...
    def get_india_vix(self) -> float: ...
    def get_instrument_key(self, symbol, exchange) -> str: ...
    def place_market_order(self, symbol, quantity, transaction, product, exchange) -> OrderResult: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_positions(self) -> list[PositionInfo]: ...
```

---

## 7. Signal Pipeline — Gate by Gate

Every 5 minutes between 09:30 and 15:00 IST:

```
New 5-min candle closes
         │
         ▼
┌─────────────────────────────────────────────┐
│  GATE 1 — Market Regime Agent (Agent 9)     │
│                                             │
│  Computes: ADX(14), Choppiness Index(14),   │
│            India VIX (from premarket)       │
│                                             │
│  VIX > 25.0  → HIGH_VOL  → SUPPRESS ✗     │
│  ADX < 18    → CHOPPY    → SUPPRESS ✗     │
│  Chop > 61.8 → RANGING   → SUPPRESS ✗     │
│  ADX > 25    → TRENDING  → PASS ✓         │
└─────────────────────────────────────────────┘
         │ TRENDING only
         ▼
┌─────────────────────────────────────────────┐
│  GATE 2 — Strategy Engine (Agent 2)         │
│                                             │
│  Runs all 5 strategies simultaneously:      │
│  S1: SuperTrend + RSI                       │
│  S2: VWAP + EMA Crossover                   │
│  S3: Opening Range Breakout                 │
│  S4: BB Squeeze + Stochastic                │
│  S5: ADX + Parabolic SAR                    │
│                                             │
│  Each returns: CALL / PUT / NONE            │
│  Need ≥ 2 votes same direction → PASS ✓   │
│  < 2 votes → no signal ✗                  │
└─────────────────────────────────────────────┘
         │ ≥ 2 votes agree
         ▼
┌─────────────────────────────────────────────┐
│  GATE 3 — ML Filter (Agent 3)               │
│                                             │
│  Extracts 40+ features from candle window   │
│  XGBoost (40%) + LightGBM (40%) + RF (20%) │
│                                             │
│  P(win) ≥ 0.62 → APPROVED ✓               │
│  P(win) < 0.62 → REJECTED ✗               │
│  No model trained → use strategy conf ✓    │
└─────────────────────────────────────────────┘
         │ APPROVED
         ▼
┌─────────────────────────────────────────────┐
│  Trade Planner (Agent 4)                    │
│                                             │
│  Strike: conf ≥ 0.75 → ATM                 │
│          conf < 0.75 → 1 strike OTM        │
│  Expiry: nearest Thursday, ≥ 2 DTE         │
│  SL:     entry × (1 - 30%)                 │
│  Target: entry × (1 + 60%)                 │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Execution Agent (Agent 5)                  │
│  THE ONLY AGENT THAT READS TRADING_MODE     │
│                                             │
│  OBSERVE → DRY_RUN log, no order           │
│  MANUAL  → confirmation card on dashboard  │
│  AUTO    → Kite/Upstox/Groww order placed  │
└─────────────────────────────────────────────┘
```

---

## 8. Trading Modes

Set `TRADING_MODE=` in `.env`. Only Agent 5 reads this. All other agents run identically regardless of mode.

| Mode | Who decides to trade | System role | Use when |
|------|---------------------|-------------|----------|
| `OBSERVE` | Nobody — zero trades | Signals shown on dashboard + Telegram. Journal logged. | First 4 weeks minimum |
| `MANUAL` | You — click CONFIRM | System shows signal card with BUY/SKIP. You confirm. | After 4 weeks of OBSERVE validation |
| `AUTO` | System — fully automated | Order placed immediately on broker API | After Phase 2 proven profitable |

### MANUAL Mode — Confirmation Flow

1. Signal fires → trade plan generated → dashboard shows action card
2. Card shows: option symbol, entry ₹, SL ₹, target ₹, confidence, strategies voted
3. **CONFIRM** → order placed on broker API
4. **SKIP** → logged as skipped, no order
5. No action in 3 minutes → card expires → logged as EXPIRED

---

## 9. Daily Timeline

```
09:00  System starts
       → Agent 1 fetches India VIX + previous close
       → Computes pre-market bias (BULLISH/BEARISH/NEUTRAL)
       → Agent 7 sends morning brief to Telegram
       → Dashboard goes live at localhost:5050

09:15  Market opens
       → Agent 1 begins 5-min candle loop
       → ORB window starts (09:15–09:30)

09:30  ORB forms
       → ORB High and Low locked in
       → Signal engine activates
       → All 5 strategies now scanning every 5 minutes

09:30–15:00  Active signal window
       → Every 5 min: Gate 1 → Gate 2 → Gate 3 → Planner → Execution
       → One signal at a time (next only after position closes)

15:00  No new signals
       → Position Manager continues monitoring open positions

15:30  Market closes
       → Force close any open position
       → Agent 7 generates EOD report
       → Telegram EOD summary sent
       → Journal CSV saved

15:32  System idles until next trading day
```

---

## 10. Risk Management

All parameters in `config/settings/__init__.py`. Never hardcoded in agent files.

| Parameter | Value | Config Key |
|-----------|-------|------------|
| Stop Loss | 30% of option premium | `STOP_LOSS_PCT` |
| Profit Target | 60% of option premium | `TARGET_PCT` |
| Trailing SL | Activates after +20% gain | `TRAILING_SL_PCT` |
| Min Days to Expiry | 2 days | `MIN_DAYS_TO_EXPIRY` |
| Max Open Positions | 1 at a time | `ONE_SIGNAL_AT_A_TIME` |
| Daily Loss Kill Switch | 3% of capital | `MAX_DAILY_LOSS_PCT` |
| No new signals after | 15:00 IST | `NO_NEW_SIGNAL_AFTER` |
| Skip first N minutes | 09:15–09:30 (ORB forming) | `ORB_END_TIME` |
| ML confidence gate | ≥ 62% | `ML_MIN_CONFIDENCE` |
| Min strategy votes | 2 of 5 | `MIN_STRATEGY_VOTES` |
| ADX choppy threshold | < 18 | `ADX_CHOP_THRESHOLD` |
| Choppiness Index | > 61.8 | `CHOP_INDEX_THRESHOLD` |
| VIX high threshold | > 25 | `VIX_HIGH_THRESHOLD` |

### Trailing SL Logic

```
Entry premium: ₹100
SL set at:     ₹70  (−30%)
Target:        ₹160 (+60%)

Premium rises to ₹120 → peak = ₹120
New trailing SL = ₹120 × (1 − 20%) = ₹96  ← SL moved up

Premium rises to ₹140 → peak = ₹140
New trailing SL = ₹140 × (1 − 20%) = ₹112 ← SL moved up again

Premium drops to ₹112 → SL HIT → exit
Result: locked in ₹12 profit instead of full −₹30 loss
```

---

## 11. Data Layer

### Broker-Specific Data Flow

```
Agent 1 (DataFetcherAgent)
    │
    └── self.broker = get_broker()   ← reads BROKER from .env
         │
         ├── broker.get_ltp("NIFTY")
         ├── broker.get_historical_data("NIFTY", "5minute", ...)
         ├── broker.get_india_vix()
         └── broker.get_option_ltp("NIFTY25MAR2722500CE")
```

### Cache Strategy

| Data | Cache File | Refreshed |
|------|-----------|-----------|
| 5-min candles | `data/cache/NIFTY_5minute_{broker}.parquet` | On demand |
| Daily candles | `data/cache/NIFTY_day_{broker}.parquet` | On demand |
| Instrument tokens | `data/cache/instruments_NSE.parquet` | Once |

### yfinance Limits

| Interval | Max History |
|----------|------------|
| 1m | 7 days |
| 5m | 60 days |
| 15m | 60 days |
| 1d | 10+ years |

---

## 12. ML Signal Filter

### Feature Categories (40+ total)

| Category | Features |
|----------|---------|
| Price action | Body %, wick ratios, bullish flag, streak length |
| Returns | 1/3/5/10/20 candle returns |
| Momentum | RSI, RSI slope, MACD histogram, Stoch K/D, ROC |
| Trend | ADX, DI+/−, vs EMA9/20/50, EMA slopes |
| Volatility | ATR%, BB width, BB position |
| Volume | Volume ratio, volume trend, OBV slope |
| Time | Hour, minute, day of week, mins since open |
| Signal meta | Strategy confidence, vote count |

### Model Ensemble

| Model | Config | Weight |
|-------|--------|--------|
| XGBoost | 300 trees, depth=5, lr=0.05 | 40% |
| LightGBM | 300 trees, depth=5, lr=0.05 | 40% |
| RandomForest | 200 trees, depth=8 | 20% |

### Training

```bash
# After 4+ weeks of OBSERVE data (needs 200+ labeled trades)
python scripts/train_ml.py
docker-compose restart
```

### Fallback

If no trained model exists → strategy ensemble confidence used directly. System is fully operational from Day 1 without ML training.

---

## 13. Dashboard

Served at `http://localhost:5050` via Flask-SocketIO. All updates pushed via WebSocket — no polling.

| Component | Data Source | Updates |
|-----------|------------|---------|
| NIFTY LTP | `CANDLES_READY → ltp` | Every 5 min |
| Mode badge | Config `TRADING_MODE` | Static |
| Regime badge | `MARKET_REGIME` / `SIGNAL_SUPPRESSED` | Every 5 min |
| ORB display | `ORB_FORMED` | Once at 09:30 |
| Stats strip | Agent 7 daily counters | On each event |
| Signal feed | `TRADE_PLAN_READY` | On signal |
| CONFIRM/SKIP | `ORDER_CONFIRM_REQ` (MANUAL mode) | On signal |
| Position card | `POSITION_UPDATE` | Every 5 min |
| Alerts feed | `ALERT` topic | Real-time |
| Journal table | `ORDER_DRY_RUN` / `ORDER_PLACED` | On signal |

---

## 14. Three-Phase Rollout

### Phase 1 — Observe (~4 weeks)

- `TRADING_MODE=OBSERVE` — zero real trades
- All 9 agents run at full capacity
- Every signal logged to journal with full detail
- Review journal CSV every evening
- **Pass criterion:** Paper win rate > 55% over 4 weeks

### Phase 2 — Manual (~4–8 weeks)

- `TRADING_MODE=MANUAL` — you confirm each trade
- Dashboard shows CONFIRM/SKIP buttons per signal
- System pre-fills strike, SL, target
- You click to place — system monitors and alerts on exit
- **Pass criterion:** Real win rate > 55%, profit factor > 1.3

### Phase 3 — Auto (ongoing)

- `TRADING_MODE=AUTO` — fully automated
- System places, manages, and exits orders
- Daily review via dashboard
- ML model retrains monthly
- Daily loss kill switch always active

---

## 15. Adding a New Broker

**Step 1 — Create the broker file:**
```python
# broker/angelone_broker.py
from broker.base_broker import BaseBroker, OrderResult, PositionInfo

class AngelOneBroker(BaseBroker):
    @property
    def broker_name(self) -> str: return "angelone"
    def get_login_url(self) -> str: ...
    def generate_session(self, auth_code: str) -> str: ...
    # ... implement all BaseBroker methods
```

**Step 2 — Register in factory:**
```python
# broker/factory.py — add ONE line to BROKER_REGISTRY
BROKER_REGISTRY = {
    "yfinance": ("broker.yfinance_broker", "YFinanceBroker"),
    "upstox":   ("broker.upstox_broker",   "UpstoxBroker"),
    "groww":    ("broker.groww_broker",     "GrowwBroker"),
    "kite":     ("broker.kite_broker",      "KiteBroker"),
    "angelone": ("broker.angelone_broker",  "AngelOneBroker"),  # ← add this
}
```

**Step 3 — Switch:**
```bash
# .env
BROKER=angelone
```

```bash
docker-compose restart
```

Zero changes anywhere else.

---

## 16. Directory Structure

```
signalforge/
├── CLAUDE.md                    ← Claude Code project memory
├── main.py                      ← Entry point — wires all 9 agents
├── Dockerfile                   ← Docker build
├── docker-compose.yml           ← Container config + volume mounts
├── requirements.txt             ← Python dependencies
├── pytest.ini                   ← Test runner config
├── .env.example                 ← Credentials template
├── .gitignore
│
├── config/
│   └── settings.py              ← ALL config — never hardcode elsewhere
│
├── core/
│   ├── bus.py                   ← Async message bus
│   └── models.py                ← Shared dataclasses
│
├── broker/
│   ├── base_broker.py           ← Abstract interface
│   ├── factory.py               ← Registry-based broker loader
│   ├── yfinance_broker.py       ← Free testing
│   ├── upstox_broker.py         ← Free account
│   ├── groww_broker.py          ← Rs 499/month
│   └── kite_broker.py           ← Rs 500/month production
│
├── agents_code/
│   ├── agent1_data/             ← Data fetcher (fetcher.py + orb.py)
│   ├── agent2_strategy/         ← Runner + 5 strategy files
│   ├── agent3_ml/               ← ML filter + feature extractor
│   ├── agent4_planner/          ← Trade planner
│   ├── agent5_execution/        ← Execution (mode-aware)
│   ├── agent6_position/         ← Position manager
│   ├── agent7_analytics/        ← Journal + EOD reporter
│   ├── agent8_dashboard/        ← Flask-SocketIO + Telegram
│   └── agent9_regime/           ← Regime classifier (Gate 1)
│
├── utils/
│   ├── llm.py                   ← Async Claude API wrapper
│   └── option_utils.py          ← Premium estimation, expiry calc
│
├── scripts/
│   ├── kite_auth.py             ← Daily Kite token refresh
│   ├── upstox_auth.py           ← Daily Upstox token refresh
│   ├── groww_auth.py            ← Daily Groww auth (TOTP automated)
│   ├── download_history.py      ← Seed data cache via yfinance
│   ├── replay.py                ← Historical signal replay
│   ├── train_ml.py              ← ML model training
│   ├── debug_regime.py          ← Diagnose regime suppression
│   ├── health_check.py          ← System verification
│   └── quick_ref.py             ← Daily cheat sheet
│
├── dashboard/
│   └── templates/index.html     ← Live dashboard UI
│
├── data/
│   └── cache/                   ← Parquet data cache (persisted)
│
├── journal/                     ← Signal journal CSVs (daily)
├── logs/                        ← Rotating daily logs
├── ml/
│   ├── saved_models/            ← Trained .pkl model files
│   └── training/                ← Training scripts
├── backtesting/
│   └── results/                 ← Backtest output CSVs + JSONs
├── tests/
│   ├── unit/                    ← Per-agent unit tests
│   └── integration/             ← Full pipeline tests
├── docs/
│   ├── architecture.md          ← This file
│   ├── api-reference.md         ← Bus topic payload schemas
│   └── onboarding.md            ← How to run locally
├── skills/                      ← Claude Code skill files per agent
├── agents/                      ← Claude Code subagent definitions
└── arc/                         ← Archived old versions
```

---

*SignalForge v1.0 — NIFTY Options Signal Engine*
*Phase 1: OBSERVE mode — validate before trading*

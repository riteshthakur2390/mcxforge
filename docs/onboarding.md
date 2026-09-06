# MCXForge — Operator Onboarding & Daily Operations Guide

This guide is for setup and daily operations of **MCXForge** (Commodity Option Buying Platform: BUY CALL / BUY PUT ONLY, starting with **SILVERM**). For architecture and algorithmic details, refer to [Architecture Reference](architecture.md).

---

## Prerequisites

Before you begin, ensure your system meets these requirements:
- **Operating System**: macOS or Linux (Python 3.11+ virtual environment).
- **Broker Account**: **Dhan HQ** API access (Client ID + Permanent Access Token) for live MCX tick feeds, option chains, and order execution. Upstox and Kite integrations are also supported via `broker/factory.py`.
- **Cached Option Scrip Master**: Cached Dhan master list at `data/cache/dhan_scrip_master.csv` containing active SILVERM options contracts.
- **Historical Data**: Clean 5-minute continuous datasets located in `data/historical/` (e.g. `SILVERMIC_dhan_5m.csv`).

---

## First-Time Setup

```bash
# 1. Navigate to workspace
cd ~/Downloads/projects/mcxforge

# 2. Configure Environment Variables
# Copy credentials from your configuration or .env.example
cp .env.example .env

# Configure your Dhan tokens in .env:
# DHAN_CLIENT_ID="1100352..."
# DHAN_ACCESS_TOKEN="eyJ0eXAi..."
# TRADING_MODE="OBSERVE"  # Options: OBSERVE (paper), MANUAL (dashboard confirm), AUTO (live)

# 3. Verify Virtual Environment and Dependencies
source venv/bin/activate
pip install -r requirements.txt

# 4. Verify Historical Data Freshness
ls -lh data/historical/SILVERMIC_dhan_5m.csv
```

---

## Daily Operational Routine

MCX Commodity trading operates across extended market hours (09:00 to 23:30/23:55 IST). Because empirical research demonstrates that the **Evening Session (17:00–23:30 IST)** captures $>85\%$ of institutional liquidity from US COMEX overlap, the system is optimized for Evening execution.

### Session Schedule

| Time (IST) | Market Phase | Action |
|---|---|---|
| **08:30 – 09:00** | Pre-Market Checks | Verify Dhan HQ token, check broker margins, ensure zero stale overnight state. |
| **09:00 – 13:00** | Asian / Morning Session | Discovery & ORB formation. (Thin liquidity; live trading suppressed by default). |
| **13:00 – 17:00** | European / Afternoon Session | London open crossover. Volatility builds. |
| **17:00 – 23:30** | US COMEX / Evening Session | **Active Live Execution**. Run Golden Ensemble on 1-Hour candles. |
| **23:25** | Intraday Square-off Cutoff | Mandatory flat exit of any remaining intraday positions before 23:30 close. |
| **23:35** | Post-Market EOD | Generate daily performance ledger and persist journal logs. |

---

## Running MCXForge

### 1. Launch Live Commodity Option Buying Paper Runner

To stream live Dhan HQ market feeds, evaluate the multi-strategy ensemble, resolve active SILVERM ATM/near-ATM options contracts, and simulate fills with exact statutory transaction charges:

```bash
# Run Evening-only paper trading on SILVERM options
./venv/bin/python scripts/live_commodity_paper_runner.py \
  --instrument SILVERM \
  --timeframe 5m \
  --session-filter EVENING_ONLY

# Or run full-day observation:
./venv/bin/python scripts/live_commodity_paper_runner.py \
  --instrument SILVERM \
  --timeframe 5m \
  --session-filter ALL
```

### 2. Monitor Live Journal & Telemetry

Active trades, entry/exit fills, trailing stops, and MTM are continuously logged to:
- Real-time JSON State: `state/live_paper_journal.json`
- Human Readable Log: `logs/commodity_paper_runner.log`
- Daily Signal CSV: `journal/commodity_signals_YYYY-MM-DD.csv`

### 3. Run Strategy & Ensemble Backtests

To run the full institutional backtest suite on real historical Dhan data:

```bash
# Run standalone 19-strategy audit on 1h timeframe
./venv/bin/python scripts/run_commodity_backtest.py \
  --timeframe 1h \
  --session-filter EVENING_ONLY

# Run 30-minute sensitivity backtest
./venv/bin/python scripts/run_commodity_backtest.py \
  --timeframe 30m \
  --session-filter EVENING_ONLY
```

---

## Trading Modes

Set in `.env` via `TRADING_MODE`:

| Mode | Behavior | Safety |
|---|---|---|
| `OBSERVE` | **Default**. Subscribes to live market data, evaluates ensemble, records simulated fills with slippage and statutory costs. No broker orders sent. | 100% Safe (Recommended) |
| `MANUAL` | Evaluates ensemble, generates trade plan, and pushes confirmation modal to Dashboard/Telegram. Requires operator approval within 180s. | Human-in-the-loop |
| `AUTO` | Fully autonomous order routing to Dhan HQ API for qualified signals passing all risk and physical delivery filters. | Unattended live execution |

---

## Risk Governance & Safety Tripwires

1. **Compulsory Physical Delivery Tender-Period Lockout**:
   - MCX contracts mandate physical delivery during the final 5 trading days.
   - The engine automatically vetoes entries and enforces mandatory square-off before tender periods commence.
2. **Statutory MCX Transaction Fee Model**:
   - Every simulated trade accounts for:
     - Brokerage: ₹20 flat
     - Exchange Turnover Fee: 0.0021%
     - Commodity Transaction Tax (CTT): 0.01% on sell-side
     - SEBI Turnover Fee: ₹10 / crore
     - GST: 18% on (Brokerage + Exchange + SEBI)
     - Stamp Duty: 0.002% on buy-side
3. **Macro Currency Circuit-Breaker**:
   - `CurrencyMacroFilter` monitors USD-INR 1-day rate of change. If $|USDINR| > 0.5\%$, trade sizing is automatically halved to protect against currency whipsaws.

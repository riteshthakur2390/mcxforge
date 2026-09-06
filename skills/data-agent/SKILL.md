# SKILL: Data Fetcher Agent (Agent 1)

## Purpose
Owns ALL market data ingestion for MCXForge. Streams live ticks, manages WebSocket connections, aggregates multi-timeframe candles (5m, 15m, 30m, 1h, 1d), and provides real-time option contract lookup for **Commodity Option Buying**, starting with **SILVERM**.

## Key Files
- `agents_code/agent1_data/fetcher.py` — Main market data fetcher & broker feed interface
- `broker/dhan_broker.py` — Dhan HQ MCX option master fetcher and contract resolver
- `data/cache/dhan_scrip_master.csv` — Cached MCX options scrip master (14,802 options)
- `data/historical/SILVERMIC_dhan_5m.csv` — Primary Dhan historical dataset (22,330 bars)

## Market Hours & Sessions (IST)
- **Market Open**: 09:00 IST
- **Asian / Morning Session**: 09:00–13:00 IST (Price discovery & ORB formation)
- **European / Afternoon Session**: 13:00–17:00 IST (London overlap)
- **US COMEX / Evening Session**: 17:00–23:30 IST (US trading overlap — captures >85% of institutional liquidity)
- **Market Close**: 23:30 IST (or 23:55 IST during US Daylight Saving Time)

## Bus Topics Published
- `PREMARKET_BIAS` — Macro bias, COMEX overnight changes, DXY status, USD-INR rate, opening gap %
- `CANDLES_READY` — Normalized OHLCV candles, timeframe, session tag, LTP, ORB levels
- `ORB_FORMED` — 09:30 locked ORB High, Low, and range
- `TICK_UPDATE` — Real-time Dhan HQ WebSocket LTP and market depth

## Dhan HQ Broker Ingestion Details
- **Exchange Segment**: `MCX_COMM`
- **Underlying Focus**: `SILVERM` (5 kg lot, tick size ₹1.0)
- **Option Chain / Scrip Master**: Filters Dhan scrip master specifically for `SM_SYMBOL_NAME == "SILVERM"`, returning strikes in step 500
- **Permanent Access Token**: Dhan tokens are permanent unless manually regenerated on web portal
- **Cached LTP Fallback**: Implements local caching to withstand transient DNS or network hiccups

## Data Invariants
- `High >= max(Open, Close)` and `Low <= min(Open, Close)`
- Zero or negative volume is validated and flagged
- Strictly rejects future lookahead bars during historical replay

# SKILL: Dashboard & Alert Agent (Agent 8)

## Purpose
Operator interface and notification hub for MCXForge. Serves real-time Flask-SocketIO dashboard and dispatches critical Telegram alerts for **Commodity Option Buying (BUY CALL / BUY PUT ONLY)**, starting with **SILVERM**.
Receives all bus events and broadcasts updates to UI and mobile endpoints.

## Key Files
- `agents_code/agent8_dashboard/app.py` — Flask + SocketIO server
- `agents_code/agent8_dashboard/alerts.py` — Telegram bot alert dispatcher
- `utils/telegram_notifier.py` — Structured Telegram notifier with rich cards

## Dashboard URL
`http://localhost:5050`

## Key UI Components
- **Commodity LTP Strip**: Real-time tick price for `SILVERM` (5 kg lot), `GOLD`, `CRUDEOIL` from Dhan HQ feed
- **Market Session Badge**: Active session indicator (`MORNING`, `AFTERNOON`, `EVENING - HIGH LIQUIDITY`)
- **Regime & Tender Warning**: Market state (`TRENDING`, `RANGING`, `CHOPPY`) and tender lockout status
- **Option Strategy Consensus Grid**: Real-time vote tallies, individual strategy direction, and confidence bars
- **Active Option Trade Card**: Real-time option premium MTM, current trailing stop, dynamic target, and point gain
- **Statutory Costs Telemetry**: Cumulative transaction taxes deducted on option turnover
- **Manual Execution Modal**: 180-second countdown prompt to approve/reject signals in `TRADING_MODE=MANUAL`

## WebSocket Events Broadcasted
`commodity_ltp_update`, `session_change`, `consensus_signal`, `order_update`, `position_update`, `position_closed`, `risk_halt_alert`

## Telegram Notifications
Sent for:
- Session open/close notifications
- Approved consensus signals (votes $\ge 2$)
- Live order fills and trailing stop updates
- Intraday square-off executions (23:25 IST)
- Risk limit tripwires or broker feed disconnects

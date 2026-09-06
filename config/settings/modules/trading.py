import os
from config.settings.utils import *


# ── TRADING MODE ──
# OBSERVE → signals shown, nothing executed, paper journal logged (Default for 4-month live testing)
# MANUAL  → signals shown, you click confirm to place order
# AUTO    → signals placed automatically
TRADING_MODE = os.getenv("TRADING_MODE", "OBSERVE").strip().upper()

# ── STRATEGY CONTEXT MODE ──
# PNL_MAXIMIZER_V1     → Production Execution (Anchors, deduplication)
# LEGACY_CONTEXT       → Shadow observation only
# ROLLING_5M_CONTEXT   → Shadow observation only
STRATEGY_CONTEXT_MODE = os.getenv("STRATEGY_CONTEXT_MODE", "PNL_MAXIMIZER_V1").strip().upper()

# ── MCX COMMODITY INSTRUMENT ──
MCX_EXCHANGE          = "MCX"
COMMODITY_SYMBOL      = os.getenv("COMMODITY_SYMBOL", "SILVERM")
SUPPORTED_COMMODITIES = ("SILVERM", "SILVERMIC", "SILVER", "GOLD", "GOLDM", "CRUDEOIL", "CRUDEOILM", "NATURALGAS", "NATGASMINI")

# Legacy symbol aliases for compatibility
NIFTY_INDEX_SYMBOL    = os.getenv("INSTRUMENT", "SILVERM")
NIFTY_OPTION_EXCHANGE = "MCX"
NIFTY_LOT_SIZE        = 5  # 1 lot SILVERM = 5 kg
NIFTY_STRIKE_STEP     = 500
SENSEX_INDEX_SYMBOL   = "MCX:GOLD"
SENSEX_OPTION_EXCHANGE = "MCX"
SENSEX_LOT_SIZE       = 1
SENSEX_STRIKE_STEP    = 100
SUPPORTED_INDEX_SYMBOLS = SUPPORTED_COMMODITIES

# ── MCX MARKET TIMING (IST) ──
# Morning session: 09:00 - 17:00 IST
# Evening session: 17:00 - 23:30 IST (23:55 in winter DST)
SYSTEM_START_TIME    = os.getenv("SYSTEM_START_TIME", "08:45")
MARKET_OPEN_TIME     = os.getenv("MARKET_OPEN_TIME", "09:00")
ORB_END_TIME         = os.getenv("ORB_END_TIME", "09:30")
SIGNAL_START_TIME    = os.getenv("SIGNAL_START_TIME", "09:30")

# Stop new intraday entries at 23:00 IST
NO_NEW_SIGNAL_AFTER  = os.getenv("NO_NEW_SIGNAL_AFTER", "23:00")

# Intraday futures position auto-exit before broker MIS square-off (~23:20 IST)
EOD_SQUARE_OFF_TIME  = os.getenv("EOD_SQUARE_OFF_TIME", "23:15")
MARKET_CLOSE_TIME    = os.getenv("MARKET_CLOSE_TIME", "23:30")
EOD_REPORT_TIME      = os.getenv("EOD_REPORT_TIME", "23:45")

# MCX does not have NSE SEBI CAS auction session; set off-market dummy window for compatibility
CAS_START_TIME       = "23:58"
CAS_END_TIME         = "23:59"
FNO_CLOSE_TIME       = "23:30"

# ── MANUAL MODE ──
MANUAL_CONFIRM_TIMEOUT = 180        # seconds before signal card expires

# ── SENSEX INSTRUMENT ──
# BSE SENSEX 30 — options expire on FRIDAY (vs NIFTY which expires Tuesday)
# Use SENSEX on Tuesday to avoid NIFTY expiry gamma risk
# SENSEX and NIFTY have ~0.97 correlation so direction signal is same
SENSEX_INDEX_SYMBOL = "BSE:SENSEX"
SENSEX_OPTION_EXCHANGE = "BFO"  # BSE F&O segment
SENSEX_LOT_SIZE = 10  # 10 units per lot (vs NIFTY 75)
SENSEX_STRIKE_STEP = 100  # 100-pt strike spacing (vs NIFTY 50)
SENSEX_BSE_SECURITY_ID = "51"  # Dhan BSE_INDEX security_id for SENSEX
SENSEX_UPSTOX_KEY = "BSE_INDEX|SENSEX"
SENSEX_DHAN_SEGMENT = "BSE_FNO"  # Dhan BSE F&O segment for options
SENSEX_UPSTOX_SEGMENT = "BFO"  # Upstox BSE F&O segment

# ── INSTRUMENT SELECTION LOGIC ──
# Tuesday  = NIFTY expiry day. Switch to SENSEX to avoid gamma risk.
# Friday   = SENSEX expiry day. Can trade SENSEX for expiry theta play.
# Other    = Trade NIFTY (default)
INSTRUMENT_THURSDAY = "NIFTY"
INSTRUMENT_FRIDAY = "NIFTY"  # SENSEX expiry day — trade SENSEX gamma
INSTRUMENT_DEFAULT = "NIFTY"  # Mon/Tue/Wed → NIFTY

# Minimum premium floors (lower for SENSEX since it has fewer retail participants)
SENSEX_MIN_PREMIUM = 50.0  # ₹50 min premium (vs NIFTY ₹20)
SENSEX_MAX_PREMIUM = 800.0  # ₹800 max (ATM on high-vol days)

import os
from config.settings.utils import *


# ── IDENTITY ──
APP_NAME   = "MCXForge"
VERSION    = "1.0.0"
INSTRUMENT = os.getenv("INSTRUMENT", "SILVERM")
COMMODITY  = os.getenv("COMMODITY", "SILVERM")

# ── ZERODHA KITE ──
KITE_API_KEY      = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET   = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

# ── LLM ROUTING ──
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_NON_SENSITIVE_PROVIDER = os.getenv(
    "LLM_NON_SENSITIVE_PROVIDER", LLM_PROVIDER
).strip().lower()
LLM_SENSITIVE_PROVIDER = os.getenv(
    "LLM_SENSITIVE_PROVIDER", LLM_PROVIDER
).strip().lower()
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "200"))
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "20"))
LLM_ENABLED = _flag("LLM_ENABLED", "true")
MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", 3))
RETRY_DELAY = int(os.getenv("OLLAMA_RETRY_DELAY", 2))
LLM_CONTEXT_MIN_RANK = float(os.getenv("LLM_CONTEXT_MIN_RANK", "0.50"))
LLM_CONTEXT_MIN_PROB = float(os.getenv("LLM_CONTEXT_MIN_PROB", "0.62"))
LLM_CONTEXT_TIMEOUT_SEC = float(os.getenv("LLM_CONTEXT_TIMEOUT_SEC", "8"))
LLM_CONTEXT_CACHE_TTL_SEC = int(os.getenv("LLM_CONTEXT_CACHE_TTL_SEC", "1800"))

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
OLLAMA_FAST_MODEL = os.getenv("OLLAMA_FAST_MODEL", "mistral:latest")
OLLAMA_SLOW_TASK_TIMEOUT = float(os.getenv("OLLAMA_SLOW_TASK_TIMEOUT", "300"))

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

# Backward-compatible alias for existing imports/logging
LLM_MODEL = os.getenv("LLM_MODEL", OLLAMA_MODEL if LLM_PROVIDER == "ollama" else (
    DEEPSEEK_MODEL if LLM_PROVIDER == "deepseek" else (
        OPENAI_MODEL if LLM_PROVIDER == "openai" else ANTHROPIC_MODEL
    )
))

# ── NOTIFICATIONS ──
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")
LIVE_TELEGRAM_BOT_TOKEN = os.getenv("LIVE_TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN)
LIVE_TELEGRAM_CHAT_ID   = os.getenv("LIVE_TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID)
BACKTEST_TELEGRAM_BOT_TOKEN = os.getenv("BACKTEST_TELEGRAM_BOT_TOKEN", "")
BACKTEST_TELEGRAM_CHAT_ID   = os.getenv("BACKTEST_TELEGRAM_CHAT_ID", "")
SOUND_ALERT_ENABLED  = True
LOG_LEVEL            = os.getenv("LOG_LEVEL", "DEBUG").strip().upper()

# ── PATHS ──
DATA_CACHE_DIR       = "data/cache"
DATA_HIST_DIR        = "data/historical"
DATA_HIST_DB_PATH    = os.getenv(
    "DATA_HIST_DB_PATH",
    os.path.join(DATA_HIST_DIR, "market_history.sqlite3"),
)
RECENT_MARKET_WINDOW_DAYS = int(os.getenv("RECENT_MARKET_WINDOW_DAYS", "90"))
HISTORICAL_ARCHIVE_INTERVALS = _parse_csv_tuple(
    os.getenv("HISTORICAL_ARCHIVE_INTERVALS", "5minute,day"),
    ("5minute", "day"),
)
BACKTEST_DATA_SOURCE_POLICY = os.getenv(
    "BACKTEST_DATA_SOURCE_POLICY",
    "MIXED",
).strip().upper()
JOURNAL_DIR          = "journal"
LOGS_DIR             = "logs"
RESULTS_DIR          = "backtesting/results"
DASHBOARD_PORT       = int(os.getenv("DASHBOARD_PORT", "5054"))

# ── BROKER SELECTION ──
# Which broker to use:
#   dhan   → Dhan HQ API (default for MCXForge, 5-year historical backtest data)
#   upstox → Upstox API
#   kite   → Zerodha KiteConnect (production)
BROKER = os.getenv("BROKER", "dhan")

# ── UPSTOX API ──
UPSTOX_API_KEY      = os.getenv("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET   = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")

# ── DHAN ──
# Generate from https://web.dhan.co, then renew before expiry via scripts/dhan_renew_tokens.py.
DHAN_CLIENT_ID    = os.getenv("DHAN_CLIENT_ID",    "")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")
DHAN_CLIENT_ID_2    = os.getenv("DHAN_CLIENT_ID_2",    "")
DHAN_ACCESS_TOKEN_2 = os.getenv("DHAN_ACCESS_TOKEN_2", "")
DHAN_CLIENT_ID_3    = os.getenv("DHAN_CLIENT_ID_3",    "")
DHAN_ACCESS_TOKEN_3 = os.getenv("DHAN_ACCESS_TOKEN_3", "")
DHAN_ORDER_EXECUTION_ENABLED = _flag("DHAN_ORDER_EXECUTION_ENABLED", "false")
DISABLE_PRIMARY_EXECUTION = _flag("DISABLE_PRIMARY_EXECUTION", "true")

# ── GROWW TRADE API ──
GROWW_API_KEY      = os.getenv("GROWW_API_KEY", "")
GROWW_API_SECRET   = os.getenv("GROWW_API_SECRET", "")
GROWW_TOTP_SECRET  = os.getenv("GROWW_TOTP_SECRET", "")
GROWW_ACCESS_TOKEN = os.getenv("GROWW_ACCESS_TOKEN", "")

# ── TELEGRAM NOTIFICATIONS ──
# Get token: https://t.me/BotFather  |  chat_id: https://t.me/userinfobot
TELEGRAM_ENABLED = bool(
    (LIVE_TELEGRAM_BOT_TOKEN and LIVE_TELEGRAM_CHAT_ID)
    or (BACKTEST_TELEGRAM_BOT_TOKEN and BACKTEST_TELEGRAM_CHAT_ID)
)

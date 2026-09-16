"""
main.py — SignalForge Entry Point
===================================
Wires all 9 agents to the message bus and runs the system.

Usage:
    python main.py                    # default OBSERVE mode (from .env)
    TRADING_MODE=MANUAL python main.py
    TRADING_MODE=AUTO   python main.py

System timeline:
    09:00  →  boots, pre-market brief
    09:15  →  candles start
    09:30  →  ORB formed, signals activate
    15:00  →  no new signals
    15:30  →  EOD report, system idles
"""

import asyncio, warnings
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", message=".*find_spec\\(\\).not found; falling back to find_module\\(\\).*")
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
warnings.filterwarnings("ignore", message=r"(?s).*If you are loading a serialized model.*")
warnings.filterwarnings("ignore", message=r"(?s).*Trying to unpickle estimator.*")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
import os
import signal
import sys
from datetime import datetime
import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")


def _log_format(record: dict) -> str:
    ts = record["time"].astimezone(IST).strftime("%H:%M:%S.%f")[:-3]
    prefix = (
        f"{ts} | {record['level'].name:<8} | "
        f"{record['name']}:{record['function']}:{record['line']} - "
    )
    return prefix + record["message"] + "\n"


def _strategy_log_format(record: dict) -> str:
    ts = record["time"].astimezone(IST).strftime("%H:%M:%S.%f")[:-3]
    return ts + " | " + record["message"] + "\n"


def _is_strategy_record(record: dict) -> bool:
    return bool(record["extra"].get("strategy_exec"))


def _is_pipeline_trace_record(record: dict) -> bool:
    return bool(record["extra"].get("pipeline_trace"))

# ── Logging setup ──────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "DEBUG").strip().upper(),
    format=_log_format,
    filter=lambda record: not _is_strategy_record(record) and not _is_pipeline_trace_record(record),
)
logger.add(
    "logs/mcxforge_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days",
    level=os.getenv("LOG_LEVEL", "DEBUG").strip().upper(),
    format=_log_format,
    filter=lambda record: not _is_strategy_record(record) and not _is_pipeline_trace_record(record),
)
logger.add(
    "logs/strategies_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days",
    level="INFO",
    format=_strategy_log_format,
    filter=_is_strategy_record,
)
logger.add(
    "logs/pipeline_trace_{time:YYYY-MM-DD}.log",
    rotation="1 day", retention="30 days",
    level="INFO",
    format=_strategy_log_format,
    filter=_is_pipeline_trace_record,
)

from config.settings import (
    APP_NAME, VERSION, TRADING_MODE,
    SYSTEM_START_TIME, EOD_REPORT_TIME, DASHBOARD_PORT, LOG_LEVEL,
)
from core.llm_router import get_llm_status

# ── Agent imports ───────────────────────────────────────────────────────────
from agents_code.agent1_data.fetcher       import DataFetcherAgent
from agents_code.agent2_strategy.runner    import StrategyAgent
from agents_code.agent2_strategy.market_context_gate import get_market_context_gate
from agents_code.agent3_ml.filter          import MLFilterAgent
from agents_code.agent4_planner.planner    import TradePlannerAgent
from agents_code.agent5_execution.executor import ExecutionAgent
from agents_code.agent6_position.manager   import PositionManagerAgent
from agents_code.agent7_analytics.journal  import AnalyticsAgent
from agents_code.agent8_dashboard.app      import DashboardAlertAgent
from agents_code.agent9_regime.classifier  import MarketRegimeAgent
from agents_code.agent10_risk.risk_guard   import RiskGuardAgent
from agents_code.agent11_shadow.shadow_agent import ShadowParameterAgent
from agents_code.agent12_lifecycle.lifecycle_auditor import TradeLifecycleAuditor
from utils.capital_manager import get_capital_manager
from utils.hot_reload import HotReloader
from utils.system_health import HealthServer
from utils.audit_trail import get_audit
from utils.live_position_reconciler import LivePositionReconciler
from utils.market_calendar import is_trading_day, nse_holiday_name

async def main() -> None:
    llm_status = get_llm_status()
    audit = get_audit()
    audit.log_system_event("STARTUP", {"mode": TRADING_MODE, "version": VERSION})
    logger.info("=" * 60)
    logger.info(f"  {APP_NAME} v{VERSION}")
    logger.info(f"  Mode:      {TRADING_MODE}")
    logger.info(f"  Dashboard: http://localhost:{DASHBOARD_PORT}")
    logger.info(f"  Start:     {SYSTEM_START_TIME} IST")
    logger.info(
        f"  LLM:       "
        f"{'ENABLED' if llm_status.get('enabled') else 'DISABLED'} | "
        f"{llm_status.get('active_provider')} | {llm_status.get('active_model')}"
    )
    logger.info(f"  Log level: {LOG_LEVEL}")
    logger.info("=" * 60)

    # ── Instantiate all 12 agents ───────────────────────────────────────────
    from core.bus import get_bus
    bus = get_bus()

    data_agent      = DataFetcherAgent()
    regime_agent    = MarketRegimeAgent()
    strategy_agent  = StrategyAgent()
    strategy_agent.set_oi_recorder(data_agent.get_oi_recorder())
    market_context_gate = get_market_context_gate(bus=bus)
    ml_agent        = MLFilterAgent()
    planner_agent   = TradePlannerAgent(data_agent=data_agent)
    execution_agent = ExecutionAgent()
    position_agent  = PositionManagerAgent(data_agent=data_agent)
    analytics_agent = AnalyticsAgent()
    risk_agent      = RiskGuardAgent()
    dashboard_agent = DashboardAlertAgent(
        position_agent=position_agent,
        analytics_agent=analytics_agent,
        risk_agent=risk_agent,
        ml_agent=ml_agent,
        data_agent=data_agent,
        strategy_agent=strategy_agent,
    )
    shadow_agent    = ShadowParameterAgent(bus=bus)
    lifecycle_agent = TradeLifecycleAuditor(bus=bus)
    capital_manager = get_capital_manager()

    # ── Register all on the message bus ───────────────────────────────────
    # Order doesn't matter — all communicate via async bus
    regime_agent.register()
    strategy_agent.register()
    market_context_gate.register()
    ml_agent.register()
    planner_agent.register()
    execution_agent.register()
    position_agent.register()
    analytics_agent.register()
    risk_agent.register()
    dashboard_agent.register()
    await shadow_agent.register()
    await lifecycle_agent.register()

    # ── Start dashboard (background thread — non-blocking) ────────────────
    dashboard_agent.start_dashboard()
    health_server = HealthServer(
        port=int(os.getenv("HEALTH_PORT", "8080")),
        capital_manager=capital_manager,
        position_manager=position_agent,
        broker=data_agent.broker,
    )
    await health_server.start()
    hot_reloader = HotReloader()
    await hot_reloader.start()
    position_reconciler = LivePositionReconciler(
        broker=data_agent.broker,
        position_manager=position_agent,
    )
    await position_reconciler.start()

    # ── Graceful shutdown ─────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    stop = asyncio.Event()

    def _shutdown(*_) -> None:
        logger.info("Shutdown signal received.")
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown)

    # ── Wait for system start time ────────────────────────────────────────
    now = datetime.now(IST).strftime("%H:%M")
    if now < SYSTEM_START_TIME:
        secs = _seconds_until(SYSTEM_START_TIME)
        logger.info(f"Waiting {secs // 60}m until {SYSTEM_START_TIME} IST...")
        await asyncio.sleep(secs)

    # ── Start data agent (drives everything via CANDLES_READY) ────────────
    data_task = asyncio.create_task(data_agent.start())
    eod_task  = asyncio.create_task(_eod_scheduler(analytics_agent, stop))

    logger.info(f"MCXForge running. Dashboard → http://localhost:{DASHBOARD_PORT}")

    # ── Run until shutdown ────────────────────────────────────────────────
    await stop.wait()

    logger.info("Stopping MCXForge...")
    audit.log_system_event("SHUTDOWN_REQUESTED", {"mode": TRADING_MODE})
    try:
        await position_agent.force_eod_exit()
    except Exception as exc:
        logger.error(f"Graceful position close failed: {exc}")
        audit.log_system_event("SHUTDOWN_POSITION_CLOSE_FAILED", {"error": str(exc)})
    await hot_reloader.stop()
    await position_reconciler.stop()
    data_agent.stop()
    data_task.cancel()
    eod_task.cancel()
    audit.log_system_event("STOPPED", {"mode": TRADING_MODE})
    logger.info("Stopped. Goodbye.")


async def _eod_scheduler(analytics: AnalyticsAgent, stop: asyncio.Event) -> None:
    """Generate EOD report at EOD_REPORT_TIME then idle."""
    triggered_for: str | None = None
    while not stop.is_set():
        now_dt = datetime.now(IST)
        today = now_dt.date()
        today_iso = today.isoformat()
        now = now_dt.strftime("%H:%M")
        analytics_day = str(getattr(analytics, "_today", "") or today_iso)
        if not is_trading_day(today):
            if triggered_for != today_iso:
                logger.info(f"Skipping EOD report on non-trading day: {today_iso} ({nse_holiday_name(today)})")
                triggered_for = today_iso
        elif analytics_day != today_iso:
            if triggered_for != today_iso:
                logger.warning(
                    f"Skipping EOD report because analytics date is stale: "
                    f"analytics={analytics_day} today={today_iso}"
                )
                triggered_for = today_iso
        elif triggered_for != today_iso and now >= EOD_REPORT_TIME:
            logger.info("Generating EOD report...")
            await analytics.generate_eod_report()
            triggered_for = today_iso
        await asyncio.sleep(30)


def _seconds_until(target: str) -> int:
    now = datetime.now(IST)
    h, m = map(int, target.split(":"))
    t    = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return max(0, int((t - now).total_seconds()))


if __name__ == "__main__":
    asyncio.run(main())

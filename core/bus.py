"""
core/bus.py — SignalForge Message Bus
======================================
Async pub/sub backbone. ALL 9 agents communicate ONLY through here.
No agent imports another agent directly — ever.

Usage:
    bus = get_bus()
    bus.subscribe(Topic.CANDLES_READY, my_async_handler)
    await bus.publish(Topic.CANDLES_READY, payload, source="Agent1")
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Awaitable, Union
from loguru import logger
import pytz

IST = pytz.timezone("Asia/Kolkata")


# ── ALL VALID BUS TOPICS ──────────────────────────────────────────────────────
class Topic:
    # Data layer
    PREMARKET_BIAS    = "PREMARKET_BIAS"
    CANDLES_READY     = "CANDLES_READY"
    ORB_FORMED        = "ORB_FORMED"
    TICK_UPDATE       = "TICK_UPDATE"

    # Regime layer
    MARKET_REGIME     = "MARKET_REGIME"
    SIGNAL_SUPPRESSED = "SIGNAL_SUPPRESSED"

    # Strategy layer
    RAW_SIGNAL        = "RAW_SIGNAL"

    # ML layer
    SIGNAL_APPROVED   = "SIGNAL_APPROVED"
    SIGNAL_REJECTED   = "SIGNAL_REJECTED"
    ML_FILTERED_SIGNAL = "SIGNAL_APPROVED"
    MODEL_RETRAINED   = "MODEL_RETRAINED"

    # Planning layer
    TRADE_PLAN_READY  = "TRADE_PLAN_READY"

    # Execution layer
    ORDER_DRY_RUN     = "ORDER_DRY_RUN"
    ORDER_CONFIRM_REQ = "ORDER_CONFIRM_REQ"
    ORDER_PLACED      = "ORDER_PLACED"
    ORDER_SKIPPED     = "ORDER_SKIPPED"

    # Position layer
    POSITION_UPDATE   = "POSITION_UPDATE"
    POSITION_CLOSED   = "POSITION_CLOSED"

    # Analytics layer
    EOD_REPORT_READY  = "EOD_REPORT_READY"

    # Shadow/Audit layer (Agent 11 + 12 — observer agents)
    SHADOW_REPORT_READY    = "SHADOW_REPORT_READY"
    LIFECYCLE_REPORT_READY = "LIFECYCLE_REPORT_READY"
    # System
    ALERT             = "ALERT"
    SYSTEM_STATUS     = "SYSTEM_STATUS"
    SYSTEM_RESET      = "SYSTEM_RESET"   # CRITICAL FIX: Agents clear internal state on reset
    CONFIG_RELOADED   = "CONFIG_RELOADED"


@dataclass
class Message:
    topic:     str
    payload:   dict
    source:    str
    timestamp: str = field(
        default_factory=lambda: datetime.now(IST).isoformat()
    )

    def to_dict(self) -> dict:
        return {"topic": self.topic, "payload": self.payload, "source": self.source, "timestamp": self.timestamp}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ── MESSAGE BUS ───────────────────────────────────────────────────────────────
class MessageBus:
    """
    Lightweight async pub/sub bus using asyncio.
    Handlers must be async coroutines.
    All messages logged at DEBUG level for full observability.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._history: list[Message] = []
        self._max_history = 1000
        self._stats: dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, handler: Callable[..., Awaitable]) -> None:
        """Register an async handler for a topic."""
        self._subscribers[topic].append(handler)
        logger.debug(
            f"[BUS] Subscribed: {handler.__qualname__} → {topic}"
        )

    def subscribe_many(self, topics: list[str], handler: Callable) -> None:
        for topic in topics:
            self.subscribe(topic, handler)

    @staticmethod
    def _message_timestamp(payload: dict) -> str:
        raw = payload.get("timestamp")
        if not raw:
            return datetime.now(IST).isoformat()
        try:
            ts = datetime.fromisoformat(str(raw))
            if ts.tzinfo is None:
                ts = IST.localize(ts)
            else:
                ts = ts.astimezone(IST)
            return ts.isoformat()
        except Exception:
            return str(raw)

    @staticmethod
    def _market_ts_suffix(payload: dict) -> str:
        if not payload.get("timestamp"):
            return ""
        normalized = MessageBus._message_timestamp(payload)
        try:
            ts = datetime.fromisoformat(normalized)
            return f" | market_ts={ts.strftime('%Y-%m-%d %H:%M IST')}"
        except Exception:
            return f" | market_ts={normalized}"

    async def publish(
        self,
        topic:   str,
        payload: dict,
        source:  str = "system"
    ) -> None:
        """
        Publish a message. All subscribers are called concurrently.
        Errors in individual handlers are caught and logged — they never
        crash the publisher.
        """
        msg = Message(
            topic=topic,
            payload=payload,
            source=source,
            timestamp=self._message_timestamp(payload),
        )

        # History ring buffer
        self._history.append(msg)
        if len(self._history) > self._max_history:
            self._history.pop(0)
        self._stats[topic] += 1

        handlers = self._subscribers.get(topic, [])
        market_ts_suffix = self._market_ts_suffix(payload)
        if not handlers:
            logger.debug(f"[BUS] {topic} — no subscribers{market_ts_suffix}")
            return

        logger.debug(
            f"[BUS] {source} → {topic} | {len(handlers)} handler(s){market_ts_suffix}"
        )

        results = await asyncio.gather(
            *[h(msg) for h in handlers],
            return_exceptions=True
        )
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                handler_name = getattr(handlers[i], "__qualname__", str(i))
                logger.error(
                    f"[BUS] Handler error | topic={topic} "
                    f"handler={handler_name} | {type(r).__name__}: {r}"
                )

    def get_history(
        self, topic: Union[str, None] = None, limit: int = 50
    ) -> list[dict]:
        msgs = (
            self._history if not topic
            else [m for m in self._history if m.topic == topic]
        )
        return [m.to_dict() for m in msgs[-limit:]]

    def get_stats(self) -> dict:
        return {
            "total_messages": sum(self._stats.values()),
            "by_topic":       dict(self._stats),
            "subscribers":    {t: len(h) for t, h in self._subscribers.items()},
        }

    def clear_history(self) -> None:
        self._history.clear()


# ── SINGLETON ─────────────────────────────────────────────────────────────────
_bus: Union[MessageBus, None] = None


def get_bus() -> MessageBus:
    global _bus
    if _bus is None:
        _bus = MessageBus()
        logger.info("[BUS] MessageBus initialized")
    return _bus


def reset_bus(notify_agents: bool = True) -> None:
    """Reset the bus and optionally notify agents to clear internal state.
    
    CRITICAL FIX: When notify_agents=True, publishes SYSTEM_RESET so agents
    can clear caches (_prefetched_features, _last_signal_*, etc.) preventing
    stale data from affecting new backtest runs.
    """
    global _bus
    if _bus and notify_agents:
        # Notify all agents to reset their internal state
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(_bus.publish(Topic.SYSTEM_RESET, {}, "system"))
        except RuntimeError:
            pass
    _bus = None


def reset_bus_sync() -> None:
    """Synchronous version for non-async contexts."""
    global _bus
    _bus = None

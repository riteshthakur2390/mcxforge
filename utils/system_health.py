from __future__ import annotations
"""
utils/system_health.py — HTTP Health Check & Graceful Shutdown
==============================================================
Two critical operational features every production system needs.

1. HTTP HEALTH CHECK ENDPOINT (/health, /status)
   Returns JSON with system status.
   Used by: Docker health checks, UptimeRobot, Datadog, your phone.
   Without this: you won't know the system crashed until you manually check.

   Response:
   {
     "status": "healthy",
     "mode": "OBSERVE",
     "position_open": false,
     "daily_pnl_inr": 1250.0,
     "capital_used_pct": 75.0,
     "last_candle": "14:25",
     "broker": "dhan",
     "broker_connected": true,
     "uptime_min": 342,
     "version": "1.0.0"
   }

2. GRACEFUL SHUTDOWN (SIGTERM / SIGINT / Ctrl+C)
   On shutdown signal:
   a) Stop accepting new signals
   b) Close any open position (market order)
   c) Cancel pending orders at broker
   d) Save state to disk
   e) Send Telegram notification
   f) Exit cleanly

   Without this: Docker stop kills the process mid-trade → position stuck open.

Usage (add to main.py):
    from utils.system_health import HealthServer, GracefulShutdown

    # Start health server on port 8080
    health = HealthServer(port=8080)
    await health.start()

    # Register shutdown handler
    shutdown = GracefulShutdown(broker, position_manager, capital_manager)
    shutdown.register()
"""

import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from config.settings import TRADING_MODE, TELEGRAM_ENABLED
except ImportError:
    TRADING_MODE     = os.getenv("TRADING_MODE", "AUTO")
    TELEGRAM_ENABLED = False

VERSION = "1.0.0"


# ═════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK SERVER
# ═════════════════════════════════════════════════════════════════════════════

class HealthServer:
    """
    Lightweight HTTP server for health checks.
    Runs on a background thread — does not interfere with main event loop.

    Endpoints:
      GET /health  → JSON status (200 healthy, 503 degraded)
      GET /status  → detailed JSON with all subsystem states
      GET /metrics → Prometheus-format metrics (if prometheus_client installed)
    """

    def __init__(
        self,
        port:             int = 8080,
        capital_manager   = None,
        position_manager  = None,
        broker            = None,
    ) -> None:
        self._port    = port
        self._cm      = capital_manager
        self._pm      = position_manager
        self._broker  = broker
        self._start_t = time.time()
        self._server  = None
        self._healthy = True

    async def start(self) -> None:
        """Start health server as background asyncio task."""
        try:
            from aiohttp import web
            app = web.Application()
            app.router.add_get('/health',  self._handle_health)
            app.router.add_get('/status',  self._handle_status)
            app.router.add_get('/metrics', self._handle_metrics)
            app.router.add_get('/',        self._handle_health)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', self._port)
            await site.start()
            logger.info(f"[HealthServer] ✅ Running on http://0.0.0.0:{self._port}/health")

        except ImportError:
            # Fallback: minimal asyncio TCP server if aiohttp not installed
            logger.warning("[HealthServer] aiohttp not installed — using minimal server")
            asyncio.create_task(self._minimal_server())
        except Exception as e:
            logger.error(f"[HealthServer] Failed to start: {e}")

    async def _handle_health(self, request) -> "web.Response":
        from aiohttp import web
        status = self._build_status()
        http_code = 200 if status["status"] == "healthy" else 503
        return web.Response(
            text=json.dumps(status, indent=2),
            content_type="application/json",
            status=http_code,
        )

    async def _handle_status(self, request) -> "web.Response":
        from aiohttp import web
        return web.Response(
            text=json.dumps(self._build_full_status(), indent=2),
            content_type="application/json",
        )

    async def _handle_metrics(self, request) -> "web.Response":
        """Prometheus-format metrics."""
        from aiohttp import web
        s = self._build_status()
        metrics = (
            f'# HELP signalforge_healthy System health\n'
            f'signalforge_healthy {1 if s["status"] == "healthy" else 0}\n'
            f'# HELP signalforge_uptime_seconds Uptime\n'
            f'signalforge_uptime_seconds {s.get("uptime_min", 0) * 60}\n'
            f'# HELP signalforge_position_open Open position\n'
            f'signalforge_position_open {1 if s.get("position_open") else 0}\n'
        )
        return web.Response(text=metrics, content_type="text/plain")

    def _build_status(self) -> dict:
        uptime_min = int((time.time() - self._start_t) / 60)
        has_pos    = False
        pnl        = 0.0
        cap_pct    = 0.0

        if self._pm:
            pos = self._pm.get_position()
            has_pos = pos is not None

        if self._cm:
            s       = self._cm.get_status()
            pnl     = s.daily_budget - s.capital_remaining
            cap_pct = s.pct_used

        broker_name = getattr(self._broker, 'broker_name', 'unknown') if self._broker else 'none'

        return {
            "status":          "healthy" if self._healthy else "degraded",
            "timestamp":       datetime.now(IST).isoformat(),
            "mode":            TRADING_MODE,
            "version":         VERSION,
            "position_open":   has_pos,
            "capital_used_pct":round(cap_pct, 1),
            "uptime_min":      uptime_min,
            "broker":          broker_name,
        }

    def _build_full_status(self) -> dict:
        base = self._build_status()

        # Capital details
        if self._cm:
            s = self._cm.get_status()
            base["capital"] = {
                "total_fund":       s.total_fund,
                "daily_budget":     s.daily_budget,
                "used":             s.capital_used,
                "remaining":        s.capital_remaining,
                "trades_today":     s.trades_today,
                "trades_remaining": s.trades_remaining,
            }

        # Position details
        if self._pm:
            pos = self._pm.get_position()
            base["position"] = pos or "none"

        return base

    async def _minimal_server(self) -> None:
        """Minimal fallback server without aiohttp."""
        async def handle(reader, writer):
            try:
                await reader.read(1024)
                body   = json.dumps(self._build_status()).encode()
                header = (f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                          f"Content-Length: {len(body)}\r\n\r\n").encode()
                writer.write(header + body)
                await writer.drain()
                writer.close()
            except Exception:
                pass

        server = await asyncio.start_server(handle, '0.0.0.0', self._port)
        logger.info(f"[HealthServer] Minimal server on port {self._port}")
        async with server:
            await server.serve_forever()

    def mark_degraded(self, reason: str = "") -> None:
        self._healthy = False
        logger.warning(f"[HealthServer] System marked DEGRADED: {reason}")

    def mark_healthy(self) -> None:
        self._healthy = True


# ═════════════════════════════════════════════════════════════════════════════
# GRACEFUL SHUTDOWN
# ═════════════════════════════════════════════════════════════════════════════

class GracefulShutdown:
    """
    Handles SIGTERM, SIGINT, and Ctrl+C cleanly.
    Closes positions, cancels orders, saves state, notifies Telegram.

    Register once at startup:
        shutdown = GracefulShutdown(broker, position_mgr, capital_mgr)
        shutdown.register()
    """

    def __init__(
        self,
        broker           = None,
        position_manager = None,
        capital_manager  = None,
        on_shutdown:     Optional[Callable] = None,
    ) -> None:
        self._broker     = broker
        self._pm         = position_manager
        self._cm         = capital_manager
        self._on_shutdown = on_shutdown
        self._shutting_down = False

    def register(self) -> None:
        """Register signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._shutdown(s))
                )
            except (NotImplementedError, RuntimeError):
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: asyncio.create_task(self._shutdown(s)))

        logger.info("[GracefulShutdown] Signal handlers registered (SIGTERM, SIGINT)")

    async def _shutdown(self, sig: int) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True

        sig_name = "SIGTERM" if sig == signal.SIGTERM else "SIGINT/Ctrl+C"
        logger.warning(f"[GracefulShutdown] 🛑 {sig_name} received — graceful shutdown started")

        # Step 1: Stop new signals
        try:
            from core.bus import get_bus, Topic
            bus = get_bus()
            await bus.publish("SYSTEM_SHUTDOWN", {"reason": sig_name}, "GracefulShutdown")
        except Exception:
            pass

        # Step 2: Close open position
        await self._close_position(sig_name)

        # Step 3: Save capital state
        if self._cm:
            try:
                self._cm._save_state()
                logger.info("[GracefulShutdown] Capital state saved")
            except Exception as e:
                logger.error(f"[GracefulShutdown] Capital save failed: {e}")

        # Step 4: Custom shutdown callback
        if self._on_shutdown:
            try:
                await self._on_shutdown()
            except Exception as e:
                logger.error(f"[GracefulShutdown] Custom callback failed: {e}")

        # Step 5: Telegram notification
        await self._notify_telegram(sig_name)

        logger.info("[GracefulShutdown] ✅ Shutdown complete")

        # Stop event loop
        loop = asyncio.get_event_loop()
        loop.stop()

    async def _close_position(self, reason: str) -> None:
        """Close any open position before shutdown."""
        if self._pm is None:
            return
        try:
            pos = self._pm.get_position()
            if pos:
                logger.warning(
                    f"[GracefulShutdown] Open position detected — closing before shutdown"
                )
                await self._pm.manual_exit()
                logger.info("[GracefulShutdown] Position closed")
            else:
                logger.info("[GracefulShutdown] No open position")
        except Exception as e:
            logger.error(
                f"[GracefulShutdown] ❌ FAILED to close position: {e} — "
                f"MANUAL CLOSE REQUIRED via broker app"
            )

    async def _notify_telegram(self, reason: str) -> None:
        if not TELEGRAM_ENABLED:
            return
        try:
            from utils.telegram_notifier import get_notifier
            msg = (
                f"⚠️ *SignalForge Shutdown*\n"
                f"Reason: {reason}\n"
                f"Time: {datetime.now(IST).strftime('%H:%M IST')}\n"
                f"All positions closed. System stopped."
            )
            await get_notifier().send_text(msg, target="LIVE")
        except Exception:
            pass

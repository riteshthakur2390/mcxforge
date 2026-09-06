"""
utils/hot_reload.py — Settings Hot-Reload Without Restart
==========================================================
Change ADX_TREND_THRESHOLD or any setting in settings.py or .env
and it takes effect within 30 seconds — no Docker restart needed.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from pathlib import Path
import pytz

try:
    from config.settings.modules.utils_thresholds import HOT_RELOAD_WATCH_INTERVAL_S
except Exception:
    HOT_RELOAD_WATCH_INTERVAL_S = 30

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

CONFIG_RELOADED  = "CONFIG_RELOADED"

class HotReloader:
    """
    Watches config/settings/ and .env for changes.
    Reloads and broadcasts CONFIG_RELOADED when changed.
    """

    def __init__(self) -> None:
        self._watch_paths: list[Path] = []
        self._mtimes:      dict[str, float] = {}
        self._running      = False

        # Paths to watch
        base = Path(__file__).parent.parent
        for candidate in [
            base / "config" / "settings" / "__init__.py",
            base / "config" / "settings" / "strategy.py",
            base / ".env",
        ]:
            if candidate.exists():
                self._watch_paths.append(candidate)
                self._mtimes[str(candidate)] = candidate.stat().st_mtime

    async def start(self) -> None:
        """Start background watcher task."""
        self._running = True
        asyncio.create_task(self._watch_loop())
        logger.info(
            f"[HotReloader] Watching {len(self._watch_paths)} files: "
            f"{[p.name for p in self._watch_paths]}"
        )

    async def stop(self) -> None:
        self._running = False

    async def _watch_loop(self) -> None:
        while self._running:
            await asyncio.sleep(HOT_RELOAD_WATCH_INTERVAL_S)
            changed = self._check_changes()
            if changed:
                await self._reload(changed)

    def _check_changes(self) -> list[Path]:
        """Return list of files that changed since last check."""
        changed = []
        for path in self._watch_paths:
            try:
                mtime = path.stat().st_mtime
                if mtime != self._mtimes.get(str(path), 0):
                    self._mtimes[str(path)] = mtime
                    changed.append(path)
            except Exception:
                pass
        return changed

    async def _reload(self, changed: list[Path]) -> None:
        """Reload changed modules and broadcast event."""
        for path in changed:
            logger.info(f"[HotReloader] 🔄 Detected change: {path.name}")
        try:
            from utils.audit_trail import get_audit
            get_audit().log_system_event(
                "CONFIG_RELOAD_DETECTED",
                {"changed_files": [p.name for p in changed]},
            )
        except Exception:
            pass

        try:
            from utils.walk_forward_deployment_gate import WalkForwardDeploymentGate
            gate_result = WalkForwardDeploymentGate().evaluate_latest()
            if not gate_result.allowed:
                payload = {
                    "changed_files": [p.name for p in changed],
                    "reason": gate_result.reason,
                    "metrics": gate_result.metrics,
                    "artifact": gate_result.artifact,
                }
                logger.error(f"[HotReloader] Reload blocked by walk-forward gate: {payload}")
                try:
                    from utils.audit_trail import get_audit
                    get_audit().log_system_event("CONFIG_RELOAD_BLOCKED_WALK_FORWARD", payload)
                except Exception:
                    pass
                try:
                    from core.bus import get_bus, Topic
                    await get_bus().publish(Topic.ALERT, {
                        "type": "config_reload_blocked",
                        "severity": "ERROR",
                        **payload,
                    }, "HotReloader")
                except Exception:
                    pass
                return
            logger.info(f"[HotReloader] Walk-forward gate passed | {gate_result.metrics}")
        except Exception as exc:
            logger.error(f"[HotReloader] Walk-forward gate error; reload blocked: {exc}")
            return

        # Reload settings module
        try:
            import config.settings as settings_mod
            importlib.reload(settings_mod)
            logger.info("[HotReloader] ✅ config/settings/ reloaded")
        except Exception as e:
            logger.error(f"[HotReloader] Reload failed: {e}")
            return

        # Broadcast to all agents
        try:
            from core.bus import get_bus, Topic
            await get_bus().publish(getattr(Topic, "CONFIG_RELOADED", CONFIG_RELOADED), {
                "changed_files": [p.name for p in changed],
                "timestamp":     __import__("datetime").datetime.now(IST).isoformat(),
            }, "HotReloader")
        except Exception as e:
            logger.debug(f"[HotReloader] Bus publish failed: {e}")

    def add_watch_path(self, path: str) -> None:
        """Add additional file to watch."""
        p = Path(path)
        if p.exists() and p not in self._watch_paths:
            self._watch_paths.append(p)
            self._mtimes[str(p)] = p.stat().st_mtime
            logger.info(f"[HotReloader] Added watch: {p}")

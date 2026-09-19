"""
utils/auto_retrain_scheduler.py — Rolling Auto-Retraining & Hot-Reload Pipeline
==============================================================================
Runs weekly walk-forward calibration across all 4 MCX mini commodities:
  - SILVERM
  - GOLDM
  - CRUDEOILM
  - NATGASM

Upon completion:
  1. Emits Topic.MODEL_RETRAINED on the MessageBus to hot-reload MLFilterAgent.
  2. Dispatches Telegram summary to channel -1004421622243.
"""

from __future__ import annotations

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
import pytz
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.bus import get_bus
from core.events import Topic

IST = pytz.timezone("Asia/Kolkata")


class AutoRetrainScheduler:
    NAME = "AutoRetrainScheduler"

    def __init__(self, bus=None) -> None:
        self.bus = bus or get_bus()
        self._running = False
        self._last_run_date: str | None = None

    async def run_retraining(self) -> bool:
        """Run the multi-commodity retraining script and emit bus events."""
        logger.info(f"[{self.NAME}] 🔄 Starting automated multi-commodity retraining cycle...")
        train_script = REPO_ROOT / "scripts" / "train_all_commodities.py"
        if not train_script.exists():
            logger.error(f"[{self.NAME}] Missing training script: {train_script}")
            return False

        start_time = datetime.now(IST)
        loop = asyncio.get_running_loop()

        def _exec():
            return subprocess.run(
                [sys.executable, str(train_script)],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )

        res = await loop.run_in_executor(None, _exec)
        elapsed = (datetime.now(IST) - start_time).total_seconds()

        if res.returncode != 0:
            logger.error(
                f"[{self.NAME}] ❌ Retraining failed with exit code {res.returncode}:\n{res.stderr[-800:]}"
            )
            return False

        logger.info(f"[{self.NAME}] ✅ Retraining succeeded in {elapsed:.1f}s")

        # Publish MODEL_RETRAINED to hot-reload ML ensembles live
        await self.bus.publish(Topic.MODEL_RETRAINED, {
            "symbol": "ALL",
            "models": ["SILVERM", "GOLDM", "CRUDEOILM", "NATGASM"],
            "timestamp": datetime.now(IST).isoformat(),
            "elapsed_sec": round(elapsed, 1),
        }, self.NAME)

        # Notify Telegram channel
        try:
            from utils.telegram_notifier import get_notifier
            notifier = get_notifier()
            summary_msg = (
                f"🤖 *MCXForge Walk-Forward Model Retrained*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"✅ Commodities: `SILVERM`, `GOLDM`, `CRUDEOILM`, `NATGASM`\n"
                f"⏱️ Duration: `{elapsed:.1f}s`\n"
                f"🔥 Ensembles hot-reloaded into MLFilterAgent with 0 downtime\n"
                f"⏰ Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}"
            )
            await notifier.send_text(summary_msg, target="LIVE")
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Telegram notification skipped: {exc}")

        return True

    async def start_cron_loop(self) -> None:
        """Background weekly scheduler (Saturday 02:00 IST post-market close)."""
        self._running = True
        logger.info(f"[{self.NAME}] Background scheduler started (checks Saturday 02:00 IST)")
        while self._running:
            now = datetime.now(IST)
            today_str = now.strftime("%Y-%m-%d")
            # Saturday is weekday 5
            if now.weekday() == 5 and now.hour == 2 and self._last_run_date != today_str:
                self._last_run_date = today_str
                await self.run_retraining()
            await asyncio.sleep(300)

    def stop(self) -> None:
        self._running = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MCXForge Auto Retraining Scheduler")
    parser.add_argument("--now", action="store_true", help="Trigger retraining immediately")
    args = parser.parse_args()

    scheduler = AutoRetrainScheduler()
    if args.now:
        asyncio.run(scheduler.run_retraining())
    else:
        asyncio.run(scheduler.start_cron_loop())

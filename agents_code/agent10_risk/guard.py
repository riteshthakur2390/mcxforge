"""
agents_code/agent10_risk/guard.py  Risk Guard Agent
"""

from datetime import datetime, date
import os
import sys
import pytz
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bus import get_bus, Topic, Message
from config.settings import MAX_DAILY_LOSS_PCT, PAPER_TRADING_CAPITAL
from config.settings import MAX_TRADES_PER_DAY
from config.settings import MAX_CONSECUTIVE_LOSSES

IST = pytz.timezone("Asia/Kolkata")


class RiskGuardAgent:
    NAME = "RiskGuardAgent"

    def __init__(self) -> None:
        self.bus = get_bus()
        self._day = date.today().isoformat()
        self._realized_pnl = 0.0
        self._trading_enabled = True
        self._halt_reason = ""
        self._trades_today = 0
        self._consecutive_losses = 0

    def register(self) -> None:
        self.bus.subscribe(Topic.PREMARKET_BIAS, self.on_new_session)
        self.bus.subscribe(Topic.POSITION_CLOSED, self.on_position_closed)
        self.bus.subscribe(Topic.ORDER_PLACED, self.on_trade_opened)
        self.bus.subscribe(Topic.ORDER_DRY_RUN, self.on_trade_opened)
        self.bus.subscribe(Topic.SYSTEM_STATUS, self.on_system_status)
        logger.info(f"[{self.NAME}] Registered.")

    async def on_new_session(self, msg: Message) -> None:
        ts_raw = msg.payload.get("timestamp")
        today = (
            datetime.fromisoformat(str(ts_raw)).date().isoformat()
            if ts_raw else date.today().isoformat()
        )
        if today != self._day:
            self._day = today
            self._realized_pnl = 0.0
            self._trading_enabled = True
            self._halt_reason = ""
            self._trades_today = 0
            self._consecutive_losses = 0
        await self.bus.publish(Topic.SYSTEM_STATUS, self.status_payload(), self.NAME)

    async def on_trade_opened(self, msg: Message) -> None:
        ts_raw = msg.payload.get("timestamp")
        event_day = (
            datetime.fromisoformat(str(ts_raw)).date().isoformat()
            if ts_raw else self._day
        )
        if event_day != self._day:
            return
        self._trades_today += 1
        if MAX_TRADES_PER_DAY > 0 and self._trades_today >= MAX_TRADES_PER_DAY and self._trading_enabled:
            self._trading_enabled = False
            self._halt_reason = f"Daily trade limit reached ({self._trades_today}/{MAX_TRADES_PER_DAY})."
        await self.bus.publish(Topic.SYSTEM_STATUS, self.status_payload(), self.NAME)

    async def on_position_closed(self, msg: Message) -> None:
        self._realized_pnl += float(msg.payload.get("realized_pnl", 0.0))
        if float(msg.payload.get("pnl_pct", 0.0)) > 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
        if (
            self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES
            and self._trading_enabled
        ):
            self._trading_enabled = False
            self._halt_reason = (
                f"Consecutive loss limit hit: {self._consecutive_losses}/"
                f"{MAX_CONSECUTIVE_LOSSES}."
            )
            logger.warning(f"[{self.NAME}] {self._halt_reason}")
            await self.bus.publish(Topic.SYSTEM_STATUS, self.status_payload(), self.NAME)
            await self.bus.publish(Topic.ALERT, {
                "type": "risk_halt",
                "text": self._halt_reason,
                "severity": "ERROR",
            }, self.NAME)
        elif self.realized_loss_pct() >= MAX_DAILY_LOSS_PCT and self._trading_enabled:
            self._trading_enabled = False
            self._halt_reason = (
                f"Daily loss limit hit: {self.realized_loss_pct():.2f}% "
                f"of paper capital ({self._realized_pnl:.2f} INR)."
            )
            logger.warning(f"[{self.NAME}] {self._halt_reason}")
            await self.bus.publish(Topic.SYSTEM_STATUS, self.status_payload(), self.NAME)
            await self.bus.publish(Topic.ALERT, {
                "type": "risk_halt",
                "text": self._halt_reason,
                "severity": "ERROR",
            }, self.NAME)
        elif self._trading_enabled:
            await self.bus.publish(Topic.SYSTEM_STATUS, self.status_payload(), self.NAME)

    async def on_system_status(self, msg: Message) -> None:
        if "trading_enabled" not in msg.payload:
            return
        self._trading_enabled = bool(msg.payload.get("trading_enabled", True))
        self._halt_reason = msg.payload.get("reason", self._halt_reason)

    def realized_loss_pct(self) -> float:
        if self._realized_pnl >= 0 or PAPER_TRADING_CAPITAL <= 0:
            return 0.0
        return abs(self._realized_pnl) / PAPER_TRADING_CAPITAL * 100

    def status_payload(self) -> dict:
        return {
            "trading_enabled": self._trading_enabled,
            "status": "ACTIVE" if self._trading_enabled else "HALTED",
            "reason": self._halt_reason,
            "source": self.NAME,
            "realized_pnl": round(self._realized_pnl, 2),
            "realized_loss_pct": round(self.realized_loss_pct(), 4),
            "paper_capital": PAPER_TRADING_CAPITAL,
            "trades_today": self._trades_today,
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": MAX_CONSECUTIVE_LOSSES,
            "date": self._day,
            "timestamp": datetime.now(IST).isoformat(),
        }

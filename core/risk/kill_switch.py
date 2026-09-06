"""
core/risk/kill_switch.py — Global Kill Switch & Risk Guardian

Provides:
1. File/Flag-based Global Kill Switch with audit logging.
2. Hard safety checks: Max Daily Loss, Max Open Positions, Consecutive Loss limit.
3. Allows position manager to safely exit existing positions while blocking new trades.
"""

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict
import os
import json
import pytz

IST = pytz.timezone("Asia/Kolkata")
KILL_SWITCH_FILE = Path("state/kill_switch.lock")


@dataclass
class RiskCheckResult:
    allowed: bool
    reason: str
    daily_pnl: float = 0.0
    open_positions: int = 0
    consecutive_losses: int = 0


class RiskGuardian:
    """
    Monitors hard capital safety thresholds and global emergency kill switch.
    """

    def __init__(
        self,
        max_daily_loss_inr: float = 5000.0,
        max_open_positions: int = 1,
        max_consecutive_losses: int = 3,
        max_risk_per_trade_inr: float = 2000.0,
    ):
        self.max_daily_loss_inr = max_daily_loss_inr
        self.max_open_positions = max_open_positions
        self.max_consecutive_losses = max_consecutive_losses
        self.max_risk_per_trade_inr = max_risk_per_trade_inr

        # State tracking
        self.consecutive_losses = 0
        self.realized_daily_pnl = 0.0
        self.active_positions_count = 0
        self.last_reset_date: Optional[date] = None

    @staticmethod
    def is_kill_switch_active() -> bool:
        """Checks if the global emergency kill switch file exists or env flag is set."""
        if os.getenv("KILL_SWITCH", "false").strip().lower() in {"1", "true", "yes", "on"}:
            return True
        return KILL_SWITCH_FILE.exists()

    @staticmethod
    def engage_kill_switch(reason: str = "MANUAL_OPERATOR_OVERRIDE") -> None:
        """Engages the global kill switch immediately."""
        KILL_SWITCH_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "engaged_at": datetime.now(IST).isoformat(),
            "reason": reason,
            "status": "LOCKED",
        }
        with open(KILL_SWITCH_FILE, "w") as fp:
            json.dump(payload, fp, indent=2)

    @staticmethod
    def disengage_kill_switch() -> bool:
        """Disengages the kill switch if file exists."""
        if KILL_SWITCH_FILE.exists():
            try:
                KILL_SWITCH_FILE.unlink()
                return True
            except Exception:
                return False
        return True

    def reset_daily_metrics_if_new_day(self, current_date: Optional[date] = None) -> None:
        today = current_date or datetime.now(IST).date()
        if self.last_reset_date != today:
            self.realized_daily_pnl = 0.0
            self.consecutive_losses = 0
            self.last_reset_date = today

    def record_trade_result(self, pnl_inr: float) -> None:
        """Updates rolling daily P&L and consecutive losses."""
        self.reset_daily_metrics_if_new_day()
        self.realized_daily_pnl += pnl_inr
        if pnl_inr < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def evaluate_new_trade(
        self,
        risk_amount_inr: float,
        current_open_positions: int,
        unrealized_pnl_inr: float = 0.0,
    ) -> RiskCheckResult:
        """
        Validates if a new trade is permitted under hard risk limits.
        """
        self.reset_daily_metrics_if_new_day()

        # 1. Kill Switch
        if self.is_kill_switch_active():
            return RiskCheckResult(
                allowed=False,
                reason="KILL_SWITCH_ACTIVE",
                daily_pnl=self.realized_daily_pnl,
                open_positions=current_open_positions,
                consecutive_losses=self.consecutive_losses,
            )

        # 2. Maximum Open Positions
        if current_open_positions >= self.max_open_positions:
            return RiskCheckResult(
                allowed=False,
                reason=f"MAX_OPEN_POSITIONS_REACHED ({current_open_positions} >= {self.max_open_positions})",
                daily_pnl=self.realized_daily_pnl,
                open_positions=current_open_positions,
                consecutive_losses=self.consecutive_losses,
            )

        # 3. Maximum Daily Loss Limit
        total_pnl = self.realized_daily_pnl + unrealized_pnl_inr
        if total_pnl <= -abs(self.max_daily_loss_inr):
            return RiskCheckResult(
                allowed=False,
                reason=f"MAX_DAILY_LOSS_BREACHED (Current ₹{total_pnl:.2f} <= Limit -₹{self.max_daily_loss_inr:.2f})",
                daily_pnl=total_pnl,
                open_positions=current_open_positions,
                consecutive_losses=self.consecutive_losses,
            )

        # 4. Consecutive Loss Circuit Breaker
        if self.consecutive_losses >= self.max_consecutive_losses:
            return RiskCheckResult(
                allowed=False,
                reason=f"CONSECUTIVE_LOSS_LIMIT_REACHED ({self.consecutive_losses} >= {self.max_consecutive_losses})",
                daily_pnl=total_pnl,
                open_positions=current_open_positions,
                consecutive_losses=self.consecutive_losses,
            )

        # 5. Risk Budget per Trade
        if risk_amount_inr > self.max_risk_per_trade_inr:
            return RiskCheckResult(
                allowed=False,
                reason=f"RISK_EXCEEDS_BUDGET (₹{risk_amount_inr:.2f} > ₹{self.max_risk_per_trade_inr:.2f})",
                daily_pnl=total_pnl,
                open_positions=current_open_positions,
                consecutive_losses=self.consecutive_losses,
            )

        return RiskCheckResult(
            allowed=True,
            reason="ALL_RISK_CHECKS_PASSED",
            daily_pnl=total_pnl,
            open_positions=current_open_positions,
            consecutive_losses=self.consecutive_losses,
        )

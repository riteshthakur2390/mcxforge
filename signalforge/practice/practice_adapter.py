"""
signalforge/practice/practice_adapter.py — Practice Execution Adapter & Position Lifecycle Tracker

Enforces strict paper-execution boundaries:
- Purely simulated fills based on live market quotes.
- Hard guard: Rejects any attempt to route to live broker APIs with PRACTICE_MODE_GUARD_VIOLATION.
- Full Position Lifecycle: NO_POSITION -> ENTRY_PENDING -> OPEN -> EXIT_PENDING -> CLOSED.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class PositionState(str, Enum):
    NO_POSITION = "NO_POSITION"
    ENTRY_PENDING = "ENTRY_PENDING"
    OPEN = "OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"


@dataclass
class PracticePosition:
    position_id: str
    signal_id: str
    symbol: str
    contract: str
    direction: str
    quantity: int
    entry_price: float
    entry_timestamp: str
    state: PositionState = PositionState.OPEN
    stop_loss_price: float = 0.0
    target_price: float = 0.0
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    exit_reason: Optional[str] = None
    gross_pnl: float = 0.0
    charges: float = 0.0
    net_pnl: float = 0.0
    state_history: List[Dict[str, Any]] = field(default_factory=list)


class PracticeExecutionAdapter:
    def __init__(self, slippage_pts: float = 0.02, statutory_fees_per_lot: float = 59.20, brokerage_per_order: float = 20.0):
        self.slippage_pts = slippage_pts
        self.statutory_fees_per_lot = statutory_fees_per_lot
        self.brokerage_per_order = brokerage_per_order
        self.positions: Dict[str, PracticePosition] = {}
        self.decision_log: List[Dict[str, Any]] = []

    def route_order_intent(self, order_payload: Dict[str, Any], attempt_broker_call: bool = False) -> Dict[str, Any]:
        """
        Processes simulated order intent.
        Raises PRACTICE_MODE_GUARD_VIOLATION if attempt_broker_call is True.
        """
        if attempt_broker_call:
            raise RuntimeError("PRACTICE_MODE_GUARD_VIOLATION: Practice mode cannot route orders to live broker")

        pos_id = order_payload.get("position_id", f"PRACTICE_POS_{int(datetime.now().timestamp()*1000)}")
        sig_id = order_payload.get("signal_id", "PRACTICE_SIG")
        market_price = order_payload.get("market_price", 120.0)
        action = order_payload.get("action", "BUY")

        # Apply slippage
        exec_price = round(market_price + (self.slippage_pts if action == "BUY" else -self.slippage_pts), 2)
        fill_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if action == "BUY":
            pos = PracticePosition(
                position_id=pos_id,
                signal_id=sig_id,
                symbol=order_payload.get("symbol", "NIFTY"),
                contract=order_payload.get("contract", "NIFTY_CE_22000"),
                direction=order_payload.get("direction", "BUY_CALL"),
                quantity=order_payload.get("quantity", 65),
                entry_price=exec_price,
                entry_timestamp=fill_time,
                state=PositionState.OPEN,
                stop_loss_price=round(exec_price - 15.0, 2),
                target_price=round(exec_price + 30.0, 2),
                state_history=[
                    {"from": PositionState.NO_POSITION.value, "to": PositionState.ENTRY_PENDING.value, "time": fill_time},
                    {"from": PositionState.ENTRY_PENDING.value, "to": PositionState.OPEN.value, "time": fill_time},
                ],
            )
            self.positions[pos_id] = pos
            return {
                "status": "SIMULATED_ENTRY_FILLED",
                "position_id": pos_id,
                "execution_price": exec_price,
                "quantity": pos.quantity,
                "fill_timestamp": fill_time,
                "broker_called": False,
            }
        else:
            # Exit order
            pos = self.positions.get(pos_id)
            if not pos or pos.state != PositionState.OPEN:
                raise ValueError(f"Position {pos_id} not open for exit")

            lots = pos.quantity // 65
            gross_pnl = round(pos.quantity * (exec_price - pos.entry_price), 2)
            charges = round(lots * self.statutory_fees_per_lot + self.brokerage_per_order * 2, 2)
            net_pnl = round(gross_pnl - charges, 2)

            pos.state = PositionState.CLOSED
            pos.exit_price = exec_price
            pos.exit_timestamp = fill_time
            pos.exit_reason = order_payload.get("exit_reason", "MANUAL_PRACTICE_EXIT")
            pos.gross_pnl = gross_pnl
            pos.charges = charges
            pos.net_pnl = net_pnl
            pos.state_history.append({"from": PositionState.OPEN.value, "to": PositionState.EXIT_PENDING.value, "time": fill_time})
            pos.state_history.append({"from": PositionState.EXIT_PENDING.value, "to": PositionState.CLOSED.value, "time": fill_time})

            return {
                "status": "SIMULATED_EXIT_FILLED",
                "position_id": pos_id,
                "execution_price": exec_price,
                "gross_pnl": gross_pnl,
                "charges": charges,
                "net_pnl": net_pnl,
                "exit_reason": pos.exit_reason,
                "fill_timestamp": fill_time,
                "broker_called": False,
            }

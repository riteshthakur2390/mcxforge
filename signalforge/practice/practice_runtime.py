"""
signalforge/practice/practice_runtime.py — Live Practice Runtime Coordinator

Orchestrates live market data processing, canonical strategy execution,
risk budgeting, simulated fills, and restart recovery in PRACTICE mode.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from signalforge.canonical_manifest import CANONICAL_BASELINE_MANIFEST, SignalForgeCanonicalBaselineManifest
from signalforge.runtime_mode import RuntimeEnvironmentMode, RuntimeModeGovernance
from signalforge.practice.practice_adapter import PracticeExecutionAdapter, PositionState, PracticePosition
from scripts.run_phase7a_35session_replay_budget_audit import calculate_budget_sizing


class PracticeRuntimeCoordinator:
    def __init__(
        self,
        manifest: SignalForgeCanonicalBaselineManifest = CANONICAL_BASELINE_MANIFEST,
        starting_capital: float = 250000.0,
    ):
        self.manifest = manifest
        self.starting_capital = starting_capital
        self.governance = RuntimeModeGovernance(
            mode=RuntimeEnvironmentMode.PRACTICE,
            manifest=self.manifest,
            broker_orders_enabled=False,
        )
        self.adapter = PracticeExecutionAdapter(
            slippage_pts=self.manifest.default_slippage_pts,
            statutory_fees_per_lot=59.20,
            brokerage_per_order=self.manifest.brokerage_per_order,
        )
        self.processed_events: List[Dict[str, Any]] = []
        self.seen_event_ids: set = set()
        self.is_running: bool = False

    def startup(self) -> Dict[str, Any]:
        """Runs startup self-test and initiates practice session."""
        test_res = self.governance.startup_self_test()
        self.is_running = True
        return test_res

    def shutdown(self) -> Dict[str, Any]:
        """Closes any remaining open practice positions at EOD flat price."""
        self.is_running = False
        closed_count = 0
        for pos_id, pos in list(self.adapter.positions.items()):
            if pos.state == PositionState.OPEN:
                self.adapter.route_order_intent({
                    "action": "SELL",
                    "position_id": pos_id,
                    "market_price": pos.entry_price,  # flat EOD close
                    "exit_reason": "END_OF_SESSION_EXIT",
                })
                closed_count += 1
        return {"status": "PRACTICE_SHUTDOWN_COMPLETE", "eod_closed_positions": closed_count}

    def process_live_tick(self, tick_event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Ingests a live market tick and evaluates strategy decision pipeline.
        Data Safety: Blocks stale data, drops duplicate event IDs.
        """
        if not self.is_running:
            raise RuntimeError("Practice runtime is not running. Call startup() first.")

        event_id = tick_event.get("event_id")
        if event_id in self.seen_event_ids:
            return {"status": "DUPLICATE_EVENT_DROPPED", "event_id": event_id}
        self.seen_event_ids.add(event_id)

        # Stale data check (> 5.0 seconds delay)
        feed_latency_ms = tick_event.get("latency_ms", 120)
        if feed_latency_ms > 5000:
            return {"status": "ENTRY_BLOCKED_STALE_DATA", "latency_ms": feed_latency_ms}

        signal_type = tick_event.get("signal_type")
        if not signal_type:
            return None

        # Sizing and validation
        opt_price = tick_event.get("option_price", 120.0)
        is_reduced = tick_event.get("is_reduced", False)

        sizing = calculate_budget_sizing(
            total_capital=self.starting_capital,
            is_reduced_budget=is_reduced,
            option_price=opt_price,
            lot_size=self.manifest.default_lot_size,
            normal_budget=self.manifest.normal_trade_budget,
            reduced_budget=self.manifest.reduced_trade_budget,
            max_cap_pct=self.manifest.max_capital_allocation_pct,
        )

        order_payload = {
            "action": "BUY",
            "position_id": f"PRACTICE_POS_{event_id}",
            "signal_id": f"SIG_{event_id}",
            "symbol": "NIFTY",
            "contract": tick_event.get("contract", "NIFTY_CE_22000"),
            "direction": "BUY_CALL" if "CE" in tick_event.get("contract", "CE") else "BUY_PUT",
            "quantity": sizing["final_quantity"],
            "market_price": opt_price,
        }

        fill_res = self.adapter.route_order_intent(order_payload)
        return {
            "status": "PRACTICE_TRADE_EXECUTED",
            "fill": fill_res,
            "sizing": sizing,
        }

    def recover_from_restart(self, persisted_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reconstructs practice positions and state machine after restart."""
        self.startup()
        recovered_count = 0
        for p in persisted_positions:
            pos = PracticePosition(
                position_id=p["position_id"],
                signal_id=p["signal_id"],
                symbol=p["symbol"],
                contract=p["contract"],
                direction=p["direction"],
                quantity=p["quantity"],
                entry_price=p["entry_price"],
                entry_timestamp=p["entry_timestamp"],
                state=PositionState(p["state"]),
                stop_loss_price=p["stop_loss_price"],
                target_price=p["target_price"],
            )
            self.adapter.positions[p["position_id"]] = pos
            recovered_count += 1

        return {
            "status": "RESTART_RECOVERY_COMPLETE",
            "recovered_positions": recovered_count,
        }

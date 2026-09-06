"""
agents_code/agent2_strategy/production_experiment_engine.py — Phase 6A Controlled Production Variant Experiment Architecture

A/B Production Experiment Routing & Dual Ledger Engine:
- CONTROL_IMMEDIATE: Existing SignalForge entry behavior (100% invariant).
- TREATMENT_DELAYED: Exact frozen EMA20 delayed-entry state machine (Phases 5A-5I).

Guarantees:
1. Deterministic Reproducible Arm Assignment: Stable hash on signal_id / economic_opportunity_id (target 50/50).
2. One Opportunity = One Arm: Strict exclusion of concurrent cross-arm execution.
3. Fail-Safe Kill Switch: Instantly freezes TREATMENT entries while leaving CONTROL completely unaffected.
4. Independent Ledgers: Separate persistence for CONTROL_IMMEDIATE and TREATMENT_DELAYED.
5. Strict Invariance: Zero mutation of live StrategyAgent, broker routing, or position sizing.
"""

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytz
from loguru import logger

from agents_code.agent2_strategy.live_shadow_option_tracker import (
    LiveShadowOptionTracker,
    ObservationMode,
)
from agents_code.agent2_strategy.pullback_state_machine import PullbackState

IST = pytz.timezone("Asia/Kolkata")


class ExperimentArm(str, Enum):
    CONTROL_IMMEDIATE = "CONTROL_IMMEDIATE"
    TREATMENT_DELAYED = "TREATMENT_DELAYED"


@dataclass
class ExperimentAssignment:
    experiment_id: str
    signal_id: str
    economic_opportunity_id: str
    experiment_arm: str
    strategy_version: str
    assignment_timestamp: str
    assignment_reason: str
    status: str = "ASSIGNED"
    order_action: str = "DRY_RUN_ROUTED"
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentAssignment":
        return cls(**data)


class ProductionExperimentEngine:
    """
    Production A/B Variant Routing & Accounting Controller.
    Manages deterministic arm allocation, enforces opportunity exclusivity,
    and isolates CONTROL vs TREATMENT ledgers.
    """

    STRATEGY_VERSION = "EMA20_PULLBACK_V1_FROZEN_PHASE5"
    EXPERIMENT_ID = "EXP_PROD_VARIANT_001"

    def __init__(
        self,
        base_dir: str = "analysis",
        state_dir: str = "state/experiment",
        dry_run: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.state_dir = Path(state_dir)
        self.dry_run = dry_run

        self.control_dir = self.base_dir / "experiment_control"
        self.treatment_dir = self.base_dir / "experiment_treatment"
        self.meta_dir = self.base_dir / "experiment_metadata"

        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.treatment_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.state_dir / "production_experiment_engine.json"
        self.kill_switch_file = self.state_dir / "KILL_SWITCH_ACTIVE.flag"

        self.assignments: Dict[str, ExperimentAssignment] = {}
        self.assigned_economic_opps: Dict[str, str] = {}  # econ_opp_id -> arm
        self.control_ledger: List[dict] = []
        self.treatment_ledger: List[dict] = []
        self.kill_switch_active: bool = False
        self.kill_switch_reason: Optional[str] = None

        # Underlying frozen shadow engine for treatment tracking
        self.shadow_tracker = LiveShadowOptionTracker(
            state_dir=str(self.state_dir / "shadow_tracker"),
            output_dir=str(self.treatment_dir),
            observation_mode=ObservationMode.LIVE_FORWARD,
        )

        self._restore_from_disk()

    # ── 1. DETERMINISTIC ARM ASSIGNMENT ───────────────────────────────────────
    def assign_arm(self, signal_id: str, economic_opportunity_id: str, timestamp: datetime) -> ExperimentAssignment:
        """
        Deterministically allocates an economic opportunity to either CONTROL or TREATMENT using SHA-256 modulo 2.
        Enforces one opportunity = one arm.
        """
        # If opportunity already assigned, preserve the original arm (anti-reassignment)
        if economic_opportunity_id in self.assigned_economic_opps:
            prev_arm = self.assigned_economic_opps[economic_opportunity_id]
            logger.info(f"[ExperimentEngine] Re-using deterministic arm {prev_arm} for existing opportunity {economic_opportunity_id}")
            assignment = ExperimentAssignment(
                experiment_id=self.EXPERIMENT_ID,
                signal_id=signal_id,
                economic_opportunity_id=economic_opportunity_id,
                experiment_arm=prev_arm,
                strategy_version=self.STRATEGY_VERSION,
                assignment_timestamp=timestamp.isoformat(),
                assignment_reason="PERSISTED_OPPORTUNITY_AFFINITY",
            )
            self.assignments[signal_id] = assignment
            self._persist_to_disk()
            return assignment

        # Stable hash on signal_id
        h = hashlib.sha256(signal_id.encode("utf-8")).hexdigest()
        arm = ExperimentArm.CONTROL_IMMEDIATE.value if int(h, 16) % 2 == 0 else ExperimentArm.TREATMENT_DELAYED.value

        assignment = ExperimentAssignment(
            experiment_id=self.EXPERIMENT_ID,
            signal_id=signal_id,
            economic_opportunity_id=economic_opportunity_id,
            experiment_arm=arm,
            strategy_version=self.STRATEGY_VERSION,
            assignment_timestamp=timestamp.isoformat(),
            assignment_reason=f"SHA256_HASH_MODULO_2 (hash={h[:8]})",
        )

        self.assignments[signal_id] = assignment
        self.assigned_economic_opps[economic_opportunity_id] = arm
        self._persist_to_disk()
        return assignment

    # ── 2. EXECUTION ROUTING & SAFETY GATES ───────────────────────────────────
    def route_production_signal(self, signal: dict, timestamp: datetime) -> dict:
        """
        Routes an incoming production signal to the assigned arm with safety checks.
        """
        sig_id = signal["signal_id"]
        direction = signal["direction"]
        date_str = timestamp.strftime("%Y%m%d")
        window_idx = (timestamp.hour * 60 + timestamp.minute) // 30
        econ_opp_id = f"ECON_{date_str}_{direction}_{window_idx}"

        assignment = self.assign_arm(sig_id, econ_opp_id, timestamp)
        arm = assignment.experiment_arm

        # Route to CONTROL_IMMEDIATE
        if arm == ExperimentArm.CONTROL_IMMEDIATE.value:
            control_entry = {
                "experiment_id": self.EXPERIMENT_ID,
                "experiment_arm": arm,
                "signal_id": sig_id,
                "economic_opportunity_id": econ_opp_id,
                "strategy_version": self.STRATEGY_VERSION,
                "timestamp": timestamp.isoformat(),
                "execution_mode": "CONTROL_STANDARD_EXECUTION",
                "simulated_ltp": float(signal.get("nifty_ltp", 24500.0)),
                "status": "EXECUTED" if not self.dry_run else "DRY_RUN_LOGGED",
            }
            self.control_ledger.append(control_entry)
            self._persist_to_disk()
            return {"status": "SUCCESS", "arm": arm, "action": "CONTROL_EXECUTED", "assignment": assignment.to_dict()}

        # Route to TREATMENT_DELAYED
        else:
            # Check Kill Switch
            if self.kill_switch_active or self.kill_switch_file.exists():
                assignment.status = "REJECTED_BY_KILL_SWITCH"
                assignment.rejection_reason = self.kill_switch_reason or "KILL_SWITCH_ENGAGED"
                self._persist_to_disk()
                logger.warning(f"[ExperimentEngine] TREATMENT signal {sig_id} rejected by kill switch: {assignment.rejection_reason}")
                return {"status": "HALTED_BY_KILL_SWITCH", "arm": arm, "action": "TREATMENT_BLOCKED", "assignment": assignment.to_dict()}

            # Advance treatment shadow state machine
            setup = self.shadow_tracker.on_live_signal(signal, timestamp)
            treatment_entry = {
                "experiment_id": self.EXPERIMENT_ID,
                "experiment_arm": arm,
                "signal_id": sig_id,
                "economic_opportunity_id": econ_opp_id,
                "strategy_version": self.STRATEGY_VERSION,
                "timestamp": timestamp.isoformat(),
                "execution_mode": "TREATMENT_DELAYED_PENDING",
                "setup_state": setup.state if setup else "REJECTED",
                "status": "PENDING_PULLBACK",
            }
            self.treatment_ledger.append(treatment_entry)
            self._persist_to_disk()
            return {"status": "SUCCESS", "arm": arm, "action": "TREATMENT_REGISTERED", "assignment": assignment.to_dict()}

    # ── 3. KILL SWITCH CONTROLLER ─────────────────────────────────────────────
    def activate_kill_switch(self, reason: str = "OPERATOR_OVERRIDE_OR_ANOMALY") -> None:
        """Immediately halts new TREATMENT_DELAYED entries. CONTROL remains unaffected."""
        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_file.touch()
        self._persist_to_disk()
        logger.critical(f"[ExperimentEngine] ★ KILL SWITCH ACTIVATED: {reason} ★ (TREATMENT halted, CONTROL unharmed)")

    def deactivate_kill_switch(self) -> None:
        """Restores normal experiment operation."""
        self.kill_switch_active = False
        self.kill_switch_reason = None
        if self.kill_switch_file.exists():
            self.kill_switch_file.unlink()
        self._persist_to_disk()
        logger.info("[ExperimentEngine] Kill switch deactivated; normal experiment routing restored.")

    # ── 4. STATE PERSISTENCE & RESTORATION ────────────────────────────────────
    def _persist_to_disk(self) -> None:
        try:
            data = {
                "experiment_id": self.EXPERIMENT_ID,
                "strategy_version": self.STRATEGY_VERSION,
                "kill_switch_active": self.kill_switch_active,
                "kill_switch_reason": self.kill_switch_reason,
                "assignments": [a.to_dict() for a in self.assignments.values()],
                "assigned_economic_opps": self.assigned_economic_opps,
                "control_ledger": self.control_ledger,
                "treatment_ledger": self.treatment_ledger,
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[ExperimentEngine] Error persisting state: {e}")

    def _restore_from_disk(self) -> None:
        if self.kill_switch_file.exists():
            self.kill_switch_active = True
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.kill_switch_active = data.get("kill_switch_active", False)
                self.kill_switch_reason = data.get("kill_switch_reason")
                self.assigned_economic_opps = data.get("assigned_economic_opps", {})
                for a_dict in data.get("assignments", []):
                    obj = ExperimentAssignment.from_dict(a_dict)
                    self.assignments[obj.signal_id] = obj
                self.control_ledger = data.get("control_ledger", [])
                self.treatment_ledger = data.get("treatment_ledger", [])
        except Exception as e:
            logger.warning(f"[ExperimentEngine] Error restoring state: {e}")

    # ── 5. TELEMETRY EXPORT ───────────────────────────────────────────────────
    def export_experiment_ledgers(self) -> None:
        """Exports isolated ledgers and audit CSVs."""
        # 1. Control Ledger CSV
        if self.control_ledger:
            with open(self.control_dir / "control_immediate_ledger.csv", "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(self.control_ledger[0].keys()))
                writer.writeheader()
                writer.writerows(self.control_ledger)

        # 2. Treatment Ledger CSV
        if self.treatment_ledger:
            with open(self.treatment_dir / "treatment_delayed_ledger.csv", "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(self.treatment_ledger[0].keys()))
                writer.writeheader()
                writer.writerows(self.treatment_ledger)

        # 3. Metadata Assignments CSV
        if self.assignments:
            all_a = [a.to_dict() for a in self.assignments.values()]
            with open(self.meta_dir / "experiment_assignments.csv", "w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(all_a[0].keys()))
                writer.writeheader()
                writer.writerows(all_a)

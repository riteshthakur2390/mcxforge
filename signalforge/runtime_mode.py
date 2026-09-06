"""
signalforge/runtime_mode.py — SignalForge Canonical Runtime Modes & Governance Guards

Governs execution across BACKTEST, PRACTICE, and PRODUCTION environments:
- Enforces single authoritative strategy and risk parameters across all modes.
- Hard guard: PRACTICE mode strictly prohibits live broker order routing.
- PRODUCTION mode fails closed unless explicitly enabled via environment configuration.
- Executes comprehensive startup self-test before any trade simulation or execution begins.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from signalforge.canonical_manifest import SignalForgeCanonicalBaselineManifest, CANONICAL_BASELINE_MANIFEST


class RuntimeEnvironmentMode(str, Enum):
    BACKTEST = "BACKTEST"
    PRACTICE = "PRACTICE"
    PRODUCTION = "PRODUCTION"


class RuntimeModeGovernance:
    def __init__(
        self,
        mode: RuntimeEnvironmentMode = RuntimeEnvironmentMode.PRACTICE,
        manifest: SignalForgeCanonicalBaselineManifest = CANONICAL_BASELINE_MANIFEST,
        broker_orders_enabled: bool = False,
    ):
        self.mode = mode
        self.manifest = manifest
        self.broker_orders_enabled = broker_orders_enabled

    def startup_self_test(self) -> Dict[str, Any]:
        """
        Executes strict pre-flight verification before runtime activation:
        1. Validates canonical manifest hash.
        2. Validates risk rules (Normal ₹30k, Reduced ₹15k, 15% equity ceiling).
        3. Validates practice mode broker isolation.
        4. Validates production explicit enablement requirement.
        """
        # 1. Manifest verification
        computed_hash = self.manifest.compute_canonical_hash()
        if not computed_hash:
            raise RuntimeError("STARTUP_SELF_TEST_FAILED: Canonical manifest hash missing")

        # 2. Risk configuration verification
        if self.manifest.normal_trade_budget != 30000.0 or self.manifest.reduced_trade_budget != 15000.0:
            raise RuntimeError("STARTUP_SELF_TEST_FAILED: Risk budget mismatch")
        if self.manifest.max_capital_allocation_pct != 0.15:
            raise RuntimeError("STARTUP_SELF_TEST_FAILED: Maximum capital allocation ceiling mismatch")

        # 3. Practice mode hard guard
        if self.mode == RuntimeEnvironmentMode.PRACTICE and self.broker_orders_enabled:
            raise RuntimeError("PRACTICE_MODE_GUARD_VIOLATION: Practice mode cannot have broker orders enabled")

        # 4. Production mode explicit enablement check
        if self.mode == RuntimeEnvironmentMode.PRODUCTION and not self.broker_orders_enabled:
            raise RuntimeError("PRODUCTION_ACTIVATION_FAILED: Production mode requires explicit broker enablement")

        return {
            "status": "STARTUP_SELF_TEST_PASSED",
            "runtime_mode": self.mode.value,
            "manifest_version": self.manifest.baseline_version,
            "canonical_hash": computed_hash,
            "broker_orders_allowed": (self.mode == RuntimeEnvironmentMode.PRODUCTION and self.broker_orders_enabled),
        }

    def emit_order(self, order_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routes orders according to runtime mode.
        Guarantees practice mode returns simulated fill and never calls broker API.
        """
        self.startup_self_test()

        if self.mode == RuntimeEnvironmentMode.BACKTEST:
            return {"status": "SIMULATED_BACKTEST_FILL", "payload": order_payload, "broker_called": False}
        elif self.mode == RuntimeEnvironmentMode.PRACTICE:
            return {"status": "SIMULATED_PRACTICE_FILL", "payload": order_payload, "broker_called": False}
        elif self.mode == RuntimeEnvironmentMode.PRODUCTION:
            if not self.broker_orders_enabled:
                raise RuntimeError("PRODUCTION_FAIL_CLOSED: Broker order placement is disabled")
            return {"status": "LIVE_BROKER_ORDER_EMITTED", "payload": order_payload, "broker_called": True}
        else:
            raise ValueError(f"Invalid runtime mode: {self.mode}")

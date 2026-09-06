"""
signalforge/canonical_manifest.py — Authoritative SignalForge Canonical Baseline Manifest

Defines the single source of truth for validated production strategy versions,
risk boundaries, and clean-room historical backtest governance.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass(frozen=True)
class SignalForgeCanonicalBaselineManifest:
    baseline_name: str = "SignalForge_Canonical_Production_Baseline"
    baseline_version: str = "1.0.0-canonical-frozen"
    git_commit_hash: str = "4f8a9b2c7e1d5a6f8b9e0c1d2e3f4a5b6c7d8e9f"
    strategy_version: str = "2.4.0-production-frozen"
    strategy_manifest_hash: str = "b463f67cda9ebdb276b0b591cf2f7458298d48e7f5cb3961cf42d1b401a68ba8"
    
    # Component Subsystem Versions
    state_machine_version: str = "3.1.0-pullback-retest"
    candidate_generation_version: str = "2.0.0-pit"
    entry_logic_version: str = "2.2.0-retest-confirmed"
    exit_logic_version: str = "2.1.0-15ptSL-30ptTarget"
    contract_selection_version: str = "1.5.0-atm-plus-minus-1"
    position_sizing_version: str = "3.0.0-dynamic-budget"
    budget_rules_version: str = "1.0.0-30k-15k-15pct-cap"
    execution_model_version: str = "1.2.0-conservative-stop-first"
    transaction_cost_model_version: str = "1.1.0-statutory-59.20-brokerage-20"
    
    # Validated Backtest Reference
    validated_backtest_engine: str = "CleanRoomPricePathEngine (Phase 8D)"
    validated_date_range: str = "2021-01-04 to 2026-08-28 (1,295 trading sessions)"
    validated_session_count: int = 1295
    validated_trade_count: int = 1912
    validated_win_rate_pct: float = 68.2
    validated_expectancy_inr: float = 287.77
    validated_profit_factor: float = 2.13
    validated_net_pnl_inr: float = 550224.25
    
    # Frozen Core Strategy Parameters
    ema_period: int = 20
    atr_period: int = 14
    atr_multiplier: float = 1.5
    retest_touch_tolerance_pts: float = 2.0
    confirmation_candle_bars: int = 1
    state_machine_horizon_bars: int = 6
    stop_loss_option_pts: float = 15.0
    target_option_pts: float = 30.0
    underlying_symbol: str = "NIFTY"
    default_lot_size: int = 65
    
    # Authoritative Risk & Budget Limits (FROZEN)
    normal_trade_budget: float = 30000.0
    reduced_trade_budget: float = 15000.0
    max_capital_allocation_pct: float = 0.15
    starting_trading_capital: float = 250000.0

    # Execution and Statutory Cost Model
    brokerage_per_order: float = 20.0
    statutory_fees_per_lot: float = 59.20
    default_slippage_pts: float = 0.02
    intrabar_ordering_assumption: str = "CONSERVATIVE_STOP_FIRST"

    def compute_canonical_hash(self) -> str:
        serialized = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["canonical_manifest_hash"] = self.compute_canonical_hash()
        return data


CANONICAL_BASELINE_MANIFEST = SignalForgeCanonicalBaselineManifest()

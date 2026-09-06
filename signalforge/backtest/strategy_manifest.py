"""
signalforge/backtest/strategy_manifest.py — Immutable Backtest Strategy Manifest

Captures and freezes the exact production parameters and configuration hash
for deterministic historical backtesting.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass(frozen=True)
class BacktestStrategyManifest:
    strategy_name: str = "SignalForge_EMA20_Pullback_DynamicBudget"
    strategy_version: str = "2.4.0-production-frozen"
    state_machine_version: str = "3.1.0-pullback-retest"
    candidate_generation_version: str = "2.0.0-pit"
    git_commit_hash: str = "4f8a9b2c7e1d5a6f8b9e0c1d2e3f4a5b6c7d8e9f"
    
    # Core Strategy Parameters (FROZEN)
    ema_period: int = 20
    atr_period: int = 14
    atr_multiplier: float = 1.5
    retest_touch_tolerance_pts: float = 2.0
    confirmation_candle_bars: int = 1
    state_machine_horizon_bars: int = 6
    stop_loss_option_pts: float = 15.0
    target_option_pts: float = 30.0
    
    # Contract Selection
    underlying_symbol: str = "NIFTY"
    strike_selection_mode: str = "ATM_PLUS_MINUS_1"
    expiry_selection_mode: str = "NEAREST_WEEKLY"
    default_lot_size: int = 65
    
    # Budget Sizing Rules (FROZEN)
    normal_trade_budget: float = 30000.0
    reduced_trade_budget: float = 15000.0
    max_capital_allocation_pct: float = 0.15
    starting_trading_capital: float = 250000.0
    
    # Execution and Statutory Cost Model
    brokerage_per_order: float = 20.0
    statutory_fees_per_lot: float = 59.20  # GST, STT, Exchange, SEBI, Stamp
    default_slippage_pts: float = 0.02
    intrabar_ordering_assumption: str = "CONSERVATIVE_STOP_FIRST"

    def compute_manifest_hash(self) -> str:
        serialized = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["manifest_hash"] = self.compute_manifest_hash()
        return data


FROZEN_BACKTEST_MANIFEST = BacktestStrategyManifest()

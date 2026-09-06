"""
core/strategies/backtest_models.py — Data Models for Commodity Backtesting
==========================================================================
Defines all domain models, configurations, and analytical structures for
institutional MCX commodity backtesting:
- Slippage models (Fixed Ticks, Fixed Points, Percentage)
- Ambiguity rules for same-candle SL/Target touch
- Contract modes (Individual vs Continuous)
- Trade records with MFE/MAE, statutory fees, slippage attribution
- Equity curve tracking and regime/time-of-day breakdowns
- Reproducible BacktestConfig and serializable BacktestResult
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date, time
from enum import Enum
import json
import os
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import pytz

from core.regime.engine import MarketRegime

IST = pytz.timezone("Asia/Kolkata")


class SlippageType(str, Enum):
    FIXED_TICKS = "FIXED_TICKS"
    FIXED_POINTS = "FIXED_POINTS"
    PERCENTAGE = "PERCENTAGE"


@dataclass
class SlippageModel:
    """Configurable slippage model for sensitivity analysis."""
    slippage_type: SlippageType = SlippageType.FIXED_TICKS
    value: float = 1.0  # 1 tick, 1 point, or 0.05%

    def calculate_slippage_points(self, price: float, tick_size: float = 1.0) -> float:
        """Computes slippage in points based on type."""
        if self.slippage_type == SlippageType.FIXED_TICKS:
            return round(self.value * tick_size, 4)
        elif self.slippage_type == SlippageType.FIXED_POINTS:
            return round(self.value, 4)
        elif self.slippage_type == SlippageType.PERCENTAGE:
            raw_pts = price * (self.value / 100.0)
            return round(round(raw_pts / tick_size) * tick_size, 4)
        return round(tick_size, 4)


class ContractMode(str, Enum):
    INDIVIDUAL_CONTRACT = "INDIVIDUAL_CONTRACT"
    CONTINUOUS_CONTRACT = "CONTINUOUS_CONTRACT"


class SameBarAmbiguityRule(str, Enum):
    """
    Deterministic rule when both SL and Target are breached in the same bar.
    Defaults to STOP_FIRST (conservative institutional assumption).
    """
    STOP_FIRST = "STOP_FIRST"          # Assume stop loss hit first (conservative)
    TARGET_FIRST = "TARGET_FIRST"      # Assume target hit first (aggressive)
    WORST_CASE = "WORST_CASE"          # Same as STOP_FIRST
    SPLIT_50_50 = "SPLIT_50_50"        # Random 50/50 probability


@dataclass
class BacktestConfig:
    """Complete, reproducible configuration for a backtest run."""
    backtest_id: str
    strategy_name: str
    strategy_version: str
    instrument_symbol: str
    timeframe: str = "5m"
    contract_mode: ContractMode = ContractMode.CONTINUOUS_CONTRACT
    contract_symbol: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    initial_capital_inr: float = 200000.0
    lots: int = 1
    slippage: SlippageModel = field(default_factory=SlippageModel)
    same_bar_rule: SameBarAmbiguityRule = SameBarAmbiguityRule.STOP_FIRST
    enable_intraday_eod_squareoff: bool = True
    eod_squareoff_time: str = "23:15"
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)
    oos_split_pct: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now(IST).isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["slippage"]["slippage_type"] = self.slippage.slippage_type.value
        d["contract_mode"] = self.contract_mode.value
        d["same_bar_rule"] = self.same_bar_rule.value
        return d


@dataclass
class BacktestTrade:
    """Detailed record of a single completed backtest trade."""
    trade_id: str
    strategy: str
    instrument: str
    contract: str
    direction: str                     # "BUY" (Long) or "SELL" (Short)
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    stop_loss: float
    target: float
    quantity: int
    lots: int
    gross_pnl_inr: float
    net_pnl_inr: float
    pnl_points: float
    slippage_pts: float
    slippage_cost_inr: float
    fees_inr: float
    mfe_pts: float
    mae_pts: float
    hold_bars: int
    hold_minutes: float
    regime: str
    time_of_day_bucket: str
    exit_reason: str
    is_ambiguous_bar: bool = False
    is_oos: bool = False               # Out-Of-Sample validation flag

    def to_dict(self) -> dict:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        d["exit_time"] = self.exit_time.isoformat()
        return d


@dataclass
class EquityPoint:
    timestamp: datetime
    trade_id: str
    gross_pnl_inr: float
    cumulative_gross_pnl: float
    fees_inr: float
    cumulative_fees: float
    slippage_cost_inr: float
    cumulative_slippage: float
    net_pnl_inr: float
    cumulative_net_pnl: float
    drawdown_inr: float
    drawdown_pct: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    expectancy_points: float = 0.0
    expectancy_inr: float = 0.0
    avg_win_inr: float = 0.0
    avg_loss_inr: float = 0.0
    largest_win_inr: float = 0.0
    largest_loss_inr: float = 0.0
    win_loss_ratio: float = 0.0
    avg_holding_time_minutes: float = 0.0
    gross_pnl_inr: float = 0.0
    total_costs_inr: float = 0.0
    total_slippage_inr: float = 0.0
    net_pnl_inr: float = 0.0
    max_drawdown_inr: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_mfe_points: float = 0.0
    avg_mae_points: float = 0.0
    mfe_mae_ratio: float = 0.0
    sl_hit_count: int = 0
    target_hit_count: int = 0
    strategy_exit_count: int = 0
    eod_exit_count: int = 0
    ambiguous_bar_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    """Comprehensive, fully serializable result bundle of a backtest execution."""
    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: List[BacktestTrade]
    equity_curve: List[EquityPoint]
    regime_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    time_of_day_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    direction_breakdown: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    oos_metrics: Optional[PerformanceMetrics] = None

    def save_to_dir(self, output_dir: str) -> str:
        """Persists all outputs to disk cleanly."""
        target_path = os.path.join(
            output_dir,
            self.config.instrument_symbol,
            self.config.strategy_name,
            self.config.backtest_id,
        )
        os.makedirs(target_path, exist_ok=True)

        # 1. configuration.json
        with open(os.path.join(target_path, "configuration.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # 2. summary.json
        summary = {
            "metrics": self.metrics.to_dict(),
            "direction_breakdown": self.direction_breakdown,
            "oos_metrics": self.oos_metrics.to_dict() if self.oos_metrics else None,
        }
        with open(os.path.join(target_path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # 3. trades.csv
        if self.trades:
            df_trades = pd.DataFrame([t.to_dict() for t in self.trades])
            df_trades.to_csv(os.path.join(target_path, "trades.csv"), index=False)

        # 4. equity_curve.csv
        if self.equity_curve:
            df_equity = pd.DataFrame([e.to_dict() for e in self.equity_curve])
            df_equity.to_csv(os.path.join(target_path, "equity_curve.csv"), index=False)

        # 5. regime_analysis.csv
        if self.regime_breakdown:
            df_regime = pd.DataFrame.from_dict(self.regime_breakdown, orient="index")
            df_regime.to_csv(os.path.join(target_path, "regime_analysis.csv"))

        # 6. time_analysis.csv
        if self.time_of_day_breakdown:
            df_time = pd.DataFrame.from_dict(self.time_of_day_breakdown, orient="index")
            df_time.to_csv(os.path.join(target_path, "time_analysis.csv"))

        return target_path

    def __iter__(self):
        """Allows unpacking as (trades, report, df_journal) for backward compatibility."""
        df_journal = pd.DataFrame([t.to_dict() for t in self.trades]) if self.trades else pd.DataFrame()
        return iter((self.trades, self.metrics, df_journal))

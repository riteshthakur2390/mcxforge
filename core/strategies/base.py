"""
core/strategies/base.py — Base Interface for MCX Commodity Strategies
====================================================================
Standard interface and signal models for all MCXForge commodity futures strategies.
Ensures uniform lifecycle methods:
- initialize()
- calculate_indicators()
- generate_signal()
- validate_signal()
- calculate_stop_loss()
- calculate_target()
- manage_position()
- exit_signal()

Guarantees:
- Zero hardcoded commodity assumptions (reads lot_size, tick_size, session from InstrumentConfig)
- Strict integration with MarketRegimeEngine and MarketShieldLayer
- Comprehensive signal telemetry for 4-month forward validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import pytz

from core.models import Direction, Position, JournalEntry
from core.regime.engine import MarketRegime, RegimeDetails
from instruments.base import InstrumentConfig, ContractSpec

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class StrategySignal:
    """
    Standard structured trade signal produced by all MCXForge commodity strategies.
    Recorded for every signal generated (both accepted and rejected) for 4-month telemetry.
    """
    timestamp: datetime
    instrument: str                         # e.g. "SILVERMIC"
    contract: str                           # e.g. "SILVERMIC-30Nov2026-FUT"
    strategy: str                           # e.g. "TrendFollowing"
    strategy_version: str                   # e.g. "1.0.0"
    direction: Direction                    # Direction.BUY, Direction.SELL, Direction.NONE
    confidence: float = 0.0                 # 0.0 to 1.0
    entry_price: float = 0.0                # Suggested entry price
    stop_loss: float = 0.0                  # Suggested stop loss level
    target: float = 0.0                     # Suggested target level
    risk_reward: float = 0.0                # Target distance / SL distance
    regime: MarketRegime = MarketRegime.RANGE
    reason: str = ""                        # Explainable setup description
    indicators: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""              # Reason if vetoed by regime/risk/shield
    decision: str = "WAIT"                  # "TRADE", "WAIT", "NO_TRADE", "ABNORMAL"
    is_valid: bool = False                  # True only if tradeable and unvetoed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["direction"] = self.direction.value
        d["regime"] = self.regime.value
        return d

    def to_journal_entry(
        self,
        execution_mode: str = "OBSERVE",
        actual_entry: float = 0.0,
        actual_exit: float = 0.0,
        slippage_pts: float = 0.0,
        costs: float = 0.0,
        pnl_inr: float = 0.0,
        mfe_points: float = 0.0,
        mae_points: float = 0.0,
        exit_reason: str = "",
    ) -> JournalEntry:
        """Converts signal to 4-month observation journal entry."""
        dt = self.timestamp.astimezone(IST) if self.timestamp.tzinfo else IST.localize(self.timestamp)
        return JournalEntry(
            date=dt.strftime("%Y-%m-%d"),
            time=dt.strftime("%H:%M:%S"),
            symbol=self.instrument,
            direction=self.direction.action,
            market_price=self.entry_price,
            strategies_fired=self.strategy,
            votes=1,
            strategy_conf=self.confidence,
            regime=self.regime.value,
            contract_symbol=self.contract,
            strategy_version=self.strategy_version,
            entry_price=self.entry_price,
            actual_entry=actual_entry or self.entry_price,
            sl_price=self.stop_loss,
            target_price=self.target,
            exit_price=actual_exit,
            actual_exit=actual_exit,
            slippage_pts=slippage_pts,
            costs=costs,
            risk_reward=self.risk_reward,
            mode=execution_mode,
            execution="EXECUTED" if self.is_valid and execution_mode != "OBSERVE" else "OBSERVED",
            pnl_inr=pnl_inr,
            mfe_points=mfe_points,
            mae_points=mae_points,
            rejection_reason=self.rejection_reason,
            exit_reason=exit_reason,
            llm_rationale=self.reason,
        )


class BaseCommodityStrategy(ABC):
    """
    Abstract Base Class for all MCXForge Commodity Trading Strategies.
    Every strategy operates on instrument specifications supplied by MCXForge
    and adheres to common execution, regime, and risk requirements.
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        permitted_regimes: Optional[List[MarketRegime]] = None,
        default_parameters: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.version = version
        self.permitted_regimes = permitted_regimes or [MarketRegime.TREND, MarketRegime.BREAKOUT]
        self.default_parameters = default_parameters or {}
        self.parameters: Dict[str, Any] = self.default_parameters.copy()
        self.instrument_config: Optional[InstrumentConfig] = None
        self.is_initialized: bool = False

    def initialize(
        self,
        instrument_config: InstrumentConfig,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initializes the strategy with instrument metadata and custom parameters.
        No hardcoded instrument parameters are permitted.
        """
        self.instrument_config = instrument_config
        self.parameters = self.default_parameters.copy()
        if parameters:
            self.parameters.update(parameters)
        self.is_initialized = True

    def round_to_tick(self, price: float) -> float:
        """Rounds a given price to the instrument's official minimum tick size."""
        if not self.instrument_config or self.instrument_config.tick_size <= 0:
            return round(price, 2)
        tick = self.instrument_config.tick_size
        return round(round(price / tick) * tick, 4)

    def is_within_session(self, current_dt: datetime) -> bool:
        """Checks if timestamp falls within instrument trading hours."""
        if not self.instrument_config:
            return True
        session = self.instrument_config.session
        open_t = datetime.strptime(session.open_time, "%H:%M").time()
        close_t = datetime.strptime(session.close_time, "%H:%M").time()
        curr_t = current_dt.time()
        return open_t <= curr_t <= close_t

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all technical indicators required by the strategy.
        Must operate without future look-ahead bias.
        """
        pass

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_contract: Optional[ContractSpec] = None,
        regime_details: Optional[RegimeDetails] = None,
    ) -> StrategySignal:
        """
        Evaluates current market data and returns a structured StrategySignal.
        """
        pass

    def validate_signal(
        self,
        signal: StrategySignal,
        regime_details: Optional[RegimeDetails] = None,
    ) -> StrategySignal:
        """
        Validates whether the signal matches the permitted market regimes.
        Rejects signal if market regime is incompatible or marked ABNORMAL.
        """
        if not signal or signal.direction == Direction.NONE:
            signal.decision = "NO_TRADE"
            signal.is_valid = False
            return signal

        if regime_details:
            r_obj = getattr(regime_details, "regime", None) or (
                regime_details.get("regime") if isinstance(regime_details, dict) else None
            )
            if r_obj:
                r_val = r_obj.value if hasattr(r_obj, "value") else str(r_obj)
                reasons = getattr(regime_details, "reasons", []) if hasattr(regime_details, "reasons") else (
                    regime_details.get("reasons", []) if isinstance(regime_details, dict) else []
                )
                if r_val == MarketRegime.ABNORMAL.value or r_obj == MarketRegime.ABNORMAL:
                    signal.rejection_reason = f"REGIME_ABNORMAL_VETO: {','.join(reasons)}"
                    signal.decision = "ABNORMAL"
                    signal.is_valid = False
                    return signal

                permitted = [r.value if hasattr(r, "value") else str(r) for r in self.permitted_regimes]
                if r_val not in permitted and r_obj not in self.permitted_regimes:
                    signal.rejection_reason = (
                        f"REGIME_INCOMPATIBLE: Strategy '{self.name}' requires {permitted}, got {r_val}"
                    )
                    signal.decision = "NO_TRADE"
                    signal.is_valid = False
                    return signal

        # Validate minimum distance for stop loss and target
        if signal.entry_price > 0 and signal.stop_loss > 0:
            sl_dist = abs(signal.entry_price - signal.stop_loss)
            min_tick = self.instrument_config.tick_size if self.instrument_config else 1.0
            if sl_dist < min_tick:
                signal.rejection_reason = f"SL_DISTANCE_TOO_SMALL ({sl_dist:.2f} < {min_tick:.2f})"
                signal.decision = "NO_TRADE"
                signal.is_valid = False
                return signal

        signal.is_valid = True
        signal.decision = "TRADE"
        return signal

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: Direction,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        """Calculates risk-managed stop loss price."""
        mult = float(self.parameters.get("sl_atr_mult", self.parameters.get("atr_sl_multiplier", 1.5)))
        sl_dist = max(atr * mult, (self.instrument_config.tick_size * 2) if self.instrument_config else 2.0)
        if direction.is_long:
            return self.round_to_tick(entry_price - sl_dist)
        else:
            return self.round_to_tick(entry_price + sl_dist)

    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        direction: Direction,
        rr_ratio: Optional[float] = None,
    ) -> float:
        """Calculates target exit price based on risk-reward geometry."""
        rr = rr_ratio if rr_ratio is not None else float(self.parameters.get("risk_reward_ratio", 2.0))
        risk_dist = abs(entry_price - stop_loss)
        target_dist = risk_dist * rr
        if direction.is_long:
            return self.round_to_tick(entry_price + target_dist)
        else:
            return self.round_to_tick(entry_price - target_dist)

    def manage_position(
        self,
        position: Position,
        current_price: float,
        df: pd.DataFrame,
    ) -> Optional[dict]:
        """
        Evaluates active open position for dynamic trailing stop adjustments or breakeven arming.
        Returns modification dict if adjustment is warranted:
        {"action": "UPDATE_SL", "new_sl": float, "reason": str}
        """
        if not position or not position.is_open:
            return None

        trailing_mult = self.parameters.get("trailing_stop_atr_mult", 0.0)
        if trailing_mult <= 0.0 or len(df) < 5:
            return None

        # Calculate current ATR
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            )
        )
        curr_atr = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        trail_distance = curr_atr * trailing_mult

        is_long = position.plan.signal.direction.is_long if position.plan and position.plan.signal else True
        curr_sl = position.sl_premium

        if is_long:
            new_sl = self.round_to_tick(current_price - trail_distance)
            if new_sl > curr_sl and new_sl < current_price:
                return {
                    "action": "UPDATE_SL",
                    "new_sl": new_sl,
                    "reason": f"TRAILING_STOP_LONG (distance {trail_distance:.1f})"
                }
        else:
            new_sl = self.round_to_tick(current_price + trail_distance)
            if (curr_sl <= 0 or new_sl < curr_sl) and new_sl > current_price:
                return {
                    "action": "UPDATE_SL",
                    "new_sl": new_sl,
                    "reason": f"TRAILING_STOP_SHORT (distance {trail_distance:.1f})"
                }

        return None

    def exit_signal(
        self,
        position: Position,
        current_price: float,
        df: pd.DataFrame,
        regime_details: Optional[RegimeDetails] = None,
    ) -> Optional[dict]:
        """
        Evaluates whether an active position should be exited prematurely
        (e.g., trend reversal, regime change to ABNORMAL, or max hold timeout).
        """
        if not position or not position.is_open:
            return None

        # Veto immediately if abnormal conditions detected
        if regime_details and regime_details.regime == MarketRegime.ABNORMAL:
            return {
                "exit": True,
                "reason": f"REGIME_ABNORMAL_EMERGENCY_EXIT ({','.join(regime_details.reasons)})",
                "price": current_price
            }

        # Check maximum holding period (if configured)
        max_bars = self.parameters.get("max_holding_bars", 0)
        if max_bars > 0:
            entry_idx = getattr(position, "entry_bar_index", None)
            if entry_idx is not None and (len(df) - entry_idx) >= max_bars:
                return {
                    "exit": True,
                    "reason": f"MAX_HOLDING_PERIOD_EXPIRED ({max_bars} bars)",
                    "price": current_price
                }

        return None

    def evaluate(
        self,
        df: pd.DataFrame,
        orb_high: Optional[float] = None,
        orb_low: Optional[float] = None,
        cache: Any = None,
        **kwargs
    ) -> dict:
        """
        Compatibility adapter for legacy multi-strategy consensus runner.
        """
        contract = kwargs.get("contract")
        regime_details = kwargs.get("regime_details")
        if isinstance(regime_details, dict):
            reg_val = str(regime_details.get("regime", regime_details.get("regime_label", "RANGE"))).upper()
            if "TREND" in reg_val:
                market_regime = MarketRegime.TREND
            elif "BREAKOUT" in reg_val:
                market_regime = MarketRegime.BREAKOUT
            elif "HIGH_VOL" in reg_val:
                market_regime = MarketRegime.HIGH_VOLATILITY
            elif "LOW_VOL" in reg_val:
                market_regime = MarketRegime.LOW_VOLATILITY
            elif "ABNORMAL" in reg_val:
                market_regime = MarketRegime.ABNORMAL
            else:
                market_regime = MarketRegime.RANGE

            regime_details = RegimeDetails(
                regime=market_regime,
                confidence=float(regime_details.get("confidence", 0.80) or 0.80),
                adx=float(regime_details.get("adx", 20.0) or 20.0),
                atr=float(regime_details.get("atr", 25.0) or 25.0),
                choppiness=float(regime_details.get("chop_index", regime_details.get("choppiness", 50.0)) or 50.0),
                is_tradeable=bool(regime_details.get("is_tradeable", True)),
                reasons=list(regime_details.get("reasons", []) or []),
            )
            kwargs["regime_details"] = regime_details
        try:
            signal = self.generate_signal(
                df,
                current_contract=contract,
                regime_details=regime_details,
                orb_high=orb_high,
                orb_low=orb_low,
                cache=cache,
                **kwargs
            )
        except TypeError:
            try:
                signal = self.generate_signal(df, current_contract=contract, regime_details=regime_details)
            except TypeError:
                try:
                    signal = self.generate_signal(df, regime_details=regime_details)
                except TypeError:
                    signal = self.generate_signal(df)
        try:
            signal = self.validate_signal(signal, regime_details=regime_details)
        except TypeError:
            signal = self.validate_signal(signal)

        direction = signal.direction if signal and signal.is_valid else Direction.NONE
        if direction in (Direction.BUY, "BUY"):
            direction = Direction.BUY_CALL
        elif direction in (Direction.SELL, "SELL"):
            direction = Direction.BUY_PUT

        return {
            "name": self.name,
            "direction": direction,
            "confidence": float(signal.confidence if (signal and signal.is_valid) else 0.0),
            "entry_price": float(signal.entry_price if signal else 0.0),
            "stop_loss": float(signal.stop_loss if signal else 0.0),
            "target": float(signal.target if signal else 0.0),
            "decision": str(signal.decision if signal else "WAIT"),
            "rejection_reason": str(signal.rejection_reason if signal else ""),
            "signal_object": signal,
        }

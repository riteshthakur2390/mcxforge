"""
core/strategies/donchian_breakout.py — Donchian Channel Breakout Strategy Family
================================================================================
Classic trend and breakout benchmark system for MCX commodity futures.
Implements the clean Richard Donchian channel breakout logic:
- Entry: Breakout of N-period High (Long) or N-period Low (Short).
- Exit: Dynamic trailing channel exit (M-period channel break) or risk-managed ATR stop.

Keeps filtering minimal to serve as an uncurve-fitted statistical benchmark.
"""

from datetime import datetime
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import pytz

from core.models import Direction, Position
from core.regime.engine import MarketRegime, RegimeDetails
from instruments.base import ContractSpec
from core.strategies.base import BaseCommodityStrategy, StrategySignal

IST = pytz.timezone("Asia/Kolkata")


class DonchianBreakoutStrategy(BaseCommodityStrategy):
    """
    Classic Donchian Channel breakout benchmark system.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "entry_lookback": 20,                # Lookback for breakout entry channel
            "exit_lookback": 10,                 # Lookback for trailing channel exit
            "breakout_buffer_atr_mult": 0.05,    # Minor buffer to avoid noisy single-tick touches
            "atr_period": 14,
            "atr_sl_multiplier": 2.0,            # Hard emergency stop distance
            "risk_reward_ratio": 2.5,
            "min_volume_ratio": 0.9,             # Light volume filter to avoid dead illiquid bars
            "max_holding_bars": 150,
        }
        super().__init__(
            name="DonchianBreakout",
            version="1.0.0",
            permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        c = data["close"]
        h = data["high"]
        l = data["low"]
        v = data["volume"]

        entry_n = self.parameters["entry_lookback"]
        exit_n = self.parameters["exit_lookback"]

        # Shift by 1 bar to strictly avoid look-ahead bias on current bar!
        data["donchian_high"] = h.rolling(entry_n, min_periods=1).max().shift(1)
        data["donchian_low"] = l.rolling(entry_n, min_periods=1).min().shift(1)

        data["donchian_exit_high"] = h.rolling(exit_n, min_periods=1).max().shift(1)
        data["donchian_exit_low"] = l.rolling(exit_n, min_periods=1).min().shift(1)

        # ATR
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["tr"] = tr
        data["atr"] = tr.rolling(self.parameters["atr_period"], min_periods=1).mean()

        # Volume MA
        data["vol_ma"] = v.rolling(20, min_periods=1).mean()
        data["vol_ratio"] = v / data["vol_ma"].replace(0, 1e-6)

        return data

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_contract: Optional[ContractSpec] = None,
        regime_details: Optional[RegimeDetails] = None,
    ) -> StrategySignal:
        now_ts = datetime.now(IST)
        if len(df) > 0 and isinstance(df.index[-1], (datetime, pd.Timestamp)):
            now_ts = df.index[-1].to_pydatetime()

        contract_sym = current_contract.symbol if current_contract else (
            self.instrument_config.symbol if self.instrument_config else "MCX_FUT"
        )
        instrument_name = self.instrument_config.symbol if self.instrument_config else "COMMODITY"

        empty_signal = StrategySignal(
            timestamp=now_ts,
            instrument=instrument_name,
            contract=contract_sym,
            strategy=self.name,
            strategy_version=self.version,
            direction=Direction.NONE,
            decision="NO_TRADE",
        )

        min_bars = self.parameters["entry_lookback"] + 5
        if len(df) < min_bars:
            empty_signal.rejection_reason = "INSUFFICIENT_DATA"
            return empty_signal

        # Check session timing
        if not self.is_within_session(now_ts):
            empty_signal.rejection_reason = "OUTSIDE_TRADING_SESSION"
            return empty_signal

        data = self.calculate_indicators(df)
        curr = data.iloc[-1]

        close = float(curr["close"])
        d_high = float(curr["donchian_high"])
        d_low = float(curr["donchian_low"])
        atr = float(curr["atr"])
        vol_ratio = float(curr["vol_ratio"])
        buffer = self.parameters["breakout_buffer_atr_mult"] * atr

        indicators_snapshot = {
            "close": close,
            "donchian_high": round(d_high, 2),
            "donchian_low": round(d_low, 2),
            "atr": round(atr, 2),
            "vol_ratio": round(vol_ratio, 2),
            "buffer": round(buffer, 2),
        }

        # Volume threshold
        if vol_ratio < self.parameters["min_volume_ratio"]:
            empty_signal.indicators = indicators_snapshot
            empty_signal.rejection_reason = f"VOLUME_TOO_LOW ({vol_ratio:.2f} < {self.parameters['min_volume_ratio']})"
            return empty_signal

        # Bullish Breakout: Close > N-period High + buffer
        if close > (d_high + buffer):
            entry_price = self.round_to_tick(close)
            sl = self.calculate_stop_loss(entry_price, Direction.BUY, atr, data)
            target = self.calculate_target(entry_price, sl, Direction.BUY)
            rr = round(abs(target - entry_price) / max(abs(entry_price - sl), 1e-6), 2)
            conf = min(0.95, round(0.65 + min(vol_ratio * 0.08, 0.25), 2))

            return StrategySignal(
                timestamp=now_ts,
                instrument=instrument_name,
                contract=contract_sym,
                strategy=self.name,
                strategy_version=self.version,
                direction=Direction.BUY,
                confidence=conf,
                entry_price=entry_price,
                stop_loss=sl,
                target=target,
                risk_reward=rr,
                regime=regime_details.regime if regime_details else MarketRegime.BREAKOUT,
                reason=f"Donchian {self.parameters['entry_lookback']}-period High Breakout above {d_high:.1f} + buffer {buffer:.1f}",
                indicators=indicators_snapshot,
                decision="TRADE",
                is_valid=True,
            )

        # Bearish Breakdown: Close < N-period Low - buffer
        if close < (d_low - buffer):
            entry_price = self.round_to_tick(close)
            sl = self.calculate_stop_loss(entry_price, Direction.SELL, atr, data)
            target = self.calculate_target(entry_price, sl, Direction.SELL)
            rr = round(abs(entry_price - target) / max(abs(sl - entry_price), 1e-6), 2)
            conf = min(0.95, round(0.65 + min(vol_ratio * 0.08, 0.25), 2))

            return StrategySignal(
                timestamp=now_ts,
                instrument=instrument_name,
                contract=contract_sym,
                strategy=self.name,
                strategy_version=self.version,
                direction=Direction.SELL,
                confidence=conf,
                entry_price=entry_price,
                stop_loss=sl,
                target=target,
                risk_reward=rr,
                regime=regime_details.regime if regime_details else MarketRegime.BREAKOUT,
                reason=f"Donchian {self.parameters['entry_lookback']}-period Low Breakdown below {d_low:.1f} - buffer {buffer:.1f}",
                indicators=indicators_snapshot,
                decision="TRADE",
                is_valid=True,
            )

        empty_signal.indicators = indicators_snapshot
        empty_signal.rejection_reason = "NO_DONCHIAN_BREAKOUT"
        return empty_signal

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: Direction,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        # Emergency stop loss based on ATR or exit lookback channel
        mult = self.parameters["atr_sl_multiplier"]
        sl_dist = atr * mult
        min_tick = self.instrument_config.tick_size if self.instrument_config else 1.0

        if direction.is_long:
            exit_low = float(df["donchian_exit_low"].iloc[-1]) if "donchian_exit_low" in df.columns else (entry_price - sl_dist)
            sl = max(entry_price - sl_dist, exit_low)
            return self.round_to_tick(min(sl, entry_price - (min_tick * 4)))
        else:
            exit_high = float(df["donchian_exit_high"].iloc[-1]) if "donchian_exit_high" in df.columns else (entry_price + sl_dist)
            sl = min(entry_price + sl_dist, exit_high)
            return self.round_to_tick(max(sl, entry_price + (min_tick * 4)))

    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        direction: Direction,
        rr_ratio: Optional[float] = None,
    ) -> float:
        rr = rr_ratio or self.parameters["risk_reward_ratio"]
        risk_dist = abs(entry_price - stop_loss)
        target_dist = risk_dist * rr

        if direction.is_long:
            return self.round_to_tick(entry_price + target_dist)
        else:
            return self.round_to_tick(entry_price - target_dist)

    def exit_signal(
        self,
        position: Position,
        current_price: float,
        df: pd.DataFrame,
        regime_details: Optional[RegimeDetails] = None,
    ) -> Optional[dict]:
        base_exit = super().exit_signal(position, current_price, df, regime_details)
        if base_exit:
            return base_exit

        if len(df) < self.parameters["exit_lookback"] + 2:
            return None

        data = self.calculate_indicators(df)
        curr = data.iloc[-1]
        exit_high = float(curr["donchian_exit_high"])
        exit_low = float(curr["donchian_exit_low"])

        is_long = position.plan.signal.direction.is_long if position.plan and position.plan.signal else True

        # Classic Donchian/Turtle channel exit:
        # Long exits when price falls below M-period low
        if is_long and current_price < exit_low:
            return {
                "exit": True,
                "reason": f"DONCHIAN_CHANNEL_EXIT_LONG (Price {current_price:.1f} < {self.parameters['exit_lookback']}-period Low {exit_low:.1f})",
                "price": current_price
            }
        # Short exits when price rises above M-period high
        elif (not is_long) and current_price > exit_high:
            return {
                "exit": True,
                "reason": f"DONCHIAN_CHANNEL_EXIT_SHORT (Price {current_price:.1f} > {self.parameters['exit_lookback']}-period High {exit_high:.1f})",
                "price": current_price
            }

        return None

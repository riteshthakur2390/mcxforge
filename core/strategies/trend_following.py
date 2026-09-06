"""
core/strategies/trend_following.py — Trend Following Strategy Family
====================================================================
Identifies genuine directional momentum in MCX commodity futures using
EMA structural alignment, ADX trend strength confirmation, and ATR risk bands.

Avoids simplistic raw MA crosses by requiring price action confirmation above/below
structural EMAs, directional indicator confirmation (+DI/-DI), and volatility filtering.
"""

from datetime import datetime
from typing import Dict, Optional, Any, List
import numpy as np
import pandas as pd
import pytz

from core.models import Direction, Position
from core.regime.engine import MarketRegime, RegimeDetails
from instruments.base import ContractSpec
from core.strategies.base import BaseCommodityStrategy, StrategySignal

IST = pytz.timezone("Asia/Kolkata")


class TrendFollowingStrategy(BaseCommodityStrategy):
    """
    Directional trend following engine for MCX commodity futures.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "fast_ema_period": 9,
            "slow_ema_period": 21,
            "trend_ema_period": 50,
            "adx_period": 14,
            "adx_threshold": 22.0,
            "atr_period": 14,
            "atr_sl_multiplier": 1.8,
            "risk_reward_ratio": 2.0,
            "trailing_stop_atr_mult": 1.5,
            "min_volume_ratio": 0.8,
            "max_holding_bars": 120,
        }
        super().__init__(
            name="TrendFollowing",
            version="1.0.0",
            permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes EMAs, ADX, ATR, and volume ratios without look-ahead bias."""
        data = df.copy()
        c = data["close"]
        h = data["high"]
        l = data["low"]
        v = data["volume"]

        # EMAs
        fast_p = self.parameters["fast_ema_period"]
        slow_p = self.parameters["slow_ema_period"]
        trend_p = self.parameters["trend_ema_period"]

        data["ema_fast"] = c.ewm(span=fast_p, adjust=False).mean()
        data["ema_slow"] = c.ewm(span=slow_p, adjust=False).mean()
        data["ema_trend"] = c.ewm(span=min(trend_p, len(c)), adjust=False).mean()

        # ATR
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["tr"] = tr
        data["atr"] = tr.rolling(self.parameters["atr_period"], min_periods=1).mean()

        # ADX (+DI / -DI)
        plus_dm = h.diff().where(lambda x: (x > 0) & (x > -l.diff()), 0.0)
        minus_dm = (-l.diff()).where(lambda x: (x > 0) & (x > h.diff()), 0.0)
        atr_series = data["atr"].replace(0, 1e-6)
        plus_di = 100.0 * (plus_dm.rolling(self.parameters["adx_period"], min_periods=1).mean() / atr_series)
        minus_di = 100.0 * (minus_dm.rolling(self.parameters["adx_period"], min_periods=1).mean() / atr_series)
        denom = (plus_di + minus_di).replace(0, 1e-6)
        dx = 100.0 * (plus_di - minus_di).abs() / denom
        data["plus_di"] = plus_di
        data["minus_di"] = minus_di
        data["adx"] = dx.rolling(self.parameters["adx_period"], min_periods=1).mean()

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
        min_bars = max(self.parameters["trend_ema_period"], 30)
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

        if len(df) < min_bars:
            empty_signal.rejection_reason = f"INSUFFICIENT_DATA ({len(df)} < {min_bars})"
            return empty_signal

        # Check session timing
        if not self.is_within_session(now_ts):
            empty_signal.rejection_reason = "OUTSIDE_TRADING_SESSION"
            return empty_signal

        data = self.calculate_indicators(df)
        curr = data.iloc[-1]
        prev = data.iloc[-2]

        close = float(curr["close"])
        ema_fast = float(curr["ema_fast"])
        ema_slow = float(curr["ema_slow"])
        ema_trend = float(curr["ema_trend"])
        adx = float(curr["adx"])
        plus_di = float(curr["plus_di"])
        minus_di = float(curr["minus_di"])
        atr = float(curr["atr"])
        vol_ratio = float(curr["vol_ratio"])

        indicators_snapshot = {
            "close": close,
            "ema_fast": round(ema_fast, 2),
            "ema_slow": round(ema_slow, 2),
            "ema_trend": round(ema_trend, 2),
            "adx": round(adx, 2),
            "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2),
            "atr": round(atr, 2),
            "vol_ratio": round(vol_ratio, 2),
        }

        # 1. Evaluate Bullish Trend
        bullish_ema = (ema_fast > ema_slow) and (close > ema_fast) and (close > ema_trend)
        bullish_adx = (adx >= self.parameters["adx_threshold"]) and (plus_di > minus_di)
        bullish_vol = vol_ratio >= self.parameters["min_volume_ratio"]

        if bullish_ema and bullish_adx and bullish_vol:
            entry_price = self.round_to_tick(close)
            sl = self.calculate_stop_loss(entry_price, Direction.BUY, atr, data)
            target = self.calculate_target(entry_price, sl, Direction.BUY)
            rr = round(abs(target - entry_price) / max(abs(entry_price - sl), 1e-6), 2)
            conf = min(0.95, round(0.60 + (adx / 100.0) * 0.35, 2))

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
                regime=regime_details.regime if regime_details else MarketRegime.TREND,
                reason=f"Bullish EMA alignment (Fast {ema_fast:.1f} > Slow {ema_slow:.1f}), ADX {adx:.1f} > {self.parameters['adx_threshold']}",
                indicators=indicators_snapshot,
                decision="TRADE",
                is_valid=True,
            )

        # 2. Evaluate Bearish Trend
        bearish_ema = (ema_fast < ema_slow) and (close < ema_fast) and (close < ema_trend)
        bearish_adx = (adx >= self.parameters["adx_threshold"]) and (minus_di > plus_di)
        bearish_vol = vol_ratio >= self.parameters["min_volume_ratio"]

        if bearish_ema and bearish_adx and bearish_vol:
            entry_price = self.round_to_tick(close)
            sl = self.calculate_stop_loss(entry_price, Direction.SELL, atr, data)
            target = self.calculate_target(entry_price, sl, Direction.SELL)
            rr = round(abs(entry_price - target) / max(abs(sl - entry_price), 1e-6), 2)
            conf = min(0.95, round(0.60 + (adx / 100.0) * 0.35, 2))

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
                regime=regime_details.regime if regime_details else MarketRegime.TREND,
                reason=f"Bearish EMA alignment (Fast {ema_fast:.1f} < Slow {ema_slow:.1f}), ADX {adx:.1f} > {self.parameters['adx_threshold']}",
                indicators=indicators_snapshot,
                decision="TRADE",
                is_valid=True,
            )

        empty_signal.indicators = indicators_snapshot
        empty_signal.rejection_reason = "TREND_CONDITIONS_NOT_MET"
        return empty_signal

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: Direction,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        mult = self.parameters["atr_sl_multiplier"]
        sl_dist = max(atr * mult, self.instrument_config.tick_size * 5 if self.instrument_config else 5.0)

        # Also consider structural swing low/high
        recent_bars = df.tail(10)
        if direction.is_long:
            structural_low = float(recent_bars["low"].min())
            atr_sl = entry_price - sl_dist
            sl = max(atr_sl, structural_low - (atr * 0.2))
            return self.round_to_tick(min(sl, entry_price - (self.instrument_config.tick_size * 2 if self.instrument_config else 2.0)))
        else:
            structural_high = float(recent_bars["high"].max())
            atr_sl = entry_price + sl_dist
            sl = min(atr_sl, structural_high + (atr * 0.2))
            return self.round_to_tick(max(sl, entry_price + (self.instrument_config.tick_size * 2 if self.instrument_config else 2.0)))

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

        if len(df) < 5:
            return None

        data = self.calculate_indicators(df)
        curr = data.iloc[-1]
        ema_fast = float(curr["ema_fast"])
        ema_slow = float(curr["ema_slow"])

        is_long = position.plan.signal.direction.is_long if position.plan and position.plan.signal else True

        # Trend reversal exit
        if is_long and ema_fast < ema_slow:
            return {
                "exit": True,
                "reason": "TREND_REVERSAL_EMA_BEARISH_CROSS",
                "price": current_price
            }
        elif (not is_long) and ema_fast > ema_slow:
            return {
                "exit": True,
                "reason": "TREND_REVERSAL_EMA_BULLISH_CROSS",
                "price": current_price
            }

        return None

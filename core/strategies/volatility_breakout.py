"""
core/strategies/volatility_breakout.py — Volatility Breakout Strategy Family
===========================================================================
Identifies price expansion emerging from defined periods of volatility compression.
Implements a strict state progression:
COMPRESSION (Volatility Squeeze) -> EXPANSION -> CONFIRMED BREAKOUT -> ENTRY.

Guards against entering on erratic spikes by enforcing abnormal-volatility ceilings
and requiring consolidation box breakout confirmation with volume surge.
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


class VolatilityBreakoutStrategy(BaseCommodityStrategy):
    """
    Volatility expansion breakout system calibrated for commodity cycles.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "keltner_period": 20,
            "keltner_mult": 1.5,
            "compression_lookback": 15,          # Lookback to verify compression occurred
            "expansion_bar_mult": 1.6,           # Bar range expansion vs recent ATR
            "volume_expansion_mult": 1.3,        # Breakout volume vs 20 SMA
            "abnormal_vol_max_atr_ratio": 2.8,   # Abort if ATR expansion is abnormally wild
            "atr_sl_multiplier": 1.6,
            "risk_reward_ratio": 2.0,
            "trailing_stop_atr_mult": 1.4,
            "max_holding_bars": 75,
        }
        super().__init__(
            name="VolatilityBreakout",
            version="1.0.0",
            permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY],
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

        # 1. ATR (short and baseline)
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["tr"] = tr
        data["atr"] = tr.rolling(14, min_periods=1).mean()
        data["baseline_atr"] = tr.rolling(min(len(tr), 50), min_periods=5).mean()
        data["atr_ratio"] = data["atr"] / data["baseline_atr"].replace(0, 1e-6)

        # 2. Bollinger Bands
        bb_p = self.parameters["bb_period"]
        bb_std = self.parameters["bb_std"]
        data["bb_mid"] = c.rolling(bb_p, min_periods=1).mean()
        bb_dev = c.rolling(bb_p, min_periods=1).std().fillna(0)
        data["bb_upper"] = data["bb_mid"] + (bb_std * bb_dev)
        data["bb_lower"] = data["bb_mid"] - (bb_std * bb_dev)
        data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_mid"].replace(0, 1e-6)

        # 3. Keltner Channels
        kc_p = self.parameters["keltner_period"]
        kc_mult = self.parameters["keltner_mult"]
        data["kc_mid"] = c.ewm(span=kc_p, adjust=False).mean()
        data["kc_upper"] = data["kc_mid"] + (kc_mult * data["atr"])
        data["kc_lower"] = data["kc_mid"] - (kc_mult * data["atr"])

        # 4. Squeeze condition (BB completely inside Keltner Channel)
        data["in_squeeze"] = (data["bb_upper"] < data["kc_upper"]) & (data["bb_lower"] > data["kc_lower"])

        # 5. Volume MA
        data["vol_ma"] = v.rolling(20, min_periods=1).mean()
        data["vol_ratio"] = v / data["vol_ma"].replace(0, 1e-6)

        # 6. Consolidation High / Low of previous compression window
        data["recent_high"] = h.rolling(self.parameters["compression_lookback"], min_periods=1).max().shift(1)
        data["recent_low"] = l.rolling(self.parameters["compression_lookback"], min_periods=1).min().shift(1)

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

        min_bars = max(self.parameters["bb_period"], self.parameters["compression_lookback"]) + 5
        if len(df) < min_bars:
            empty_signal.rejection_reason = "INSUFFICIENT_DATA"
            return empty_signal

        # Check session timing
        if not self.is_within_session(now_ts):
            empty_signal.rejection_reason = "OUTSIDE_TRADING_SESSION"
            return empty_signal

        data = self.calculate_indicators(df)
        curr = data.iloc[-1]
        prev = data.iloc[-2]

        close = float(curr["close"])
        high = float(curr["high"])
        low = float(curr["low"])
        tr = float(curr["tr"])
        atr = float(curr["atr"])
        atr_ratio = float(curr["atr_ratio"])
        vol_ratio = float(curr["vol_ratio"])
        recent_high = float(curr["recent_high"])
        recent_low = float(curr["recent_low"])

        indicators_snapshot = {
            "close": close,
            "atr": round(atr, 2),
            "atr_ratio": round(atr_ratio, 2),
            "vol_ratio": round(vol_ratio, 2),
            "recent_high": round(recent_high, 2),
            "recent_low": round(recent_low, 2),
            "in_squeeze_now": bool(curr["in_squeeze"]),
            "was_in_squeeze": bool(data["in_squeeze"].iloc[-self.parameters["compression_lookback"]:-1].any()),
        }

        # 1. Abnormal Volatility Check (Ceiling)
        if atr_ratio > self.parameters["abnormal_vol_max_atr_ratio"]:
            empty_signal.indicators = indicators_snapshot
            empty_signal.rejection_reason = f"ABNORMAL_VOLATILITY_CEILING ({atr_ratio:.2f} > {self.parameters['abnormal_vol_max_atr_ratio']})"
            empty_signal.decision = "ABNORMAL"
            return empty_signal

        # 2. Check that compression previously occurred within recent lookback
        had_compression = indicators_snapshot["was_in_squeeze"]
        if not had_compression:
            empty_signal.indicators = indicators_snapshot
            empty_signal.rejection_reason = "NO_PRIOR_COMPRESSION_DETECTED"
            return empty_signal

        # 3. Expansion verification: Bar range expanding vs ATR and volume expanding
        is_expanding_bar = tr >= self.parameters["expansion_bar_mult"] * atr
        is_expanding_vol = vol_ratio >= self.parameters["volume_expansion_mult"]

        # Long Breakout: Close breaking above compression ceiling with expansion
        if close > recent_high and (is_expanding_bar or is_expanding_vol):
            entry_price = self.round_to_tick(close)
            sl = self.calculate_stop_loss(entry_price, Direction.BUY, atr, data)
            target = self.calculate_target(entry_price, sl, Direction.BUY)
            rr = round(abs(target - entry_price) / max(abs(entry_price - sl), 1e-6), 2)
            conf = min(0.95, round(0.65 + min(vol_ratio * 0.1, 0.25), 2))

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
                reason=f"Volatility expansion breakout above compression box {recent_high:.1f} (Vol {vol_ratio:.1f}x, ATR ratio {atr_ratio:.2f})",
                indicators=indicators_snapshot,
                decision="TRADE",
                is_valid=True,
            )

        # Short Breakdown: Close breaking below compression floor with expansion
        if close < recent_low and (is_expanding_bar or is_expanding_vol):
            entry_price = self.round_to_tick(close)
            sl = self.calculate_stop_loss(entry_price, Direction.SELL, atr, data)
            target = self.calculate_target(entry_price, sl, Direction.SELL)
            rr = round(abs(entry_price - target) / max(abs(sl - entry_price), 1e-6), 2)
            conf = min(0.95, round(0.65 + min(vol_ratio * 0.1, 0.25), 2))

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
                reason=f"Volatility expansion breakdown below compression box {recent_low:.1f} (Vol {vol_ratio:.1f}x, ATR ratio {atr_ratio:.2f})",
                indicators=indicators_snapshot,
                decision="TRADE",
                is_valid=True,
            )

        empty_signal.indicators = indicators_snapshot
        empty_signal.rejection_reason = "NO_EXPANSION_BREAKOUT"
        return empty_signal

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: Direction,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        mult = self.parameters["atr_sl_multiplier"]
        sl_dist = atr * mult
        min_tick = self.instrument_config.tick_size if self.instrument_config else 1.0

        if direction.is_long:
            return self.round_to_tick(entry_price - max(sl_dist, min_tick * 5))
        else:
            return self.round_to_tick(entry_price + max(sl_dist, min_tick * 5))

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

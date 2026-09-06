"""
core/strategies/ma_crossover.py — Dual Moving Average Crossover Strategy
========================================================================
Institutional Dual Moving Average Trend Strategy:
- Fast EMA (default 9) crossing Slow EMA (default 21).
- Optional Momentum confirmation via Rate of Change (ROC) or 50 EMA baseline.
- Dynamic ATR-based stop loss (1.5 * ATR) and 1:2.0+ risk-to-reward target.
- Trailing stop as trend extends.
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import pytz

from core.models import Direction
from core.regime.engine import MarketRegime, RegimeDetails
from core.strategies.base import BaseCommodityStrategy, StrategySignal

IST = pytz.timezone("Asia/Kolkata")


class MovingAverageCrossoverStrategy(BaseCommodityStrategy):
    """
    EMA(9) / EMA(21) crossover strategy with ATR-based stop loss.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "fast_period": 9,
            "slow_period": 21,
            "trend_filter_period": 50,
            "use_trend_filter": False,
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "target_atr_mult": 3.0,
            "min_roc": 0.0,             # Rate of change confirmation
        }
        super().__init__(
            name="MACrossover",
            version="1.0.0",
            permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]

        fast_p = int(self.parameters.get("fast_period", 9))
        slow_p = int(self.parameters.get("slow_period", 21))
        trend_p = int(self.parameters.get("trend_filter_period", 50))

        data["ema_fast"] = close.ewm(span=fast_p, adjust=False).mean()
        data["ema_slow"] = close.ewm(span=slow_p, adjust=False).mean()
        data["ema_trend"] = close.ewm(span=trend_p, adjust=False).mean()

        # Rate of change (10 bars)
        data["roc"] = ((close - close.shift(10)) / close.shift(10).replace(0, np.nan)) * 100.0

        # ATR
        atr_p = int(self.parameters.get("atr_period", 14))
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr"] = tr.rolling(atr_p).mean()

        return data

    def generate_signal(
        self,
        df: pd.DataFrame,
        regime_details: Optional[RegimeDetails] = None,
        current_time: Optional[datetime] = None,
    ) -> StrategySignal:
        ts = df.index[-1] if hasattr(df.index[-1], "to_pydatetime") else datetime.now(IST)
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        elif not isinstance(ts, datetime):
            ts = datetime.now(IST)

        none_sig = StrategySignal(
            timestamp=ts,
            instrument=self.instrument_config.symbol if self.instrument_config else "MCX",
            contract=f"{self.instrument_config.symbol}-FRONT" if self.instrument_config else "MCX-FRONT",
            strategy=self.name,
            strategy_version=self.version,
            direction=Direction.NONE,
        )

        min_bars = int(self.parameters.get("trend_filter_period", 50)) + 10
        if len(df) < min_bars:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]
        prev = ind.iloc[-2]

        c = float(bar["close"])
        fast_now = float(bar["ema_fast"])
        slow_now = float(bar["ema_slow"])
        fast_prev = float(prev["ema_fast"])
        slow_prev = float(prev["ema_slow"])
        trend_filter = float(bar["ema_trend"])
        roc = float(bar["roc"]) if not np.isnan(bar["roc"]) else 0.0
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        sl_mult = float(self.parameters.get("sl_atr_mult", 1.5))
        tgt_mult = float(self.parameters.get("target_atr_mult", 3.0))
        use_filter = bool(self.parameters.get("use_trend_filter", False))

        sig_dir = Direction.NONE
        conf = 0.0

        # Bullish Crossover: Fast crosses above Slow
        if fast_prev <= slow_prev and fast_now > slow_now:
            if not use_filter or c > trend_filter:
                sig_dir = Direction.BUY
                conf = 0.75 if roc > 0 else 0.65
                sl = self.round_to_tick(c - (atr * sl_mult))
                target = self.round_to_tick(c + (atr * tgt_mult))

        # Bearish Crossover: Fast crosses below Slow
        elif fast_prev >= slow_prev and fast_now < slow_now:
            if not use_filter or c < trend_filter:
                sig_dir = Direction.SELL
                conf = 0.75 if roc < 0 else 0.65
                sl = self.round_to_tick(c + (atr * sl_mult))
                target = self.round_to_tick(c - (atr * tgt_mult))

        if sig_dir == Direction.NONE:
            return none_sig

        rr = round(abs(target - c) / max(abs(c - sl), 1e-4), 2)

        return StrategySignal(
            timestamp=ts,
            instrument=self.instrument_config.symbol if self.instrument_config else "MCX",
            contract=f"{self.instrument_config.symbol}-FRONT" if self.instrument_config else "MCX-FRONT",
            strategy=self.name,
            strategy_version=self.version,
            direction=sig_dir,
            confidence=conf,
            entry_price=c,
            stop_loss=sl,
            target=target,
            risk_reward=rr,
            decision="TRADE",
            is_valid=True,
            metadata={"fast_ema": fast_now, "slow_ema": slow_now, "roc": roc, "atr": atr},
        )

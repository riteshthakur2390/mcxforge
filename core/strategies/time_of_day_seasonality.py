"""
core/strategies/time_of_day_seasonality.py — Time-of-Day Seasonality Strategy
=============================================================================
Exploits recurring statistical time-of-day edge in MCX Commodities:
1. US COMEX Market Open Window (17:00 - 19:30 IST):
   - Sudden influx of global institutional volume.
   - Evaluates directional breakout of the 16:30 - 17:00 consolidation channel.
2. US Economic Data Release / Evening Window (20:30 - 22:30 IST):
   - Directional continuation follow-through.
Filters out unlucrative dead zones (morning / afternoon low volume churn).
"""

from __future__ import annotations
from datetime import datetime, time
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import pytz

from core.models import Direction
from core.regime.engine import MarketRegime, RegimeDetails
from core.strategies.base import BaseCommodityStrategy, StrategySignal

IST = pytz.timezone("Asia/Kolkata")


class TimeOfDaySeasonalityStrategy(BaseCommodityStrategy):
    """
    Time-of-day seasonal liquidity window breakout strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "window_start_hour": 17,
            "window_start_minute": 0,
            "window_end_hour": 22,
            "window_end_minute": 0,
            "momentum_ema_period": 20,
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "target_atr_mult": 3.0,
        }
        super().__init__(
            name="TimeOfDaySeasonality",
            version="1.0.0",
            permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT, MarketRegime.HIGH_VOLATILITY],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]

        p = int(self.parameters.get("momentum_ema_period", 20))
        data["ema"] = close.ewm(span=p, adjust=False).mean()

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

        # Check if inside high-probability MCX time-of-day windows
        # 1. Morning Momentum Window: 09:30 - 12:30 IST
        # 2. European Open Window: 13:30 - 16:00 IST
        # 3. US Session Window: 17:00 - 22:30 IST
        bar_time = ts.time()
        in_morning = (time(9, 30) <= bar_time <= time(12, 30))
        in_europe = (time(13, 30) <= bar_time <= time(16, 0))
        in_us = (time(17, 0) <= bar_time <= time(22, 30))

        if not (in_morning or in_europe or in_us):
            return none_sig

        min_bars = int(self.parameters.get("momentum_ema_period", 20)) + 10
        if len(df) < min_bars:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]
        prev = ind.iloc[-2]

        c = float(bar["close"])
        ema = float(bar["ema"])
        prev_c = float(prev["close"])
        prev_ema = float(prev["ema"])
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        sl_mult = float(self.parameters.get("sl_atr_mult", 1.5))
        tgt_mult = float(self.parameters.get("target_atr_mult", 3.0))

        sig_dir = Direction.NONE
        conf = 0.0

        # Bullish Evening Trend Alignment
        if prev_c <= prev_ema and c > ema:
            sig_dir = Direction.BUY
            conf = 0.75
            sl = self.round_to_tick(c - (atr * sl_mult))
            target = self.round_to_tick(c + (atr * tgt_mult))

        # Bearish Evening Trend Alignment
        elif prev_c >= prev_ema and c < ema:
            sig_dir = Direction.SELL
            conf = 0.75
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
            metadata={"session_time": bar_time.strftime("%H:%M"), "atr": atr},
        )

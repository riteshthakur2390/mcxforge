"""
core/strategies/momentum_volume_breakout.py — Momentum Breakout with Volume Confirmation
========================================================================================
Institutional Breakout Logic:
- Price breaks out of N-bar highest high (e.g. 20-bar Donchian channel top).
- Confirmed by a Volume Spike: Volume > 1.5x of 20-period volume moving average.
- Confirmed by MACD Histogram: MACD histogram > 0 and expanding for longs (or < 0 for shorts).
- Stop loss placed at ATR trailing or middle of breakout candle.
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


class MomentumVolumeBreakoutStrategy(BaseCommodityStrategy):
    """
    Price N-bar breakout confirmed by Volume Spike and MACD histogram.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "channel_period": 20,
            "volume_multiplier": 1.5,   # Volume > 1.5x average
            "volume_sma_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "target_atr_mult": 3.0,
        }
        super().__init__(
            name="MomentumVolumeBreakout",
            version="1.0.0",
            permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        p = int(self.parameters.get("channel_period", 20))
        data["channel_high"] = high.shift(1).rolling(p).max()
        data["channel_low"] = low.shift(1).rolling(p).min()

        # Volume SMA
        v_p = int(self.parameters.get("volume_sma_period", 20))
        data["volume_sma"] = volume.rolling(v_p).mean()

        # MACD
        fast_p = int(self.parameters.get("macd_fast", 12))
        slow_p = int(self.parameters.get("macd_slow", 26))
        sig_p = int(self.parameters.get("macd_signal", 9))
        ema_fast = close.ewm(span=fast_p, adjust=False).mean()
        ema_slow = close.ewm(span=slow_p, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=sig_p, adjust=False).mean()
        data["macd_hist"] = macd_line - signal_line

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

        min_bars = int(self.parameters.get("macd_slow", 26)) + 15
        if len(df) < min_bars:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]
        prev = ind.iloc[-2]

        c = float(bar["close"])
        vol = float(bar["volume"])
        vol_sma = float(bar["volume_sma"]) if not np.isnan(bar["volume_sma"]) else 1.0
        ch_high = float(bar["channel_high"])
        ch_low = float(bar["channel_low"])
        macd_hist = float(bar["macd_hist"])
        macd_hist_prev = float(prev["macd_hist"])
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        vol_mult = float(self.parameters.get("volume_multiplier", 1.5))
        sl_mult = float(self.parameters.get("sl_atr_mult", 1.5))
        tgt_mult = float(self.parameters.get("target_atr_mult", 3.0))

        sig_dir = Direction.NONE
        conf = 0.0

        # Bullish Breakout: Close > Channel High + Volume Spike + MACD Hist Positive & Rising
        if c > ch_high and vol >= (vol_sma * vol_mult) and macd_hist > 0 and macd_hist > macd_hist_prev:
            sig_dir = Direction.BUY
            vol_ratio = vol / max(vol_sma, 1.0)
            conf = min(0.88, 0.65 + (vol_ratio - vol_mult) * 0.05)
            sl = self.round_to_tick(c - (atr * sl_mult))
            target = self.round_to_tick(c + (atr * tgt_mult))

        # Bearish Breakdown: Close < Channel Low + Volume Spike + MACD Hist Negative & Falling
        elif c < ch_low and vol >= (vol_sma * vol_mult) and macd_hist < 0 and macd_hist < macd_hist_prev:
            sig_dir = Direction.SELL
            vol_ratio = vol / max(vol_sma, 1.0)
            conf = min(0.88, 0.65 + (vol_ratio - vol_mult) * 0.05)
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
            confidence=round(conf, 2),
            entry_price=c,
            stop_loss=sl,
            target=target,
            risk_reward=rr,
            decision="TRADE",
            is_valid=True,
            metadata={"volume_ratio": round(vol / max(vol_sma, 1.0), 2), "macd_hist": round(macd_hist, 2), "atr": atr},
        )

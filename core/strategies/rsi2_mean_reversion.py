"""
core/strategies/rsi2_mean_reversion.py — Larry Connors 2-Period RSI Mean Reversion Strategy
==========================================================================================
Classic Quantitative Mean Reversion (Larry Connors):
- Major Trend Filter: Close > 200 SMA (for longs) or Close < 200 SMA (for shorts).
- Extreme Pullback Trigger:
  - BUY when 2-period RSI drops below 10 (extreme oversold dip in bull market).
  - Exit when Close crosses above 5-period SMA or ATR profit target.
  - SELL when 2-period RSI rises above 90 (extreme overbought spike in bear market).
  - Exit when Close crosses below 5-period SMA or ATR profit target.
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


class RSI2MeanReversionStrategy(BaseCommodityStrategy):
    """
    Larry Connors RSI(2) Short-Term Mean Reversion Strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "trend_sma_period": 200,     # Major trend filter
            "rsi_period": 2,             # Fast 2-period RSI
            "rsi_oversold": 10.0,        # Buy entry below 10
            "rsi_overbought": 90.0,      # Sell entry above 90
            "exit_sma_period": 5,        # Fast exit target
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "target_atr_mult": 2.5,
        }
        super().__init__(
            name="RSI2MeanReversion",
            version="1.0.0",
            permitted_regimes=[MarketRegime.TREND, MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # 200 SMA
        trend_p = int(self.parameters.get("trend_sma_period", 200))
        # If df has fewer bars than 200, adapt smoothly to 50
        actual_trend_p = trend_p if len(data) >= trend_p else min(len(data) // 2, 50)
        data["sma_trend"] = close.rolling(max(actual_trend_p, 10)).mean()

        # 5 SMA for mean reversion exit
        exit_p = int(self.parameters.get("exit_sma_period", 5))
        data["sma_exit"] = close.rolling(exit_p).mean()

        # 2-period RSI
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain, index=data.index).rolling(2).mean()
        avg_loss = pd.Series(loss, index=data.index).rolling(2).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        data["rsi2"] = 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)

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

        if len(df) < 25:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]

        c = float(bar["close"])
        rsi2 = float(bar["rsi2"]) if not np.isnan(bar["rsi2"]) else 50.0
        sma_trend = float(bar["sma_trend"]) if not np.isnan(bar["sma_trend"]) else c
        sma_exit = float(bar["sma_exit"]) if not np.isnan(bar["sma_exit"]) else c
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        os_thresh = float(self.parameters.get("rsi_oversold", 10.0))
        ob_thresh = float(self.parameters.get("rsi_overbought", 90.0))
        sl_mult = float(self.parameters.get("sl_atr_mult", 1.5))
        tgt_mult = float(self.parameters.get("target_atr_mult", 2.5))

        sig_dir = Direction.NONE
        conf = 0.0

        # Connors Long: Close > 200 SMA AND RSI(2) < 10
        if c > sma_trend and rsi2 <= os_thresh:
            sig_dir = Direction.BUY
            conf = min(0.88, 0.70 + (os_thresh - rsi2) * 0.015)
            sl = self.round_to_tick(c - (atr * sl_mult))
            target = self.round_to_tick(max(sma_exit, c + (atr * tgt_mult)))

        # Connors Short: Close < 200 SMA AND RSI(2) > 90
        elif c < sma_trend and rsi2 >= ob_thresh:
            sig_dir = Direction.SELL
            conf = min(0.88, 0.70 + (rsi2 - ob_thresh) * 0.015)
            sl = self.round_to_tick(c + (atr * sl_mult))
            target = self.round_to_tick(min(sma_exit, c - (atr * tgt_mult)))

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
            metadata={"rsi2": round(rsi2, 2), "sma_trend": round(sma_trend, 2), "atr": atr},
        )

"""
core/strategies/bb_mean_reversion.py — Bollinger Band Mean Reversion Strategy
=============================================================================
Fades price touching outer Bollinger Bands in a ranging market (ADX < 20 filter).
Institutional logic:
- BUY when price touches/dips below lower band AND ADX < 20 (non-trending market).
  Target: Middle Band (20 SMA). Stop Loss: 1.5 * ATR below lower band.
- SELL when price touches/crosses above upper band AND ADX < 20.
  Target: Middle Band (20 SMA). Stop Loss: 1.5 * ATR above upper band.
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


class BollingerBandMeanReversionStrategy(BaseCommodityStrategy):
    """
    Bollinger Band Mean Reversion with ADX ranging filter.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "bb_period": 20,
            "bb_std": 2.0,
            "adx_period": 14,
            "adx_threshold": 28.0,      # Fade extreme band touches even in moderate trends
            "rsi_period": 14,
            "rsi_oversold": 35.0,
            "rsi_overbought": 65.0,
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "risk_reward_ratio": 2.0,
        }
        super().__init__(
            name="BBMeanReversion",
            version="1.0.0",
            permitted_regimes=[MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY, MarketRegime.TREND],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Bollinger Bands
        period = int(self.parameters.get("bb_period", 20))
        std_mult = float(self.parameters.get("bb_std", 2.0))
        sma = close.rolling(period).mean()
        rstd = close.rolling(period).std()
        data["bb_mid"] = sma
        data["bb_upper"] = sma + (std_mult * rstd)
        data["bb_lower"] = sma - (std_mult * rstd)

        # ATR
        atr_p = int(self.parameters.get("atr_period", 14))
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr"] = tr.rolling(atr_p).mean()

        # ADX
        adx_p = int(self.parameters.get("adx_period", 14))
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr_smooth = tr.rolling(adx_p).sum()
        plus_di = 100.0 * (pd.Series(plus_dm, index=data.index).rolling(adx_p).sum() / tr_smooth.replace(0, np.nan))
        minus_di = 100.0 * (pd.Series(minus_dm, index=data.index).rolling(adx_p).sum() / tr_smooth.replace(0, np.nan))
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        data["adx"] = dx.rolling(adx_p).mean().fillna(15.0)

        # RSI
        rsi_p = int(self.parameters.get("rsi_period", 14))
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain, index=data.index).rolling(rsi_p).mean()
        avg_loss = pd.Series(loss, index=data.index).rolling(rsi_p).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        data["rsi"] = 100.0 - (100.0 / (1.0 + rs))

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

        min_bars = int(self.parameters.get("bb_period", 20)) + 15
        if len(df) < min_bars:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]
        prev_bar = ind.iloc[-2]

        c = float(bar["close"])
        l = float(bar["low"])
        h = float(bar["high"])
        bb_upper = float(bar["bb_upper"])
        bb_lower = float(bar["bb_lower"])
        bb_mid = float(bar["bb_mid"])
        adx = float(bar["adx"]) if not np.isnan(bar["adx"]) else 15.0
        rsi = float(bar["rsi"]) if not np.isnan(bar["rsi"]) else 50.0
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        adx_thresh = float(self.parameters.get("adx_threshold", 20.0))
        sl_mult = float(self.parameters.get("sl_atr_mult", 1.5))
        rsi_os = float(self.parameters.get("rsi_oversold", 35.0))
        rsi_ob = float(self.parameters.get("rsi_overbought", 65.0))

        # Filter: market must not be an extreme runaway trend (ADX <= 35)
        if adx > 35.0:
            return none_sig

        sig_dir = Direction.NONE
        conf = 0.0

        # Bullish Mean Reversion: low pierced/touched lower band and closed back above or near it
        if (l <= bb_lower or c <= bb_lower * 1.002) and rsi <= rsi_os:
            sig_dir = Direction.BUY
            conf = min(0.85, 0.60 + (rsi_os - rsi) * 0.01 + (adx_thresh - adx) * 0.005)
            sl = self.round_to_tick(bb_lower - (atr * sl_mult))
            target = self.round_to_tick(bb_mid)
        # Bearish Mean Reversion: high pierced/touched upper band and closed back below or near it
        elif (h >= bb_upper or c >= bb_upper * 0.998) and rsi >= rsi_ob:
            sig_dir = Direction.SELL
            conf = min(0.85, 0.60 + (rsi - rsi_ob) * 0.01 + (adx_thresh - adx) * 0.005)
            sl = self.round_to_tick(bb_upper + (atr * sl_mult))
            target = self.round_to_tick(bb_mid)
        else:
            return none_sig

        rr = round(abs(target - c) / max(abs(c - sl), 1e-4), 2)
        if rr < 0.8:
            # Expand target if middle band is too close
            if sig_dir == Direction.BUY:
                target = self.round_to_tick(c + (abs(c - sl) * 1.5))
            else:
                target = self.round_to_tick(c - (abs(c - sl) * 1.5))
            rr = 1.5

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
            metadata={"bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid, "adx": adx, "rsi": rsi},
        )

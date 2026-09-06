"""
core/strategies/order_flow_delta.py — Order Flow & Cumulative Delta Imbalance Strategy
======================================================================================
Market Microstructure & Order Flow Imbalance:
- Calculates Bar Delta Proxy:
  Delta = Volume * ((Close - Open) / max(High - Low, 0.001))
- Computes Cumulative Volume Delta (CVD) over rolling window (e.g. 14 bars).
- Determines Delta Imbalance:
  - Strong positive cumulative delta with price consolidating -> Institutional absorption / aggressive buyers.
  - Generates BUY when Cumulative Delta crosses above delta threshold and price breaks candle high.
  - Generates SELL when Cumulative Delta crosses below negative delta threshold and price breaks candle low.
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


class OrderFlowDeltaStrategy(BaseCommodityStrategy):
    """
    Cumulative Volume Delta (CVD) Order Flow Imbalance Strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "cvd_window": 14,
            "delta_zscore_threshold": 1.75,  # Statistical significance of delta surge
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "target_atr_mult": 3.0,
        }
        super().__init__(
            name="OrderFlowDelta",
            version="1.0.0",
            permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        open_ = data["open"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # Bar delta estimate from candlestick mechanics
        candle_range = (high - low).replace(0, np.nan)
        body = close - open_
        delta_pct = (body / candle_range).clip(-1.0, 1.0).fillna(0.0)
        data["bar_delta"] = volume * delta_pct

        # Cumulative Delta over window
        w = int(self.parameters.get("cvd_window", 14))
        data["cvd"] = data["bar_delta"].rolling(w).sum()
        cvd_mean = data["cvd"].rolling(w * 2).mean()
        cvd_std = data["cvd"].rolling(w * 2).std()
        data["cvd_zscore"] = (data["cvd"] - cvd_mean) / cvd_std.replace(0, np.nan)

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

        min_bars = int(self.parameters.get("cvd_window", 14)) * 3
        if len(df) < min_bars:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]
        prev = ind.iloc[-2]

        c = float(bar["close"])
        z = float(bar["cvd_zscore"]) if not np.isnan(bar["cvd_zscore"]) else 0.0
        z_prev = float(prev["cvd_zscore"]) if not np.isnan(prev["cvd_zscore"]) else 0.0
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        thresh = float(self.parameters.get("delta_zscore_threshold", 1.75))
        sl_mult = float(self.parameters.get("sl_atr_mult", 1.5))
        tgt_mult = float(self.parameters.get("target_atr_mult", 3.0))

        sig_dir = Direction.NONE
        conf = 0.0

        # Aggressive Buy Delta Imbalance
        if z >= thresh and z > z_prev and c > float(bar["open"]):
            sig_dir = Direction.BUY
            conf = min(0.85, 0.65 + (z - thresh) * 0.08)
            sl = self.round_to_tick(c - (atr * sl_mult))
            target = self.round_to_tick(c + (atr * tgt_mult))

        # Aggressive Sell Delta Imbalance
        elif z <= -thresh and z < z_prev and c < float(bar["open"]):
            sig_dir = Direction.SELL
            conf = min(0.85, 0.65 + (abs(z) - thresh) * 0.08)
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
            metadata={"delta_zscore": round(z, 2), "atr": atr},
        )

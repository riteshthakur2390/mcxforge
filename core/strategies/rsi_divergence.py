"""
core/strategies/rsi_divergence.py — RSI Divergence Strategy
===========================================================
Detects swing pivots in Price and RSI(14):
- Bullish Divergence: Price makes a Lower Low (LL) while RSI makes a Higher Low (HL)
  -> Institutional accumulation / momentum exhaustion -> BUY entry.
- Bearish Divergence: Price makes a Higher High (HH) while RSI makes a Lower High (LH)
  -> Institutional distribution / momentum exhaustion -> SELL entry.
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
import pandas as pd
import pytz

from core.models import Direction
from core.regime.engine import MarketRegime, RegimeDetails
from core.strategies.base import BaseCommodityStrategy, StrategySignal

IST = pytz.timezone("Asia/Kolkata")


class RSIDivergenceStrategy(BaseCommodityStrategy):
    """
    Pivot-based RSI Divergence strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "rsi_period": 14,
            "pivot_lookback": 5,        # Bars on each side to define a swing pivot
            "divergence_window": 35,    # Max bars between two swing points
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "target_atr_mult": 3.0,
            "min_rsi_diff": 2.0,        # Minimum RSI divergence delta
        }
        super().__init__(
            name="RSIDivergence",
            version="1.0.0",
            permitted_regimes=[MarketRegime.RANGE, MarketRegime.TREND, MarketRegime.BREAKOUT],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # RSI(14)
        rsi_p = int(self.parameters.get("rsi_period", 14))
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain, index=data.index).rolling(rsi_p).mean()
        avg_loss = pd.Series(loss, index=data.index).rolling(rsi_p).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        data["rsi"] = 100.0 - (100.0 / (1.0 + rs)).fillna(50.0)

        # ATR
        atr_p = int(self.parameters.get("atr_period", 14))
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr"] = tr.rolling(atr_p).mean()

        return data

    def _find_swing_lows(self, lows: np.ndarray, rsis: np.ndarray, lookback: int) -> List[Tuple[int, float, float]]:
        """Returns list of (bar_idx, low_price, rsi_value) for swing lows."""
        swings = []
        n = len(lows)
        for i in range(lookback, n - lookback):
            window = lows[i - lookback : i + lookback + 1]
            if lows[i] == np.min(window):
                swings.append((i, lows[i], rsis[i]))
        return swings

    def _find_swing_highs(self, highs: np.ndarray, rsis: np.ndarray, lookback: int) -> List[Tuple[int, float, float]]:
        """Returns list of (bar_idx, high_price, rsi_value) for swing highs."""
        swings = []
        n = len(highs)
        for i in range(lookback, n - lookback):
            window = highs[i - lookback : i + lookback + 1]
            if highs[i] == np.max(window):
                swings.append((i, highs[i], rsis[i]))
        return swings

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

        min_bars = int(self.parameters.get("divergence_window", 35)) + 15
        if len(df) < min_bars:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]
        c = float(bar["close"])
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        lows = ind["low"].values
        highs = ind["high"].values
        rsis = ind["rsi"].values
        n = len(lows)

        lookback = int(self.parameters.get("pivot_lookback", 5))
        max_dist = int(self.parameters.get("divergence_window", 35))
        min_rsi_diff = float(self.parameters.get("min_rsi_diff", 2.0))
        sl_mult = float(self.parameters.get("sl_atr_mult", 1.5))
        tgt_mult = float(self.parameters.get("target_atr_mult", 3.0))

        swing_lows = self._find_swing_lows(lows, rsis, lookback)
        swing_highs = self._find_swing_highs(highs, rsis, lookback)

        sig_dir = Direction.NONE
        conf = 0.0

        # Check Bullish Divergence (Price Lower Low, RSI Higher Low)
        if len(swing_lows) >= 2:
            prev_idx, prev_low, prev_rsi = swing_lows[-2]
            curr_idx, curr_low, curr_rsi = swing_lows[-1]
            
            # Most recent swing must have formed recently (within lookback + 3 bars)
            if (n - 1 - curr_idx) <= (lookback + 2) and (curr_idx - prev_idx) <= max_dist:
                if curr_low < prev_low and curr_rsi > (prev_rsi + min_rsi_diff):
                    sig_dir = Direction.BUY
                    conf = min(0.85, 0.65 + (curr_rsi - prev_rsi) * 0.01)
                    sl = self.round_to_tick(curr_low - (atr * sl_mult))
                    target = self.round_to_tick(c + (atr * tgt_mult))

        # Check Bearish Divergence (Price Higher High, RSI Lower High)
        if sig_dir == Direction.NONE and len(swing_highs) >= 2:
            prev_idx, prev_high, prev_rsi = swing_highs[-2]
            curr_idx, curr_high, curr_rsi = swing_highs[-1]

            if (n - 1 - curr_idx) <= (lookback + 2) and (curr_idx - prev_idx) <= max_dist:
                if curr_high > prev_high and curr_rsi < (prev_rsi - min_rsi_diff):
                    sig_dir = Direction.SELL
                    conf = min(0.85, 0.65 + (prev_rsi - curr_rsi) * 0.01)
                    sl = self.round_to_tick(curr_high + (atr * sl_mult))
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
            metadata={"divergence_type": "BULLISH" if sig_dir == Direction.BUY else "BEARISH", "atr": atr},
        )

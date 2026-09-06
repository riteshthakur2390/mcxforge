"""
core/strategies/gold_silver_pairs.py — Gold-Silver Ratio Pairs & Statistical Arbitrage
======================================================================================
Institutional Statistical Arbitrage (Quant Fund / Citadel style):
- Tracks the Gold-Silver ratio: Ratio = Gold_Price / Silver_Price.
- Calculates rolling Z-score: z = (Ratio - mean(Ratio, 60)) / std(Ratio, 60).
- When z > +2.0: Ratio is at upper extreme -> Silver is undervalued relative to Gold.
  -> Strong BUY signal for Silver (or Long Silver / Short Gold spread).
- When z < -2.0: Ratio is at lower extreme -> Silver is overextended relative to Gold.
  -> Strong SELL signal for Silver (or Short Silver / Long Gold spread).
- Mean reversion target: z reverts back to equilibrium (|z| < 0.5).
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


class GoldSilverPairsStrategy(BaseCommodityStrategy):
    """
    Gold-Silver Ratio Z-score Mean Reversion Statistical Arbitrage Strategy.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "zscore_window": 60,
            "zscore_entry_threshold": 2.0,   # Enter when |z| > 2.0
            "zscore_exit_threshold": 0.5,    # Exit when |z| < 0.5
            "atr_period": 14,
            "sl_atr_mult": 2.0,
            "target_atr_mult": 3.5,
            # Synthetic ratio baseline if explicit Gold series is omitted
            "fallback_synthetic_ratio": True,
        }
        super().__init__(
            name="GoldSilverPairs",
            version="1.0.0",
            permitted_regimes=[MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY, MarketRegime.TREND],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)
        self.gold_df: Optional[pd.DataFrame] = None

    def set_gold_data(self, gold_df: pd.DataFrame) -> None:
        """Sets the synchronised secondary instrument (Gold) for spread calculation."""
        self.gold_df = gold_df.copy()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]

        window = int(self.parameters.get("zscore_window", 60))

        # Check if paired Gold series is available
        if self.gold_df is not None and not self.gold_df.empty:
            common_idx = data.index.intersection(self.gold_df.index)
            if len(common_idx) > 20:
                gold_close = self.gold_df.loc[common_idx, "close"]
                silver_close = close.loc[common_idx]
                ratio = gold_close / silver_close.replace(0, np.nan)
                ratio_mean = ratio.rolling(window).mean()
                ratio_std = ratio.rolling(window).std()
                zscore = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)
                data["gs_ratio"] = ratio
                data["zscore"] = zscore
            else:
                self._compute_synthetic_spread(data, window)
        else:
            self._compute_synthetic_spread(data, window)

        # ATR
        atr_p = int(self.parameters.get("atr_period", 14))
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr"] = tr.rolling(atr_p).mean()

        return data

    def _compute_synthetic_spread(self, data: pd.DataFrame, window: int) -> None:
        """
        When standalone Silver data is passed without parallel Gold feed,
        computes the structural relative value z-score using price detrended ratio.
        """
        c = data["close"]
        ma_long = c.rolling(window).mean()
        ma_std = c.rolling(window).std()
        # Inverted z-score of price distance represents relative undervaluation/overvaluation
        data["zscore"] = -(c - ma_long) / ma_std.replace(0, np.nan)
        data["gs_ratio"] = ma_long / c.replace(0, np.nan)

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

        min_bars = int(self.parameters.get("zscore_window", 60)) + 10
        if len(df) < min_bars:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]
        prev = ind.iloc[-2]

        c = float(bar["close"])
        z = float(bar["zscore"]) if not np.isnan(bar["zscore"]) else 0.0
        z_prev = float(prev["zscore"]) if not np.isnan(prev["zscore"]) else 0.0
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        z_thresh = float(self.parameters.get("zscore_entry_threshold", 2.0))
        sl_mult = float(self.parameters.get("sl_atr_mult", 2.0))
        tgt_mult = float(self.parameters.get("target_atr_mult", 3.5))

        sig_dir = Direction.NONE
        conf = 0.0

        # z > +2.0: Gold/Silver ratio is extremely high -> Silver is cheap -> BUY Silver
        if z >= z_thresh and z_prev < z:
            sig_dir = Direction.BUY
            conf = min(0.85, 0.65 + (z - z_thresh) * 0.08)
            sl = self.round_to_tick(c - (atr * sl_mult))
            target = self.round_to_tick(c + (atr * tgt_mult))

        # z < -2.0: Gold/Silver ratio is extremely low -> Silver is expensive -> SELL Silver
        elif z <= -z_thresh and z_prev > z:
            sig_dir = Direction.SELL
            conf = min(0.85, 0.65 + (abs(z) - z_thresh) * 0.08)
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
            metadata={"zscore": round(z, 2), "atr": atr},
        )

"""
core/strategies/term_structure.py — Contango / Backwardation Term Structure Strategy
====================================================================================
Institutional Commodity Term Structure Edge:
- Analyzes the calendar spread between Near-Month and Far-Month MCX Futures:
  Spread = Far_Month_Price - Near_Month_Price
- In standard carry markets, commodities trade in Contango (Far > Near due to storage, financing, insurance).
- When physical demand spikes or spot supply tightens:
  - Spread contracts sharply or inverts into Backwardation (Near > Far).
  - `if spread < spread.rolling(30).mean() - 1.0 * spread.rolling(30).std():`
    -> Bullish near-month signal ("bullish_near").
- When inventories flood the physical market:
  - Super-contango emerges (Spread > mean + 1.5 * std).
  - -> Bearish near-month signal ("bearish_near").
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


class TermStructureStrategy(BaseCommodityStrategy):
    """
    Term Structure Contango/Backwardation calendar spread strategy for MCX Commodity Futures.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "spread_window": 30,
            "backwardation_std_mult": 1.0,     # Z-score deviation below mean
            "contango_std_mult": 1.5,          # Z-score deviation above mean
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "target_atr_mult": 3.0,
        }
        super().__init__(
            name="TermStructure",
            version="1.0.0",
            permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT, MarketRegime.HIGH_VOLATILITY, MarketRegime.RANGE],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)
        self.far_month_df: Optional[pd.DataFrame] = None

    def set_far_month_data(self, far_df: pd.DataFrame) -> None:
        """Sets the synchronised far-month contract data."""
        self.far_month_df = far_df.copy()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]

        w = int(self.parameters.get("spread_window", 30))

        if self.far_month_df is not None and not self.far_month_df.empty:
            common_idx = data.index.intersection(self.far_month_df.index)
            if len(common_idx) > 10:
                far_c = self.far_month_df.loc[common_idx, "close"]
                near_c = close.loc[common_idx]
                spread = far_c - near_c
                data["calendar_spread"] = spread
                sp_mean = spread.rolling(w).mean()
                sp_std = spread.rolling(w).std()
                data["spread_zscore"] = (spread - sp_mean) / sp_std.replace(0, np.nan)
            else:
                self._compute_proxy_term_structure(data, w)
        else:
            self._compute_proxy_term_structure(data, w)

        # ATR
        atr_p = int(self.parameters.get("atr_period", 14))
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["atr"] = tr.rolling(atr_p).mean()

        return data

    def _compute_proxy_term_structure(self, data: pd.DataFrame, window: int) -> None:
        """
        When standalone near-month data is passed, estimates term structure tension
        via roll momentum & basis velocity (rate of change vs 30-bar baseline).
        """
        c = data["close"]
        mom = c - c.shift(window)
        mom_mean = mom.rolling(window).mean()
        mom_std = mom.rolling(window).std()
        # Inverted momentum acts as backwardation proxy
        data["spread_zscore"] = -(mom - mom_mean) / mom_std.replace(0, np.nan)
        data["calendar_spread"] = mom

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

        min_bars = int(self.parameters.get("spread_window", 30)) + 10
        if len(df) < min_bars:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]
        prev = ind.iloc[-2]

        c = float(bar["close"])
        z = float(bar["spread_zscore"]) if not np.isnan(bar["spread_zscore"]) else 0.0
        z_prev = float(prev["spread_zscore"]) if not np.isnan(prev["spread_zscore"]) else 0.0
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        bw_thresh = float(self.parameters.get("backwardation_std_mult", 1.0))
        ct_thresh = float(self.parameters.get("contango_std_mult", 1.5))
        sl_mult = float(self.parameters.get("sl_atr_mult", 1.5))
        tgt_mult = float(self.parameters.get("target_atr_mult", 3.0))

        sig_dir = Direction.NONE
        conf = 0.0
        state = ""

        # Backwardation Shift: Spread drops below mean - 1.0 std -> Physical squeeze -> Bullish Near Month
        if z <= -bw_thresh and z < z_prev:
            sig_dir = Direction.BUY
            conf = min(0.85, 0.65 + (abs(z) - bw_thresh) * 0.08)
            state = "BACKWARDATION_PHYSICAL_TIGHTNESS"
            sl = self.round_to_tick(c - (atr * sl_mult))
            target = self.round_to_tick(c + (atr * tgt_mult))

        # Extreme Contango Shift: Spread expands above mean + 1.5 std -> Physical inventory glut -> Bearish Near Month
        elif z >= ct_thresh and z > z_prev:
            sig_dir = Direction.SELL
            conf = min(0.85, 0.65 + (z - ct_thresh) * 0.08)
            state = "SUPER_CONTANGO_SUPPLY_GLUT"
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
            metadata={"term_structure_state": state, "spread_zscore": round(z, 2), "atr": atr},
        )

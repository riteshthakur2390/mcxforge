"""
core/strategies/calendar_seasonality.py — Institutional Calendar Seasonality Strategy
======================================================================================
Silver in India exhibits pronounced, statistically backtestable calendar tendencies:
1. Pre-Diwali / Festive Demand Window:
   - Months September & October (leading up to Dhanteras/Diwali).
   - High physical wedding/festive demand creates structural long accumulation bias.
   - `if month in [9, 10] and day_of_month < 15: bias = 'long'`
2. Akshaya Tritiya Spring Window:
   - Late April / early May physical bullion surge.
3. Post-Budget / Tax Season:
   - Early February import duty adjustment volatility.
4. Month-End / Quarterly COMEX Rollover Cycle:
   - Expiry roll pressures into the active contract.
"""

from __future__ import annotations
from datetime import datetime, date
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import pytz

from core.models import Direction
from core.regime.engine import MarketRegime, RegimeDetails
from core.strategies.base import BaseCommodityStrategy, StrategySignal

IST = pytz.timezone("Asia/Kolkata")


class CalendarSeasonalityStrategy(BaseCommodityStrategy):
    """
    Exploits recurring macroeconomic and festive calendar windows in Indian commodity markets.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "diwali_months": [9, 10],            # September, October pre-festive
            "diwali_max_day": 20,                # Accumulation window
            "spring_month": 4,                   # April Akshaya Tritiya window
            "trend_filter_period": 20,
            "atr_period": 14,
            "sl_atr_mult": 1.5,
            "target_atr_mult": 3.0,
        }
        super().__init__(
            name="CalendarSeasonality",
            version="1.0.0",
            permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT, MarketRegime.RANGE],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        close = data["close"]
        high = data["high"]
        low = data["low"]

        p = int(self.parameters.get("trend_filter_period", 20))
        data["trend_ema"] = close.ewm(span=p, adjust=False).mean()

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

        min_bars = int(self.parameters.get("trend_filter_period", 20)) + 5
        if len(df) < min_bars:
            return none_sig

        ind = self.calculate_indicators(df)
        bar = ind.iloc[-1]
        c = float(bar["close"])
        ema = float(bar["trend_ema"])
        atr = float(bar["atr"]) if not np.isnan(bar["atr"]) else (c * 0.005)

        m = ts.month
        d = ts.day

        diwali_months = self.parameters.get("diwali_months", [9, 10])
        diwali_max_day = int(self.parameters.get("diwali_max_day", 20))
        spring_m = int(self.parameters.get("spring_month", 4))

        sl_mult = float(self.parameters.get("sl_atr_mult", 1.5))
        tgt_mult = float(self.parameters.get("target_atr_mult", 3.0))

        is_diwali_window = (m in diwali_months and d <= diwali_max_day)
        is_spring_window = (m == spring_m and 10 <= d <= 28)

        sig_dir = Direction.NONE
        conf = 0.0
        reason = ""

        # Pre-Diwali Bullish Accumulation
        if is_diwali_window and c >= ema * 0.995:
            sig_dir = Direction.BUY
            conf = 0.78
            reason = f"Pre-Diwali physical demand accumulation window (Month {m}, Day {d})"
            sl = self.round_to_tick(c - (atr * sl_mult))
            target = self.round_to_tick(c + (atr * tgt_mult))

        # Spring Bullion Festive Window
        elif is_spring_window and c >= ema * 0.995:
            sig_dir = Direction.BUY
            conf = 0.72
            reason = f"Akshaya Tritiya spring bullion window (Month {m}, Day {d})"
            sl = self.round_to_tick(c - (atr * sl_mult))
            target = self.round_to_tick(c + (atr * tgt_mult))

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
            metadata={"calendar_window": reason, "month": m, "day": d, "atr": atr},
        )

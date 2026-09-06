"""
core/strategies/currency_macro_filter.py — USD-INR & DXY Macro Overlay Filter
=============================================================================
Silver traded on MCX in INR terms is a direct mathematical derivative of:
  MCX Silver (INR) = [COMEX Silver (USD/oz) * USDINR * (1 + Import Duty + Cess)] / 31.1035

A technical signal generated on MCX price can be completely invalidated or whipsawed
if USD-INR undergoes an intraday or interday shock move:
- If |USDINR_change_1d| > 0.5%: Macro currency volatility alert.
  -> Action: Reduce position sizing by 50% (risk scaling).
- If |USDINR_change_1d| > 1.2%: Macro currency shock.
  -> Action: Block aggressive breakouts or reduce position size to 25%.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd


class MacroVolatilityLevel(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"       # |change| > 0.5%
    EXTREME = "EXTREME"         # |change| > 1.2%


@dataclass
class MacroFilterResult:
    level: MacroVolatilityLevel
    usdinr_change_pct: float
    dxy_change_pct: float
    position_size_multiplier: float
    is_safe_to_trade: bool
    reason: str


class CurrencyMacroFilter:
    """
    USD-INR and DXY Macro Overlay Filter for MCX Commodity Strategies.
    """

    def __init__(
        self,
        elevated_threshold_pct: float = 0.50,   # 0.5% 1-day move
        extreme_threshold_pct: float = 1.20,    # 1.2% 1-day move
    ):
        self.elevated_threshold_pct = elevated_threshold_pct
        self.extreme_threshold_pct = extreme_threshold_pct

    def evaluate(
        self,
        usdinr_change_pct: float = 0.0,
        dxy_change_pct: float = 0.0,
    ) -> MacroFilterResult:
        abs_usdinr = abs(usdinr_change_pct)
        abs_dxy = abs(dxy_change_pct)

        if abs_usdinr >= self.extreme_threshold_pct or abs_dxy >= 1.5:
            return MacroFilterResult(
                level=MacroVolatilityLevel.EXTREME,
                usdinr_change_pct=round(usdinr_change_pct, 3),
                dxy_change_pct=round(dxy_change_pct, 3),
                position_size_multiplier=0.25,
                is_safe_to_trade=False,
                reason=f"Extreme currency shock detected (USD-INR: {usdinr_change_pct:+.2f}%, DXY: {dxy_change_pct:+.2f}%). Sizing scaled to 25%.",
            )
        elif abs_usdinr >= self.elevated_threshold_pct or abs_dxy >= 0.75:
            return MacroFilterResult(
                level=MacroVolatilityLevel.ELEVATED,
                usdinr_change_pct=round(usdinr_change_pct, 3),
                dxy_change_pct=round(dxy_change_pct, 3),
                position_size_multiplier=0.50,
                is_safe_to_trade=True,
                reason=f"Elevated USD-INR volatility ({usdinr_change_pct:+.2f}%). Reducing position size by 50%.",
            )
        else:
            return MacroFilterResult(
                level=MacroVolatilityLevel.NORMAL,
                usdinr_change_pct=round(usdinr_change_pct, 3),
                dxy_change_pct=round(dxy_change_pct, 3),
                position_size_multiplier=1.00,
                is_safe_to_trade=True,
                reason="Currency macro environment stable.",
            )

    def extract_change_from_df(self, usdinr_df: Optional[pd.DataFrame]) -> float:
        """Extracts latest 1-day percentage change from a USD-INR daily or intraday dataframe."""
        if usdinr_df is None or len(usdinr_df) < 2:
            return 0.0
        c = usdinr_df["close"].values
        chg = ((c[-1] - c[-2]) / c[-2]) * 100.0
        return float(chg)

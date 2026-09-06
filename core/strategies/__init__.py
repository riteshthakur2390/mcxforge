"""
core.strategies — Multi-Instrument Strategy Governance
"""

from core.strategies.base import (
    BaseCommodityStrategy,
    StrategySignal,
)
from core.strategies.trend_following import TrendFollowingStrategy
from core.strategies.orb import OpeningRangeBreakoutStrategy
from core.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from core.strategies.volatility_breakout import VolatilityBreakoutStrategy
from core.strategies.donchian_breakout import DonchianBreakoutStrategy
from core.strategies.registry import (
    StrategyStatus,
    StrategyDefinition,
    MASTER_STRATEGY_CATALOG,
    StrategyRegistry,
)

__all__ = [
    "BaseCommodityStrategy",
    "StrategySignal",
    "TrendFollowingStrategy",
    "OpeningRangeBreakoutStrategy",
    "VWAPMeanReversionStrategy",
    "VolatilityBreakoutStrategy",
    "DonchianBreakoutStrategy",
    "StrategyStatus",
    "StrategyDefinition",
    "MASTER_STRATEGY_CATALOG",
    "StrategyRegistry",
]

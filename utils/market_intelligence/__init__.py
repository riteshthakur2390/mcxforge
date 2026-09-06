"""
utils/market_intelligence package
================================================================================
Combines:
  1. The new extensible Market Intelligence Engine + plugins (PCR, TrapDetection, OI, FII, VIX, MaxPain)
  2. Legacy MarketIntelligence monitor classes for backwards compatibility
"""

from utils.market_intelligence.base import MarketContextPlugin, ContextScore
from utils.market_intelligence.engine import MarketIntelligenceEngine, TradeQualityReport, get_market_intelligence_engine
from utils.market_intelligence.legacy import (
    MarketIntelligence,
    BreadthSignal,
    PCRSignal,
    VWAPCheck,
    VIXTrend,
    MarketBreadthMonitor,
    PCRMonitor,
    VWAPMonitor,
    VIXTrendMonitor,
)

__all__ = [
    # New Engine
    "MarketContextPlugin",
    "ContextScore",
    "TradeQualityReport",
    "MarketIntelligenceEngine",
    "get_market_intelligence_engine",
    # Legacy Classes
    "MarketIntelligence",
    "BreadthSignal",
    "PCRSignal",
    "VWAPCheck",
    "VIXTrend",
    "MarketBreadthMonitor",
    "PCRMonitor",
    "VWAPMonitor",
    "VIXTrendMonitor",
]

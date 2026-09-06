"""
core/strategies/registry.py — Strategy Governance & Per-Instrument Configuration

Classifies strategies into:
- KEEP: Reusable directly on MCX commodity futures
- ADAPT: Adapted for MCX hours (09:00 - 23:30), commodity sessions, futures pricing
- RESEARCH: Retained for historical study but disabled by default
- REMOVED: Excluded from active commodity engine (Options / NIFTY-specific)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from core.regime.engine import MarketRegime


class StrategyStatus(str, Enum):
    KEEP = "KEEP"
    ADAPT = "ADAPT"
    RESEARCH = "RESEARCH"
    REMOVED = "REMOVED"


@dataclass(frozen=True)
class StrategyDefinition:
    id: str                             # e.g., "S1"
    name: str                           # e.g., "SuperTrend+RSI"
    category: str                       # e.g., "Trend / Momentum"
    status: StrategyStatus
    permitted_regimes: List[MarketRegime]
    min_candles: int
    description: str
    requires_live_broker: bool = False


# Complete Catalog of all 34 Strategies with Audit Classifications
MASTER_STRATEGY_CATALOG: Dict[str, StrategyDefinition] = {
    # ── 5 CORE MCX COMMODITY STRATEGY FAMILIES ─────────────────────────────
    "TrendFollowing": StrategyDefinition(
        id="TF_01", name="TrendFollowing", category="Trend",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
        min_candles=30, description="EMA structural alignment, ADX trend filter, and ATR risk bands",
    ),
    "OpeningRangeBreakout": StrategyDefinition(
        id="ORB_01", name="OpeningRangeBreakout", category="Breakout",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.HIGH_VOLATILITY, MarketRegime.TREND],
        min_candles=15, description="MCX session opening price discovery breakout with volume confirmation",
    ),
    "VWAPMeanReversion": StrategyDefinition(
        id="VWAP_MR_01", name="VWAPMeanReversion", category="Mean Reversion",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY],
        min_candles=25, description="Intraday stretch fade toward VWAP equilibrium (vetoed in strong trends)",
    ),
    "VolatilityBreakout": StrategyDefinition(
        id="VOL_BO_01", name="VolatilityBreakout", category="Volatility",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY],
        min_candles=25, description="Squeeze compression expansion breakout with abnormal volatility guard",
    ),
    "DonchianBreakout": StrategyDefinition(
        id="DONCH_01", name="DonchianBreakout", category="Breakout",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
        min_candles=25, description="Classic N-period Donchian channel breakout statistical benchmark",
    ),

    # ── NEW TOP 10 INSTITUTIONAL COMMODITY STRATEGIES ─────────────────────────
    "BBMeanReversion": StrategyDefinition(
        id="BB_MR_01", name="BBMeanReversion", category="Mean Reversion",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY],
        min_candles=25, description="Fading outer Bollinger Bands in ranging markets with ADX < 20 filter",
    ),
    "MACrossover": StrategyDefinition(
        id="MA_X_01", name="MACrossover", category="Trend",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
        min_candles=30, description="Dual EMA(9/21) crossover with ATR stop and momentum confirmation",
    ),
    "RSIDivergence": StrategyDefinition(
        id="RSI_DIV_01", name="RSIDivergence", category="Momentum Reversal",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.RANGE, MarketRegime.TREND, MarketRegime.BREAKOUT],
        min_candles=35, description="Swing pivot-based RSI bullish/bearish momentum divergence",
    ),
    "MomentumVolumeBreakout": StrategyDefinition(
        id="MOM_VOL_01", name="MomentumVolumeBreakout", category="Breakout",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY],
        min_candles=30, description="Price N-bar high/low breakout confirmed by volume spike and MACD histogram",
    ),
    "GoldSilverPairs": StrategyDefinition(
        id="PAIRS_01", name="GoldSilverPairs", category="Statistical Arbitrage",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY, MarketRegime.TREND],
        min_candles=60, description="Gold-Silver price ratio rolling Z-score mean reversion statistical arbitrage",
    ),
    "OrderFlowDelta": StrategyDefinition(
        id="OF_DELTA_01", name="OrderFlowDelta", category="Microstructure",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND, MarketRegime.HIGH_VOLATILITY],
        min_candles=30, description="Cumulative Volume Delta (CVD) aggressive buyer/seller imbalance",
    ),
    "TimeOfDaySeasonality": StrategyDefinition(
        id="TOD_SEASON_01", name="TimeOfDaySeasonality", category="Seasonality",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT, MarketRegime.HIGH_VOLATILITY],
        min_candles=25, description="Statistical time-of-day edge during US COMEX opening liquidity window",
    ),
    "RSI2MeanReversion": StrategyDefinition(
        id="RSI2_MR_01", name="RSI2MeanReversion", category="Mean Reversion",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY],
        min_candles=25, description="Larry Connors 2-period RSI extreme oversold/overbought pullback reversion",
    ),
    "CalendarSeasonality": StrategyDefinition(
        id="CAL_SEASON_01", name="CalendarSeasonality", category="Seasonality",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT, MarketRegime.RANGE],
        min_candles=25, description="Pre-Diwali and Akshaya Tritiya physical commodity demand cycle",
    ),
    "TermStructure": StrategyDefinition(
        id="TERM_STRUCT_01", name="TermStructure", category="Term Structure",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT, MarketRegime.HIGH_VOLATILITY, MarketRegime.RANGE],
        min_candles=30, description="Contango and Backwardation near-vs-far calendar spread physical supply/demand stress",
    ),

    # ── KEEP (Directly Reusable on MCX Futures) ──────────────────────────────
    "SuperTrend+RSI": StrategyDefinition(
        id="S1", name="SuperTrend+RSI", category="Trend",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
        min_candles=20, description="SuperTrend trend flip with RSI momentum confirmation",
    ),
    "VWAP+EMA": StrategyDefinition(
        id="S2", name="VWAP+EMA", category="Trend",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
        min_candles=20, description="Intraday VWAP rejection and EMA alignment",
    ),
    "ORB": StrategyDefinition(
        id="S3", name="ORB", category="Breakout",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.HIGH_VOLATILITY],
        min_candles=4, description="09:00-09:30 Opening Range Breakout with volume expansion",
    ),
    "BBSqueeze": StrategyDefinition(
        id="S4", name="BBSqueeze", category="Volatility",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND],
        min_candles=20, description="Bollinger Band squeeze expansion into trend",
    ),
    "ADX+PSAR": StrategyDefinition(
        id="S5", name="ADX+PSAR", category="Trend",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND],
        min_candles=20, description="Strong directional trend with Parabolic SAR trailing stop",
    ),
    "FVG": StrategyDefinition(
        id="S6", name="FVG", category="Structure",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
        min_candles=15, description="Fair Value Gap liquidity imbalance retest",
    ),
    "UTBot": StrategyDefinition(
        id="S7", name="UTBot", category="Trend",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
        min_candles=20, description="ATR trailing stop key-value crossover",
    ),
    "Ichimoku": StrategyDefinition(
        id="S9", name="Ichimoku", category="Trend",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND],
        min_candles=52, description="Multi-timeframe Tenkan/Kijun cloud confirmation",
    ),
    "VolumeProfile": StrategyDefinition(
        id="S10", name="VolumeProfile", category="Structure",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.RANGE, MarketRegime.BREAKOUT],
        min_candles=30, description="VPOC and Value Area High/Low rejection/break",
    ),
    "LiqSweep": StrategyDefinition(
        id="S11", name="LiqSweep", category="Structure",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.RANGE, MarketRegime.HIGH_VOLATILITY],
        min_candles=20, description="Liquidity stop-hunt rejection and reversal",
    ),
    "PriceAction": StrategyDefinition(
        id="S12", name="PriceAction", category="Structure",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.RANGE],
        min_candles=20, description="Support/Resistance candle patterns at key levels",
    ),
    "HeikinAshi": StrategyDefinition(
        id="S28", name="HeikinAshi", category="Trend",
        status=StrategyStatus.KEEP,
        permitted_regimes=[MarketRegime.TREND],
        min_candles=15, description="Smoothed Heikin Ashi trend persistence",
    ),

    # ── ADAPT (Modified for MCX Trading Sessions & Commodity Futures) ────────
    "CPR": StrategyDefinition(
        id="S8", name="CPR", category="Structure",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND, MarketRegime.RANGE],
        min_candles=20, description="Central Pivot Range calculated from prior session high/low/close",
    ),
    "OIAnalysis": StrategyDefinition(
        id="S13", name="OIAnalysis", category="Participation",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.TREND],
        min_candles=20, description="Futures Open Interest buildup / long unwinding / short covering",
    ),
    "AMD": StrategyDefinition(
        id="S15", name="AMD", category="Structure",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND],
        min_candles=30, description="Accumulation-Manipulation-Distribution across MCX sessions",
    ),
    "GapDirection": StrategyDefinition(
        id="S16", name="GapDirection", category="Momentum",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND],
        min_candles=15, description="Overnight gap fill / continuation relative to global commodities",
    ),
    "SMC": StrategyDefinition(
        id="S17", name="SMC", category="Structure",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.TREND, MarketRegime.BREAKOUT],
        min_candles=50, description="Smart Money Concepts break-of-structure (BOS) & change-of-character (CHoCH)",
    ),
    "GapMomentum": StrategyDefinition(
        id="S22", name="GapMomentum", category="Momentum",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.BREAKOUT],
        min_candles=20, description="Momentum extension during 17:00 IST US market opening overlap",
    ),
    "SqueezeMomentum": StrategyDefinition(
        id="S25", name="SqueezeMomentum", category="Volatility",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND],
        min_candles=20, description="John Carter Squeeze indicator adapted for commodity tick sizes",
    ),
    "EMASlope": StrategyDefinition(
        id="S27", name="EMASlope", category="Trend",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.TREND],
        min_candles=20, description="Linear regression slope of EMA20",
    ),
    "OpeningRangeBias": StrategyDefinition(
        id="S31", name="OpeningRangeBias", category="Breakout",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.TREND],
        min_candles=15, description="First 30-minute high/low directional bias",
    ),

    # ── ADAPTED ACTIVE STRATEGIES ───────────────────────────────────────────
    "ValueArea": StrategyDefinition(
        id="S21", name="ValueArea", category="Structure",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.RANGE],
        min_candles=30, description="Multi-day Value Area acceptance/rejection",
    ),
    "ADXRising": StrategyDefinition(
        id="S23", name="ADXRising", category="Trend",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.TREND],
        min_candles=25, description="ADX slope acceleration",
    ),
    "RangeSpread": StrategyDefinition(
        id="S24", name="RangeSpread", category="Volatility",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.RANGE],
        min_candles=20, description="Bar range / volume spread analysis",
    ),
    "StochRSI": StrategyDefinition(
        id="S26", name="StochRSI", category="Momentum",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.RANGE],
        min_candles=20, description="Stochastic RSI oscillator overbought/oversold",
    ),
    "VWAPExtreme": StrategyDefinition(
        id="S29", name="VWAPExtreme", category="Mean Reversion",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.RANGE],
        min_candles=20, description="Mean reversion from 2.5+ ATR distance from VWAP",
    ),
    "ElliottWave": StrategyDefinition(
        id="S34", name="ElliottWave", category="Wave Theory",
        status=StrategyStatus.ADAPT,
        permitted_regimes=[MarketRegime.TREND],
        min_candles=40, description="Algorithmic wave 3 / wave 5 projection",
    ),

    # ── REMOVED (Options / NIFTY Equity Index Specific) ──────────────────────
    "IVContraction": StrategyDefinition(
        id="S14", name="IVContraction", category="Options",
        status=StrategyStatus.REMOVED, permitted_regimes=[], min_candles=0,
        description="REMOVED: Option IV crush logic",
    ),
    "SkewHunter": StrategyDefinition(
        id="S17_opt", name="SkewHunter", category="Options",
        status=StrategyStatus.REMOVED, permitted_regimes=[], min_candles=0,
        description="REMOVED: Option volatility skew",
    ),
    "ExpiryWeek": StrategyDefinition(
        id="S19", name="ExpiryWeek", category="Equity Index",
        status=StrategyStatus.REMOVED, permitted_regimes=[], min_candles=0,
        description="REMOVED: NIFTY Thursday weekly expiry anomaly",
    ),
    "StrikeMomentum": StrategyDefinition(
        id="S20", name="StrikeMomentum", category="Options",
        status=StrategyStatus.REMOVED, permitted_regimes=[], min_candles=0,
        description="REMOVED: Option strike momentum",
    ),
    "VIXDivergence": StrategyDefinition(
        id="S30", name="VIXDivergence", category="Equity Index",
        status=StrategyStatus.REMOVED, permitted_regimes=[], min_candles=0,
        description="REMOVED: India VIX vs NIFTY index divergence",
    ),
    "GammaExposure": StrategyDefinition(
        id="S32", name="GammaExposure", category="Options",
        status=StrategyStatus.REMOVED, permitted_regimes=[], min_candles=0,
        description="REMOVED: Option dealer gamma exposure (GEX)",
    ),
    "HeroZero": StrategyDefinition(
        id="S33", name="HeroZero", category="Options",
        status=StrategyStatus.REMOVED, permitted_regimes=[], min_candles=0,
        description="REMOVED: 0DTE options expiry gamble",
    ),
}


class StrategyRegistry:
    """
    Manages active production strategies per commodity instrument.
    Enforces status isolation (KEEP/ADAPT enabled, RESEARCH/REMOVED excluded).
    """

    @classmethod
    def get_eligible_strategies_for_instrument(cls, symbol: str = "SILVERMIC") -> List[StrategyDefinition]:
        """
        Returns the active, production-approved strategies configured for the instrument.
        """
        from instruments.registry import get_instrument_strategy_config
        config = get_instrument_strategy_config(symbol)
        enabled_names: List[str] = config.get("enabled_strategies", [])

        active_list: List[StrategyDefinition] = []
        for name in enabled_names:
            definition = MASTER_STRATEGY_CATALOG.get(name)
            if definition and definition.status in (StrategyStatus.KEEP, StrategyStatus.ADAPT):
                active_list.append(definition)
        return active_list

    @classmethod
    def is_strategy_permitted_in_regime(cls, strategy_name: str, regime: MarketRegime) -> bool:
        """Checks if a strategy is authorized to trade in the given regime."""
        strat = MASTER_STRATEGY_CATALOG.get(strategy_name)
        if not strat:
            return False
        return regime in strat.permitted_regimes

    @classmethod
    def get_strategy_class_map(cls) -> Dict[str, Any]:
        """Lazy load and return mapping of strategy names to strategy classes."""
        from core.strategies.trend_following import TrendFollowingStrategy
        from core.strategies.orb import OpeningRangeBreakoutStrategy
        from core.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
        from core.strategies.volatility_breakout import VolatilityBreakoutStrategy
        from core.strategies.donchian_breakout import DonchianBreakoutStrategy

        return {
            "TrendFollowing": TrendFollowingStrategy,
            "TrendFollowingStrategy": TrendFollowingStrategy,
            "OpeningRangeBreakout": OpeningRangeBreakoutStrategy,
            "OpeningRangeBreakoutStrategy": OpeningRangeBreakoutStrategy,
            "ORB": OpeningRangeBreakoutStrategy,
            "VWAPMeanReversion": VWAPMeanReversionStrategy,
            "VWAPMeanReversionStrategy": VWAPMeanReversionStrategy,
            "VolatilityBreakout": VolatilityBreakoutStrategy,
            "VolatilityBreakoutStrategy": VolatilityBreakoutStrategy,
            "DonchianBreakout": DonchianBreakoutStrategy,
            "DonchianBreakoutStrategy": DonchianBreakoutStrategy,
        }

    @classmethod
    def create_strategy(
        cls,
        name: str,
        instrument_config: Any,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Instantiates and initializes a commodity strategy with instrument metadata and parameters.
        """
        class_map = cls.get_strategy_class_map()
        klass = class_map.get(name)
        if not klass:
            raise ValueError(f"Unknown commodity strategy: '{name}'. Available: {list(class_map.keys())}")
        strategy_instance = klass()
        strategy_instance.initialize(instrument_config, parameters=parameters)
        return strategy_instance

    @classmethod
    def get_all_commodity_strategies(
        cls,
        instrument_config: Any,
        parameters_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Instantiates all 5 canonical commodity strategy families."""
        canonical_names = [
            "TrendFollowing",
            "OpeningRangeBreakout",
            "VWAPMeanReversion",
            "VolatilityBreakout",
            "DonchianBreakout",
        ]
        res = {}
        for name in canonical_names:
            params = (parameters_map or {}).get(name)
            res[name] = cls.create_strategy(name, instrument_config, parameters=params)
        return res

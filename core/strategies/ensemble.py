"""
core/strategies/ensemble.py — Unified Multi-Strategy Commodity Ensemble Engine
=============================================================================
Unifies all strategy families for MCX Commodity Futures (SILVERMIC, GOLD, CRUDEOIL):
1. 5 Core Commodity Strategies:
   - TrendFollowingStrategy
   - OpeningRangeBreakoutStrategy
   - VWAPMeanReversionStrategy
   - VolatilityBreakoutStrategy
   - DonchianBreakoutStrategy
2. Ported SignalForge Strategy Suite:
   - SuperTrend + RSI (S1)
   - UT Bot ATR Trailing Stop (S7)
   - Smart Money Concepts / Order Blocks (S17)
   - Squeeze Momentum (S25)
   - VWAP Extreme (S29)
   - Ichimoku Cloud (S9)
3. Multi-Strategy Confluence Voting & Consensus:
   - Aggregates simultaneous strategy signals
   - Determines agreement / confluence
   - Attributes P&L and performance contribution per strategy
4. MCX Session Tagging & Filtering:
   - MORNING: 09:00 - 13:00 IST (Asian hours)
   - AFTERNOON: 13:00 - 17:00 IST (European open)
   - EVENING: 17:00 - 23:30 IST (US / COMEX open - peak commodity volatility)
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, time, timedelta
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import pytz

from instruments.base import InstrumentConfig
from instruments import SILVERMIC_CONFIG, SILVERM_CONFIG
from core.models import Direction
from core.regime.engine import MarketRegime, MarketRegimeEngine, RegimeDetails
from core.strategies.base import BaseCommodityStrategy, StrategySignal
from core.strategies.trend_following import TrendFollowingStrategy
from core.strategies.orb import OpeningRangeBreakoutStrategy
from core.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy
from core.strategies.volatility_breakout import VolatilityBreakoutStrategy
from core.strategies.donchian_breakout import DonchianBreakoutStrategy
from core.strategies.bb_mean_reversion import BollingerBandMeanReversionStrategy
from core.strategies.ma_crossover import MovingAverageCrossoverStrategy
from core.strategies.rsi_divergence import RSIDivergenceStrategy
from core.strategies.momentum_volume_breakout import MomentumVolumeBreakoutStrategy
from core.strategies.gold_silver_pairs import GoldSilverPairsStrategy
from core.strategies.order_flow_delta import OrderFlowDeltaStrategy
from core.strategies.time_of_day_seasonality import TimeOfDaySeasonalityStrategy
from core.strategies.rsi2_mean_reversion import RSI2MeanReversionStrategy
from core.strategies.calendar_seasonality import CalendarSeasonalityStrategy
from core.strategies.term_structure import TermStructureStrategy
from core.strategies.currency_macro_filter import CurrencyMacroFilter
from core.strategies.backtest_models import (
    SlippageModel,
    SlippageType,
    SameBarAmbiguityRule,
    BacktestTrade,
    EquityPoint,
    PerformanceMetrics,
)
from utils.brokerage_calculator import calculate_commodity_trade_charges

IST = pytz.timezone("Asia/Kolkata")

# ── SignalForge Strategy Categories Mapping ─────────────────────────────────
STRATEGY_CATEGORIES: Dict[str, str] = {
    # Trend
    "TrendFollowing": "TREND",
    "SuperTrend+RSI": "TREND",
    "Ichimoku": "TREND",
    "MACrossover": "TREND",
    "ElliottWave": "TREND",
    "EMASlope": "TREND",
    # Momentum & Reversal
    "RSI": "MOMENTUM_REVERSAL",
    "RSIDivergence": "MOMENTUM_REVERSAL",
    "RSI2MeanReversion": "MOMENTUM_REVERSAL",
    "StochRSI": "MOMENTUM_REVERSAL",
    "StrikeMomentum": "MOMENTUM_REVERSAL",
    "IVContraction": "MOMENTUM_REVERSAL",
    # Market Structure & Order Flow
    "VWAP+EMA": "PRICE_ACTION_STRUCTURE",
    "SMC": "PRICE_ACTION_STRUCTURE",
    "SMC_OrderBlocks": "PRICE_ACTION_STRUCTURE",
    "VolumeProfile": "PRICE_ACTION_STRUCTURE",
    "CPR": "PRICE_ACTION_STRUCTURE",
    "BBMeanReversion": "PRICE_ACTION_STRUCTURE",
    "BBSqueeze": "PRICE_ACTION_STRUCTURE",
    "VolatilityBreakout": "PRICE_ACTION_STRUCTURE",
    "DonchianBreakout": "PRICE_ACTION_STRUCTURE",
    "OpeningRangeBreakout": "PRICE_ACTION_STRUCTURE",
    "GammaExposure": "PRICE_ACTION_STRUCTURE",
    "GammaExplosion": "PRICE_ACTION_STRUCTURE",
    # Multi-Asset, Macro & Seasonality
    "GoldSilverPairs": "MULTI_ASSET_SEASONALITY",
    "TermStructure": "MULTI_ASSET_SEASONALITY",
    "CalendarSeasonality": "MULTI_ASSET_SEASONALITY",
    "TimeOfDaySeasonality": "MULTI_ASSET_SEASONALITY",
    "OrderFlowDelta": "MULTI_ASSET_SEASONALITY",
    "OpeningRangeBias": "MULTI_ASSET_SEASONALITY",
    "RangeSpread": "MULTI_ASSET_SEASONALITY",
    "MultiTimeframeConfluence": "MULTI_ASSET_SEASONALITY",
    "ExpiryWeek": "MULTI_ASSET_SEASONALITY",
}


def count_independent_categories(strategy_names: List[str]) -> int:
    """Counts number of distinct strategic categories among fired strategies."""
    cats = set()
    for s in strategy_names:
        cats.add(STRATEGY_CATEGORIES.get(s, "OTHER"))
    return len(cats)




class MCXSession(str, Enum):
    MORNING = "MORNING"       # 09:00 - 13:00 IST
    AFTERNOON = "AFTERNOON"   # 13:00 - 17:00 IST
    EVENING = "EVENING"       # 17:00 - 23:30 IST
    OFF_MARKET = "OFF_MARKET" # Outside market hours

    @classmethod
    def from_time(cls, t: time) -> MCXSession:
        if time(9, 0) <= t < time(13, 0):
            return cls.MORNING
        elif time(13, 0) <= t < time(17, 0):
            return cls.AFTERNOON
        elif time(17, 0) <= t <= time(23, 30):
            return cls.EVENING
        else:
            return cls.OFF_MARKET


class SignalForgeStrategyAdapter(BaseCommodityStrategy):
    """
    Adapts SignalForge options strategies (S1, S7, S17, S25, etc.) into
    standard BaseCommodityStrategy for MCX Commodity Futures.
    """

    def __init__(
        self,
        sf_strategy_class: Any,
        name: Optional[str] = None,
        atr_period: int = 14,
        sl_atr_mult: float = 1.5,
        target_atr_mult: float = 3.0,
    ):
        raw_instance = sf_strategy_class() if callable(sf_strategy_class) else sf_strategy_class
        strat_name = name or getattr(raw_instance, "name", raw_instance.__class__.__name__)
        super().__init__(name=strat_name, version="1.0.0")
        self.sf_strategy = raw_instance
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.target_atr_mult = target_atr_mult
        self.parameters = {
            "atr_period": atr_period,
            "sl_atr_mult": sl_atr_mult,
            "target_atr_mult": target_atr_mult,
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

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

        if len(df) < max(20, self.atr_period + 5):
            return none_sig

        try:
            res = self.sf_strategy.evaluate(df)
        except Exception:
            return none_sig

        if not res or not isinstance(res, dict):
            return none_sig

        raw_dir = res.get("direction", Direction.NONE)
        conf = float(res.get("confidence", 0.0) or 0.0)

        # Map SignalForge BUY_CALL -> BUY (Long), BUY_PUT -> SELL (Short)
        if raw_dir in (Direction.BUY, Direction.BUY_CALL) or str(raw_dir) == "Direction.BUY_CALL":
            sig_dir = Direction.BUY
        elif raw_dir in (Direction.SELL, Direction.BUY_PUT) or str(raw_dir) == "Direction.BUY_PUT":
            sig_dir = Direction.SELL
        else:
            return none_sig

        close = float(df["close"].iloc[-1])
        high = df["high"].tail(self.atr_period).values
        low = df["low"].tail(self.atr_period).values
        cl = df["close"].tail(self.atr_period + 1).values
        tr_list = []
        for j in range(len(high)):
            tr = max(high[j] - low[j], abs(high[j] - cl[j]), abs(low[j] - cl[j]))
            tr_list.append(tr)
        atr = float(np.mean(tr_list)) if tr_list else max(close * 0.005, 10.0)

        tick = self.instrument_config.tick_size if self.instrument_config else 1.0

        if sig_dir == Direction.BUY:
            sl = self.round_to_tick(close - (atr * self.sl_atr_mult))
            target = self.round_to_tick(close + (atr * self.target_atr_mult))
        else:
            sl = self.round_to_tick(close + (atr * self.sl_atr_mult))
            target = self.round_to_tick(close - (atr * self.target_atr_mult))

        rr = round(abs(target - close) / max(abs(close - sl), 1e-4), 2)

        return StrategySignal(
            timestamp=ts,
            instrument=self.instrument_config.symbol if self.instrument_config else "MCX",
            contract=f"{self.instrument_config.symbol}-FRONT" if self.instrument_config else "MCX-FRONT",
            strategy=self.name,
            strategy_version=self.version,
            direction=sig_dir,
            confidence=conf,
            entry_price=close,
            stop_loss=sl,
            target=target,
            risk_reward=rr,
            decision="TRADE",
            is_valid=True,
            metadata=res.get("meta", {}),
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: Direction,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        if direction.is_long:
            raw_sl = entry_price - (atr * self.sl_atr_mult)
        else:
            raw_sl = entry_price + (atr * self.sl_atr_mult)
        return self.round_to_tick(raw_sl)

    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        direction: Direction,
        rr_ratio: Optional[float] = None,
    ) -> float:
        ratio = rr_ratio if rr_ratio is not None else 2.0
        risk = abs(entry_price - stop_loss)
        if direction.is_long:
            raw_target = entry_price + (risk * ratio)
        else:
            raw_target = entry_price - (risk * ratio)
        return self.round_to_tick(raw_target)



SIGNALFORGE_STRATEGY_CATALOG = [
    ("SuperTrend+RSI", "agents_code.agent2_strategy.s1_supertrend_rsi", "SuperTrendRSI"),
    ("VWAP+EMA", "agents_code.agent2_strategy.s2_vwap_ema", "VWAPEMACross"),
    ("ORB", "agents_code.agent2_strategy.s3_orb", "ORBStrategy"),
    ("BBSqueeze", "agents_code.agent2_strategy.s4_bb_squeeze", "BBSqueeze"),
    ("ADX+PSAR", "agents_code.agent2_strategy.s5_adx_psar", "ADXParabolicSAR"),
    ("FVG", "agents_code.agent2_strategy.s6_fvg", "FVGStrategy"),
    ("UTBot", "agents_code.agent2_strategy.s7_utbot", "UTBotStrategy"),
    ("CPR", "agents_code.agent2_strategy.s8_cpr", "CPRStrategy"),
    ("Ichimoku", "agents_code.agent2_strategy.s9_ichimoku", "IchimokuStrategy"),
    ("VolumeProfile", "agents_code.agent2_strategy.s10_volume_profile", "VolumeProfileStrategy"),
    ("LiqSweep", "agents_code.agent2_strategy.s11_liquidity_sweep", "LiquiditySweepStrategy"),
    ("PriceAction", "agents_code.agent2_strategy.s12_price_action", "PriceActionStrategy"),
    ("OIAnalysis", "agents_code.agent2_strategy.s13_oi_analysis", "OIAnalysisStrategy"),
    ("AMD", "agents_code.agent2_strategy.s15_amd", "AMDStrategy"),
    ("GapDirection", "agents_code.agent2_strategy.s16_gap_direction", "GapDirectionStrategy"),
    ("SMC", "agents_code.agent2_strategy.s17_smc", "SMCStrategy"),
    ("ValueArea", "agents_code.agent2_strategy.s20_volume_profile", "VolumeProfileStrategy"),
    ("GapMomentum", "agents_code.agent2_strategy.s22_gap_momentum", "GapMomentumStrategy"),
    ("ADXRising", "agents_code.agent2_strategy.s23_adx_rising", "ADXRisingStrategy"),
    ("RangeSpread", "agents_code.agent2_strategy.s24_range_spread", "RangeSpreadStrategy"),
    ("SqueezeMomentum", "agents_code.agent2_strategy.s25_squeeze_momentum", "SqueezeMomentumStrategy"),
    ("StochRSI", "agents_code.agent2_strategy.s26_stoch_rsi", "StochRSIStrategy"),
    ("EMASlope", "agents_code.agent2_strategy.s27_ema_slope", "EMASlopeStrategy"),
    ("HeikinAshi", "agents_code.agent2_strategy.s28_heikin_ashi", "HeikinAshiStrategy"),
    ("VWAPExtreme", "agents_code.agent2_strategy.s29_vwap_extreme", "VWAPExtremeStrategy"),
    ("OpeningRangeBias", "agents_code.agent2_strategy.s31_opening_range_bias", "OpeningRangeBiasStrategy"),
    # Excluded from offline backtest: ElliottWave (lagging) and StrikeMomentum (requires live websocket options chain; proxy causes counter-trend false entries)
    # ("ElliottWave", "agents_code.agent2_strategy.s34_elliott_wave", "ElliottWaveStrategy"),
    # ("StrikeMomentum", "agents_code.agent2_strategy.s21_strike_momentum", "StrikeMomentumStrategy"),
    ("GammaExposure", "agents_code.agent2_strategy.s32_gamma_exposure", "GammaExposureStrategy"),
    ("IVContraction", "agents_code.agent2_strategy.s14_iv_contraction", "IVContractionStrategy"),
    ("ExpiryWeek", "agents_code.agent2_strategy.s19_expiry_week", "ExpiryWeekStrategy"),
]


def build_catalog_27_strategy_suite() -> List[BaseCommodityStrategy]:
    """Instantiates all 27 canonical strategies adapted for MCX Commodity Futures."""
    suite: List[BaseCommodityStrategy] = []
    for name, mod_path, cls_name in SIGNALFORGE_STRATEGY_CATALOG:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            suite.append(SignalForgeStrategyAdapter(cls, name=name))
        except Exception as e:
            pass
    return suite


def build_default_strategy_suite(catalog_only: bool = False) -> List[BaseCommodityStrategy]:
    """
    Instantiates strategies for MCX Commodity Futures.
    Includes ALL 27 canonical strategies adapted from the catalog,
    along with dedicated institutional commodity models.
    """
    if catalog_only:
        return build_catalog_27_strategy_suite()

    suite: List[BaseCommodityStrategy] = [
        TrendFollowingStrategy(),
        OpeningRangeBreakoutStrategy(),
        VWAPMeanReversionStrategy(),
        VolatilityBreakoutStrategy(),
        DonchianBreakoutStrategy(),
        BollingerBandMeanReversionStrategy(),
        MovingAverageCrossoverStrategy(),
        RSIDivergenceStrategy(),
        MomentumVolumeBreakoutStrategy(),
        GoldSilverPairsStrategy(),
        OrderFlowDeltaStrategy(),
        TimeOfDaySeasonalityStrategy(),
        RSI2MeanReversionStrategy(),
        CalendarSeasonalityStrategy(),
        TermStructureStrategy(),
    ]

    # Add all 27 canonical adapted strategies from the catalog
    for name, mod_path, cls_name in SIGNALFORGE_STRATEGY_CATALOG:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            suite.append(SignalForgeStrategyAdapter(cls, name=name))
        except Exception:
            pass

    return suite


TOXIC_COMMODITY_STRATEGIES = {
    # Lagging breakout and false-whipsaw models that erode capital in chop
    "EMASlope", "DonchianBreakout", "ValueArea", "OpeningRangeBreakout",
    "UTBot", "StochRSI", "MomentumVolumeBreakout", "OIAnalysis", "FVG", "ADX+PSAR",
    "VolatilityBreakout", "SuperTrend+RSI", "BBMeanReversion", "RangeSpread", "Ichimoku",
    "SqueezeMomentum", "GoldSilverPairs",
}


def build_quality_strategy_suite() -> List[BaseCommodityStrategy]:
    """
    Returns only verified alpha-generating strategies for MCX Commodity Futures,
    pruning lagging breakout and false-whipsaw models that erode capital in chop.
    """
    full_suite = build_default_strategy_suite()
    return [s for s in full_suite if s.name not in TOXIC_COMMODITY_STRATEGIES]



@dataclass
class EnsembleTradeRecord:
    """Ensemble trade with multi-strategy attribution and MCX session details."""
    trade_id: str
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    lots: int
    gross_pnl_inr: float
    fees_inr: float
    net_pnl_inr: float
    pnl_pct: float
    pnl_points: float
    slippage_cost_inr: float
    exit_reason: str
    hold_bars: int
    hold_minutes: float
    session: str
    strategies_fired: List[str]
    lead_strategy: str
    votes: int
    categories: int = 1
    mfe_pts: float = 0.0
    mae_pts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entry_time"] = self.entry_time.isoformat()
        d["exit_time"] = self.exit_time.isoformat()
        d["strategies_fired"] = "+".join(self.strategies_fired)
        return d


@dataclass
class EnsembleRunResult:
    """Consolidated performance results from an ensemble backtest run."""
    symbol: str
    timeframe: str
    total_days: int
    trades: List[EnsembleTradeRecord]
    equity_curve: List[EquityPoint]
    metrics: PerformanceMetrics
    strategy_contributions: Dict[str, Dict[str, Any]]
    session_breakdown: Dict[str, Dict[str, Any]]
    initial_capital: float = 200_000.0
    ending_capital: float = 200_000.0
    peak_margin_seen: float = 0.0


class CommodityEnsembleEngine:
    """
    Executes unified multi-strategy backtests where all strategies contribute simultaneously.
    Provides vote aggregation, confluence gating, session tracking, and P&L attribution.
    """

    def __init__(
        self,
        strategies: Optional[List[BaseCommodityStrategy]] = None,
        instrument_config: InstrumentConfig = SILVERM_CONFIG,
        slippage_model: Optional[SlippageModel] = None,
        same_bar_rule: SameBarAmbiguityRule = SameBarAmbiguityRule.STOP_FIRST,
        regime_engine: Optional[MarketRegimeEngine] = None,
        min_votes: int = 1,
        min_categories: int = 1,
        max_positions: int = 1,
        capital: float = 200_000.0,
    ):
        self.strategies = strategies if strategies is not None else build_default_strategy_suite()
        self.config = instrument_config
        self.slippage_model = slippage_model or SlippageModel(SlippageType.FIXED_TICKS, 1.0)
        self.same_bar_rule = same_bar_rule
        self.regime_engine = regime_engine or MarketRegimeEngine()
        self.min_votes = min_votes
        self.min_categories = min_categories
        self.max_positions = max_positions
        self.capital = capital

        # Initialize all strategies
        for s in self.strategies:
            s.initialize(self.config)

    def run(
        self,
        df: pd.DataFrame,
        timeframe: str = "15m",
        allowed_sessions: Optional[List[MCXSession]] = None,
        lots: int = 1,
    ) -> EnsembleRunResult:
        """
        Executes simultaneous multi-strategy backtesting over df.
        """
        if df.empty:
            empty_m = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            return EnsembleRunResult(
                symbol=self.config.symbol,
                timeframe=timeframe,
                total_days=0,
                trades=[],
                equity_curve=[],
                metrics=empty_m,
                strategy_contributions={},
                session_breakdown={},
            )

        work_df = df.copy()
        if not isinstance(work_df.index, pd.DatetimeIndex):
            ts_col = "timestamp" if "timestamp" in work_df.columns else work_df.columns[0]
            work_df[ts_col] = pd.to_datetime(work_df[ts_col])
            work_df = work_df.set_index(ts_col)

        if work_df.index.tz is None:
            work_df.index = work_df.index.tz_localize(IST)
        else:
            work_df.index = work_df.index.tz_convert(IST)

        work_df = work_df.sort_index()

        tick_size = self.config.tick_size
        lot_size = self.config.lot_size
        qty = lots * lot_size
        multiplier = self.config.tick_value / (max(lot_size, 1) * max(tick_size, 1e-6))

        margin_obj = getattr(self.config, "margin", None)
        margin_pct = (getattr(margin_obj, "normal_margin_pct", 10.0) / 100.0) if margin_obj else 0.10

        completed_trades: List[EnsembleTradeRecord] = []
        equity_curve: List[EquityPoint] = []
        current_cash = self.capital
        peak_equity = self.capital
        peak_margin_seen = 0.0

        # Position tracking
        pos_open = False
        pos_dir: Optional[Direction] = None
        pos_entry_price = 0.0
        pos_entry_time: Optional[datetime] = None
        pos_sl = 0.0
        pos_target = 0.0
        pos_trailing_sl = 0.0
        pos_session = "MORNING"
        pos_strategies: List[str] = []
        pos_lead = ""
        pos_votes = 0
        pos_entry_idx = 0
        mfe_pts = 0.0
        mae_pts = 0.0

        pending_signal: Optional[Dict[str, Any]] = None
        trade_id_counter = 0

        warmup = 30
        n_bars = len(work_df)

        for i in range(warmup, n_bars):
            bar = work_df.iloc[i]
            bar_time = work_df.index[i].to_pydatetime()
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            # 1. Execute pending orders on bar open
            if pending_signal and not pos_open:
                sig_data = pending_signal
                pending_signal = None

                is_long = sig_data["direction"] == Direction.BUY
                slip_pts = self.slippage_model.calculate_slippage_points(bar_open, tick_size)
                fill_price = bar_open + slip_pts if is_long else bar_open - slip_pts

                pos_open = True
                pos_dir = sig_data["direction"]
                pos_entry_price = fill_price
                pos_entry_time = bar_time
                pos_sl = sig_data["stop_loss"]
                pos_target = sig_data["target"]
                pos_trailing_sl = sig_data["stop_loss"]
                pos_session = sig_data["session"]
                pos_strategies = sig_data["strategies"]
                pos_lead = sig_data["lead_strategy"]
                pos_votes = sig_data["votes"]
                pos_categories = sig_data.get("categories", 1)
                pos_entry_idx = i
                mfe_pts = 0.0
                mae_pts = 0.0

                req_margin = fill_price * qty * multiplier * margin_pct
                peak_margin_seen = max(peak_margin_seen, req_margin)

            # 2. Manage open position exits
            if pos_open:
                is_long = pos_dir == Direction.BUY
                # MFE / MAE tracking
                if is_long:
                    fav = bar_high - pos_entry_price
                    adv = pos_entry_price - bar_low
                else:
                    fav = pos_entry_price - bar_low
                    adv = bar_high - pos_entry_price
                mfe_pts = max(mfe_pts, fav)
                mae_pts = max(mae_pts, adv)

                # ── SignalForge Multi-Tier Progressive Profit Protection Ladder ──
                # As implemented in Position Manager (agent6) & ProfitLadder:
                # When position moves X% into profit, lock stop loss into progressive profit tiers.
                risk_pts = max(abs(pos_entry_price - pos_sl), 50.0)
                fav_pct = (fav / max(pos_entry_price, 1e-4)) * 100.0
                opt_gain_pct = fav_pct * 20.0  # Option leverage approx (Delta 0.50, Premium ~2.5% of spot)
                candles_held = i - pos_entry_idx

                # Allow position to breathe during initial entry swings (decision candles >= 2)
                if candles_held >= 2:
                    # Tier 1: Breakeven Lock (Option +18% / Spot +0.70% / 1.20R move) -> Lock Breakeven (+0.15R buffer)
                    if fav >= 1.20 * risk_pts or opt_gain_pct >= 18.0:
                        if is_long:
                            pos_trailing_sl = max(pos_trailing_sl, pos_entry_price + (0.15 * risk_pts))
                        else:
                            pos_trailing_sl = min(pos_trailing_sl, pos_entry_price - (0.15 * risk_pts))

                    # Tier 2: Solid Expansion (Option +28% / Spot +1.10% / 1.80R move) -> Lock +0.80R Profit
                    if fav >= 1.80 * risk_pts or opt_gain_pct >= 28.0:
                        if is_long:
                            pos_trailing_sl = max(pos_trailing_sl, pos_entry_price + (0.80 * risk_pts))
                        else:
                            pos_trailing_sl = min(pos_trailing_sl, pos_entry_price - (0.80 * risk_pts))

                    # Tier 3: High Momentum (Option +40% / Spot +1.50% / 2.50R move) -> Lock +1.50R Profit
                    if fav >= 2.50 * risk_pts or opt_gain_pct >= 40.0:
                        if is_long:
                            pos_trailing_sl = max(pos_trailing_sl, pos_entry_price + (1.50 * risk_pts))
                        else:
                            pos_trailing_sl = min(pos_trailing_sl, pos_entry_price - (1.50 * risk_pts))

                    # Tier 4: Super Runner (Option +55%+ / Spot +2.0%+ / 3.20R+) -> Dynamic Trailing 0.70R behind Peak
                    if fav >= 3.20 * risk_pts or opt_gain_pct >= 55.0:
                        peak_level = pos_entry_price + mfe_pts if is_long else pos_entry_price - mfe_pts
                        if is_long:
                            pos_trailing_sl = max(pos_trailing_sl, peak_level - (0.70 * risk_pts))
                        else:
                            pos_trailing_sl = min(pos_trailing_sl, peak_level + (0.70 * risk_pts))

                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""
                slip_pts = self.slippage_model.calculate_slippage_points(bar_close, tick_size)

                # Check if current trailing SL is at/above breakeven (profitable protection)
                is_trailing_profitable = (pos_trailing_sl >= pos_entry_price) if is_long else (pos_trailing_sl <= pos_entry_price)
                sl_reason_code = "TRAILING_SL" if is_trailing_profitable else "SL_HIT"

                # EOD Squareoff at 23:15
                if bar_time.time() >= time(23, 15):
                    exit_triggered = True
                    exit_price = bar_close - slip_pts if is_long else bar_close + slip_pts
                    exit_reason = "EOD_SQUAREOFF"

                # Check SL / Target
                if not exit_triggered:
                    target_hit = bar_high >= pos_target if is_long else bar_low <= pos_target
                    sl_hit = bar_low <= pos_trailing_sl if is_long else bar_high >= pos_trailing_sl

                    if target_hit and sl_hit:
                        if self.same_bar_rule == SameBarAmbiguityRule.TARGET_FIRST:
                            exit_triggered = True
                            exit_price = pos_target - slip_pts if is_long else pos_target + slip_pts
                            exit_reason = "TARGET_HIT"
                        else:
                            exit_triggered = True
                            exit_price = pos_trailing_sl - slip_pts if is_long else pos_trailing_sl + slip_pts
                            exit_reason = sl_reason_code
                    elif target_hit:
                        exit_triggered = True
                        exit_price = pos_target - slip_pts if is_long else pos_target + slip_pts
                        exit_reason = "TARGET_HIT"
                    elif sl_hit:
                        exit_triggered = True
                        exit_price = pos_trailing_sl - slip_pts if is_long else pos_trailing_sl + slip_pts
                        exit_reason = sl_reason_code

                if exit_triggered:
                    trade_id_counter += 1
                    pnl_pts = (exit_price - pos_entry_price) if is_long else (pos_entry_price - exit_price)
                    gross_pnl = pnl_pts * qty * multiplier

                    # Real MCX transaction costs
                    charges = calculate_commodity_trade_charges(
                        entry_price=pos_entry_price,
                        exit_price=exit_price,
                        quantity=qty,
                        direction="BUY" if is_long else "SELL",
                        tick_size=tick_size,
                        tick_value=self.config.tick_value,
                    )
                    total_fees = charges.total_charges
                    net_pnl = gross_pnl - total_fees
                    pnl_pct = (pnl_pts / max(pos_entry_price, 1e-4)) * 100.0


                    hold_bars = i - pos_entry_idx
                    hold_min = hold_bars * (15.0 if "15" in timeframe else (5.0 if "5" in timeframe else (30.0 if "30" in timeframe else 60.0)))

                    record = EnsembleTradeRecord(
                        trade_id=f"T{trade_id_counter:04d}",
                        symbol=self.config.symbol,
                        direction="BUY" if is_long else "SELL",
                        entry_time=pos_entry_time,
                        exit_time=bar_time,
                        entry_price=round(pos_entry_price, 2),
                        exit_price=round(exit_price, 2),
                        quantity=qty,
                        lots=lots,
                        gross_pnl_inr=round(gross_pnl, 2),
                        fees_inr=round(total_fees, 2),
                        net_pnl_inr=round(net_pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        pnl_points=round(pnl_pts, 2),
                        slippage_cost_inr=round(slip_pts * 2 * qty * multiplier, 2),
                        exit_reason=exit_reason,
                        hold_bars=hold_bars,
                        hold_minutes=round(hold_min, 1),
                        session=pos_session,
                        strategies_fired=pos_strategies,
                        lead_strategy=pos_lead,
                        votes=pos_votes,
                        categories=pos_categories,
                        mfe_pts=round(mfe_pts, 2),
                        mae_pts=round(mae_pts, 2),
                    )
                    completed_trades.append(record)

                    current_cash += net_pnl
                    peak_equity = max(peak_equity, current_cash)
                    dd_inr = peak_equity - current_cash
                    dd_pct = (dd_inr / peak_equity * 100.0) if peak_equity > 0 else 0.0

                    cum_gross = sum(t.gross_pnl_inr for t in completed_trades)
                    cum_fees = sum(t.fees_inr for t in completed_trades)
                    cum_slip = sum(t.slippage_cost_inr for t in completed_trades)
                    cum_net = sum(t.net_pnl_inr for t in completed_trades)

                    equity_curve.append(
                        EquityPoint(
                            timestamp=bar_time,
                            trade_id=record.trade_id,
                            gross_pnl_inr=record.gross_pnl_inr,
                            cumulative_gross_pnl=round(cum_gross, 2),
                            fees_inr=record.fees_inr,
                            cumulative_fees=round(cum_fees, 2),
                            slippage_cost_inr=record.slippage_cost_inr,
                            cumulative_slippage=round(cum_slip, 2),
                            net_pnl_inr=record.net_pnl_inr,
                            cumulative_net_pnl=round(cum_net, 2),
                            drawdown_inr=round(dd_inr, 2),
                            drawdown_pct=round(dd_pct, 2),
                        )
                    )


                    pos_open = False

            # 3. Evaluate strategies at bar close for next bar entry
            session_now = MCXSession.from_time(bar_time.time())
            if allowed_sessions and session_now not in allowed_sessions:
                continue

            if not pos_open and bar_time.time() < time(22, 30):
                lookback_start = max(0, i - 150)
                slice_df = work_df.iloc[lookback_start : i + 1]

                buy_votes = 0
                sell_votes = 0
                buy_fired: List[str] = []
                sell_fired: List[str] = []
                buy_signals: List[StrategySignal] = []
                sell_signals: List[StrategySignal] = []

                regime_info = self.regime_engine.classify(slice_df)

                for strat in self.strategies:
                    try:
                        sig = strat.generate_signal(slice_df, regime_details=regime_info)
                    except TypeError:
                        try:
                            sig = strat.generate_signal(slice_df)
                        except Exception:
                            continue
                    except Exception:
                        continue

                    if sig.is_valid and sig.direction != Direction.NONE:
                        if sig.direction == Direction.BUY:
                            buy_votes += 1
                            buy_fired.append(strat.name)
                            buy_signals.append(sig)
                        elif sig.direction == Direction.SELL:
                            sell_votes += 1
                            sell_fired.append(strat.name)
                            sell_signals.append(sig)

                # Check vote consensus
                if buy_votes >= self.min_votes and buy_votes > sell_votes and buy_signals:
                    best_sig = max(buy_signals, key=lambda s: s.confidence)
                    sl = best_sig.stop_loss if best_sig.stop_loss < bar_close else round(bar_close - max(150.0, bar_close * 0.007), 1)
                    tp = best_sig.target if best_sig.target > bar_close else round(bar_close + max(300.0, bar_close * 0.014), 1)
                    pending_signal = {
                        "direction": Direction.BUY,
                        "stop_loss": sl,
                        "target": tp,
                        "session": session_now.value,
                        "strategies": buy_fired,
                        "lead_strategy": best_sig.strategy,
                        "votes": buy_votes,
                    }
                elif sell_votes >= self.min_votes and sell_votes > buy_votes and sell_signals:
                    best_sig = max(sell_signals, key=lambda s: s.confidence)
                    sl = best_sig.stop_loss if best_sig.stop_loss > bar_close else round(bar_close + max(150.0, bar_close * 0.007), 1)
                    tp = best_sig.target if best_sig.target < bar_close else round(bar_close - max(300.0, bar_close * 0.014), 1)
                    pending_signal = {
                        "direction": Direction.SELL,
                        "stop_loss": sl,
                        "target": tp,
                        "session": session_now.value,
                        "strategies": sell_fired,
                        "lead_strategy": best_sig.strategy,
                        "votes": sell_votes,
                    }

        # Calculate metrics
        metrics = self._calculate_metrics(completed_trades, equity_curve)
        strat_contrib = self._calculate_strategy_contributions(completed_trades)
        session_breakdown = self._calculate_session_breakdown(completed_trades)

        total_days = len(set(work_df.index.date))

        return EnsembleRunResult(
            symbol=self.config.symbol,
            timeframe=timeframe,
            total_days=total_days,
            trades=completed_trades,
            equity_curve=equity_curve,
            metrics=metrics,
            strategy_contributions=strat_contrib,
            session_breakdown=session_breakdown,
            initial_capital=self.capital,
            ending_capital=round(current_cash, 2),
            peak_margin_seen=round(peak_margin_seen, 2),
        )

    def _calculate_metrics(
        self,
        trades: List[EnsembleTradeRecord],
        equity_curve: List[EquityPoint],
    ) -> PerformanceMetrics:
        if not trades:
            return PerformanceMetrics()

        n = len(trades)
        wins = [t for t in trades if t.net_pnl_inr > 0]
        losses = [t for t in trades if t.net_pnl_inr <= 0]
        w_cnt = len(wins)
        l_cnt = len(losses)
        win_rate = (w_cnt / n * 100.0) if n else 0.0

        gross = sum(t.gross_pnl_inr for t in trades)
        fees = sum(t.fees_inr for t in trades)
        slippage = sum(t.slippage_cost_inr for t in trades)
        net = sum(t.net_pnl_inr for t in trades)

        gross_win = sum(t.gross_pnl_inr for t in wins)
        gross_loss = abs(sum(t.gross_pnl_inr for t in losses))
        pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

        expectancy = net / n if n else 0.0
        avg_w = sum(t.net_pnl_inr for t in wins) / w_cnt if w_cnt else 0.0
        avg_l = abs(sum(t.net_pnl_inr for t in losses) / l_cnt) if l_cnt else 0.0
        wl_ratio = (avg_w / avg_l) if avg_l > 0 else 0.0

        max_dd_inr = max((p.drawdown_inr for p in equity_curve), default=0.0)
        max_dd_pct = max((p.drawdown_pct for p in equity_curve), default=0.0)

        # Sharpe & Sortino based on trade returns
        pnls = [t.net_pnl_inr for t in trades]
        std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 1.0
        sharpe = (float(np.mean(pnls)) / max(std_pnl, 1e-4) * math.sqrt(250)) if std_pnl > 0 else 0.0

        downside = [p for p in pnls if p < 0]
        downside_std = float(np.std(downside)) if len(downside) > 1 else 1.0
        sortino = (float(np.mean(pnls)) / max(downside_std, 1e-4) * math.sqrt(250)) if downside_std > 0 else 0.0
        calmar = (net / max(max_dd_inr, 1.0)) if max_dd_inr > 0 else 0.0

        avg_min = sum(t.hold_minutes for t in trades) / n
        avg_mfe = sum(t.mfe_pts for t in trades) / n
        avg_mae = sum(t.mae_pts for t in trades) / n
        mfe_mae_ratio = (avg_mfe / avg_mae) if avg_mae > 0 else 0.0

        return PerformanceMetrics(
            total_trades=n,
            winning_trades=w_cnt,
            losing_trades=l_cnt,
            win_rate_pct=round(win_rate, 2),
            gross_pnl_inr=round(gross, 2),
            total_costs_inr=round(fees, 2),
            total_slippage_inr=round(slippage, 2),
            net_pnl_inr=round(net, 2),
            profit_factor=round(pf, 2),
            expectancy_inr=round(expectancy, 2),
            avg_win_inr=round(avg_w, 2),
            avg_loss_inr=round(avg_l, 2),
            win_loss_ratio=round(wl_ratio, 2),
            max_drawdown_inr=round(max_dd_inr, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            avg_holding_time_minutes=round(avg_min, 1),
            avg_mfe_points=round(avg_mfe, 2),
            avg_mae_points=round(avg_mae, 2),
            mfe_mae_ratio=round(mfe_mae_ratio, 2),
            sl_hit_count=sum(1 for t in trades if "SL" in t.exit_reason),
            target_hit_count=sum(1 for t in trades if "TARGET" in t.exit_reason),
            eod_exit_count=sum(1 for t in trades if "EOD" in t.exit_reason),
        )


    def _calculate_strategy_contributions(self, trades: List[EnsembleTradeRecord]) -> Dict[str, Dict[str, Any]]:
        contrib: Dict[str, Dict[str, Any]] = {}
        total_net = sum(t.net_pnl_inr for t in trades)

        for strat in self.strategies:
            s_name = strat.name
            participating = [t for t in trades if s_name in t.strategies_fired]
            t_cnt = len(participating)
            wins = len([t for t in participating if t.net_pnl_inr > 0])
            losses = len([t for t in participating if t.net_pnl_inr <= 0])
            gross = sum(t.gross_pnl_inr for t in participating)
            fees = sum(t.fees_inr for t in participating)
            net = sum(t.net_pnl_inr for t in participating)
            pf = (sum(t.gross_pnl_inr for t in participating if t.gross_pnl_inr > 0) /
                  max(abs(sum(t.gross_pnl_inr for t in participating if t.gross_pnl_inr <= 0)), 1e-4))
            contrib_pct = (net / max(abs(total_net), 1e-4) * 100.0) if total_net != 0 else 0.0

            contrib[s_name] = {
                "trades": t_cnt,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round((wins / t_cnt * 100.0) if t_cnt else 0.0, 1),
                "gross_pnl_inr": round(gross, 2),
                "fees_inr": round(fees, 2),
                "net_pnl_inr": round(net, 2),
                "profit_factor": round(pf, 2),
                "contribution_pct": round(contrib_pct, 1),
            }
        return contrib

    def _calculate_session_breakdown(self, trades: List[EnsembleTradeRecord]) -> Dict[str, Dict[str, Any]]:
        breakdown: Dict[str, Dict[str, Any]] = {}
        for sess in [MCXSession.MORNING.value, MCXSession.AFTERNOON.value, MCXSession.EVENING.value]:
            sess_trades = [t for t in trades if t.session == sess]
            t_cnt = len(sess_trades)
            wins = len([t for t in sess_trades if t.net_pnl_inr > 0])
            gross = sum(t.gross_pnl_inr for t in sess_trades)
            fees = sum(t.fees_inr for t in sess_trades)
            net = sum(t.net_pnl_inr for t in sess_trades)
            pf = (sum(t.gross_pnl_inr for t in sess_trades if t.gross_pnl_inr > 0) /
                  max(abs(sum(t.gross_pnl_inr for t in sess_trades if t.gross_pnl_inr <= 0)), 1e-4))

            breakdown[sess] = {
                "trades": t_cnt,
                "wins": wins,
                "win_rate_pct": round((wins / t_cnt * 100.0) if t_cnt else 0.0, 1),
                "gross_pnl_inr": round(gross, 2),
                "fees_inr": round(fees, 2),
                "net_pnl_inr": round(net, 2),
                "profit_factor": round(pf, 2),
            }
        return breakdown

"""
core/strategies/backtest_portfolio.py — Multi-Strategy Combined Portfolio Simulator
===================================================================================
Simulates the actual combined portfolio of multiple strategies running together under:
- Shared Capital (tiers: ₹1,00,000, ₹2,00,000, ₹5,00,000, ₹10,00,000)
- Shared Risk Limits:
  * Maximum concurrent open positions (e.g. max 2 positions across all strategies)
  * Daily loss limit (e.g. ₹5,000 max daily loss stops new entries for the day)
- Conflict resolution for simultaneous signals (directional veto / prioritization)
- Margin utilization tracking (SILVERMIC ~10% margin ~₹8,500-₹10,000/lot)
- Portfolio-level equity curve and drawdown tracking
- Signal correlation and concurrence analysis
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import pytz

from core.models import Direction
from core.regime.engine import MarketRegimeEngine, MarketRegime, RegimeDetails
from instruments.base import InstrumentConfig
from instruments import SILVERMIC_CONFIG
from core.strategies.base import BaseCommodityStrategy, StrategySignal
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


@dataclass
class PortfolioPosition:
    """Active open position in the multi-strategy portfolio."""
    strategy_name: str
    direction: Direction
    entry_time: datetime
    entry_price: float
    stop_loss: float
    target: float
    lots: int
    quantity: int
    margin_blocked: float
    highest_price: float
    lowest_price: float
    entry_bar_idx: int
    regime: str
    trailing_sl: float = 0.0


@dataclass
class PortfolioRunSummary:
    """Consolidated performance results of a combined portfolio simulation."""
    capital_tier_inr: float
    initial_capital_inr: float
    ending_capital_inr: float
    net_pnl_inr: float
    gross_pnl_inr: float
    total_costs_inr: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_inr: float
    max_drawdown_inr: float
    max_drawdown_pct: float
    max_margin_utilization_pct: float
    max_concurrent_positions: int
    days_daily_loss_hit: int
    simultaneous_signals_count: int
    conflicting_signals_count: int
    strategy_trade_counts: Dict[str, int] = field(default_factory=dict)
    strategy_net_pnls: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CommodityPortfolioBacktestEngine:
    """
    Simulates unified portfolio execution across multiple strategies
    with shared capital, margin constraints, and risk governance.
    """

    def __init__(
        self,
        capital: float = 200_000.0,
        max_concurrent_positions: int = 2,
        daily_loss_limit_inr: float = 5_000.0,
        slippage_model: Optional[SlippageModel] = None,
        same_bar_rule: SameBarAmbiguityRule = SameBarAmbiguityRule.STOP_FIRST,
        regime_engine: Optional[MarketRegimeEngine] = None,
    ):
        self.capital = capital
        self.max_concurrent_positions = max_concurrent_positions
        self.daily_loss_limit_inr = daily_loss_limit_inr
        self.slippage_model = slippage_model or SlippageModel(SlippageType.FIXED_TICKS, 1.0)
        self.same_bar_rule = same_bar_rule
        self.regime_engine = regime_engine or MarketRegimeEngine()

    def run_portfolio(
        self,
        strategies: List[BaseCommodityStrategy],
        instrument_config: InstrumentConfig,
        df: pd.DataFrame,
        timeframe: str = "5m",
        lots_per_trade: int = 1,
    ) -> Tuple[PortfolioRunSummary, List[BacktestTrade], List[EquityPoint]]:
        """
        Executes simultaneous multi-strategy bar-by-bar backtest with shared capital and limits.
        """
        if df.empty or not strategies:
            empty_summary = PortfolioRunSummary(
                capital_tier_inr=self.capital,
                initial_capital_inr=self.capital,
                ending_capital_inr=self.capital,
                net_pnl_inr=0.0,
                gross_pnl_inr=0.0,
                total_costs_inr=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                expectancy_inr=0.0,
                max_drawdown_inr=0.0,
                max_drawdown_pct=0.0,
                max_margin_utilization_pct=0.0,
                max_concurrent_positions=0,
                days_daily_loss_hit=0,
                simultaneous_signals_count=0,
                conflicting_signals_count=0,
            )
            return empty_summary, [], []

        # Prepare data
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

        # Initialize strategies
        for s in strategies:
            s.initialize(instrument_config)

        # State tracking
        current_cash = self.capital
        peak_equity = self.capital
        open_positions: List[PortfolioPosition] = []
        completed_trades: List[BacktestTrade] = []
        equity_curve: List[EquityPoint] = []

        pending_signals: List[Tuple[BaseCommodityStrategy, StrategySignal]] = []

        current_day: Optional[date] = None
        day_realized_pnl = 0.0
        daily_loss_locked = False
        days_daily_loss_hit = 0

        max_margin_seen = 0.0
        simultaneous_signals_count = 0
        conflicting_signals_count = 0

        margin_obj = getattr(instrument_config, "margin", None)
        margin_pct = (getattr(margin_obj, "normal_margin_pct", 10.0) / 100.0) if margin_obj else 0.10

        n_bars = len(work_df)
        trade_counter = 0

        # Bar-by-bar simulation
        for i in range(n_bars):
            bar = work_df.iloc[i]
            bar_time = work_df.index[i]
            bar_date = bar_time.date()

            # 1. Day Rollover & Daily Loss Limit Reset
            if current_day != bar_date:
                current_day = bar_date
                day_realized_pnl = 0.0
                daily_loss_locked = False

            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            # 2. Execute Pending Orders from Bar N-1 at Bar N Open
            if pending_signals and not daily_loss_locked:
                # Check for conflicts among pending signals
                buy_signals = [item for item in pending_signals if item[1].direction == Direction.BUY]
                sell_signals = [item for item in pending_signals if item[1].direction == Direction.SELL]

                if len(pending_signals) > 1:
                    simultaneous_signals_count += 1

                # If simultaneous opposite signals appear, cancel conflicting orders for safety
                if buy_signals and sell_signals:
                    conflicting_signals_count += 1
                    # Conflicted: take highest confidence or skip both
                    pending_signals = []
                else:
                    for strat, sig in pending_signals:
                        # Check capacity
                        if len(open_positions) >= self.max_concurrent_positions:
                            break

                        # Check if this strategy already has an open position
                        strat_has_pos = any(p.strategy_name == strat.name for p in open_positions)
                        if strat_has_pos:
                            continue

                        # Calculate margin required
                        qty = lots_per_trade * instrument_config.lot_size
                        contract_val = bar_open * qty
                        margin_req = contract_val * margin_pct

                        # Check available cash
                        blocked_now = sum(p.margin_blocked for p in open_positions)
                        if (current_cash - blocked_now) < margin_req:
                            # Insufficient margin to open new position
                            continue

                        # Execute with slippage
                        slip_pts = self.slippage_model.calculate_slippage_points(bar_open, instrument_config.tick_size)
                        if sig.direction == Direction.BUY:
                            fill_price = strat.round_to_tick(bar_open + slip_pts)
                        else:
                            fill_price = strat.round_to_tick(bar_open - slip_pts)

                        new_pos = PortfolioPosition(
                            strategy_name=strat.name,
                            direction=sig.direction,
                            entry_time=bar_time,
                            entry_price=fill_price,
                            stop_loss=sig.stop_loss,
                            target=sig.target,
                            lots=lots_per_trade,
                            quantity=qty,
                            margin_blocked=margin_req,
                            highest_price=fill_price,
                            lowest_price=fill_price,
                            entry_bar_idx=i,
                            regime=sig.regime.value if hasattr(sig.regime, "value") else str(sig.regime),
                            trailing_sl=sig.stop_loss,
                        )
                        open_positions.append(new_pos)

                pending_signals = []

            # 3. Track Intra-Bar Extremes for Open Positions
            for pos in open_positions:
                pos.highest_price = max(pos.highest_price, bar_high)
                pos.lowest_price = min(pos.lowest_price, bar_low)

            # 4. Check Exits (Target, Stop Loss, Trailing Stop, EOD Squareoff)
            remaining_positions: List[PortfolioPosition] = []
            is_eod = bar_time.time() >= time(23, 15)

            for pos in open_positions:
                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""
                is_ambiguous = False

                slip_pts = self.slippage_model.calculate_slippage_points(bar_close, instrument_config.tick_size)

                if pos.direction == Direction.BUY:
                    target_touched = bar_high >= pos.target
                    sl_touched = bar_low <= pos.trailing_sl

                    if target_touched and sl_touched:
                        is_ambiguous = True
                        if self.same_bar_rule == SameBarAmbiguityRule.TARGET_FIRST:
                            exit_triggered = True
                            exit_price = strat.round_to_tick(pos.target - slip_pts)
                            exit_reason = "TARGET_HIT"
                        else:
                            exit_triggered = True
                            exit_price = strat.round_to_tick(pos.trailing_sl - slip_pts)
                            exit_reason = "SL_HIT"
                    elif target_touched:
                        exit_triggered = True
                        exit_price = strat.round_to_tick(pos.target - slip_pts)
                        exit_reason = "TARGET_HIT"
                    elif sl_touched:
                        exit_triggered = True
                        exit_price = strat.round_to_tick(pos.trailing_sl - slip_pts)
                        exit_reason = "SL_HIT"
                    elif is_eod:
                        exit_triggered = True
                        exit_price = strat.round_to_tick(bar_close - slip_pts)
                        exit_reason = "EOD_SQUAREOFF"

                else:  # SELL (Short)
                    target_touched = bar_low <= pos.target
                    sl_touched = bar_high >= pos.trailing_sl

                    if target_touched and sl_touched:
                        is_ambiguous = True
                        if self.same_bar_rule == SameBarAmbiguityRule.TARGET_FIRST:
                            exit_triggered = True
                            exit_price = strat.round_to_tick(pos.target + slip_pts)
                            exit_reason = "TARGET_HIT"
                        else:
                            exit_triggered = True
                            exit_price = strat.round_to_tick(pos.trailing_sl + slip_pts)
                            exit_reason = "SL_HIT"
                    elif target_touched:
                        exit_triggered = True
                        exit_price = strat.round_to_tick(pos.target + slip_pts)
                        exit_reason = "TARGET_HIT"
                    elif sl_touched:
                        exit_triggered = True
                        exit_price = strat.round_to_tick(pos.trailing_sl + slip_pts)
                        exit_reason = "SL_HIT"
                    elif is_eod:
                        exit_triggered = True
                        exit_price = strat.round_to_tick(bar_close + slip_pts)
                        exit_reason = "EOD_SQUAREOFF"

                if exit_triggered:
                    trade_counter += 1
                    # Compute P&L & Real Statutory Fees
                    pnl_pts = (exit_price - pos.entry_price) if pos.direction == Direction.BUY else (pos.entry_price - exit_price)
                    gross_pnl = pnl_pts * pos.quantity * instrument_config.tick_value

                    # Real Statutory Fees & P&L
                    charges = calculate_commodity_trade_charges(
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        quantity=pos.quantity,
                        direction="BUY" if pos.direction == Direction.BUY else "SELL",
                        tick_size=instrument_config.tick_size,
                        tick_value=instrument_config.tick_value,
                    )
                    gross_pnl = charges.gross_pnl_inr
                    total_fees = float(charges.total_charges)
                    net_pnl = charges.net_pnl_inr

                    # MFE / MAE
                    if pos.direction == Direction.BUY:
                        mfe = max(0.0, pos.highest_price - pos.entry_price)
                        mae = max(0.0, pos.entry_price - pos.lowest_price)
                    else:
                        mfe = max(0.0, pos.entry_price - pos.lowest_price)
                        mae = max(0.0, pos.highest_price - pos.entry_price)

                    hold_bars = i - pos.entry_bar_idx
                    hold_min = float((bar_time - pos.entry_time).total_seconds() / 60.0)

                    completed_trade = BacktestTrade(
                        trade_id=f"PT_{trade_counter}_{pos.strategy_name}",
                        strategy=pos.strategy_name,
                        instrument=instrument_config.symbol,
                        contract=instrument_config.symbol,
                        direction="BUY" if pos.direction == Direction.BUY else "SELL",
                        entry_time=pos.entry_time,
                        entry_price=pos.entry_price,
                        exit_time=bar_time,
                        exit_price=exit_price,
                        stop_loss=pos.stop_loss,
                        target=pos.target,
                        quantity=pos.quantity,
                        lots=pos.lots,
                        gross_pnl_inr=round(gross_pnl, 2),
                        net_pnl_inr=round(net_pnl, 2),
                        pnl_points=round(pnl_pts, 2),
                        slippage_pts=round(slip_pts * 2, 2),
                        slippage_cost_inr=round(slip_pts * 2 * pos.quantity * instrument_config.tick_value, 2),
                        fees_inr=round(total_fees, 2),
                        mfe_pts=round(mfe, 2),
                        mae_pts=round(mae, 2),
                        hold_bars=hold_bars,
                        hold_minutes=hold_min,
                        regime=pos.regime,
                        time_of_day_bucket="MCX Session",
                        exit_reason=exit_reason,
                        is_ambiguous_bar=is_ambiguous,
                    )
                    completed_trades.append(completed_trade)

                    # Update portfolio cash
                    current_cash += net_pnl
                    day_realized_pnl += net_pnl

                    # Check daily loss limit
                    if day_realized_pnl <= -self.daily_loss_limit_inr and not daily_loss_locked:
                        daily_loss_locked = True
                        days_daily_loss_hit += 1

                    # Track Equity
                    peak_equity = max(peak_equity, current_cash)
                    dd_inr = peak_equity - current_cash
                    dd_pct = (dd_inr / peak_equity * 100.0) if peak_equity > 0 else 0.0

                    cum_gross = sum(t.gross_pnl_inr for t in completed_trades)
                    cum_fees = sum(t.fees_inr for t in completed_trades)
                    cum_net = sum(t.net_pnl_inr for t in completed_trades)

                    equity_curve.append(
                        EquityPoint(
                            timestamp=bar_time,
                            trade_id=completed_trade.trade_id,
                            gross_pnl_inr=completed_trade.gross_pnl_inr,
                            cumulative_gross_pnl=round(cum_gross, 2),
                            fees_inr=completed_trade.fees_inr,
                            cumulative_fees=round(cum_fees, 2),
                            slippage_cost_inr=completed_trade.slippage_cost_inr,
                            cumulative_slippage=0.0,
                            net_pnl_inr=completed_trade.net_pnl_inr,
                            cumulative_net_pnl=round(cum_net, 2),
                            drawdown_inr=round(dd_inr, 2),
                            drawdown_pct=round(dd_pct, 2),
                        )
                    )
                else:
                    remaining_positions.append(pos)

            open_positions = remaining_positions

            # Track peak margin utilization
            current_margin = sum(p.margin_blocked for p in open_positions)
            max_margin_seen = max(max_margin_seen, current_margin)

            # 5. Generate Signals for Next Bar at Bar N Close (Strict No-Lookahead)
            if not daily_loss_locked and len(open_positions) < self.max_concurrent_positions and not is_eod:
                # Slice history up to current bar
                lookback_start = max(0, i - 150)
                history_slice = work_df.iloc[lookback_start : i + 1]

                regime_details = self.regime_engine.classify(history_slice)

                for strat in strategies:
                    # Skip if strategy already has an open position
                    if any(p.strategy_name == strat.name for p in open_positions):
                        continue

                    sig = strat.generate_signal(history_slice, regime_details=regime_details)
                    if sig.direction != Direction.NONE and sig.is_valid:
                        pending_signals.append((strat, sig))

        # Close any lingering positions at end of dataset
        if open_positions:
            last_bar = work_df.iloc[-1]
            last_time = work_df.index[-1]
            last_close = float(last_bar["close"])
            slip_pts = self.slippage_model.calculate_slippage_points(last_close, instrument_config.tick_size)

            for pos in open_positions:
                trade_counter += 1
                exit_price = strat.round_to_tick(last_close - slip_pts) if pos.direction == Direction.BUY else strat.round_to_tick(last_close + slip_pts)
                pnl_pts = (exit_price - pos.entry_price) if pos.direction == Direction.BUY else (pos.entry_price - exit_price)
                gross_pnl = pnl_pts * pos.quantity * instrument_config.tick_value

                # Real Statutory Fees & P&L
                charges = calculate_commodity_trade_charges(
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    quantity=pos.quantity,
                    direction="BUY" if pos.direction == Direction.BUY else "SELL",
                    tick_size=instrument_config.tick_size,
                    tick_value=instrument_config.tick_value,
                )
                gross_pnl = charges.gross_pnl_inr
                total_fees = float(charges.total_charges)
                net_pnl = charges.net_pnl_inr

                completed_trades.append(
                    BacktestTrade(
                        trade_id=f"PT_{trade_counter}_{pos.strategy_name}",
                        strategy=pos.strategy_name,
                        instrument=instrument_config.symbol,
                        contract=instrument_config.symbol,
                        direction="BUY" if pos.direction == Direction.BUY else "SELL",
                        entry_time=pos.entry_time,
                        entry_price=pos.entry_price,
                        exit_time=last_time,
                        exit_price=exit_price,
                        stop_loss=pos.stop_loss,
                        target=pos.target,
                        quantity=pos.quantity,
                        lots=pos.lots,
                        gross_pnl_inr=round(gross_pnl, 2),
                        net_pnl_inr=round(net_pnl, 2),
                        pnl_points=round(pnl_pts, 2),
                        slippage_pts=round(slip_pts * 2, 2),
                        slippage_cost_inr=round(slip_pts * 2 * pos.quantity * instrument_config.tick_value, 2),
                        fees_inr=round(total_fees, 2),
                        mfe_pts=0.0,
                        mae_pts=0.0,
                        hold_bars=n_bars - pos.entry_bar_idx,
                        hold_minutes=float((last_time - pos.entry_time).total_seconds() / 60.0),
                        regime=pos.regime,
                        time_of_day_bucket="MCX Session",
                        exit_reason="END_OF_DATA",
                    )
                )
                current_cash += net_pnl

        # Calculate portfolio summary metrics
        total_tr = len(completed_trades)
        wins = [t for t in completed_trades if t.net_pnl_inr > 0]
        losses = [t for t in completed_trades if t.net_pnl_inr < 0]

        win_rate = (len(wins) / total_tr * 100.0) if total_tr > 0 else 0.0
        tot_gross = sum(t.gross_pnl_inr for t in completed_trades)
        tot_fees = sum(t.fees_inr for t in completed_trades)
        tot_net = sum(t.net_pnl_inr for t in completed_trades)

        tot_win_pnl = sum(t.net_pnl_inr for t in wins)
        tot_loss_pnl = abs(sum(t.net_pnl_inr for t in losses))
        profit_factor = (tot_win_pnl / tot_loss_pnl) if tot_loss_pnl > 0 else (99.0 if tot_win_pnl > 0 else 0.0)
        expectancy = (tot_net / total_tr) if total_tr > 0 else 0.0

        max_dd_inr = 0.0
        max_dd_pct = 0.0
        if equity_curve:
            max_dd_inr = max(e.drawdown_inr for e in equity_curve)
            max_dd_pct = max(e.drawdown_pct for e in equity_curve)

        max_util_pct = (max_margin_seen / self.capital * 100.0) if self.capital > 0 else 0.0

        strat_counts: Dict[str, int] = {}
        strat_pnls: Dict[str, float] = {}
        for t in completed_trades:
            strat_counts[t.strategy] = strat_counts.get(t.strategy, 0) + 1
            strat_pnls[t.strategy] = round(strat_pnls.get(t.strategy, 0.0) + t.net_pnl_inr, 2)

        summary = PortfolioRunSummary(
            capital_tier_inr=self.capital,
            initial_capital_inr=self.capital,
            ending_capital_inr=round(current_cash, 2),
            net_pnl_inr=round(tot_net, 2),
            gross_pnl_inr=round(tot_gross, 2),
            total_costs_inr=round(tot_fees, 2),
            total_trades=total_tr,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            expectancy_inr=round(expectancy, 2),
            max_drawdown_inr=round(max_dd_inr, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_margin_utilization_pct=round(max_util_pct, 2),
            max_concurrent_positions=self.max_concurrent_positions,
            days_daily_loss_hit=days_daily_loss_hit,
            simultaneous_signals_count=simultaneous_signals_count,
            conflicting_signals_count=conflicting_signals_count,
            strategy_trade_counts=strat_counts,
            strategy_net_pnls=strat_pnls,
        )

        return summary, completed_trades, equity_curve

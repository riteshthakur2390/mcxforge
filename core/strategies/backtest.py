"""
core/strategies/backtest.py — Institutional Backtesting Engine for MCX Commodity Strategies
==========================================================================================
Provides complete, reproducible, non-lookahead backtesting for MCX commodity futures:
- Strict zero look-ahead bias (indicators on candle N, execution on candle N+1 open)
- Realistic execution with configurable slippage (ticks, points, percentage)
- Conservative deterministic rule for same-bar SL/Target touch (STOP_FIRST default)
- Full statutory MCX fees: Brokerage, CTT, Exchange turnover, SEBI charges, Stamp duty, GST
- Excursion telemetry: Maximum Favorable (MFE) & Maximum Adverse (MAE) tracking
- Multi-timeframe support: 1m, 3m, 5m, 15m, 30m, 1h
- Futures contract modes: Individual (with 5-day physical tender lockout) vs Continuous
- Out-of-sample (OOS) train/test partitioning
- Walk-forward rolling window analysis
- Slippage sensitivity analysis (0, 1, 2, 3, 5 ticks)
- Detailed breakdowns: Regime, Time-of-Day, Long vs Short, Equity Curve
- Clean CLI interface with output persistence
"""

import argparse
from datetime import datetime, time, timedelta, date
import json
import os
import sys
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
import pytz

from core.models import Direction, Position, TradePlan, RawSignal, JournalEntry
from core.regime.engine import MarketRegimeEngine, MarketRegime, RegimeDetails
from instruments.base import InstrumentConfig, ContractSpec
from instruments import (
    SILVERMIC_CONFIG,
    GOLDM_CONFIG,
    CRUDEOILM_CONFIG,
    NATGASM_CONFIG,
    get_instrument_config,
)
from core.strategies.base import BaseCommodityStrategy, StrategySignal
from core.strategies.registry import StrategyRegistry
from core.strategies.backtest_models import (
    SlippageType,
    SlippageModel,
    ContractMode,
    SameBarAmbiguityRule,
    BacktestConfig,
    BacktestTrade,
    EquityPoint,
    PerformanceMetrics,
    BacktestResult,
)
from core.strategies.backtest_data import HistoricalDataLoader, resample_ohlcv
from utils.brokerage_calculator import calculate_commodity_trade_charges

IST = pytz.timezone("Asia/Kolkata")


class CommodityBacktestEngine:
    """
    Institutional backtest engine strictly enforcing zero lookahead bias
    and realistic commodity futures market mechanics.
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        slippage_ticks: Optional[int] = None,
        same_bar_rule: SameBarAmbiguityRule = SameBarAmbiguityRule.STOP_FIRST,
        regime_engine: Optional[MarketRegimeEngine] = None,
        enable_intraday_eod_squareoff: bool = True,
        eod_squareoff_time: str = "23:15",
    ):
        if slippage_model is not None:
            self.slippage_model = slippage_model
        elif slippage_ticks is not None:
            self.slippage_model = SlippageModel(SlippageType.FIXED_TICKS, float(slippage_ticks))
        else:
            self.slippage_model = SlippageModel(SlippageType.FIXED_TICKS, 1.0)
        self.same_bar_rule = same_bar_rule
        self.regime_engine = regime_engine or MarketRegimeEngine()
        self.enable_intraday_eod_squareoff = enable_intraday_eod_squareoff
        self.eod_time = datetime.strptime(eod_squareoff_time, "%H:%M").time()

    def _get_time_of_day_bucket(self, t: time) -> str:
        """Categorizes trade time into standard MCX trading phases."""
        if t < time(11, 0):
            return "09:00-11:00 (Opening Discovery)"
        elif t < time(14, 0):
            return "11:00-14:00 (European Pre-Market)"
        elif t < time(17, 0):
            return "14:00-17:00 (European Active)"
        elif t < time(20, 0):
            return "17:00-20:00 (US Opening Overlap)"
        else:
            return "20:00-23:30 (US Main Session)"

    def run(
        self,
        strategy: BaseCommodityStrategy,
        instrument_config: InstrumentConfig,
        df: pd.DataFrame,
        timeframe: str = "5m",
        lots: int = 1,
        contract_mode: ContractMode = ContractMode.CONTINUOUS_CONTRACT,
        contract_expiry: Optional[date] = None,
        parameters: Optional[Dict[str, Any]] = None,
        is_oos: bool = False,
    ) -> BacktestResult:
        """
        Executes non-lookahead backtest on historical OHLCV data.
        """
        if df is None or len(df) < 10:
            raise ValueError(f"Insufficient historical data: {len(df) if df is not None else 0} bars (min 10 required).")

        # Initialize strategy
        strategy.initialize(instrument_config, parameters=parameters)

        tick_size = instrument_config.tick_size
        lot_size = instrument_config.lot_size
        multiplier = instrument_config.tick_value / max(tick_size, 1e-6)
        qty = lots * lot_size

        # Create Config
        backtest_id = f"BT_{strategy.name}_{instrument_config.symbol}_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}"
        config = BacktestConfig(
            backtest_id=backtest_id,
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            instrument_symbol=instrument_config.symbol,
            timeframe=timeframe,
            contract_mode=contract_mode,
            contract_symbol=f"{instrument_config.symbol}-FRONT",
            start_time=df.index[0].isoformat() if hasattr(df.index[0], "isoformat") else str(df.index[0]),
            end_time=df.index[-1].isoformat() if hasattr(df.index[-1], "isoformat") else str(df.index[-1]),
            lots=lots,
            slippage=self.slippage_model,
            same_bar_rule=self.same_bar_rule,
            strategy_parameters=strategy.parameters.copy(),
        )

        trades: List[BacktestTrade] = []
        equity_curve: List[EquityPoint] = []

        # Position tracking
        pos_open = False
        pos_direction: Optional[Direction] = None
        pos_entry_price = 0.0
        pos_entry_time: Optional[datetime] = None
        pos_sl = 0.0
        pos_target = 0.0
        pos_entry_bar = 0
        pos_regime = MarketRegime.RANGE.value
        pos_contract = f"{instrument_config.symbol}-FRONT"
        pos_slippage_pts = 0.0
        mfe_pts = 0.0
        mae_pts = 0.0
        pos_ambiguous = False

        # Order queue: signal generated on candle i enters on candle i+1 open
        pending_signal: Optional[StrategySignal] = None

        cum_gross = 0.0
        cum_fees = 0.0
        cum_slippage = 0.0
        cum_net = 0.0
        peak_net = 0.0

        warmup_bars = 25
        n_bars = len(df)

        for i in range(warmup_bars, n_bars):
            bar = df.iloc[i]
            bar_time = df.index[i] if isinstance(df.index[i], (datetime, pd.Timestamp)) else datetime.now(IST)
            if hasattr(bar_time, "to_pydatetime"):
                bar_time = bar_time.to_pydatetime()

            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            # ── 1. EXECUTE PENDING ENTRY ON CANDLE OPEN ───────────────────────
            if pending_signal and not pos_open:
                sig = pending_signal
                pending_signal = None

                is_long = sig.direction.is_long
                slip_pts = self.slippage_model.calculate_slippage_points(bar_open, tick_size)
                fill_price = bar_open + slip_pts if is_long else bar_open - slip_pts
                fill_price = strategy.round_to_tick(fill_price)

                pos_open = True
                pos_direction = sig.direction
                pos_entry_price = fill_price
                pos_entry_time = bar_time
                pos_sl = sig.stop_loss
                pos_target = sig.target
                pos_entry_bar = i
                pos_regime = sig.regime.value if hasattr(sig.regime, "value") else str(sig.regime)
                pos_contract = sig.contract
                pos_slippage_pts = slip_pts
                mfe_pts = 0.0
                mae_pts = 0.0
                pos_ambiguous = False

            # ── 2. EVALUATE EXITS IF POSITION IS OPEN ─────────────────────────
            if pos_open:
                is_long = pos_direction.is_long

                # MFE / MAE tracking
                if is_long:
                    fav = bar_high - pos_entry_price
                    adv = pos_entry_price - bar_low
                else:
                    fav = pos_entry_price - bar_low
                    adv = bar_high - pos_entry_price

                mfe_pts = max(mfe_pts, fav)
                mae_pts = max(mae_pts, adv)

                exit_triggered = False
                exit_price = 0.0
                exit_reason = ""
                exit_slip = self.slippage_model.calculate_slippage_points(bar_close, tick_size)

                # Check Physical Delivery Tender Lockout (Individual Contract)
                if contract_mode == ContractMode.INDIVIDUAL_CONTRACT and contract_expiry:
                    if instrument_config.is_in_tender_period(contract_expiry, bar_time.date()):
                        exit_triggered = True
                        exit_price = bar_open - exit_slip if is_long else bar_open + exit_slip
                        exit_reason = "TENDER_PERIOD_LOCKOUT"

                # Check Same-Bar SL & Target Conflict
                if not exit_triggered:
                    sl_hit = (bar_low <= pos_sl) if is_long else (bar_high >= pos_sl)
                    target_hit = (bar_high >= pos_target) if is_long else (bar_low <= pos_target)

                    if sl_hit and target_hit:
                        pos_ambiguous = True
                        if self.same_bar_rule in (SameBarAmbiguityRule.STOP_FIRST, SameBarAmbiguityRule.WORST_CASE):
                            exit_triggered = True
                            exit_price = min(pos_sl, bar_open) - exit_slip if is_long else max(pos_sl, bar_open) + exit_slip
                            exit_reason = "SL_HIT"
                        elif self.same_bar_rule == SameBarAmbiguityRule.TARGET_FIRST:
                            exit_triggered = True
                            exit_price = max(pos_target, bar_open) - exit_slip if is_long else min(pos_target, bar_open) + exit_slip
                            exit_reason = "TARGET_HIT"
                        elif self.same_bar_rule == SameBarAmbiguityRule.SPLIT_50_50:
                            # Deterministic 50/50 using trade index
                            if len(trades) % 2 == 0:
                                exit_triggered = True
                                exit_price = min(pos_sl, bar_open) - exit_slip if is_long else max(pos_sl, bar_open) + exit_slip
                                exit_reason = "SL_HIT"
                            else:
                                exit_triggered = True
                                exit_price = max(pos_target, bar_open) - exit_slip if is_long else min(pos_target, bar_open) + exit_slip
                                exit_reason = "TARGET_HIT"

                    elif sl_hit:
                        exit_triggered = True
                        exit_price = min(pos_sl, bar_open) - exit_slip if is_long else max(pos_sl, bar_open) + exit_slip
                        exit_reason = "SL_HIT"

                    elif target_hit:
                        exit_triggered = True
                        exit_price = max(pos_target, bar_open) - exit_slip if is_long else min(pos_target, bar_open) + exit_slip
                        exit_reason = "TARGET_HIT"

                    # Check EOD Intraday Squareoff
                    elif self.enable_intraday_eod_squareoff and bar_time.time() >= self.eod_time:
                        exit_triggered = True
                        exit_price = bar_close - exit_slip if is_long else bar_close + exit_slip
                        exit_reason = "EOD_SQUAREOFF"

                    # Check Position Management Trailing Stop & Custom Exit Signal
                    else:
                        mock_plan = TradePlan(
                            signal=RawSignal(
                                symbol=instrument_config.symbol,
                                direction=pos_direction,
                                confidence=0.8,
                                votes=1,
                                strategies_fired=[strategy.name]
                            ),
                            entry_price=pos_entry_price,
                            sl_price=pos_sl,
                            target_price=pos_target,
                            desired_lots=lots,
                            lot_size=lot_size,
                            tick_size=tick_size,
                        )
                        mock_pos = Position(plan=mock_plan, entry_premium=pos_entry_price, current_premium=bar_close)
                        mock_pos.entry_bar_index = pos_entry_bar

                        # Trailing stop ratchet
                        trail_update = strategy.manage_position(mock_pos, bar_close, df.iloc[max(0, i - 100):i+1])
                        if trail_update and trail_update.get("action") == "UPDATE_SL":
                            new_sl = trail_update["new_sl"]
                            if is_long and new_sl > pos_sl:
                                pos_sl = new_sl
                            elif (not is_long) and (pos_sl <= 0 or new_sl < pos_sl):
                                pos_sl = new_sl

                        # Strategy Custom Exit Signal
                        strat_exit = strategy.exit_signal(mock_pos, bar_close, df.iloc[max(0, i - 100):i+1])
                        if strat_exit and strat_exit.get("exit"):
                            exit_triggered = True
                            exit_price = bar_close - exit_slip if is_long else bar_close + exit_slip
                            exit_reason = strat_exit.get("reason", "STRATEGY_EXIT")

                # Finalize Exit
                if exit_triggered:
                    exit_price = strategy.round_to_tick(exit_price)
                    pnl_pts = (exit_price - pos_entry_price) if is_long else (pos_entry_price - exit_price)

                    charges = calculate_commodity_trade_charges(
                        entry_price=pos_entry_price,
                        exit_price=exit_price,
                        quantity=qty,
                        direction="BUY" if is_long else "SELL",
                        tick_size=tick_size,
                        tick_value=instrument_config.tick_value,
                    )

                    total_slip_pts = pos_slippage_pts + exit_slip
                    total_slip_inr = total_slip_pts * qty * multiplier
                    hold_bars = i - pos_entry_bar
                    tf_mins = 5
                    if timeframe.endswith("m"):
                        try: tf_mins = int(timeframe[:-1])
                        except: pass
                    elif timeframe.endswith("h"):
                        try: tf_mins = int(timeframe[:-1]) * 60
                        except: pass
                    hold_minutes = float(hold_bars * tf_mins)

                    trade = BacktestTrade(
                        trade_id=f"BT_{len(trades)+1:04d}",
                        strategy=strategy.name,
                        instrument=instrument_config.symbol,
                        contract=pos_contract,
                        direction="BUY" if is_long else "SELL",
                        entry_time=pos_entry_time,
                        entry_price=pos_entry_price,
                        exit_time=bar_time,
                        exit_price=exit_price,
                        stop_loss=pos_sl,
                        target=pos_target,
                        quantity=qty,
                        lots=lots,
                        gross_pnl_inr=charges.gross_pnl_inr,
                        net_pnl_inr=charges.net_pnl_inr,
                        pnl_points=pnl_pts,
                        slippage_pts=total_slip_pts,
                        slippage_cost_inr=total_slip_inr,
                        fees_inr=charges.total_charges,
                        mfe_pts=mfe_pts,
                        mae_pts=mae_pts,
                        hold_bars=hold_bars,
                        hold_minutes=hold_minutes,
                        regime=pos_regime,
                        time_of_day_bucket=self._get_time_of_day_bucket(pos_entry_time.time()),
                        exit_reason=exit_reason,
                        is_ambiguous_bar=pos_ambiguous,
                        is_oos=is_oos,
                    )
                    trades.append(trade)

                    # Update Cumulative Equity
                    cum_gross += trade.gross_pnl_inr
                    cum_fees += trade.fees_inr
                    cum_slippage += trade.slippage_cost_inr
                    cum_net += trade.net_pnl_inr
                    peak_net = max(peak_net, cum_net)
                    dd_inr = peak_net - cum_net
                    dd_pct = (dd_inr / max(config.initial_capital_inr + peak_net, 1.0)) * 100.0

                    equity_curve.append(EquityPoint(
                        timestamp=bar_time,
                        trade_id=trade.trade_id,
                        gross_pnl_inr=trade.gross_pnl_inr,
                        cumulative_gross_pnl=cum_gross,
                        fees_inr=trade.fees_inr,
                        cumulative_fees=cum_fees,
                        slippage_cost_inr=trade.slippage_cost_inr,
                        cumulative_slippage=cum_slippage,
                        net_pnl_inr=trade.net_pnl_inr,
                        cumulative_net_pnl=cum_net,
                        drawdown_inr=dd_inr,
                        drawdown_pct=dd_pct,
                    ))

                    # Reset position
                    pos_open = False
                    pos_direction = None

            # ── 3. EVALUATE STRATEGY ON CANDLE I (ENTRY ON CANDLE I+1) ────────
            if not pos_open and pending_signal is None:
                # Tender period block: do not generate signals during tender period
                if contract_mode == ContractMode.INDIVIDUAL_CONTRACT and contract_expiry:
                    if instrument_config.is_in_tender_period(contract_expiry, bar_time.date()):
                        continue

                # Provide bounded lookback window (up to 200 bars) for O(1) per-bar evaluation
                history_df = df.iloc[max(0, i - 200):i+1]
                regime_details = self.regime_engine.classify(history_df)

                sig = strategy.generate_signal(history_df, regime_details=regime_details)
                sig = strategy.validate_signal(sig, regime_details=regime_details)

                if sig.is_valid and sig.direction != Direction.NONE and sig.decision == "TRADE":
                    pending_signal = sig

        # Calculate Performance Metrics & Breakdowns
        metrics = self._calculate_metrics(trades, equity_curve, config.initial_capital_inr)
        regime_breakdown = self._breakdown_by_key(trades, lambda t: t.regime)
        time_breakdown = self._breakdown_by_key(trades, lambda t: t.time_of_day_bucket)
        direction_breakdown = self._breakdown_by_key(trades, lambda t: t.direction)

        return BacktestResult(
            config=config,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            regime_breakdown=regime_breakdown,
            time_of_day_breakdown=time_breakdown,
            direction_breakdown=direction_breakdown,
        )

    def _calculate_metrics(
        self,
        trades: List[BacktestTrade],
        equity: List[EquityPoint],
        initial_capital: float,
    ) -> PerformanceMetrics:
        """Computes comprehensive risk and return analytics."""
        if not trades:
            return PerformanceMetrics()

        n = len(trades)
        longs = [t for t in trades if t.direction == "BUY"]
        shorts = [t for t in trades if t.direction == "SELL"]
        wins = [t for t in trades if t.net_pnl_inr > 0]
        losses = [t for t in trades if t.net_pnl_inr < 0]
        breakevens = [t for t in trades if t.net_pnl_inr == 0]

        win_rate = (len(wins) / n) * 100.0
        gross_pnl = sum(t.gross_pnl_inr for t in trades)
        fees = sum(t.fees_inr for t in trades)
        slippage = sum(t.slippage_cost_inr for t in trades)
        net_pnl = sum(t.net_pnl_inr for t in trades)

        gross_wins = sum(t.gross_pnl_inr for t in wins)
        gross_losses = abs(sum(t.gross_pnl_inr for t in losses))
        profit_factor = round(gross_wins / max(gross_losses, 1e-6), 2) if gross_losses > 0 else 999.0

        pnl_pts = [t.pnl_points for t in trades]
        net_inrs = [t.net_pnl_inr for t in trades]
        expectancy_pts = round(float(np.mean(pnl_pts)), 2)
        expectancy_inr = round(float(np.mean(net_inrs)), 2)

        avg_win = float(np.mean([t.net_pnl_inr for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t.net_pnl_inr for t in losses])) if losses else 0.0
        largest_win = max([t.net_pnl_inr for t in trades], default=0.0)
        largest_loss = min([t.net_pnl_inr for t in trades], default=0.0)
        win_loss_ratio = round(abs(avg_win / max(abs(avg_loss), 1e-6)), 2) if avg_loss != 0 else 0.0

        holding_times = [t.hold_minutes for t in trades]
        avg_holding = float(np.mean(holding_times)) if holding_times else 0.0

        # Drawdown
        max_dd_inr = max([e.drawdown_inr for e in equity], default=0.0)
        max_dd_pct = max([e.drawdown_pct for e in equity], default=0.0)

        # Consecutive Wins / Losses
        max_consec_w = 0
        max_consec_l = 0
        curr_w = 0
        curr_l = 0
        for t in trades:
            if t.net_pnl_inr > 0:
                curr_w += 1
                curr_l = 0
                max_consec_w = max(max_consec_w, curr_w)
            elif t.net_pnl_inr < 0:
                curr_l += 1
                curr_w = 0
                max_consec_l = max(max_consec_l, curr_l)
            else:
                curr_w = 0
                curr_l = 0

        # Sharpe & Sortino (per-trade)
        std_pnl = float(np.std(net_inrs)) if len(net_inrs) > 1 else 0.0
        downside_pnls = [x for x in net_inrs if x < 0]
        downside_std = float(np.std(downside_pnls)) if len(downside_pnls) > 1 else 0.0

        sharpe = round(float(expectancy_inr / max(std_pnl, 1e-6)) * np.sqrt(252), 2) if std_pnl > 0 else 0.0
        sortino = round(float(expectancy_inr / max(downside_std, 1e-6)) * np.sqrt(252), 2) if downside_std > 0 else 0.0
        calmar = round(net_pnl / max(max_dd_inr, 1e-6), 2) if max_dd_inr > 0 else 0.0

        # MFE / MAE
        mfe_vals = [t.mfe_pts for t in trades]
        mae_vals = [t.mae_pts for t in trades]
        avg_mfe = round(float(np.mean(mfe_vals)), 2) if mfe_vals else 0.0
        avg_mae = round(float(np.mean(mae_vals)), 2) if mae_vals else 0.0
        mfe_mae_r = round(avg_mfe / max(avg_mae, 1e-6), 2)

        return PerformanceMetrics(
            total_trades=n,
            long_trades=len(longs),
            short_trades=len(shorts),
            winning_trades=len(wins),
            losing_trades=len(losses),
            breakeven_trades=len(breakevens),
            win_rate_pct=round(win_rate, 2),
            profit_factor=profit_factor,
            expectancy_points=expectancy_pts,
            expectancy_inr=expectancy_inr,
            avg_win_inr=round(avg_win, 2),
            avg_loss_inr=round(avg_loss, 2),
            largest_win_inr=round(largest_win, 2),
            largest_loss_inr=round(largest_loss, 2),
            win_loss_ratio=win_loss_ratio,
            avg_holding_time_minutes=round(avg_holding, 1),
            gross_pnl_inr=round(gross_pnl, 2),
            total_costs_inr=round(fees, 2),
            total_slippage_inr=round(slippage, 2),
            net_pnl_inr=round(net_pnl, 2),
            max_drawdown_inr=round(max_dd_inr, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_consecutive_wins=max_consec_w,
            max_consecutive_losses=max_consec_l,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_mfe_points=avg_mfe,
            avg_mae_points=avg_mae,
            mfe_mae_ratio=mfe_mae_r,
            sl_hit_count=sum(1 for t in trades if t.exit_reason == "SL_HIT"),
            target_hit_count=sum(1 for t in trades if t.exit_reason == "TARGET_HIT"),
            strategy_exit_count=sum(1 for t in trades if t.exit_reason == "STRATEGY_EXIT"),
            eod_exit_count=sum(1 for t in trades if t.exit_reason == "EOD_SQUAREOFF"),
            ambiguous_bar_count=sum(1 for t in trades if t.is_ambiguous_bar),
        )

    def _breakdown_by_key(self, trades: List[BacktestTrade], key_fn) -> Dict[str, Dict[str, Any]]:
        """Groups trades by key and computes sub-metrics."""
        groups: Dict[str, List[BacktestTrade]] = {}
        for t in trades:
            k = str(key_fn(t))
            groups.setdefault(k, []).append(t)

        out = {}
        for k, sub_trades in groups.items():
            wins = [t for t in sub_trades if t.net_pnl_inr > 0]
            losses = [t for t in sub_trades if t.net_pnl_inr < 0]
            gw = sum(t.gross_pnl_inr for t in wins)
            gl = abs(sum(t.gross_pnl_inr for t in losses))
            pf = round(gw / max(gl, 1e-6), 2) if gl > 0 else (999.0 if gw > 0 else 0.0)
            net_pnl = sum(t.net_pnl_inr for t in sub_trades)
            wr = round((len(wins) / len(sub_trades)) * 100.0, 1)

            out[k] = {
                "trades": len(sub_trades),
                "win_rate_pct": wr,
                "net_pnl_inr": round(net_pnl, 2),
                "profit_factor": pf,
                "expectancy_inr": round(float(np.mean([t.net_pnl_inr for t in sub_trades])), 2),
            }
        return out

    # ── OUT-OF-SAMPLE (OOS) PARTITIONING ───────────────────────────────────────
    def run_oos(
        self,
        strategy: BaseCommodityStrategy,
        instrument_config: InstrumentConfig,
        df: pd.DataFrame,
        timeframe: str = "5m",
        split_pct: float = 0.30, # 30% out-of-sample
        lots: int = 1,
    ) -> Tuple[BacktestResult, BacktestResult]:
        """
        Splits data chronologically into in-sample (train) and out-of-sample (test).
        Returns (in_sample_result, out_of_sample_result).
        """
        split_idx = int(len(df) * (1.0 - split_pct))
        df_is = df.iloc[:split_idx]
        df_oos = df.iloc[split_idx:]

        res_is = self.run(strategy, instrument_config, df_is, timeframe=timeframe, lots=lots, is_oos=False)
        res_oos = self.run(strategy, instrument_config, df_oos, timeframe=timeframe, lots=lots, is_oos=True)
        res_is.oos_metrics = res_oos.metrics
        return res_is, res_oos

    # ── WALK-FORWARD ANALYSIS ──────────────────────────────────────────────────
    def run_walk_forward(
        self,
        strategy: BaseCommodityStrategy,
        instrument_config: InstrumentConfig,
        df: pd.DataFrame,
        timeframe: str = "5m",
        n_windows: int = 3,
        train_ratio: float = 0.70,
        lots: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Runs walk-forward rolling window analysis.
        Each window trains on train slice and validates on out-of-sample test slice.
        """
        window_size = len(df) // n_windows
        results = []

        for w in range(n_windows):
            start = w * window_size
            end = min(start + window_size, len(df))
            sub_df = df.iloc[start:end]

            if len(sub_df) < 30:
                continue

            split_idx = int(len(sub_df) * train_ratio)
            train_df = sub_df.iloc[:split_idx]
            test_df = sub_df.iloc[split_idx:]

            res_train = self.run(strategy, instrument_config, train_df, timeframe=timeframe, lots=lots, is_oos=False)
            res_test = self.run(strategy, instrument_config, test_df, timeframe=timeframe, lots=lots, is_oos=True)

            results.append({
                "window": w + 1,
                "train_start": str(train_df.index[0]),
                "train_end": str(train_df.index[-1]),
                "test_start": str(test_df.index[0]),
                "test_end": str(test_df.index[-1]),
                "train_net_pnl": res_train.metrics.net_pnl_inr,
                "train_profit_factor": res_train.metrics.profit_factor,
                "test_net_pnl": res_test.metrics.net_pnl_inr,
                "test_profit_factor": res_test.metrics.profit_factor,
                "test_trades": res_test.metrics.total_trades,
                "test_win_rate": res_test.metrics.win_rate_pct,
            })
        return results

    # ── SLIPPAGE SENSITIVITY ───────────────────────────────────────────────────
    def run_slippage_sensitivity(
        self,
        strategy: BaseCommodityStrategy,
        instrument_config: InstrumentConfig,
        df: pd.DataFrame,
        ticks_list: List[int] = [0, 1, 2, 3, 5],
        timeframe: str = "5m",
        lots: int = 1,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Evaluates strategy resilience under multiple slippage stress levels.
        """
        results = {}
        for ticks in ticks_list:
            engine = CommodityBacktestEngine(
                slippage_model=SlippageModel(SlippageType.FIXED_TICKS, float(ticks)),
                same_bar_rule=self.same_bar_rule,
            )
            res = engine.run(strategy, instrument_config, df, timeframe=timeframe, lots=lots)
            results[ticks] = {
                "slippage_ticks": ticks,
                "slippage_points": ticks * instrument_config.tick_size,
                "net_pnl_inr": res.metrics.net_pnl_inr,
                "profit_factor": res.metrics.profit_factor,
                "expectancy_inr": res.metrics.expectancy_inr,
                "max_drawdown_inr": res.metrics.max_drawdown_inr,
                "total_trades": res.metrics.total_trades,
            }
        return results


# ── COMMAND-LINE INTERFACE ─────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="MCXForge Institutional Commodity Backtest Runner")
    parser.add_argument("--instrument", type=str, default="SILVERMIC", help="Commodity symbol (SILVERMIC, GOLDM, CRUDEOILM, NATGASM)")
    parser.add_argument("--strategy", type=str, default="TrendFollowing", help="Strategy name or ALL")
    parser.add_argument("--timeframe", type=str, default="5m", help="Candle timeframe (1m, 3m, 5m, 15m, 30m, 1h)")
    parser.add_argument("--data", type=str, required=False, default="data/examples/silvermic_sample.csv", help="Path to OHLCV CSV file")
    parser.add_argument("--slippage-ticks", type=int, default=1, help="Slippage in ticks")
    parser.add_argument("--same-bar-rule", type=str, default="STOP_FIRST", choices=["STOP_FIRST", "TARGET_FIRST", "SPLIT_50_50"], help="Ambiguity resolution")
    parser.add_argument("--oos-split", type=float, default=None, help="Optional out-of-sample test split ratio (e.g. 0.30)")
    parser.add_argument("--output-dir", type=str, default="backtests", help="Output directory to save backtest artifacts")
    parser.add_argument("--sensitivity", action="store_true", help="Run slippage sensitivity sweep (0, 1, 2, 3, 5 ticks)")
    return parser.parse_args()


def main():
    args = parse_args()
    inst_cfg = get_instrument_config(args.instrument)
    raw_df = HistoricalDataLoader.load(args.data)

    # Resample if needed
    if args.timeframe != "5m":
        df = resample_ohlcv(raw_df, timeframe=args.timeframe)
    else:
        df = raw_df

    slip_model = SlippageModel(SlippageType.FIXED_TICKS, float(args.slippage_ticks))
    ambiguity_rule = SameBarAmbiguityRule(args.same_bar_rule)
    engine = CommodityBacktestEngine(slippage_model=slip_model, same_bar_rule=ambiguity_rule)

    strategy_names = [
        "TrendFollowing",
        "OpeningRangeBreakout",
        "VWAPMeanReversion",
        "VolatilityBreakout",
        "DonchianBreakout",
    ] if args.strategy.upper() == "ALL" else [args.strategy]

    print(f"\n{'='*70}")
    print(f"MCXForge Backtest Suite — {inst_cfg.symbol} ({args.timeframe})")
    print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"Slippage: {args.slippage_ticks} ticks ({args.slippage_ticks * inst_cfg.tick_size} pts) | Same-Bar Rule: {ambiguity_rule.value}")
    print(f"{'='*70}\n")

    for strat_name in strategy_names:
        strat = StrategyRegistry.create_strategy(strat_name, inst_cfg)

        if args.sensitivity:
            print(f"\n--- Slippage Sensitivity Sweep: {strat_name} on {inst_cfg.symbol} ---")
            sweep = engine.run_slippage_sensitivity(strat, inst_cfg, df, timeframe=args.timeframe)
            df_sweep = pd.DataFrame.from_dict(sweep, orient="index")
            print(df_sweep.to_string())
            continue

        if args.oos_split and args.oos_split > 0:
            res_is, res_oos = engine.run_oos(strat, inst_cfg, df, timeframe=args.timeframe, split_pct=args.oos_split)
            path = res_is.save_to_dir(args.output_dir)
            print(f"[{strat_name}] Train Net P&L: ₹{res_is.metrics.net_pnl_inr:,.2f} (PF: {res_is.metrics.profit_factor}) | "
                  f"OOS Net P&L: ₹{res_oos.metrics.net_pnl_inr:,.2f} (PF: {res_oos.metrics.profit_factor}) | Saved to {path}")
        else:
            res = engine.run(strat, inst_cfg, df, timeframe=args.timeframe)
            path = res.save_to_dir(args.output_dir)
            m = res.metrics
            print(f"[{strat_name}] Trades: {m.total_trades} | Win Rate: {m.win_rate_pct}% | "
                  f"Gross P&L: ₹{m.gross_pnl_inr:,.2f} | Costs: ₹{m.total_costs_inr:,.2f} | "
                  f"Net P&L: ₹{m.net_pnl_inr:,.2f} | PF: {m.profit_factor} | Max DD: ₹{m.max_drawdown_inr:,.2f} ({m.max_drawdown_pct}%) | "
                  f"Saved to: {path}")


if __name__ == "__main__":
    main()

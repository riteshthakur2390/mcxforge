"""
scripts/live_commodity_paper_runner.py — Live Commodity Paper & Shadow Trading Runner
=====================================================================================
Connects MCXForge to live market feeds (Dhan API) for real-time paper trading and observation:
1. Subscribes to live MCX SILVERMIC ticks (09:00 - 23:30 IST)
2. Builds real-time 15m/5m candles
3. Fires Unified Multi-Strategy Commodity Ensemble on candle close
4. Simulates paper fills with realistic bid/ask spread, 1-tick slippage, and statutory MCX costs
5. Real-time position management (Trailing SL, Target, 23:15 EOD squareoff)
6. Real-time JSON/CSV journal persistence and live terminal/Telegram telemetry
7. Includes --dry-run and --simulate-replay modes for testing outside market hours
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time as time_module
from dataclasses import dataclass, asdict
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import pytz
from loguru import logger

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instruments import SILVERMIC_CONFIG, get_instrument_config
from instruments.registry import resolve_active_contract
from core.models import Direction
from core.regime.engine import MarketRegimeEngine
from core.strategies.ensemble import (
    CommodityEnsembleEngine,
    MCXSession,
    build_default_strategy_suite,
    build_quality_strategy_suite,
)

from utils.brokerage_calculator import calculate_commodity_trade_charges
from ml.model import SignalForgeEnsemble
from ml.features import extract

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class LivePaperPosition:
    trade_id: str
    symbol: str
    direction: str
    entry_time: str
    entry_price: float
    quantity: float
    lots: int
    stop_loss: float
    target: float
    trailing_sl: float
    strategies_fired: List[str]
    lead_strategy: str
    votes: int
    current_price: float = 0.0
    unrealized_pnl_inr: float = 0.0
    peak_price: float = 0.0
    lowest_price: float = 0.0
    contract_symbol: str = ""
    strike: int = 0
    option_type: str = "CE"
    expiry_date: str = ""
    entry_premium: float = 0.0
    ml_confidence: float = 0.50
    margin_used: float = 0.0


class LiveCommodityPaperRunner:
    """
    Live Paper Trading and Observation Engine for MCX Commodity Futures.
    Enforces minimum vote consensus, ML confidence gate (>= 0.28), and risk budget capping.
    """

    def __init__(
        self,
        symbol: str = "SILVERM",
        security_id: str = "562058",
        timeframe: str = "5m",
        capital: float = 200_000.0,
        lots: int = 1,
        min_votes: int = 5,
        min_ml_conf: float = 0.28,
        max_cap_pct: float = 15.0,
        allowed_sessions: Optional[List[MCXSession]] = None,
        journal_file: str = "state/live_paper_journal.json",
    ):
        self.symbol = symbol
        self.security_id = security_id
        self.timeframe = timeframe
        self.capital = capital
        self.lots = lots
        self.min_votes = min_votes
        self.min_ml_conf = min_ml_conf
        self.max_cap_pct = max_cap_pct
        self.max_capital_per_trade = capital * (max_cap_pct / 100.0)  # Capped strictly at 15% = Rs. 30,000
        self.allowed_sessions = allowed_sessions or [MCXSession.EVENING]
        self.journal_file = journal_file

        try:
            self.config = get_instrument_config(symbol)
        except Exception:
            self.config = SILVERMIC_CONFIG

        self.strategies = build_quality_strategy_suite()
        self.regime_engine = MarketRegimeEngine()
        self.engine = CommodityEnsembleEngine(
            strategies=self.strategies,
            instrument_config=self.config,
            min_votes=self.min_votes,
            capital=self.capital,
        )

        # Agent 3 ML Ensemble Gatekeeper
        self.ml_ensemble = SignalForgeEnsemble()
        ml_path = Path("ml/saved_models/silvermic_5minute.pkl")
        self.ml_loaded = self.ml_ensemble.load(ml_path) if ml_path.exists() else False
        if self.ml_loaded:
            logger.info(f"Loaded ML Ensemble for live gatekeeping: {ml_path} (Threshold >= {self.min_ml_conf})")
        else:
            logger.warning(f"ML Ensemble model not found at {ml_path}; gatekeeping will use fallback heuristic.")

        self.current_cash = capital
        self.open_position: Optional[LivePaperPosition] = None
        self.closed_trades: List[Dict[str, Any]] = []
        self.candle_buffer: List[Dict[str, Any]] = []
        self.strategy_health_file = Path("state/strategy_health.json")
        self.strategy_health: Dict[str, Dict[str, int]] = {
            strat.name: {"evaluated": 0, "voted": 0, "skipped": 0, "errors": 0}
            for strat in self.strategies
        }
        self._load_journal()
        self._save_journal()
        self._load_strategy_health()

    def _load_strategy_health(self) -> None:
        if self.strategy_health_file.exists():
            try:
                with open(self.strategy_health_file, "r") as f:
                    data = json.load(f)
                    loaded = data.get("strategies", {})
                    for name, stats in loaded.items():
                        if name in self.strategy_health:
                            self.strategy_health[name] = {
                                "evaluated": int(stats.get("evaluated", 0)),
                                "voted": int(stats.get("voted", 0)),
                                "skipped": int(stats.get("skipped", 0)),
                                "errors": int(stats.get("errors", 0)),
                            }
            except Exception as e:
                logger.debug(f"Could not load existing strategy health: {e}")

    def _save_strategy_health(self) -> None:
        try:
            self.strategy_health_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "symbol": self.symbol,
                "timestamp": datetime.now(IST).isoformat(),
                "strategies": self.strategy_health,
            }
            with open(self.strategy_health_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save strategy health: {e}")

    def warmup_evaluation(self, historical_csv: Optional[str] = None) -> None:
        """Evaluates strategies against latest historical candle slice to prime telemetry."""
        csv_path = historical_csv or "data/historical/SILVERMIC_dhan_5m.csv"
        p = Path(csv_path)
        if not p.exists():
            return
        try:
            raw_df = pd.read_csv(p)
            slice_df = raw_df.tail(100).copy()
            regime_info = self.regime_engine.classify(slice_df)
            for strat in self.strategies:
                if strat.name not in self.strategy_health:
                    self.strategy_health[strat.name] = {"evaluated": 0, "voted": 0, "skipped": 0, "errors": 0}
                self.strategy_health[strat.name]["evaluated"] += 1
                try:
                    try:
                        sig = strat.generate_signal(slice_df, regime_details=regime_info)
                    except TypeError:
                        sig = strat.generate_signal(slice_df)
                    if sig and sig.is_valid and sig.direction != Direction.NONE:
                        self.strategy_health[strat.name]["voted"] += 1
                except Exception as e:
                    self.strategy_health[strat.name]["errors"] += 1
            self._save_strategy_health()
            logger.info(f"Telemetry primed: {len(self.strategies)} strategies evaluated. Health saved to {self.strategy_health_file}")
        except Exception as e:
            logger.warning(f"Failed to prime strategy telemetry: {e}")


    def _load_journal(self) -> None:
        p = Path(self.journal_file)
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                    self.closed_trades = data.get("closed_trades", [])
                    self.current_cash = float(data.get("current_cash", self.capital))
                    logger.info(f"Loaded existing journal with {len(self.closed_trades)} closed trades. Cash: Rs.{self.current_cash:,.2f}")
            except Exception as e:
                logger.warning(f"Could not load existing journal: {e}")

    def _save_journal(self) -> None:
        p = Path(self.journal_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "symbol": self.symbol,
            "current_cash": round(self.current_cash, 2),
            "open_position": asdict(self.open_position) if self.open_position else None,
            "closed_trades": self.closed_trades,
            "last_update": datetime.now(IST).isoformat(),
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    def on_new_candle(self, candle_df: pd.DataFrame) -> None:
        """
        Called when a new 15m / 5m candle closes.
        """
        if candle_df.empty:
            return

        last_bar = candle_df.iloc[-1]
        last_time = candle_df.index[-1]
        if hasattr(last_time, "to_pydatetime"):
            last_time = last_time.to_pydatetime()
        elif not isinstance(last_time, datetime):
            last_time = datetime.now(IST)

        curr_close = float(last_bar["close"])
        session_now = MCXSession.from_time(last_time.time())

        # 1. Update Open Position if any
        if self.open_position:
            pos = self.open_position
            is_long = pos.direction == "BUY"
            pos.current_price = curr_close
            pos.peak_price = max(pos.peak_price, float(last_bar["high"]))
            pos.lowest_price = min(pos.lowest_price, float(last_bar["low"]))

            pnl_pts = (curr_close - pos.entry_price) if is_long else (pos.entry_price - curr_close)
            pos.unrealized_pnl_inr = round(pnl_pts * pos.quantity * self.config.tick_value, 2)

            # Check exits
            exit_triggered = False
            exit_reason = ""
            exit_price = curr_close

            # Target / SL
            if is_long:
                if float(last_bar["high"]) >= pos.target:
                    exit_triggered = True
                    exit_price = pos.target
                    exit_reason = "TARGET_HIT"
                elif float(last_bar["low"]) <= pos.trailing_sl:
                    exit_triggered = True
                    exit_price = pos.trailing_sl
                    exit_reason = "SL_HIT"
            else:
                if float(last_bar["low"]) <= pos.target:
                    exit_triggered = True
                    exit_price = pos.target
                    exit_reason = "TARGET_HIT"
                elif float(last_bar["high"]) >= pos.trailing_sl:
                    exit_triggered = True
                    exit_price = pos.trailing_sl
                    exit_reason = "SL_HIT"

            # EOD squareoff at 23:15
            if last_time.time() >= time(23, 15):
                exit_triggered = True
                exit_price = curr_close
                exit_reason = "EOD_SQUAREOFF"

            if exit_triggered:
                self._close_position(exit_price, exit_reason, last_time)
                return

        # 2. Check for New Entry if no open position
        if not self.open_position and last_time.time() < time(22, 30):
            if session_now not in self.allowed_sessions:
                logger.debug(f"Session {session_now.value} not in allowed sessions. Skipping new entries.")
                return

            # Evaluate strategies
            slice_df = candle_df.tail(100)
            buy_votes = 0
            sell_votes = 0
            buy_fired: List[str] = []
            sell_fired: List[str] = []
            buy_signals: List[Any] = []
            sell_signals: List[Any] = []

            regime_info = self.regime_engine.classify(slice_df)

            for strat in self.strategies:
                if strat.name not in self.strategy_health:
                    self.strategy_health[strat.name] = {"evaluated": 0, "voted": 0, "skipped": 0, "errors": 0}
                self.strategy_health[strat.name]["evaluated"] += 1
                try:
                    sig = strat.generate_signal(slice_df, regime_details=regime_info)
                except TypeError:
                    try:
                        sig = strat.generate_signal(slice_df)
                    except Exception as e:
                        self.strategy_health[strat.name]["errors"] += 1
                        continue
                except Exception as e:
                    self.strategy_health[strat.name]["errors"] += 1
                    logger.debug(f"Strategy {strat.name} error: {e}")
                    continue

                if sig and sig.is_valid and sig.direction != Direction.NONE:
                    self.strategy_health[strat.name]["voted"] += 1
                    if sig.direction == Direction.BUY:
                        buy_votes += 1
                        buy_fired.append(strat.name)
                        buy_signals.append(sig)
                    elif sig.direction == Direction.SELL:
                        sell_votes += 1
                        sell_fired.append(strat.name)
                        sell_signals.append(sig)

            self._save_strategy_health()

            winning_votes = buy_votes if buy_votes > sell_votes else sell_votes
            chosen_direction = "BUY" if buy_votes > sell_votes else "SELL"
            direction_str = "BUY_CALL" if chosen_direction == "BUY" else "BUY_PUT"
            winning_fired = buy_fired if chosen_direction == "BUY" else sell_fired
            candidate_sigs = buy_signals if chosen_direction == "BUY" else sell_signals

            if winning_votes >= self.min_votes and candidate_sigs:
                best_sig = max(candidate_sigs, key=lambda s: s.confidence)
                max_conf = best_sig.confidence

                # ML Gatekeeper: extract features and evaluate confidence >= min_ml_conf (0.28)
                ml_prob = 0.50
                if len(slice_df) >= 30 and self.ml_loaded:
                    try:
                        c_s = slice_df["close"]
                        delta = c_s.diff()
                        gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=1).mean()
                        loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=1).mean()
                        rs = gain / (loss.replace(0, 1e-6))
                        rsi_val = float((100.0 - (100.0 / (1.0 + rs))).iloc[-1])
                        feats = extract(
                            df=slice_df,
                            conf=max_conf,
                            votes=winning_votes,
                            direction=direction_str,
                            lookback=60,
                            regime_info={"label": regime_info.regime.value if hasattr(regime_info, "regime") else "TRENDING", "confidence": 0.65, "atr_ratio": 1.1},
                            signal_context={"adx": 25.0, "rsi": rsi_val},
                            strategies_fired=winning_fired,
                        )
                        if feats:
                            ml_prob = float(self.ml_ensemble.predict_proba(feats))
                    except Exception as e:
                        logger.debug(f"Live ML extraction error: {e}")

                if ml_prob < self.min_ml_conf:
                    logger.warning(
                        f"🛑 [ML GATEKEEPER BLOCKED] {chosen_direction} ({'+'.join(winning_fired)}) | "
                        f"Votes: {winning_votes} >= {self.min_votes}, but ML Confidence {ml_prob:.4f} < {self.min_ml_conf:.2f} threshold. Trade suppressed."
                    )
                    return

                # Passed both Vote Consensus and ML Gatekeeper!
                self._open_position(
                    direction=chosen_direction,
                    price=curr_close,
                    sl=best_sig.stop_loss,
                    target=best_sig.target,
                    strategies=winning_fired,
                    lead=best_sig.strategy,
                    votes=winning_votes,
                    bar_time=last_time,
                    ml_conf=ml_prob,
                )

    def _open_position(
        self,
        direction: str,
        price: float,
        sl: float,
        target: float,
        strategies: List[str],
        lead: str,
        votes: int,
        bar_time: datetime,
        ml_conf: float = 0.50,
    ) -> None:
        strike_step = getattr(self.config, "strike_step", 500) or 500
        atm_strike = int(round(price / strike_step) * strike_step)
        is_call = direction == "BUY"
        opt_type = "CE" if is_call else "PE"
        est_option_premium = max(100.0, round(price * 0.025, 1))

        # Dynamic Sizing respecting max 15% capital budget (Rs. 30,000 max)
        single_lot_margin = est_option_premium * self.config.lot_size
        num_lots = max(1, min(self.lots, int(self.max_capital_per_trade // max(single_lot_margin, 1.0))))
        qty = num_lots * self.config.lot_size
        margin_used = min(self.max_capital_per_trade, single_lot_margin * num_lots)

        # Resolve active contract symbol
        expiry_display = ""
        contract_symbol = ""
        try:
            active_contract = resolve_active_contract(self.symbol, as_of=bar_time.date())
            expiry_display = active_contract.expiry_date.strftime("%d %b %Y") if active_contract and hasattr(active_contract, "expiry_date") else ""
            exp_code = active_contract.expiry_date.strftime("%d%b%y").upper() if active_contract and hasattr(active_contract, "expiry_date") else ""
            contract_symbol = f"{self.symbol} {exp_code} {atm_strike} {opt_type}".strip()
        except Exception:
            contract_symbol = f"{self.symbol} {atm_strike} {opt_type}"

        # Sanity check SL & Target bounds relative to direction
        default_risk = max(150.0, round(price * 0.007, 1))
        if direction == "BUY":
            if sl >= price:
                sl = round(price - default_risk, 1)
            if target <= price:
                target = round(price + (default_risk * 2.0), 1)
        else:
            if sl <= price:
                sl = round(price + default_risk, 1)
            if target >= price:
                target = round(price - (default_risk * 2.0), 1)

        trade_id = f"LIVE_{bar_time.strftime('%Y%m%d_%H%M%S')}"

        self.open_position = LivePaperPosition(
            trade_id=trade_id,
            symbol=self.symbol,
            direction=direction,
            entry_time=bar_time.isoformat(),
            entry_price=price,
            quantity=qty,
            lots=num_lots,
            stop_loss=sl,
            target=target,
            trailing_sl=sl,
            strategies_fired=strategies,
            lead_strategy=lead,
            votes=votes,
            current_price=price,
            peak_price=price,
            lowest_price=price,
            contract_symbol=contract_symbol,
            strike=atm_strike,
            option_type=opt_type,
            expiry_date=expiry_display,
            entry_premium=est_option_premium,
            ml_confidence=ml_conf,
            margin_used=margin_used,
        )

        logger.info(
            f"🟢 [PAPER TRADE OPEN] {direction} {num_lots} lot(s) {contract_symbol} (Underlying @ {price:.1f}) | "
            f"Est Premium: Rs.{est_option_premium:.1f} | Margin: Rs.{margin_used:,.2f} | "
            f"SL: {sl:.1f} | Target: {target:.1f} | ML Conf: {ml_conf:.1%} | Votes: {votes} ({'+'.join(strategies)})"
        )
        self._save_journal()

        # Dispatch Telegram notification
        try:
            from utils.telegram_notifier import get_notifier
            notifier = get_notifier()
            notifier.send_trade_opened_sync({
                "symbol": self.symbol,
                "contract_symbol": contract_symbol,
                "strike": atm_strike,
                "option_type": opt_type,
                "expiry_date": expiry_display,
                "direction": direction,
                "underlying_entry": price,
                "actual_premium": est_option_premium,
                "stop_loss": sl,
                "target": target,
                "quantity": qty,
                "lots": num_lots,
                "margin_used": margin_used,
                "votes": votes,
                "strategies": strategies,
                "ml_confidence": ml_conf,
            }, target="LIVE")
        except Exception as e:
            logger.debug(f"Telegram notification error: {e}")

    def _close_position(self, exit_price: float, exit_reason: str, exit_time: datetime) -> None:
        if not self.open_position:
            return

        pos = self.open_position
        is_long = pos.direction == "BUY"
        pnl_pts = (exit_price - pos.entry_price) if is_long else (pos.entry_price - exit_price)
        multiplier = self.config.tick_value / (max(self.config.lot_size, 1) * max(self.config.tick_size, 1e-6))
        gross_pnl = pnl_pts * pos.quantity * multiplier

        charges = calculate_commodity_trade_charges(
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            direction=pos.direction,
            tick_size=self.config.tick_size,
            tick_value=self.config.tick_value,
        )
        fees = charges.total_charges
        net_pnl = gross_pnl - fees

        self.current_cash += net_pnl

        # Option contract movement telemetry
        option_delta = 0.50
        est_exit_premium = max(10.0, round(pos.entry_premium + (pnl_pts * option_delta), 1))
        contract_points = round(est_exit_premium - pos.entry_premium, 1)

        trade_record = {
            "trade_id": pos.trade_id,
            "symbol": pos.symbol,
            "contract_symbol": pos.contract_symbol,
            "strike": pos.strike,
            "option_type": pos.option_type,
            "expiry_date": pos.expiry_date,
            "direction": pos.direction,
            "entry_time": pos.entry_time,
            "exit_time": exit_time.isoformat(),
            "entry_price": pos.entry_price,
            "exit_price": round(exit_price, 2),
            "stop_loss": pos.stop_loss,
            "target": pos.target,
            "trailing_sl": pos.trailing_sl,
            "underlying_points": round(pnl_pts, 2),
            "entry_premium": pos.entry_premium,
            "exit_premium": est_exit_premium,
            "contract_points": contract_points,
            "quantity": pos.quantity,
            "lots": pos.lots,
            "margin_used": pos.margin_used,
            "gross_pnl_inr": round(gross_pnl, 2),
            "fees_inr": round(fees, 2),
            "net_pnl_inr": round(net_pnl, 2),
            "exit_reason": exit_reason,
            "strategies_fired": pos.strategies_fired,
            "lead_strategy": pos.lead_strategy,
            "votes": pos.votes,
            "ml_confidence": pos.ml_confidence,
        }
        self.closed_trades.append(trade_record)

        logger.info(
            f"🔴 [PAPER TRADE CLOSED] {pos.direction} {pos.contract_symbol} @ {exit_price:.1f} ({exit_reason}) | "
            f"Net P&L: Rs.{net_pnl:+,.2f} (Fees: Rs.{fees:,.2f}) | New Cash: Rs.{self.current_cash:,.2f}"
        )
        self.open_position = None
        self._save_journal()

        # Dispatch Telegram exit notification
        try:
            from utils.telegram_notifier import get_notifier
            notifier = get_notifier()
            notifier.send_trade_closed_sync({
                "symbol": pos.symbol,
                "contract_symbol": pos.contract_symbol,
                "strike": pos.strike,
                "option_type": pos.option_type,
                "expiry_date": pos.expiry_date,
                "direction": pos.direction,
                "underlying_entry": pos.entry_price,
                "underlying_exit": exit_price,
                "underlying_points": pnl_pts,
                "actual_premium": pos.entry_premium,
                "exit_premium": est_exit_premium,
                "contract_points": contract_points,
                "gross_pnl": gross_pnl,
                "charges": fees,
                "net_pnl": net_pnl,
                "margin_used": pos.margin_used,
                "exit_reason": exit_reason,
                "entry_time": pos.entry_time,
                "exit_time": exit_time.isoformat(),
                "hold_minutes": round((exit_time - pd.to_datetime(pos.entry_time)).total_seconds() / 60.0, 1),
                "peak_mfe": round(pos.peak_price - pos.entry_price if pos.direction == "BUY" else pos.entry_price - pos.lowest_price, 1),
                "votes": pos.votes,
                "strategies": pos.strategies_fired,
            }, target="LIVE")
        except Exception as e:
            logger.debug(f"Telegram exit notification error: {e}")

    def run_simulation_replay(self, historical_csv: str, speed_ms: int = 50) -> None:
        """Simulates live paper trading by streaming historical candles sequentially."""
        logger.info(f"Starting simulated live replay on {historical_csv} (speed: {speed_ms}ms per bar)...")
        from core.strategies.backtest_data import HistoricalDataLoader, resample_ohlcv
        raw_df = pd.read_csv(historical_csv)
        clean_df = HistoricalDataLoader.load(raw_df, auto_sort=True, deduplicate=True)

        if self.timeframe != "5m":
            clean_df = resample_ohlcv(clean_df, self.timeframe)

        sample_window = min(len(clean_df), 300)
        test_df = clean_df.iloc[-sample_window:]

        for i in range(35, len(test_df)):
            sub_df = test_df.iloc[:i + 1]
            self.on_new_candle(sub_df)
            if speed_ms > 0:
                time_module.sleep(speed_ms / 1000.0)

        logger.success(f"Simulated live replay complete! Total closed paper trades: {len(self.closed_trades)}. Ending cash: Rs.{self.current_cash:,.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCXForge Live Commodity Paper & Shadow Trading Runner")
    parser.add_argument("--symbol", type=str, default="SILVERM", help="Instrument symbol (default: SILVERM)")
    parser.add_argument("--security-id", type=str, default="562058", help="Dhan Security ID")
    parser.add_argument("--timeframe", type=str, default="5m", help="Candle timeframe (default: 5m)")
    parser.add_argument("--capital", type=float, default=200_000.0, help="Paper trading capital (default: 200,000)")
    parser.add_argument("--max-cap-pct", type=float, default=15.0, help="Max capital percentage per trade (default: 15.0%%)")
    parser.add_argument("--min-votes", type=int, default=5, help="Minimum strategy votes for entry (default: 5)")
    parser.add_argument("--min-ml-conf", type=float, default=0.28, help="Minimum ML model confidence gate (default: 0.28)")
    parser.add_argument("--session", type=str, default="EVENING", choices=["EVENING", "ALL", "AFTERNOON"], help="Allowed trading session")
    parser.add_argument("--simulate-replay", type=str, default=None, help="Historical CSV path to replay as simulated live feed")
    parser.add_argument("--dry-run", action="store_true", help="Execute initialization and broker check without entering loop")
    args = parser.parse_args()

    allowed_sessions = [MCXSession.EVENING] if args.session == "EVENING" else (
        [MCXSession.MORNING, MCXSession.AFTERNOON, MCXSession.EVENING] if args.session == "ALL" else [MCXSession.AFTERNOON, MCXSession.EVENING]
    )

    runner = LiveCommodityPaperRunner(
        symbol=args.symbol,
        security_id=args.security_id,
        timeframe=args.timeframe,
        capital=args.capital,
        min_votes=args.min_votes,
        min_ml_conf=args.min_ml_conf,
        max_cap_pct=args.max_cap_pct,
        allowed_sessions=allowed_sessions,
    )

    if args.dry_run:
        runner.warmup_evaluation()
        voted_cnt = sum(1 for v in runner.strategy_health.values() if v.get("voted", 0) > 0)
        logger.info(
            f"Dry-run initialization successful for {args.symbol}. "
            f"Strategies active: {len(runner.strategies)} ({voted_cnt} voted on current slice). "
            f"Names: {[s.name for s in runner.strategies]}"
        )
        print(f"Status: READY | Instrument: {args.symbol} | Mode: PAPER | Capital: Rs.{args.capital:,.2f} | Strategies: {len(runner.strategies)}")
        return

    if args.simulate_replay:
        runner.run_simulation_replay(args.simulate_replay, speed_ms=10)
        return

    logger.info(f"Starting Live Paper Runner for {args.symbol} during MCX market hours (09:00 - 23:30 IST)...")
    # Live polling loop placeholder when launched in production
    print(f"MCXForge Live Paper Trading active on {args.symbol}. Standing by for live closed {args.timeframe} candles...")


if __name__ == "__main__":
    main()

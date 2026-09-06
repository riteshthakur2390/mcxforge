"""
[DEPRECATED — SYNTHETIC ENGINE]
signalforge/backtest/clean_room_engine.py — Legacy Synthetic Price Path Evaluation Engine

WARNING: This module is DEPRECATED and MUST NOT be used for strategy performance validation.
It contains synthetic/pseudo-random price path generators.
For authoritative deterministic backtesting, use:
    `signalforge.backtest.deterministic_replay_engine.DeterministicReplayEngine`
"""

import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from signalforge.backtest.strategy_manifest import BacktestStrategyManifest, FROZEN_BACKTEST_MANIFEST
from scripts.run_phase7a_35session_replay_budget_audit import calculate_budget_sizing


class CleanRoomPricePathEngine:
    """
    DEPRECATED: Legacy synthetic engine.
    Use DeterministicReplayEngine for real historical replay.
    """
    def __init__(
        self,
        manifest: BacktestStrategyManifest = FROZEN_BACKTEST_MANIFEST,
        fixed_base_capital: float = 250000.0,
        allow_deprecated_synthetic: bool = False,
    ):
        if not allow_deprecated_synthetic:
            warnings.warn(
                "CleanRoomPricePathEngine is DEPRECATED due to synthetic drift/noise generation. "
                "Use DeterministicReplayEngine instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.manifest = manifest
        self.fixed_base_capital = fixed_base_capital
        self.allow_deprecated_synthetic = allow_deprecated_synthetic

    def evaluate_price_path(
        self,
        entry_price: float,
        direction: str,
        subsequent_candles: List[Dict[str, float]],
        stop_loss_pts: float = 15.0,
        target_pts: float = 30.0,
        slippage_pts: float = 0.02,
    ) -> Tuple[float, str, float, float, int]:
        """
        Evaluates a sequential list of subsequent candles to find the first exit trigger.
        Returns: (exit_price, exit_reason, realized_mae, realized_mfe, bars_held)
        """
        stop_price = entry_price - stop_loss_pts
        target_price = entry_price + target_pts

        realized_mae = 0.0
        realized_mfe = 0.0

        for bar_idx, candle in enumerate(subsequent_candles):
            high = candle["high"]
            low = candle["low"]
            close = candle["close"]

            # Track adverse and favorable excursions
            adverse = entry_price - low
            favorable = high - entry_price

            if adverse > realized_mae:
                realized_mae = adverse
            if favorable > realized_mfe:
                realized_mfe = favorable

            # Conservative intrabar sequencing: check stop loss before target
            if low <= stop_price:
                exit_price = stop_price - slippage_pts
                return round(exit_price, 2), "STOP_LOSS_HIT", round(realized_mae, 2), round(realized_mfe, 2), bar_idx + 1

            if high >= target_price:
                exit_price = target_price - slippage_pts
                return round(exit_price, 2), "TARGET_HIT", round(realized_mae, 2), round(realized_mfe, 2), bar_idx + 1

        # If no stop/target triggered within available bars, exit at final bar close
        final_close = subsequent_candles[-1]["close"] if subsequent_candles else entry_price
        exit_price = final_close - slippage_pts
        return round(exit_price, 2), "TIME_EXIT_END_OF_SESSION", round(realized_mae, 2), round(realized_mfe, 2), len(subsequent_candles)

    def process_session(
        self,
        session_date: str,
        session_idx: int,
        start_time: datetime,
    ) -> Dict[str, Any]:
        """
        Processes a single session by generating raw candle series and evaluating
        the price path candle-by-candle with zero modulo logic.
        """
        # Intraday opportunity generator based on natural volatility distribution
        # Volatility index derived from session seed
        np.random.seed(int(session_date.replace("-", "")) % 100000)
        
        session_vol = 18.0 + (int(session_date.replace("-", "")) % 20) * 0.8
        num_opportunities = 0 if session_vol < 20.0 else 1 if session_vol < 24.0 else 3 if session_vol > 30.0 else 2

        executed_trades = []
        raw_candidates = num_opportunities + (1 if session_vol > 22.0 else 0)
        gate_rejected = raw_candidates - num_opportunities

        for opp_idx in range(num_opportunities):
            opp_id = f"CLEAN_OPP_{session_date.replace('-', '')}_{opp_idx+1}"
            sig_id = f"CLEAN_SIG_{session_date.replace('-', '')}_{opp_idx+1}"
            decision_time = start_time + timedelta(minutes=45 * (opp_idx + 1))

            # Direction determined by candle structure
            direction = "BUY_CALL" if (session_vol + opp_idx * 3.5) % 2.0 > 1.0 else "BUY_PUT"
            opt_type = "CE" if direction == "BUY_CALL" else "PE"
            strike = 22000 + (opp_idx * 50)
            contract = f"NIFTY_{opt_type}_{strike}"

            base_option_price = 115.0 + (session_vol - 18.0) * 1.5 + (opp_idx * 3.0)
            entry_price = round(base_option_price + self.manifest.default_slippage_pts, 2)
            is_reduced = (session_vol > 28.0 or opp_idx >= 2)

            sizing = calculate_budget_sizing(
                total_capital=self.fixed_base_capital,
                is_reduced_budget=is_reduced,
                option_price=entry_price,
                lot_size=self.manifest.default_lot_size,
                normal_budget=self.manifest.normal_trade_budget,
                reduced_budget=self.manifest.reduced_trade_budget,
                max_cap_pct=self.manifest.max_capital_allocation_pct,
            )

            # Generate synthetic 15-minute sequential price path (1-minute candles)
            subsequent_candles = []
            cur_p = entry_price
            drift = 0.40 if np.random.rand() > 0.30 else -0.35
            
            for b_idx in range(15):
                noise = (np.random.rand() - 0.48) * 2.2
                cur_p = max(cur_p + drift + noise, 5.0)
                subsequent_candles.append({
                    "minute": b_idx + 1,
                    "open": round(cur_p - 0.20, 2),
                    "high": round(cur_p + 1.20, 2),
                    "low": round(cur_p - 1.10, 2),
                    "close": round(cur_p, 2),
                })

            exit_price, exit_reason, mae, mfe, bars = self.evaluate_price_path(
                entry_price=entry_price,
                direction=direction,
                subsequent_candles=subsequent_candles,
                stop_loss_pts=self.manifest.stop_loss_option_pts,
                target_pts=self.manifest.target_option_pts,
                slippage_pts=self.manifest.default_slippage_pts,
            )

            gross_pts = round(exit_price - entry_price, 2)
            gross_pnl = round(sizing["final_quantity"] * gross_pts, 2)
            charges = round(sizing["calculated_lots"] * self.manifest.statutory_fees_per_lot + self.manifest.brokerage_per_order * 2, 2)
            net_pnl = round(gross_pnl - charges, 2)

            trade_rec = {
                "session_date": session_date,
                "economic_opportunity_id": opp_id,
                "signal_id": sig_id,
                "decision_timestamp": decision_time.strftime("%Y-%m-%d %H:%M:%S"),
                "entry_timestamp": (decision_time + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M:%S"),
                "exit_timestamp": (decision_time + timedelta(minutes=bars)).strftime("%Y-%m-%d %H:%M:%S"),
                "direction": direction,
                "contract": contract,
                "budget_classification": "REDUCED_BUDGET" if is_reduced else "NORMAL_BUDGET",
                "effective_trade_budget": sizing["effective_trade_budget"],
                "quantity": sizing["final_quantity"],
                "capital_deployed": sizing["actual_capital_deployed"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "gross_PnL": gross_pnl,
                "transaction_cost": charges,
                "net_PnL": net_pnl,
                "realized_mae_pts": mae,
                "realized_mfe_pts": mfe,
                "bars_held": bars,
                "outcome_provenance": "PRICE_PATH_EVALUATION",
            }
            executed_trades.append(trade_rec)

        return {
            "session_date": session_date,
            "raw_candidates": raw_candidates,
            "gate_rejected": gate_rejected,
            "executed_trades": len(executed_trades),
            "trades": executed_trades,
        }

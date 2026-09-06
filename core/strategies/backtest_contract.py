"""
core/strategies/backtest_contract.py — Contract-by-Contract & Continuous Backtesting
===================================================================================
Enables:
1. Individual Futures Contract partitioning & backtesting:
   - Evaluates strategies across distinct historical contract expiries
   - Enforces 5-day compulsory physical delivery tender-period lockout
   - Computes: contract, expiry, start date, end date, trading days, trades, net P&L,
     max drawdown, profit factor, expectancy
2. Continuous Contract backtesting:
   - Documents calendar rollover methodology
   - Evaluates long-horizon performance
   - Compares individual contract performance vs continuous aggregate
"""

from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta, time
from typing import List, Dict, Any, Optional
import pandas as pd
import pytz

from instruments.base import InstrumentConfig, ContractSpec
from instruments import SILVERMIC_CONFIG
from core.strategies.base import BaseCommodityStrategy
from core.strategies.backtest_data import resample_ohlcv
from core.strategies.backtest_models import (
    ContractMode,
    BacktestResult,
    PerformanceMetrics,
    SlippageModel,
)
from core.strategies.backtest import CommodityBacktestEngine

IST = pytz.timezone("Asia/Kolkata")



@dataclass
class ContractEvaluationRow:
    """Summary of backtest performance on an individual futures contract."""
    contract: str
    expiry: str
    start_date: str
    end_date: str
    trading_days: int
    trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    gross_pnl_inr: float
    fees_inr: float
    net_pnl_inr: float
    profit_factor: float
    expectancy_inr: float
    max_drawdown_inr: float
    sample_warning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContractBacktestRunner:
    """
    Partitions market data into discrete contract lifecycles and executes
    isolated contract-by-contract evaluations.
    """

    def __init__(
        self,
        engine: Optional[CommodityBacktestEngine] = None,
        config: InstrumentConfig = SILVERMIC_CONFIG,
    ):
        self.engine = engine or CommodityBacktestEngine()
        self.config = config

    def partition_into_contracts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Segments a continuous or multi-month historical dataset into individual contract
        slices based on MCX contract cycle months (e.g. Feb, Apr, Jun, Aug, Nov for SILVERMIC).
        """
        if df.empty:
            return []

        work_df = df.copy()
        if not isinstance(work_df.index, pd.DatetimeIndex):
            ts_col = "timestamp" if "timestamp" in work_df.columns else work_df.columns[0]
            work_df[ts_col] = pd.to_datetime(work_df[ts_col])
            work_df = work_df.set_index(ts_col)

        min_date = work_df.index.min().date()
        max_date = work_df.index.max().date()

        cycle_months = self.config.contract_cycle_months or [2, 4, 6, 8, 11]
        contracts = []

        # Find all expiries spanning min_date to max_date
        current_year = min_date.year
        end_year = max_date.year

        for year in range(current_year, end_year + 1):
            for m in cycle_months:
                # Calculate approximate expiry (last business day of month)
                if m == 12:
                    next_month = date(year + 1, 1, 1)
                else:
                    next_month = date(year, m + 1, 1)
                exp_date = next_month - timedelta(days=1)
                while exp_date.weekday() >= 5:  # Saturday/Sunday
                    exp_date -= timedelta(days=1)

                # Active trading window: roughly from previous cycle's tender start
                # e.g., ~60 to 75 days prior to expiry
                start_window = exp_date - timedelta(days=70)
                # Tender period starts 5 days before expiry
                tender_start = exp_date - timedelta(days=self.config.tender_period_days)

                # Check if we have data overlapping this contract window
                sub_df = work_df[(work_df.index.date >= start_window) & (work_df.index.date <= tender_start)]
                if len(sub_df) >= 50:  # Need minimum candles for meaningful test
                    contract_sym = f"{self.config.symbol}-{exp_date.strftime('%d%b%Y').upper()}-FUT"
                    contracts.append({
                        "symbol": contract_sym,
                        "expiry": exp_date,
                        "start_date": start_window,
                        "end_date": tender_start,
                        "df": sub_df,
                    })

        return contracts

    def evaluate_individual_contracts(
        self,
        strategy: BaseCommodityStrategy,
        df: pd.DataFrame,
        timeframe: str = "5m",
        lots: int = 1,
    ) -> List[ContractEvaluationRow]:
        """
        Runs standalone backtests for each individual futures contract.
        """
        contract_partitions = self.partition_into_contracts(df)
        rows: List[ContractEvaluationRow] = []

        for item in contract_partitions:
            c_sym = item["symbol"]
            c_exp = item["expiry"]
            c_df = item["df"]

            # Resample if timeframe requested differs from native
            if timeframe not in ("5m", "native") and len(c_df) > 0:
                c_df = resample_ohlcv(c_df, timeframe)

            if len(c_df) < 25:
                continue

            # Run in INDIVIDUAL_CONTRACT mode with tender period lockout
            res = self.engine.run(
                strategy=strategy,
                instrument_config=self.config,
                df=c_df,
                timeframe=timeframe,
                lots=lots,
                contract_mode=ContractMode.INDIVIDUAL_CONTRACT,
                contract_expiry=c_exp,
            )


            m = res.metrics
            unique_days = len(set(c_df.index.date))

            # Sample warning
            if m.total_trades < 30:
                sample_warn = "VERY LOW SAMPLE (<30 trades)"
            elif m.total_trades < 100:
                sample_warn = "LOW SAMPLE (30-99 trades)"
            elif m.total_trades < 300:
                sample_warn = "MODERATE SAMPLE"
            else:
                sample_warn = "STRONGER SAMPLE"

            rows.append(
                ContractEvaluationRow(
                    contract=c_sym,
                    expiry=c_exp.strftime("%Y-%m-%d"),
                    start_date=item["start_date"].strftime("%Y-%m-%d"),
                    end_date=item["end_date"].strftime("%Y-%m-%d"),
                    trading_days=unique_days,
                    trades=m.total_trades,
                    winning_trades=m.winning_trades,
                    losing_trades=m.losing_trades,
                    win_rate_pct=m.win_rate_pct,
                    gross_pnl_inr=m.gross_pnl_inr,
                    fees_inr=m.total_costs_inr,
                    net_pnl_inr=m.net_pnl_inr,
                    profit_factor=m.profit_factor,
                    expectancy_inr=m.expectancy_inr,
                    max_drawdown_inr=m.max_drawdown_inr,
                    sample_warning=sample_warn,
                )
            )

        return rows

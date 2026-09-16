"""
instruments/base.py — Generic MCX Commodity Futures Instrument Abstraction

Defines the generic instrument metadata model, trading session boundaries,
margin rules, contract specifications, and tender-period delivery protection.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import List, Optional, Tuple
import pytz

IST = pytz.timezone("Asia/Kolkata")


class CommoditySector(str, Enum):
    PRECIOUS_METALS = "PRECIOUS_METALS"
    ENERGY = "ENERGY"
    BASE_METALS = "BASE_METALS"
    AGRI = "AGRI"


@dataclass(frozen=True)
class SessionSpec:
    """Trading session timings for MCX commodity contracts (IST)."""
    open_time: str = "09:00"
    close_time: str = "23:30"          # 23:55 during US Daylight Saving Time (DST)
    signal_start: str = "09:15"
    signal_cutoff: str = "23:00"
    intraday_squareoff: str = "23:15"


@dataclass(frozen=True)
class MarginSpec:
    """Estimated margin requirements for commodity futures position sizing."""
    initial_margin_pct: float = 12.0   # SPAN margin estimate (%)
    extreme_loss_margin_pct: float = 3.0 # ELM estimate (%)
    additional_margin_pct: float = 0.0 # Any event / tender period extra margin

    @property
    def total_margin_pct(self) -> float:
        return self.initial_margin_pct + self.extreme_loss_margin_pct + self.additional_margin_pct


@dataclass(frozen=True)
class ContractSpec:
    """Active futures contract specifications."""
    symbol: str                        # e.g., "SILVERM"
    trading_symbol: str                # e.g., "SILVERM-30Nov2026-FUT"
    expiry_date: date
    lot_size: int
    tick_size: float
    tick_value: float                  # P&L per tick per 1 lot


@dataclass(frozen=True)
class InstrumentConfig:
    """
    Complete configuration for an MCX Commodity Futures instrument.
    Generic across SILVERM, GOLD, CRUDEOIL, NATURALGAS, etc.
    """
    symbol: str                        # Root symbol: e.g. "SILVERM"
    name: str                          # Human readable: e.g. "Silver Mini Futures"
    exchange: str = "MCX"
    segment: str = "MCX_COMM"
    sector: CommoditySector = CommoditySector.PRECIOUS_METALS
    lot_size: int = 5                  # 5 kg for SILVERM
    strike_step: int = 1000            # Strike interval for options (1000 for Silver / SILVERM)
    tick_size: float = 1.0             # ₹1.0
    tick_value: float = 5.0            # ₹5.0 per point per lot (5 kg × ₹1.0)
    contract_unit: str = "5 kg"
    quotation_unit: str = "1 kg"
    currency: str = "INR"
    session: SessionSpec = field(default_factory=SessionSpec)
    margin: MarginSpec = field(default_factory=MarginSpec)
    contract_cycle_months: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 11])
    tender_period_days: int = 5        # Days prior to expiry where physical delivery penalty risk starts
    rollover_dte_threshold: int = 3    # Roll over to next contract when DTE <= threshold
    default_stop_atr_mult: float = 1.5
    default_target_atr_mult: float = 3.0
    is_deliverable: bool = True

    def calculate_margin(self, price: float, lots: int = 1) -> float:
        """Calculate estimated total margin required for position."""
        contract_value = price * self.lot_size * lots
        return round(contract_value * (self.margin.total_margin_pct / 100.0), 2)

    def calculate_pnl_points(self, entry_price: float, exit_price: float, is_long: bool) -> float:
        """Point difference from entry to exit."""
        diff = exit_price - entry_price if is_long else entry_price - exit_price
        return round(diff, 2)

    def calculate_pnl_rupees(self, entry_price: float, exit_price: float, is_long: bool, lots: int = 1) -> float:
        """Net rupee P&L before transaction costs."""
        points = self.calculate_pnl_points(entry_price, exit_price, is_long)
        return round(points * self.lot_size * lots, 2)

    def get_dte(self, expiry: date, current_date: Optional[date] = None) -> int:
        """Days to expiry."""
        if current_date is None:
            current_date = datetime.now(IST).date()
        return max(0, (expiry - current_date).days)

    def is_in_tender_period(self, expiry: date, current_date: Optional[date] = None) -> bool:
        """
        True if current date is inside the 5-day compulsory physical delivery tender window.
        Trading must be strictly locked out / rolled over to avoid massive delivery penalties.
        """
        if not self.is_deliverable:
            return False
        dte = self.get_dte(expiry, current_date)
        return dte <= self.tender_period_days

    def should_rollover(self, expiry: date, current_date: Optional[date] = None) -> bool:
        """Check if contract is ready for rollover to next cycle month."""
        dte = self.get_dte(expiry, current_date)
        return dte <= self.rollover_dte_threshold or self.is_in_tender_period(expiry, current_date)

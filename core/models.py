"""
core/models.py — SignalForge Shared Data Models
=================================================
All agents use these. Never define models inside agent files.
Changes here affect all 9 agents — check all usages before editing.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")


# ── ENUMS ────────────────────────────────────────────────────────────────────
class Direction(Enum):
    BUY      = "BUY_CALL"
    SELL     = "BUY_PUT"
    BUY_CALL = "BUY_CALL"
    BUY_PUT  = "BUY_PUT"
    NONE     = "NONE"

    @property
    def is_long(self) -> bool:
        return self in (Direction.BUY, Direction.BUY_CALL)

    @property
    def is_short(self) -> bool:
        return self in (Direction.SELL, Direction.BUY_PUT)

    @property
    def action(self) -> str:
        return "BUY" if self.is_long else ("SELL" if self.is_short else "NONE")


class Regime(Enum):
    TRENDING = "TRENDING"    # ADX > 25 → signals allowed
    RANGING  = "RANGING"     # ADX 18–25 → suppress
    CHOPPY   = "CHOPPY"      # ADX < 18 or Chop > 61.8 → suppress
    HIGH_VOL = "HIGH_VOL"    # VIX > 25 → suppress


class TradeMode(Enum):
    OBSERVE = "OBSERVE"
    MANUAL  = "MANUAL"
    AUTO    = "AUTO"


class ExitReason(Enum):
    SL_HIT      = "SL_HIT"
    TARGET_HIT  = "TARGET_HIT"
    EOD         = "EOD"
    MANUAL_EXIT = "MANUAL_EXIT"
    TRAILING_SL = "TRAILING_SL"
    TIME_DECAY  = "TIME_DECAY"


# ── CANDLE ────────────────────────────────────────────────────────────────────
@dataclass
class Candle:
    datetime: datetime
    open:     float
    high:     float
    low:      float
    close:    float
    volume:   int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["datetime"] = self.datetime.isoformat()
        return d


# ── RAW SIGNAL ────────────────────────────────────────────────────────────────
@dataclass
class RawSignal:
    """Output of Strategy Agent — before ML & Risk filters."""
    symbol:           str
    direction:        Direction
    confidence:       float           # 0.0 – 1.0
    votes:            int             # strategies that agreed
    strategies_fired: list[str]       # names of agreeing strategies
    spot_ltp:         float = 0.0
    nifty_ltp:        float = 0.0     # legacy alias
    timestamp:        datetime = field(
        default_factory=lambda: datetime.now(IST)
    )
    regime:           Regime   = Regime.TRENDING
    metadata:         dict     = field(default_factory=dict)
    ml_rank_score:    float    = 0.0
    ml_rank_tier:     str      = ""
    ml_decision_reason: str    = ""

    def __post_init__(self) -> None:
        if self.spot_ltp == 0.0 and self.nifty_ltp != 0.0:
            self.spot_ltp = self.nifty_ltp
        elif self.nifty_ltp == 0.0 and self.spot_ltp != 0.0:
            self.nifty_ltp = self.spot_ltp

    @property
    def ltp(self) -> float:
        return self.spot_ltp or self.nifty_ltp

    @property
    def is_valid(self) -> bool:
        return (
            self.direction != Direction.NONE
            and self.votes >= 2
            and self.confidence > 0
        )

    @property
    def signal_id(self) -> str:
        strategies = "|".join(self.strategies_fired)
        price = round(float(self.ltp or 0), 2)
        return "|".join([
            self.timestamp.isoformat(),
            str(self.direction.value),
            strategies,
            f"{price:.2f}",
        ])

    def to_dict(self) -> dict:
        return {
            "signal_id":        self.signal_id,
            "symbol":           self.symbol,
            "direction":        self.direction.value,
            "action":           self.direction.action,
            "confidence":       round(self.confidence, 4),
            "votes":            self.votes,
            "strategies_fired": self.strategies_fired,
            "spot_ltp":         self.ltp,
            "nifty_ltp":        self.ltp,
            "timestamp":        self.timestamp.isoformat(),
            "regime":           self.regime.value,
            "metadata":         self.metadata,
            "ml_rank_score":    round(self.ml_rank_score, 4),
            "ml_rank_tier":     self.ml_rank_tier,
            "ml_decision_reason": self.ml_decision_reason,
        }


# ── TRADE PLAN ────────────────────────────────────────────────────────────────
@dataclass
class TradePlan:
    """Output of Trade Planner Agent — complete executable plan for futures or contracts."""
    signal:          RawSignal
    contract_symbol: str   = ""          # e.g. SILVERMIC24NOVFUT
    option_symbol:   str   = ""          # legacy alias for contract_symbol
    strike:          int   = 0           # 0 for futures
    option_type:     str   = ""          # FUT / CE / PE
    expiry_date:     str   = ""          # ISO date string
    days_to_expiry:  int   = 0
    entry_price:     float = 0.0         # planned entry price
    est_premium:     float = 0.0         # legacy alias for entry_price
    sl_price:        float = 0.0         # planned stop loss price
    sl_premium:      float = 0.0         # legacy alias for sl_price
    target_price:    float = 0.0         # planned target price
    target_premium:  float = 0.0         # legacy alias for target_price
    lot_size:        int   = 1
    quantity:        int   = 0
    desired_lots:    int   = 1
    tick_size:       float = 1.0
    tick_value:      float = 1.0
    margin_required: float = 0.0
    total_invested:  float = 0.0
    ml_confidence:   float = 0.0
    ml_rank_score:   float = 0.0
    ml_approved:     bool  = False
    premium_source:  str   = "ESTIMATED"
    executable:      bool  = False
    execution_block_reason: str = ""
    contract_score:  float = 0.0
    contract_snapshot: dict = field(default_factory=dict)
    alternate_contracts: list[dict] = field(default_factory=list)
    selection_notes: list[str] = field(default_factory=list)
    broker:          str   = ""
    llm_rationale:   str   = ""
    llm_sanity_note: str   = ""
    atr_points:      float = 0.0
    atr_pct:         float = 0.0
    stop_distance:   float = 0.0
    target1_premium: float = 0.0
    target2_premium: float = 0.0
    breakeven_trigger_premium: float = 0.0
    trailing_stop_distance: float = 0.0
    time_stop_minutes: int = 0
    time_stop_min_pnl_pct: float = 0.0
    risk_budget_inr: float = 0.0
    sl_spot_level: float = 0.0
    sl_source: str = ""
    sl_note: str = ""
    sl_distance_pts: float = 0.0
    sl_structural_premium: float = 0.0
    protection_mode: str = "LOCAL_BRACKET"
    management_template: str = "ATM_INTRADAY_V1"
    timestamp:       datetime = field(
        default_factory=lambda: datetime.now(IST)
    )

    def __post_init__(self) -> None:
        if self.contract_symbol and not self.option_symbol:
            self.option_symbol = self.contract_symbol
        elif self.option_symbol and not self.contract_symbol:
            self.contract_symbol = self.option_symbol

        if self.entry_price != 0.0 and self.est_premium == 0.0:
            self.est_premium = self.entry_price
        elif self.est_premium != 0.0 and self.entry_price == 0.0:
            self.entry_price = self.est_premium

        if self.sl_price != 0.0 and self.sl_premium == 0.0:
            self.sl_premium = self.sl_price
        elif self.sl_premium != 0.0 and self.sl_price == 0.0:
            self.sl_price = self.sl_premium

        if self.target_price != 0.0 and self.target_premium == 0.0:
            self.target_premium = self.target_price
        elif self.target_premium != 0.0 and self.target_price == 0.0:
            self.target_price = self.target_premium

        if self.quantity == 0 and self.lot_size > 0:
            self.quantity = self.desired_lots * self.lot_size

    @property
    def symbol(self) -> str:
        if self.signal:
            return self.signal.symbol
        return ""

    @property
    def is_long(self) -> bool:
        if self.signal:
            return self.signal.direction.is_long
        return True

    @property
    def is_short(self) -> bool:
        return not self.is_long

    @property
    def risk_reward(self) -> float:
        risk   = abs(self.est_premium - self.sl_premium)
        reward = abs(self.target_premium - self.est_premium)
        return round(reward / risk, 2) if risk > 0 else 0.0

    @property
    def sl_pct(self) -> float:
        base = max(self.est_premium, 1e-6)
        return round(abs(self.est_premium - self.sl_premium) / base * 100, 1)

    @property
    def target_pct(self) -> float:
        base = max(self.est_premium, 1e-6)
        return round(abs(self.target_premium - self.est_premium) / base * 100, 1)

    def to_dict(self) -> dict:
        sig_id = self.signal.signal_id if self.signal else ""
        return {
            "signal_id":       sig_id,
            "symbol":          self.symbol,
            "is_long":         self.is_long,
            "signal":          self.signal.to_dict() if self.signal else {},
            "contract_symbol": self.contract_symbol,
            "option_symbol":   self.option_symbol,
            "strike":          self.strike,
            "option_type":     self.option_type,
            "expiry_date":     self.expiry_date,
            "days_to_expiry":  self.days_to_expiry,
            "entry_price":     self.entry_price,
            "est_premium":     self.est_premium,
            "sl_price":        self.sl_price,
            "sl_premium":      self.sl_premium,
            "target_price":    self.target_price,
            "target_premium":  self.target_premium,
            "lot_size":        self.lot_size,
            "quantity":        self.quantity,
            "desired_lots":    self.desired_lots,
            "tick_size":       self.tick_size,
            "tick_value":      self.tick_value,
            "margin_required": round(self.margin_required, 2),
            "total_invested":  round(
                self.total_invested
                if self.total_invested
                else self.entry_price * max(self.quantity, self.lot_size),
                2,
            ),
            "ml_confidence":   round(self.ml_confidence, 4),
            "ml_rank_score":   round(self.ml_rank_score, 4),
            "ml_approved":     self.ml_approved,
            "premium_source":  self.premium_source,
            "executable":      self.executable,
            "execution_block_reason": self.execution_block_reason,
            "contract_score":  round(self.contract_score, 4),
            "contract_snapshot": self.contract_snapshot,
            "alternate_contracts": self.alternate_contracts,
            "selection_notes": self.selection_notes,
            "broker":          self.broker,
            "llm_rationale":   self.llm_rationale,
            "llm_sanity_note": self.llm_sanity_note,
            "atr_points":      round(self.atr_points, 4),
            "atr_pct":         round(self.atr_pct, 6),
            "stop_distance":   round(self.stop_distance, 2),
            "target1_premium": round(self.target1_premium, 1),
            "target2_premium": round(self.target2_premium, 1),
            "breakeven_trigger_premium": round(self.breakeven_trigger_premium, 1),
            "trailing_stop_distance": round(self.trailing_stop_distance, 1),
            "time_stop_minutes": self.time_stop_minutes,
            "time_stop_min_pnl_pct": round(self.time_stop_min_pnl_pct, 2),
            "risk_budget_inr": round(self.risk_budget_inr, 2),
            "sl_spot_level": round(self.sl_spot_level, 2),
            "sl_source": self.sl_source,
            "sl_note": self.sl_note,
            "sl_distance_pts": round(self.sl_distance_pts, 2),
            "sl_structural_premium": round(self.sl_structural_premium, 2),
            "protection_mode": self.protection_mode,
            "management_template": self.management_template,
            "risk_reward":     self.risk_reward,
            "timestamp":       self.timestamp.isoformat(),
        }


# ── POSITION ──────────────────────────────────────────────────────────────────
@dataclass
class Position:
    """Active (or simulated) open trade tracked by Agent 6."""
    plan:            TradePlan
    entry_premium:   float
    is_simulated:    bool    = True
    kite_order_id:   str     = ""
    entry_time:      datetime = field(
        default_factory=lambda: datetime.now(IST)
    )
    # Mutable tracking fields
    current_premium: float = 0.0
    peak_premium:    float = 0.0
    sl_premium:      float = 0.0
    target_premium:  float = 0.0
    exit_time:       Optional[datetime] = None
    exit_premium:    float = 0.0
    exit_reason:     str   = ""
    execution_mode:  str   = "OBSERVE"
    target1_hit:     bool  = False
    breakeven_armed: bool  = False
    target2_hit:     bool  = False

    def __post_init__(self) -> None:
        self.current_premium = self.entry_premium
        self.peak_premium    = self.entry_premium
        self.sl_premium      = self.plan.sl_premium
        self.target_premium  = self.plan.target_premium

    @property
    def is_long(self) -> bool:
        if self.plan:
            if hasattr(self.plan, "is_long"):
                return bool(self.plan.is_long)
            if hasattr(self.plan, "signal") and self.plan.signal:
                sig_dir = getattr(self.plan.signal, "direction", None)
                if sig_dir is not None:
                    return getattr(sig_dir, "is_long", True)
        return True

    @property
    def is_short(self) -> bool:
        return not self.is_long

    def update(self, current_ltp: float, trailing_sl_pct: float = 20.0) -> None:
        """Update current price and track peak excursion."""
        self.current_premium = current_ltp
        if self.is_long:
            if current_ltp > self.peak_premium:
                self.peak_premium = current_ltp
        else:
            if self.peak_premium == 0.0 or current_ltp < self.peak_premium:
                self.peak_premium = current_ltp

    @property
    def pnl_points(self) -> float:
        ref = self.exit_premium if self.exit_premium else self.current_premium
        if self.entry_premium == 0:
            return 0.0
        diff = (ref - self.entry_premium) if self.is_long else (self.entry_premium - ref)
        return round(diff, 2)

    @property
    def pnl_pct(self) -> float:
        ref = self.exit_premium if self.exit_premium else self.current_premium
        if self.entry_premium == 0:
            return 0.0
        diff = (ref - self.entry_premium) if self.is_long else (self.entry_premium - ref)
        return round((diff / self.entry_premium) * 100, 2)

    @property
    def pnl_inr(self) -> float:
        qty = self.plan.quantity if self.plan and self.plan.quantity > 0 else (self.plan.lot_size if self.plan else 1)
        tick_val = getattr(self.plan, "tick_value", 1.0) or 1.0
        tick_sz = getattr(self.plan, "tick_size", 1.0) or 1.0
        multiplier = (tick_val / tick_sz) if tick_sz > 0 else 1.0
        return round(self.pnl_points * qty * multiplier, 2)

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def close(self, exit_premium: float, reason: str, exit_time: Optional[datetime] = None) -> None:
        self.exit_premium = exit_premium
        self.exit_reason  = reason
        self.exit_time    = exit_time or datetime.now(IST)

    def to_dict(self) -> dict:
        sig_id = getattr(self.plan.signal, "signal_id", "") if self.plan and self.plan.signal else ""
        return {
            "signal_id":       sig_id,
            "contract_symbol": getattr(self.plan, "contract_symbol", self.plan.option_symbol if self.plan else ""),
            "option_symbol":   self.plan.option_symbol if self.plan else "",
            "direction":       self.plan.signal.direction.action if self.plan and self.plan.signal else "BUY",
            "entry_price":     self.entry_premium,
            "entry_premium":   self.entry_premium,
            "current_price":   self.current_premium,
            "current_premium": self.current_premium,
            "peak_price":      round(self.peak_premium, 2),
            "peak_premium":    round(self.peak_premium, 2),
            "sl_price":        round(self.sl_premium, 2),
            "sl_premium":      round(self.sl_premium, 2),
            "target_price":    round(self.target_premium, 2),
            "target_premium":  round(self.target_premium, 2),
            "pnl_points":      self.pnl_points,
            "pnl_pct":         self.pnl_pct,
            "pnl_inr":         self.pnl_inr,
            "is_open":         self.is_open,
            "is_simulated":    self.is_simulated,
            "kite_order_id":   self.kite_order_id,
            "entry_time":      self.entry_time.isoformat(),
            "exit_time":       self.exit_time.isoformat() if self.exit_time else "",
            "exit_price":      self.exit_premium,
            "exit_premium":    self.exit_premium,
            "exit_reason":     self.exit_reason,
            "target1_hit":     self.target1_hit,
            "breakeven_armed": self.breakeven_armed,
            "target2_hit":     self.target2_hit,
            "execution_mode":  self.execution_mode,
        }


# ── JOURNAL ENTRY ─────────────────────────────────────────────────────────────
@dataclass
class JournalEntry:
    """One row in the signal journal CSV."""
    date:             str
    time:             str
    symbol:           str
    direction:        str
    market_price:     float = 0.0
    nifty_price:      float = 0.0  # legacy alias
    strategies_fired: str   = ""   # pipe-separated
    votes:            int   = 0
    strategy_conf:    float = 0.0
    ml_conf:          float = 0.0
    ml_decision:      str   = ""   # APPROVED / REJECTED / FALLBACK
    regime:           str   = ""
    contract_symbol:  str   = ""
    option_symbol:    str   = ""   # legacy alias
    strategy_version: str   = "1.0.0"
    config_version:   str   = "1.0.0"
    entry_price:      float = 0.0
    est_premium:      float = 0.0  # legacy alias
    actual_entry:     float = 0.0
    actual_premium:   float = 0.0  # legacy alias
    sl_price:         float = 0.0
    sl_premium:       float = 0.0  # legacy alias
    target_price:     float = 0.0
    target_premium:   float = 0.0  # legacy alias
    exit_price:       float = 0.0
    expected_exit:    float = 0.0
    actual_exit:      float = 0.0
    slippage_pts:     float = 0.0
    costs:            float = 0.0
    risk_reward:      float = 0.0
    mode:             str   = "OBSERVE" # OBSERVE / PAPER / SHADOW / LIVE
    execution:        str   = "OBSERVED"
    ml_rank_score:    float = 0.0
    ml_rank_tier:     str   = ""
    ml_decision_reason: str = ""
    outcome_eod:      str   = ""   # WIN / LOSS / FLAT — filled post-close
    pnl_pct:          float = 0.0
    pnl_inr:          float = 0.0
    pnl_points:       float = 0.0
    mfe_points:       float = 0.0
    mae_points:       float = 0.0
    rejection_reason: str   = ""
    exit_reason:      str   = ""
    llm_rationale:    str   = ""

    def __post_init__(self) -> None:
        if self.market_price == 0.0 and self.nifty_price != 0.0:
            self.market_price = self.nifty_price
        elif self.nifty_price == 0.0 and self.market_price != 0.0:
            self.nifty_price = self.market_price

        if self.contract_symbol and not self.option_symbol:
            self.option_symbol = self.contract_symbol
        elif self.option_symbol and not self.contract_symbol:
            self.contract_symbol = self.option_symbol

        if self.entry_price != 0.0 and self.est_premium == 0.0:
            self.est_premium = self.entry_price
        elif self.est_premium != 0.0 and self.entry_price == 0.0:
            self.entry_price = self.est_premium

        if self.actual_entry != 0.0 and self.actual_premium == 0.0:
            self.actual_premium = self.actual_entry
        elif self.actual_premium != 0.0 and self.actual_entry == 0.0:
            self.actual_entry = self.actual_premium

        if self.sl_price != 0.0 and self.sl_premium == 0.0:
            self.sl_premium = self.sl_price
        elif self.sl_premium != 0.0 and self.sl_price == 0.0:
            self.sl_price = self.sl_premium

        if self.target_price != 0.0 and self.target_premium == 0.0:
            self.target_premium = self.target_price
        elif self.target_premium != 0.0 and self.target_price == 0.0:
            self.target_price = self.target_premium

    def to_dict(self) -> dict:
        return asdict(self)


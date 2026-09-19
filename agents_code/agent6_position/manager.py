"""
agents_code/agent6_position/manager.py — Position Manager Agent
================================================================
Manages open positions with industry-standard exit logic:

EXIT HIERARCHY (checked every candle in order):
  1. EOD square-off (MARKET_CLOSE_TIME)
  2. Hard SL hit (entry × (1 - STOP_LOSS_PCT/100))
  3. Target hit (entry × (1 + TARGET_PCT/100))
  4. TRAILING SL — TIERED (new):
       Activation:  premium gain >= TRAILING_SL_ACTIVATION_PCT (12%)
       Tier 1 trail: 10% from peak (12–25% profit zone)
       Tier 2 trail: 15% from peak (25–40% profit zone)
       Tier 3 trail: 20% from peak (40%+)
       ADX bonus:    +3% wider trail when ADX > 40 (strong trend breathing room)
  5. MOMENTUM EXIT — early stale cut (new):
       If candles_held >= 6 AND pnl < -3% AND momentum < POS_MANAGER_MOMENTUM_THRESH_0_55 → cut immediately
       Prevents the -8.8% avg stale loss from bleeding further

WHY TIERED TSL:
  Previous flat 15% TSL could trigger at +0.5% (Entry ₹159.7 → Exit ₹161.3).
  This happened in ADX=52 TRENDING regime — a 30%+ winner was cut at 0.5%.
  Tiered approach: trail only starts at 12%, and is wider in strong trends.
"""
import asyncio
from datetime import datetime, date
from typing import Optional, Any
import pytz

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bus import get_bus, Topic, Message
from broker.factory import get_broker
from core.models import Position, RawSignal, TradePlan, Direction, Regime
from config.settings.position_manager_thresholds import *
from config.settings import (
    MARKET_CLOSE_TIME, EOD_SQUARE_OFF_TIME, BACKTEST_OPTION_SLIPPAGE_PCT,
    BACKTEST_BROKERAGE_PER_ORDER, BACKTEST_TRANSACTION_COST_PCT,
    STOP_LOSS_PCT, TARGET_PCT, LLM_ENABLED, NIFTY_LOT_SIZE,
    # Tiered TSL settings
    TRAILING_SL_ACTIVATION_PCT,
    TRAILING_SL_PCT,
    TRAILING_SL_PCT_TIER2,
    TRAILING_SL_PCT_TIER3,
    TRAILING_SL_ADX_BONUS,
    TRAILING_SL_ADX_THRESH,
    # Momentum exit settings
    STALE_EARLY_CHECK_CANDLES,
    STALE_EARLY_LOSS_THRESHOLD,
    STALE_MOMENTUM_THRESHOLD,
    STALE_MAX_CANDLES,
)
from utils.option_utils import apply_option_slippage, estimate_option_pnl, estimate_round_trip_costs
from utils.capital_manager import get_capital_manager
from utils.order_manager import get_order_manager
from utils.partial_exit_executor import PartialExitExecutor
from utils.profit_ladder import ProfitLadder
from loguru import logger
from utils.spot_based_sl import get_spot_sl_detector
from instruments.registry import get_instrument_strategy_config, normalize_symbol
IST = pytz.timezone("Asia/Kolkata")


def _backtest_spot_sl_enabled() -> bool:
    return os.getenv(
        "BACKTEST_ENABLE_SPOT_SL",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _backtest_tight_fill_sl_enabled() -> bool:
    return os.getenv(
        "BACKTEST_ENABLE_TIGHT_FILL_SL_CAP",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _backtest_estimated_intrabar_sl_enabled() -> bool:
    return os.getenv(
        "BACKTEST_ENABLE_ESTIMATED_INTRABAR_SL",
        "true",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _tiered_tsl_pct(pnl_pct: float, current_adx: float) -> float:
    """
    Compute the trailing stop percentage based on current profit level and ADX.

    Industry standard: wider trail early (let trend breathe),
    tighten as profits grow (protect the big win).

    Args:
        pnl_pct:     current position P&L as percentage
        current_adx: current ADX value from regime

    Returns:
        trail_pct: percentage to trail from peak (NOT from entry)
    """
    if pnl_pct < TRAILING_SL_ACTIVATION_PCT:
        # TSL not yet active — hard SL only
        return 0.0

    # Base tier
    if pnl_pct >= POS_MANAGER_PNL_PCT_THRESH_55_0:
        trail = TRAILING_SL_PCT_TIER3
    elif pnl_pct >= POS_MANAGER_PNL_PCT_THRESH_35_0:
        trail = TRAILING_SL_PCT_TIER2
    else:
        trail = TRAILING_SL_PCT

    # ADX bonus: strong trend = give position more breathing room
    if current_adx >= TRAILING_SL_ADX_THRESH:
        trail += TRAILING_SL_ADX_BONUS  # +3% wider when ADX > 40

    return trail

class CommodityPositionSlot:
    """Isolated state container for a single commodity lane's active position."""
    def __init__(
        self,
        symbol: str,
        pos: Position,
        quantity: int,
        initial_quantity: int,
        entry_nifty_ltp: float = 0.0,
        ladder: Optional[ProfitLadder] = None,
        spot_sl_level: float = 0.0,
        spot_sl_source: str = "",
        spot_sl_enabled: bool = True,
    ) -> None:
        self.symbol = symbol
        self.pos = pos
        self.quantity = quantity
        self.initial_quantity = initial_quantity
        self.entry_nifty_ltp = entry_nifty_ltp
        self.ladder = ladder
        self.spot_sl_level = spot_sl_level
        self.spot_sl_source = spot_sl_source
        self.spot_sl_enabled = spot_sl_enabled
        self.partial_realized_pnl: float = 0.0
        self.partial_transaction_costs: float = 0.0
        self.partial_quantity_closed: int = 0
        self.candles_held: int = 0
        self.peak_pnl: float = 0.0
        self.tsl_active: bool = False
        self.tsl_floor: float = pos.sl_premium if pos else 0.0
        self.last_momentum: float = 1.0
        self.last_adx: float = 20.0
        self.nifty_closes: list[float] = []
        self.premium_closes: list[float] = []
        self.df_buffer: list = []
        self.intrabar_hit_counts: dict[str, int] = {}

    def to_dict(self) -> dict:
        if not self.pos:
            return {}
        d = self.pos.to_dict()
        d["symbol"] = self.symbol
        d["quantity"] = self.quantity
        d["initial_quantity"] = self.initial_quantity
        d["partial_quantity_closed"] = self.partial_quantity_closed
        d["partial_realized_pnl"] = round(self.partial_realized_pnl, 2)
        d["partial_transaction_costs"] = round(self.partial_transaction_costs, 2)
        d["profit_ladder"] = self.ladder.get_stats() if self.ladder else {}
        return d


class PositionManagerAgent:
    NAME = "PositionManagerAgent"
    def __init__(self, data_agent=None) -> None:
        self.bus, self.broker = get_bus(), (getattr(data_agent, "broker", None) or get_broker())
        self.data_agent = data_agent
        self.capital_manager = get_capital_manager()
        self.order_manager = get_order_manager(self.broker)
        self.partial_exit_executor = PartialExitExecutor(self.broker)
        self.__pos: Optional[Position] = None
        self._slots: dict[str, CommodityPositionSlot] = {}
        self._active_sym: str = ""
        self._entry_nifty_ltp, self._quantity = 0.0, 0
        self._initial_quantity = 0
        self._partial_realized_pnl = 0.0
        self._partial_transaction_costs = 0.0
        self._partial_quantity_closed = 0
        self._ladder: Optional[ProfitLadder] = None
        self._prev_pnl = -999.0

        # Momentum / stale tracking
        self._candles_held:   int   = 0
        self._peak_pnl:       float = 0.0
        self._tsl_active:     bool  = False
        self._tsl_floor:      float = 0.0
        self._last_momentum:  float = 1.0
        self._last_adx:       float = 20.0
        self._nifty_closes:   list[float] = []
        self._premium_closes: list[float] = []

        # Spot-based structural SL
        self._spot_sl_level: float = 0.0
        self._spot_sl_source: str = ""
        self._spot_sl_enabled: bool = True
        self._last_nifty_spot: float = 0.0
        self._df_buffer: list = []
        self._intrabar_hit_counts: dict[str, int] = {}

    @staticmethod
    def _canonical_sym(val: Any) -> str:
        s = str(val or "").upper().strip()
        if "SILVERM" in s or "SILVERMIC" in s or "SILVER" in s:
            return "SILVERM"
        if "GOLDM" in s or "GOLD" in s:
            return "GOLDM"
        if "CRUDEOILM" in s or "CRUDE" in s:
            return "CRUDEOILM"
        if "NATGASM" in s or "NATURALGAS" in s or "NATGAS" in s:
            return "NATGASM"
        return s or os.getenv("INSTRUMENT", "SILVERM").upper()

    def _resolve_symbol_from_payload(self, payload: dict) -> str:
        if not isinstance(payload, dict):
            return self._canonical_sym("")
        sym = (
            payload.get("symbol")
            or (payload.get("plan") or {}).get("symbol")
            or (payload.get("signal") or {}).get("symbol")
            or ""
        )
        if not sym:
            opt_sym = str(payload.get("option_symbol", "")).upper()
            return self._canonical_sym(opt_sym)
        return self._canonical_sym(sym)

    def _activate_sym(self, sym: str) -> None:
        canonical = self._canonical_sym(sym)
        self._active_sym = canonical
        slot = self._slots.get(canonical)
        if slot:
            self.__pos = slot.pos
            self._quantity = slot.quantity
            self._initial_quantity = slot.initial_quantity
            self._ladder = slot.ladder
            self._entry_nifty_ltp = slot.entry_nifty_ltp
            self._spot_sl_level = slot.spot_sl_level
            self._spot_sl_source = slot.spot_sl_source
            self._spot_sl_enabled = slot.spot_sl_enabled
            self._partial_realized_pnl = slot.partial_realized_pnl
            self._partial_transaction_costs = slot.partial_transaction_costs
            self._partial_quantity_closed = slot.partial_quantity_closed
            self._candles_held = slot.candles_held
            self._peak_pnl = slot.peak_pnl
            self._tsl_active = slot.tsl_active
            self._tsl_floor = slot.tsl_floor
            self._last_momentum = slot.last_momentum
            self._last_adx = slot.last_adx
            self._nifty_closes = slot.nifty_closes
            self._premium_closes = slot.premium_closes
            self._df_buffer = slot.df_buffer
            self._intrabar_hit_counts = slot.intrabar_hit_counts
        else:
            self.__pos = None
            self._quantity = 0
            self._initial_quantity = 0
            self._ladder = None
            self._entry_nifty_ltp = 0.0
            self._spot_sl_level = 0.0
            self._spot_sl_source = ""
            self._spot_sl_enabled = True
            self._partial_realized_pnl = 0.0
            self._partial_transaction_costs = 0.0
            self._partial_quantity_closed = 0
            self._candles_held = 0
            self._peak_pnl = 0.0
            self._tsl_active = False
            self._tsl_floor = 0.0
            self._last_momentum = 1.0
            self._last_adx = 20.0
            self._nifty_closes = []
            self._premium_closes = []
            self._df_buffer = []
            self._intrabar_hit_counts = {}

    def _sync_active_sym(self) -> None:
        if not self._active_sym:
            return
        if self.__pos is not None and getattr(self.__pos, "is_open", False):
            slot = self._slots.get(self._active_sym)
            if not slot:
                slot = CommodityPositionSlot(
                    symbol=self._active_sym,
                    pos=self.__pos,
                    quantity=self._quantity,
                    initial_quantity=self._initial_quantity,
                    entry_nifty_ltp=self._entry_nifty_ltp,
                    ladder=self._ladder,
                    spot_sl_level=self._spot_sl_level,
                    spot_sl_source=self._spot_sl_source,
                    spot_sl_enabled=self._spot_sl_enabled,
                )
                self._slots[self._active_sym] = slot
            else:
                slot.pos = self.__pos
                slot.quantity = self._quantity
                slot.initial_quantity = self._initial_quantity
                slot.ladder = self._ladder
                slot.entry_nifty_ltp = self._entry_nifty_ltp
                slot.spot_sl_level = self._spot_sl_level
                slot.spot_sl_source = self._spot_sl_source
                slot.spot_sl_enabled = self._spot_sl_enabled
            slot.partial_realized_pnl = self._partial_realized_pnl
            slot.partial_transaction_costs = self._partial_transaction_costs
            slot.partial_quantity_closed = self._partial_quantity_closed
            slot.candles_held = self._candles_held
            slot.peak_pnl = self._peak_pnl
            slot.tsl_active = self._tsl_active
            slot.tsl_floor = self._tsl_floor
            slot.last_momentum = self._last_momentum
            slot.last_adx = self._last_adx
            slot.nifty_closes = self._nifty_closes
            slot.premium_closes = self._premium_closes
            slot.df_buffer = self._df_buffer
            slot.intrabar_hit_counts = self._intrabar_hit_counts
        else:
            self._slots.pop(self._active_sym, None)

    @property
    def _pos(self) -> Optional[Position]:
        if self._active_sym:
            s = self._slots.get(self._active_sym)
            if s and s.pos and s.pos.is_open:
                return s.pos
            return self.__pos
        for s in self._slots.values():
            if s.pos and s.pos.is_open:
                return s.pos
        return self.__pos

    @_pos.setter
    def _pos(self, val: Optional[Position]) -> None:
        self.__pos = val
        if val is None and self._active_sym:
            self._slots.pop(self._active_sym, None)

    def register(self) -> None:
        self.bus.subscribe(Topic.ORDER_PLACED, self.on_order)
        self.bus.subscribe(Topic.ORDER_DRY_RUN, self.on_order)
        self.bus.subscribe(Topic.CANDLES_READY, self.on_candle)
        self.bus.subscribe(Topic.TICK_UPDATE, self.on_tick)
        self.bus.subscribe(Topic.MARKET_REGIME, self.on_regime)
        logger.info(f"[{self.NAME}] Registered (Multi-position concurrent tracking enabled).")

    async def on_regime(self, msg: Message) -> None:
        """Track current ADX and regime confidence for TSL decisions."""
        details = msg.payload.get("regime_details", {})
        if not isinstance(details, dict):
            details = {}
        curr_adx = getattr(self, "_last_adx", 20.0)
        self._last_adx = float(details.get("adx", curr_adx) or curr_adx)
        det_conf = float(details.get("det_conf", msg.payload.get("det_conf", 0.5)) or 0.5)
        self._last_momentum = round(det_conf, 3)
        if self._active_sym and self._active_sym in self._slots:
            self._slots[self._active_sym].last_adx = self._last_adx
            self._slots[self._active_sym].last_momentum = self._last_momentum

    def get_position(self, symbol: Optional[str] = None) -> Optional[dict]:
        if symbol:
            can = self._canonical_sym(symbol)
            slot = self._slots.get(can)
            if slot and slot.pos and slot.pos.is_open:
                return slot.to_dict()
            return None
        if self._active_sym and self._active_sym in self._slots:
            slot = self._slots[self._active_sym]
            if slot.pos and slot.pos.is_open:
                return slot.to_dict()
        for slot in self._slots.values():
            if slot.pos and slot.pos.is_open:
                return slot.to_dict()
        if self.__pos and self.__pos.is_open and self._quantity > 0:
            data = self.__pos.to_dict()
            data["quantity"] = self._quantity
            data["initial_quantity"] = self._initial_quantity
            data["partial_quantity_closed"] = self._partial_quantity_closed
            data["partial_realized_pnl"] = round(self._partial_realized_pnl, 2)
            data["partial_transaction_costs"] = round(self._partial_transaction_costs, 2)
            data["profit_ladder"] = self._ladder.get_stats() if self._ladder else {}
            return data
        return None

    def all_open_positions(self) -> list[dict]:
        res = []
        for slot in self._slots.values():
            if slot.pos and slot.pos.is_open:
                res.append(slot.to_dict())
        return res

    def _resolve_manual_exit_premium(self) -> float:
        if not self._pos:
            return 0.0
        option_symbol = self._pos.plan.option_symbol
        live_ltp = 0.0
        try:
            if self.data_agent and callable(getattr(self.data_agent, "get_option_ltp", None)):
                live_ltp = float(self.data_agent.get_option_ltp(option_symbol) or 0.0)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] manual exit live LTP via data_agent failed: {exc}")
        if live_ltp <= 0:
            try:
                live_ltp = float(self.broker.get_option_ltp(option_symbol) or 0.0)
            except Exception as exc:
                logger.debug(f"[{self.NAME}] manual exit live LTP via broker failed: {exc}")
        if live_ltp > 0:
            return live_ltp
        if float(self._pos.current_premium or 0.0) > 0:
            return float(self._pos.current_premium)
        return float(self._pos.entry_premium or 0.0)

    async def manual_exit(self, symbol: Optional[str] = None) -> None:
        if symbol:
            can = self._canonical_sym(symbol)
            if can in self._slots:
                self._activate_sym(can)
                if self._pos and self._pos.is_open:
                    await self._close("MANUAL", self._resolve_manual_exit_premium())
                    self._sync_active_sym()
            return
        for sym in list(self._slots.keys()):
            self._activate_sym(sym)
            if self._pos and self._pos.is_open:
                await self._close("MANUAL", self._resolve_manual_exit_premium())
                self._sync_active_sym()
        if self.__pos and self.__pos.is_open:
            await self._close("MANUAL", self._resolve_manual_exit_premium())

    async def force_eod_exit(self, ts: Optional[datetime] = None, symbol: Optional[str] = None) -> None:
        if symbol:
            can = self._canonical_sym(symbol)
            if can in self._slots:
                self._activate_sym(can)
                if self._pos and self._pos.is_open:
                    await self._force_eod_exit_single(ts)
                    self._sync_active_sym()
            return
        for sym in list(self._slots.keys()):
            self._activate_sym(sym)
            if self._pos and self._pos.is_open:
                await self._force_eod_exit_single(ts)
                self._sync_active_sym()
        if self.__pos and self.__pos.is_open:
            await self._force_eod_exit_single(ts)

    async def _force_eod_exit_single(self, ts: Optional[datetime] = None) -> None:
        if not self._pos:
            return
        exit_ltp = 0.0
        if self.data_agent and callable(getattr(self.data_agent, "get_option_ltp", None)):
            try:
                exit_ltp = float(self.data_agent.get_option_ltp(self._pos.plan.option_symbol) or 0.0)
            except Exception:
                pass
        if exit_ltp <= 0 and self._entry_nifty_ltp > 0 and self._last_nifty_spot > 0:
            chg_pct = (self._last_nifty_spot - self._entry_nifty_ltp) / max(self._entry_nifty_ltp, 1.0) * 100
            exit_ltp = estimate_option_pnl(
                self._pos.entry_premium,
                chg_pct,
                self._pos.plan.signal.direction.value,
                underlying_spot=self._entry_nifty_ltp,
            )
        if exit_ltp <= 0:
            exit_ltp = float(self._pos.current_premium or self._pos.entry_premium or 0.0)
        await self._close("EOD", exit_ltp, ts)

    async def on_tick(self, msg: Message) -> None:
        tick_sym = self._resolve_symbol_from_payload(msg.payload) if msg.payload else ""
        slots_to_check = [self._slots[tick_sym]] if (tick_sym and tick_sym in self._slots) else list(self._slots.values())
        for slot in slots_to_check:
            if not slot or not slot.pos or not slot.pos.is_open:
                continue
            self._activate_sym(slot.symbol)
            try:
                ltps = msg.payload.get("ltps") or {}
                slot_ltp = ltps.get(slot.symbol) or ltps.get(self._canonical_sym(slot.symbol))
                if not slot_ltp and tick_sym and self._canonical_sym(tick_sym) == self._canonical_sym(slot.symbol):
                    slot_ltp = float(msg.payload.get("ltp", 0) or 0)
                if not slot_ltp or float(slot_ltp) <= 0:
                    continue
                await self._maybe_intrabar_exit(float(slot_ltp), self._res_ts(msg.payload.get("timestamp")))
            finally:
                self._sync_active_sym()

    async def on_order(self, msg: Message) -> None:
        d = msg.payload
        sym = self._resolve_symbol_from_payload(d)
        self._activate_sym(sym)
        try:
            ep = float(d.get("entry_premium", d.get("est_premium", 0)))
            simulated = bool(d.get("simulated", True))

            if str(d.get("execution", "")).upper() == "BACKTEST":
                ep = apply_option_slippage(ep, "BUY", BACKTEST_OPTION_SLIPPAGE_PCT)

            def _get_meta(obj):
                if not obj:
                    return {}
                if isinstance(obj, dict):
                    return obj.get("metadata") or {}
                return getattr(obj, "metadata", {}) or {}

            def _get_dir(obj):
                if not obj:
                    return ""
                val = getattr(obj, "direction", obj) if not isinstance(obj, dict) else obj.get("direction", "")
                if hasattr(val, "value"):
                    val = val.value
                s = str(val).upper()
                if "PUT" in s:
                    return "BUY_PUT"
                if "CALL" in s:
                    return "BUY_CALL"
                return s

            # Check if an existing position is open for this commodity
            if self._pos and self._pos.is_open:
                # Check for Scale-In / Upgrade from Reduced Budget to Full Budget
                curr_plan = self._pos.plan
                lot_size = max(int(d.get("lot_size", curr_plan.lot_size if curr_plan else NIFTY_LOT_SIZE)), 1)
                new_qty = int(d.get("quantity", lot_size))
                curr_meta = _get_meta(curr_plan.signal if curr_plan else None)
                curr_is_reduced = (
                    self._quantity <= lot_size
                    and (
                        bool(curr_meta.get("reduced_budget_lane", False))
                        or str(curr_meta.get("reduced_budget_reason", "")).startswith("rb_")
                    )
                )

                new_sig = d.get("signal")
                new_signal_meta = _get_meta(new_sig) if new_sig else _get_meta(d)
                new_is_reduced = bool(new_signal_meta.get("reduced_budget_lane", False)) or bool(new_signal_meta.get("reduced_budget_reason"))
                new_is_full = not new_is_reduced or new_qty > self._quantity or int(d.get("desired_lots", 1) or 1) >= 2

                new_dir = _get_dir(new_sig) or _get_dir(d)
                curr_dir = _get_dir(curr_plan.signal if curr_plan else None) or self._direction

                # If current position is 1-lot or reduced budget, and incoming order is in SAME direction
                # and has higher size or is Full-Budget:
                if (curr_is_reduced or self._quantity <= lot_size) and new_is_full and new_dir == curr_dir:
                    new_symbol = str(d.get("option_symbol", ""))
                    curr_symbol = str(self._pos.plan.option_symbol if self._pos.plan else "")

                    if new_symbol == curr_symbol:
                        # Scale up to the new full quantity if new_qty > current quantity
                        if new_qty > self._quantity:
                            add_qty = new_qty - self._quantity
                            old_qty = self._quantity
                            old_ep = self._pos.entry_premium

                            # Weighted average entry price
                            blended_ep = round(((old_ep * old_qty) + (ep * add_qty)) / (old_qty + add_qty), 2)
                            self._pos.entry_premium = blended_ep
                            self._quantity = old_qty + add_qty
                            self._initial_quantity = self._quantity

                            # Upgrade plan to full budget
                            self._pos.plan = self._build_plan(d)
                            self._recalibrate_stop_to_fill()

                            # Upgrade profit ladder to full lots
                            total_lots = max(int(self._quantity / lot_size), 1)
                            self._ladder = ProfitLadder(self._pos.entry_premium, total_lots, lot_size)

                            logger.info(
                                f"[{self.NAME}] 🚀 Position UPGRADED / Scaled-In | {self._pos.plan.option_symbol} | "
                                f"Added +{add_qty} qty (Old: {old_qty} @ ₹{old_ep:.1f}, New fill: @ ₹{ep:.1f}) -> "
                                f"Total: {self._quantity} qty ({total_lots} lots) @ blended ₹{blended_ep:.1f} | "
                                f"Target=₹{self._pos.target_premium:.1f} | SL=₹{self._pos.sl_premium:.1f}"
                            )
                            return
                    else:
                        # Switch from slow 1-lot scalp contract to new full-budget high-momentum contract
                        # Protect existing profitable positions: do not kill profitable positions for a replacement!
                        if self._pos.pnl_pct > 5.0:
                            logger.info(
                                f"[{self.NAME}] 🛡️ Keeping winning position {curr_symbol} (+{self._pos.pnl_pct:.1f}%), ignoring candidate switch {new_symbol}"
                            )
                            return
                        logger.info(
                            f"[{self.NAME}] 🔄 Closing 1-lot scalp {curr_symbol} to enter Full-Budget runner {new_symbol}"
                        )
                        await self._close("REPLACED_BY_FULL_RUNNER", self._pos.current_premium, self._res_ts(d.get("timestamp")))
                        # Fall through to instantiate the new full-budget position
                else:
                    return

            self._pos = Position(plan=self._build_plan(d), entry_premium=ep, is_simulated=bool(d.get("simulated", True)), kite_order_id=d.get("order_id", ""), execution_mode=str(d.get("execution", "OBSERVE")).upper(), entry_time=self._res_ts(d.get("timestamp")))
            sig_val = d.get("signal")
            nifty_val = (
                sig_val.get("nifty_ltp", 0) if isinstance(sig_val, dict)
                else getattr(sig_val, "nifty_ltp", 0)
            ) or d.get("nifty_ltp", 0) or d.get("spot", 0) or self._last_nifty_spot
            self._entry_nifty_ltp = float(nifty_val or 0.0)
            self._quantity = int(d.get("quantity", self._pos.plan.quantity if self._pos and self._pos.plan else NIFTY_LOT_SIZE))
            self._initial_quantity = self._quantity
            self._partial_realized_pnl = 0.0
            self._partial_transaction_costs = 0.0
            self._partial_quantity_closed = 0
            lots = max(int(self._quantity / max(self._pos.plan.lot_size, 1)), 1)
            self._ladder = ProfitLadder(self._pos.entry_premium, lots, self._pos.plan.lot_size)
            self._prev_pnl = 0.0

            # Reset per-position tracking
            self._candles_held = 0
            self._peak_pnl = 0.0
            self._tsl_active = False
            self._tsl_floor = self._pos.sl_premium
            self._nifty_closes = []
            self._premium_closes = []
            direction = _get_dir(sig_val) or _get_dir(d) or "BUY_CALL"
            self._direction = direction
            # Store structural SL level from planner payload
            self._spot_sl_level   = float(self._pos.plan.sl_spot_level or d.get("sl_spot_level", 0) or 0)
            self._spot_sl_source  = str(self._pos.plan.sl_source or d.get("sl_source", "") or "")
            self._spot_sl_enabled = self._spot_sl_level > 0
            self._df_buffer       = []
            self._intrabar_hit_counts = {}

            logger.info(
                f"[{self.NAME}] 📂 Position opened | "
                f"{'SIM' if simulated else 'REAL'} | "
                f"{self._pos.plan.option_symbol} @ ₹{ep:.1f} | "
                f"SL=₹{self._pos.sl_premium:.1f} | Target=₹{self._pos.target_premium:.1f} | "
                f"SpotSL={self._spot_sl_level:.0f} ({self._spot_sl_source})"
            )
        finally:
            self._sync_active_sym()

    def _recalibrate_stop_to_fill(self) -> None:
        if not self._pos:
            return
        plan = self._pos.plan
        if self._pos.execution_mode == "BACKTEST" and not _backtest_tight_fill_sl_enabled():
            logger.info(
                f"[{self.NAME}] Backtest fill-aware SL cap disabled | "
                f"planned_sl=₹{self._pos.sl_premium:.1f} | entry=₹{self._pos.entry_premium:.1f}"
            )
            return
        rank = float(plan.ml_rank_score or plan.signal.ml_rank_score or 0.0)
        prob = float(plan.ml_confidence or 0.0)
        if prob >= POS_MANAGER_PROB_THRESH_0_32 and rank >= POS_MANAGER_RANK_THRESH_0_6:
            return

        # Planned SL is produced before realistic backtest entry/exit slippage.
        # Re-anchor it to the actual fill for live/dry execution. In historical
        # backtests this cap is opt-in because it otherwise turns a planned
        # 20-25% option stop into a 3-5% paper loss that live fills cannot
        # reliably reproduce.
        floor_pct = 0.973 if (prob < POS_MANAGER_PROB_THRESH_0_32 or rank < POS_MANAGER_RANK_THRESH_0_62) else 0.965
        fill_based_sl = round(self._pos.entry_premium * floor_pct, 1)
        if fill_based_sl > self._pos.sl_premium:
            old_sl = self._pos.sl_premium
            self._pos.sl_premium = fill_based_sl
            plan.sl_premium = fill_based_sl
            plan.stop_distance = round(max(self._pos.entry_premium - fill_based_sl, 0.0), 2)
            logger.info(
                f"[{self.NAME}] Fill-aware SL cap | "
                f"SL ₹{old_sl:.1f}->₹{fill_based_sl:.1f} | "
                f"entry=₹{self._pos.entry_premium:.1f} | p={prob:.3f} rank={rank:.3f}"
            )

    async def on_candle(self, msg: Message) -> None:
        p, ts = msg.payload, self._res_ts(msg.payload.get("timestamp"))
        candle_sym = self._resolve_symbol_from_payload(p)
        slot = self._slots.get(candle_sym)
        if not slot or not slot.pos or not slot.pos.is_open:
            if not (self.__pos and self.__pos.is_open and self._canonical_sym(getattr(self._pos.plan, "symbol", "")) == candle_sym):
                return
        self._activate_sym(candle_sym)
        try:
            await self._on_candle_inner(msg, p, ts)
        finally:
            self._sync_active_sym()

    async def _on_candle_inner(self, msg: Message, p: dict, ts: datetime) -> None:
        ltps = p.get("ltps") or {}
        pos_sym = getattr(self._pos.plan, "symbol", "") or self._active_sym or ""
        slot_spot = ltps.get(pos_sym) or ltps.get(self._canonical_sym(pos_sym))
        nifty = float(slot_spot or p.get("ltp") or p.get("close", 0) or 0)
        profile = self._management_profile()
        self._candles_held += 1
        use_spot_sl = self._pos.execution_mode != "BACKTEST" or _backtest_spot_sl_enabled()
        spot_sl_breached = self._spot_sl_breached(nifty) if use_spot_sl else False

        # Backtest hard-stop check uses candle extremes before close-level logic.
        # Estimated option paths use this by default so paper losses are not
        # understated versus live option stops; disable only for diagnostics via
        # BACKTEST_ENABLE_ESTIMATED_INTRABAR_SL=false.
        if self._pos.execution_mode == "BACKTEST" and "low" in p:
            worst = float(p["low"]) if self._pos.plan.signal.direction == Direction.BUY_CALL else float(p["high"])
            worst_option = estimate_option_pnl(
                self._pos.entry_premium,
                (worst - self._entry_nifty_ltp) / max(self._entry_nifty_ltp, 1) * 100,
                self._pos.plan.signal.direction.value,
                underlying_spot=self._entry_nifty_ltp,
            )
            catastrophic_floor = self._premium_disaster_floor(profile)
            estimated_backtest_path = bool(self._pos.is_simulated)
            if worst_option <= self._pos.sl_premium and (
                (
                    (not estimated_backtest_path or _backtest_estimated_intrabar_sl_enabled())
                    and
                    (not use_spot_sl or not self._spot_sl_enabled)
                    and self._candles_held > profile["decision_candles"]
                )
                or spot_sl_breached
                or worst_option <= catastrophic_floor
                or self._pos.breakeven_armed
            ):
                reason = "TRAILING_SL" if self._pos.breakeven_armed else ("SPOT_SL" if spot_sl_breached else "SL_HIT")
                await self._close(reason, self._pos.sl_premium, ts); return

        nifty_ltp = nifty
        option_sym = self._pos.plan.option_symbol

        use_data_ltp = bool(self.data_agent)
        current_ltp = 0.0
        if use_data_ltp:
            try:
                current_ltp = float(self.data_agent.get_option_ltp(option_sym) or 0.0)
            except Exception:
                current_ltp = 0.0
        is_futures = (
            getattr(self._pos.plan, "option_type", "") == "FUT"
            or getattr(self._pos.plan, "premium_source", "") == "COMMODITY_FUTURES"
            or (hasattr(self._pos.plan, "symbol") and self._pos.plan.symbol in ("SILVERMIC", "GOLD", "GOLDM", "CRUDEOIL", "CRUDEOILM", "NATURALGAS", "NATGASM"))
        )
        if is_futures:
            current_ltp = nifty_ltp if nifty_ltp > 0 else self._pos.entry_premium
        elif current_ltp <= 0:
            chg_pct = ((nifty_ltp - self._entry_nifty_ltp)
                       / max(self._entry_nifty_ltp, 1) * 100
                       if self._entry_nifty_ltp else 0)
            current_ltp = estimate_option_pnl(
                self._pos.entry_premium,
                chg_pct,
                self._pos.plan.signal.direction.value,
                underlying_spot=self._entry_nifty_ltp,
            )

        self._pos.update(current_ltp)
        pnl_pct = self._pos.pnl_pct
        logger.info(f"[{self.NAME}] CANDLE {ts.strftime('%H:%M')} | sym={option_sym} | nifty={nifty_ltp} | entry_nifty={self._entry_nifty_ltp} | ltp={current_ltp:.1f} | pnl={pnl_pct:.1f}%")
        self._record_trade_path(nifty_ltp, current_ltp)
        if pnl_pct > self._peak_pnl:
            self._peak_pnl = pnl_pct
        self._upd_exit(current_ltp, ts)

        await self._maybe_partial_exit(option_sym, current_ltp, pnl_pct, ts)
        if not self._pos or not self._pos.is_open:
            return

        trail_pct = _tiered_tsl_pct(self._peak_pnl, self._last_adx)
        if profile["runner"] and self._peak_pnl < 15.0:
            trail_pct = 0.0

        # Dynamic Profit Protection & Breakeven Lock on Momentum Bursts
        if profile.get("runner"):
            if trail_pct > 0 or self._peak_pnl >= 10.0:
                peak_premium = self._pos.entry_premium * (1 + self._peak_pnl / 100)
                effective_trail = trail_pct if trail_pct > 0 else 12.0
                new_tsl_floor = peak_premium * (1 - effective_trail / 100)

                # Tiered Progressive Profit Protection for Runners:
                if self._peak_pnl >= 10.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.005)
                if self._peak_pnl >= 12.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.04)
                if self._peak_pnl >= 15.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.08)
                if self._peak_pnl >= 20.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.12)
                if self._peak_pnl >= 25.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.16)
                if self._peak_pnl >= 35.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.22)
                if self._peak_pnl >= 50.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.35)
                if self._peak_pnl >= 75.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.55)
                if self._peak_pnl >= 100.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.75)

                # Dynamic peak giveback cap: never give back > 5.5% once peak >= 12%
                if self._peak_pnl >= 12.0:
                    new_tsl_floor = max(new_tsl_floor, peak_premium * 0.945)

                if new_tsl_floor > self._tsl_floor:
                    self._tsl_floor = new_tsl_floor
                    self._pos.sl_premium = max(self._pos.sl_premium, round(self._tsl_floor, 1))

                if not self._tsl_active:
                    self._tsl_active = True
                    logger.info(
                        f"[{self.NAME}] 🔒 RUNNER PROFIT PROTECTION ACTIVATED | "
                        f"peak={self._peak_pnl:.1f}% | "
                        f"trail={effective_trail:.1f}% | "
                        f"ADX={self._last_adx:.1f} | "
                        f"floor=₹{self._tsl_floor:.1f}"
                    )
        else:
            # Scalps / Non-Runners: Tight progressive profit lock
            if trail_pct > 0 or self._peak_pnl >= 5.0:
                peak_premium = self._pos.entry_premium * (1 + self._peak_pnl / 100)
                new_tsl_floor = peak_premium * (1 - trail_pct / 100) if trail_pct > 0 else self._pos.entry_premium * 1.002

                # 1. Momentum Burst (Peak >= 5.0%): Lock stop to Breakeven (+0.2%)
                if self._peak_pnl >= 5.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.002)
                # 2. Solid Expansion (Peak >= 10.0%): Lock at least +4.0% profit
                if self._peak_pnl >= 10.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.04)
                # 3. High Momentum (Peak >= 15.0%): Lock at least +8.0% profit
                if self._peak_pnl >= 15.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.08)
                # 4. Major Extension (Peak >= 20.0%): Lock at least +12.0% profit
                if self._peak_pnl >= 20.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.12)
                # 5. Massive Runner (Peak >= 30.0%): Lock at least +20.0% profit
                if self._peak_pnl >= 30.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.20)
                if self._peak_pnl >= 50.0:
                    new_tsl_floor = max(new_tsl_floor, self._pos.entry_premium * 1.35)

                if self._peak_pnl >= 10.0:
                    new_tsl_floor = max(new_tsl_floor, peak_premium * 0.95)

                if new_tsl_floor > self._tsl_floor:
                    self._tsl_floor = new_tsl_floor
                    self._pos.sl_premium = max(self._pos.sl_premium, round(self._tsl_floor, 1))

                if not self._tsl_active:
                    self._tsl_active = True
                    logger.info(
                        f"[{self.NAME}] 🔒 SCALP PROFIT PROTECTION ACTIVATED | "
                        f"peak={self._peak_pnl:.1f}% | "
                        f"trail={trail_pct:.1f}% | "
                        f"ADX={self._last_adx:.1f} | "
                        f"floor=₹{self._tsl_floor:.1f}"
                    )

        await self.bus.publish(Topic.POSITION_UPDATE, {
            "option_symbol": option_sym,
            "entry_premium": self._pos.entry_premium,
            "current_ltp": current_ltp,
            "pnl_pct": round(pnl_pct, 2),
            "peak_pnl": round(self._peak_pnl, 2),
            "tsl_active": self._tsl_active,
            "tsl_floor": round(self._tsl_floor, 1),
            "trail_pct": trail_pct,
            "candles_held": self._candles_held,
            "quantity": self._quantity,
            "initial_quantity": self._initial_quantity,
            "partial_quantity_closed": self._partial_quantity_closed,
            "partial_realized_pnl": round(self._partial_realized_pnl, 2),
            "partial_transaction_costs": round(self._partial_transaction_costs, 2),
            "profit_ladder": self._ladder.get_stats() if self._ladder else {},
            "spot_sl_enabled": self._spot_sl_enabled,
            "spot_sl_level": round(self._spot_sl_level, 1),
            "spot_sl_source": self._spot_sl_source,
            "spot_sl_breached": spot_sl_breached,
            "simulated": self._pos.is_simulated,
            "timestamp": datetime.now(IST).isoformat(),
        }, self.NAME)

        now = ts.strftime("%H:%M")
        mins = (ts - self._pos.entry_time).total_seconds() / 60

        if now >= MARKET_CLOSE_TIME or (self._pos.execution_mode == "BACKTEST" and now >= "15:25"):
            await self._close("EOD", current_ltp, ts); return
        if use_spot_sl and spot_sl_breached:
            logger.info(
                f"[{self.NAME}] Spot SL breached | spot={nifty:.1f} "
                f"level={self._spot_sl_level:.1f} source={self._spot_sl_source}"
            )
            await self._close("SPOT_SL", current_ltp, ts); return
        is_pos_long = getattr(self._pos, "is_long", True)
        sl_condition = (current_ltp <= self._pos.sl_premium) if is_pos_long else (current_ltp >= self._pos.sl_premium)
        if sl_condition:
            catastrophic_floor = self._premium_disaster_floor(profile)
            if (
                use_spot_sl
                and self._spot_sl_enabled
                and not spot_sl_breached
                and current_ltp > catastrophic_floor
                and not self._pos.breakeven_armed
            ):
                self._prev_pnl = pnl_pct
                return
            if (
                self._candles_held <= int(profile["decision_candles"])
                and current_ltp > catastrophic_floor
                and not self._pos.breakeven_armed
            ):
                self._prev_pnl = pnl_pct
                return
            exit_premium = self._pos.sl_premium if self._pos.execution_mode == "BACKTEST" else current_ltp
            await self._close("SL_HIT", exit_premium, ts)
            await self.bus.publish("SL_HIT_EVENT", {
                "timestamp": ts.isoformat(),
                "option_symbol": option_sym,
            }, self.NAME)
            return
        tsl_condition = (current_ltp < self._tsl_floor) if is_pos_long else (current_ltp > self._tsl_floor)
        if self._tsl_active and tsl_condition:
            await self._close("TRAILING_SL", current_ltp, ts); return
        is_single_lot = self._quantity <= max(int(self._pos.plan.lot_size or 1), 1)
        tgt_condition = (current_ltp >= self._pos.plan.target_premium) if is_pos_long else (current_ltp <= self._pos.plan.target_premium)
        if (not profile.get("runner", False) or is_single_lot) and tgt_condition:
            await self._close("TARGET_HIT", current_ltp, ts); return
        exit_signal = self._exit_signal_reason(pnl_pct=pnl_pct, nifty=nifty, mins=mins, profile=profile)
        if exit_signal:
            await self._close(exit_signal, current_ltp, ts); return
        # Trader Intuition: Don't cut if trade is in green or momentum is strong
        # Skip global stale exits for reduced budget lane & HeroZero — they exit via profile stale_minutes/SL only
        is_reduced_or_hero = bool(
            (self._pos.plan.signal.metadata or {}).get("reduced_budget_lane")
            or (self._pos.plan.signal.metadata or {}).get("reduced_budget_reason")
            or str((self._pos.plan.signal.metadata or {}).get("budget_lane", "")).upper() == "HERO_ZERO"
            or profile.get("is_hero_zero", False)
        )
        if not is_reduced_or_hero and not profile.get("runner", False):
            if self._candles_held >= STALE_EARLY_CHECK_CANDLES:
                if pnl_pct < STALE_EARLY_LOSS_THRESHOLD and self._last_momentum < STALE_MOMENTUM_THRESHOLD:
                    if self._change_of_character() or self._premium_erosion() or pnl_pct <= POS_MANAGER_PNL_PCT_THRESH_minus_5_5:
                        logger.warning(f"[{self.NAME}] ⚡ MOMENTUM_EXIT | pnl={pnl_pct:.1f}% | mom={self._last_momentum:.2f}")
                        await self._close("STALE_LOSS", current_ltp, ts); return

            if self._candles_held >= STALE_MAX_CANDLES:
                # If trade is in green (> 1%), give it more time (Trader perspective: don't choke a winner)
                if pnl_pct < POS_MANAGER_PNL_PCT_THRESH_0_5: 
                    # Cut only when the trade has actually lost structure or premium is eroding.
                    if (
                        self._last_momentum < POS_MANAGER_LAST_MOMENTUM_THRESH_0_5
                        and (self._change_of_character() or self._premium_erosion())
                    ) or (
                        self._candles_held >= STALE_MAX_CANDLES + 5
                        and self._premium_erosion()
                    ):
                        await self._close("STALE_LOSS", current_ltp, ts); return

        time_stop_minutes = int(self._pos.plan.time_stop_minutes + profile["time_extension_minutes"])
        time_stop_floor = float(self._pos.plan.time_stop_min_pnl_pct + profile["time_floor_offset"])
        if profile["runner"]:
            time_stop_floor = min(time_stop_floor, 0.75)
        if now >= EOD_SQUARE_OFF_TIME:
            if pnl_pct <= POS_MANAGER_PNL_PCT_THRESH_0_0: await self._close("EOD_LOSS", current_ltp, ts); return
            if pnl_pct < self._prev_pnl - 1.0: await self._close("EOD_BOOK", current_ltp, ts); return
        if mins >= time_stop_minutes and pnl_pct < time_stop_floor:
            # Time alone is not an exit. Require CoC or premium erosion.
            if (pnl_pct < POS_MANAGER_PNL_PCT_THRESH_0_0 or self._last_momentum < POS_MANAGER_LAST_MOMENTUM_THRESH_0_45) and (
                self._change_of_character() or self._premium_erosion()
            ):
                await self._close("TIME_DECAY", current_ltp, ts); return
        if mins > profile["stale_minutes"] and pnl_pct < profile["stale_loss_pct"]:
            nifty_change = (nifty - self._entry_nifty_ltp) / max(self._entry_nifty_ltp, 1e-9) * 100
            is_against = (self._pos.plan.signal.direction == Direction.BUY_CALL and nifty_change < -0.05) or (
                self._pos.plan.signal.direction == Direction.BUY_PUT and nifty_change > 0.05
            )
            stale_cut = -12.0 if profile["runner"] else -8.0
            coc = self._change_of_character()
            erosion = self._premium_erosion()
            severe_loss = pnl_pct <= stale_cut - 2.5
            if (is_against and (coc or self._last_momentum < POS_MANAGER_LAST_MOMENTUM_THRESH_0_5)) or (pnl_pct <= stale_cut and (coc or erosion or severe_loss)):
                await self._close("STALE_LOSS", current_ltp, ts); return
        metadata = (self._pos.plan.signal.metadata or {}) if self._pos.plan and self._pos.plan.signal else {}
        is_reduced = bool(metadata.get("reduced_budget_lane", False)) or bool(metadata.get("reduced_budget_reason"))
        if mins > profile["flat_minutes"] and profile["flat_low_pct"] < pnl_pct < profile["flat_high_pct"]:
            if is_reduced or (not profile["runner"] and self._premium_erosion()):
                await self._close("TIME_DECAY", current_ltp, ts); return

        self._prev_pnl = pnl_pct

    def _spot_sl_breached(self, nifty_spot: float) -> bool:
        if not self._pos or not self._spot_sl_enabled or self._spot_sl_level <= 0:
            return False
        spot = float(nifty_spot or 0.0)
        if spot <= 0:
            return False
        self._last_nifty_spot = spot
        return get_spot_sl_detector().is_sl_breached(
            current_spot=spot,
            sl_level=self._spot_sl_level,
            direction=self._pos.plan.signal.direction.value,
            require_close=True,
        )

    def _premium_disaster_floor(self, profile: dict) -> float:
        if not self._pos:
            return 0.0
        if profile.get("is_hero_zero"):
            floor_pct = 0.50  # 50% max catastrophic floor for HeroZero 0DTE trades
        elif self._spot_sl_enabled:
            # When spot SL is active, spot structure governs normal exit.
            # Catastrophic floor is only a backstop for extreme black swan / spot feed failure.
            floor_pct = 0.60 if profile.get("runner") else 0.65
        else:
            floor_pct = 0.65 if profile.get("runner") else 0.70
        return self._pos.entry_premium * floor_pct

    def _paper_exit_premium(self, option_sym: str, fallback_ltp: float, ts: datetime) -> float:
        if (
            not self._pos
            or not self._pos.is_simulated
            or self._pos.execution_mode == "BACKTEST"
            or not self.data_agent
        ):
            return fallback_ltp
        getter = getattr(self.data_agent, "get_option_candle_close", None)
        if not callable(getter):
            return fallback_ltp
        try:
            candle_close = float(getter(option_sym, ts, interval="1minute") or 0.0)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] option candle close lookup failed: {exc}")
            return fallback_ltp
        if candle_close <= 0:
            return fallback_ltp
        if abs(candle_close - fallback_ltp) >= 0.05:
            logger.info(
                f"[{self.NAME}] Paper exit aligned to option candle close | "
                f"{option_sym} quote=₹{fallback_ltp:.2f} close=₹{candle_close:.2f} "
                f"ts={ts.isoformat()}"
            )
        return candle_close

    def _intrabar_option_ltp(self, nifty_ltp: float) -> float:
        if not self._pos:
            return 0.0
        option_sym = self._pos.plan.option_symbol
        current_ltp = 0.0
        try:
            if self.data_agent and callable(getattr(self.data_agent, "get_option_ltp", None)):
                current_ltp = float(self.data_agent.get_option_ltp(option_sym) or 0.0)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] intrabar option LTP via data_agent failed: {exc}")
        if current_ltp <= 0:
            try:
                current_ltp = float(self.broker.get_option_ltp(option_sym) or 0.0)
            except Exception as exc:
                logger.debug(f"[{self.NAME}] intrabar option LTP via broker failed: {exc}")
        if current_ltp > 0:
            return current_ltp
        if self._entry_nifty_ltp > 0:
            chg_pct = (nifty_ltp - self._entry_nifty_ltp) / max(self._entry_nifty_ltp, 1.0) * 100
            return estimate_option_pnl(
                self._pos.entry_premium,
                chg_pct,
                self._pos.plan.signal.direction.value,
                underlying_spot=self._entry_nifty_ltp,
            )
        return float(self._pos.current_premium or self._pos.entry_premium or 0.0)

    def _intrabar_hit(self, key: str, hit: bool) -> bool:
        if not hit:
            self._intrabar_hit_counts.pop(key, None)
            return False
        count = int(self._intrabar_hit_counts.get(key, 0)) + 1
        self._intrabar_hit_counts[key] = count
        return count >= 2

    async def _maybe_intrabar_exit(self, nifty_ltp: float, ts: datetime) -> None:
        if not self._pos or not self._pos.is_open:
            return
        profile = self._management_profile()
        current_ltp = float(self._intrabar_option_ltp(nifty_ltp) or 0.0)
        if current_ltp <= 0:
            return

        self._pos.update(current_ltp)
        pnl_pct = self._pos.pnl_pct
        if self._pos.execution_mode != "BACKTEST" and pnl_pct > self._peak_pnl:
            self._peak_pnl = pnl_pct
            # Real-time Momentum Breakeven Lock & Giveback Protection on Intrabar Surges
            be_surge_threshold = float(profile.get("breakeven_trigger_pct", 10.0))
            if profile.get("fragile_experimental"):
                be_surge_threshold = min(be_surge_threshold, 8.0)
            if self._peak_pnl >= be_surge_threshold:
                new_floor = max(self._tsl_floor, self._pos.entry_premium * 1.005)
                if self._peak_pnl >= 12.0:
                    new_floor = max(new_floor, self._pos.entry_premium * 1.04)
                if self._peak_pnl >= 15.0:
                    new_floor = max(new_floor, self._pos.entry_premium * 1.08)
                if self._peak_pnl >= 20.0:
                    new_floor = max(new_floor, self._pos.entry_premium * 1.12)
                if self._peak_pnl >= 25.0:
                    new_floor = max(new_floor, self._pos.entry_premium * 1.16)
                if self._peak_pnl >= 35.0:
                    new_floor = max(new_floor, self._pos.entry_premium * 1.22)
                if self._peak_pnl >= 50.0:
                    new_floor = max(new_floor, self._pos.entry_premium * 1.35)

                # Dynamic peak giveback cap: once peak >= 12%, do not give back more than giveback_cap_pct from peak
                if self._peak_pnl >= 12.0:
                    giveback_pct = float(profile.get("giveback_cap_pct", 5.5))
                    giveback_mult = 1.0 - (giveback_pct / 100.0)
                    peak_prem = self._pos.entry_premium * (1.0 + self._peak_pnl / 100.0)
                    new_floor = max(new_floor, peak_prem * giveback_mult)

                if new_floor > self._tsl_floor:
                    self._tsl_floor = new_floor
                    self._pos.sl_premium = max(self._pos.sl_premium, round(self._tsl_floor, 1))
                if not self._tsl_active:
                    self._tsl_active = True
        if self._pos.execution_mode != "BACKTEST":
            self._upd_exit(current_ltp, ts)

        spot_sl_breached = self._spot_sl_breached(nifty_ltp)
        catastrophic_floor = self._premium_disaster_floor(profile)
        if spot_sl_breached:
            await self._close("SPOT_SL", current_ltp, ts)
            return
        if current_ltp <= catastrophic_floor:
            await self._close("SL_HIT", current_ltp, ts)
            return

        stop_buffer = max(self._pos.sl_premium * 0.002, 0.25)
        target_buffer = max(self._pos.plan.target_premium * 0.002, 0.25)
        tsl_buffer = max(self._tsl_floor * 0.002, 0.25) if self._tsl_active else 0.0

        hard_sl_hit = current_ltp <= max(self._pos.sl_premium - stop_buffer, 0.0)
        target_hit = current_ltp >= self._pos.plan.target_premium + target_buffer
        trailing_hit = self._tsl_active and current_ltp <= max(self._tsl_floor - tsl_buffer, 0.0)

        if self._intrabar_hit("hard_sl", hard_sl_hit):
            await self._close("SL_HIT", current_ltp, ts)
            return
        is_single_lot = self._quantity <= max(int(self._pos.plan.lot_size or 1), 1)
        if (not profile.get("runner", False) or is_single_lot) and self._intrabar_hit("target", target_hit):
            await self._close("TARGET_HIT", current_ltp, ts)
            return
        if self._intrabar_hit("tsl", trailing_hit):
            await self._close("TRAILING_SL", current_ltp, ts)
            return

        # Real-time Intrabar Partial Profit Booking (T1 / T2 ladder) on live ticks
        if self._pos and self._pos.is_open and self._ladder and self._quantity > self._pos.plan.lot_size:
            await self._maybe_partial_exit(self._pos.plan.option_symbol, current_ltp, pnl_pct, ts)

    def _upd_exit(self, ltp: float, ts: datetime) -> None:
        entry, stop = self._pos.entry_premium, self._pos.plan.stop_distance
        if stop <= 0: return
        pnl = self._pos.pnl_pct
        profile = self._management_profile()
        metadata = (self._pos.plan.signal.metadata or {}) if self._pos.plan and self._pos.plan.signal else {}
        is_reduced = bool(metadata.get("reduced_budget_lane", False)) or bool(metadata.get("reduced_budget_reason"))

        # ── TIGHT INTRABAR SCALP TRAILING FOR REDUCED BUDGET LANE ─────────────
        if is_reduced:
            # Scalp target is ~18%. Lock in profits progressively once +10% is reached:
            # 1. At +10.0% gain -> Lock in +4.0% profit (guaranteed green)
            # 2. At +12.5% gain -> Lock in +8.0% profit
            # 3. At +15.0% gain -> Lock in +11.5% profit
            if pnl >= 15.0:
                self._pos.breakeven_armed = True
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.115, 1))
            elif pnl >= 12.5:
                self._pos.breakeven_armed = True
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.08, 1))
            elif pnl >= 10.0:
                self._pos.breakeven_armed = True
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.04, 1))
            elif pnl >= 6.5 and not self._pos.breakeven_armed:
                # Early break-even protection for scalps
                self._pos.breakeven_armed = True
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.005, 1))

            # Dynamic peak trail for scalps: if peak >= +12%, never let price drop >4.5% below peak
            if self._peak_pnl >= 12.0:
                peak_trail_sl = round(self._pos.peak_premium * 0.955, 1)
                self._pos.sl_premium = max(self._pos.sl_premium, peak_trail_sl)
            return

        # Profit floors are intentionally conservative. Previous logic armed
        # from +6% and then trailed off tiny fill-aware stop distances, turning
        # normal pullbacks into PROFIT_PROTECT before trades could reach target.
        breakeven_trigger = float(profile.get("breakeven_trigger_pct", 10.0))
        if profile.get("fragile_experimental"):
            breakeven_trigger = min(breakeven_trigger, 7.5)
        if pnl >= breakeven_trigger and not self._pos.breakeven_armed:
            self._pos.breakeven_armed = True
            lock_pct = 1.002 if profile.get("fragile_experimental") else 1.005
            self._pos.sl_premium = max(self._pos.sl_premium, round(entry * lock_pct, 1))

        if profile.get("is_hero_zero"):
            # Multiplier trailing ladder for HeroZero
            if pnl >= 50.0:
                self._pos.breakeven_armed = True
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.10, 1))
            if pnl >= 100.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.50, 1))
            if pnl >= 200.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 2.20, 1))
            if pnl >= 300.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 2.80, 1))
            if pnl >= 400.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 3.50, 1))
        elif profile["runner"]:
            if pnl >= 12.0:
                self._pos.breakeven_armed = True
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.04, 1))
            if pnl >= 15.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.08, 1))
            if pnl >= 20.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.12, 1))
            if pnl >= 25.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.16, 1))
            if pnl >= 35.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.22, 1))
            if pnl >= 50.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.35, 1))
            if pnl >= 65.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.45, 1))
            if pnl >= 95.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.65, 1))
        else:
            if pnl >= 10.0:
                self._pos.breakeven_armed = True
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.04, 1))
            if pnl >= 15.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.08, 1))
            if pnl >= 20.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.12, 1))
            if pnl >= 28.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.18, 1))
            if pnl >= 42.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.28, 1))
            if pnl >= 65.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.45, 1))
            if pnl >= 95.0:
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.65, 1))

        # Dynamic peak trail for all normal/runner positions: if peak >= +12%, never give back > giveback_cap_pct from peak
        if self._peak_pnl >= 12.0:
            giveback_pct = float(profile.get("giveback_cap_pct", 5.5))
            giveback_mult = 1.0 - (giveback_pct / 100.0)
            peak_trail_sl = round(self._pos.peak_premium * giveback_mult, 1)
            self._pos.sl_premium = max(self._pos.sl_premium, peak_trail_sl)

        if not self._pos.breakeven_armed:
            r = (ltp - entry) / max(stop, 0.1)
            # Only cut risk if r >= 1.0 (trade has proved direction with 1R gain), locking breakeven
            if r >= 1.0:
                self._pos.breakeven_armed = True
                self._pos.sl_premium = max(self._pos.sl_premium, round(entry * 1.005, 1))
        elif self._peak_pnl >= (32.0 if profile["runner"] else 28.0):
            pnl_val = self._pos.pnl_pct
            base = stop * (1.50 if profile["runner"] else 1.25)
            if profile["runner"]:
                trail_mult = 3.5 if pnl_val >= 55 else 2.8
            else:
                trail_mult = 1.7 if pnl_val >= 45 else 2.2
            self._pos.sl_premium = max(self._pos.sl_premium, round(self._pos.peak_premium - base * trail_mult, 1))

    def _enrich_profile(self, profile: dict) -> dict:
        plan = self._pos.plan if self._pos else None
        sym = getattr(plan, "symbol", "") or self._active_sym or os.getenv("INSTRUMENT", "CRUDEOILM")
        inst_cfg = get_instrument_strategy_config(str(sym))
        profile["giveback_cap_pct"] = float(inst_cfg.get("giveback_cap_pct", 5.5))
        profile["breakeven_trigger_pct"] = float(inst_cfg.get("breakeven_trigger_pct", 10.0))
        profile["target2_pct"] = float(inst_cfg.get("target2_pct", 22.0))
        profile["stop_loss_pct"] = float(inst_cfg.get("stop_loss_pct", 10.0))
        profile["inst_cfg"] = inst_cfg
        return profile

    def _management_profile(self) -> dict:
        plan = self._pos.plan
        metadata = plan.signal.metadata or {}
        is_hero_zero = (
            bool(metadata.get("hero_zero_lane", False))
            or str(metadata.get("budget_lane", "")).upper() == "HERO_ZERO"
            or "HeroZero" in {str(s).strip() for s in (plan.signal.strategies_fired or [])}
        )
        if is_hero_zero:
            return self._enrich_profile({
                "runner": True,
                "fragile_experimental": False,
                "is_hero_zero": True,
                "decision_candles": 3,
                "stale_minutes": 35,             # 35-minute hold window for 0DTE expiry breakout
                "stale_loss_pct": -25.0,         # 25% SL tolerance for HeroZero options
                "flat_minutes": 40,
                "flat_low_pct": -5.0,
                "flat_high_pct": 5.0,
                "time_extension_minutes": 20,
                "time_floor_offset": -2.0,
            })
        is_reduced = (
            self._quantity <= max(int(plan.lot_size or NIFTY_LOT_SIZE), 1)
            and (
                bool(metadata.get("reduced_budget_lane", False))
                or str(metadata.get("reduced_budget_reason", "")).startswith("rb_")
            )
        )
        if is_reduced and int(plan.signal.votes or 0) < 6:
            return self._enrich_profile({
                "runner": False,
                "fragile_experimental": False,
                "decision_candles": 3,
                "stale_minutes": 50,             # Allow 50-minute window to hit target
                "stale_loss_pct": -10.0,         # 10% SL already set by planner
                "flat_minutes": 65,              # 65-minute flat exit (prevents premature cutoffs)
                "flat_low_pct": -2.0,
                "flat_high_pct": 2.0,
                "time_extension_minutes": 10,
                "time_floor_offset": 0.25,
            })
        setup = ((metadata.get("_context") or {}).get("setup") or {})
        setup_type = str(setup.get("setup_type", "") or "").lower()
        setup_strength = float(setup.get("setup_strength", 0.0) or 0.0)
        strategies = {str(s).strip() for s in plan.signal.strategies_fired if str(s).strip()}
        rank = float(plan.ml_rank_score or plan.signal.ml_rank_score or 0.0)
        direction = plan.signal.direction.value
        votes = int(plan.signal.votes or 0)
        structure_bias = str((metadata.get("structure_bias") or setup.get("structure_bias") or "")).upper()
        if not structure_bias:
            context = setup.get("context") or {}
            structure_state = ((context.get("market_structure") or {}).get("structure_state") or {})
            structure_bias = str(structure_state.get("bias", "") or "").upper()
        aligned_bias = (
            (direction == "BUY_CALL" and structure_bias == "BULLISH")
            or (direction == "BUY_PUT" and structure_bias == "BEARISH")
        )
        experimental = {"ValueArea", "ADXRising", "RangeSpread", "StrikeMomentum", "GapMomentum"}
        legacy_structure = {"FVG", "ORB", "Ichimoku", "OIAnalysis", "CPR", "VolumeProfile"}
        fragile_experimental = bool(strategies & experimental) and not bool(strategies & legacy_structure)

        negative_edge = (
            setup_type == "breakout"
            and direction == "BUY_PUT"
            and "SkewHunter" in strategies
            and (rank < POS_MANAGER_RANK_THRESH_0_62 and setup_strength < 0.74)
        )
        skew_adx_runner = (
            direction == "BUY_PUT"
            and {"SuperTrend+RSI", "ADX+PSAR", "SkewHunter"}.issubset(strategies)
            and setup_type == "vote_aligned"
            and rank >= POS_MANAGER_RANK_THRESH_0_58
            and setup_strength >= 0.80
            and votes >= 3
        )
        call_displacement_runner = (
            direction == "BUY_CALL"
            and setup_type == "vote_aligned"
            and {"FVG", "OIAnalysis"}.issubset(strategies)
            and votes >= 6
            and rank >= POS_MANAGER_RANK_THRESH_0_62
            and setup_strength >= 0.84
        )
        breakout_runner = (
            setup_type == "breakout"
            and setup_strength >= 0.72
            and rank >= POS_MANAGER_RANK_THRESH_0_52
            and {"BBSqueeze", "FVG"}.issubset(strategies)
            and "SkewHunter" not in strategies
        )
        anchor_consensus_runner = (
            votes >= 6
            and setup_type in {"vote_aligned", "trend_pullback", "breakout"}
            and bool(strategies.intersection({"VolumeProfile", "FVG", "ElliottWave", "StrikeMomentum", "RangeSpread", "SkewHunter", "BBSqueeze"}))
        )
        runner = (
            not negative_edge
            and (aligned_bias or skew_adx_runner or call_displacement_runner or breakout_runner or anchor_consensus_runner)
            and setup_type in {"vote_aligned", "trend_pullback", "breakout"}
            and (
                skew_adx_runner
                or call_displacement_runner
                or breakout_runner
                or anchor_consensus_runner
                or (votes >= 5 and rank >= POS_MANAGER_RANK_THRESH_0_6)
                or (rank >= 0.70 and setup_strength >= 0.70)
                or (rank >= POS_MANAGER_RANK_THRESH_0_66 and setup_strength >= 0.86 and votes >= 3)
                or (rank >= POS_MANAGER_RANK_THRESH_0_63 and setup_strength >= 0.82 and votes >= 4 and "VWAP+EMA" in strategies)
            )
        )
        lots_held = max(
            int(getattr(plan, "desired_lots", 1) or 1),
            int(getattr(plan, "lots", 1) or 1),
            int(metadata.get("lots", 1) or 1),
            int(self._quantity // max(int(getattr(plan, "lot_size", NIFTY_LOT_SIZE) or NIFTY_LOT_SIZE), 1)),
        )
        if not negative_edge and lots_held >= 2:
            runner = True

        if negative_edge:
            return self._enrich_profile({
                "runner": False,
                "fragile_experimental": True,
                "decision_candles": 3,
                "stale_minutes": 15,
                "stale_loss_pct": -5.0,
                "flat_minutes": 30,
                "flat_low_pct": -0.50,
                "flat_high_pct": 0.75,
                "time_extension_minutes": -25,
                "time_floor_offset": 0.75,
            })
        if runner:
            return self._enrich_profile({
                "runner": True,
                "fragile_experimental": False,
                "decision_candles": 4,
                "stale_minutes": 25,
                "stale_loss_pct": -8.0,
                "flat_minutes": 45,
                "flat_low_pct": -0.5,
                "flat_high_pct": 0.75,
                "time_extension_minutes": 25,
                "time_floor_offset": -1.0,
            })
        return self._enrich_profile({
            "runner": False,
            "fragile_experimental": fragile_experimental,
            "decision_candles": 3,
            "stale_minutes": 10 if fragile_experimental else 15,
            "stale_loss_pct": -5.0 if fragile_experimental else -7.0,
            "flat_minutes": 15 if fragile_experimental else 30,
            "flat_low_pct": -0.50,
            "flat_high_pct": 1.25 if fragile_experimental else 1.0,
            "time_extension_minutes": -25 if fragile_experimental else -10,
            "time_floor_offset": 0.75 if fragile_experimental else 0.35,
        })

    def _exit_signal_reason(self, *, pnl_pct: float, nifty: float, mins: float, profile: dict) -> str:
        if not self._pos:
            return ""

        entry = max(self._entry_nifty_ltp, 1e-9)
        nifty_change = (nifty - entry) / entry * 100
        direction = self._pos.plan.signal.direction
        against = (
            (direction == Direction.BUY_CALL and nifty_change < -0.05)
            or (direction == Direction.BUY_PUT and nifty_change > 0.05)
        )
        favorable = (
            (direction == Direction.BUY_CALL and nifty_change > 0.05)
            or (direction == Direction.BUY_PUT and nifty_change < -0.05)
        )
        coc = self._change_of_character()
        erosion = self._premium_erosion()

        # Profit protection is a late-stage exit, not the primary target
        # substitute. Earlier thresholds capped most winners below +20%.
        if profile["runner"]:
            if self._peak_pnl >= 45.0 and pnl_pct <= max(18.0, self._peak_pnl * 0.50) and (coc or erosion):
                return "PROFIT_PROTECT"
            if self._peak_pnl >= 75.0 and pnl_pct <= self._peak_pnl * 0.58:
                return "PROFIT_PROTECT"
            return ""
        if self._peak_pnl >= 30.0 and pnl_pct <= max(8.0, self._peak_pnl * 0.42) and (coc and erosion):
            return "PROFIT_PROTECT"
        if self._peak_pnl >= 45.0 and pnl_pct <= max(14.0, self._peak_pnl * 0.55) and (coc or erosion):
            return "PROFIT_PROTECT"
        if self._peak_pnl >= 70.0 and pnl_pct <= self._peak_pnl * 0.62:
            return "PROFIT_PROTECT"

        # Do not classify the first few candles as stale. After the decision
        # window, exit only when price is against the signal or momentum is weak.
        if self._candles_held <= int(profile["decision_candles"]):
            return ""
        if profile.get("fragile_experimental") and not profile["runner"]:
            if pnl_pct <= POS_MANAGER_PNL_PCT_THRESH_minus_1_1 and (against or coc or erosion or self._peak_pnl < 3.0):
                return "SELL_SIGNAL"
            if mins >= 15 and self._peak_pnl < 5.0 and pnl_pct < POS_MANAGER_PNL_PCT_THRESH_1_0:
                return "NO_FOLLOW_THROUGH"
        if not profile.get("runner", False):
            if pnl_pct <= POS_MANAGER_PNL_PCT_THRESH_minus_2_2 and (coc or (against and self._last_momentum < POS_MANAGER_LAST_MOMENTUM_THRESH_0_68)):
                return "SELL_SIGNAL"
            if mins >= 25 and pnl_pct < POS_MANAGER_PNL_PCT_THRESH_minus_1_2 and (coc or erosion) and not favorable:
                return "SELL_SIGNAL"
        if mins >= 45 and pnl_pct < POS_MANAGER_PNL_PCT_THRESH_0_5 and self._peak_pnl < 6.0 and erosion:
            # Skip this global exit for reduced_budget_lane — the profile's stale_minutes/SL handles it
            if not ((self._pos.plan.signal.metadata or {}).get("reduced_budget_lane")
                    or (self._pos.plan.signal.metadata or {}).get("reduced_budget_reason")):
                return "NO_FOLLOW_THROUGH"
        return ""

    def _record_trade_path(self, nifty_ltp: float, option_ltp: float) -> None:
        self._nifty_closes.append(float(nifty_ltp or 0.0))
        self._premium_closes.append(float(option_ltp or 0.0))
        if len(self._nifty_closes) > 12:
            self._nifty_closes = self._nifty_closes[-12:]
        if len(self._premium_closes) > 12:
            self._premium_closes = self._premium_closes[-12:]

    def _change_of_character(self) -> bool:
        if not self._pos or len(self._nifty_closes) < 4:
            return False
        a, b, c = self._nifty_closes[-3:]
        entry = max(self._entry_nifty_ltp, 1e-9)
        change = (c - entry) / entry * 100
        if self._pos.plan.signal.direction == Direction.BUY_CALL:
            return (a > b > c) or change <= -0.10
        return (a < b < c) or change >= 0.10

    def _premium_erosion(self) -> bool:
        if not self._pos or len(self._premium_closes) < 4 or len(self._nifty_closes) < 4:
            return False
        direction = self._pos.plan.signal.direction
        entry_spot = max(self._entry_nifty_ltp, 1e-9)
        spot = self._nifty_closes[-1]
        spot_change = (spot - entry_spot) / entry_spot * 100
        favorable_spot = (
            (direction == Direction.BUY_CALL and spot_change >= 0.06)
            or (direction == Direction.BUY_PUT and spot_change <= -0.06)
        )
        premium_pnl = (self._premium_closes[-1] - self._pos.entry_premium) / max(self._pos.entry_premium, 1e-9) * 100
        recent_peak = max(self._premium_closes[-4:-1])
        premium_rollover = self._premium_closes[-1] < recent_peak * 0.975
        return (favorable_spot and premium_pnl < 1.0) or (premium_rollover and premium_pnl < 0.5)

    def _peak_pnl_pct(self) -> float:
        if not self._pos or self._pos.entry_premium <= 0: return 0.0
        return round((self._pos.peak_premium - self._pos.entry_premium) / self._pos.entry_premium * 100.0, 2)

    async def _maybe_partial_exit(self, option_sym: str, current_ltp: float, pnl_pct: float, ts: datetime) -> None:
        if not self._pos or not self._ladder or self._quantity <= self._pos.plan.lot_size:
            return
        action = self._ladder.check(current_ltp)
        if not action.book_now:
            return
        result = await self.partial_exit_executor.partial_exit(
            symbol=option_sym,
            lots_to_close=action.lots_to_book,
            lot_size=self._pos.plan.lot_size,
            current_premium=current_ltp,
            simulated=self._pos.is_simulated,
            reason=action.reason,
        )
        if not result.success:
            await self.bus.publish(Topic.ALERT, {
                "type": "partial_exit_failed",
                "severity": "WARNING",
                "option_symbol": option_sym,
                "reason": result.reason,
                "timestamp": ts.isoformat(),
            }, self.NAME)
            return

        self._ladder.record_booking(action, result.fill_price)
        self._quantity = max(self._quantity - result.quantity, 0)
        self._partial_quantity_closed += result.quantity
        partial_costs = 0.0
        if self._pos.execution_mode == "BACKTEST":
            partial_costs = estimate_round_trip_costs(
                self._pos.entry_premium,
                result.fill_price,
                result.quantity,
                BACKTEST_BROKERAGE_PER_ORDER,
                BACKTEST_TRANSACTION_COST_PCT,
            )
        self._partial_transaction_costs += partial_costs
        self._partial_realized_pnl += round((result.fill_price - self._pos.entry_premium) * result.quantity - partial_costs, 2)
        await self.bus.publish(Topic.POSITION_UPDATE, {
            "type": "partial_exit",
            "option_symbol": option_sym,
            "exit_premium": result.fill_price,
            "entry_premium": self._pos.entry_premium,
            "pnl_pct": round(pnl_pct, 2),
            "level": action.level,
            "lots_closed": action.lots_to_book,
            "quantity_closed": result.quantity,
            "quantity_remaining": self._quantity,
            "partial_realized_pnl": round(self._partial_realized_pnl, 2),
            "partial_transaction_costs": round(self._partial_transaction_costs, 2),
            "profit_ladder": self._ladder.get_stats(),
            "order_id": result.order_id,
            "account_results": result.account_results or [],
            "simulated": result.simulated,
            "timestamp": ts.isoformat(),
        }, self.NAME)

    async def _close(self, reason: str, ltp: float, ts: Optional[datetime] = None) -> None:
        if not self._pos: return
        closing_pos = self._pos
        closing_quantity = self._quantity
        closing_initial_qty = self._initial_quantity
        closing_ladder = self._ladder
        closing_partial_pnl = self._partial_realized_pnl
        closing_partial_costs = self._partial_transaction_costs
        closing_partial_qty_closed = self._partial_quantity_closed
        closing_peak_pnl = self._peak_pnl
        closing_candles_held = self._candles_held
        closing_tsl_active = self._tsl_active
        closing_sym = self._active_sym or self._resolve_symbol_from_payload({"option_symbol": closing_pos.plan.option_symbol if closing_pos and closing_pos.plan else ""})

        # Mark position as None immediately so subsequent events see position closed
        self._pos = None
        if closing_sym in self._slots:
            self._slots.pop(closing_sym, None)
        self._active_sym = ""

        logger.info(f"[{self.NAME}] CLOSING POSITION | reason={reason} | sym={closing_pos.plan.option_symbol if closing_pos and closing_pos.plan else 'NA'} | ltp={ltp} | ts={ts}")
        # Use wall-clock time for real trades to match broker/audit logs.
        # Simulated paper exits keep the candle timestamp so journaled prices
        # can align to the completed option candle close.
        # Use candle/passed ts for backtests to maintain consistency.
        if closing_pos.execution_mode != "BACKTEST" and not closing_pos.is_simulated:
            ts = datetime.now(IST)
        else:
            ts = ts or datetime.now(IST)

        if closing_pos.execution_mode == "BACKTEST":
            ltp = apply_option_slippage(ltp, "SELL", BACKTEST_OPTION_SLIPPAGE_PCT)
            costs = estimate_round_trip_costs(closing_pos.entry_premium, ltp, closing_quantity, BACKTEST_BROKERAGE_PER_ORDER, BACKTEST_TRANSACTION_COST_PCT)
        else: costs = 0.0
        is_short = bool(closing_pos.is_short)
        tick_val = getattr(closing_pos.plan, "tick_value", 1.0) or 1.0
        pnl_pts = (closing_pos.entry_premium - ltp) if is_short else (ltp - closing_pos.entry_premium)
        remaining_realized = pnl_pts * closing_quantity * tick_val
        realized_pnl = round(closing_partial_pnl + remaining_realized - costs, 2)
        total_qty = max(closing_initial_qty, closing_quantity + closing_partial_qty_closed, 1)
        pnl = (pnl_pts / max(closing_pos.entry_premium, 1.0)) * 100
        if reason == "SL_HIT" and pnl > 0 and getattr(self, "_tsl_active", False):
            reason = "PROFIT_PROTECT"
        trade_id = closing_pos.plan.option_symbol
        ltp = self._paper_exit_premium(trade_id, ltp, ts)
        pnl_pts = (closing_pos.entry_premium - ltp) if is_short else (ltp - closing_pos.entry_premium)
        remaining_realized = pnl_pts * closing_quantity * tick_val
        realized_pnl = round(closing_partial_pnl + remaining_realized - costs, 2)
        pnl = (pnl_pts / max(closing_pos.entry_premium, 1.0)) * 100
        if closing_ladder:
            closing_ladder.record_final_close(ltp)
        ladder_summary = closing_ladder.bookings_summary if closing_ladder else []
        avg_exit_pct = closing_ladder.avg_exit_pct if closing_ladder else round(pnl, 2)
        exit_order_id = ""
        exit_account_results = []
        if not closing_pos.is_simulated and closing_quantity > 0:
            lots_to_close = max(int(closing_quantity / max(closing_pos.plan.lot_size, 1)), 1)
            exit_result = await self.partial_exit_executor.partial_exit(
                symbol=closing_pos.plan.option_symbol,
                lots_to_close=lots_to_close,
                lot_size=closing_pos.plan.lot_size,
                current_premium=ltp,
                simulated=False,
                reason=reason,
            )
            if not exit_result.success:
                self._pos = closing_pos
                await self.bus.publish(Topic.ALERT, {
                    "type": "exit_order_failed",
                    "severity": "ERROR",
                    "option_symbol": closing_pos.plan.option_symbol,
                    "reason": exit_result.reason,
                    "timestamp": ts.isoformat(),
                }, self.NAME)
                logger.error(
                    f"[{self.NAME}] Close aborted; broker SELL failed | "
                    f"{closing_pos.plan.option_symbol} | {exit_result.reason}"
                )
                try:
                    from utils.telegram_notifier import get_notifier
                    notifier = get_notifier()
                    fail_msg = (
                        f"🚨 *EXIT FAILED — MANUAL INTERVENTION REQUIRED*\n"
                        f"Symbol: `{closing_pos.plan.option_symbol}`\n"
                        f"Reason: `{exit_result.reason}`\n"
                        f"Action: Position kept open in tracker. Close manually in terminal!"
                    )
                    await notifier.send_text(fail_msg, target="LIVE", parse_mode="markdown")
                except Exception as err:
                    logger.warning(f"[{self.NAME}] Telegram exit fail alert failed: {err}")
                return
            ltp = exit_result.fill_price
            exit_order_id = exit_result.order_id
            exit_account_results = exit_result.account_results or []
        closing_pos.close(ltp, reason, ts)
        self.capital_manager.record_trade_closed(trade_id)
        self.order_manager.record_position_closed(trade_id)
        try:
            from utils.bot_trade_registry import mark_bot_trade_closed
            mark_bot_trade_closed(trade_id)
        except Exception:
            pass
        sig_dict = closing_pos.plan.signal.to_dict()
        sig_id = getattr(closing_pos.plan.signal, "signal_id", "") or sig_dict.get("signal_id", "")
        await self.bus.publish(Topic.POSITION_CLOSED, {
            "symbol":        closing_sym,
            "signal_id":     sig_id,
            "option_symbol": closing_pos.plan.option_symbol,
            "signal":        sig_dict,
            "direction":     closing_pos.plan.signal.direction.value,
            "exit_reason":   reason,
            "exit_premium":  ltp,
            "entry_premium": closing_pos.entry_premium,
            "pnl_pct":       round(pnl, 2),
            "realized_pnl":  realized_pnl,
            "pnl_inr":       realized_pnl,
            "transaction_costs": round(costs, 2),
            "lots":          max(int(total_qty / max(closing_pos.plan.lot_size, 1)), 1),
            "quantity":      total_qty,
            "final_quantity_closed": closing_quantity,
            "partial_quantity_closed": closing_partial_qty_closed,
            "partial_realized_pnl": round(closing_partial_pnl, 2),
            "partial_transaction_costs": round(closing_partial_costs, 2),
            "partial_bookings": ladder_summary,
            "exit_order_id": exit_order_id,
            "exit_account_results": exit_account_results,
            "avg_exit_pct": avg_exit_pct,
            "peak_pnl":      closing_peak_pnl,
            "candles_held":  closing_candles_held,
            "tsl_activated": closing_tsl_active,
            "simulated":     closing_pos.is_simulated,
            "entry_time":    closing_pos.entry_time.isoformat(),
            "exit_time":     closing_pos.exit_time.isoformat() if closing_pos.exit_time else "",
            "timestamp":     datetime.now(IST).isoformat(),
        }, self.NAME)

        # Only clear tracking fields if a new position wasn't opened while awaiting
        if self._pos is None or self._pos is closing_pos:
            self._pos          = None
            self._candles_held = 0
            self._peak_pnl     = 0.0
            self._tsl_active   = False
            self._quantity = 0
            self._initial_quantity = 0
            self._partial_realized_pnl = 0.0
            self._partial_transaction_costs = 0.0
            self._partial_quantity_closed = 0
            self._ladder = None
            self._spot_sl_level = 0.0
            self._spot_sl_source = ""
            self._spot_sl_enabled = False
            self._last_nifty_spot = 0.0
            self._df_buffer = []
            self._intrabar_hit_counts = {}

    def _res_ts(self, raw: Optional[str]) -> datetime:
        t = datetime.fromisoformat(str(raw)) if raw else datetime.now(IST)
        return t if t.tzinfo else IST.localize(t)

    def _build_plan(self, d: dict) -> TradePlan:
        s = d.get("signal", {})
        ml_rank_score = float(d.get("ml_rank_score", s.get("ml_rank_score", 0.0)) or 0.0)
        ml_confidence = float(d.get("ml_confidence", s.get("ml_confidence", d.get("confidence", 0.0))) or 0.0)
        metadata = dict(s.get("metadata", {}) or {})
        if d.get("structure_bias") and not metadata.get("structure_bias"):
            metadata["structure_bias"] = d.get("structure_bias")
        signal = RawSignal(
            symbol=str(s.get("symbol") or d.get("symbol") or os.getenv("INSTRUMENT", "SILVERM")),
            direction=Direction(s.get("direction", "NONE")),
            confidence=float(s.get("confidence", 0)),
            votes=int(s.get("votes", 0)),
            strategies_fired=s.get("strategies_fired", []),
            nifty_ltp=float(s.get("nifty_ltp", 0)),
            regime=Regime(s.get("regime", "TRENDING")),
            metadata=metadata,
            ml_rank_score=ml_rank_score,
            ml_rank_tier=str(d.get("ml_rank_tier", s.get("ml_rank_tier", "")) or ""),
            ml_decision_reason=str(d.get("ml_decision_reason", s.get("ml_decision_reason", "")) or ""),
        )
        return TradePlan(
            signal=signal,
            option_symbol=str(d.get("option_symbol", "")),
            strike=int(d.get("strike", 0)),
            option_type=str(d.get("option_type", "")),
            expiry_date=str(d.get("expiry_date", "")),
            days_to_expiry=int(d.get("days_to_expiry", 0)),
            est_premium=float(d.get("entry_premium", d.get("est_premium", 0)) or 0),
            sl_premium=float(d.get("sl_premium", 0)),
            target_premium=float(d.get("target_premium", 0)),
            lot_size=int(d.get("lot_size", NIFTY_LOT_SIZE)),
            quantity=int(d.get("quantity", 0)),
            desired_lots=int(d.get("desired_lots", d.get("lots", 1)) or 1),
            total_invested=float(d.get("total_invested", 0.0) or 0.0),
            ml_confidence=ml_confidence,
            ml_rank_score=ml_rank_score,
            ml_approved=bool(d.get("ml_approved", True)),
            premium_source=str(d.get("premium_source", "")),
            contract_score=float(d.get("contract_score", 0.0) or 0.0),
            contract_snapshot=d.get("contract_snapshot", {}) or {},
            atr_points=float(d.get("atr_points", 0)),
            stop_distance=float(d.get("stop_distance", 15.0)),
            target1_premium=float(d.get("target1_premium", 0)),
            target2_premium=float(d.get("target2_premium", 0)),
            breakeven_trigger_premium=float(d.get("breakeven_trigger_premium", 0)),
            trailing_stop_distance=float(d.get("trailing_stop_distance", 0)),
            time_stop_minutes=int(d.get("time_stop_minutes", 0)),
            time_stop_min_pnl_pct=float(d.get("time_stop_min_pnl_pct", 0)),
            risk_budget_inr=float(d.get("risk_budget_inr", 0.0) or 0.0),
            sl_spot_level=float(d.get("sl_spot_level", 0.0) or 0.0),
            sl_source=str(d.get("sl_source", "") or ""),
            sl_note=str(d.get("sl_note", "") or ""),
            sl_distance_pts=float(d.get("sl_distance_pts", 0.0) or 0.0),
            sl_structural_premium=float(d.get("sl_structural_premium", 0.0) or 0.0),
        )

    async def force_close_from_broker(self, reason: str = "MANUAL_BROKER_EXIT", exit_premium: float = 0.0) -> None:
        """Called by LivePositionReconciler when broker position is confirmed closed manually."""
        if not self._pos:
            return
        closing_pos = self._pos
        closing_quantity = self._quantity
        trade_id = closing_pos.plan.option_symbol
        logger.info(f"[{self.NAME}] ⚡ Syncing internal position close from broker state | sym={trade_id} | reason={reason}")
        current_ltp = exit_premium if exit_premium > 0 else (closing_pos.current_premium or closing_pos.entry_premium)
        self._pos = None
        self._quantity = 0
        ts = datetime.now(IST)
        is_short = bool(closing_pos.is_short)
        tick_val = getattr(closing_pos.plan, "tick_value", 1.0) or 1.0
        pnl_pts = (closing_pos.entry_premium - current_ltp) if is_short else (current_ltp - closing_pos.entry_premium)
        realized_pnl = round(pnl_pts * closing_quantity * tick_val, 2)
        pnl_pct = round((pnl_pts / max(closing_pos.entry_premium, 1.0)) * 100, 2)
        closing_pos.close(current_ltp, reason, ts)
        self.capital_manager.record_trade_closed(trade_id)
        self.order_manager.record_position_closed(trade_id)
        try:
            from utils.bot_trade_registry import mark_bot_trade_closed
            mark_bot_trade_closed(trade_id)
        except Exception:
            pass
        sig_dict = closing_pos.plan.signal.to_dict()
        sig_id = getattr(closing_pos.plan.signal, "signal_id", "") or sig_dict.get("signal_id", "")
        await self.bus.publish(Topic.POSITION_CLOSED, {
            "signal_id": sig_id,
            "option_symbol": closing_pos.plan.option_symbol,
            "direction": closing_pos.plan.signal.direction.value,
            "entry_premium": closing_pos.entry_premium,
            "exit_premium": current_ltp,
            "quantity": closing_quantity,
            "pnl": realized_pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
            "timestamp": ts.isoformat(),
            "simulated": closing_pos.is_simulated,
        }, self.NAME)
        self._pos = None
        self._entry_nifty_ltp = 0.0
        self._candles_held = 0
        self._peak_pnl = 0.0
        self._tsl_active = False
        self._tsl_floor = 0.0

    async def adopt_broker_position(self, broker_pos: dict) -> None:
        """
        Adopts an open broker position that was orphaned due to desync, restart, or premature close.
        Re-instantiates active tracking in PositionManagerAgent so SL, Target, Spot SL, Trailing SL,
        and EOD square-off will protect and manage the live trade.
        """
        if self._pos and self._pos.is_open:
            logger.info(f"[{self.NAME}] Internal position already open; ignoring adopt request for {broker_pos.get('symbol')}")
            return

        raw_sym = str(broker_pos.get("symbol", "") or "").strip()
        qty = int(broker_pos.get("quantity", 0) or 0)
        avg_price = float(broker_pos.get("avg_price", 0.0) or 0.0)
        ltp = float(broker_pos.get("ltp", 0.0) or avg_price)
        account = str(broker_pos.get("account", "primary") or "primary")

        from utils.bot_trade_registry import get_bot_trade_info, register_bot_order
        from utils.option_utils import canonical_option_sym

        reg_info = get_bot_trade_info(raw_sym)
        plan_meta = (reg_info or {}).get("plan_meta", {}) if reg_info else {}
        stored_plan = plan_meta.get("plan") or {}

        opt_sym = canonical_option_sym(raw_sym) or raw_sym
        ep = float(plan_meta.get("entry_premium") or stored_plan.get("entry_premium") or avg_price or ltp)
        if ep <= 0:
            ep = ltp or 95.5
        sl_prem = float(plan_meta.get("sl_premium") or stored_plan.get("sl_premium") or round(ep * 0.90, 2))
        tgt_prem = float(plan_meta.get("target_premium") or stored_plan.get("target_premium") or round(ep * 1.30, 2))
        spot_sl = float(plan_meta.get("sl_spot_level") or stored_plan.get("sl_spot_level") or 0.0)
        spot_source = str(plan_meta.get("sl_source") or stored_plan.get("sl_source") or "SWING_HIGH")
        dir_str = str(plan_meta.get("direction") or ("BUY_PUT" if "PE" in opt_sym else "BUY_CALL")).upper()
        order_id = str(plan_meta.get("order_id") or ((reg_info or {}).get("orders", [{}])[-1].get("order_id", "")) if reg_info else "")

        import re
        strike_m = re.search(r"(\d{5})(CE|PE)", opt_sym)
        strike_val = int(strike_m.group(1)) if strike_m else 23900
        opt_type = strike_m.group(2) if strike_m else ("PE" if "PE" in opt_sym else "CE")

        # Build plan dictionary and use existing _build_plan
        d = dict(stored_plan) if stored_plan else {}
        d.update({
            "signal": stored_plan.get("signal") or {
                "direction": dir_str,
                "confidence": 0.8,
                "strategy_name": str(plan_meta.get("strategy") or "SuperTrend+RSI"),
                "nifty_ltp": self._last_nifty_spot or 23900.0,
                "regime": "TRENDING",
            },
            "option_symbol": opt_sym,
            "strike": strike_val,
            "option_type": opt_type,
            "expiry_date": str(stored_plan.get("expiry_date") or date.today().isoformat()),
            "days_to_expiry": int(stored_plan.get("days_to_expiry", 5) or 5),
            "entry_premium": ep,
            "sl_premium": sl_prem,
            "target_premium": tgt_prem,
            "lot_size": NIFTY_LOT_SIZE,
            "quantity": qty,
            "desired_lots": max(1, qty // NIFTY_LOT_SIZE),
            "total_invested": round(ep * qty, 2),
            "sl_spot_level": spot_sl,
            "sl_source": spot_source,
            "order_id": order_id,
            "simulated": False,
            "execution": "REAL",
            "timestamp": datetime.now(IST).isoformat(),
        })

        built_plan = self._build_plan(d)
        self._pos = Position(
            plan=built_plan,
            entry_premium=ep,
            is_simulated=False,
            kite_order_id=order_id,
            execution_mode="REAL",
            entry_time=datetime.now(IST),
        )
        self._pos.current_premium = ltp
        self._quantity = qty
        self._initial_quantity = qty
        self._partial_realized_pnl = 0.0
        self._partial_transaction_costs = 0.0
        self._partial_quantity_closed = 0
        lots = max(int(self._quantity / max(self._pos.plan.lot_size, 1)), 1)
        self._ladder = ProfitLadder(self._pos.entry_premium, lots, self._pos.plan.lot_size)
        self._prev_pnl = 0.0
        self._candles_held = 0
        self._peak_pnl = 0.0
        self._tsl_active = False
        self._tsl_floor = self._pos.sl_premium
        self._nifty_closes = []
        self._premium_closes = []
        self._direction = dir_str
        self._spot_sl_level = spot_sl
        self._spot_sl_source = spot_source
        self._spot_sl_enabled = self._spot_sl_level > 0
        self._df_buffer = []
        self._intrabar_hit_counts = {}

        # Re-register in bot trade registry so is_active is True
        register_bot_order(
            symbol=opt_sym,
            order_id=self._pos.kite_order_id,
            quantity=qty,
            transaction="BUY",
            account=account,
            strategy=built_plan.signal.strategy_name if built_plan.signal else "SuperTrend+RSI",
            plan_meta=d,
        )
        self.order_manager.record_position_opened(opt_sym, {
            "order_id": self._pos.kite_order_id,
            "quantity": qty,
            "entry_premium": ep,
        })
        self.capital_manager.record_trade_opened(
            trade_id=opt_sym,
            premium=ep,
            quantity=qty,
            signal_id=getattr(built_plan.signal, "signal_id", opt_sym) if built_plan.signal else opt_sym,
            account_label=account,
        )

        logger.info(
            f"[{self.NAME}] 🛡️ Successfully adopted open broker position {opt_sym} | "
            f"qty={qty} @ ₹{ep:.1f} | SL=₹{sl_prem:.1f} | Tgt=₹{tgt_prem:.1f} | "
            f"SpotSL={spot_sl} ({spot_source}) | LTP=₹{ltp:.1f}"
        )


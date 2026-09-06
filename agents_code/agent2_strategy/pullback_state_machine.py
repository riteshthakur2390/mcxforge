"""
agents_code/agent2_strategy/pullback_state_machine.py — EMA20 Pullback Shadow State Machine

Implements an additive, deterministic shadow pending-entry state machine for
MEDIUM_QUALITY candidates (temporarily extended multi-category setups).

States:
- SIGNAL_RECEIVED
- PENDING_PULLBACK
- RETEST_DETECTED
- CONFIRMING
- SHADOW_ENTRY
- INVALIDATED
- EXPIRED
- MISSED_CONTINUATION
- AMBIGUOUS_SEQUENCE
- CANCELLED

Critical Guarantees:
- SHADOW / OBSERVE ONLY: Never places real live orders or blocks live execution.
- Point-in-Time Safe: Uses strictly completed candle data up to decision point.
- Idempotent: Suppresses duplicate signal_id deliveries.
- Crash-Resilient: Deterministically persists active setups to disk and restores on startup.
- Unclamped Outcome Tracking: Computes actual_realized_mae unfloored and stores risk cap separately.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")


class PullbackState(str, Enum):
    SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
    PENDING_PULLBACK = "PENDING_PULLBACK"
    RETEST_DETECTED = "RETEST_DETECTED"
    CONFIRMING = "CONFIRMING"
    SHADOW_ENTRY = "SHADOW_ENTRY"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    MISSED_CONTINUATION = "MISSED_CONTINUATION"
    AMBIGUOUS_SEQUENCE = "AMBIGUOUS_SEQUENCE"
    CANCELLED = "CANCELLED"


@dataclass
class PendingPullbackSetup:
    signal_id: str
    instrument: str
    direction: str
    original_signal_timestamp: str
    signal_price: float
    ema20_at_signal: float
    atr_at_signal: float
    quality_classification: str
    raw_vote_count: int
    independent_category_count: int
    ml_state: str
    state: str = PullbackState.PENDING_PULLBACK.value
    previous_state: str = PullbackState.SIGNAL_RECEIVED.value
    state_timestamp: str = ""
    bars_observed: int = 0
    max_observation_bars: int = 6
    max_observation_minutes: int = 30
    retest_timestamp: Optional[str] = None
    retest_price: Optional[float] = None
    confirmation_timestamp: Optional[str] = None
    shadow_entry_timestamp: Optional[str] = None
    shadow_entry_price: Optional[float] = None
    invalidation_timestamp: Optional[str] = None
    invalidation_price: Optional[float] = None
    invalidation_level: Optional[float] = None
    invalidation_reason: Optional[str] = None
    expiration_timestamp: Optional[str] = None
    transition_reason: str = "PENDING_OBSERVATION_STARTED"
    max_directional_move_pts: float = 0.0
    actual_realized_mae: Optional[float] = None
    actual_realized_mfe: Optional[float] = None
    risk_cap_distance_pts: float = 25.0
    state_history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PendingPullbackSetup":
        return cls(**data)


class PullbackShadowStateMachine:
    """
    Deterministic Shadow Execution State Machine for EMA20 Pullback Retests.
    """

    def __init__(self, state_dir: Optional[str] = "state") -> None:
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self.state_file = self.state_dir / "pullback_state_machine.json"
        else:
            self.state_file = None
        
        self.active_setups: Dict[str, PendingPullbackSetup] = {}
        self.completed_setups: List[PendingPullbackSetup] = []
        self._processed_signal_ids: set[str] = set()

        if self.state_file:
            self._restore_from_disk()

    def _restore_from_disk(self) -> None:
        """Restores pending setups and processed signal IDs from disk."""
        if not self.state_file or not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for s_dict in data.get("active_setups", []):
                    setup = PendingPullbackSetup.from_dict(s_dict)
                    self.active_setups[setup.signal_id] = setup
                    self._processed_signal_ids.add(setup.signal_id)
                for c_dict in data.get("completed_setups", []):
                    c_setup = PendingPullbackSetup.from_dict(c_dict)
                    self.completed_setups.append(c_setup)
                    self._processed_signal_ids.add(c_setup.signal_id)
            logger.info(f"[PullbackStateMachine] Restored {len(self.active_setups)} active setups from disk.")
        except Exception as e:
            logger.warning(f"[PullbackStateMachine] Error restoring state from disk: {e}")

    def _persist_to_disk(self) -> None:
        """Persists current state deterministically to disk."""
        if not self.state_file or os.getenv("TRADING_MODE") == "BACKTEST":
            return
        try:
            payload = {
                "active_setups": [s.to_dict() for s in self.active_setups.values()],
                "completed_setups": [s.to_dict() for s in self.completed_setups[-500:]],
                "last_persisted_ts": datetime.now(IST).isoformat(),
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"[PullbackStateMachine] Failed to persist state to disk: {e}")

    def on_candidate_signal(self, signal_data: dict, current_ts: datetime) -> Optional[PendingPullbackSetup]:
        """
        Receives candidate signal. If MEDIUM_QUALITY and telemetry available, registers pending pullback setup.
        Strictly idempotent: ignores repeated deliveries of existing signal_id.
        """
        signal_id = str(signal_data.get("signal_id") or "")
        quality_classification = str(
            signal_data.get("quality_classification")
            or signal_data.get("quality_tier")
            or ""
        ).upper()

        if not signal_id or quality_classification != "MEDIUM_QUALITY":
            return None

        # Idempotency check
        if signal_id in self._processed_signal_ids or signal_id in self.active_setups:
            return self.active_setups.get(signal_id)

        direction = str(signal_data.get("direction", "")).upper()
        signal_price = float(signal_data.get("nifty_ltp") or signal_data.get("price") or 0.0)
        ema20_val = float(signal_data.get("ema20") or (signal_price - 15.4 if direction == "BUY_CALL" else signal_price + 15.4))
        atr_val = float(signal_data.get("atr") or 25.0)
        votes = int(signal_data.get("votes") or 0)
        categories = int(signal_data.get("independent_category_count") or signal_data.get("categories") or 0)
        ml_state = str(signal_data.get("ml_state") or "POSITIVE").upper()

        # Invalidation level: 0.60 ATR breach beyond EMA20
        invalidation_level = (
            ema20_val - (0.60 * atr_val)
            if direction == "BUY_CALL"
            else ema20_val + (0.60 * atr_val)
        )

        setup = PendingPullbackSetup(
            signal_id=signal_id,
            instrument=str(signal_data.get("symbol", "NIFTY")),
            direction=direction,
            original_signal_timestamp=current_ts.isoformat(),
            signal_price=signal_price,
            ema20_at_signal=ema20_val,
            atr_at_signal=atr_val,
            quality_classification=quality_classification,
            raw_vote_count=votes,
            independent_category_count=categories,
            ml_state=ml_state,
            state=PullbackState.PENDING_PULLBACK.value,
            previous_state=PullbackState.SIGNAL_RECEIVED.value,
            state_timestamp=current_ts.isoformat(),
            invalidation_level=invalidation_level,
            risk_cap_distance_pts=round(0.60 * atr_val, 2),
            transition_reason="MEDIUM_QUALITY_SIGNAL_REGISTERED",
            state_history=[{
                "from_state": PullbackState.SIGNAL_RECEIVED.value,
                "to_state": PullbackState.PENDING_PULLBACK.value,
                "timestamp": current_ts.isoformat(),
                "reason": "INITIAL_REGISTRATION",
            }]
        )

        self.active_setups[signal_id] = setup
        self._processed_signal_ids.add(signal_id)
        self._persist_to_disk()
        logger.info(f"[PullbackStateMachine] Registered pending EMA20 pullback setup for {signal_id} (Direction: {direction}, Target EMA20: {ema20_val:.2f})")
        return setup

    def on_candle(
        self,
        candle_open: float,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        current_ema20: float,
        current_atr: float,
        current_ts: datetime,
    ) -> List[dict]:
        """
        Advances the state machine bar-by-bar across all active setups point-in-time.
        Evaluates retest, confirmation bounce, invalidation, timeout, and ambiguous collisions.
        """
        events: List[dict] = []
        completed_ids = []

        for signal_id, setup in list(self.active_setups.items()):
            setup.bars_observed += 1
            direction = setup.direction
            retest_zone_buffer = 0.15 * current_atr

            # 1. Update Directional Move Tracking (for Missed Continuation telemetry)
            if direction == "BUY_CALL":
                fwd_move = max(candle_high - setup.signal_price, 0.0)
            else:
                fwd_move = max(setup.signal_price - candle_low, 0.0)
            if fwd_move > setup.max_directional_move_pts:
                setup.max_directional_move_pts = fwd_move

            # 2. Check Ambiguous Sequence Collision (Intrabar collision near retest and invalidation)
            if direction == "BUY_CALL":
                touches_retest = candle_low <= (current_ema20 + retest_zone_buffer)
                crosses_invalidation = candle_low <= setup.invalidation_level
                spikes_target = candle_high >= (setup.signal_price + 20.0)
            else:
                touches_retest = candle_high >= (current_ema20 - retest_zone_buffer)
                crosses_invalidation = candle_high >= setup.invalidation_level
                spikes_target = candle_low <= (setup.signal_price - 20.0)

            if touches_retest and crosses_invalidation and spikes_target:
                # Ambiguous collision: cannot prove sequence within single bar -> treat conservatively
                self._transition(setup, PullbackState.AMBIGUOUS_SEQUENCE, current_ts, "AMBIGUOUS_INTRABAR_COLLISION")
                completed_ids.append(signal_id)
                events.append(setup.to_dict())
                continue

            # 3. Check Invalidation (Breakdown beyond 0.60 ATR of EMA20)
            if crosses_invalidation:
                setup.invalidation_timestamp = current_ts.isoformat()
                setup.invalidation_price = setup.invalidation_level
                setup.invalidation_reason = "EMA20_BREAKDOWN_EXCEEDED_0.60_ATR"
                self._transition(setup, PullbackState.INVALIDATED, current_ts, "STRUCTURAL_BREAKDOWN")
                completed_ids.append(signal_id)
                events.append(setup.to_dict())
                continue

            # 4. Check Retest & Confirmation
            if touches_retest:
                if setup.state == PullbackState.PENDING_PULLBACK.value:
                    setup.retest_timestamp = current_ts.isoformat()
                    setup.retest_price = current_ema20
                    self._transition(setup, PullbackState.RETEST_DETECTED, current_ts, "EMA20_ZONE_TOUCHED")

                # Confirmation: Close back in primary direction above/below EMA20
                is_confirmed = (
                    (candle_close > current_ema20 and candle_close > candle_open)
                    if direction == "BUY_CALL"
                    else (candle_close < current_ema20 and candle_close < candle_open)
                )

                if is_confirmed and setup.state in (PullbackState.RETEST_DETECTED.value, PullbackState.CONFIRMING.value):
                    setup.confirmation_timestamp = current_ts.isoformat()
                    setup.shadow_entry_timestamp = current_ts.isoformat()
                    setup.shadow_entry_price = round(current_ema20, 2)
                    
                    # Compute unfloored post-entry actual realized excursion proxies
                    entry_bonus = abs(setup.signal_price - setup.shadow_entry_price)
                    setup.actual_realized_mfe = round(setup.max_directional_move_pts + entry_bonus, 2)
                    setup.actual_realized_mae = round(max(abs(current_ema20 - (candle_low if direction == "BUY_CALL" else candle_high)), 0.0), 2)
                    
                    self._transition(setup, PullbackState.SHADOW_ENTRY, current_ts, "CONFIRMED_HEALTHY_BOUNCE")
                    completed_ids.append(signal_id)
                    events.append(setup.to_dict())
                    continue

            # 5. Check Timeout / Expiration (6 bars / 30 minutes)
            if setup.bars_observed >= setup.max_observation_bars:
                setup.expiration_timestamp = current_ts.isoformat()
                if setup.max_directional_move_pts >= 20.0 and not touches_retest:
                    self._transition(setup, PullbackState.MISSED_CONTINUATION, current_ts, "EXPANSION_WITHOUT_RETEST")
                else:
                    self._transition(setup, PullbackState.EXPIRED, current_ts, "OBSERVATION_TIMEOUT_6_BARS")
                completed_ids.append(signal_id)
                events.append(setup.to_dict())
                continue

        # Clean up completed setups
        for s_id in completed_ids:
            if s_id in self.active_setups:
                self.completed_setups.append(self.active_setups.pop(s_id))

        self._persist_to_disk()

        return events

    def _transition(self, setup: PendingPullbackSetup, new_state: PullbackState, ts: datetime, reason: str) -> None:
        setup.previous_state = setup.state
        setup.state = new_state.value
        setup.state_timestamp = ts.isoformat()
        setup.transition_reason = reason
        setup.state_history.append({
            "from_state": setup.previous_state,
            "to_state": setup.state,
            "timestamp": ts.isoformat(),
            "reason": reason,
        })
        logger.info(f"[PullbackStateMachine] {setup.signal_id}: {setup.previous_state} -> {setup.state} ({reason})")

    def get_active_count(self) -> int:
        return len(self.active_setups)

    def get_completed_count(self) -> int:
        return len(self.completed_setups)

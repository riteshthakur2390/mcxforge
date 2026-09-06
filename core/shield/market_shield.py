"""
core/shield/market_shield.py — Abnormal Market Protection & Shield Layer

Implements the multi-stage veto gate:
Market Data -> Strategy Signal -> Regime Check -> Risk Check -> Abnormal Market Check -> Execution

Outputs 4 strict verdicts:
- TRADE      (All conditions clear, trade approved)
- WAIT       (Transient condition, e.g. spread wide, retry in 30s)
- NO_TRADE   (Conditions unsuitable for strategy, e.g. regime mismatch)
- ABNORMAL   (Severe anomaly, e.g. stale data, spike, tender lockout, disconnect)
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Optional, List, Dict
import pytz

from instruments.base import InstrumentConfig
from core.regime.engine import MarketRegime, RegimeDetails

IST = pytz.timezone("Asia/Kolkata")


class ShieldVerdict(str, Enum):
    TRADE = "TRADE"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"
    ABNORMAL = "ABNORMAL"


@dataclass(frozen=True)
class ShieldDecision:
    verdict: ShieldVerdict
    reason: str
    stage_vetoed: Optional[str] = None
    telemetry: Dict = field(default_factory=dict)

    @property
    def is_executable(self) -> bool:
        return self.verdict == ShieldVerdict.TRADE


class MarketShieldLayer:
    """
    Reusable multi-factor protection layer for MCX commodity futures trading.
    """

    def __init__(
        self,
        max_stale_seconds: int = 600,       # 10 minutes max candle staleness
        max_spread_pts: float = 8.0,        # Maximum allowable bid-ask spread
        max_bar_atr_ratio: float = 3.5,     # Single bar spike threshold
        max_slippage_pts: float = 12.0,     # Max allowable slippage before halting
    ):
        self.max_stale_seconds = max_stale_seconds
        self.max_spread_pts = max_spread_pts
        self.max_bar_atr_ratio = max_bar_atr_ratio
        self.max_slippage_pts = max_slippage_pts

    def evaluate_pre_execution(
        self,
        instrument: InstrumentConfig,
        strategy_name: str,
        regime_details: RegimeDetails,
        permitted_regimes: List[MarketRegime],
        last_candle_time: Optional[datetime],
        current_ltp: float,
        bid_price: Optional[float] = None,
        ask_price: Optional[float] = None,
        broker_connected: bool = True,
        kill_switch_active: bool = False,
        expiry_date: Optional[date] = None,
        current_time: Optional[datetime] = None,
    ) -> ShieldDecision:
        """
        Runs the 5-stage veto checklist in strict order of defense.
        """
        now = current_time or datetime.now(IST)

        # ── 1. GLOBAL KILL SWITCH ─────────────────────────────────────────────
        if kill_switch_active:
            return ShieldDecision(
                verdict=ShieldVerdict.ABNORMAL,
                reason="GLOBAL_KILL_SWITCH_ACTIVE",
                stage_vetoed="KILL_SWITCH",
                telemetry={"kill_switch": True},
            )

        # ── 2. BROKER / CONNECTIVITY CHECK ────────────────────────────────────
        if not broker_connected:
            return ShieldDecision(
                verdict=ShieldVerdict.ABNORMAL,
                reason="BROKER_DISCONNECTED_OR_API_FAILURE",
                stage_vetoed="CONNECTIVITY",
                telemetry={"broker_connected": False},
            )

        # ── 3. TENDER-PERIOD PHYSICAL DELIVERY LOCKOUT ────────────────────────
        if expiry_date and instrument.is_in_tender_period(expiry_date, now.date()):
            dte = instrument.get_dte(expiry_date, now.date())
            return ShieldDecision(
                verdict=ShieldVerdict.ABNORMAL,
                reason=f"TENDER_PERIOD_LOCKOUT (DTE {dte} <= {instrument.tender_period_days} days to physical delivery)",
                stage_vetoed="TENDER_PERIOD",
                telemetry={"dte": dte, "tender_days": instrument.tender_period_days},
            )

        # ── 4. STALE MARKET DATA CHECK ────────────────────────────────────────
        if last_candle_time:
            # Normalize timezone
            if last_candle_time.tzinfo is None:
                last_candle_time = IST.localize(last_candle_time)
            age_sec = (now - last_candle_time).total_seconds()
            # Only check during active market session
            time_str = now.strftime("%H:%M")
            if instrument.session.open_time <= time_str <= instrument.session.close_time:
                if age_sec > self.max_stale_seconds:
                    return ShieldDecision(
                        verdict=ShieldVerdict.ABNORMAL,
                        reason=f"STALE_MARKET_DATA (Last candle {int(age_sec)}s ago > {self.max_stale_seconds}s limit)",
                        stage_vetoed="MARKET_DATA",
                        telemetry={"candle_age_seconds": age_sec},
                    )

        # ── 5. ABNORMAL PRICE ACTION / VOLATILITY CHECK ───────────────────────
        if regime_details.regime == MarketRegime.ABNORMAL:
            return ShieldDecision(
                verdict=ShieldVerdict.ABNORMAL,
                reason=f"ABNORMAL_PRICE_ACTION ({'; '.join(regime_details.reasons)})",
                stage_vetoed="REGIME",
                telemetry={"regime": regime_details.regime.value},
            )

        # ── 6. STRATEGY REGIME COMPATIBILITY CHECK ────────────────────────────
        if permitted_regimes and regime_details.regime not in permitted_regimes:
            return ShieldDecision(
                verdict=ShieldVerdict.NO_TRADE,
                reason=f"REGIME_MISMATCH ({strategy_name} requires {', '.join(r.value for r in permitted_regimes)}; current is {regime_details.regime.value})",
                stage_vetoed="REGIME_POLICY",
                telemetry={"current_regime": regime_details.regime.value},
            )

        # ── 7. EXCESSIVE SPREAD CHECK (TRANSIENT WAIT) ────────────────────────
        if bid_price and ask_price and bid_price > 0 and ask_price > 0:
            spread = round(ask_price - bid_price, 2)
            if spread > self.max_spread_pts:
                return ShieldDecision(
                    verdict=ShieldVerdict.WAIT,
                    reason=f"EXCESSIVE_SPREAD ({spread} pts > {self.max_spread_pts} pts threshold)",
                    stage_vetoed="SPREAD",
                    telemetry={"spread": spread, "bid": bid_price, "ask": ask_price},
                )

        # ── 8. SESSION CUTOFF CHECK ───────────────────────────────────────────
        now_hm = now.strftime("%H:%M")
        if now_hm > instrument.session.signal_cutoff:
            return ShieldDecision(
                verdict=ShieldVerdict.NO_TRADE,
                reason=f"PAST_SIGNAL_CUTOFF ({now_hm} > {instrument.session.signal_cutoff} IST)",
                stage_vetoed="SESSION_TIMING",
                telemetry={"current_time": now_hm},
            )

        # ── ALL DEFENSE GATES PASSED ──────────────────────────────────────────
        return ShieldDecision(
            verdict=ShieldVerdict.TRADE,
            reason="ALL_SHIELD_CHECKS_SATISFIED",
            stage_vetoed=None,
            telemetry={
                "instrument": instrument.symbol,
                "regime": regime_details.regime.value,
                "strategy": strategy_name,
                "timestamp": now.isoformat(),
            },
        )

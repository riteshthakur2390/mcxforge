"""
utils/advanced_filters.py — 5 Final Missing Edge Features
==========================================================

1. OI CHANGE VELOCITY TRACKER
   How fast is OI building at each strike? Fast build = fresh institutional money.
   NSE publishes option chain every 3 min. Diff between snapshots = velocity.

2. PUT WALL / CALL WALL DYNAMIC TRACKER
   Max OI strike on call side = resistance ceiling (CALL WALL).
   Max OI strike on put side = support floor (PUT WALL).
   Movement of walls during the day = smart money repositioning.

3. PROFIT FACTOR ADAPTIVE GATE
   Per-strategy real-time profit factor from last 20 trades.
   PF < 1.2 → auto-suppress that strategy today.
   PF > 2.0 → lower its vote threshold (trust it more).

4. CORRELATION-BASED SIGNAL DEDUPLICATION
   SuperTrend + ADX are both trend-following = NOT 2 independent votes.
   Deduplicate correlated strategies before counting votes.
   Real signal quality = number of UNCORRELATED confirmations.

5. EXPIRY DAY THETA ACCELERATION EXIT
   After 1:30 PM on Thursday, ATM theta is exponential, not linear.
   Force-close ALL open positions by 13:45 on expiry day.
   Prevents theta destruction in last 45 minutes.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from config.settings import NIFTY_STRIKE_STEP, JOURNAL_DIR
except ImportError:
    NIFTY_STRIKE_STEP = 50
    JOURNAL_DIR       = "journal"


# ══════════════════════════════════════════════════════════════════════════════
# 1. OI CHANGE VELOCITY TRACKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OIVelocitySignal:
    direction:          str       # "BUY_CALL" | "BUY_PUT" | "NEUTRAL"
    confidence_boost:   float
    fast_build_strike:  Optional[int]   # strike where OI building fastest
    fast_unwind_strike: Optional[int]   # strike where OI unwinding fastest
    call_oi_velocity:   float     # net call OI change rate (lots/min)
    put_oi_velocity:    float     # net put OI change rate (lots/min)
    note:               str


class OIVelocityTracker:
    """
    Tracks OI change velocity across all strikes.
    NSE publishes option chain every 3 min — fetch and diff.

    Fast OI buildup at strike above spot = fresh call writing = bearish signal
    Fast OI buildup at strike below spot = fresh put writing = bullish signal
    OI unwinding at ATM = institutional closing position = contra signal
    """

    SNAPSHOT_INTERVAL_MIN = 3   # NSE updates every 3 min

    def __init__(self) -> None:
        # {strike: deque of (timestamp, ce_oi, pe_oi)}
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self._last_snapshot_time: Optional[datetime] = None

    def update_snapshot(
        self,
        ce_oi_by_strike: dict[int, int],
        pe_oi_by_strike: dict[int, int],
        spot:            float,
    ) -> OIVelocitySignal:
        """
        Call every time you fetch a fresh option chain snapshot.
        Computes OI velocity at each strike and returns directional signal.

        Args:
            ce_oi_by_strike: {strike: call_open_interest}
            pe_oi_by_strike: {strike: put_open_interest}
            spot:            current NIFTY spot price
        """
        now = datetime.now(IST)

        # Record this snapshot
        for strike in set(list(ce_oi_by_strike.keys()) + list(pe_oi_by_strike.keys())):
            ce = ce_oi_by_strike.get(strike, 0)
            pe = pe_oi_by_strike.get(strike, 0)
            self._history[strike].append((now, ce, pe))

        self._last_snapshot_time = now

        # Need at least 2 snapshots to compute velocity
        if all(len(v) < 2 for v in self._history.values()):
            return self._neutral("need 2+ snapshots")

        # Compute velocity per strike
        call_vel: dict[int, float] = {}
        put_vel:  dict[int, float] = {}

        for strike, hist in self._history.items():
            if len(hist) < 2:
                continue
            prev_t, prev_ce, prev_pe = hist[-2]
            curr_t, curr_ce, curr_pe = hist[-1]
            elapsed = max((curr_t - prev_t).seconds / 60, 1)  # minutes
            call_vel[strike] = (curr_ce - prev_ce) / elapsed   # lots per minute
            put_vel[strike]  = (curr_pe - prev_pe) / elapsed

        if not call_vel:
            return self._neutral("no velocity data")

        # ATM reference
        atm = int(round(spot / NIFTY_STRIKE_STEP) * NIFTY_STRIKE_STEP)

        # Strikes above/below spot
        otm_calls = {k: v for k, v in call_vel.items() if k > spot}
        otm_puts  = {k: v for k, v in put_vel.items()  if k < spot}

        # Fastest building OTM call = new call writing = bearish pressure
        fast_call_strike = max(otm_calls, key=otm_calls.get) if otm_calls else None
        fast_put_strike  = max(otm_puts,  key=otm_puts.get)  if otm_puts  else None

        call_vel_total = sum(max(v, 0) for v in otm_calls.values())
        put_vel_total  = sum(max(v, 0) for v in otm_puts.values())

        # Key insight: NEW call writing above spot = institutions selling calls
        # = they expect price to stay BELOW that level = mildly bearish
        # But NEW put writing below spot = institutions selling puts
        # = they expect price to stay ABOVE that level = mildly bullish

        # For OPTION BUYERS, we want to follow the DIRECTION of OI buildup,
        # not trade against it.
        # If large put OI building rapidly at 22000 = support level forming = BUY CALL

        if put_vel_total > call_vel_total * 1.5 and put_vel_total > 500:
            direction = "BUY_CALL"
            boost     = min(0.06, put_vel_total / 10000)
            note      = (f"Put OI building fast ({put_vel_total:.0f} lots/min at "
                        f"{fast_put_strike}) → support forming → BUY_CALL")
        elif call_vel_total > put_vel_total * 1.5 and call_vel_total > 500:
            direction = "BUY_PUT"
            boost     = min(0.06, call_vel_total / 10000)
            note      = (f"Call OI building fast ({call_vel_total:.0f} lots/min at "
                        f"{fast_call_strike}) → resistance forming → BUY_PUT")
        else:
            direction = "NEUTRAL"
            boost     = 0.0
            note      = f"OI velocity balanced. Call={call_vel_total:.0f} Put={put_vel_total:.0f} lots/min"

        logger.debug(f"[OIVelocity] {note}")
        return OIVelocitySignal(
            direction          = direction,
            confidence_boost   = round(boost, 4),
            fast_build_strike  = fast_call_strike if direction == "BUY_PUT" else fast_put_strike,
            fast_unwind_strike = None,
            call_oi_velocity   = round(call_vel_total, 1),
            put_oi_velocity    = round(put_vel_total, 1),
            note               = note,
        )

    @staticmethod
    def _neutral(reason: str) -> OIVelocitySignal:
        return OIVelocitySignal("NEUTRAL", 0.0, None, None, 0.0, 0.0, reason)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PUT WALL / CALL WALL DYNAMIC TRACKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WallSignal:
    call_wall:          int       # highest OI call strike (resistance)
    put_wall:           int       # highest OI put strike (support)
    call_wall_oi:       int
    put_wall_oi:        int
    wall_range_high:    int       # expected trading range top
    wall_range_low:     int       # expected trading range bottom
    spot_position:      str       # "INSIDE_RANGE" | "ABOVE_CALL_WALL" | "BELOW_PUT_WALL"
    wall_signal:        str       # "BUY_CALL" | "BUY_PUT" | "NEUTRAL"
    call_wall_moved:    bool      # wall moved since last snapshot
    put_wall_moved:     bool
    note:               str


class PutCallWallTracker:
    """
    Tracks the PUT WALL and CALL WALL dynamically throughout the session.
    Wall movement = institutional repositioning = directional signal.

    CALL WALL = strike with highest call OI above spot = resistance ceiling
    PUT WALL  = strike with highest put OI below spot = support floor

    Price between walls: rangebound expected — option selling environment
    Price breaking above call wall: call writers covering = explosive up move
    Price breaking below put wall: put writers covering = explosive down move
    Wall RISING (repositioning higher): institutions getting more bullish
    Wall FALLING: institutions repositioning bearishly
    """

    def __init__(self) -> None:
        self._prev_call_wall: Optional[int] = None
        self._prev_put_wall:  Optional[int] = None
        self._wall_history:   list[dict]    = []

    def update(
        self,
        spot:            float,
        ce_oi_by_strike: dict[int, int],
        pe_oi_by_strike: dict[int, int],
    ) -> WallSignal:
        """
        Update wall levels from latest option chain snapshot.

        Call every time you have fresh OI data (every 3-15 min).
        """
        if not ce_oi_by_strike or not pe_oi_by_strike:
            return self._neutral(spot)

        # Call wall: highest OI strike ABOVE spot
        calls_above = {k: v for k, v in ce_oi_by_strike.items() if k > spot}
        puts_below  = {k: v for k, v in pe_oi_by_strike.items() if k < spot}

        # If spot has broken above ALL call strikes, use nearest call strike as wall
        if not calls_above and ce_oi_by_strike:
            max_call_strike = max(ce_oi_by_strike.keys())
            if spot > max_call_strike:
                # Spot broken through all call walls - extremely bullish
                put_wall = max(puts_below, key=puts_below.get) if puts_below else int(round(spot/50)*50)-200
                put_wall_oi = puts_below.get(put_wall, 0) if puts_below else 0
                note = (f"Spot {spot:.0f} ABOVE all call strikes "
                       f"(max={max_call_strike}) → extremely bullish → BUY_CALL")
                return WallSignal(max_call_strike, put_wall, 0, put_wall_oi,
                                 max_call_strike, put_wall, "ABOVE_CALL_WALL", "BUY_CALL",
                                 False, False, note)

        if not calls_above or not puts_below:
            return self._neutral(spot)

        call_wall     = max(calls_above, key=calls_above.get)
        put_wall      = max(puts_below,  key=puts_below.get)
        call_wall_oi  = calls_above[call_wall]
        put_wall_oi   = puts_below[put_wall]

        # Check if walls moved
        call_moved = (self._prev_call_wall is not None and
                      call_wall != self._prev_call_wall)
        put_moved  = (self._prev_put_wall is not None and
                      put_wall != self._prev_put_wall)

        # Signal from wall position
        if spot > call_wall:
            position = "ABOVE_CALL_WALL"
            # Call wall was breached — call writers covering — bullish explosive
            signal = "BUY_CALL"
            note   = (f"Spot {spot:.0f} ABOVE call wall {call_wall} "
                     f"(OI={call_wall_oi:,}) → call writers covering → EXPLOSIVE UP")
        elif spot < put_wall:
            position = "BELOW_PUT_WALL"
            signal   = "BUY_PUT"
            note     = (f"Spot {spot:.0f} BELOW put wall {put_wall} "
                       f"(OI={put_wall_oi:,}) → put writers covering → EXPLOSIVE DOWN")
        else:
            position = "INSIDE_RANGE"
            # Inside the walls — who has more OI?
            if put_wall_oi > call_wall_oi * 1.3:
                signal = "BUY_CALL"   # stronger support below = bullish
                note   = (f"Inside range [{put_wall}-{call_wall}]. "
                         f"Put wall stronger (OI={put_wall_oi:,} vs {call_wall_oi:,}) → mild CALL")
            elif call_wall_oi > put_wall_oi * 1.3:
                signal = "BUY_PUT"
                note   = (f"Inside range [{put_wall}-{call_wall}]. "
                         f"Call wall stronger (OI={call_wall_oi:,}) → mild PUT")
            else:
                signal = "NEUTRAL"
                note   = (f"Inside range [{put_wall}-{call_wall}]. "
                         f"Balanced walls → rangebound expected")

        # Wall movement adds confidence
        if call_moved and call_wall > (self._prev_call_wall or 0):
            note += f" | CALL WALL RISING ({self._prev_call_wall}→{call_wall}) = bullish repositioning"
        if put_moved and put_wall > (self._prev_put_wall or 0):
            note += f" | PUT WALL RISING ({self._prev_put_wall}→{put_wall}) = bullish"
        if call_moved and call_wall < (self._prev_call_wall or 999999):
            note += f" | CALL WALL FALLING = bearish repositioning"

        self._prev_call_wall = call_wall
        self._prev_put_wall  = put_wall
        self._wall_history.append({
            "time": datetime.now(IST).isoformat(),
            "call_wall": call_wall, "put_wall": put_wall, "spot": spot,
        })

        logger.debug(f"[WallTracker] {note}")

        return WallSignal(
            call_wall       = call_wall,
            put_wall        = put_wall,
            call_wall_oi    = call_wall_oi,
            put_wall_oi     = put_wall_oi,
            wall_range_high = call_wall,
            wall_range_low  = put_wall,
            spot_position   = position,
            wall_signal     = signal,
            call_wall_moved = call_moved,
            put_wall_moved  = put_moved,
            note            = note,
        )

    def _neutral(self, spot: float) -> WallSignal:
        atm = int(round(spot / 50) * 50)
        return WallSignal(atm+200, atm-200, 0, 0, atm+200, atm-200,
                          "INSIDE_RANGE", "NEUTRAL", False, False, "No OI data")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROFIT FACTOR ADAPTIVE GATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyAdaptiveScore:
    strategy_name:   str
    profit_factor:   float    # sum(wins) / sum(losses) over last N trades
    win_rate:        float
    sample_size:     int
    gate_action:     str      # "SUPPRESS" | "NORMAL" | "TRUST_MORE"
    vote_threshold:  int      # adjusted MIN_VOTES for this strategy
    conf_boost:      float    # confidence boost when this strategy fires
    note:            str


class ProfitFactorGate:
    """
    Real-time per-strategy profit factor tracker.
    Adapts strategy trust level based on recent live performance.
    
    This is superior to monthly walk-forward because it reacts
    within 20 trades — typically 2-3 weeks of live data.
    
    Self-correcting: bad strategy auto-suppresses, good strategy gets more weight.
    """

    LOOKBACK_TRADES = 20     # evaluate last N trades per strategy
    PF_SUPPRESS     = 1.20   # profit factor below this → suppress
    PF_TRUST_MORE   = 2.00   # profit factor above this → boost

    def __init__(self) -> None:
        # {strategy_name: deque of (pnl_pct,)}
        self._trade_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.LOOKBACK_TRADES)
        )

    def record_trade(self, strategy_name: str, pnl_pct: float) -> None:
        """Record a completed trade outcome for a strategy."""
        self._trade_history[strategy_name].append(pnl_pct)
        logger.debug(f"[PFGate] {strategy_name}: pnl={pnl_pct:+.1f}%")

    def record_trade_multi(self, strategies: list[str], pnl_pct: float) -> None:
        """Record outcome for multiple strategies that contributed to a trade."""
        for s in strategies:
            self.record_trade(s, pnl_pct)

    def evaluate(self, strategy_name: str, default_min_votes: int = 2) -> StrategyAdaptiveScore:
        """
        Get current adaptive gate status for a strategy.

        Returns gate_action:
          SUPPRESS   → don't count this strategy's vote today
          NORMAL     → count normally
          TRUST_MORE → count as 2 votes (strong recent performance)
        """
        hist = list(self._trade_history.get(strategy_name, []))

        if len(hist) < 5:
            return StrategyAdaptiveScore(
                strategy_name  = strategy_name,
                profit_factor  = 1.5,    # assume neutral if insufficient data
                win_rate       = 0.50,
                sample_size    = len(hist),
                gate_action    = "NORMAL",
                vote_threshold = default_min_votes,
                conf_boost     = 0.0,
                note           = f"Insufficient data ({len(hist)} trades). Using neutral."
            )

        wins   = [p for p in hist if p > 0]
        losses = [p for p in hist if p <= 0]
        pf     = (sum(wins) / max(sum(abs(l) for l in losses), 0.01)) if losses else 99.0
        wr     = len(wins) / len(hist)

        if pf < self.PF_SUPPRESS:
            action  = "SUPPRESS"
            thresh  = default_min_votes + 1   # harder to pass (needs more votes)
            boost   = -0.03
            note    = (f"{strategy_name} PF={pf:.2f} < {self.PF_SUPPRESS} "
                      f"({len(wins)}W/{len(losses)}L, WR={wr:.0%}) → SUPPRESSED today")
        elif pf > self.PF_TRUST_MORE:
            action  = "TRUST_MORE"
            thresh  = max(1, default_min_votes - 1)   # easier to pass
            boost   = +0.04
            note    = (f"{strategy_name} PF={pf:.2f} > {self.PF_TRUST_MORE} "
                      f"({len(wins)}W/{len(losses)}L, WR={wr:.0%}) → TRUST MORE")
        else:
            action  = "NORMAL"
            thresh  = default_min_votes
            boost   = 0.0
            note    = f"{strategy_name} PF={pf:.2f} — normal operation"

        if action in ("SUPPRESS",):
            logger.warning(f"[PFGate] {note}")
        else:
            logger.debug(f"[PFGate] {note}")

        return StrategyAdaptiveScore(
            strategy_name  = strategy_name,
            profit_factor  = round(pf, 3),
            win_rate       = round(wr, 3),
            sample_size    = len(hist),
            gate_action    = action,
            vote_threshold = thresh,
            conf_boost     = boost,
            note           = note,
        )

    def get_all_scores(self, default_min_votes: int = 2) -> dict[str, StrategyAdaptiveScore]:
        """Get adaptive scores for all tracked strategies."""
        return {
            name: self.evaluate(name, default_min_votes)
            for name in self._trade_history
        }

    def get_suppressed_strategies(self) -> list[str]:
        """Return list of currently suppressed strategies."""
        return [
            name for name, hist in self._trade_history.items()
            if len(hist) >= 5 and self.evaluate(name).gate_action == "SUPPRESS"
        ]


# ══════════════════════════════════════════════════════════════════════════════
# 4. CORRELATION-BASED SIGNAL DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════

# Strategy correlation groups (pre-computed from category knowledge)
# Strategies in the same group are correlated — only count 1 from each group
STRATEGY_CORRELATION_GROUPS = {
    "TREND_FOLLOWING": [
        "SuperTrendRSI", "ADXPsar", "ADXRising", "EMASlope",
        "HeikinAshi", "Ichimoku", "SuperTrend+RSI", "ADX+PSAR",
    ],
    "MOMENTUM": [
        "SqueezeMomentum", "StochRSI", "UTBot",
        "GapMomentum", "VWAPExtreme",
    ],
    "MEAN_REVERSION": [
        "BBSqueeze", "VWAPExtreme", "CPR", "StochRSI", "VWAP+EMA",
    ],
    "VOLUME_BASED": [
        "VolumeProfile", "VWAPema", "StrikeMomentum", "LiquiditySweep",
        "ValueArea",
    ],
    "OPTIONS_SPECIFIC": [
        "OIAnalysis", "IVContraction", "SkewHunter", "GammaExposure",
        "ExpiryWeek", "StrikeMomentum",
    ],
    "STRUCTURE_BASED": [
        "ORB", "SMC", "PriceAction", "FVG", "OpeningRangeBias",
        "AMD", "GapDirection",
    ],
    "DIVERGENCE": [
        "VIXDivergence", "RangeSpread", "WyckoffPhase",
    ],
}

# Reverse lookup: {strategy_name: group_name}
_STRAT_TO_GROUP: dict[str, str] = {}
for grp, strats in STRATEGY_CORRELATION_GROUPS.items():
    for s in strats:
        _STRAT_TO_GROUP[s] = grp


def deduplicate_signals(
    fired_strategies: list[str],
    min_uncorrelated_votes: int = 2,
) -> tuple[int, list[str], bool]:
    """
    Deduplicate correlated strategy signals.

    Instead of counting raw votes (5 trend-following = 5 votes),
    count UNCORRELATED votes (5 trend-following = 1 vote from that group).

    Args:
        fired_strategies:       list of strategy names that fired
        min_uncorrelated_votes: minimum unique-group votes needed

    Returns:
        (uncorrelated_vote_count, representative_strategies, passes_gate)
    """
    seen_groups:         set[str] = set()
    representative:      list[str] = []
    ungrouped_votes:     int = 0

    for strat in fired_strategies:
        group = _STRAT_TO_GROUP.get(strat)
        if group:
            if group not in seen_groups:
                seen_groups.add(group)
                representative.append(strat)
        else:
            # Uncategorised strategy = counts as its own vote
            ungrouped_votes += 1
            representative.append(strat)

    uncorrelated_votes = len(seen_groups) + ungrouped_votes
    passes             = uncorrelated_votes >= min_uncorrelated_votes

    return uncorrelated_votes, representative, passes


# ══════════════════════════════════════════════════════════════════════════════
# 5. EXPIRY DAY THETA ACCELERATION EXIT
# ══════════════════════════════════════════════════════════════════════════════

EXPIRY_FORCE_CLOSE_TIME = dtime(13, 45)   # force close by 1:45 PM on expiry
EXPIRY_CAUTION_TIME     = dtime(13, 0)    # reduce size after 1:00 PM on expiry
EXPIRY_AVOID_ENTRY_TIME = dtime(13, 30)   # no NEW entries after 1:30 PM on expiry


@dataclass
class ExpiryTheta:
    is_expiry_day:         bool
    current_time:          dtime
    force_close_now:       bool   # True = close immediately
    avoid_new_entries:     bool   # True = don't enter new trades
    size_multiplier:       float  # 1.0 normal, 0.5 reduced, 0.0 no trades
    theta_decay_rate:      float  # estimated % per 5-min candle at current time
    premium_remaining_pct: float  # approx % of premium value remaining
    note:                  str


class ExpiryThetaManager:
    """
    Manages expiry-day theta acceleration.
    
    On expiry (Thursday), ATM options lose value EXPONENTIALLY after 1 PM.
    A ₹100 ATM option at 9:15 AM becomes worth ~₹20 by 1:00 PM
    and ~₹5 by 2:30 PM even with ZERO price movement.
    
    This module:
    1. Detects expiry day (Thursday, or last Thursday for monthly)
    2. Applies exponential decay model to estimate remaining value
    3. Forces position closure before theta destroys remaining value
    4. Blocks new entries after 1:30 PM on expiry
    """

    # Approximate ATM premium remaining % by time on expiry day
    # (NSE 2020-2025 historical average, ATM options)
    THETA_SCHEDULE = {
        dtime(9,  15): 1.000,
        dtime(9,  45): 0.850,
        dtime(10, 15): 0.720,
        dtime(10, 45): 0.610,
        dtime(11, 15): 0.500,
        dtime(11, 45): 0.400,
        dtime(12, 15): 0.300,
        dtime(12, 45): 0.210,
        dtime(13, 15): 0.130,
        dtime(13, 45): 0.060,
        dtime(14, 15): 0.020,
        dtime(14, 45): 0.005,
    }

    def check(self) -> ExpiryTheta:
        """Check current expiry theta status. Call every candle."""
        now     = datetime.now(IST)
        today   = now.date()
        is_expiry = today.weekday() == 3   # Thursday = expiry for weekly

        if not is_expiry:
            return ExpiryTheta(
                is_expiry_day=False,
                current_time=now.time(),
                force_close_now=False,
                avoid_new_entries=False,
                size_multiplier=1.0,
                theta_decay_rate=0.0,
                premium_remaining_pct=1.0,
                note="Not expiry day — normal operation",
            )

        current_t = now.time()

        # Get nearest theta schedule entry
        prem_remaining = 1.0
        for sched_time, prem_pct in sorted(self.THETA_SCHEDULE.items()):
            if current_t >= sched_time:
                prem_remaining = prem_pct
            else:
                break

        # Theta decay rate (% per 5-min candle)
        sched_keys = sorted(self.THETA_SCHEDULE.keys())
        theta_rate = 0.0
        for i in range(1, len(sched_keys)):
            if sched_keys[i-1] <= current_t < sched_keys[i]:
                dt    = (datetime.combine(today, sched_keys[i]) -
                         datetime.combine(today, sched_keys[i-1])).seconds / 60 / 5
                d_prem = (self.THETA_SCHEDULE[sched_keys[i-1]] -
                         self.THETA_SCHEDULE[sched_keys[i]])
                theta_rate = d_prem / max(dt, 1) * 100   # % per 5-min candle
                break

        # Status
        force_close      = current_t >= EXPIRY_FORCE_CLOSE_TIME
        avoid_new        = current_t >= EXPIRY_AVOID_ENTRY_TIME
        size_mult        = 1.0

        if current_t >= EXPIRY_FORCE_CLOSE_TIME:
            size_mult = 0.0
            note = (f"⚠️ EXPIRY FORCE CLOSE — {current_t.strftime('%H:%M')} "
                   f"past 13:45 — {prem_remaining*100:.0f}% premium remaining. "
                   f"CLOSE ALL POSITIONS NOW.")
        elif current_t >= EXPIRY_AVOID_ENTRY_TIME:
            size_mult = 0.0
            note = (f"No new entries after {EXPIRY_AVOID_ENTRY_TIME.strftime('%H:%M')} "
                   f"on expiry. Premium={prem_remaining*100:.0f}% remaining. "
                   f"Hold existing positions until 13:45 force-close.")
        elif current_t >= EXPIRY_CAUTION_TIME:
            size_mult = 0.25
            note = (f"EXPIRY CAUTION — {current_t.strftime('%H:%M')} "
                   f"Premium decaying at {theta_rate:.1f}%/candle. "
                   f"Only {prem_remaining*100:.0f}% premium remaining. "
                   f"Reduce size to 25%. Exit targets lowered.")
        else:
            note = (f"Expiry day — {current_t.strftime('%H:%M')} "
                   f"Premium={prem_remaining*100:.0f}% remaining. "
                   f"Theta={theta_rate:.1f}%/candle. Trade normally but stay alert.")

        if force_close:
            logger.warning(f"[ExpiryTheta] {note}")
        else:
            logger.debug(f"[ExpiryTheta] {note}")

        return ExpiryTheta(
            is_expiry_day         = True,
            current_time          = current_t,
            force_close_now       = force_close,
            avoid_new_entries     = avoid_new,
            size_multiplier       = size_mult,
            theta_decay_rate      = round(theta_rate, 2),
            premium_remaining_pct = round(prem_remaining, 3),
            note                  = note,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

_oi_vel:   OIVelocityTracker | None = None
_wall:     PutCallWallTracker | None = None
_pf_gate:  ProfitFactorGate | None  = None
_expiry:   ExpiryThetaManager | None = None


def get_oi_velocity()  -> OIVelocityTracker:
    global _oi_vel
    if _oi_vel is None: _oi_vel = OIVelocityTracker()
    return _oi_vel

def get_wall_tracker() -> PutCallWallTracker:
    global _wall
    if _wall is None: _wall = PutCallWallTracker()
    return _wall

def get_pf_gate()      -> ProfitFactorGate:
    global _pf_gate
    if _pf_gate is None: _pf_gate = ProfitFactorGate()
    return _pf_gate

def get_expiry_theta() -> ExpiryThetaManager:
    global _expiry
    if _expiry is None: _expiry = ExpiryThetaManager()
    return _expiry

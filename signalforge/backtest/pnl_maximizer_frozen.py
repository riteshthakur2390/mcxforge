"""
signalforge/backtest/pnl_maximizer_frozen.py — Immutable Frozen Candidate Specification

VERSION_IDENTIFIER = "P&L_MAXIMIZER_V1_FROZEN_20260830_V1"
CHECKSUM_SPEC = "f9a2c810d7b04e6c981297e68b3c102a"

Rules:
1. Single-position bar deduplication (at most 1 trade per bar).
2. Morning consolidation trap filter: Between 10:00 and 11:30 IST, require >= 6 votes.
3. Friday afternoon chop filter: No new entries on Friday >= 13:00 IST.
4. Premium ceiling: Option contract entry price <= 180.0.
5. Structural anchor requirement: At least 1 verified anchor required:
   {"VolumeProfile", "RangeSpread", "StrikeMomentum", "FVG", "ElliottWave", "SkewHunter"}.
   If missing anchor, require >= 5 votes.
   If ADX+PSAR present without anchor, reject.
6. Execution & Risk bounds:
   - ₹30,000 normal trade budget (votes >= 5)
   - ₹15,000 reduced trade budget (votes == 4)
   - 15% maximum portfolio equity allocation cap
   - Single position concurrent limit
   - Stop loss 15 option pts, Target 30 option pts.
"""

from dataclasses import dataclass
from typing import Set, Tuple, Dict, Any

FROZEN_VERSION_ID: str = "P&L_MAXIMIZER_V1_FROZEN_20260830_V1"

VERIFIED_ANCHOR_STRATEGIES: Set[str] = {
    "VolumeProfile",
    "RangeSpread",
    "StrikeMomentum",
    "FVG",
    "ElliottWave",
    "SkewHunter",
}

MORNING_FILTER_START_HOUR: float = 10.0
MORNING_FILTER_END_HOUR: float = 11.5
MORNING_MIN_VOTES: int = 6

FRIDAY_RESTRICTION_START_HOUR: float = 13.0
PREMIUM_CEILING_MAX_INR: float = 180.0
BASE_CONSENSUS_VOTES: int = 4
ELEVATED_NON_ANCHOR_VOTES: int = 5

@dataclass(frozen=True)
class PnlMaximizerFrozenSpec:
    version_id: str = FROZEN_VERSION_ID
    morning_min_votes: int = MORNING_MIN_VOTES
    premium_ceiling: float = PREMIUM_CEILING_MAX_INR
    friday_cutoff_hour: float = FRIDAY_RESTRICTION_START_HOUR
    anchor_strategies: Tuple[str, ...] = tuple(sorted(VERIFIED_ANCHOR_STRATEGIES))

    def evaluate_candidate(
        self,
        strategy_votes: Set[str],
        vote_count: int,
        day_of_week: str,
        hour_of_day: float,
    ) -> Tuple[bool, str]:
        has_anchor = bool(strategy_votes.intersection(VERIFIED_ANCHOR_STRATEGIES))

        # 1. Morning consolidation trap filter
        if MORNING_FILTER_START_HOUR <= hour_of_day < MORNING_FILTER_END_HOUR and vote_count < self.morning_min_votes:
            return False, f"morning_trap_low_votes: {vote_count} < {self.morning_min_votes}"

        # 2. Friday afternoon chop filter
        if day_of_week == "Friday" and hour_of_day >= self.friday_cutoff_hour:
            return False, f"friday_afternoon_chop: Friday >= {self.friday_cutoff_hour:.1f}h"

        # 3. Anchor strategy requirement
        if not has_anchor and vote_count < ELEVATED_NON_ANCHOR_VOTES:
            return False, f"missing_structural_anchor: require >= {ELEVATED_NON_ANCHOR_VOTES} votes"

        # 4. Toxic correlation filter
        if "ADX+PSAR" in strategy_votes and not has_anchor:
            return False, "toxic_pair_no_anchor: ADX+PSAR without structural lead"

        return True, "PASS"

IMMUTABLE_MAXIMIZER_SPEC = PnlMaximizerFrozenSpec()

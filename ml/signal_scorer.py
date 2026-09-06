"""
ml/signal_scorer.py — Signal Quality Scoring
=============================================
WHAT THIS ADDS (not in any existing code):

1. WEIGHTED VOTES: Not all strategy votes are equal.
   S9 Ichimoku (5-condition confluence) should count more than
   S1 SuperTrend (single ATR flip). Weights reflect signal quality.

2. STRATEGY DIVERSITY BONUS: If 3 strategies agree but they are
   S1 (SuperTrend) + S5 (ADX+PSAR) + S7 (UTBot) — all three are
   ATR-based momentum indicators. They are not independent confirmations.
   A bonus is given when strategies from DIFFERENT categories agree.

3. SIGNAL FRESHNESS: A SuperTrend flip happened 8 candles ago but
   price has since retraced slightly. That signal is stale.
   Fresher signals (crossover happened in last 2 candles) score higher.

4. INTRADAY TIMING SCORE: NIFTY specific —
   09:30-10:30 = ORB session, momentum is fresh (higher score)
   10:30-12:00 = mid-morning, follow-through likely (medium score)
   12:00-14:00 = lunch chop, avoid (lower score)
   14:00-15:00 = afternoon trending, good (medium score)

5. TRANSACTION COST FILTER: Options with premium < ₹50 should be avoided.
   Even a 30% target = ₹15 gain, but broker charges alone eat ₹20+.
   Minimum premium filter makes each trade worth the friction.

USAGE in runner.py:
    from ml.signal_scorer import SignalQualityScorer
    scorer = SignalQualityScorer()
    score  = scorer.score(strategies_fired, direction, ltp, timestamp, conf)
    if score.total < ML_MIN_TOTAL_SCORE:        return   # skip weak signal
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd
import pytz

from config.settings import (
    ML_STRATEGY_WEIGHTS as STRATEGY_WEIGHTS,
    ML_STRATEGY_CATEGORIES as STRATEGY_CATEGORIES,
    ML_TIMING_SCORES as TIMING_SCORE,
    ML_QUALITY_MIN_SCORE as MIN_SCORE,
    ML_QUALITY_MIN_PREMIUM as MIN_PREMIUM,
    ML_QUALITY_DIVERSITY_BONUS as DIVERSITY_BONUS,
    ML_MAX_TOTAL_CONFIDENCE,
    ML_MIN_TOTAL_SCORE,
    ML_CHEAP_OPTION_PENALTY_FACTOR,
    ML_TIMING_FALLBACK_SESSION_SCORE,
    ML_TIMING_FALLBACK_DEFAULT_SCORE,
)

IST = pytz.timezone("Asia/Kolkata")


@dataclass
class QualityScore:
    """Breakdown of signal quality components."""
    weighted_votes:   float   # sum of strategy weights
    diversity_bonus:  float   # bonus for multi-category agreement
    timing_score:     float   # intraday timing quality
    base_conf:        float   # strategy confidence from ensemble
    total:            float   # final combined score
    categories_hit:   list    # which categories voted
    tradeable:        bool    # True if score passes minimum threshold
    reason:           str     # human-readable explanation


class SignalQualityScorer:
    """
    Scores signal quality using weighted votes, diversity, and timing.
    Drop-in addition to runner.py — does not modify existing logic.
    """

    MIN_SCORE         = MIN_SCORE   # below this → skip signal
    MIN_PREMIUM       = MIN_PREMIUM   # minimum option premium to trade (₹)
    DIVERSITY_BONUS   = DIVERSITY_BONUS   # added per extra category beyond first

    def score(
        self,
        strategies_fired: list[str],
        direction:        str,
        confidence:       float,
        timestamp:        Optional[pd.Timestamp] = None,
        option_premium:   float = 100.0,
    ) -> QualityScore:
        """
        Compute quality score for a signal.

        Args:
            strategies_fired: list of strategy names that voted
            direction:        "BUY_CALL" or "BUY_PUT"
            confidence:       ensemble confidence from runner
            timestamp:        candle timestamp (for timing score)
            option_premium:   estimated entry premium in ₹

        Returns:
            QualityScore with breakdown and tradeable flag
        """
        # 1. Weighted vote score
        raw_votes = sum(
            STRATEGY_WEIGHTS.get(s, 1.0) for s in strategies_fired
        )
        # Normalize: 2 votes at weight 1.0 each → 1.0
        # 2 votes at weight 1.5 each → 1.5, etc.
        n_votes = max(len(strategies_fired), 1)
        weighted = raw_votes / n_votes   # avg weight per vote

        # 2. Diversity bonus: how many different categories voted
        cats_hit = []
        for cat, members in STRATEGY_CATEGORIES.items():
            if any(s in members for s in strategies_fired):
                cats_hit.append(cat)
        n_cats = len(cats_hit)
        diversity = max(0.0, (n_cats - 1) * self.DIVERSITY_BONUS)

        # 3. Timing score
        t_score = self._timing_score(timestamp)

        # 4. Premium filter — penalise very cheap options
        if option_premium < self.MIN_PREMIUM:
            t_score *= ML_CHEAP_OPTION_PENALTY_FACTOR   # reduce quality score for cheap options

        # 5. Combine: weighted sum instead of pure multiplication to avoid over-suppression
        # confidence and weighted votes are primary (70%), timing is secondary (30%)
        combined = ((weighted * confidence) * 0.7 + t_score * 0.3) + diversity

        # Normalize to roughly 0-1 range
        # A perfect signal (weight=1.5, timing=1.0, conf=0.80, diversity=0.10)
        # scores = 1.5 × 1.0 × 0.80 + 0.10 = 1.30 → cap at 0.95
        total = round(min(ML_MAX_TOTAL_CONFIDENCE, combined), 4)

        tradeable = (total >= self.MIN_SCORE and option_premium >= self.MIN_PREMIUM)
        reason    = self._reason(weighted, t_score, cats_hit, total, option_premium)

        return QualityScore(
            weighted_votes  = round(weighted, 3),
            diversity_bonus = round(diversity, 3),
            timing_score    = round(t_score, 3),
            base_conf       = round(confidence, 3),
            total           = total,
            categories_hit  = cats_hit,
            tradeable       = tradeable,
            reason          = reason,
        )

    def _timing_score(self, ts: Optional[pd.Timestamp]) -> float:
        if ts is None:
            return ML_TIMING_FALLBACK_SESSION_SCORE
        try:
            h, m = int(ts.hour), int(ts.minute)
            for (h0, m0, h1, m1), score in TIMING_SCORE.items():
                if (h0 * 60 + m0) <= (h * 60 + m) < (h1 * 60 + m1):
                    return score
            return ML_TIMING_FALLBACK_DEFAULT_SCORE   # outside defined windows
        except Exception:
            return ML_TIMING_FALLBACK_SESSION_SCORE

    def _reason(
        self,
        weighted:      float,
        t_score:       float,
        cats_hit:      list,
        total:         float,
        premium:       float,
    ) -> str:
        parts = [
            f"weighted_votes={weighted:.2f}",
            f"timing={t_score:.2f}",
            f"categories={cats_hit}",
            f"premium=₹{premium:.0f}",
            f"→ score={total:.3f}",
        ]
        if total < self.MIN_SCORE:
            parts.append("❌ below threshold")
        elif premium < self.MIN_PREMIUM:
            parts.append("❌ premium too low")
        else:
            parts.append("✅ tradeable")
        return " | ".join(parts)
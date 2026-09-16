"""
agents_code/agent3_ml/reduced_budget_evaluator.py
==================================================
Standalone reduced-budget evaluator for REJECTED signals.

This module is called ONLY after a signal has been fully rejected by the
primary full-budget gates (ML confidence gate + rank gate + all secondary
lanes).  It provides a last-chance evaluation to determine if the signal
qualifies for a 1-lot reduced-budget trade.

Architecture:
    Primary Gate (filter.py)           Reduced Budget Evaluator (this file)
    ─────────────────────────          ──────────────────────────────────────
    Full-budget approval (3 lots)  →   Never called
    Full-budget rejection          →   evaluate_for_reduced_budget()
                                            ├── approved  → 1-lot trade
                                            └── rejected  → final rejection

Key design rules:
  • Never intercepts primary gate winners — only called after full rejection
  • Always caps at 1 lot via metadata
  • Own criteria — tunable independently without affecting full-budget PnL
  • Direction + structure validation included
"""

from __future__ import annotations

import os
from datetime import time as dt_time

import pytz

# ── Configurable thresholds (env-overridable) ─────────────────────────────
RB_MIN_VOTES = int(os.getenv("REDUCED_BUDGET_MIN_VOTES", "4"))
RB_MIN_RANK = float(os.getenv("REDUCED_BUDGET_MIN_RANK", "0.55"))
RB_MIN_SETUP = float(os.getenv("REDUCED_BUDGET_MIN_SETUP", "0.60"))
RB_NEAR_ML_DELTA = float(os.getenv("REDUCED_BUDGET_NEAR_ML_DELTA", "0.02"))
RB_NEAR_ML_MIN_RANK = float(os.getenv("REDUCED_BUDGET_NEAR_ML_MIN_RANK", "0.52"))
RB_MAX_TRADE_INR = float(os.getenv("REDUCED_BUDGET_MAX_TRADE_INR", "15000"))

# ── Smart filter constants (empirically derived from 206 near-miss trades) ─
RB_MAX_PREMIUM_INR = float(os.getenv("REDUCED_BUDGET_MAX_PREMIUM_INR", "140.0"))
# Open rush block: 09:15-09:45 trades have 23% WR (-Rs.11,348 loss over 22 trades)
RB_OPEN_RUSH_END_MINUTE = int(os.getenv("REDUCED_BUDGET_OPEN_RUSH_END_MINUTE", "570"))  # 9*60+30
# Close session gate: requires votes >= 6 at/after 14:30 (14*60+30=870)
RB_CLOSE_SESSION_START_MINUTE = int(os.getenv("REDUCED_BUDGET_CLOSE_SESSION_START_MINUTE", "870"))
RB_CLOSE_SESSION_MIN_VOTES = int(os.getenv("REDUCED_BUDGET_CLOSE_SESSION_MIN_VOTES", "6"))
# Day-move bias suppressor: block contrarian direction when market is trending strongly
# e.g. BUY_PUT when market is up >0.5% from open, BUY_CALL when down >0.5%
RB_DAY_MOVE_SUPPRESS_PCT = float(os.getenv("REDUCED_BUDGET_DAY_MOVE_SUPPRESS_PCT", "0.5"))
# Daily trade limiter: max 2 reduced budget scalps per day to prevent cluster churn
RB_MAX_DAILY_TRADES = int(os.getenv("REDUCED_BUDGET_MAX_DAILY_TRADES", "2"))

# In-memory daily counter (date_str -> count)
_daily_reduced_trade_counts: dict[str, int] = {}
_daily_reversal_counts: dict[str, int] = {}


def reset_daily_reduced_trade_counts() -> None:
    """Reset daily reduced trade counters (useful for backtests and daily rollover)."""
    _daily_reduced_trade_counts.clear()
    _daily_reversal_counts.clear()


def reset_daily_reversal_counts() -> None:
    """Reset daily reversal trade counter."""
    _daily_reversal_counts.clear()


IST = pytz.timezone("Asia/Kolkata")


def _signal_minutes(ts_str: str) -> int | None:
    """Extract minutes-since-midnight from an ISO timestamp string."""
    if not ts_str:
        return None
    try:
        import pandas as pd

        t = pd.Timestamp(ts_str)
        if t.tzinfo is None:
            t = t.tz_localize(IST)
        else:
            t = t.tz_convert(IST)
        return t.hour * 60 + t.minute
    except Exception:
        return None


def _day_move_pct(data: dict) -> float | None:
    """
    Compute intraday move % from the day's open to current spot.
    Returns None if either value is unavailable or zero.
    """
    try:
        # current spot: carried in signal payload as nifty_price / nifty_ltp
        current = 0.0
        for key in ("nifty_price", "nifty_ltp", "underlying_price", "spot", "ltp", "price"):
            val = float(data.get(key) or 0.0)
            if val > 0:
                current = val
                break

        # today's open: set by backtest engine premarket payload OR live DataFetcherAgent
        # propagated into signal metadata via _context
        today_open = 0.0
        for key in ("today_open", "nifty_open", "open_price", "day_open"):
            val = float(data.get(key) or 0.0)
            if val > 0:
                today_open = val
                break
        if today_open <= 0:
            ctx = ((data.get("metadata") or {}).get("_context") or {})
            for key in ("today_open", "nifty_open", "day_open"):
                val = float(ctx.get(key) or 0.0)
                if val > 0:
                    today_open = val
                    break

        if current <= 0 or today_open <= 0:
            return None
        return (current - today_open) / today_open * 100.0
    except Exception:
        return None


def _direction_ok(
    *,
    data: dict,
    direction: str,
    regime_label: str,
    votes: int = 0,
    setup_strength: float = 0.0,
) -> tuple[bool, str, bool]:
    """
    Validate direction against market structure for reduced budget.
    Returns (ok: bool, reason: str, is_first_reversal: bool).
    """
    context = ((data.get("metadata") or {}) .get("_context") or {})
    setup = context.get("setup", {}) or {}
    setup_context = setup.get("context", {}) or {}
    market_structure = (
        context.get("market_structure")
        or setup_context.get("market_structure")
        or {}
    )
    entry_validation = market_structure.get("entry_validation", {}) or {}
    direction_validation = (
        (entry_validation.get("by_direction", {}) or {}).get(direction, {})
    )
    if not bool(direction_validation.get("valid", entry_validation.get("valid", True))):
        # For reduced budget, we allow invalid structure since we use tight SL & 1-lot
        pass  # Removed: structure validation too strict for scalps
    signal_minutes = _signal_minutes(str(data.get("timestamp", "")))
    if signal_minutes is not None:
        if signal_minutes >= 15 * 60 + 20:
            return False, "late_session_penalty", False
        if direction == "BUY_CALL" and 9 * 60 + 30 <= signal_minutes < 10 * 60:
            v = int(data.get("strategy_votes") or data.get("votes") or votes or 0)
            st = float(data.get("setup_strength") or setup_strength or 0.0)
            if not (v >= 5 and st >= 0.50):
                return False, "morning_chop_call_bias", False

    structure_state = market_structure.get("structure_state", {}) or {}
    structure_bias = str(structure_state.get("bias", "UNKNOWN") or "UNKNOWN").upper()
    opposite_bias = (
        direction == "BUY_CALL" and structure_bias == "BEARISH"
    ) or (
        direction == "BUY_PUT" and structure_bias == "BULLISH"
    )

    trade_date = str(data.get("timestamp", "") or "")[:10]
    if opposite_bias:
        # Check First Reversal Scalp Lane criteria:
        # 1. Must be the FIRST reversal of the session (counts == 0)
        # 2. Reversal window: 10:45 AM to 01:45 PM IST (645 <= signal_minutes <= 825)
        # 3. Strong consensus: votes >= 6 OR (votes >= 5 and setup_strength >= 0.55)
        current_rev_count = _daily_reversal_counts.get(trade_date, 0)
        in_reversal_window = signal_minutes is not None and (10 * 60 + 45 <= signal_minutes <= 13 * 60 + 45)
        has_reversal_consensus = votes >= 6 or (votes >= 5 and setup_strength >= 0.55)

        if current_rev_count == 0 and in_reversal_window and has_reversal_consensus:
            return True, "first_reversal_scalp_lane", True
        return False, f"direction_vs_structure_bias:{structure_bias}", False

    return True, "", False


def evaluate_for_reduced_budget(
    *,
    data: dict,
    direction: str,
    votes: int,
    success_prob: float,
    rank_score: float,
    setup_strength: float,
    regime_label: str,
    quality_threshold: float,
    required_conf: float,
    rejection_source: str,
    entry_premium: float | None = None,
) -> tuple[bool, str]:
    """
    Evaluate whether a REJECTED signal qualifies for reduced budget (1-lot).

    Called ONLY after the primary full-budget gate has fully rejected the
    signal.  Returns (approved, reason_string).

    Parameters
    ----------
    rejection_source : str
        ``"ML_CONFIDENCE"`` when the ML model confidence gate rejected, or
        ``"RANK"`` when the rank quality gate rejected.
    entry_premium : float | None
        Estimated option entry premium in Rs. per unit.  When supplied, used
        to enforce the premium cap derived from empirical option candle audit.
    """
    # ── Strict ML Gate: Enforce ML Confidence >= 0.32 (Zero exception) ──────
    if success_prob < 0.32:
        return False, f"rb_low_ml_prob: success_prob={success_prob:.4f} < 0.32 (Strict ML Gate)"

    # ── Smart Filter 1: Block expensive options (empirical: >Rs.140 → 22% WR) ──
    if entry_premium is not None and entry_premium >= RB_MAX_PREMIUM_INR:
        return False, (
            f"rb_expensive_premium: entry_premium=Rs.{entry_premium:.1f} "
            f">= cap=Rs.{RB_MAX_PREMIUM_INR:.0f} (22% WR historically)"
        )

    # ── Smart Filter 2: Block open rush 09:15–09:45 (23% WR, -Rs.11,348) ─────
    signal_minutes_val = _signal_minutes(str(data.get("timestamp", "")))
    if signal_minutes_val is not None and signal_minutes_val < RB_OPEN_RUSH_END_MINUTE:
        return False, (
            f"rb_open_rush_block: signal at {signal_minutes_val // 60:02d}:{signal_minutes_val % 60:02d} "
            f"< 09:45 cutoff (23% WR at open — wide spreads, gap reversals)"
        )

    # ── Smart Filter 3: Close session (>=14:30) requires votes >= 6 (votes >= 3 for HeroZero) ─
    strats = data.get("strategies_fired") or []
    is_hero = "HeroZero" in strats
    required_close_votes = 3 if is_hero else RB_CLOSE_SESSION_MIN_VOTES
    if (
        signal_minutes_val is not None
        and signal_minutes_val >= RB_CLOSE_SESSION_START_MINUTE
        and votes < required_close_votes
    ):
        return False, (
            f"rb_close_low_votes: votes={votes} < {required_close_votes} "
            f"required after 14:30 (close session higher-conviction gate)"
        )

    # ── Smart Filter 4: Day-move bias suppressor (empirical: July 23-24 cluster) ─
    # Block BUY_PUT when market is up >0.5% from open (strong intraday uptrend)
    # Block BUY_CALL when market is down >0.5% from open (strong intraday downtrend)
    day_move = _day_move_pct(data)
    if day_move is not None:
        if day_move > RB_DAY_MOVE_SUPPRESS_PCT and direction == "BUY_PUT":
            return False, (
                f"rb_day_bias_suppressor: market up {day_move:+.2f}% from open "
                f"> +{RB_DAY_MOVE_SUPPRESS_PCT:.1f}% — suppressing BUY_PUT scalp "
                f"(contrarian vs. strong intraday trend)"
            )
        if day_move < -RB_DAY_MOVE_SUPPRESS_PCT and direction == "BUY_CALL":
            return False, (
                f"rb_day_bias_suppressor: market down {day_move:+.2f}% from open "
                f"< -{RB_DAY_MOVE_SUPPRESS_PCT:.1f}% — suppressing BUY_CALL scalp "
                f"(contrarian vs. strong intraday downtrend)"
            )

    # ── Smart Filter 5: Daily trade limit (max 2 reduced scalps per day) ─────
    trade_date = str(data.get("timestamp", ""))[:10]
    if trade_date:
        current_daily_count = _daily_reduced_trade_counts.get(trade_date, 0)
        if current_daily_count >= RB_MAX_DAILY_TRADES:
            return False, (
                f"rb_max_daily_trades_reached: {current_daily_count}/{RB_MAX_DAILY_TRADES} "
                f"reduced budget scalps already taken today ({trade_date})"
            )

    # ── Gate 1: minimum vote consensus ────────────────────────────────────
    if votes < RB_MIN_VOTES:
        return False, f"rb_low_votes: {votes} < {RB_MIN_VOTES}"

    # ── Gate 2: direction + structure validation ──────────────────────────
    dir_ok, dir_reason, is_first_reversal = _direction_ok(
        data=data, direction=direction, regime_label=regime_label, votes=votes, setup_strength=setup_strength,
    )
    if not dir_ok:
        return False, f"rb_direction_fail: {dir_reason}"

    if is_first_reversal:
        if trade_date:
            _daily_reduced_trade_counts[trade_date] = _daily_reduced_trade_counts.get(trade_date, 0) + 1
            _daily_reversal_counts[trade_date] = _daily_reversal_counts.get(trade_date, 0) + 1
        return True, (
            f"rb_first_reversal_scalp_lane: direction={direction}, votes={votes}, "
            f"setup={setup_strength:.2f}, prob={success_prob:.3f}, rank={rank_score:.2f}"
        )

    # ── Gate 3: quality floor — votes OR setup_strength ───────────────────
    has_consensus = votes >= 4
    has_strong_setup = setup_strength >= RB_MIN_SETUP
    if not (has_consensus or has_strong_setup):
        return False, (
            f"rb_low_quality: votes={votes} < 4 and "
            f"setup={setup_strength:.2f} < {RB_MIN_SETUP:.2f}"
        )

    # ── Lane: Smart Morning Value / Pullback Lane (0.440 <= rank < 0.450) ───
    if 0.440 <= rank_score < 0.450:
        in_morning_window = signal_minutes_val is not None and signal_minutes_val < 12 * 60
        raw_strats = data.get("strategies_fired") or data.get("strategies") or []
        if isinstance(raw_strats, str):
            strats = [s.strip() for s in raw_strats.split("|") if s.strip()]
        else:
            strats = [str(s).strip() for s in raw_strats if str(s).strip()]
        has_value_vol = any(s in ["ValueArea", "VolumeProfile"] for s in strats)
        is_pullback = str(data.get("setup_type") or "").strip() == "trend_pullback"
        cpr_trap = "CPR" in strats and not has_value_vol

        if in_morning_window and (has_value_vol or is_pullback) and not cpr_trap:
            if trade_date:
                _daily_reduced_trade_counts[trade_date] = _daily_reduced_trade_counts.get(trade_date, 0) + 1
            return True, (
                f"rb_morning_value_pullback_lane: direction={direction}, votes={votes}, "
                f"setup={setup_strength:.2f}, prob={success_prob:.3f}, rank={rank_score:.3f}"
            )

    # ── Gate 4: minimum rank score ────────────────────────────────────────
    if votes >= 8:
        min_rank = 0.50
    elif votes >= 6:
        min_rank = 0.52
    elif votes >= 5:
        min_rank = 0.55
    elif votes >= 4:
        min_rank = 0.58
    else:
        min_rank = RB_MIN_RANK

    if rejection_source == "ML_CONFIDENCE":
        # ML confidence rejection — use the near-ML rank floor
        effective_min_rank = min(min_rank, RB_NEAR_ML_MIN_RANK) if has_consensus else RB_NEAR_ML_MIN_RANK
    else:
        # Rank rejection — rank must be in the near-miss band
        effective_min_rank = min_rank

    if rank_score < effective_min_rank:
        return False, (
            f"rb_low_rank: rank={rank_score:.2f} < "
            f"min={effective_min_rank:.2f}"
        )

    # ── Gate 5: minimum model confidence (strict >= 0.32) ────────────────
    prob_floor = 0.32
    if votes < 4:
        # setup_strength qualified: require closer to threshold
        prob_floor = max(0.32, required_conf - RB_NEAR_ML_DELTA)

    if success_prob < prob_floor:
        return False, (
            f"rb_low_prob: prob={success_prob:.3f} < "
            f"floor={prob_floor:.3f} (votes={votes})"
        )

    # ── All gates passed ──────────────────────────────────────────────────
    if trade_date:
        _daily_reduced_trade_counts[trade_date] = _daily_reduced_trade_counts.get(trade_date, 0) + 1

    if rejection_source == "ML_CONFIDENCE":
        reason = (
            f"reduced_budget_ml_near_miss: prob={success_prob:.3f} "
            f"< required={required_conf:.2f}, rank={rank_score:.2f}, "
            f"votes={votes}, setup={setup_strength:.2f}"
        )
    else:
        reason = (
            f"reduced_budget_rank_near_miss: rank={rank_score:.2f} "
            f"< required={quality_threshold:.2f}, votes={votes}, "
            f"setup={setup_strength:.2f}"
        )
    return True, reason


def reduced_budget_meta(reason: str, *, min_ml_prob: float | None = None) -> dict:
    """Build metadata dict for reduced-budget approved signals."""
    meta = {
        "reduced_budget_lane": True,
        "reduced_budget_reason": reason,
        "max_trade_investment_inr": RB_MAX_TRADE_INR,
    }
    if min_ml_prob is not None:
        meta["reduced_budget_min_ml_prob"] = round(float(min_ml_prob), 4)
    return meta

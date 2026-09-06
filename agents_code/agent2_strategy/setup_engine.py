from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from agents_code.agent2_strategy.market_structure import MarketStructureLiquidityEngine
from core.models import Direction
from config.settings import (
    SETUP_VOTE_BASE_STRENGTH, SETUP_VOTE_SCORE_BONUS_MAX,
    SETUP_VOTE_SCORE_THRESHOLD, SETUP_VOTE_SCORE_MULT,
    SETUP_VOTE_COUNT_BONUS_MAX, SETUP_VOTE_COUNT_THRESHOLD,
    SETUP_VOTE_COUNT_MULT, SETUP_VOTE_TRENDING_BONUS,
    SETUP_VOTE_BIAS_BONUS, SETUP_VOTE_CONF_MAX,
    SETUP_VOTE_CONF_MIN, SETUP_ZONE_PAD_EMA_MULT,
    SETUP_ZONE_PAD_LTP_MULT, SETUP_ZONE_PAD_EMA_REDUCED_MULT,
    SETUP_ZONE_PAD_INFERRED_MULT,
    SETUP_STOP_PAD_ATR_MULT, SETUP_STOP_PAD_LTP_MULT,
    SETUP_MOVE_TARGET_ATR_MULT, SETUP_MOVE_TARGET_EMA_MULT,
    SETUP_MOVE_TARGET_LTP_MULT, SETUP_TREND_NEAR_EMA_ATR_MULT,
    SETUP_TREND_NEAR_EMA_LTP_MULT, SETUP_TREND_PULLBACK_ATR_MULT,
    SETUP_TREND_PULLBACK_STOP_MULT, SETUP_TREND_CONFIRM_ATR_MULT,
    SETUP_TREND_ADX_THRESHOLD, SETUP_TREND_STRENGTH_BASE,
    SETUP_TREND_STRENGTH_ADX_MULT, SETUP_TREND_NON_TRENDING_PENALTY,
    SETUP_TREND_STOP_ATR_MULT, SETUP_TREND_STOP_LTP_MULT,
    SETUP_TREND_MOVE_ATR_MULT, SETUP_TREND_MOVE_EMA_MULT,
    SETUP_BREAKOUT_RANGE_ATR_MULT, SETUP_BREAKOUT_RANGE_LTP_MULT,
    SETUP_BREAKOUT_ADX_THRESHOLD, SETUP_BREAKOUT_BUFFER_ATR_MULT,
    SETUP_BREAKOUT_BUFFER_LTP_MULT, SETUP_BREAKOUT_VOL_MULT_CALL,
    SETUP_BREAKOUT_VOL_MULT_PUT, SETUP_BREAKOUT_STRENGTH_BASE,
    SETUP_BREAKOUT_STRENGTH_VOL_MULT, SETUP_BREAKOUT_STRENGTH_BB_THRESHOLD,
    SETUP_BREAKOUT_STRENGTH_BB_DIVISOR, SETUP_BREAKOUT_STRENGTH_BB_MULT,
    SETUP_BREAKOUT_MOVE_RANGE_MULT, SETUP_BREAKOUT_MOVE_ATR_MULT,
    SETUP_REVERSION_STRETCH_ATR_MULT, SETUP_REVERSION_STRETCH_LTP_MULT,
    SETUP_REVERSION_STRENGTH_BASE, SETUP_REVERSION_STRENGTH_STRETCH_MULT,
    SETUP_REVERSION_ENTRY_ATR_MULT, SETUP_REVERSION_STOP_ATR_MULT,
    SETUP_REVERSION_MOVE_STRETCH_MULT, SETUP_REVERSION_MOVE_ATR_MULT,
    SETUP_STRUCTURE_BOOST_EVENT, SETUP_STRUCTURE_BOOST_BOS,
    SETUP_STRUCTURE_BOOST_CHOCH, SETUP_STRUCTURE_BOOST_BIAS,
    SETUP_STRUCTURE_BOOST_REVERSION, SETUP_STRUCTURE_BOOST_BREAKOUT,
    SETUP_STRUCTURE_PENALTY_SOFT, SETUP_STRUCTURE_PENALTY_HARD,
    SETUP_STRUCTURE_PENALTY_AVOID, SETUP_STRUCTURE_PENALTY_MIDDLE,
    SETUP_REVERSION_CLOSE_LOCATION_THRESHOLD, SETUP_REVERSION_MIN_ATR,
    SETUP_REVERSION_STRETCH_RATIO_MAX, SETUP_REVERSION_ZONE_THRESHOLD_HIGH,
    SETUP_REVERSION_ZONE_THRESHOLD_LOW, SETUP_STRUCTURE_VALIDATION_BOS_THRESHOLD_LOW,
    SETUP_STRUCTURE_VALIDATION_BOS_THRESHOLD_HIGH, SETUP_STRUCTURE_VALIDATION_REVERSION_THRESHOLD_LOW,
    SETUP_STRUCTURE_VALIDATION_REVERSION_THRESHOLD_HIGH,
    SETUP_BREAKOUT_SCAN_LOOKBACK, ML_RANK_TIER_LOW,
)
from config.settings.strategy import RUNNER_CONF_MAX, S3_MIN_DF_LEN, S12_MIN_DF_LEN, S6_MIN_DF_LEN


@dataclass(frozen=True)
class TradeSetup:
    setup_type: str
    direction: str
    setup_strength: float
    entry_zone_low: float
    entry_zone_high: float
    stop_loss_level: float
    expected_move: float
    context: dict

    def to_dict(self) -> dict:
        data = asdict(self)
        data["setup_strength"] = round(float(self.setup_strength), 4)
        data["entry_zone_low"] = round(float(self.entry_zone_low), 2)
        data["entry_zone_high"] = round(float(self.entry_zone_high), 2)
        data["stop_loss_level"] = round(float(self.stop_loss_level), 2)
        data["expected_move"] = round(float(self.expected_move), 2)
        return data


class TradeSetupEngine:
    def __init__(self) -> None:
        self._structure_engine = MarketStructureLiquidityEngine()

    def evaluate(
        self,
        *,
        df: pd.DataFrame,
        cache,
        regime_details: dict,
    ) -> TradeSetup | None:
        structure_snapshot = self._structure_engine.evaluate(
            df=df,
            cache=cache,
            regime_details=regime_details,
        )
        candidates = []
        for builder in (
            self._trend_pullback,
            self._breakout,
            self._mean_reversion,
        ):
            setup = builder(
                df=df,
                cache=cache,
                regime_details=regime_details,
                structure_snapshot=structure_snapshot,
            )
            if setup is not None:
                candidates.append(setup)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.setup_strength)

    def infer_vote_aligned_setup(
        self,
        *,
        df: pd.DataFrame,
        cache,
        regime_details: dict,
        direction: str,
        weighted_score: float,
        votes: int,
    ) -> TradeSetup | None:
        if len(df) < S3_MIN_DF_LEN:
            return None

        structure_snapshot = self._structure_engine.evaluate(
            df=df,
            cache=cache,
            regime_details=regime_details,
        )
        close = float(df["close"].iloc[-1])
        ema20 = float(
            (
                cache.ema_20.iloc[-1]
                if cache and getattr(cache, "ema_20", None) is not None
                else close
            )
            or close
        )
        atr = float(
            (
                cache.atr_14.iloc[-1]
                if cache and getattr(cache, "atr_14", None) is not None
                else (df["high"] - df["low"]).tail(14).mean()
            )
            or 0.0
        )
        regime_label = str(
            ((regime_details or {}).get("detailed_regime", {}) or {}).get("label", "RANGING")
        ).upper()
        structure_state = structure_snapshot.get("structure_state", {}) or {}

        base_strength = SETUP_VOTE_BASE_STRENGTH
        base_strength += min(SETUP_VOTE_SCORE_BONUS_MAX, max(0.0, float(weighted_score) - SETUP_VOTE_SCORE_THRESHOLD) * SETUP_VOTE_SCORE_MULT)
        base_strength += min(SETUP_VOTE_COUNT_BONUS_MAX, max(0, int(votes) - SETUP_VOTE_COUNT_THRESHOLD) * SETUP_VOTE_COUNT_MULT)
        if regime_label == "TRENDING":
            base_strength += SETUP_VOTE_TRENDING_BONUS
        if (
            (direction == Direction.BUY_CALL.value and structure_state.get("bias") == "BULLISH")
            or (direction == Direction.BUY_PUT.value and structure_state.get("bias") == "BEARISH")
        ):
            base_strength += SETUP_VOTE_BIAS_BONUS
        base_strength = min(SETUP_VOTE_CONF_MAX, max(SETUP_VOTE_CONF_MIN, round(base_strength, 4)))

        zone_pad = max(atr * SETUP_ZONE_PAD_EMA_REDUCED_MULT, close * SETUP_ZONE_PAD_LTP_MULT)
        stop_pad = max(atr * SETUP_STOP_PAD_ATR_MULT, close * SETUP_STOP_PAD_LTP_MULT)
        move_target = max(atr * SETUP_MOVE_TARGET_ATR_MULT, abs(close - ema20) * SETUP_MOVE_TARGET_EMA_MULT, close * SETUP_MOVE_TARGET_LTP_MULT)

        if direction == Direction.BUY_CALL.value:
            candidate = TradeSetup(
                setup_type="vote_aligned",
                direction=direction,
                setup_strength=base_strength,
                entry_zone_low=min(close - zone_pad, ema20),
                entry_zone_high=max(close, ema20 + zone_pad * SETUP_ZONE_PAD_INFERRED_MULT),
                stop_loss_level=close - stop_pad,
                expected_move=move_target,
                context={
                    "ema20": round(ema20, 2),
                    "atr": round(atr, 2),
                    "inferred_from_votes": True,
                },
            )
        else:
            candidate = TradeSetup(
                setup_type="vote_aligned",
                direction=direction,
                setup_strength=base_strength,
                entry_zone_low=min(close, ema20 - zone_pad * SETUP_ZONE_PAD_INFERRED_MULT),
                entry_zone_high=max(close + zone_pad, ema20),
                stop_loss_level=close + stop_pad,
                expected_move=move_target,
                context={
                    "ema20": round(ema20, 2),
                    "atr": round(atr, 2),
                    "inferred_from_votes": True,
                },
            )
        return self._with_structure_context(candidate, structure_snapshot)

    def _trend_pullback(self, *, df: pd.DataFrame, cache, regime_details: dict, structure_snapshot: dict) -> TradeSetup | None:
        detailed = regime_details.get("detailed_regime", {}) or {}
        is_trending = str(detailed.get("label", "")).upper() == "TRENDING"
        if len(df) < S12_MIN_DF_LEN:
            return None
        adx = float((cache.adx.iloc[-1] if cache and cache.adx is not None else 0.0) or 0.0)
        ema20 = float((cache.ema_20.iloc[-1] if cache and cache.ema_20 is not None else df["close"].iloc[-1]) or df["close"].iloc[-1])
        ema50 = float((cache.ema_50.iloc[-1] if cache and cache.ema_50 is not None else df["close"].iloc[-1]) or df["close"].iloc[-1])
        atr = float((cache.atr_14.iloc[-1] if cache and cache.atr_14 is not None else (df["high"] - df["low"]).tail(14).mean()) or 0.0)
        close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        low = float(df["low"].iloc[-1])
        high = float(df["high"].iloc[-1])
        recent_low = float(df["low"].tail(3).min())
        recent_high = float(df["high"].tail(3).max())
        near_ema = abs(close - ema20) <= max(atr * SETUP_TREND_NEAR_EMA_ATR_MULT, close * SETUP_TREND_NEAR_EMA_LTP_MULT)
        bullish_pullback = recent_low <= ema20 + atr * SETUP_TREND_PULLBACK_ATR_MULT and close >= ema20 - atr * SETUP_TREND_PULLBACK_STOP_MULT
        bearish_pullback = recent_high >= ema20 - atr * SETUP_TREND_PULLBACK_ATR_MULT and close <= ema20 + atr * SETUP_TREND_PULLBACK_STOP_MULT

        if close > ema50 and bullish_pullback and close >= prev_close - atr * SETUP_TREND_CONFIRM_ATR_MULT and near_ema and adx >= SETUP_TREND_ADX_THRESHOLD:
            strength = min(RUNNER_CONF_MAX, SETUP_TREND_STRENGTH_BASE + max(0.0, adx - SETUP_TREND_ADX_THRESHOLD) * SETUP_TREND_STRENGTH_ADX_MULT)
            if not is_trending:
                strength -= SETUP_TREND_NON_TRENDING_PENALTY
            if strength < SETUP_TREND_STRENGTH_BASE: return None
            return self._with_structure_context(TradeSetup(
                setup_type="trend_pullback",
                direction=Direction.BUY_CALL.value,
                setup_strength=round(strength, 4),
                entry_zone_low=min(ema20, close),
                entry_zone_high=max(ema20, close),
                stop_loss_level=ema20 - max(atr * SETUP_TREND_STOP_ATR_MULT, close * SETUP_TREND_STOP_LTP_MULT),
                expected_move=max(atr * SETUP_TREND_MOVE_ATR_MULT, abs(close - ema20) * SETUP_TREND_MOVE_EMA_MULT),
                context={"adx": round(adx, 2), "ema20": round(ema20, 2), "ema50": round(ema50, 2)},
            ), structure_snapshot)
        if close < ema50 and bearish_pullback and close <= prev_close + atr * SETUP_TREND_CONFIRM_ATR_MULT and near_ema and adx >= SETUP_TREND_ADX_THRESHOLD:
            strength = min(RUNNER_CONF_MAX, SETUP_TREND_STRENGTH_BASE + max(0.0, adx - SETUP_TREND_ADX_THRESHOLD) * SETUP_TREND_STRENGTH_ADX_MULT)
            if not is_trending:
                strength -= SETUP_TREND_NON_TRENDING_PENALTY
            if strength < SETUP_TREND_STRENGTH_BASE: return None
            return self._with_structure_context(TradeSetup(
                setup_type="trend_pullback",
                direction=Direction.BUY_PUT.value,
                setup_strength=round(strength, 4),
                entry_zone_low=min(ema20, close),
                entry_zone_high=max(ema20, close),
                stop_loss_level=ema20 + max(atr * SETUP_TREND_STOP_ATR_MULT, close * SETUP_TREND_STOP_LTP_MULT),
                expected_move=max(atr * SETUP_TREND_MOVE_ATR_MULT, abs(close - ema20) * SETUP_TREND_MOVE_EMA_MULT),
                context={"adx": round(adx, 2), "ema20": round(ema20, 2), "ema50": round(ema50, 2)},
            ), structure_snapshot)
        return None

    def _breakout(self, *, df: pd.DataFrame, cache, regime_details: dict, structure_snapshot: dict) -> TradeSetup | None:
        if len(df) < S6_MIN_DF_LEN:
            return None
        close = float(df["close"].iloc[-1])
        lookback = df.tail(SETUP_BREAKOUT_SCAN_LOOKBACK)
        range_high = float(lookback["high"].iloc[:-1].max())
        range_low = float(lookback["low"].iloc[:-1].min())
        range_width = max(range_high - range_low, 0.01)
        atr = float((cache.atr_14.iloc[-1] if cache and cache.atr_14 is not None else (df["high"] - df["low"]).tail(14).mean()) or 0.0)
        bb_width = float((cache.bb_width.iloc[-1] if cache and cache.bb_width is not None else 0.0) or 0.0)
        vol_ratio = float((cache.vol_ratio.iloc[-1] if cache and cache.vol_ratio is not None else 1.0) or 1.0)
        adx = float((cache.adx.iloc[-1] if cache and cache.adx is not None else 0.0) or 0.0)
        if range_width > max(atr * SETUP_BREAKOUT_RANGE_ATR_MULT, close * SETUP_BREAKOUT_RANGE_LTP_MULT):
            return None
        if adx < SETUP_BREAKOUT_ADX_THRESHOLD:
            return None
        breakout_buffer = max(atr * SETUP_BREAKOUT_BUFFER_ATR_MULT, close * SETUP_BREAKOUT_BUFFER_LTP_MULT)

        if (close > range_high + breakout_buffer or float(df["high"].iloc[-1]) > range_high + breakout_buffer) and close >= range_high and vol_ratio >= SETUP_BREAKOUT_VOL_MULT_CALL:
            strength = min(0.93, SETUP_BREAKOUT_STRENGTH_BASE + vol_ratio * SETUP_BREAKOUT_STRENGTH_VOL_MULT + max(0.0, SETUP_BREAKOUT_STRENGTH_BB_THRESHOLD - bb_width / SETUP_BREAKOUT_STRENGTH_BB_DIVISOR) * SETUP_BREAKOUT_STRENGTH_BB_MULT)
            return self._with_structure_context(TradeSetup(
                setup_type="breakout",
                direction=Direction.BUY_CALL.value,
                setup_strength=round(strength, 4),
                entry_zone_low=range_high,
                entry_zone_high=close,
                stop_loss_level=range_low,
                expected_move=max(range_width * SETUP_BREAKOUT_MOVE_RANGE_MULT, atr * SETUP_BREAKOUT_MOVE_ATR_MULT),
                context={"range_high": round(range_high, 2), "range_low": round(range_low, 2), "vol_ratio": round(vol_ratio, 2), "adx": round(adx, 2)},
            ), structure_snapshot)
        if (close < range_low - breakout_buffer or float(df["low"].iloc[-1]) < range_low - breakout_buffer) and close <= range_low and vol_ratio >= SETUP_BREAKOUT_VOL_MULT_PUT:
            strength = min(0.93, SETUP_BREAKOUT_STRENGTH_BASE + vol_ratio * SETUP_BREAKOUT_STRENGTH_VOL_MULT + max(0.0, SETUP_BREAKOUT_STRENGTH_BB_THRESHOLD - bb_width / SETUP_BREAKOUT_STRENGTH_BB_DIVISOR) * SETUP_BREAKOUT_STRENGTH_BB_MULT)
            return self._with_structure_context(TradeSetup(
                setup_type="breakout",
                direction=Direction.BUY_PUT.value,
                setup_strength=round(strength, 4),
                entry_zone_low=close,
                entry_zone_high=range_low,
                stop_loss_level=range_high,
                expected_move=max(range_width * SETUP_BREAKOUT_MOVE_RANGE_MULT, atr * SETUP_BREAKOUT_MOVE_ATR_MULT),
                context={"range_high": round(range_high, 2), "range_low": round(range_low, 2), "vol_ratio": round(vol_ratio, 2), "adx": round(adx, 2)},
            ), structure_snapshot)
        return None

    def _mean_reversion(self, *, df: pd.DataFrame, cache, regime_details: dict, structure_snapshot: dict) -> TradeSetup | None:
        detailed = regime_details.get("detailed_regime", {}) or {}
        if str(detailed.get("label", "")).upper() != "RANGING":
            return None
        if len(df) < S6_MIN_DF_LEN:
            return None
        close = float(df["close"].iloc[-1])
        open_ = float(df["open"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        vwap = float((cache.vwap.iloc[-1] if cache and cache.vwap is not None else close) or close)
        atr = float((cache.atr_14.iloc[-1] if cache and cache.atr_14 is not None else (df["high"] - df["low"]).tail(14).mean()) or 0.0)
        stretch = close - vwap
        close_location = (close - float(df["low"].iloc[-1])) / max(float(df["high"].iloc[-1]) - float(df["low"].iloc[-1]), 0.01)
        if abs(stretch) < max(atr * SETUP_REVERSION_STRETCH_ATR_MULT, close * SETUP_REVERSION_STRETCH_LTP_MULT):
            return None

        if stretch < 0 and (close > open_ or close >= prev_close or close_location >= SETUP_REVERSION_CLOSE_LOCATION_THRESHOLD):
            strength = min(0.90, SETUP_REVERSION_STRENGTH_BASE + min(abs(stretch) / max(atr, SETUP_REVERSION_MIN_ATR), SETUP_REVERSION_STRETCH_RATIO_MAX) * SETUP_REVERSION_STRENGTH_STRETCH_MULT)
            return self._with_structure_context(TradeSetup(
                setup_type="mean_reversion",
                direction=Direction.BUY_CALL.value,
                setup_strength=round(strength, 4),
                entry_zone_low=close - atr * SETUP_REVERSION_ENTRY_ATR_MULT,
                entry_zone_high=close,
                stop_loss_level=close - atr * SETUP_REVERSION_STOP_ATR_MULT,
                expected_move=max(abs(stretch) * SETUP_REVERSION_MOVE_STRETCH_MULT, atr * SETUP_REVERSION_MOVE_ATR_MULT),
                context={"vwap": round(vwap, 2), "stretch": round(stretch, 2)},
            ), structure_snapshot)
        if stretch > 0 and (close < open_ or close <= prev_close or close_location <= SETUP_REVERSION_CLOSE_LOCATION_THRESHOLD):
            strength = min(0.90, SETUP_REVERSION_STRENGTH_BASE + min(abs(stretch) / max(atr, SETUP_REVERSION_MIN_ATR), SETUP_REVERSION_STRETCH_RATIO_MAX) * SETUP_REVERSION_STRENGTH_STRETCH_MULT)
            return self._with_structure_context(TradeSetup(
                setup_type="mean_reversion",
                direction=Direction.BUY_PUT.value,
                setup_strength=round(strength, 4),
                entry_zone_low=close,
                entry_zone_high=close + atr * SETUP_REVERSION_ENTRY_ATR_MULT,
                stop_loss_level=close + atr * SETUP_REVERSION_STOP_ATR_MULT,
                expected_move=max(abs(stretch) * SETUP_REVERSION_MOVE_STRETCH_MULT, atr * SETUP_REVERSION_MOVE_ATR_MULT),
                context={"vwap": round(vwap, 2), "stretch": round(stretch, 2)},
            ), structure_snapshot)
        return None

    @staticmethod
    def _with_structure_context(setup: TradeSetup, structure_snapshot: dict) -> TradeSetup | None:
        validation = (structure_snapshot.get("entry_validation", {}) or {}).get("by_direction", {})
        validation_root = structure_snapshot.get("entry_validation", {}) or {}
        direction_validation = validation.get(setup.direction, {})
        event = structure_snapshot.get("liquidity_event", {}) or {}
        structure_state = structure_snapshot.get("structure_state", {}) or {}
        range_position = float(validation_root.get("range_position", 0.5) or 0.5)
        validation_reasons = [str(reason) for reason in direction_validation.get("reasons", [])]
        strength_penalty = 0.0
        if not direction_validation.get("valid", False):
            only_structure_gate = bool(validation_reasons) and all(
                "liquidity sweep" in reason or "structure shift" in reason
                for reason in validation_reasons
            )
            aligned_bias = (
                (setup.direction == Direction.BUY_CALL.value and structure_state.get("bias") == "BULLISH")
                or (setup.direction == Direction.BUY_PUT.value and structure_state.get("bias") == "BEARISH")
            )
            continuation_ok = (
                setup.setup_type in {"trend_pullback", "vote_aligned"}
                and aligned_bias
                and only_structure_gate
                and not validation_root.get("middle_range", False)
            )
            breakout_ok = (
                setup.setup_type == "breakout"
                and only_structure_gate
                and not validation_root.get("middle_range", False)
                and (
                    (setup.direction == Direction.BUY_CALL.value and range_position >= SETUP_STRUCTURE_VALIDATION_BOS_THRESHOLD_HIGH)
                    or (setup.direction == Direction.BUY_PUT.value and range_position <= SETUP_STRUCTURE_VALIDATION_BOS_THRESHOLD_LOW)
                )
            )
            mean_reversion_ok = (
                setup.setup_type == "mean_reversion"
                and (
                    (setup.direction == Direction.BUY_CALL.value and range_position <= SETUP_STRUCTURE_VALIDATION_REVERSION_THRESHOLD_LOW)
                    or (setup.direction == Direction.BUY_PUT.value and range_position >= SETUP_STRUCTURE_VALIDATION_REVERSION_THRESHOLD_HIGH)
                )
            )
            # Preserve setup candidates and let the explicit runner/ML gates
            # decide based on degraded setup strength rather than silently
            # killing the candle here.
            if continuation_ok or breakout_ok or mean_reversion_ok:
                strength_penalty += SETUP_STRUCTURE_PENALTY_SOFT
            else:
                strength_penalty += SETUP_STRUCTURE_PENALTY_HARD
        boost = 0.0
        if event.get("direction") == setup.direction and event.get("type") != "NONE":
            boost += SETUP_STRUCTURE_BOOST_EVENT
        if structure_state.get("bos_direction") == setup.direction:
            boost += SETUP_STRUCTURE_BOOST_BOS
        if structure_state.get("choch_direction") == setup.direction:
            boost += SETUP_STRUCTURE_BOOST_CHOCH
        if (
            setup.setup_type in {"trend_pullback", "vote_aligned"}
            and boost == 0.0
            and (
                (setup.direction == Direction.BUY_CALL.value and structure_state.get("bias") == "BULLISH")
                or (setup.direction == Direction.BUY_PUT.value and structure_state.get("bias") == "BEARISH")
            )
        ):
            boost += SETUP_STRUCTURE_BOOST_BIAS
        if setup.setup_type == "mean_reversion":
            if setup.direction == Direction.BUY_CALL.value and range_position <= SETUP_REVERSION_ZONE_THRESHOLD_LOW:
                boost += SETUP_STRUCTURE_BOOST_REVERSION
            if setup.direction == Direction.BUY_PUT.value and range_position >= SETUP_REVERSION_ZONE_THRESHOLD_HIGH:
                boost += SETUP_STRUCTURE_BOOST_REVERSION
        if setup.setup_type == "breakout" and validation_root.get("middle_range", False) is False:
            boost += SETUP_STRUCTURE_BOOST_BREAKOUT
        if validation_root.get("avoid", False):
            strength_penalty += SETUP_STRUCTURE_PENALTY_AVOID
        if validation_root.get("middle_range", False):
            strength_penalty += SETUP_STRUCTURE_PENALTY_MIDDLE
        adjusted_strength = min(0.97, round(setup.setup_strength + boost - strength_penalty, 4))
        if adjusted_strength < ML_RANK_TIER_LOW:
            return None
        return TradeSetup(
            setup_type=setup.setup_type,
            direction=setup.direction,
            setup_strength=adjusted_strength,
            entry_zone_low=setup.entry_zone_low,
            entry_zone_high=setup.entry_zone_high,
            stop_loss_level=setup.stop_loss_level,
            expected_move=setup.expected_move,
            context={
                **setup.context,
                "market_structure": structure_snapshot,
            },
        )

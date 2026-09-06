"""
agents_code/agent2_strategy/s34_elliott_wave.py — Elliott Wave Strategy
=========================================================================

Intent for SignalForge:
- detect swing pivots with a lightweight ZigZag
- map the most recent pivots into a probable Elliott sequence
- trigger only on actionable completion states:
  1. Wave 4 confirmed -> enter Wave 5 continuation
  2. A-B-C corrective completion -> reversal entry

This implementation is deliberately conservative. It is not a full wave-count
engine. It is a structured swing-pattern confirmer designed to contribute one
vote into the existing multi-strategy stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:
    from core.models import Direction
except ImportError:
    class Direction:
        NONE = "NONE"
        BUY_CALL = "BUY_CALL"
        BUY_PUT = "BUY_PUT"


MIN_CANDLES = 40
ZIGZAG_PCT = 0.15

FIB_W2_MIN = 0.382
FIB_W2_MAX = 0.786
FIB_W4_MIN = 0.146
FIB_W4_MAX = 0.500
FIB_B_MIN = 0.382
FIB_B_MAX = 0.786
FIB_TOL = 0.05

MIN_W3_TO_W1_RATIO = 1.0
MIN_W5_EXTENSION_RATIO = 0.618
MAX_W5_EXTENSION_RATIO = 1.272

RSI_BULL_MIN = 48.0
RSI_BEAR_MAX = 52.0


@dataclass(frozen=True)
class Pivot:
    idx: int
    price: float
    kind: str  # "HIGH" | "LOW"


class ElliottWaveStrategy:
    name = "ElliottWave"

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None, **kwargs) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if df is None or len(df) < MIN_CANDLES:
            return none
        try:
            return self._evaluate(df)
        except Exception:
            return none

    def _evaluate(self, df: pd.DataFrame) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        # Point-in-time safety: require confirmed pivots to avoid premature counter-trend triggers on unconfirmed tail candles
        pivots = self._zigzag(df, include_unconfirmed=False)
        if len(pivots) < 5:
            return none

        current_close = float(df["close"].iloc[-1])
        current_rsi = self._rsi(df["close"])

        impulse_up = self._check_wave5_setup(pivots[-5:], current_close, bullish=True)
        if impulse_up:
            return self._build_signal(Direction.BUY_CALL, impulse_up, current_rsi)

        impulse_down = self._check_wave5_setup(pivots[-5:], current_close, bullish=False)
        if impulse_down:
            return self._build_signal(Direction.BUY_PUT, impulse_down, current_rsi)

        if len(pivots) >= 4:
            corrective_up = self._check_abc_completion(pivots[-4:], current_close, bullish=True)
            if corrective_up:
                return self._build_signal(Direction.BUY_CALL, corrective_up, current_rsi)

            corrective_down = self._check_abc_completion(pivots[-4:], current_close, bullish=False)
            if corrective_down:
                return self._build_signal(Direction.BUY_PUT, corrective_down, current_rsi)

        return none

    def _zigzag(self, df: pd.DataFrame, include_unconfirmed: bool = False) -> list[Pivot]:
        highs = [float(v) for v in df["high"].values]
        lows = [float(v) for v in df["low"].values]
        closes = [float(v) for v in df["close"].values]
        threshold = ZIGZAG_PCT / 100.0

        if not closes:
            return []

        pivots: list[Pivot] = []
        candidate_price = closes[0]
        candidate_idx = 0
        direction: str | None = None

        for i in range(1, len(closes)):
            hi = highs[i]
            lo = lows[i]

            if direction is None:
                up_move = (hi - candidate_price) / max(abs(candidate_price), 1e-9)
                down_move = (candidate_price - lo) / max(abs(candidate_price), 1e-9)
                if up_move >= threshold:
                    pivots.append(Pivot(candidate_idx, candidate_price, "LOW"))
                    direction = "UP"
                    candidate_price = hi
                    candidate_idx = i
                elif down_move >= threshold:
                    pivots.append(Pivot(candidate_idx, candidate_price, "HIGH"))
                    direction = "DOWN"
                    candidate_price = lo
                    candidate_idx = i
                continue

            if direction == "UP":
                if hi >= candidate_price:
                    candidate_price = hi
                    candidate_idx = i
                retrace = (candidate_price - lo) / max(abs(candidate_price), 1e-9)
                if retrace >= threshold:
                    pivots.append(Pivot(candidate_idx, candidate_price, "HIGH"))
                    direction = "DOWN"
                    candidate_price = lo
                    candidate_idx = i
            else:
                if lo <= candidate_price:
                    candidate_price = lo
                    candidate_idx = i
                bounce = (hi - candidate_price) / max(abs(candidate_price), 1e-9)
                if bounce >= threshold:
                    pivots.append(Pivot(candidate_idx, candidate_price, "LOW"))
                    direction = "UP"
                    candidate_price = hi
                    candidate_idx = i

        if include_unconfirmed:
            if direction == "UP":
                pivots.append(Pivot(candidate_idx, candidate_price, "HIGH"))
            elif direction == "DOWN":
                pivots.append(Pivot(candidate_idx, candidate_price, "LOW"))

        cleaned: list[Pivot] = []
        for pivot in pivots:
            if cleaned and cleaned[-1].kind == pivot.kind:
                prev = cleaned[-1]
                if pivot.kind == "HIGH" and pivot.price >= prev.price:
                    cleaned[-1] = pivot
                elif pivot.kind == "LOW" and pivot.price <= prev.price:
                    cleaned[-1] = pivot
            else:
                cleaned.append(pivot)
        return cleaned

    def _check_wave5_setup(
        self,
        pivots: list[Pivot],
        current_close: float,
        *,
        bullish: bool,
    ) -> Optional[dict]:
        if len(pivots) != 5:
            return None
        p0, p1, p2, p3, p4 = pivots

        if bullish:
            expected = ["LOW", "HIGH", "LOW", "HIGH", "LOW"]
            if [p.kind for p in pivots] != expected:
                return None
            w1 = p1.price - p0.price
            w2 = p1.price - p2.price
            w3 = p3.price - p2.price
            w4 = p3.price - p4.price
            if min(w1, w2, w3, w4) <= 0:
                return None
            w2_retrace = w2 / w1
            w4_retrace = w4 / w3
            if not self._within(w2_retrace, FIB_W2_MIN, FIB_W2_MAX):
                return None
            if not self._within(w4_retrace, FIB_W4_MIN, FIB_W4_MAX):
                return None
            if p2.price <= p0.price:
                return None
            if p4.price <= p1.price:
                return None
            if w3 < max(w1, 1e-9) * MIN_W3_TO_W1_RATIO:
                return None
            if current_close <= p4.price:
                return None
            projected_w5_min = p4.price + (w1 * MIN_W5_EXTENSION_RATIO)
            projected_w5_max = p4.price + (w1 * MAX_W5_EXTENSION_RATIO)
            return {
                "pattern": "wave5_continuation",
                "wave_side": "bullish",
                "entry_level": current_close,
                "confirmation_level": p4.price,
                "target_level": projected_w5_min,
                "stretch_target": projected_w5_max,
                "w2_retrace": w2_retrace,
                "w4_retrace": w4_retrace,
                "w1_size": w1,
                "w3_size": w3,
                "pivot_span": p4.idx - p0.idx,
            }

        expected = ["HIGH", "LOW", "HIGH", "LOW", "HIGH"]
        if [p.kind for p in pivots] != expected:
            return None
        w1 = p0.price - p1.price
        w2 = p2.price - p1.price
        w3 = p2.price - p3.price
        w4 = p4.price - p3.price
        if min(w1, w2, w3, w4) <= 0:
            return None
        w2_retrace = w2 / w1
        w4_retrace = w4 / w3
        if not self._within(w2_retrace, FIB_W2_MIN, FIB_W2_MAX):
            return None
        if not self._within(w4_retrace, FIB_W4_MIN, FIB_W4_MAX):
            return None
        if p2.price >= p0.price:
            return None
        if p4.price >= p1.price:
            return None
        if w3 < max(w1, 1e-9) * MIN_W3_TO_W1_RATIO:
            return None
        if current_close >= p4.price:
            return None
        projected_w5_min = p4.price - (w1 * MIN_W5_EXTENSION_RATIO)
        projected_w5_max = p4.price - (w1 * MAX_W5_EXTENSION_RATIO)
        return {
            "pattern": "wave5_continuation",
            "wave_side": "bearish",
            "entry_level": current_close,
            "confirmation_level": p4.price,
            "target_level": projected_w5_min,
            "stretch_target": projected_w5_max,
            "w2_retrace": w2_retrace,
            "w4_retrace": w4_retrace,
            "w1_size": w1,
            "w3_size": w3,
            "pivot_span": p4.idx - p0.idx,
        }

    def _check_abc_completion(
        self,
        pivots: list[Pivot],
        current_close: float,
        *,
        bullish: bool,
    ) -> Optional[dict]:
        if len(pivots) != 4:
            return None
        p0, pa, pb, pc = pivots

        if bullish:
            expected = ["HIGH", "LOW", "HIGH", "LOW"]
            if [p.kind for p in pivots] != expected:
                return None
            wave_a = p0.price - pa.price
            wave_b = pb.price - pa.price
            wave_c = pb.price - pc.price
            if min(wave_a, wave_b, wave_c) <= 0:
                return None
            b_retrace = wave_b / wave_a
            if not self._within(b_retrace, FIB_B_MIN, FIB_B_MAX):
                return None
            if pc.price > pa.price:
                return None
            if current_close <= pc.price:
                return None
            return {
                "pattern": "abc_reversal",
                "wave_side": "bullish",
                "entry_level": current_close,
                "confirmation_level": pc.price,
                "target_level": current_close + wave_a,
                "stretch_target": current_close + (wave_a * 1.618),
                "b_retrace": b_retrace,
                "wave_a_size": wave_a,
                "wave_c_size": wave_c,
                "pivot_span": pc.idx - p0.idx,
            }

        expected = ["LOW", "HIGH", "LOW", "HIGH"]
        if [p.kind for p in pivots] != expected:
            return None
        wave_a = pa.price - p0.price
        wave_b = pa.price - pb.price
        wave_c = pc.price - pb.price
        if min(wave_a, wave_b, wave_c) <= 0:
            return None
        b_retrace = wave_b / wave_a
        if not self._within(b_retrace, FIB_B_MIN, FIB_B_MAX):
            return None
        if pc.price < pa.price:
            return None
        if current_close >= pc.price:
            return None
        return {
            "pattern": "abc_reversal",
            "wave_side": "bearish",
            "entry_level": current_close,
            "confirmation_level": pc.price,
            "target_level": current_close - wave_a,
            "stretch_target": current_close - (wave_a * 1.618),
            "b_retrace": b_retrace,
            "wave_a_size": wave_a,
            "wave_c_size": wave_c,
            "pivot_span": pc.idx - p0.idx,
        }

    def _build_signal(self, direction: str, info: dict, rsi: float) -> dict:
        confidence = 0.62
        if info["pattern"] == "wave5_continuation":
            confidence += 0.08
            confidence += min(0.04, max(0.0, info.get("w3_size", 0.0) / max(info.get("w1_size", 1e-9), 1e-9) - 1.0) * 0.03)
        else:
            confidence += 0.04

        if direction == Direction.BUY_CALL and rsi >= RSI_BULL_MIN:
            confidence += 0.03
        elif direction == Direction.BUY_PUT and rsi <= RSI_BEAR_MAX:
            confidence += 0.03

        confidence = round(min(0.82, confidence), 4)
        return {
            "direction": direction,
            "confidence": confidence,
            "name": self.name,
            "meta": {
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in info.items()},
                "rsi": round(rsi, 2),
                "zigzag_pct": ZIGZAG_PCT,
            },
        }

    @staticmethod
    def _within(value: float, low: float, high: float) -> bool:
        return (low - FIB_TOL) <= value <= (high + FIB_TOL)

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float:
        try:
            diff = close.diff().dropna()
            gain = diff.clip(lower=0).ewm(span=period, adjust=False).mean()
            loss = (-diff).clip(lower=0).ewm(span=period, adjust=False).mean()
            rs = gain.iloc[-1] / max(loss.iloc[-1], 1e-10)
            return float(100 - (100 / (1 + rs)))
        except Exception:
            return 50.0

"""
Context Alignment Score (CAS) Module
====================================
Computes a unified real-time alignment score across:
1. Multi-Timeframe (MTF) Trend (5m, 15m, Daily EMA / SuperTrend alignment)
2. Option Microstructure & Flow (PCR velocity, Put/Call wall shifts, volume ratio)
3. Technical Momentum (VWAP stance, RSI momentum, closing candle expansion)
4. Strategy Consensus & Directional Symmetry

Output:
- cas_score (0.0 to 1.0): Unified context alignment strength.
- cas_verdict: "STRONG_BULLISH", "MODERATE_BULLISH", "NEUTRAL", "MODERATE_BEARISH", "STRONG_BEARISH"
- allows_structural_override: True if CAS is strong enough (>= 0.65) to override stale morning bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd


@dataclass(frozen=True)
class CASResult:
    score: float                         # 0.0 to 1.0
    direction: str                       # BUY_CALL, BUY_PUT, NONE
    verdict: str                         # STRONG_BULLISH, MODERATE_BULLISH, NEUTRAL, MODERATE_BEARISH, STRONG_BEARISH
    allows_structural_override: bool     # True if strong enough to trade dynamic regime flips
    components: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "direction": self.direction,
            "verdict": self.verdict,
            "allows_structural_override": self.allows_structural_override,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "notes": self.notes,
        }


class ContextAlignmentScore:
    """
    Evaluates multi-dimensional alignment across price action, order flow,
    and derivatives structure.
    """

    OVERRIDE_THRESHOLD = 0.65  # CAS >= 0.65 overrides static morning bias

    @classmethod
    def evaluate(
        cls,
        *,
        direction: str,
        df: Optional[pd.DataFrame] = None,
        context: Optional[dict[str, Any]] = None,
        market_intel: Optional[dict[str, Any]] = None,
        votes: int = 0,
        ml_rank_score: float = 0.0,
        is_expiry_session: bool = False,
    ) -> CASResult:
        context = context or {}
        market_intel = market_intel or {}
        direction = str(direction or "NONE").upper()

        if direction not in {"BUY_CALL", "BUY_PUT"}:
            return CASResult(
                score=0.0,
                direction="NONE",
                verdict="NEUTRAL",
                allows_structural_override=False,
                components={},
                notes=["No directional bias provided"],
            )

        is_call = direction == "BUY_CALL"
        notes: list[str] = []
        components: dict[str, float] = {}

        # ── 1. Multi-Timeframe Alignment Component (1m, 5m, 10m, 15m, 30m) (Weight: 25%) ─
        # Combines:
        # - 1m micro-momentum (fastest confirmation of turning point)
        # - 5m & 10m structural trend (primary intraday swing)
        # - 15m & 30m anchor trend (macro institutional flow)
        mtf_1m_bias  = str(context.get("bias_1m") or context.get("1m_trend") or context.get("micro_trend") or "").upper()
        mtf_5m_bias  = str(context.get("bias_5m") or context.get("5m_trend") or context.get("hybrid_5m", {}).get("bias") or "").upper()
        mtf_10m_bias = str(context.get("bias_10m") or context.get("10m_trend") or "").upper()
        mtf_15m_bias = str(context.get("mtf_bias") or context.get("bias_15m") or context.get("15m_trend") or context.get("structure_bias") or "").upper()
        mtf_30m_bias = str(context.get("bias_30m") or context.get("30m_trend") or "").upper()

        tf_scores = []
        # 1m Evaluation
        if mtf_1m_bias in {"BULLISH", "BEARISH"}:
            tf_scores.append(1.0 if ((is_call and mtf_1m_bias == "BULLISH") or (not is_call and mtf_1m_bias == "BEARISH")) else 0.0)
            notes.append(f"1m {mtf_1m_bias.lower()}")
        elif df is not None and len(df) >= 3:
            # Infer 1m micro-momentum from recent candle action
            c_diff = df["close"].iloc[-1] - df["close"].iloc[-2]
            if (is_call and c_diff > 0) or (not is_call and c_diff < 0):
                tf_scores.append(1.0)
                notes.append("1m tick aligned")
            else:
                tf_scores.append(0.3)

        # 5m Evaluation
        if mtf_5m_bias in {"BULLISH", "BEARISH"}:
            tf_scores.append(1.0 if ((is_call and mtf_5m_bias == "BULLISH") or (not is_call and mtf_5m_bias == "BEARISH")) else 0.0)
            notes.append(f"5m {mtf_5m_bias.lower()}")

        # 10m Evaluation
        if mtf_10m_bias in {"BULLISH", "BEARISH"}:
            tf_scores.append(1.0 if ((is_call and mtf_10m_bias == "BULLISH") or (not is_call and mtf_10m_bias == "BEARISH")) else 0.0)
            notes.append(f"10m {mtf_10m_bias.lower()}")

        # 15m Evaluation
        if mtf_15m_bias in {"BULLISH", "BEARISH"}:
            tf_scores.append(1.0 if ((is_call and mtf_15m_bias == "BULLISH") or (not is_call and mtf_15m_bias == "BEARISH")) else 0.0)
            notes.append(f"15m {mtf_15m_bias.lower()}")

        # 30m Evaluation
        if mtf_30m_bias in {"BULLISH", "BEARISH"}:
            tf_scores.append(1.0 if ((is_call and mtf_30m_bias == "BULLISH") or (not is_call and mtf_30m_bias == "BEARISH")) else 0.0)
            notes.append(f"30m {mtf_30m_bias.lower()}")

        if tf_scores:
            mtf_score = sum(tf_scores) / len(tf_scores)
        else:
            mtf_score = 0.50
        components["mtf"] = mtf_score

        # ── 2. Option Microstructure / Flow Component (Weight: 30%) ─────────────
        flow_score = 0.50
        pcr = float(context.get("pcr") or market_intel.get("pcr") or 1.0)
        pcr_trend15 = float(context.get("pcr_trend15") or 0.0)
        wall_signal = str(context.get("wall_signal") or market_intel.get("wall_signal") or "").upper()

        if is_call:
            # Bullish flow: PCR rising or Put wall stronger / rising
            flow_points = 0.50
            if pcr_trend15 > 0.10:
                flow_points += 0.25
                notes.append(f"PCR rising fast (+{pcr_trend15:.2f})")
            elif pcr > 1.15:
                flow_points += 0.15
            if "CALL" in wall_signal or "BULLISH" in wall_signal:
                flow_points += 0.25
                notes.append("Put wall support dominant")
            flow_score = min(1.0, flow_points)
        else:
            # Bearish flow: PCR dropping or Call wall resistance dominant
            flow_points = 0.50
            if pcr_trend15 < -0.10:
                flow_points += 0.25
                notes.append(f"PCR falling fast ({pcr_trend15:.2f})")
            elif pcr < 0.85:
                flow_points += 0.15
            if "PUT" in wall_signal or "BEARISH" in wall_signal:
                flow_points += 0.25
                notes.append("Call wall resistance dominant")
            flow_score = min(1.0, flow_points)
        components["flow"] = flow_score

        # ── 3. Technical Momentum & VWAP Stance (Weight: 25%) ──────────────────
        tech_score = 0.50
        if df is not None and len(df) >= 10:
            try:
                closes = df["close"].values
                vols = df["volume"].values
                spot = closes[-1]
                tp = (df["high"] + df["low"] + df["close"]) / 3
                vwap = float((tp * vols).cumsum().iloc[-1] / max(vols.cumsum().iloc[-1], 1e-9))
                
                tech_points = 0.50
                # VWAP position
                if is_call and spot > vwap:
                    tech_points += 0.25
                    notes.append("Spot above VWAP")
                elif not is_call and spot < vwap:
                    tech_points += 0.25
                    notes.append("Spot below VWAP")
                else:
                    tech_points -= 0.15

                # 3-candle short term velocity
                roc3 = (closes[-1] - closes[-3]) / max(closes[-3], 1.0) * 100.0
                if is_call and roc3 > 0.10:
                    tech_points += 0.25
                elif not is_call and roc3 < -0.10:
                    tech_points += 0.25
                tech_score = max(0.0, min(1.0, tech_points))
            except Exception:
                tech_score = 0.50
        components["tech_momentum"] = tech_score

        # ── 4. Consensus & ML Conviction (Weight: 20%) ─────────────────────────
        consensus_score = min(1.0, max(0.0, (votes / 8.0) * 0.5 + float(ml_rank_score) * 0.5))
        components["consensus"] = consensus_score

        # ── Calculate Composite Score ──────────────────────────────────────────
        total_score = (
            components["mtf"] * 0.25
            + components["flow"] * 0.30
            + components["tech_momentum"] * 0.25
            + components["consensus"] * 0.20
        )

        # Expiry boost: on 0DTE expiry afternoon, strong momentum + flow alignment is extra explosive
        if is_expiry_session and total_score >= 0.58:
            total_score = min(1.0, total_score + 0.07)
            notes.append("0DTE expiry convexity boost applied")

        total_score = round(total_score, 4)

        if total_score >= 0.75:
            verdict = "STRONG_BULLISH" if is_call else "STRONG_BEARISH"
        elif total_score >= 0.60:
            verdict = "MODERATE_BULLISH" if is_call else "MODERATE_BEARISH"
        else:
            verdict = "NEUTRAL"

        allows_override = total_score >= cls.OVERRIDE_THRESHOLD

        return CASResult(
            score=total_score,
            direction=direction,
            verdict=verdict,
            allows_structural_override=allows_override,
            components=components,
            notes=notes,
        )

"""
core/regime/engine.py — Explainable Market Regime Classification Engine

Classifies market state into 6 distinct, explainable regimes:
1. TREND           — Strong directional drift (ADX > 22, clean EMA alignment)
2. RANGE           — Mean-reverting compression around VWAP (Choppiness > 50, ADX < 20)
3. BREAKOUT        — Expanding volatility or range breakout (ORB breach, BB expansion)
4. HIGH_VOLATILITY — Extreme excursions, wide bars (ATR > 1.8x baseline)
5. LOW_VOLATILITY  — Muted consolidation, tight ATR (ATR < 0.65x baseline)
6. ABNORMAL        — Single-bar spike > 3.5 ATR, volume anomaly, erratic spreads
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
import pandas as pd


class MarketRegime(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    BREAKOUT = "BREAKOUT"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    ABNORMAL = "ABNORMAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RegimeDetails:
    regime: MarketRegime
    confidence: float = 0.80
    adx: float = 20.0
    atr: float = 25.0
    atr_ratio: float = 1.0
    choppiness: float = 50.0
    dist_vwap_atr: float = 0.5
    reasons: list[str] = field(default_factory=list)
    is_tradeable: bool = True


class MarketRegimeEngine:
    """
    Computes explainable, point-in-time market regime on 5-minute candle history.
    """

    def __init__(self, baseline_window: int = 50):
        self.baseline_window = baseline_window

    def classify(self, df: pd.DataFrame, current_price: Optional[float] = None) -> RegimeDetails:
        """
        Classifies market regime from completed candle history.
        Zero future lookahead.
        """
        if df is None or len(df) < 14:
            return RegimeDetails(
                regime=MarketRegime.RANGE,
                confidence=0.50,
                adx=15.0,
                atr=10.0,
                atr_ratio=1.0,
                choppiness=50.0,
                dist_vwap_atr=0.0,
                reasons=["INSUFFICIENT_HISTORY (<14 candles)"],
                is_tradeable=True,
            )

        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        v = df["volume"].astype(float).replace(0, 1.0)
        curr_p = float(current_price or c.iloc[-1])

        # 1. ATR
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        prev_atr = float(tr.shift(1).rolling(14, min_periods=1).mean().iloc[-1]) if len(tr) > 1 else float(tr.iloc[0])
        atr = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        baseline_atr = float(tr.rolling(min(len(tr), self.baseline_window), min_periods=5).mean().iloc[-1])
        atr_ratio = round(atr / max(baseline_atr, 1e-6), 2)

        # 2. ADX
        plus_dm = h.diff().where(lambda x: (x > 0) & (x > -l.diff()), 0.0)
        minus_dm = (-l.diff()).where(lambda x: (x > 0) & (x > h.diff()), 0.0)
        atr_series = tr.rolling(14, min_periods=1).mean().replace(0, 1e-6)
        plus_di = 100.0 * (plus_dm.rolling(14, min_periods=1).mean() / atr_series)
        minus_di = 100.0 * (minus_dm.rolling(14, min_periods=1).mean() / atr_series)
        denom = (plus_di + minus_di).replace(0, 1e-6)
        dx = 100.0 * (plus_di - minus_di).abs() / denom
        adx = float(dx.rolling(14, min_periods=1).mean().iloc[-1])

        # 3. Choppiness Index (14 periods)
        n = min(len(df), 14)
        sum_tr = tr.rolling(n, min_periods=1).sum().iloc[-1]
        max_h = h.rolling(n, min_periods=1).max().iloc[-1]
        min_l = l.rolling(n, min_periods=1).min().iloc[-1]
        range_hl = max_h - min_l
        if n > 1 and range_hl > 0 and sum_tr > 0:
            choppiness = float(100.0 * np.log10(sum_tr / range_hl) / np.log10(n))
        else:
            choppiness = 50.0


        # 4. VWAP & EMAs
        cum_vol = v.cumsum()
        cum_pv = (c * v).cumsum()
        vwap = float((cum_pv / cum_vol).iloc[-1])
        dist_vwap_atr = round(abs(curr_p - vwap) / max(atr, 1.0), 2)

        ema9 = float(c.ewm(span=9, adjust=False).mean().iloc[-1])
        ema20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(c.ewm(span=min(50, len(c)), adjust=False).mean().iloc[-1])

        last_bar_range = float(tr.iloc[-1])
        reasons = []

        # ── 5. REGIME EVALUATION (Strict Priority Order) ──────────────────────────
        # Check A: ABNORMAL (Wild spike, erratic candle)
        ref_atr = prev_atr if prev_atr > 0 else atr
        if last_bar_range > 3.0 * ref_atr or (baseline_atr > 0 and last_bar_range > 3.5 * baseline_atr):
            reasons.append(f"EXTREME_SPIKE_DETECTED ({last_bar_range:.1f} > 3x baseline ATR {ref_atr:.1f})")
            return RegimeDetails(
                regime=MarketRegime.ABNORMAL,
                confidence=0.90,
                adx=round(adx, 1),
                atr=round(atr, 2),
                atr_ratio=atr_ratio,
                choppiness=round(choppiness, 1),
                dist_vwap_atr=dist_vwap_atr,
                reasons=reasons,
                is_tradeable=False,
            )

        # Check B: HIGH VOLATILITY
        if atr_ratio >= 1.80 or last_bar_range > 2.2 * atr:
            reasons.append(f"HIGH_VOLATILITY_EXPANSION (ATR ratio: {atr_ratio}x baseline)")
            return RegimeDetails(
                regime=MarketRegime.HIGH_VOLATILITY,
                confidence=0.85,
                adx=round(adx, 1),
                atr=round(atr, 2),
                atr_ratio=atr_ratio,
                choppiness=round(choppiness, 1),
                dist_vwap_atr=dist_vwap_atr,
                reasons=reasons,
                is_tradeable=True,
            )

        # Check C: LOW VOLATILITY / SLEEP
        if atr_ratio <= 0.65 and adx < 18:
            reasons.append(f"COMPRESSED_LOW_VOLATILITY (ATR ratio: {atr_ratio}x, ADX {adx:.1f})")
            return RegimeDetails(
                regime=MarketRegime.LOW_VOLATILITY,
                confidence=0.80,
                adx=round(adx, 1),
                atr=round(atr, 2),
                atr_ratio=atr_ratio,
                choppiness=round(choppiness, 1),
                dist_vwap_atr=dist_vwap_atr,
                reasons=reasons,
                is_tradeable=False,  # Muted opportunity
            )

        # Check D: BREAKOUT (09:15-09:30 ORB expansion or BB breach with volume)
        if len(df) >= 4:
            orb_high = float(h.iloc[:3].max())
            orb_low = float(l.iloc[:3].min())
            if (curr_p > orb_high or curr_p < orb_low) and dist_vwap_atr > 1.2:
                reasons.append(f"ORB_BREAKOUT_ACTIVE (Outside initial 15m range [{orb_low:.1f}, {orb_high:.1f}])")
                return RegimeDetails(
                    regime=MarketRegime.BREAKOUT,
                    confidence=0.80,
                    adx=round(adx, 1),
                    atr=round(atr, 2),
                    atr_ratio=atr_ratio,
                    choppiness=round(choppiness, 1),
                    dist_vwap_atr=dist_vwap_atr,
                    reasons=reasons,
                    is_tradeable=True,
                )

        # Check E: TREND (Sustained directional momentum)
        is_bull_trend = (curr_p > vwap) and (ema9 > ema20 > ema50) and (adx >= 22)
        is_bear_trend = (curr_p < vwap) and (ema9 < ema20 < ema50) and (adx >= 22)
        if is_bull_trend or is_bear_trend:
            dir_str = "BULLISH" if is_bull_trend else "BEARISH"
            reasons.append(f"STRONG_{dir_str}_TREND (ADX {adx:.1f}, aligned EMAs)")
            return RegimeDetails(
                regime=MarketRegime.TREND,
                confidence=round(min(0.95, 0.60 + (adx / 100.0)), 2),
                adx=round(adx, 1),
                atr=round(atr, 2),
                atr_ratio=atr_ratio,
                choppiness=round(choppiness, 1),
                dist_vwap_atr=dist_vwap_atr,
                reasons=reasons,
                is_tradeable=True,
            )

        # Check F: RANGE / CONSOLIDATION (Default when non-trending)
        reasons.append(f"RANGE_BOUND_CONSOLIDATION (Chop {choppiness:.1f}, ADX {adx:.1f})")
        return RegimeDetails(
            regime=MarketRegime.RANGE,
            confidence=round(min(0.90, choppiness / 100.0), 2),
            adx=round(adx, 1),
            atr=round(atr, 2),
            atr_ratio=atr_ratio,
            choppiness=round(choppiness, 1),
            dist_vwap_atr=dist_vwap_atr,
            reasons=reasons,
            is_tradeable=True,
        )

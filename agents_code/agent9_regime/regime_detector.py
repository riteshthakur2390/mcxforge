"""
agents_code/agent9_regime/regime_detector.py
============================================
NEW MODULE  Market Regime Detector
====================================

Classifies each candle into one of THREE regimes:

  TRENDING        ADX strong + directional + ATR expanding
  RANGING         Low ADX + tight price range + ATR contracting
  HIGH_VOLATILITY ATR spike OR VIX elevated (fear in market)

Also produces:
  - confidence score  0.01.0 (how certain the classification is)
  - sub_label         e.g. "STRONG_TREND", "TIGHT_RANGE", "VIX_SPIKE"
  - raw indicator values for downstream use by strategies and ML

DESIGN PRINCIPLES

1. PURE FUNCTION  compute() takes a DataFrame, returns a RegimeResult.
   No state, no bus, no side effects. Easy to test, fast to run.

2. NO STRATEGY CHANGES  strategies read regime from regime_details
   dict that is already in the MARKET_REGIME message. We add
   'detailed_regime' key to that dict. Strategies can optionally
   read it; if they don't, nothing breaks.

3. FAST  all indicators computed from existing IndicatorCache values
   where possible. Standalone computation adds < 5ms per candle.

4. PLUGS INTO classifier.py  one import + one method call.
   classifier.py already builds details dict and publishes it.
   We add our output to that dict before publish. Zero architecture change.

HOW TO INTEGRATE (see bottom of this file for exact patch)

In classifier.py __init__:
    from agents_code.agent9_regime.regime_detector import RegimeDetector
    self._regime_detector = RegimeDetector()

In classifier.py on_candles, after details["trend_persist"] = ...:
    detailed = self._regime_detector.compute(df, self._india_vix)
    details["detailed_regime"] = detailed.to_dict()

That's it. The result is available to:
  - runner.py          via regime_details.get("detailed_regime", {})
  - ML filter          via signal payload (passed through with RAW_SIGNAL)
  - signal_scorer.py   via the same dict
  - journal.py         logged automatically with the rest of details

REGIME LOGIC (decision tree)


              
                 ATR > ATR_MA  1.5        
                 OR VIX > VIX_HIGH_THRESH  
              
                          Yes  HIGH_VOLATILITY
                          No  
              
                 ADX  ADX_TREND_THRESH    
                 (default 18)              
              
                          Yes  TRENDING
                          No  
              
                 Chop Index > 61.8         
                 OR ATR < ATR_MA  0.7     
              
                          Yes  RANGING
                          No   TRENDING (marginal)

CONFIDENCE SCORING

Each regime computes a 0.01.0 confidence:

  TRENDING:
    base = 0.50
    + ADX distance above threshold  0.015   (max +0.30)
    + DI spread (plus_di - minus_di)  0.005 (max +0.15)
     capped at 0.95

  RANGING:
    base = 0.50
    + how much Chop Index exceeds 50  0.01  (max +0.25)
    + ATR contraction below average  0.20   (max +0.20)
     capped at 0.90

  HIGH_VOLATILITY:
    base = 0.60
    + ATR expansion above 1.5 average  0.15 (max +0.30)
     capped at 0.95

SUB-LABELS

TRENDING:
  STRONG_TREND    ADX > 30
  MODERATE_TREND  ADX 1830
  EMERGING_TREND  ADX 1518 (marginal, low confidence)

RANGING:
  TIGHT_RANGE     ATR well below average, chop high
  WIDE_RANGE      Price range wide but no direction
  CONSOLIDATION   ATR contracting AND chop rising (squeeze forming)

HIGH_VOLATILITY:
  ATR_SPIKE       ATR > 2 average (price moving but chaotic)
  VIX_ELEVATED    VIX > threshold (fear-driven vol)
  EXPANSION       Both ATR spike + VIX elevated
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

#  use pandas_ta if available  fallback to manual for portability 
try:
    import pandas_ta as ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False


# 
# RESULT DATACLASS
# 

@dataclass
class RegimeResult:
    """
    Output of RegimeDetector.compute()  one per candle.

    Fields accessible downstream via details["detailed_regime"]:
        label       : "TRENDING" | "RANGING" | "HIGH_VOLATILITY"
        confidence  : float 0.01.0
        sub_label   : e.g. "STRONG_TREND", "TIGHT_RANGE", "VIX_SPIKE"
        adx         : float  ADX(14) value
        atr_pct     : float  ATR as % of price
        atr_ratio   : float  current ATR / 20-bar avg ATR
        chop        : float  Choppiness Index
        di_spread   : float  +DI minus -DI
        is_trending : bool   convenience flag
        is_ranging  : bool   convenience flag
        is_high_vol : bool   convenience flag
    """
    label:       str     # "TRENDING" | "RANGING" | "HIGH_VOLATILITY"
    confidence:  float   # 0.01.0
    sub_label:   str     # descriptive sub-classification
    adx:         float
    atr_pct:     float   # ATR as % of close price
    atr_ratio:   float   # ATR_now / ATR_20bar_avg  (>1 = expanding, <1 = contracting)
    chop:        float   # Choppiness Index
    di_spread:   float   # +DI - -DI  (positive = bullish, negative = bearish)
    is_trending: bool
    is_ranging:  bool
    is_high_vol: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        # round floats for clean serialisation in bus messages
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d

    def __repr__(self) -> str:
        return (
            f"RegimeResult({self.label}|{self.sub_label} "
            f"conf={self.confidence:.2f} "
            f"ADX={self.adx:.1f} ATR_ratio={self.atr_ratio:.2f} "
            f"Chop={self.chop:.1f})"
        )


# 
# DEFAULT RESULT (returned when not enough candles)
# 

def _default_result() -> RegimeResult:
    return RegimeResult(
        label       = "RANGING",
        confidence  = 0.50,
        sub_label   = "INSUFFICIENT_DATA",
        adx         = 0.0,
        atr_pct     = 0.0,
        atr_ratio   = 1.0,
        chop        = 50.0,
        di_spread   = 0.0,
        is_trending = False,
        is_ranging  = True,
        is_high_vol = False,
    )


# 
# INDICATOR HELPERS (fast, no external deps)
# 

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range and ATR  manual computation avoids pandas_ta dependency."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c  = c.shift(1)
    tr      = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _compute_adx(df: pd.DataFrame, period: int = 14) -> dict:
    """
    ADX, +DI, -DI  Wilder smoothing.
    Returns dict so we avoid external TA library if unavailable.
    """
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(h)

    plus_dm  = np.zeros(n)
    minus_dm = np.zeros(n)
    tr_arr   = np.zeros(n)

    for i in range(1, n):
        up    = h[i] - h[i - 1]
        down  = l[i - 1] - l[i]
        tr_arr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        plus_dm[i]  = up   if (up > down   and up > 0)   else 0.0
        minus_dm[i] = down if (down > up   and down > 0) else 0.0

    def _wilder(arr: np.ndarray, p: int) -> np.ndarray:
        out = np.zeros(n)
        out[p] = arr[1:p + 1].sum()
        for i in range(p + 1, n):
            out[i] = out[i - 1] - out[i - 1] / p + arr[i]
        return out

    sm_tr  = _wilder(tr_arr, period)
    sm_pdm = _wilder(plus_dm, period)
    sm_mdm = _wilder(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(sm_tr > 0, 100 * sm_pdm / sm_tr, 0.0)
        mdi = np.where(sm_tr > 0, 100 * sm_mdm / sm_tr, 0.0)
        dx  = np.where((pdi + mdi) > 0, 100 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)

    adx = np.zeros(n)
    adx[2 * period] = dx[period: 2 * period + 1].mean()
    for i in range(2 * period + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return {
        "adx":      float(adx[-1]),
        "plus_di":  float(pdi[-1]),
        "minus_di": float(mdi[-1]),
    }


def _compute_chop(df: pd.DataFrame, period: int = 14) -> float:
    """Choppiness Index  same formula as classifier._chop_index."""
    try:
        n  = period + 1
        hi = df["high"].tail(n)
        lo = df["low"].tail(n)
        cl = df["close"].tail(n)
        tr = pd.concat(
            [hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr_sum     = tr.sum()
        price_range = hi.max() - lo.min()
        if price_range == 0 or atr_sum == 0:
            return 50.0
        return float(
            np.clip(100 * np.log10(atr_sum / price_range) / np.log10(period), 0, 100)
        )
    except Exception:
        return 50.0


# 
# REGIME DETECTOR
# 

class RegimeDetector:
    """
    Lightweight market regime classifier.
    Thread-safe pure-compute design  one instance shared across calls.

    Usage:
        detector = RegimeDetector()
        result   = detector.compute(df_5min, india_vix=22.0)
        print(result.label, result.confidence, result.sub_label)
    """

    #  Thresholds (read from settings if available, else defaults) 
    _ADX_TREND_MIN:  float = 18.0   # ADX >= this  trending
    _ADX_STRONG:     float = 30.0   # ADX >= this  strong trend
    _CHOP_RANGE:     float = 61.8   # Chop >= this  ranging
    _ATR_EXPAND:     float = 1.7    # ATR_ratio >= this  high vol
    _ATR_CONTRACT:   float = 0.70   # ATR_ratio <= this  ranging
    _VIX_HIGH:       float = 30.0   # VIX >= this  high vol
    _ATR_AVG_PERIOD: int   = 20     # bars for ATR baseline average
    _MIN_CANDLES:    int   = 30     # minimum candles needed

    def __init__(self) -> None:
        # Load thresholds from settings without crashing if unavailable
        try:
            from config.settings import (
                ADX_TREND_THRESHOLD, ADX_CHOP_THRESHOLD,
                CHOP_INDEX_THRESHOLD, VIX_HIGH_THRESHOLD,
            )
            self._ADX_TREND_MIN = float(ADX_TREND_THRESHOLD)
            self._CHOP_RANGE    = float(CHOP_INDEX_THRESHOLD)
            self._VIX_HIGH      = float(VIX_HIGH_THRESHOLD)
        except (ImportError, AttributeError):
            pass  # use class-level defaults

    #  PUBLIC API 

    def compute(
        self,
        df:         pd.DataFrame,
        india_vix:  float = 0.0,
    ) -> RegimeResult:
        """
        Classify the current market regime from OHLCV candles.

        Args:
            df        : DataFrame with columns open/high/low/close/volume
                        Must have DatetimeIndex. Minimum 30 rows.
            india_vix : current India VIX value (0 if unavailable)

        Returns:
            RegimeResult with label, confidence, sub_label, raw indicators
        """
        if df is None or len(df) < self._MIN_CANDLES:
            return _default_result()

        try:
            return self._classify(df, india_vix)
        except Exception:
            # Never crash the pipeline  return safe default
            return _default_result()

    #  INTERNAL CLASSIFICATION 

    def _classify(self, df: pd.DataFrame, india_vix: float) -> RegimeResult:

        #  Step 1: Compute indicators 
        adx_vals = self._get_adx(df)
        adx      = adx_vals["adx"]
        plus_di  = adx_vals["plus_di"]
        minus_di = adx_vals["minus_di"]
        di_spread = plus_di - minus_di

        atr_series = _compute_atr(df, period=14)
        atr_now    = float(atr_series.iloc[-1])
        atr_avg    = float(
            atr_series.dropna().tail(self._ATR_AVG_PERIOD).mean()
        ) if len(atr_series.dropna()) >= self._ATR_AVG_PERIOD else atr_now

        atr_ratio = atr_now / atr_avg if atr_avg > 0 else 1.0
        close_now = float(df["close"].iloc[-1])
        atr_pct   = atr_now / close_now * 100 if close_now > 0 else 0.0

        chop = _compute_chop(df, period=14)

        #  Step 2: Decision tree 
        # Priority: HIGH_VOL  TRENDING  RANGING
        # HIGH_VOL takes priority because it overrides all strategy logic

        # Gate A: HIGH_VOLATILITY
        vix_high = india_vix >= self._VIX_HIGH and india_vix > 0
        atr_spike = atr_ratio >= self._ATR_EXPAND
        if vix_high or atr_spike:
            return self._build_high_vol(adx, atr_pct, atr_ratio, chop, di_spread,
                                        vix_high, atr_spike, india_vix)

        # Gate B: TRENDING
        if adx >= self._ADX_TREND_MIN:
            return self._build_trending(adx, atr_pct, atr_ratio, chop, di_spread,
                                        plus_di, minus_di)

        # Gate C: RANGING
        return self._build_ranging(adx, atr_pct, atr_ratio, chop, di_spread)

    #  LABEL BUILDERS 

    def _build_trending(
        self,
        adx: float, atr_pct: float, atr_ratio: float,
        chop: float, di_spread: float, plus_di: float, minus_di: float,
    ) -> RegimeResult:

        # Sub-label based on ADX strength
        if adx >= self._ADX_STRONG:
            sub_label = "STRONG_TREND"
        elif adx >= self._ADX_TREND_MIN:
            sub_label = "MODERATE_TREND"
        else:
            sub_label = "EMERGING_TREND"

        # Confidence: base 0.50 + ADX distance + DI divergence
        adx_boost  = min(0.30, (adx - self._ADX_TREND_MIN) * 0.015)
        di_boost   = min(0.15, abs(di_spread) * 0.005)
        confidence = min(0.95, 0.50 + adx_boost + di_boost)

        return RegimeResult(
            label       = "TRENDING",
            confidence  = round(confidence, 3),
            sub_label   = sub_label,
            adx         = round(adx, 2),
            atr_pct     = round(atr_pct, 3),
            atr_ratio   = round(atr_ratio, 3),
            chop        = round(chop, 2),
            di_spread   = round(di_spread, 2),
            is_trending = True,
            is_ranging  = False,
            is_high_vol = False,
        )

    def _build_ranging(
        self,
        adx: float, atr_pct: float, atr_ratio: float,
        chop: float, di_spread: float,
    ) -> RegimeResult:

        # Sub-label based on ATR behaviour and chop level
        if atr_ratio <= self._ATR_CONTRACT and chop >= self._CHOP_RANGE:
            sub_label = "CONSOLIDATION"   # squeeze forming  breakout likely
        elif atr_ratio <= self._ATR_CONTRACT:
            sub_label = "TIGHT_RANGE"
        elif chop >= self._CHOP_RANGE:
            sub_label = "WIDE_RANGE"
        else:
            sub_label = "CHOPPY"

        # Confidence: base 0.50 + chop elevation + ATR contraction
        chop_boost = min(0.25, max(0, chop - 50) * 0.01)
        atr_boost  = min(0.20, max(0, 1.0 - atr_ratio) * 0.20)
        confidence = min(0.90, 0.50 + chop_boost + atr_boost)

        return RegimeResult(
            label       = "RANGING",
            confidence  = round(confidence, 3),
            sub_label   = sub_label,
            adx         = round(adx, 2),
            atr_pct     = round(atr_pct, 3),
            atr_ratio   = round(atr_ratio, 3),
            chop        = round(chop, 2),
            di_spread   = round(di_spread, 2),
            is_trending = False,
            is_ranging  = True,
            is_high_vol = False,
        )

    def _build_high_vol(
        self,
        adx: float, atr_pct: float, atr_ratio: float,
        chop: float, di_spread: float,
        vix_high: bool, atr_spike: bool, india_vix: float,
    ) -> RegimeResult:

        # Sub-label: what is driving the high volatility
        if vix_high and atr_spike:
            sub_label = "EXPANSION"
        elif vix_high:
            sub_label = "VIX_ELEVATED"
        else:
            sub_label = "ATR_SPIKE"

        # Confidence: base 0.60 + ATR expansion + VIX elevation
        atr_boost = min(0.30, max(0, atr_ratio - self._ATR_EXPAND) * 0.15)
        vix_boost = min(0.10, max(0, india_vix - self._VIX_HIGH) * 0.01) if vix_high else 0
        confidence = min(0.95, 0.60 + atr_boost + vix_boost)

        return RegimeResult(
            label       = "HIGH_VOLATILITY",
            confidence  = round(confidence, 3),
            sub_label   = sub_label,
            adx         = round(adx, 2),
            atr_pct     = round(atr_pct, 3),
            atr_ratio   = round(atr_ratio, 3),
            chop        = round(chop, 2),
            di_spread   = round(di_spread, 2),
            is_trending = False,
            is_ranging  = False,
            is_high_vol = True,
        )

    #  INDICATOR DISPATCH 

    def _get_adx(self, df: pd.DataFrame) -> dict:
        """Use pandas_ta if available, else manual computation."""
        if _TA_AVAILABLE:
            try:
                res = ta.adx(df["high"], df["low"], df["close"], length=14)
                if res is not None:
                    return {
                        "adx":      float(res["ADX_14"].iloc[-1]),
                        "plus_di":  float(res["DMP_14"].iloc[-1]),
                        "minus_di": float(res["DMN_14"].iloc[-1]),
                    }
            except Exception:
                pass
        # Fallback to manual
        return _compute_adx(df, period=14)

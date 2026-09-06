"""
agents_code/agent9_regime/vix_slope.py  VIX Trend Tracker
============================================================
NEW FILE. Tracks VIX 5-day slope and publishes it as regime context.

WHY: VIX level (e.g. VIX=22) tells you current fear.
     VIX SLOPE tells you if fear is INCREASING or DECREASING.
     - Rising VIX = institutional hedging, momentum trades work (buy puts)
     - Falling VIX = complacency, mean-reversion trades work (buy calls)
     - Flat VIX    = no edge from VIX alone

USAGE: Add to PREMARKET_BIAS payload and read in agent9 classifier.

HOW TO INTEGRATE in classifier.py:
    # In __init__:
    from agents_code.agent9_regime.vix_slope import VIXSlope
    self._vix_slope = VIXSlope()

    # In on_premarket:
    self._vix_slope.update(self._india_vix)

    # In on_candles:
    details["vix_slope"] = self._vix_slope.slope
    details["vix_regime"] = self._vix_slope.regime

    # In runner.py on_candles:
    vix_regime = regime_details.get("vix_regime", "FLAT")
    # Use for position sizing: rising VIX  smaller size
"""

from collections import deque
import statistics


class VIXSlope:
    """Tracks VIX over N days and computes slope."""

    WINDOW = 5   # 5-day slope

    def __init__(self) -> None:
        self._history: deque = deque(maxlen=self.WINDOW)
        self.slope    = 0.0
        self.regime   = "FLAT"   # "RISING" | "FALLING" | "FLAT"

    def update(self, vix_value: float) -> None:
        if vix_value <= 0:
            return
        self._history.append(vix_value)
        self._compute()

    def _compute(self) -> None:
        if len(self._history) < 3:
            self.slope  = 0.0
            self.regime = "FLAT"
            return
        vals    = list(self._history)
        n       = len(vals)
        x_mean  = (n - 1) / 2
        numer   = sum((i - x_mean) * (v - statistics.mean(vals)) for i, v in enumerate(vals))
        denom   = sum((i - x_mean) ** 2 for i in range(n))
        self.slope = round(numer / denom if denom > 0 else 0.0, 4)

        if self.slope > 0.5:
            self.regime = "RISING"
        elif self.slope < -0.5:
            self.regime = "FALLING"
        else:
            self.regime = "FLAT"

    def to_dict(self) -> dict:
        return {"slope": self.slope, "regime": self.regime, "history": list(self._history)}
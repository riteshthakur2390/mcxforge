"""
utils/greeks_filter.py — Options Greeks Pre-Trade Filter
=========================================================
NEW FILE. Checks whether a selected option is worth trading
based on estimated Greeks.

WHY GREEKS MATTER:
  Current system picks strikes by confidence level (ATM vs OTM).
  It does NOT check:
  - Delta: how much the option moves per ₹1 NIFTY move
    → Deep OTM with delta=0.15 barely moves even if direction is right
    → ATM has delta≈0.5, 100-point NIFTY move = ₹50 option move
  - Theta: daily time decay
    → DTE=1 (Thursday expiry): theta is ₹5-15/candle → premium bleeds fast
    → DTE=5: theta ≈ ₹2-3/day → manageable
  - IV vs HV: if IV is much higher than HV, option is expensive
    → Buying expensive options = lower expected value

FILTERING RULES:
  Minimum delta:      0.30  (option must move at least ₹30 per ₹100 NIFTY move)
  Maximum theta/day:  ₹8    (premium can't decay more than ₹8/day from theta alone)
  DTE minimum:        2     (don't buy on expiry day — theta kills you)

USAGE in planner.py:
  from utils.greeks_filter import GreeksFilter
  gf     = GreeksFilter()
  result = gf.check(nifty_ltp=22000, strike=22100, opt_type="CE",
                    est_premium=85, dte=3)
  if not result.tradeable:
      logger.info(f"Greeks filter: {result.reason}")
      return
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class GreeksResult:
    tradeable:     bool
    est_delta:     float
    est_theta_day: float
    iv_estimate:   float
    reason:        str

    @property
    def delta(self) -> float:
        return self.est_delta

    @property
    def theta(self) -> float:
        return self.est_theta_day

    @property
    def greeks_ok(self) -> bool:
        return self.tradeable

    def __getitem__(self, item: str):
        if item in ("delta", "est_delta"):
            return self.est_delta
        if item in ("theta", "est_theta_day"):
            return self.est_theta_day
        if item == "greeks_ok":
            return self.greeks_ok
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


class GreeksFilter:
    """
    Estimates Black-Scholes Greeks from market observables.
    Uses simplified BSM approximations — accurate enough for filtering.
    No broker API needed.
    """

    MIN_DELTA      = 0.30   # option must have meaningful directional sensitivity
    MAX_THETA_DAY  = 8.0    # max acceptable daily theta decay in ₹
    MIN_DTE        = 0      # allow intraday trading on expiry and near-expiry days

    def check(
        self,
        nifty_ltp:   float = 0.0,
        strike:      int = 0,
        opt_type:    str = "FUT",    # "CE", "PE", or "FUT"
        est_premium: float = 0.0,
        dte:         int = 0,
        allow_expiry_day: bool = False,
        is_hero_zero: bool = False,
        min_delta:   float | None = None,
        **kwargs,
    ) -> GreeksResult:
        """
        Check if this instrument meets Greeks requirements.
        Bypasses options Greeks for commodity futures.
        """
        premium_src = kwargs.get("premium_source", "")
        if opt_type in ("FUT", "FUTURE", "") or premium_src == "COMMODITY_FUTURES":
            return GreeksResult(
                tradeable=True,
                est_delta=1.0,
                est_theta_day=0.0,
                iv_estimate=0.0,
                reason="Commodity futures contract: delta=1.0, zero theta decay",
            )

        if dte < self.MIN_DTE and not allow_expiry_day:
            return GreeksResult(
                tradeable     = False,
                est_delta     = 0.0,
                est_theta_day = 0.0,
                iv_estimate   = 0.0,
                reason        = f"DTE={dte} < MIN_DTE={self.MIN_DTE} (too close to expiry)",
            )

        if est_premium <= 0 or nifty_ltp <= 0:
            return GreeksResult(
                tradeable=False, est_delta=0, est_theta_day=0,
                iv_estimate=0, reason="Invalid premium or LTP"
            )

        # ── Estimate IV from premium (simplified BSM inversion) ──────────
        iv = self._estimate_iv(nifty_ltp, strike, est_premium, dte)

        # ── Estimate Delta ────────────────────────────────────────────────
        delta = self._estimate_delta(nifty_ltp, strike, iv, dte, opt_type)

        # ── Estimate Theta (daily) ────────────────────────────────────────
        theta = self._estimate_theta(nifty_ltp, strike, iv, dte, est_premium, opt_type)

        # ── Determine effective Delta threshold ───────────────────────────
        if min_delta is not None:
            effective_min_delta = float(min_delta)
        elif is_hero_zero:
            effective_min_delta = 0.04  # allow cheap OTM expiry lottery contracts
        elif allow_expiry_day or dte <= 1:
            effective_min_delta = 0.12  # allow ATM/near-OTM strikes on expiry day
        else:
            effective_min_delta = self.MIN_DELTA  # standard 0.30

        # ── Apply filters ─────────────────────────────────────────────────
        if abs(delta) < effective_min_delta:
            return GreeksResult(
                tradeable     = False,
                est_delta     = round(delta, 3),
                est_theta_day = round(theta, 2),
                iv_estimate   = round(iv, 3),
                reason        = (f"Delta={delta:.2f} < {effective_min_delta:.2f} — "
                                 f"OTM option won't move enough"),
            )

        # Theta decay limit: ₹8/day for standard Nifty options (premium ~₹100),
        # or up to 6% of premium/day for commodities and high-premium options (Silver, Gold where premium is ₹4000-8000)
        theta_base_limit = max(self.MAX_THETA_DAY, est_premium * 0.06)
        max_theta = theta_base_limit * (6.0 if (allow_expiry_day or dte <= 1) else 1.0)
        if theta > max_theta:
            return GreeksResult(
                tradeable     = False,
                est_delta     = round(delta, 3),
                est_theta_day = round(theta, 2),
                iv_estimate   = round(iv, 3),
                reason        = (f"Theta=₹{theta:.1f}/day > MAX ₹{max_theta:.1f} — "
                                 f"too much time decay for DTE={dte}"),
            )

        return GreeksResult(
            tradeable     = True,
            est_delta     = round(delta, 3),
            est_theta_day = round(theta, 2),
            iv_estimate   = round(iv, 3),
            reason        = f"OK | delta={delta:.2f} | theta=₹{theta:.1f}/day",
        )

    # ── APPROXIMATIONS ────────────────────────────────────────────────────

    @staticmethod
    def _estimate_iv(S: float, K: int, premium: float, dte: int) -> float:
        """
        Simplified IV estimate using Brenner-Subrahmanyam approximation.
        IV ≈ premium × sqrt(2π / T) / S
        where T = dte / 252 (annual fraction)
        """
        T = max(dte / 252, 0.001)
        iv = (premium / S) * math.sqrt(2 * math.pi / T)
        return max(0.05, min(iv, 2.0))   # clamp to [5%, 200%]

    @staticmethod
    def _estimate_delta(S: float, K: int, iv: float, dte: int, opt_type: str) -> float:
        """
        Simplified delta using BSM d1.
        N(d1) approximated using logistic function.
        """
        T = max(dte / 252, 0.001)
        try:
            d1 = (math.log(S / K) + 0.5 * iv**2 * T) / (iv * math.sqrt(T))
            # Logistic approximation of N(d1)
            nd1 = 1 / (1 + math.exp(-1.7 * d1))
            if opt_type == "CE":
                return nd1
            else:
                return nd1 - 1.0   # Put delta = N(d1) - 1
        except Exception:
            return 0.5 if opt_type == "CE" else -0.5

    @staticmethod
    def _estimate_theta(S: float, K: int, iv: float, dte: int, premium: float, opt_type: str = "CE") -> float:
        """
        Daily theta in ₹ terms.
        Theta only decays the EXTRINSIC (time) value of an option, never intrinsic value.
        """
        intrinsic = max(0.0, (S - K) if opt_type == "CE" else (K - S))
        extrinsic = max(0.0, premium - intrinsic)
        
        safe_dte = max(dte, 0.5)
        decay_rate = 1.0 / (2.0 * safe_dte)
        theta_pct  = iv * decay_rate * 0.5
        return min(extrinsic, extrinsic * theta_pct)

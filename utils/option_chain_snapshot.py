"""
utils/option_chain_snapshot.py — Option Chain Snapshot Writer
==============================================================
Captures a full option chain snapshot at the moment a signal fires.
This is the most valuable data asset for improving ML models over time.

WHY THIS MATTERS:
  Without snapshots, you can only answer "what was the price?"
  With snapshots, you can answer "what did the entire options market
  say about this moment?" — IV skew, OI distribution, PCR, max pain.

  Sensibull, Opstra, NSE Option Chain — all build their analysis on
  exactly this data. We capture it automatically at signal time.

WHAT IT CAPTURES (per signal):
  For each strike from ATM-500 to ATM+500 (in 50-pt steps = 21 strikes):
    - CE: IV, LTP, volume, OI, OI change, delta proxy
    - PE: IV, LTP, volume, OI, OI change, delta proxy
  Plus derived metrics:
    - PCR (Put-Call Ratio by OI): market fear gauge
    - Max Pain: the strike where option writers lose least
    - IV skew: OTM PE IV vs OTM CE IV
    - Total OI at each strike

STORAGE:
  journal/option_chain/YYYY-MM-DD/HH-MM_NIFTY_<strike>.json
  One file per signal. Small: ~8KB per snapshot.
  Queryable by date/time/outcome for ML training.

USAGE:
  snapshot = OptionChainSnapshot(broker)
  data = await snapshot.capture(spot=22150, signal_direction="BUY_CALL")
  snapshot.save(data, trade_id="SF-20260514-001")

  # For backtest (no broker):
  data = snapshot.build_proxy(spot=22150, hv=0.15, signal_direction="BUY_CALL")
"""

import json
import math
import os
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from config.settings import JOURNAL_DIR
except ImportError:
    JOURNAL_DIR = "journal"

def _get_default_step(symbol: str = "") -> int:
    sym = symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))
    try:
        from utils.instrument_selector import get_instrument
        return int(get_instrument(sym).strike_step)
    except Exception:
        return 500

SNAPSHOT_DIR = Path(JOURNAL_DIR) / "option_chain"
STRIKES_EACH_SIDE = 10  # ATM ± 10 strikes = 21 strikes total


def _atm(spot: float, step: int = 500) -> int:
    s = max(int(step or 500), 1)
    return int(round(spot / s) * s)


def _proxy_iv(hv: float, moneyness_steps: int) -> float:
    """Volatility smile approximation."""
    smile = abs(moneyness_steps) * 0.08
    return round(hv * (1.0 + smile), 4)


def _bs_delta(spot: float, strike: float, T: float,
              iv: float, opt_type: str = "CE") -> float:
    """Simplified BS delta."""
    try:
        d1 = (math.log(spot / strike) + (0.065 + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        return round(nd1 if opt_type == "CE" else nd1 - 1.0, 4)
    except Exception:
        return 0.5 if opt_type == "CE" else -0.5


class OptionChainSnapshot:
    """
    Captures and persists option chain data at signal time.
    Works with or without live broker (proxy mode for backtest).
    """

    def __init__(self, broker=None) -> None:
        self._broker = broker
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    async def capture(
        self,
        spot:             float,
        signal_direction: str,
        dte:              int   = 5,
        hv:               float = 0.15,
    ) -> dict:
        """
        Capture live option chain (if broker available) or proxy chain.

        Args:
            spot:             current NIFTY spot price
            signal_direction: "BUY_CALL" or "BUY_PUT"
            dte:              days to expiry
            hv:               historical volatility (for proxy mode)

        Returns:
            dict: full chain snapshot with all strikes + derived metrics
        """
        if self._broker is not None:
            try:
                return await self._live_capture(spot, signal_direction, dte)
            except Exception:
                pass
        return self.build_proxy(spot, signal_direction, dte, hv)

    def build_proxy(
        self,
        spot:             float,
        signal_direction: str,
        dte:              int   = 5,
        hv:               float = 0.15,
    ) -> dict:
        """
        Build a synthetic option chain using Black-Scholes proxy.
        Used in backtest and when broker is unavailable.
        """
        sym    = str(os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
        step   = _get_default_step(sym)
        atm    = _atm(spot, step=step)
        T      = max(dte, 0.5) / 365.0
        r      = 0.065
        n      = STRIKES_EACH_SIDE

        strikes_range = range(atm - n * step, atm + n * step + 1, step)
        chain = {}

        for k in strikes_range:
            steps     = (k - atm) // step
            iv_ce     = _proxy_iv(hv, steps)
            iv_pe     = _proxy_iv(hv, -steps)
            delta_ce  = _bs_delta(spot, k, T, iv_ce, "CE")
            delta_pe  = _bs_delta(spot, k, T, iv_pe, "PE")

            # Synthetic LTP from BS
            d1_ce = (math.log(spot / k) + (r + 0.5 * iv_ce**2) * T) / (iv_ce * math.sqrt(T))
            d2_ce = d1_ce - iv_ce * math.sqrt(T)
            nd1   = 0.5 * (1 + math.erf(d1_ce / math.sqrt(2)))
            nd2   = 0.5 * (1 + math.erf(d2_ce / math.sqrt(2)))
            ltp_ce = max(spot * nd1 - k * math.exp(-r * T) * nd2, 0.05)
            ltp_pe = max(ltp_ce + k * math.exp(-r * T) - spot, 0.05)

            # Synthetic volume and OI (higher at ATM, falls off OTM)
            atm_factor  = max(0.1, 1.0 - abs(steps) * 0.15)
            base_oi     = int(50000 * atm_factor)
            base_vol    = int(8000 * atm_factor)

            chain[str(k)] = {
                "strike": k,
                "moneyness_steps": steps,
                "CE": {
                    "iv":        round(iv_ce, 4),
                    "ltp":       round(ltp_ce, 2),
                    "delta":     delta_ce,
                    "volume":    base_vol,
                    "oi":        base_oi,
                    "oi_change": 0,
                },
                "PE": {
                    "iv":        round(iv_pe, 4),
                    "ltp":       round(ltp_pe, 2),
                    "delta":     delta_pe,
                    "volume":    int(base_vol * 0.9),
                    "oi":        int(base_oi * 1.1),
                    "oi_change": 0,
                },
            }

        return self._derive_metrics(chain, spot, atm, signal_direction, dte, "proxy")

    def save(self, snapshot: dict, trade_id: str = "") -> str:
        """Save snapshot to JSON file. Returns file path."""
        today    = date.today().isoformat()
        now_str  = datetime.now(IST).strftime("%H-%M")
        atm      = snapshot.get("atm_strike", 0)
        day_dir  = SNAPSHOT_DIR / today
        day_dir.mkdir(exist_ok=True)

        sym      = str(os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
        suffix   = f"_{trade_id}" if trade_id else ""
        fname    = f"{now_str}_{sym}_{atm}{suffix}.json"
        fpath    = day_dir / fname

        with open(fpath, "w") as f:
            json.dump(snapshot, f, indent=2)

        return str(fpath)

    def load(self, date_str: str, time_str: str) -> Optional[dict]:
        """Load a saved snapshot. date_str='2026-05-14', time_str='10-30'"""
        day_dir = SNAPSHOT_DIR / date_str
        for f in day_dir.glob(f"{time_str}*.json"):
            with open(f) as fp:
                return json.load(fp)
        return None

    # ── LIVE CAPTURE ─────────────────────────────────────────────────────────

    async def _live_capture(
        self, spot: float, direction: str, dte: int
    ) -> dict:
        """
        Fetch real option chain from broker.
        Currently implemented for Upstox / Dhan via option_ltp calls.
        """
        from utils.option_utils import build_option_symbol
        from datetime import date as ddate, timedelta

        # Find next Thursday
        d = ddate.today()
        days = (3 - d.weekday()) % 7 or 7
        expiry = d + timedelta(days=days)

        sym   = str(os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))).upper()
        step  = _get_default_step(sym)
        atm   = _atm(spot, step=step)
        chain = {}

        for steps in range(-STRIKES_EACH_SIDE, STRIKES_EACH_SIDE + 1):
            k = atm + steps * step
            ce_sym = build_option_symbol(sym, expiry, k, "CE")
            pe_sym = build_option_symbol(sym, expiry, k, "PE")

            ltp_ce = self._broker.get_option_ltp(ce_sym) or 0.1
            ltp_pe = self._broker.get_option_ltp(pe_sym) or 0.1

            T = max(dte, 0.5) / 365.0

            def get_iv(ltp, strike, opt_type):
                from agents_code.agent2_strategy.s18_skew_hunter import _implied_vol
                return _implied_vol(spot, float(strike), T, 0.065, ltp, opt_type)

            chain[str(k)] = {
                "strike": k,
                "moneyness_steps": steps,
                "CE": {"iv": get_iv(ltp_ce, k, "CE"), "ltp": ltp_ce,
                       "delta": _bs_delta(spot, k, T, 0.15, "CE"),
                       "volume": 0, "oi": 0, "oi_change": 0},
                "PE": {"iv": get_iv(ltp_pe, k, "PE"), "ltp": ltp_pe,
                       "delta": _bs_delta(spot, k, T, 0.15, "PE"),
                       "volume": 0, "oi": 0, "oi_change": 0},
            }

        return self._derive_metrics(chain, spot, atm, direction, dte, "live")

    # ── DERIVED METRICS ───────────────────────────────────────────────────────

    @staticmethod
    def _derive_metrics(
        chain: dict, spot: float, atm: int,
        direction: str, dte: int, mode: str
    ) -> dict:
        """
        Compute derived option chain metrics:
        PCR, Max Pain, IV Skew, OI distribution.
        """
        strikes = sorted(int(k) for k in chain)

        # PCR (Put-Call Ratio by OI)
        total_ce_oi = sum(chain[str(k)]["CE"]["oi"] for k in strikes)
        total_pe_oi = sum(chain[str(k)]["PE"]["oi"] for k in strikes)
        pcr = round(total_pe_oi / max(total_ce_oi, 1), 3)

        # Max Pain (strike where option writers lose least)
        max_pain_losses = {}
        for expiry_strike in strikes:
            total_loss = 0
            for k in strikes:
                # CE writers lose when spot > strike
                ce_oi = chain[str(k)]["CE"]["oi"]
                if expiry_strike > k:
                    total_loss += ce_oi * (expiry_strike - k)
                # PE writers lose when spot < strike
                pe_oi = chain[str(k)]["PE"]["oi"]
                if expiry_strike < k:
                    total_loss += pe_oi * (k - expiry_strike)
            max_pain_losses[expiry_strike] = total_loss

        max_pain = min(max_pain_losses, key=max_pain_losses.get)

        # IV Skew: OTM PE IV vs OTM CE IV (1 strike OTM each)
        otm_ce_k = str(atm + 100)
        otm_pe_k = str(atm - 100)
        iv_otm_ce = chain.get(otm_ce_k, {}).get("CE", {}).get("iv", 0)
        iv_otm_pe = chain.get(otm_pe_k, {}).get("PE", {}).get("iv", 0)
        iv_atm_ce = chain.get(str(atm), {}).get("CE", {}).get("iv", 0.15)
        iv_skew   = round(iv_otm_pe - iv_otm_ce, 4)   # > 0 = put skew = bearish

        return {
            "timestamp":       datetime.now(IST).isoformat(),
            "date":            date.today().isoformat(),
            "spot":            round(spot, 2),
            "atm_strike":      atm,
            "dte":             dte,
            "direction":       direction,
            "mode":            mode,
            # Derived
            "pcr":             pcr,
            "max_pain":        max_pain,
            "iv_skew":         iv_skew,
            "iv_atm_ce":       round(iv_atm_ce, 4),
            "iv_otm_ce":       round(iv_otm_ce, 4),
            "iv_otm_pe":       round(iv_otm_pe, 4),
            "total_ce_oi":     total_ce_oi,
            "total_pe_oi":     total_pe_oi,
            # Full chain
            "chain":           chain,
        }


async def write_option_chain_snapshot(
    *,
    spot: float,
    signal_direction: str,
    broker=None,
    trade_id: str = "",
    dte: int = 5,
    hv: float = 0.15,
) -> tuple[dict, str]:
    """
    Capture and persist an option-chain snapshot through one stable utility API.

    Returns:
        (snapshot, saved_path)
    """
    writer = OptionChainSnapshot(broker)
    snapshot = await writer.capture(
        spot=spot,
        signal_direction=signal_direction,
        dte=dte,
        hv=hv,
    )
    path = writer.save(snapshot, trade_id=trade_id)
    return snapshot, path

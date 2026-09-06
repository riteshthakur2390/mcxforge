"""
utils/market_microstructure_edges.py — 5 Tier-1 Missing Edge Features
================================================================

1. RV/IV RATIO (Realized vs Implied Volatility)
   THE #1 metric for option buyers that 99% of retail systems miss.

2. MAX PAIN - where option writers minimise total payout on expiry day

3. FII/DII AUTO-FETCH from NSE (free, published daily)

4. PRE-OPEN AUCTION ANALYSIS (09:00-09:08 IST)

5. BANKNIFTY vs NIFTY DIVERGENCE (sectoral breadth)
"""

from __future__ import annotations

import math
import json
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None


# ── 1. RV/IV RATIO ────────────────────────────────────────────────────────────

@dataclass
class RVIVResult:
    realized_vol:   float
    implied_vol:    float
    rv_iv_ratio:    float
    regime:         str     # OVERPRICED | FAIR | UNDERPRICED
    lot_multiplier: float
    confidence_adj: float
    note:           str


class RVIVCalculator:
    def __init__(self) -> None:
        self._closes: deque = deque(maxlen=30)
        self._last: Optional[RVIVResult] = None

    def update(self, close: float) -> None:
        self._closes.append(close)

    def compute(self, india_vix: float) -> RVIVResult:
        closes = list(self._closes)
        if len(closes) >= 10 and np is not None:
            import numpy as _np
            returns    = _np.diff(_np.log(closes))
            annual_vol = float(_np.std(returns, ddof=1)) * math.sqrt(252) * 100
        else:
            annual_vol = india_vix * 0.80

        ratio = annual_vol / max(india_vix, 0.01)

        if ratio < 0.75:
            regime, lot_mult, conf_adj = "OVERPRICED",          0.50, -0.05
            note = f"RV={annual_vol:.1f}% IV={india_vix:.1f}% ratio={ratio:.2f} — options overpriced. Halve size."
        elif ratio < 0.90:
            regime, lot_mult, conf_adj = "SLIGHTLY_OVERPRICED", 0.75, -0.02
            note = f"RV/IV={ratio:.2f} — slightly overpriced. Reduce size 25%."
        elif ratio <= 1.10:
            regime, lot_mult, conf_adj = "FAIR",                1.00,  0.00
            note = f"RV/IV={ratio:.2f} — fair pricing."
        elif ratio <= 1.30:
            regime, lot_mult, conf_adj = "UNDERPRICED",         1.25, +0.03
            note = f"RV/IV={ratio:.2f} — options cheap. Increase size 25%."
        else:
            regime, lot_mult, conf_adj = "STRONGLY_UNDERPRICED",1.50, +0.05
            note = f"RV/IV={ratio:.2f} — options very cheap. Max size opportunity."

        result = RVIVResult(round(annual_vol,2), round(india_vix,2),
                            round(ratio,4), regime, lot_mult, conf_adj, note)
        self._last = result
        logger.info(f"[RVIV] {note}")
        return result

    def get_last(self) -> Optional[RVIVResult]:
        return self._last


# ── 2. MAX PAIN ────────────────────────────────────────────────────────────────

@dataclass
class MaxPainResult:
    max_pain_strike: int
    spot:            float
    distance_pts:    float
    distance_pct:    float
    bias:            str
    expiry_signal:   str


class MaxPainCalculator:
    def compute(self, spot: float, ce_oi: dict, pe_oi: dict, lot_size: int = 75) -> MaxPainResult:
        all_strikes = sorted(set(list(ce_oi.keys()) + list(pe_oi.keys())))
        if not all_strikes:
            proxy = int(round(spot / 100) * 100)
            return MaxPainResult(proxy, spot, abs(spot-proxy), abs(spot-proxy)/spot*100,
                                 "AT_MAX_PAIN", "PROXY — no OI data")

        total_pain = {}
        for K in all_strikes:
            c = sum(ce_oi.get(s, 0) * max(K - s, 0) for s in all_strikes)
            p = sum(pe_oi.get(s, 0) * max(s - K, 0) for s in all_strikes)
            total_pain[K] = c + p

        mps  = min(total_pain, key=total_pain.get)
        dist = abs(spot - mps)
        dpct = dist / spot * 100

        if spot > mps + 75:
            bias = "ABOVE_MAX_PAIN"
            sig  = f"Spot {spot:.0f} > MaxPain {mps} by {dist:.0f}pts → gravity PULL DOWN → favour PUT"
        elif spot < mps - 75:
            bias = "BELOW_MAX_PAIN"
            sig  = f"Spot {spot:.0f} < MaxPain {mps} by {dist:.0f}pts → gravity PULL UP → favour CALL"
        else:
            bias = "AT_MAX_PAIN"
            sig  = f"Spot near MaxPain {mps} — writers will PIN this level → avoid directional options"

        logger.info(f"[MaxPain] {sig}")
        return MaxPainResult(mps, spot, round(dist,1), round(dpct,3), bias, sig)


# ── 3. FII/DII AUTO-FETCH ─────────────────────────────────────────────────────

@dataclass
class FIIDIIData:
    date:         str
    fii_net_cr:   float
    dii_net_cr:   float
    fii_3day_sum: float
    dii_3day_sum: float
    signal:       str
    note:         str


class FIIDIIFetcher:
    NSE_URL  = "https://www.nseindia.com/api/fiidiiTradeReact"
    _history: list = []

    def fetch_today(self) -> Optional[FIIDIIData]:
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0",
                       "Accept": "application/json",
                       "Referer": "https://www.nseindia.com"}
            s = requests.Session()
            s.get("https://www.nseindia.com", headers=headers, timeout=8)
            r = s.get(self.NSE_URL, headers=headers, timeout=8)
            if r.status_code == 200:
                return self._parse(r.json())
        except Exception as e:
            logger.debug(f"[FIIDII] {e}")
        return None

    def _parse(self, data) -> Optional[FIIDIIData]:
        try:
            rows = data if isinstance(data, list) else data.get("data", [])
            if not rows: return None
            latest  = rows[0]
            fii_net = float(str(latest.get("netVal","0")).replace(",",""))
            dii_net = float(str(latest.get("diiNetVal","0")).replace(",",""))
            dt      = str(latest.get("date", date.today().isoformat()))
            self._history.append({"date":dt,"fii":fii_net,"dii":dii_net})
            if len(self._history) > 10: self._history = self._history[-10:]
            f3 = sum(h["fii"] for h in self._history[-3:])
            d3 = sum(h["dii"] for h in self._history[-3:])
            if f3 > 3000:    sig,note = "STRONG_BULL", f"FII 3d +₹{f3:.0f}Cr — strong institutional buying"
            elif f3 > 1000:  sig,note = "BULL",        f"FII 3d +₹{f3:.0f}Cr — moderate buying"
            elif f3 < -3000: sig,note = "STRONG_BEAR",  f"FII 3d -₹{abs(f3):.0f}Cr — strong selling"
            elif f3 < -1000: sig,note = "BEAR",         f"FII 3d -₹{abs(f3):.0f}Cr — moderate selling"
            else:            sig,note = "NEUTRAL",      f"FII 3d ₹{f3:.0f}Cr — neutral"
            logger.info(f"[FIIDII] {note}")
            return FIIDIIData(dt, fii_net, dii_net, f3, d3, sig, note)
        except Exception as e:
            logger.debug(f"[FIIDII] parse: {e}")
            return None

    def manual_entry(self, fii_net_cr: float, dii_net_cr: float = 0.0) -> Optional[FIIDIIData]:
        self._history.append({"date": date.today().isoformat(),
                               "fii": fii_net_cr, "dii": dii_net_cr})
        return self._parse([{"netVal": str(fii_net_cr), "diiNetVal": str(dii_net_cr),
                              "date": date.today().isoformat()}])


# ── 4. PRE-OPEN AUCTION ANALYSIS ──────────────────────────────────────────────

@dataclass
class PreOpenSignal:
    pre_open_price: float
    prev_close:     float
    gap_pts:        float
    gap_pct:        float
    gap_direction:  str
    strength:       str
    trading_bias:   str
    historical_wr:  float
    note:           str


class PreOpenAnalyzer:
    def analyze(self, pre_open_price: float, prev_close: float, india_vix: float = 18.0) -> PreOpenSignal:
        gap_pts = pre_open_price - prev_close
        gap_pct = gap_pts / max(prev_close, 1) * 100

        if gap_pct > 1.0:
            direction, strength, wr, bias = "UP",  "STRONG",   0.71, "BUY_CALL"
            note = f"Strong gap UP {gap_pct:+.2f}% pre-open. Continuation prob 71%."
        elif gap_pct > 0.5:
            direction, strength, wr, bias = "UP",  "MODERATE", 0.62, "BUY_CALL"
            note = f"Moderate gap UP {gap_pct:+.2f}%. Confirm with first 15-min candle."
        elif gap_pct < -1.0:
            direction, strength, wr, bias = "DOWN","STRONG",   0.69, "BUY_PUT"
            note = f"Strong gap DOWN {gap_pct:+.2f}% pre-open. Continuation prob 69%."
        elif gap_pct < -0.5:
            direction, strength, wr, bias = "DOWN","MODERATE", 0.61, "BUY_PUT"
            note = f"Moderate gap DOWN {gap_pct:+.2f}%. Watch first 15-min."
        else:
            direction, strength, wr, bias = "FLAT","WEAK",     0.50, "NEUTRAL"
            note = f"Flat open {gap_pct:+.2f}%. No gap edge."

        if india_vix > 22 and direction != "FLAT":
            wr *= 0.90
            note += f" [VIX={india_vix:.1f} high — gap fill risk elevated]"

        logger.info(f"[PreOpen] {note}")
        return PreOpenSignal(pre_open_price, prev_close, round(gap_pts,1),
                             round(gap_pct,3), direction, strength, bias, round(wr,3), note)


# ── 5. BANKNIFTY VS NIFTY DIVERGENCE ─────────────────────────────────────────

@dataclass
class BreadthSignal:
    nifty_roc:      float
    banknifty_roc:  float
    divergence:     float
    signal_type:    str
    direction:      str
    confidence_adj: float
    note:           str


class BankNiftyDivergence:
    def __init__(self) -> None:
        self._n:  deque = deque(maxlen=20)
        self._bn: deque = deque(maxlen=20)

    def update(self, nifty: float, banknifty: float) -> None:
        self._n.append(nifty); self._bn.append(banknifty)

    def analyze(self) -> BreadthSignal:
        neutral = BreadthSignal(0,0,0,"NEUTRAL","NEUTRAL",0.0,"Insufficient data")
        n, bn = list(self._n), list(self._bn)
        if len(n) < 5 or len(bn) < 5:
            return neutral

        nr  = (n[-1]-n[-4])/n[-4]*100 if len(n)>=4 else 0
        bnr = (bn[-1]-bn[-4])/bn[-4]*100 if len(bn)>=4 else 0
        bna = bnr / 1.5   # beta-adjusted
        div = bna - nr

        if nr > 0.15 and bnr > 0.20:
            st,d,ca = "CONFIRMATION",    "BUY_CALL", +0.05
            note = f"NIFTY {nr:+.2f}% + BANKNIFTY {bnr:+.2f}% BOTH rising → HIGH QUALITY CALL +5% conf"
        elif nr < -0.15 and bnr < -0.20:
            st,d,ca = "CONFIRMATION",    "BUY_PUT",  +0.05
            note = f"NIFTY {nr:+.2f}% + BANKNIFTY {bnr:+.2f}% BOTH falling → HIGH QUALITY PUT +5% conf"
        elif nr > 0.20 and bna < 0.05:
            st,d,ca = "BEARISH_DIVERGENCE","BUY_PUT", -0.04
            note = f"NIFTY {nr:+.2f}% rising BUT BANKNIFTY lagging → WEAK RALLY fade -4% conf"
        elif nr < -0.20 and bna > -0.05:
            st,d,ca = "BULLISH_DIVERGENCE","BUY_CALL",+0.03
            note = f"NIFTY {nr:+.2f}% falling BUT BANKNIFTY holding → bounce likely +3% conf"
        else:
            st,d,ca = "NEUTRAL","NEUTRAL",0.0
            note = f"No divergence. NIFTY={nr:+.2f}% BN={bnr:+.2f}%"

        logger.debug(f"[BNDiv] {note}")
        return BreadthSignal(round(nr,3), round(bnr,3), round(div,3), st, d, ca, note)


# ── COMBINED CONTEXT ──────────────────────────────────────────────────────────

@dataclass
class MicrostructureContext:
    rv_iv_ratio:     float
    rv_iv_regime:    str
    lot_multiplier:  float
    rviv_conf_adj:   float
    max_pain_strike: Optional[int]
    max_pain_bias:   str
    fii_signal:      str
    fii_3day_net:    float
    fii_conf_adj:    float
    pre_open_bias:   str
    pre_open_gap_pct:float
    bn_signal:       str
    bn_direction:    str
    bn_conf_adj:     float
    total_conf_adj:  float
    overall_bias:    str
    note:            str

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class MarketMicrostructure:
    def __init__(self) -> None:
        self.rv_iv    = RVIVCalculator()
        self.max_pain = MaxPainCalculator()
        self.fii_dii  = FIIDIIFetcher()
        self.pre_open = PreOpenAnalyzer()
        self.bn_div   = BankNiftyDivergence()
        self._fii:    Optional[FIIDIIData]   = None
        self._po:     Optional[PreOpenSignal] = None
        self._vix:    float = 18.0

    def update_candle(self, nifty: float, banknifty: float, vix: float) -> None:
        self.rv_iv.update(nifty)
        self.bn_div.update(nifty, banknifty)
        self._vix = vix

    def update_pre_open(self, pre_open_price: float, prev_close: float) -> None:
        self._po = self.pre_open.analyze(pre_open_price, prev_close, self._vix)

    async def refresh_fii(self) -> None:
        self._fii = self.fii_dii.fetch_today()
        if not self._fii:
            logger.debug("[Microstructure] FII/DII fetch failed")

    def get_context(self, india_vix: float, spot: float,
                    ce_oi: dict = None, pe_oi: dict = None,
                    is_expiry: bool = False) -> MicrostructureContext:
        # 1. RV/IV
        rviv = self.rv_iv.compute(india_vix)
        # 2. Max Pain
        mp_strike, mp_bias = None, "N/A"
        if is_expiry and ce_oi and pe_oi:
            mp = self.max_pain.compute(spot, ce_oi, pe_oi)
            mp_strike, mp_bias = mp.max_pain_strike, mp.bias
        # 3. FII/DII
        fii_sig, fii_3d, fii_conf = "NEUTRAL", 0.0, 0.0
        if self._fii:
            fii_sig  = self._fii.signal
            fii_3d   = self._fii.fii_3day_sum
            fii_conf = +0.03 if fii_sig in ("STRONG_BULL","BULL") else \
                       -0.03 if fii_sig in ("STRONG_BEAR","BEAR") else 0.0
        # 4. Pre-Open
        po_bias, po_gap = "NEUTRAL", 0.0
        if self._po:
            po_bias = self._po.trading_bias
            po_gap  = self._po.gap_pct
        # 5. BankNifty
        bn     = self.bn_div.analyze()
        total  = round(rviv.confidence_adj + fii_conf + bn.confidence_adj, 4)
        bulls  = sum([fii_sig in ("STRONG_BULL","BULL"),
                      po_bias=="BUY_CALL", bn.direction=="BUY_CALL",
                      rviv.regime in ("UNDERPRICED","STRONGLY_UNDERPRICED")])
        bears  = sum([fii_sig in ("STRONG_BEAR","BEAR"),
                      po_bias=="BUY_PUT", bn.direction=="BUY_PUT"])
        overall = ("BULLISH" if bulls>=3 else "BEARISH" if bears>=3 else
                   "MILDLY_BULLISH" if bulls>bears else
                   "MILDLY_BEARISH" if bears>bulls else "NEUTRAL")
        note = (f"RVIV={rviv.rv_iv_ratio:.2f}({rviv.regime}) | "
                f"FII={fii_sig}(₹{fii_3d:.0f}Cr) | "
                f"PreOpen={po_bias}({po_gap:+.2f}%) | "
                f"BN={bn.signal_type} | Adj={total:+.3f}")
        return MicrostructureContext(
            rv_iv_ratio=rviv.rv_iv_ratio, rv_iv_regime=rviv.regime,
            lot_multiplier=rviv.lot_multiplier, rviv_conf_adj=rviv.confidence_adj,
            max_pain_strike=mp_strike, max_pain_bias=mp_bias,
            fii_signal=fii_sig, fii_3day_net=fii_3d, fii_conf_adj=fii_conf,
            pre_open_bias=po_bias, pre_open_gap_pct=po_gap,
            bn_signal=bn.signal_type, bn_direction=bn.direction,
            bn_conf_adj=bn.confidence_adj, total_conf_adj=total,
            overall_bias=overall, note=note,
        )


_ms: MarketMicrostructure | None = None
def get_microstructure() -> MarketMicrostructure:
    global _ms
    if _ms is None: _ms = MarketMicrostructure()
    return _ms

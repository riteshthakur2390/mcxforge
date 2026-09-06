"""
S13 v1.2: OI Analysis - Enhanced with Multi-Strike logging & IV Confirmation
====================================================================
Now logs ATM, ITM, OTM strikes specifically for NIFTY.
Uses combination of ATM and near-strike OI for stronger signals.
v1.2: Added IV confirmation for better compatibility with S14.
"""
import pandas as pd
import numpy as np
from core.models import Direction
from config.settings import strategy as s
from loguru import logger

class OIAnalysisStrategy:
    name = "OIAnalysis"
    def __init__(self) -> None:
        self._recorder = None

    def set_recorder(self, recorder) -> None:
        """Called by Agent 1 to inject the OIRecorder instance."""
        self._recorder = recorder

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        if len(df) < s.S13_VOL_MA_PERIOD:
            return none

        close = float(df["close"].iloc[-1])
        volume = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0)
        vol_ma = float(volume.rolling(s.S13_VOL_MA_PERIOD).mean().iloc[-1])
        vol_ratio = float(volume.iloc[-1]) / max(vol_ma, 1)

        # PATH A: Real OI history via recorder
        if self._recorder is not None:
            result = self._evaluate_with_recorder(df, close, max(vol_ratio, 1.0))
            if result["direction"] != Direction.NONE:
                return result

        # Basic volume gate for historical OI/proxy paths. Live recorder OI is
        # already the primary flow signal and must not be blocked by zero index
        # volume at startup.
        if vol_ratio < s.S13_VOL_MULT:
            return none

        # PATH B: Historical option OI persisted by the option OHLCV collector.
        result = self._evaluate_with_historical_option_oi(df, close, vol_ratio)
        if result["direction"] != Direction.NONE:
            return result

        # PATH C: CMF proxy
        return self._evaluate_with_proxy(df, close, vol_ratio)

    def _evaluate_with_historical_option_oi(self, df: pd.DataFrame, close: float, vol_ratio: float) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        required = {"opt_oi_ATM_CE", "opt_oi_ATM_PE"}
        if not required.issubset(df.columns) or len(df) <= s.S13_OI_LOOKBACK:
            return none

        def oi_change(col: str) -> float:
            series = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
            now = float(series.iloc[-1])
            prev = float(series.iloc[-s.S13_OI_LOOKBACK - 1])
            if now <= 0 or prev <= 0:
                return 0.0
            return (now - prev) / prev * 100.0

        ce_atm_chg = oi_change("opt_oi_ATM_CE")
        pe_atm_chg = oi_change("opt_oi_ATM_PE")
        ce_itm_chg = oi_change("opt_oi_ITM_CE")
        ce_otm_chg = oi_change("opt_oi_OTM_CE")
        pe_itm_chg = oi_change("opt_oi_ITM_PE")
        pe_otm_chg = oi_change("opt_oi_OTM_PE")
        if not any(abs(v) >= s.S13_OI_CHANGE_THRESH for v in (ce_atm_chg, pe_atm_chg, ce_itm_chg, ce_otm_chg, pe_itm_chg, pe_otm_chg)):
            return none

        price_prev = float(df["close"].iloc[-s.S13_PRICE_LOOKBACK])
        price_chg_pct = (close - price_prev) / max(price_prev, 1) * 100
        is_price_up = price_chg_pct > s.S13_PRICE_THRESH
        is_price_down = price_chg_pct < -s.S13_PRICE_THRESH

        ce_falling = (ce_atm_chg < -s.S13_OI_CHANGE_THRESH) or (ce_otm_chg < -s.S13_OI_CHANGE_THRESH)
        pe_writing = (pe_atm_chg > s.S13_OI_CHANGE_THRESH) or (pe_itm_chg > s.S13_OI_CHANGE_THRESH)
        if is_price_up and ce_falling and pe_writing:
            return {
                "direction": Direction.BUY_CALL,
                "confidence": self._score(vol_ratio, price_chg_pct, "historical_option_oi", True),
                "name": self.name,
                "meta": {
                    "scenario": "historical_oi_bullish",
                    "ce_atm_chg": round(ce_atm_chg, 1),
                    "pe_atm_chg": round(pe_atm_chg, 1),
                },
            }

        pe_falling = (pe_atm_chg < -s.S13_OI_CHANGE_THRESH) or (pe_otm_chg < -s.S13_OI_CHANGE_THRESH)
        ce_writing = (ce_atm_chg > s.S13_OI_CHANGE_THRESH) or (ce_itm_chg > s.S13_OI_CHANGE_THRESH)
        if is_price_down and pe_falling and ce_writing:
            return {
                "direction": Direction.BUY_PUT,
                "confidence": self._score(vol_ratio, abs(price_chg_pct), "historical_option_oi", True),
                "name": self.name,
                "meta": {
                    "scenario": "historical_oi_bearish",
                    "ce_atm_chg": round(ce_atm_chg, 1),
                    "pe_atm_chg": round(pe_atm_chg, 1),
                },
            }

        return none

    def _evaluate_with_recorder(self, df: pd.DataFrame, close: float, vol_ratio: float) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        # ATM strike based on current price (rounded to 50 for Nifty)
        atm = int(round(close / 50) * 50)
        itm1_ce, otm1_ce = atm - 50, atm + 50
        itm1_pe, otm1_pe = atm + 50, atm - 50

        # Fetch OI changes over last 3 candles
        ce_atm_chg = self._recorder.get_oi_change(strike=atm, option_type="CE", lookback=s.S13_OI_LOOKBACK)
        pe_atm_chg = self._recorder.get_oi_change(strike=atm, option_type="PE", lookback=s.S13_OI_LOOKBACK)
        ce_itm_chg = self._recorder.get_oi_change(strike=itm1_ce, option_type="CE", lookback=s.S13_OI_LOOKBACK)
        ce_otm_chg = self._recorder.get_oi_change(strike=otm1_ce, option_type="CE", lookback=s.S13_OI_LOOKBACK)
        pe_itm_chg = self._recorder.get_oi_change(strike=itm1_pe, option_type="PE", lookback=s.S13_OI_LOOKBACK)
        pe_otm_chg = self._recorder.get_oi_change(strike=otm1_pe, option_type="PE", lookback=s.S13_OI_LOOKBACK)
        
        # Fetch IVs
        ce_iv = self._recorder.get_iv_at_candle(0, strike=atm, option_type="CE") or 0.14
        pe_iv = self._recorder.get_iv_at_candle(0, strike=atm, option_type="PE") or 0.14
        ce_iv_prev = self._recorder.get_iv_at_candle(3, strike=atm, option_type="CE") or 0.14
        pe_iv_prev = self._recorder.get_iv_at_candle(3, strike=atm, option_type="PE") or 0.14

        if ce_atm_chg is None or pe_atm_chg is None:
            return none

        # LOGGING
        logger.info(
            f"[S13-OI-IV] NIFTY={close:.1f} | ATM={atm} | "
            f"CE_OI_CHG: ATM={ce_atm_chg:.1f}%, ITM({itm1_ce})={ce_itm_chg or 0:.1f}%, OTM({otm1_ce})={ce_otm_chg or 0:.1f}% | "
            f"PE_OI_CHG: ATM={pe_atm_chg:.1f}%, ITM({itm1_pe})={pe_itm_chg or 0:.1f}%, OTM({otm1_pe})={pe_otm_chg or 0:.1f}% | "
            f"IV: CE={ce_iv:.2f}, PE={pe_iv:.2f}"
        )

        # Price direction
        price_prev = float(df["close"].iloc[-s.S13_PRICE_LOOKBACK])
        price_chg_pct = (close - price_prev) / max(price_prev, 1) * 100
        
        is_price_up = price_chg_pct > s.S13_PRICE_THRESH
        is_price_down = price_chg_pct < -s.S13_PRICE_THRESH
        
        # BULLISH: Price UP + CE OI Falling (Short Covering) + PE OI Rising (Writing)
        ce_falling = (ce_atm_chg < -s.S13_OI_CHANGE_THRESH) or ((ce_otm_chg or 0) < -s.S13_OI_CHANGE_THRESH)
        pe_writing = (pe_atm_chg > s.S13_OI_CHANGE_THRESH) or ((pe_itm_chg or 0) > s.S13_OI_CHANGE_THRESH)
        
        if is_price_up and ce_falling and pe_writing:
            # IV Confirmation: CE IV stable or falling is better for buying calls
            iv_confirmed = ce_iv <= ce_iv_prev * 1.05 # Allow small 5% rise
            conf = self._score(vol_ratio, price_chg_pct, "bullish_buildup", True)
            if iv_confirmed: conf += 0.05
            
            return {
                "direction": Direction.BUY_CALL,
                "confidence": round(min(0.88, conf), 2),
                "name": self.name,
                "meta": {
                    "scenario": "multi_strike_bullish",
                    "ce_atm_chg": round(ce_atm_chg, 1),
                    "pe_atm_chg": round(pe_atm_chg, 1),
                    "ce_iv": round(ce_iv, 2),
                    "iv_confirmed": iv_confirmed
                }
            }

        # BEARISH: Price DOWN + PE OI Falling (Short Covering) + CE OI Rising (Writing)
        pe_falling = (pe_atm_chg < -s.S13_OI_CHANGE_THRESH) or ((pe_otm_chg or 0) < -s.S13_OI_CHANGE_THRESH)
        ce_writing = (ce_atm_chg > s.S13_OI_CHANGE_THRESH) or ((ce_itm_chg or 0) > s.S13_OI_CHANGE_THRESH)

        if is_price_down and pe_falling and ce_writing:
            # IV Confirmation: PE IV stable or falling is better for buying puts
            iv_confirmed = pe_iv <= pe_iv_prev * 1.05
            conf = self._score(vol_ratio, abs(price_chg_pct), "bearish_buildup", True)
            if iv_confirmed: conf += 0.05

            return {
                "direction": Direction.BUY_PUT,
                "confidence": round(min(0.88, conf), 2),
                "name": self.name,
                "meta": {
                    "scenario": "multi_strike_bearish",
                    "ce_atm_chg": round(ce_atm_chg, 1),
                    "pe_atm_chg": round(pe_atm_chg, 1),
                    "pe_iv": round(pe_iv, 2),
                    "iv_confirmed": iv_confirmed
                }
            }

        return none

    def _evaluate_with_proxy(self, df: pd.DataFrame, close: float, vol_ratio: float) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}
        price_prev = float(df["close"].iloc[-s.S13_PRICE_LOOKBACK])
        price_chg_pct = (close - price_prev) / max(price_prev, 1) * 100

        if abs(price_chg_pct) < s.S13_PRICE_THRESH:
            return none

        cmf = self._compute_cmf(df.tail(s.S13_PCR_LOOKBACK + s.S13_CMF_PERIOD))
        if price_chg_pct > s.S13_PRICE_THRESH and cmf > s.S13_CMF_THRESHOLD:
            conf = self._score(vol_ratio, price_chg_pct, "proxy", False)
            return {"direction": Direction.BUY_CALL, "confidence": conf, "name": self.name}
        if price_chg_pct < -s.S13_PRICE_THRESH and cmf < -s.S13_CMF_THRESHOLD:
            conf = self._score(vol_ratio, abs(price_chg_pct), "proxy", False)
            return {"direction": Direction.BUY_PUT, "confidence": conf, "name": self.name}
        return none

    @staticmethod
    def _compute_cmf(df: pd.DataFrame, period: int = 10) -> float:
        try:
            h, l, c, v = df["high"], df["low"], df["close"], df["volume"]
            mfm = ((c - l) - (h - c)) / (h - l).replace(0, 0.001)
            return float((mfm * v).rolling(period).sum().iloc[-1] / v.rolling(period).sum().iloc[-1])
        except: return 0.0

    @staticmethod
    def _score(vol_ratio, price_move, scenario, real_oi) -> float:
        conf = s.S13_CONF_BASE
        if real_oi: conf += 0.05
        if price_move > 0.3: conf += 0.03
        if vol_ratio > 1.5: conf += 0.02
        return round(min(0.85, conf), 2)

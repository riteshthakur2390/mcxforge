"""
agents_code/agent9_regime/classifier.py  Market Regime Agent
"""

import asyncio
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Optional
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from core.bus import get_bus, Topic, Message
from core.models import Regime
from agents_code.agent2_strategy.market_structure import MarketStructureLiquidityEngine
from agents_code.agent9_regime.mtf_context import MTFContext
from agents_code.agent9_regime.vix_slope   import VIXSlope
from agents_code.agent9_regime.regime_detector import RegimeDetector
from agents_code.agent9_regime.wyckoff_phase_detector import WyckoffPhaseDetector
from config.settings import (
    ADX_TREND_THRESHOLD, ADX_CHOP_THRESHOLD,
    CHOP_INDEX_THRESHOLD, VIX_HIGH_THRESHOLD,GAP_BIAS_THRESHOLD_PCT,
    TREND_PERSIST_CANDLES, LLM_ENABLED,
)
from utils.llm import TaskType, call_llm_context_async
from utils.pipeline_logging import log_pipeline_stage


class MarketRegimeAgent:
    NAME = "RegimeAgent"

    def __init__(self) -> None:
        self.bus           = get_bus()
        self._india_vix = 0.0
        self._prev_close = 0.0 
        self._gap_bias = None 
        self._last_regime  = Regime.RANGING
        self._allowed      = 0
        self._suppressed   = 0
        self._mtf             = MTFContext()
        self._vix_slope       = VIXSlope()
        self._regime_detector = RegimeDetector()
        self._wyckoff_detector = WyckoffPhaseDetector()
        self._structure_engine = MarketStructureLiquidityEngine()
        self._last_regime_note_key = ""

    def register(self) -> None:
        self.bus.subscribe(Topic.CANDLES_READY,  self.on_candles)
        self.bus.subscribe(Topic.PREMARKET_BIAS, self.on_premarket)
        logger.info(f"[{self.NAME}] Registered.")

    async def on_premarket(self, msg: Message) -> None:
        self._india_vix = float(msg.payload.get("india_vix", 0))
        self._vix_slope.update(self._india_vix)
        logger.info(f"[{self.NAME}] VIX = {self._india_vix:.1f}")
        self._prev_close = float(
            msg.payload.get("prev_close", msg.payload.get("nifty_prev_close", 0))
        )
        self._gap_bias   = self._compute_gap_bias(msg.payload)
        logger.info(
            f"[{self.NAME}] VIX={self._india_vix:.1f} | "
            f"gap_bias={self._gap_bias or 'NONE'}"
        )

    async def on_candles(self, msg: Message) -> None:
        candles = msg.payload.get("candles", [])
        if not candles:
            return

        df = self._to_df(candles)
        detailed = self._regime_detector.compute(df, self._india_vix)
        regime = self._map_detailed_regime(detailed.label)
        _, details = self._classify(df)
        details["trend_persist"] = self._trend_persistence(df)
        details["detailed_regime"] = detailed.to_dict()
        details["regime_label"] = detailed.label
        details["regime_confidence"] = detailed.confidence
        details["regime_sub_label"] = detailed.sub_label
        wyckoff = self._wyckoff_detector.compute(df)
        details["wyckoff"] = wyckoff.to_dict()
        details["market_structure"] = self._structure_engine.evaluate(
            df=df,
            cache=None,
            regime_details={"detailed_regime": detailed.to_dict(), "wyckoff": details["wyckoff"]},
        )
        mtf = self._mtf.compute(df)
        details["mtf"] = self._mtf.to_dict()
        details["gap_bias"] = self._gap_bias
        details["is_expiry"] = self._is_expiry_day(df)
        details["india_vix"] = self._india_vix
        details["vix"] = self._india_vix
        details["vix_slope"] = self._vix_slope.slope
        details["vix_regime"] = self._vix_slope.regime

        wyckoff_rec = str((details.get("wyckoff") or {}).get("wyckoff_recommendation", "NEUTRAL") or "NEUTRAL").upper()
        has_wyckoff_bias = "FAVOR" in wyckoff_rec

        signal_permitted = (
            regime == Regime.TRENDING
            or (regime == Regime.RANGING and self._allow_ranging_signals(details, detailed))
            or (regime == Regime.HIGH_VOL and (has_wyckoff_bias or (0.0 < self._india_vix <= 22.0)))
        )
        suppression_reason = "" if signal_permitted else self._reason(regime, details, detailed)
        base = {
            **msg.payload,
            "regime": regime.value,
            "regime_details": details,
            "signal_permitted": signal_permitted,
            "suppression_reason": suppression_reason,
        }

        await self.bus.publish(Topic.MARKET_REGIME, base, self.NAME)
        log_pipeline_stage(
            self.NAME,
            "regime_filter",
            "passed" if signal_permitted else "filtered",
            reason=suppression_reason,
            market_ts=str(msg.payload.get("timestamp", "")),
            regime=regime.value,
            det_label=detailed.label,
            det_conf=f"{detailed.confidence:.2f}",
            adx=f"{float(details.get('adx', 0.0) or 0.0):.1f}",
            chop=f"{float(details.get('chop_index', 0.0) or 0.0):.1f}",
            atr_ratio=f"{float(details.get('detailed_regime', {}).get('atr_ratio', 0.0) or 0.0):.2f}",
            vix=f"{self._india_vix:.1f}",
            gap_bias=self._gap_bias or "NONE",
            wyckoff=details.get("wyckoff", {}).get("wyckoff_phase", "NEUTRAL"),
            wyckoff_rec=details.get("wyckoff", {}).get("wyckoff_recommendation", "NEUTRAL"),
        )
        if LLM_ENABLED:
            asyncio.create_task(
                self._maybe_publish_regime_context(
                    regime=regime,
                    details=details,
                    detailed=detailed,
                    timestamp=str(msg.payload.get("timestamp", "")),
                    signal_permitted=signal_permitted,
                )
            )

        if signal_permitted:
            self._allowed += 1
            mtf_str = (
                f"mtf={details.get('mtf', {}).get('bias', '?')} "
                f"(conf={details.get('mtf', {}).get('confidence', 0):.2f})"
            ) if "mtf" in details else "mtf=N/A"

            if self._allowed % 5 == 1:
                logger.info(
                    f"[{self.NAME}] [OK] TRENDING | "
                    f"det={detailed.label}({detailed.confidence:.2f}) | "
                    f"ADX={details.get('adx', 0):.1f} | "
                    f"allowed={self._allowed} candles today"
                )
        else:
            self._suppressed += 1
            reason = suppression_reason
            if self._suppressed % 10 == 1:
                total = self._allowed + self._suppressed
                logger.info(
                    f"[{self.NAME}] [SILENT] {regime.value} | det={detailed.label}({detailed.confidence:.2f}) | {reason} | "
                    f"suppressed={self._suppressed}/{total} candles today"
                )
            await self.bus.publish(Topic.SIGNAL_SUPPRESSED, {
                "regime": regime.value, "reason": reason,
                "details": details,
                "message": f"[SILENT] {regime.value}  {reason}",
                "timestamp": msg.payload.get("timestamp", ""),
            }, self.NAME)

    def _classify(self, df: pd.DataFrame) -> tuple[Regime, dict]:
        details: dict = {}
        if self._india_vix > VIX_HIGH_THRESHOLD:
            details["vix"] = self._india_vix
            return Regime.HIGH_VOL, details

        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is not None:
            adx      = float(adx_df["ADX_14"].iloc[-1])
            plus_di  = float(adx_df["DMP_14"].iloc[-1])
            minus_di = float(adx_df["DMN_14"].iloc[-1])
        else:
            adx = plus_di = minus_di = 0.0
        details.update({"adx": round(adx, 2), "plus_di": round(plus_di, 2), "minus_di": round(minus_di, 2)})
        chop = self._chop_index(df, period=14)
        details["chop_index"] = round(chop, 2)
        
        if adx < ADX_CHOP_THRESHOLD:
            return Regime.CHOPPY, details
        if chop > CHOP_INDEX_THRESHOLD and adx < ADX_TREND_THRESHOLD:
            return Regime.RANGING, details
        return Regime.TRENDING, details

    @staticmethod
    def _map_detailed_regime(label: str) -> Regime:
        normalized = str(label or "").upper()
        if normalized == "TRENDING":
            return Regime.TRENDING
        if normalized == "HIGH_VOLATILITY":
            return Regime.HIGH_VOL
        return Regime.RANGING

    @staticmethod
    def _trend_persistence(df: pd.DataFrame) -> int:
        closes = df["close"].values
        if len(closes) < 2:
            return 0
        direction = 1 if closes[-1] > closes[-2] else -1
        count = 1
        for i in range(2, min(len(closes), 15)):
            c = 1 if closes[-i] > closes[-i - 1] else -1
            if c == direction:
                count += 1
            else:
                break
        return count

    def _compute_gap_bias(self, payload: dict) -> Optional[str]:
        gap_pct = float(payload.get("gap_pct", 0))
        if gap_pct >= GAP_BIAS_THRESHOLD_PCT:
            return "BUY_CALL"
        if gap_pct <= -GAP_BIAS_THRESHOLD_PCT:
            return "BUY_PUT"
        return None

    @staticmethod
    def _is_expiry_day(df: pd.DataFrame) -> bool:
        try:
            last_date = df.index[-1]
            d = last_date.date() if hasattr(last_date, 'date') else last_date
            from utils.option_utils import is_expiry_day
            sym = os.getenv("INSTRUMENT", "SILVERM")
            return is_expiry_day(d, sym)
        except Exception:
            return False

    @staticmethod
    def _allow_ranging_signals(details: dict, detailed) -> bool:
        try:
            adx = float(details.get("adx", 0.0) or 0.0)
            chop = float(details.get("chop_index", 100.0) or 100.0)
            sub = str(getattr(detailed, "sub_label", "") or "").upper()
        except Exception:
            return False
        if sub == "WIDE_RANGE":
            return False
        return adx >= 16.0 and chop <= 58.0

    @staticmethod
    def _chop_index(df: pd.DataFrame, period: int = 14) -> float:
        try:
            n     = period + 1
            hi    = df["high"].tail(n)
            lo    = df["low"].tail(n)
            cl    = df["close"].tail(n)
            tr    = pd.concat([hi - lo, (hi - cl.shift(1)).abs(), (lo - cl.shift(1)).abs()], axis=1).max(axis=1)
            atr_sum     = tr.sum()
            price_range = hi.max() - lo.min()
            if price_range == 0 or atr_sum == 0:
                return 50.0
            chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
            return float(np.clip(chop, 0, 100))
        except Exception:
            return 50.0

    @staticmethod
    def _reason(regime: Regime, details: dict, detailed) -> str:
        if getattr(detailed, "label", "") == "HIGH_VOLATILITY":
            return f"detailed={detailed.sub_label} | conf={detailed.confidence:.2f}"
        if getattr(detailed, "label", "") == "RANGING":
            return f"detailed={detailed.sub_label} | chop={details.get('chop_index', 0):.1f}"
        if regime == Regime.HIGH_VOL:
            vix = details.get("vix", 0)
            return f"VIX={vix:.1f}" if vix > 0 else f"High Volatility (ATR={details.get('atr', 0):.1f})"
        if regime == Regime.CHOPPY:
            return f"ADX={details.get('adx', 0):.1f} | Chop={details.get('chop_index', 0):.1f}"
        return f"ADX={details.get('adx', 0):.1f}"

    @staticmethod
    def _to_df(candles: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.set_index("datetime").sort_index()

    async def _maybe_publish_regime_context(self, **kwargs) -> None:
        pass

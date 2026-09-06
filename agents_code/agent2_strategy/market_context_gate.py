"""
agents_code/agent2_strategy/market_context_gate.py — Pipeline Integration
================================================================================
Sits between raw strategy signals and the ML filter. Per spec: "Every
generated signal must pass through a Market Context Engine before being
published." Attaches the Trade Quality Score + full explainability data
to every signal — it does NOT replace the ML filter, it enriches what
reaches it (and what reaches the trader/UI downstream).

Separate from signal generation (agent2_strategy/runner.py) and separate
from the ML approval gate (agent3_ml/filter.py) — exactly the "modular
architecture, separate Market Context Engine from signal generation
logic" requirement.

WIRING:
  RAW_SIGNAL → [this agent attaches trade_quality] → re-published as
  RAW_SIGNAL (enriched) for the ML filter to consume, OR SIGNAL_SUPPRESSED
  if quality is below the (default-disabled) configurable threshold.
"""

from __future__ import annotations
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

try:
    from core.bus import Topic
except ImportError:
    class Topic:
        RAW_SIGNAL = "RAW_SIGNAL"
        SIGNAL_SUPPRESSED = "SIGNAL_SUPPRESSED"

from utils.market_intelligence.engine import get_market_intelligence_engine

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from config.settings import MARKET_INTEL_LOG_ALL_SCORES
except ImportError:
    MARKET_INTEL_LOG_ALL_SCORES = True


class MarketContextGate:
    """
    Agent 2.5 (conceptually): the Market Context Engine's pipeline
    attachment point. Reads RAW_SIGNAL, enriches with Trade Quality,
    republishes RAW_SIGNAL_ENRICHED for downstream consumers (ML filter,
    dashboard, logs).
    """

    NAME = "MarketContextGate"

    def __init__(self, bus=None) -> None:
        self.bus = bus
        self.engine = get_market_intelligence_engine()

    def register(self) -> None:
        if self.bus is None:
            try:
                from core.bus import get_bus
                self.bus = get_bus()
            except Exception:
                return
        self.bus.subscribe(Topic.RAW_SIGNAL, self.on_raw_signal)
        logger.info(f"[{self.NAME}] Registered — Market Intelligence Engine active with "
                    f"{len(self.engine._plugins)} plugins")

    async def on_raw_signal(self, msg) -> None:
        payload = dict(msg.payload)

        # Loop prevention guard: if msg came from MarketContextGate or already enriched, return
        if getattr(msg, "source", None) == self.NAME or "market_intelligence" in payload:
            return

        ctx = self._build_ctx(payload)
        report = self.engine.evaluate(ctx)

        payload["market_intelligence"] = report.to_dict()
        payload["trade_quality"] = report.trade_quality
        payload["trade_quality_band"] = report.quality_band
        payload["signal_output"] = report.to_signal_output(payload.get("symbol", "NIFTY"))

        # ── Context Alignment Score (CAS) ─────────────────────────────────────
        try:
            from utils.context_alignment_score import ContextAlignmentScore
            from utils.option_utils import is_expiry_day
            direction = str(payload.get("direction", "") or "").upper()
            ts_str = str(payload.get("timestamp") or "")
            try:
                sig_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                sig_dt = datetime.now(IST)
            is_exp = is_expiry_day(sig_dt.date(), symbol=str(payload.get("symbol", "NIFTY")))
            is_expiry_afternoon = is_exp and (sig_dt.hour >= 14 and sig_dt.minute >= 15)

            cas_res = ContextAlignmentScore.evaluate(
                direction=direction,
                context=payload.get("metadata", {}).get("_context", {}),
                market_intel=report.to_dict(),
                votes=int(payload.get("votes", 0) or 0),
                ml_rank_score=float(payload.get("ml_rank_score", 0.0) or 0.0),
                is_expiry_session=is_expiry_afternoon,
            )
            payload["cas"] = cas_res.to_dict()
            if "metadata" not in payload:
                payload["metadata"] = {}
            if "_context" not in payload["metadata"]:
                payload["metadata"]["_context"] = {}
            payload["metadata"]["_context"]["cas"] = cas_res.to_dict()
            payload["metadata"]["_context"]["cas_score"] = cas_res.score
            payload["metadata"]["_context"]["cas_override"] = cas_res.allows_structural_override
            logger.info(
                f"[{self.NAME}] CAS={cas_res.score:.2f} ({cas_res.verdict}) | "
                f"override={cas_res.allows_structural_override} | notes={cas_res.notes}"
            )
        except Exception as e:
            logger.debug(f"[{self.NAME}] CAS evaluation error: {e}")

        if MARKET_INTEL_LOG_ALL_SCORES:
            for s in report.plugin_scores:
                logger.debug(f"[{self.NAME}]   {s.plugin_name}: {s.score_contribution:+.1f} "
                            f"({s.label}) — {s.explanation}")

        if report.suppressed:
            logger.warning(f"[{self.NAME}] SUPPRESSED: {report.suppression_reason}")
            await self.bus.publish(Topic.SIGNAL_SUPPRESSED, payload, self.NAME)
            return

        # Republish enriched — ML filter and everything downstream sees
        # the SAME topic, just with market_intelligence and CAS attached.
        await self.bus.publish(Topic.RAW_SIGNAL, payload, self.NAME)

    def _build_ctx(self, payload: dict) -> dict:
        """
        Maps the raw signal payload + attached market data into the
        shared ctx dict every plugin reads from. Unpacks nested metadata
        and falls back to live recorder snapshots when present.
        """
        meta = payload.get("metadata") or {}
        context = meta.get("_context") or {}

        spot = (
            payload.get("spot")
            or payload.get("nifty_ltp")
            or meta.get("nifty_ltp")
            or meta.get("spot")
            or context.get("spot")
        )
        pcr = payload.get("pcr") or meta.get("pcr") or context.get("pcr")
        india_vix = (
            payload.get("india_vix")
            or meta.get("india_vix")
            or meta.get("vix")
            or context.get("india_vix")
            or context.get("vix")
        )
        if india_vix is None:
            try:
                from broker import get_active_broker
                b_vix = float(get_active_broker().get_india_vix() or 0.0)
                if 8.0 <= b_vix <= 80.0:
                    india_vix = b_vix
            except Exception:
                pass
        ce_oi = payload.get("ce_oi_by_strike") or meta.get("ce_oi_by_strike") or context.get("ce_oi_by_strike")
        pe_oi = payload.get("pe_oi_by_strike") or meta.get("pe_oi_by_strike") or context.get("pe_oi_by_strike")
        put_oi_change_pct = (
            payload.get("put_oi_change_pct")
            or meta.get("put_oi_change_pct")
            or meta.get("pe_oi_chg_pct")
            or context.get("put_oi_change_pct")
        )

        # Fallback to latest OI recorder snapshot if OI / PCR not in payload
        if ce_oi is None or pe_oi is None or pcr is None:
            try:
                from pathlib import Path
                import pandas as pd
                today_str = datetime.now(IST).strftime("%Y-%m-%d")
                oi_fpath = Path("data/oi_history") / f"oi_{today_str}.parquet"
                if oi_fpath.exists():
                    df_oi = pd.read_parquet(oi_fpath)
                    if not df_oi.empty:
                        latest_oi = df_oi.iloc[-1]
                        if pcr is None:
                            pcr = float(latest_oi.get("pcr", 0.0)) or None
                        if ce_oi is None or pe_oi is None:
                            atm = int(latest_oi.get("atm_strike", 0))
                            if atm > 0:
                                ce_oi = {
                                    atm - 100: float(latest_oi.get("itm2_ce_oi", 0)),
                                    atm - 50: float(latest_oi.get("itm1_ce_oi", 0)),
                                    atm: float(latest_oi.get("atm_ce_oi", 0)),
                                    atm + 50: float(latest_oi.get("otm1_ce_oi", 0)),
                                    atm + 100: float(latest_oi.get("otm2_ce_oi", 0)),
                                }
                                pe_oi = {
                                    atm - 100: float(latest_oi.get("itm2_pe_oi", 0)),
                                    atm - 50: float(latest_oi.get("itm1_pe_oi", 0)),
                                    atm: float(latest_oi.get("atm_pe_oi", 0)),
                                    atm + 50: float(latest_oi.get("otm1_pe_oi", 0)),
                                    atm + 100: float(latest_oi.get("otm2_pe_oi", 0)),
                                }
            except Exception:
                pass

        return {
            "direction":                  payload.get("direction", "BUY_CALL"),
            "spot":                       spot,
            "pcr":                        pcr,
            "regime":                     payload.get("regime") or meta.get("regime") or context.get("regime", ""),
            "is_expiry_day":              payload.get("is_expiry_day") or meta.get("is_expiry_day", False),
            "gap_direction":              payload.get("gap_direction") or meta.get("gap_direction", ""),
            "put_oi_change_pct":          put_oi_change_pct,
            "put_premium_change_pct":      payload.get("put_premium_change_pct") or meta.get("put_premium_change_pct"),
            "vix_change_pct":             payload.get("vix_change_pct") or meta.get("vix_change_pct") or meta.get("vix_roc"),
            "india_vix":                  india_vix,
            "ce_oi_by_strike":            ce_oi,
            "pe_oi_by_strike":            pe_oi,
            "fii_signal":                 payload.get("fii_signal") or meta.get("fii_signal"),
            "price_exhaustion":           payload.get("price_exhaustion") or meta.get("price_exhaustion", False),
            "oi_divergence":              payload.get("oi_divergence") or meta.get("oi_divergence", False),
            "call_writing_heavy":         payload.get("call_writing_heavy") or meta.get("call_writing_heavy", False),
            "put_writing_heavy":          payload.get("put_writing_heavy") or meta.get("put_writing_heavy", False),
            "vwap_failure":               payload.get("vwap_failure") or meta.get("vwap_failure", False),
            "momentum_weakening":         payload.get("momentum_weakening") or meta.get("momentum_weakening", False),
            "large_rejection_candle":     payload.get("large_rejection_candle") or meta.get("large_rejection_candle", False),
            "support_resistance_failure": payload.get("support_resistance_failure") or meta.get("support_resistance_failure", False),
        }


def get_market_context_gate(bus=None) -> MarketContextGate:
    return MarketContextGate(bus)

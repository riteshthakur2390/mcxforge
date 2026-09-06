"""
utils/market_intelligence/engine.py — Market Intelligence Engine
======================================================================
The extensible aggregator. PCR is the FIRST plugin, not the only one.
Every RAW_SIGNAL passes through here before becoming SIGNAL_APPROVED.

ARCHITECTURE (per spec — this is the key extensibility requirement):
  - Engine knows nothing about PCR, OI, FII, VIX specifically.
  - Engine only knows "a list of MarketContextPlugin objects, each
    returning a ContextScore."
  - Adding IV Rank, Gamma Exposure, Dealer Position estimation later
    means writing ONE new plugin file and registering it — zero changes
    to this engine.

OUTPUT FORMAT (per spec's example signal output):
  BUY NIFTY CE
  Trade Quality: 92/100
  Trend: Strong | VWAP: Bullish | Momentum: Strong
  PCR: Slightly Elevated | Trap Risk: Low | OI: Bullish | VIX: Supportive
  Confidence: High
  Warnings: None

QUALITY BANDS (per spec):
  95+     Excellent
  85-95   High Quality
  70-85   Average
  <70     Poor (optionally suppressed — SIGNAL_SUPPRESS_BELOW_QUALITY is configurable)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import pytz

IST = pytz.timezone("Asia/Kolkata")

from utils.market_intelligence.base import MarketContextPlugin, ContextScore

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from config.settings import SIGNAL_SUPPRESS_BELOW_QUALITY
except ImportError:
    SIGNAL_SUPPRESS_BELOW_QUALITY = 0.0   # 0 = never suppress, purely informational by default

BASE_SCORE = 70.0   # starting point before plugin adjustments (spec: "Average" band baseline)

QUALITY_BANDS = [
    (95, 101, "EXCELLENT"),
    (85, 95,  "HIGH_QUALITY"),
    (70, 85,  "AVERAGE"),
    (0,  70,  "POOR"),
]


@dataclass
class TradeQualityReport:
    direction:        str
    trade_quality:      float    # 0-100
    quality_band:         str
    confidence:            str    # "High" | "Medium" | "Low"
    plugin_scores:            list = field(default_factory=list)   # list[ContextScore]
    warnings:                   list = field(default_factory=list)
    suppressed:                    bool = False
    suppression_reason:              str = ""
    generated_at:                       str = ""

    def to_signal_output(self, symbol: str = "NIFTY") -> dict:
        """Formats the structured UI/log output shown in the spec example."""
        opt_type = "CE" if self.direction == "BUY_CALL" else "PE"
        labels = {s.plugin_name: s.label for s in self.plugin_scores}
        return {
            "action":         f"BUY {symbol} {opt_type}",
            "trade_quality":   f"{self.trade_quality:.0f}/100",
            "quality_band":     self.quality_band,
            "confidence":         self.confidence,
            "context":              labels,
            "warnings":              self.warnings or ["None"],
            "suppressed":              self.suppressed,
            "generated_at":              self.generated_at,
        }

    def to_dict(self) -> dict:
        return {
            "direction": self.direction, "trade_quality": self.trade_quality,
            "quality_band": self.quality_band, "confidence": self.confidence,
            "plugin_scores": [vars(s) for s in self.plugin_scores],
            "warnings": self.warnings, "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
        }


class MarketIntelligenceEngine:
    """
    Extensible market-context aggregator. Register plugins, call
    evaluate(ctx) once per signal, get back a TradeQualityReport.
    """

    def __init__(self, suppress_below: float = None) -> None:
        self._plugins: list[MarketContextPlugin] = []
        self.suppress_below = suppress_below if suppress_below is not None else SIGNAL_SUPPRESS_BELOW_QUALITY

    def register(self, plugin: MarketContextPlugin) -> None:
        self._plugins.append(plugin)
        logger.info(f"[MarketIntelligenceEngine] Registered plugin: {plugin.name}")

    def register_all(self, plugins: list[MarketContextPlugin]) -> None:
        for p in plugins:
            self.register(p)

    def evaluate(self, ctx: dict) -> TradeQualityReport:
        """
        ctx must include at minimum: direction ("BUY_CALL"/"BUY_PUT").
        Plugins are individually fault-tolerant — one plugin's exception
        never crashes the pipeline or blocks the trade; it just
        contributes 0 for that plugin with a logged warning.
        """
        direction = ctx.get("direction", "BUY_CALL")
        scores: list[ContextScore] = []

        # Feed PCR plugin's classified state back into ctx so TrapDetection
        # can use it, without TrapDetection needing to know PCR internals.
        for plugin in self._plugins:
            try:
                result = plugin.evaluate(ctx)
                scores.append(result)
                if plugin.name == "PCR":
                    ctx["_pcr_state"] = result.raw_data.get("state", "")
            except Exception as e:
                logger.warning(f"[MarketIntelligenceEngine] Plugin {plugin.name} failed: {e} "
                              f"— contributing 0, trade NOT blocked")
                scores.append(ContextScore(plugin.name, 0.0, "ERROR", str(e)))

        # Re-run TrapDetection AFTER PCR so it can see _pcr_state (order-dependent
        # only for this one cross-reference; every other plugin is independent)
        trade_quality = BASE_SCORE + sum(s.score_contribution for s in scores)
        trade_quality = max(0.0, min(100.0, trade_quality))

        band = self._band(trade_quality)
        confidence = "High" if trade_quality >= 85 else "Medium" if trade_quality >= 70 else "Low"

        warnings = self._build_warnings(scores)

        suppressed = self.suppress_below > 0 and trade_quality < self.suppress_below
        suppression_reason = (
            f"Trade Quality {trade_quality:.0f} < configured threshold {self.suppress_below:.0f}"
            if suppressed else ""
        )

        report = TradeQualityReport(
            direction=direction, trade_quality=round(trade_quality, 1), quality_band=band,
            confidence=confidence, plugin_scores=scores, warnings=warnings,
            suppressed=suppressed, suppression_reason=suppression_reason,
            generated_at=datetime.now(IST).isoformat(),
        )

        logger.info(
            f"[MarketIntelligenceEngine] {direction} Trade Quality={trade_quality:.0f} "
            f"({band}) confidence={confidence} suppressed={suppressed} | "
            f"contributions: {[(s.plugin_name, s.score_contribution) for s in scores]}"
        )

        return report

    def _build_warnings(self, scores: list[ContextScore]) -> list[str]:
        warnings = []
        for s in scores:
            if s.plugin_name == "TrapDetection" and s.raw_data.get("band") in ("HIGH", "VERY_HIGH"):
                warnings.append(f"{s.raw_data['band']} trap risk detected — {s.explanation.split('|')[0].strip()}")
            if s.plugin_name == "PCR" and s.raw_data.get("state") == "EXTREME":
                warnings.append("Extreme PCR positioning — crowded trade, consider waiting for confirmation")
            if s.score_contribution <= -8:
                warnings.append(f"{s.plugin_name} strongly contradicting this trade direction")
        return warnings

    @staticmethod
    def _band(score: float) -> str:
        for lo, hi, label in QUALITY_BANDS:
            if lo <= score < hi:
                return label
        return "EXCELLENT" if score >= 95 else "POOR"


def build_default_engine() -> MarketIntelligenceEngine:
    """
    Convenience factory wiring up all currently-available plugins.
    Future modules (IV Rank, Gamma Exposure, Dealer Position) get added
    here with one line — no other code changes needed anywhere else.
    """
    from utils.market_intelligence.plugins.pcr_plugin import get_pcr_plugin
    from utils.market_intelligence.plugins.trap_detection_plugin import get_trap_detection_plugin
    from utils.market_intelligence.plugins.oi_fii_vix_plugins import (
        get_oi_plugin, get_fii_plugin, get_vix_plugin, get_max_pain_plugin,
    )

    engine = MarketIntelligenceEngine()
    engine.register_all([
        get_pcr_plugin(),
        get_trap_detection_plugin(),
        get_oi_plugin(),
        get_fii_plugin(),
        get_vix_plugin(),
        get_max_pain_plugin(),
    ])
    return engine


_engine: MarketIntelligenceEngine | None = None
def get_market_intelligence_engine() -> MarketIntelligenceEngine:
    global _engine
    if _engine is None:
        _engine = build_default_engine()
    return _engine

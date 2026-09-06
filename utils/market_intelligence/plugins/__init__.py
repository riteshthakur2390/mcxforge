"""
utils/market_intelligence/plugins package
"""
from utils.market_intelligence.plugins.pcr_plugin import PCRPlugin, get_pcr_plugin
from utils.market_intelligence.plugins.trap_detection_plugin import TrapDetectionEngine, TrapDetectionEngine as TrapDetectionPlugin
from utils.market_intelligence.plugins.oi_fii_vix_plugins import OIPlugin, FIIPlugin, FIIPlugin as FIIDIIPlugin, VIXPlugin, MaxPainPlugin

__all__ = [
    "PCRPlugin",
    "get_pcr_plugin",
    "TrapDetectionEngine",
    "TrapDetectionPlugin",
    "OIPlugin",
    "FIIPlugin",
    "FIIDIIPlugin",
    "VIXPlugin",
    "MaxPainPlugin",
]

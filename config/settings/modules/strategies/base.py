import os
from config.settings.utils import *




MIN_STRATEGY_VOTES = 4
ALLOW_SUBMIN_VOTE_EARLY_TRIGGER = os.getenv(
    "ALLOW_SUBMIN_VOTE_EARLY_TRIGGER",
    "true",
).lower() in {"1", "true", "yes", "on"}

MIN_STRATEGY_CONF = 0.45

STRATEGY_BASE_WEIGHTS = _parse_float_map(
    os.getenv(
        "STRATEGY_BASE_WEIGHTS",
        (
            "SuperTrend+RSI:1.10,"
            "VWAP+EMA:1.08,"
            "ORB:1.12,"
            "BBSqueeze:1.00,"
            "ADX+PSAR:1.12,"
            "FVG:0.98,"
            "UTBot:1.02,"
            "CPR:1.05,"
            "Ichimoku:1.10,"
            "VolumeProfile:1.06,"
            "LiqSweep:0.96,"
            "PriceAction:0.98,"
            "OIAnalysis:1.00,"
            "IVContraction:1.04,OIIVConfluence:1.15,"
            "ValueArea:0.90,"
            "StrikeMomentum:1.02,"
            "GapMomentum:1.04,"
            "ADXRising:0.88,"
            "RangeSpread:1.02,"
            "SqueezeMomentum:1.00,"
            "StochRSI:1.00,"
            "EMASlope:1.00,"
            "HeikinAshi:1.00,"
            "VWAPExtreme:1.00,"
            "VIXDivergence:1.00,"
            "OpeningRangeBias:1.00,"
            "GammaExposure:1.00"
        ),
    ),
    {},
)

STRATEGY_REGIME_WEIGHTS = {
    "TRENDING": _parse_float_map(
        os.getenv(
            "STRATEGY_REGIME_WEIGHTS_TRENDING",
            (
                "SuperTrend+RSI:1.15,VWAP+EMA:1.12,ORB:1.12,BBSqueeze:0.96,"
                "ADX+PSAR:1.15,FVG:1.05,UTBot:1.08,CPR:1.02,Ichimoku:1.15,"
                "VolumeProfile:1.06,LiqSweep:0.94,PriceAction:0.95,"
                "OIAnalysis:1.02,IVContraction:1.05,OIIVConfluence:1.20,"
                "ValueArea:0.90,StrikeMomentum:1.04,GapMomentum:1.06,"
                "ADXRising:0.88,RangeSpread:0.98,"
                "SqueezeMomentum:1.10,StochRSI:1.05,EMASlope:1.10,HeikinAshi:1.08,"
                "VWAPExtreme:0.95,VIXDivergence:0.95,OpeningRangeBias:1.10,GammaExposure:1.00"
            ),
        ),
        {},
    ),
    "RANGING": _parse_float_map(
        os.getenv(
            "STRATEGY_REGIME_WEIGHTS_RANGING",
            (
                "SuperTrend+RSI:0.92,VWAP+EMA:0.96,ORB:0.90,BBSqueeze:1.12,"
                "ADX+PSAR:0.92,FVG:1.08,UTBot:0.97,CPR:1.12,Ichimoku:0.93,"
                "VolumeProfile:1.08,LiqSweep:1.10,PriceAction:1.12,"
                "OIAnalysis:0.98,IVContraction:1.08,OIIVConfluence:1.15,"
                "ValueArea:0.96,StrikeMomentum:0.94,GapMomentum:0.92,"
                "ADXRising:0.86,RangeSpread:1.12,"
                "SqueezeMomentum:0.95,StochRSI:1.10,EMASlope:0.95,HeikinAshi:0.92,"
                "VWAPExtreme:1.15,VIXDivergence:1.15,OpeningRangeBias:0.90,GammaExposure:1.10"
            ),
        ),
        {},
    ),
    "HIGH_VOLATILITY": _parse_float_map(
        os.getenv(
            "STRATEGY_REGIME_WEIGHTS_HIGH_VOLATILITY",
            (
                "SuperTrend+RSI:0.90,VWAP+EMA:0.92,ORB:0.88,BBSqueeze:0.96,"
                "ADX+PSAR:0.92,FVG:1.02,UTBot:0.96,CPR:0.94,Ichimoku:0.90,"
                "VolumeProfile:1.00,LiqSweep:1.05,PriceAction:1.02,"
                "OIAnalysis:0.96,IVContraction:1.10,OIIVConfluence:1.25,"
                "ValueArea:0.90,StrikeMomentum:1.00,GapMomentum:0.94,"
                "ADXRising:0.86,RangeSpread:1.02,"
                "SqueezeMomentum:1.00,StochRSI:1.00,EMASlope:1.00,HeikinAshi:1.00,"
                "VWAPExtreme:1.10,VIXDivergence:1.10,OpeningRangeBias:1.00,GammaExposure:1.10"
            ),
        ),
        {},
    ),
}

EARLY_TRIGGER_MIN_CONFIDENCE = float(os.getenv("EARLY_TRIGGER_MIN_CONFIDENCE", "0.74"))

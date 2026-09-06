"""S3: Opening Range Breakout  valid 09:3012:00 by candle timestamp."""
import pandas as pd
import pytz
from core.models import Direction
from config.settings.strategy import (
    S3_MAX_HOUR, S3_VOL_MA_PERIOD, S3_VOL_INCREASE_FACTOR,
    S3_BUFFER_RANGE_MULTIPLIER, S3_BUFFER_MIN_POINTS, S3_RETEST_BUFFER_MULTIPLIER,
    S3_CONF_MAX, S3_CONF_BASE, S3_CONF_STRENGTH_MULTIPLIER
)

IST = pytz.timezone("Asia/Kolkata")


class ORBStrategy:
    name = "ORB"

    def evaluate(self, df: pd.DataFrame, orb_high=None, orb_low=None) -> dict:
        none = {"direction": Direction.NONE, "confidence": 0.0, "name": self.name}

        if orb_high is None or orb_low is None:
            return none

        candle_ts = df.index[-1]
        if candle_ts.tzinfo is None:
            candle_ts = IST.localize(candle_ts)
        else:
            candle_ts = candle_ts.tz_convert(IST)
        if candle_ts.hour >= S3_MAX_HOUR:
            return none  # ORB only valid morning session

        orb_range = orb_high - orb_low
        if orb_range <= 0:
            return none

        close   = df["close"].iloc[-1]
        close_prev = df["close"].iloc[-2]
        high = df["high"].iloc[-1]
        low = df["low"].iloc[-1]
        vol_ma  = df["volume"].rolling(S3_VOL_MA_PERIOD).mean().iloc[-1]
        vol_ok  = df["volume"].iloc[-1] > S3_VOL_INCREASE_FACTOR * vol_ma
        buffer  = max(orb_range * S3_BUFFER_RANGE_MULTIPLIER, S3_BUFFER_MIN_POINTS)
        breakout_up = close > (orb_high + buffer)
        breakout_dn = close < (orb_low - buffer)
        retest_up = low <= (orb_high + buffer * S3_RETEST_BUFFER_MULTIPLIER) and close > orb_high and close > close_prev
        retest_dn = high >= (orb_low - buffer * S3_RETEST_BUFFER_MULTIPLIER) and close < orb_low and close < close_prev

        if (breakout_up or retest_up) and vol_ok:
            strength = (close - orb_high) / orb_range
            conf = round(min(S3_CONF_MAX, S3_CONF_BASE + strength * S3_CONF_STRENGTH_MULTIPLIER), 4)
            return {"direction": Direction.BUY_CALL, "confidence": conf,
                    "name": self.name,
                    "meta": {"orb_high": orb_high, "breakout_str": round(strength, 2)}}

        if (breakout_dn or retest_dn) and vol_ok:
            strength = (orb_low - close) / orb_range
            conf = round(min(S3_CONF_MAX, S3_CONF_BASE + strength * S3_CONF_STRENGTH_MULTIPLIER), 4)
            return {"direction": Direction.BUY_PUT, "confidence": conf,
                    "name": self.name,
                    "meta": {"orb_low": orb_low, "breakdown_str": round(strength, 2)}}

        return none

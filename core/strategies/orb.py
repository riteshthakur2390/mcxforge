"""
core/strategies/orb.py — Opening Range Breakout (ORB) Strategy Family
=====================================================================
Identifies the opening price discovery range for MCX commodity futures
and executes confirmed breakouts with volume and buffer validation.

Respects MCX opening session hours (09:00 IST) rather than equity assumptions.
Filters out overextended or abnormally tight ranges, and restricts entries
to an active intraday window.
"""

from datetime import datetime, time, timedelta
from typing import Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import pytz

from core.models import Direction, Position
from core.regime.engine import MarketRegime, RegimeDetails
from instruments.base import ContractSpec
from core.strategies.base import BaseCommodityStrategy, StrategySignal

IST = pytz.timezone("Asia/Kolkata")


class OpeningRangeBreakoutStrategy(BaseCommodityStrategy):
    """
    Opening Range Breakout calibrated for MCX commodity futures sessions.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "orb_duration_minutes": 30,         # 09:00 to 09:30 range
            "breakout_buffer_atr_mult": 0.15,   # Buffer above/below ORB
            "min_range_atr_mult": 0.5,          # Avoid dead, noisy ranges
            "max_range_atr_mult": 3.0,          # Avoid massive exhaustion ranges
            "volume_multiplier": 1.25,          # Breakout bar volume vs 20 SMA
            "risk_reward_ratio": 1.8,
            "sl_range_ratio": 0.5,              # SL placed at midpoint of ORB or ATR
            "trailing_stop_atr_mult": 1.2,
            "trading_window_hours": 4.0,        # Entries only within 4h after ORB
            "max_holding_bars": 90,
        }
        super().__init__(
            name="OpeningRangeBreakout",
            version="1.0.0",
            permitted_regimes=[MarketRegime.BREAKOUT, MarketRegime.HIGH_VOLATILITY, MarketRegime.TREND],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates ATR and volume moving average."""
        data = df.copy()
        c = data["close"]
        h = data["high"]
        l = data["low"]
        v = data["volume"]

        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["tr"] = tr
        data["atr"] = tr.rolling(14, min_periods=1).mean()
        data["vol_ma"] = v.rolling(20, min_periods=1).mean()
        data["vol_ratio"] = v / data["vol_ma"].replace(0, 1e-6)

        return data

    def _determine_opening_range(self, df: pd.DataFrame, session_open_time_str: str) -> Tuple[Optional[float], Optional[float], Optional[datetime]]:
        """
        Extracts ORB High and Low for the current trading day based on instrument open time.
        """
        if df.empty:
            return None, None, None

        open_parts = [int(p) for p in session_open_time_str.split(":")]
        open_time = time(open_parts[0], open_parts[1])
        orb_duration = self.parameters["orb_duration_minutes"]

        # Find today's session candles
        latest_dt = df.index[-1] if isinstance(df.index[-1], (datetime, pd.Timestamp)) else None
        if not latest_dt:
            return None, None, None

        today_date = latest_dt.date()
        # Filter candles for current trading day
        day_mask = [idx.date() == today_date if isinstance(idx, (datetime, pd.Timestamp)) else False for idx in df.index]
        day_df = df[day_mask]

        if day_df.empty:
            return None, None, None

        if getattr(df.index, "tz", None) is not None:
            session_start = datetime.combine(today_date, open_time).replace(tzinfo=df.index.tz)
        else:
            session_start = datetime.combine(today_date, open_time)
        orb_end = session_start + timedelta(minutes=orb_duration)

        orb_candles = day_df[(day_df.index >= session_start) & (day_df.index <= orb_end)]
        if len(orb_candles) < 2:
            return None, None, None

        orb_high = float(orb_candles["high"].max())
        orb_low = float(orb_candles["low"].min())
        return orb_high, orb_low, orb_end

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_contract: Optional[ContractSpec] = None,
        regime_details: Optional[RegimeDetails] = None,
    ) -> StrategySignal:
        now_ts = datetime.now(IST)
        if len(df) > 0 and isinstance(df.index[-1], (datetime, pd.Timestamp)):
            now_ts = df.index[-1].to_pydatetime()

        contract_sym = current_contract.symbol if current_contract else (
            self.instrument_config.symbol if self.instrument_config else "MCX_FUT"
        )
        instrument_name = self.instrument_config.symbol if self.instrument_config else "COMMODITY"
        session_open = self.instrument_config.session.open_time if self.instrument_config else "09:00"

        empty_signal = StrategySignal(
            timestamp=now_ts,
            instrument=instrument_name,
            contract=contract_sym,
            strategy=self.name,
            strategy_version=self.version,
            direction=Direction.NONE,
            decision="NO_TRADE",
        )

        if len(df) < 6:
            empty_signal.rejection_reason = "INSUFFICIENT_DATA"
            return empty_signal

        orb_high, orb_low, orb_end = self._determine_opening_range(df, session_open)
        if orb_high is None or orb_low is None or orb_end is None:
            empty_signal.rejection_reason = "OPENING_RANGE_NOT_FORMED_YET"
            return empty_signal

        # Entry window check: Can only enter after ORB formation and within trading_window_hours
        window_end = orb_end + timedelta(hours=self.parameters["trading_window_hours"])
        if now_ts < orb_end:
            empty_signal.decision = "WAIT"
            empty_signal.rejection_reason = f"WAITING_FOR_ORB_COMPLETION (ends at {orb_end.strftime('%H:%M')})"
            return empty_signal

        if now_ts > window_end:
            empty_signal.rejection_reason = f"ORB_WINDOW_EXPIRED ({window_end.strftime('%H:%M')})"
            return empty_signal

        data = self.calculate_indicators(df)
        curr = data.iloc[-1]
        close = float(curr["close"])
        high = float(curr["high"])
        low = float(curr["low"])
        atr = float(curr["atr"])
        vol_ratio = float(curr["vol_ratio"])

        orb_range = orb_high - orb_low
        buffer = self.parameters["breakout_buffer_atr_mult"] * atr

        # Range filter checks
        min_range = self.parameters["min_range_atr_mult"] * atr
        max_range = self.parameters["max_range_atr_mult"] * atr

        # Prior Day Range Location Bias (merged from OpeningRangeBias)
        prev_day_bias = "NEUTRAL"
        if len(df) > 20:
            today_date = now_ts.date() if hasattr(now_ts, "date") else None
            if today_date:
                prev_day_mask = [idx.date() < today_date if hasattr(idx, "date") else False for idx in df.index]
                prev_day_candles = df[prev_day_mask]
                if len(prev_day_candles) >= 5:
                    prev_date = prev_day_candles.index[-1].date()
                    prev_session = prev_day_candles[[idx.date() == prev_date for idx in prev_day_candles.index]]
                    prev_high = float(prev_session["high"].max())
                    prev_low = float(prev_session["low"].min())
                    prev_range = prev_high - prev_low
                    if prev_range > 0:
                        today_candles = df[[idx.date() == today_date for idx in df.index]]
                        if len(today_candles) > 0:
                            today_open = float(today_candles["open"].iloc[0])
                            open_pos = (today_open - prev_low) / prev_range
                            if open_pos >= 0.70:
                                prev_day_bias = "BULL"
                            elif open_pos <= 0.30:
                                prev_day_bias = "BEAR"

        indicators_snapshot = {
            "close": close,
            "orb_high": round(orb_high, 2),
            "orb_low": round(orb_low, 2),
            "orb_range": round(orb_range, 2),
            "atr": round(atr, 2),
            "buffer": round(buffer, 2),
            "vol_ratio": round(vol_ratio, 2),
            "prev_day_bias": prev_day_bias,
        }

        if orb_range < min_range:
            empty_signal.indicators = indicators_snapshot
            empty_signal.rejection_reason = f"ORB_RANGE_TOO_NARROW ({orb_range:.1f} < {min_range:.1f})"
            return empty_signal

        if orb_range > max_range:
            empty_signal.indicators = indicators_snapshot
            empty_signal.rejection_reason = f"ORB_RANGE_TOO_EXTENDED ({orb_range:.1f} > {max_range:.1f})"
            return empty_signal

        # Bullish Breakout Check
        bullish_break = (close > orb_high + buffer) and (vol_ratio >= self.parameters["volume_multiplier"])
        if bullish_break:
            entry_price = self.round_to_tick(close)
            sl = self.calculate_stop_loss(entry_price, Direction.BUY, atr, data)
            target = self.calculate_target(entry_price, sl, Direction.BUY)
            rr = round(abs(target - entry_price) / max(abs(entry_price - sl), 1e-6), 2)
            bias_bonus = 0.08 if prev_day_bias == "BULL" else 0.0
            conf = min(0.95, round(0.65 + min(vol_ratio * 0.1, 0.25) + bias_bonus, 2))
            reason_str = f"Bullish ORB Breakout above {orb_high:.1f} + {buffer:.1f} with {vol_ratio:.1f}x volume"
            if prev_day_bias == "BULL":
                reason_str += " | Confirmed by Prior Day Upper 30% Open Location Bias"

            return StrategySignal(
                timestamp=now_ts,
                instrument=instrument_name,
                contract=contract_sym,
                strategy=self.name,
                strategy_version=self.version,
                direction=Direction.BUY,
                confidence=conf,
                entry_price=entry_price,
                stop_loss=sl,
                target=target,
                risk_reward=rr,
                regime=regime_details.regime if regime_details else MarketRegime.BREAKOUT,
                reason=reason_str,
                indicators=indicators_snapshot,
                decision="TRADE",
                is_valid=True,
            )

        # Bearish Breakdown Check
        bearish_break = (close < orb_low - buffer) and (vol_ratio >= self.parameters["volume_multiplier"])
        if bearish_break:
            entry_price = self.round_to_tick(close)
            sl = self.calculate_stop_loss(entry_price, Direction.SELL, atr, data)
            target = self.calculate_target(entry_price, sl, Direction.SELL)
            rr = round(abs(target - entry_price) / max(abs(entry_price - sl), 1e-6), 2)
            bias_bonus = 0.08 if prev_day_bias == "BEAR" else 0.0
            conf = min(0.95, round(0.65 + min(vol_ratio * 0.1, 0.25) + bias_bonus, 2))
            reason_str = f"Bearish ORB Breakdown below {orb_low:.1f} - {buffer:.1f} with {vol_ratio:.1f}x volume"
            if prev_day_bias == "BEAR":
                reason_str += " | Confirmed by Prior Day Lower 30% Open Location Bias"

            return StrategySignal(
                timestamp=now_ts,
                instrument=instrument_name,
                contract=contract_sym,
                strategy=self.name,
                strategy_version=self.version,
                direction=Direction.SELL,
                confidence=conf,
                entry_price=entry_price,
                stop_loss=sl,
                target=target,
                risk_reward=rr,
                regime=regime_details.regime if regime_details else MarketRegime.BREAKOUT,
                reason=reason_str,
                indicators=indicators_snapshot,
                decision="TRADE",
                is_valid=True,
            )


        empty_signal.indicators = indicators_snapshot
        empty_signal.rejection_reason = "NO_ORB_BREAKOUT"
        return empty_signal

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: Direction,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        session_open = self.instrument_config.session.open_time if self.instrument_config else "09:00"
        orb_high, orb_low, _ = self._determine_opening_range(df, session_open)

        if orb_high and orb_low:
            orb_mid = (orb_high + orb_low) / 2.0
            if direction.is_long:
                sl = max(orb_mid, entry_price - (atr * 1.5))
                return self.round_to_tick(min(sl, entry_price - (self.instrument_config.tick_size * 2 if self.instrument_config else 2.0)))
            else:
                sl = min(orb_mid, entry_price + (atr * 1.5))
                return self.round_to_tick(max(sl, entry_price + (self.instrument_config.tick_size * 2 if self.instrument_config else 2.0)))

        sl_dist = atr * 1.5
        return self.round_to_tick(entry_price - sl_dist if direction.is_long else entry_price + sl_dist)

    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        direction: Direction,
        rr_ratio: Optional[float] = None,
    ) -> float:
        rr = rr_ratio or self.parameters["risk_reward_ratio"]
        risk_dist = abs(entry_price - stop_loss)
        target_dist = risk_dist * rr

        if direction.is_long:
            return self.round_to_tick(entry_price + target_dist)
        else:
            return self.round_to_tick(entry_price - target_dist)

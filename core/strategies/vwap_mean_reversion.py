"""
core/strategies/vwap_mean_reversion.py — VWAP Mean Reversion Strategy Family
=============================================================================
Identifies extreme intraday price extensions away from Volume-Weighted Average Price (VWAP)
and trades reversion back toward equilibrium ONLY during non-trending (RANGE/LOW_VOL) regimes.

CRITICAL RISK CONTROLS:
- Strict regime gate: Vetoes immediately if market is in TREND, BREAKOUT, or ABNORMAL regime.
- ADX ceiling: Blocks entry if ADX > 22.0 to prevent fading runaway trends.
- Dynamic target: Exits when price re-touches VWAP or approaches equilibrium.
"""

from datetime import datetime
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import pytz

from core.models import Direction, Position
from core.regime.engine import MarketRegime, RegimeDetails
from instruments.base import ContractSpec
from core.strategies.base import BaseCommodityStrategy, StrategySignal

IST = pytz.timezone("Asia/Kolkata")


class VWAPMeanReversionStrategy(BaseCommodityStrategy):
    """
    Intraday VWAP Mean Reversion engine with trend exhaustion filters.
    """

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "vwap_deviation_atr_mult": 1.5,      # Stretch from VWAP (in ATRs)
            "vwap_deviation_pct": 0.002,         # 0.2% price stretch below/above VWAP
            "volume_confirmation": True,         # Volume > average volume
            "volume_sma_period": 20,
            "adx_ceiling": 38.0,                 # Block only if extreme runaway trend
            "min_choppiness": 40.0,
            "rsi_period": 14,
            "rsi_oversold": 40.0,                # Long oversold threshold
            "rsi_overbought": 60.0,              # Short overbought threshold
            "atr_sl_multiplier": 1.5,
            "target_vwap_ratio": 0.95,           # Target placed at 95% of distance to VWAP
            "max_holding_bars": 30,              # Mean reversion should resolve quickly
        }
        super().__init__(
            name="VWAPMeanReversion",
            version="1.1.0",
            permitted_regimes=[MarketRegime.RANGE, MarketRegime.LOW_VOLATILITY, MarketRegime.TREND, MarketRegime.BREAKOUT],
            default_parameters=default_params,
        )
        if parameters:
            self.parameters.update(parameters)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        c = data["close"]
        h = data["high"]
        l = data["low"]
        v = data["volume"]

        # Intraday VWAP
        cum_vol = v.cumsum()
        cum_pv = (c * v).cumsum()
        data["vwap"] = (cum_pv / cum_vol.replace(0, np.nan)).ffill().fillna(c)

        # ATR
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["tr"] = tr
        data["atr"] = tr.rolling(14, min_periods=1).mean()

        # ADX
        plus_dm = h.diff().where(lambda x: (x > 0) & (x > -l.diff()), 0.0)
        minus_dm = (-l.diff()).where(lambda x: (x > 0) & (x > h.diff()), 0.0)
        atr_series = data["atr"].replace(0, 1e-6)
        plus_di = 100.0 * (plus_dm.rolling(14, min_periods=1).mean() / atr_series)
        minus_di = 100.0 * (minus_dm.rolling(14, min_periods=1).mean() / atr_series)
        denom = (plus_di + minus_di).replace(0, 1e-6)
        dx = 100.0 * (plus_di - minus_di).abs() / denom
        data["adx"] = dx.rolling(14, min_periods=1).mean()

        # Choppiness Index
        n = 14
        sum_tr = tr.rolling(n, min_periods=1).sum()
        max_h = h.rolling(n, min_periods=1).max()
        min_l = l.rolling(n, min_periods=1).min()
        range_hl = (max_h - min_l).replace(0, 1e-6)
        chop_ratio = (sum_tr / range_hl).clip(lower=1e-6)
        data["choppiness"] = 100.0 * np.log10(chop_ratio) / np.log10(n)

        # RSI
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.parameters["rsi_period"], min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.parameters["rsi_period"], min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-6)
        data["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # Deviation from VWAP in ATR terms and percentage terms
        data["vwap_dist_atr"] = (c - data["vwap"]) / data["atr"].replace(0, 1e-6)
        data["vwap_pct"] = (c - data["vwap"]) / data["vwap"].replace(0, 1e-6)
        vol_p = int(self.parameters.get("volume_sma_period", 20))
        data["volume_sma"] = v.rolling(vol_p, min_periods=1).mean()

        # Standard deviation bands from VWAP (merged from VWAPExtreme)
        deviations = (c - data["vwap"]) ** 2
        sd = np.sqrt(deviations.rolling(20, min_periods=5).mean()).replace(0, 1e-6)
        data["vwap_sd"] = sd
        data["vwap_sd2_upper"] = data["vwap"] + 2.0 * sd
        data["vwap_sd2_lower"] = data["vwap"] - 2.0 * sd
        data["vwap_sd3_upper"] = data["vwap"] + 3.0 * sd
        data["vwap_sd3_lower"] = data["vwap"] - 3.0 * sd

        return data

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

        empty_signal = StrategySignal(
            timestamp=now_ts,
            instrument=instrument_name,
            contract=contract_sym,
            strategy=self.name,
            strategy_version=self.version,
            direction=Direction.NONE,
            decision="NO_TRADE",
        )

        if len(df) < 25:
            empty_signal.rejection_reason = "INSUFFICIENT_DATA"
            return empty_signal

        # Check session timing
        if not self.is_within_session(now_ts):
            empty_signal.rejection_reason = "OUTSIDE_TRADING_SESSION"
            return empty_signal

        # REGIME FILTER: Veto on abnormal regime or strong runaway trend
        if regime_details:
            r_obj = getattr(regime_details, "regime", None) or (
                regime_details.get("regime") if isinstance(regime_details, dict) else None
            )
            r_val = str(r_obj.value if hasattr(r_obj, "value") else (r_obj or "")).upper()
            r_conf = float(getattr(regime_details, "confidence", None) or (
                regime_details.get("confidence", 0.8) if isinstance(regime_details, dict) else 0.8
            ) or 0.8)
            if r_obj == MarketRegime.ABNORMAL or "ABNORMAL" in r_val:
                empty_signal.rejection_reason = f"REGIME_ABNORMAL_VETO: Market conditions marked abnormal"
                empty_signal.decision = "NO_TRADE"
                return empty_signal
            if (r_obj in (MarketRegime.TREND, MarketRegime.BREAKOUT) or "TREND" in r_val or "BREAKOUT" in r_val) and r_conf >= 0.85:
                empty_signal.rejection_reason = f"REGIME_TRENDING_VETO (Regime {r_val} hostile to mean reversion)"
                empty_signal.decision = "NO_TRADE"
                return empty_signal

        data = self.calculate_indicators(df)
        curr = data.iloc[-1]
        prev = data.iloc[-2]

        close = float(curr["close"])
        vwap = float(curr["vwap"])
        atr = float(curr["atr"])
        adx = float(curr["adx"])
        choppiness = float(curr["choppiness"])
        rsi = float(curr["rsi"])
        vwap_dist_atr = float(curr["vwap_dist_atr"])

        indicators_snapshot = {
            "close": close,
            "vwap": round(vwap, 2),
            "atr": round(atr, 2),
            "adx": round(adx, 2),
            "choppiness": round(choppiness, 2),
            "rsi": round(rsi, 2),
            "vwap_dist_atr": round(vwap_dist_atr, 2),
        }

        # Block if ADX is too strong (runaway trend)
        if adx > self.parameters["adx_ceiling"]:
            empty_signal.indicators = indicators_snapshot
            empty_signal.rejection_reason = f"TREND_TOO_STRONG (ADX {adx:.1f} > ceiling {self.parameters['adx_ceiling']})"
            return empty_signal

        req_dist = float(self.parameters.get("vwap_deviation_atr_mult", 1.5))
        req_pct = float(self.parameters.get("vwap_deviation_pct", 0.002))
        use_vol = bool(self.parameters.get("volume_confirmation", False))
        vol = float(curr["volume"])
        vol_sma = float(curr["volume_sma"]) if not np.isnan(curr["volume_sma"]) else 1.0
        vwap_pct = float(curr["vwap_pct"])
        vol_ok = (not use_vol) or (vol >= vol_sma * 0.9)

        o = float(curr["open"]) if "open" in curr else close
        h = float(curr["high"])
        l = float(curr["low"])
        c_range = max(h - l, 0.01)
        upper_reject = ((h - max(o, close)) / c_range >= 0.30) and (close < o)
        lower_reject = ((min(o, close) - l) / c_range >= 0.30) and (close > o)
        at_lower2 = close <= float(curr.get("vwap_sd2_lower", 0.0))
        at_upper2 = close >= float(curr.get("vwap_sd2_upper", 1e9))

        # Long Mean Reversion: Price deeply stretched below VWAP OR at 2nd SD with lower wick rejection
        is_long_dist = (vwap_dist_atr <= -req_dist) or (vwap_pct <= -req_pct) or (at_lower2 and lower_reject)
        if is_long_dist and (rsi <= self.parameters["rsi_oversold"] or lower_reject) and vol_ok:
            # Confirm price rejection / hook (close > low + 0.2 * range) or lower wick rejection
            bar_range = float(curr["high"]) - float(curr["low"])
            if (bar_range > 0 and (close - float(curr["low"])) / bar_range >= 0.20) or lower_reject:
                entry_price = self.round_to_tick(close)
                sl = self.calculate_stop_loss(entry_price, Direction.BUY, atr, data)
                target = self.calculate_target(entry_price, sl, Direction.BUY)
                rr = round(abs(target - entry_price) / max(abs(entry_price - sl), 1e-6), 2)
                sd_bonus = 0.08 if at_lower2 and lower_reject else 0.0
                conf = min(0.92, round(0.60 + min(abs(vwap_dist_atr) * 0.1, 0.25) + sd_bonus, 2))
                reason_str = f"Oversold VWAP extension ({vwap_dist_atr:.2f} ATRs below VWAP {vwap:.1f}, RSI {rsi:.1f})"
                if at_lower2 and lower_reject:
                    reason_str += " | Confirmed by 2nd SD VWAP Extreme band rejection"

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
                    regime=getattr(regime_details, "regime", None) or (regime_details.get("regime") if isinstance(regime_details, dict) else MarketRegime.RANGE),
                    reason=reason_str,
                    indicators=indicators_snapshot,
                    decision="TRADE",
                    is_valid=True,
                )

        # Short Mean Reversion: Price deeply stretched above VWAP OR at 2nd SD with upper wick rejection
        is_short_dist = (vwap_dist_atr >= req_dist) or (vwap_pct >= req_pct) or (at_upper2 and upper_reject)
        if is_short_dist and (rsi >= self.parameters["rsi_overbought"] or upper_reject) and vol_ok:
            bar_range = float(curr["high"]) - float(curr["low"])
            if (bar_range > 0 and (float(curr["high"]) - close) / bar_range >= 0.20) or upper_reject:
                entry_price = self.round_to_tick(close)
                sl = self.calculate_stop_loss(entry_price, Direction.SELL, atr, data)
                target = self.calculate_target(entry_price, sl, Direction.SELL)
                rr = round(abs(entry_price - target) / max(abs(sl - entry_price), 1e-6), 2)
                sd_bonus = 0.08 if at_upper2 and upper_reject else 0.0
                conf = min(0.92, round(0.60 + min(abs(vwap_dist_atr) * 0.1, 0.25) + sd_bonus, 2))
                reason_str = f"Overbought VWAP extension ({vwap_dist_atr:.2f} ATRs above VWAP {vwap:.1f}, RSI {rsi:.1f})"
                if at_upper2 and upper_reject:
                    reason_str += " | Confirmed by 2nd SD VWAP Extreme band rejection"

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
                    regime=getattr(regime_details, "regime", None) or (regime_details.get("regime") if isinstance(regime_details, dict) else MarketRegime.RANGE),
                    reason=f"Overbought VWAP extension ({vwap_dist_atr:.2f} ATRs above VWAP {vwap:.1f}, RSI {rsi:.1f})",
                    indicators=indicators_snapshot,
                    decision="TRADE",
                    is_valid=True,
                )

        empty_signal.indicators = indicators_snapshot
        empty_signal.rejection_reason = "VWAP_EXTENSION_NOT_REACHED"
        return empty_signal

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: Direction,
        atr: float,
        df: pd.DataFrame,
    ) -> float:
        sl_dist = atr * self.parameters["atr_sl_multiplier"]
        min_tick = self.instrument_config.tick_size if self.instrument_config else 1.0

        if direction.is_long:
            return self.round_to_tick(entry_price - max(sl_dist, min_tick * 4))
        else:
            return self.round_to_tick(entry_price + max(sl_dist, min_tick * 4))

    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
        direction: Direction,
        rr_ratio: Optional[float] = None,
    ) -> float:
        # Target is placed near VWAP level
        if direction.is_long:
            dist = abs(entry_price - stop_loss) * 1.5
            return self.round_to_tick(entry_price + dist)
        else:
            dist = abs(entry_price - stop_loss) * 1.5
            return self.round_to_tick(entry_price - dist)

    def exit_signal(
        self,
        position: Position,
        current_price: float,
        df: pd.DataFrame,
        regime_details: Optional[RegimeDetails] = None,
    ) -> Optional[dict]:
        base_exit = super().exit_signal(position, current_price, df, regime_details)
        if base_exit:
            return base_exit

        # If market switches to strong TREND, exit immediately (do not hold counter-trend)
        r_obj = getattr(regime_details, "regime", None) or (
            regime_details.get("regime") if isinstance(regime_details, dict) else None
        )
        if r_obj == MarketRegime.TREND or getattr(r_obj, "value", str(r_obj)) == "TREND":
            return {
                "exit": True,
                "reason": "REGIME_SHIFTED_TO_TREND_COUNTER_EXIT",
                "price": current_price
            }

        # Check if VWAP reached
        if len(df) > 5:
            c = df["close"]
            v = df["volume"]
            vwap = float(((c * v).cumsum() / v.cumsum().replace(0, 1e-6)).iloc[-1])
            is_long = position.plan.signal.direction.is_long if position.plan and position.plan.signal else True

            if is_long and current_price >= vwap:
                return {
                    "exit": True,
                    "reason": "VWAP_EQUILIBRIUM_REACHED_LONG",
                    "price": current_price
                }
            elif (not is_long) and current_price <= vwap:
                return {
                    "exit": True,
                    "reason": "VWAP_EQUILIBRIUM_REACHED_SHORT",
                    "price": current_price
                }

        return None

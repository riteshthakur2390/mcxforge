from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SwingPoint:
    kind: str
    price: float
    index: int


class MarketStructureLiquidityEngine:
    def evaluate(self, *, df: pd.DataFrame, cache=None, regime_details: dict | None = None) -> dict:
        regime_details = regime_details or {}
        close = float(df["close"].iloc[-1]) if len(df) else 0.0
        atr = float(
            (
                cache.atr_14.iloc[-1]
                if cache is not None and getattr(cache, "atr_14", None) is not None
                else (df["high"] - df["low"]).tail(14).mean()
            ) or 0.0
        )
        tolerance = max(atr * 0.18, close * 0.0008, 0.15)
        lookback = min(len(df), 80)
        window = df.tail(lookback)

        swings = self._find_swings(window)
        structure_state = self._structure_state(window=window, swings=swings, tolerance=tolerance)
        liquidity_zones = self._liquidity_zones(df=df, window=window, swings=swings, tolerance=tolerance)
        liquidity_event = self._liquidity_event(
            window=window,
            structure_state=structure_state,
            liquidity_zones=liquidity_zones,
            tolerance=tolerance,
        )
        entry_validation = self._entry_validation(
            window=window,
            close=close,
            atr=atr,
            structure_state=structure_state,
            liquidity_zones=liquidity_zones,
            liquidity_event=liquidity_event,
        )
        return {
            "structure_state": structure_state,
            "liquidity_zones": liquidity_zones,
            "liquidity_event": liquidity_event,
            "entry_validation": entry_validation,
        }

    def _find_swings(self, df: pd.DataFrame, span: int = 2) -> list[SwingPoint]:
        highs = df["high"].tolist()
        lows = df["low"].tolist()
        swings: list[SwingPoint] = []
        for idx in range(span, len(df) - span):
            high = float(highs[idx])
            low = float(lows[idx])
            if all(high >= float(highs[idx - j]) for j in range(1, span + 1)) and all(
                high >= float(highs[idx + j]) for j in range(1, span + 1)
            ):
                swings.append(SwingPoint(kind="high", price=high, index=idx))
            if all(low <= float(lows[idx - j]) for j in range(1, span + 1)) and all(
                low <= float(lows[idx + j]) for j in range(1, span + 1)
            ):
                swings.append(SwingPoint(kind="low", price=low, index=idx))
        return swings

    def _structure_state(self, *, window: pd.DataFrame, swings: list[SwingPoint], tolerance: float) -> dict:
        highs = [s for s in swings if s.kind == "high"]
        lows = [s for s in swings if s.kind == "low"]
        last_high_state = "UNKNOWN"
        last_low_state = "UNKNOWN"
        prior_bias = "NEUTRAL"
        if len(highs) >= 2:
            last_high_state = "HH" if highs[-1].price > highs[-2].price + tolerance else "LH"
        if len(lows) >= 2:
            last_low_state = "HL" if lows[-1].price > lows[-2].price + tolerance else "LL"
        if len(highs) >= 3 and len(lows) >= 3:
            prev_high_state = "HH" if highs[-2].price > highs[-3].price + tolerance else "LH"
            prev_low_state = "HL" if lows[-2].price > lows[-3].price + tolerance else "LL"
            if prev_high_state == "HH" and prev_low_state == "HL":
                prior_bias = "BULLISH"
            elif prev_high_state == "LH" and prev_low_state == "LL":
                prior_bias = "BEARISH"

        if last_high_state == "HH" and last_low_state == "HL":
            bias = "BULLISH"
        elif last_high_state == "LH" and last_low_state == "LL":
            bias = "BEARISH"
        else:
            bias = "RANGING"

        close = float(window["close"].iloc[-1])
        bos = "NONE"
        bos_direction = "NONE"
        choch = "NONE"
        choch_direction = "NONE"
        if highs and close > highs[-1].price + tolerance * 0.35:
            bos = "BULLISH_BOS"
            bos_direction = "BUY_CALL"
        elif lows and close < lows[-1].price - tolerance * 0.35:
            bos = "BEARISH_BOS"
            bos_direction = "BUY_PUT"
        if prior_bias == "BEARISH" and bos_direction == "BUY_CALL":
            choch = "BULLISH_CHOCH"
            choch_direction = "BUY_CALL"
        elif prior_bias == "BULLISH" and bos_direction == "BUY_PUT":
            choch = "BEARISH_CHOCH"
            choch_direction = "BUY_PUT"

        return {
            "high_sequence": last_high_state,
            "low_sequence": last_low_state,
            "bias": bias,
            "bos": bos,
            "bos_direction": bos_direction,
            "choch": choch,
            "choch_direction": choch_direction,
        }

    def _liquidity_zones(
        self,
        *,
        df: pd.DataFrame,
        window: pd.DataFrame,
        swings: list[SwingPoint],
        tolerance: float,
    ) -> list[dict]:
        zones: list[dict] = []
        highs = [s for s in swings if s.kind == "high"]
        lows = [s for s in swings if s.kind == "low"]
        if len(highs) >= 2 and abs(highs[-1].price - highs[-2].price) <= tolerance:
            zones.append(
                {
                    "kind": "equal_highs",
                    "price": round((highs[-1].price + highs[-2].price) / 2, 2),
                    "strength": 2,
                }
            )
        if len(lows) >= 2 and abs(lows[-1].price - lows[-2].price) <= tolerance:
            zones.append(
                {
                    "kind": "equal_lows",
                    "price": round((lows[-1].price + lows[-2].price) / 2, 2),
                    "strength": 2,
                }
            )

        session_dates = sorted({idx.date() for idx in df.index})
        if len(session_dates) >= 2:
            prev_day = session_dates[-2]
            prev_df = df[[idx.date() == prev_day for idx in df.index]]
            if not prev_df.empty:
                zones.append(
                    {
                        "kind": "prev_day_high",
                        "price": round(float(prev_df["high"].max()), 2),
                        "strength": 3,
                    }
                )
                zones.append(
                    {
                        "kind": "prev_day_low",
                        "price": round(float(prev_df["low"].min()), 2),
                        "strength": 3,
                    }
                )

        prices = sorted([float(s.price) for s in swings[-8:]])
        clusters: list[list[float]] = []
        for price in prices:
            if not clusters or abs(price - clusters[-1][-1]) > tolerance:
                clusters.append([price])
            else:
                clusters[-1].append(price)
        for cluster in clusters:
            if len(cluster) >= 2:
                zones.append(
                    {
                        "kind": "sr_cluster",
                        "price": round(sum(cluster) / len(cluster), 2),
                        "strength": len(cluster),
                    }
                )
        return zones[:8]

    def _liquidity_event(
        self,
        *,
        window: pd.DataFrame,
        structure_state: dict,
        liquidity_zones: list[dict],
        tolerance: float,
    ) -> dict:
        if len(window) < 2:
            return {"type": "NONE", "direction": "NONE", "level": 0.0}
        candle = window.iloc[-1]
        prior = window.iloc[-2]
        range_high = float(window["high"].iloc[-10:-1].max()) if len(window) >= 10 else float(window["high"].iloc[:-1].max())
        range_low = float(window["low"].iloc[-10:-1].min()) if len(window) >= 10 else float(window["low"].iloc[:-1].min())
        zone_prices = [float(z["price"]) for z in liquidity_zones]
        upper_level = max(zone_prices + [range_high]) if zone_prices else range_high
        lower_level = min(zone_prices + [range_low]) if zone_prices else range_low

        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        if high > upper_level + tolerance and close < upper_level:
            return {
                "type": "liquidity_sweep_high",
                "direction": "BUY_PUT",
                "level": round(upper_level, 2),
                "fake_breakout": close < range_high and float(prior["close"]) <= range_high + tolerance,
            }
        if low < lower_level - tolerance and close > lower_level:
            return {
                "type": "liquidity_sweep_low",
                "direction": "BUY_CALL",
                "level": round(lower_level, 2),
                "fake_breakout": close > range_low and float(prior["close"]) >= range_low - tolerance,
            }
        if structure_state.get("bos_direction") != "NONE":
            return {
                "type": "structure_break",
                "direction": structure_state.get("bos_direction", "NONE"),
                "level": round(close, 2),
                "fake_breakout": False,
            }
        return {"type": "NONE", "direction": "NONE", "level": 0.0, "fake_breakout": False}

    def _entry_validation(
        self,
        *,
        window: pd.DataFrame,
        close: float,
        atr: float,
        structure_state: dict,
        liquidity_zones: list[dict],
        liquidity_event: dict,
    ) -> dict:
        recent = window.tail(min(len(window), 20))
        range_high = float(recent["high"].max())
        range_low = float(recent["low"].min())
        width = max(range_high - range_low, 0.01)
        range_pos = (close - range_low) / width
        middle_range = 0.38 <= range_pos <= 0.62

        upper_zone_distance = min(
            [float(z["price"]) - close for z in liquidity_zones if float(z["price"]) > close],
            default=999999.0,
        )
        lower_zone_distance = min(
            [close - float(z["price"]) for z in liquidity_zones if float(z["price"]) < close],
            default=999999.0,
        )
        zone_buffer = max(atr * 0.35, close * 0.0015, 0.3)

        structure_call = structure_state.get("bos_direction") == "BUY_CALL" or structure_state.get("choch_direction") == "BUY_CALL"
        structure_put = structure_state.get("bos_direction") == "BUY_PUT" or structure_state.get("choch_direction") == "BUY_PUT"
        sweep_call = liquidity_event.get("direction") == "BUY_CALL" and liquidity_event.get("type") != "NONE"
        sweep_put = liquidity_event.get("direction") == "BUY_PUT" and liquidity_event.get("type") != "NONE"

        call_reasons = []
        put_reasons = []
        if not (sweep_call or structure_call):
            call_reasons.append("need liquidity sweep low or bullish structure shift")
        if not (sweep_put or structure_put):
            put_reasons.append("need liquidity sweep high or bearish structure shift")
        if middle_range and liquidity_event.get("type") == "NONE":
            call_reasons.append("mid-range entry")
            put_reasons.append("mid-range entry")
        if upper_zone_distance <= zone_buffer:
            call_reasons.append("long into nearby liquidity")
        if lower_zone_distance <= zone_buffer:
            put_reasons.append("short into nearby liquidity")

        by_direction = {
            "BUY_CALL": {"valid": len(call_reasons) == 0, "reasons": call_reasons},
            "BUY_PUT": {"valid": len(put_reasons) == 0, "reasons": put_reasons},
        }
        valid = by_direction["BUY_CALL"]["valid"] or by_direction["BUY_PUT"]["valid"]
        reasons = []
        if not valid:
            reasons.extend(call_reasons[:1] or put_reasons[:1])
        return {
            "valid": valid,
            "avoid": not valid,
            "middle_range": middle_range,
            "range_position": round(range_pos, 4),
            "nearest_upper_liquidity": round(upper_zone_distance, 2) if upper_zone_distance < 999999 else None,
            "nearest_lower_liquidity": round(lower_zone_distance, 2) if lower_zone_distance < 999999 else None,
            "reasons": reasons,
            "by_direction": by_direction,
        }

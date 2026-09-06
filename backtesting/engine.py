"""
backtesting/engine.py — Reusable historical backtest engine
"""
from __future__ import annotations

import asyncio, warnings
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
import sys
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
import pytz
from loguru import logger

import config.settings as settings
from config.settings import (
    DATA_CACHE_DIR, BACKTEST_TIMEFRAME, LOGS_DIR, CANDLE_LOOKBACK,
    BACKTEST_DATA_SOURCE_POLICY,
    RECENT_MARKET_WINDOW_DAYS, HISTORICAL_ARCHIVE_INTERVALS,
    ML_THRESHOLD_OVERRIDE, ML_SECONDARY_THRESHOLD,
    ML_SECONDARY_MIN_RAW_CONFIDENCE, SIGNAL_APPROVAL_DAILY_BUDGET,
    MAX_TRADES_PER_DAY, TIME_STOP_MINUTES, TIME_STOP_MIN_PNL_PCT,
    BACKTEST_BROKERAGE_PER_ORDER, BACKTEST_OPTION_SLIPPAGE_PCT,
    BACKTEST_TRANSACTION_COST_PCT,
    NIFTY_LOT_SIZE, ML_MODELS_DIR,
)
from core.bus import get_bus, reset_bus, reset_bus_sync, Topic
from broker.factory import get_active_broker_name, get_broker
from broker.base_broker import OptionContract
from utils.cache_manager import (
    ensure_backtest_cache,
    interval_capability_days,
    recent_source_window_days,
)
from data.historical_store import HistoricalCandleStore
from data.option_volume import enrich_underlying_with_option_volume
from utils.option_utils import build_option_symbol

IST = pytz.timezone("Asia/Kolkata")
INTRADAY_INTERVALS = ("1minute", "3minute", "5minute", "15minute", "30minute")


def _explicit_backtest_historical_option_premium() -> bool:
    return os.getenv(
        "BACKTEST_ENABLE_HISTORICAL_OPTION_PREMIUM",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _require_backtest_historical_option_premium() -> bool:
    return os.getenv(
        "BACKTEST_REQUIRE_HISTORICAL_OPTION_PREMIUM",
        "false",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _candle_hash(candles: list[dict[str, Any]]) -> str:
    compact: list[dict[str, Any]] = []
    for row in candles:
        compact.append({
            "timestamp": row.get("timestamp") or row.get("datetime") or row.get("date"),
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "volume": row.get("volume"),
        })
    return hashlib.sha256(
        json.dumps(compact, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


class BacktestMarketDataAdapter:
    """Historical market-data adapter exposing the live data-agent methods."""

    def __init__(
        self,
        *,
        underlying_df: pd.DataFrame,
        interval: str,
        broker: str,
        cache_dir: str,
        symbol: str = "NIFTY",
        use_option_premium: bool = False,
    ) -> None:
        self.symbol = symbol
        self.interval = interval
        self.broker_name = broker
        self.use_option_premium = bool(use_option_premium)
        self._underlying_df = underlying_df.copy().sort_index()
        self._current_ts: pd.Timestamp | None = None
        option_start = self._underlying_df.index.min() if not self._underlying_df.empty else None
        option_end = (
            self._underlying_df.index.max() + pd.Timedelta(days=1)
            if not self._underlying_df.empty
            else None
        )
        self._option_df = self._load_option_cache(
            cache_dir,
            symbol,
            interval,
            broker,
            start=str(option_start) if option_start is not None else None,
            end=str(option_end) if option_end is not None else None,
        )
        self._oi_iv_df = self._load_oi_history()
        option_start = option_end = "n/a"
        expiries = strikes = 0
        if not self._option_df.empty and "_ts" in self._option_df.columns:
            option_start = str(self._option_df["_ts"].min())
            option_end = str(self._option_df["_ts"].max())
            expiries = int(self._option_df.get("expiry", pd.Series(dtype=object)).nunique())
            strikes = int(self._option_df.get("strike", pd.Series(dtype=object)).nunique())
        logger.info(
            f"[backtest] MarketDataAdapter initialized | "
            f"option_rows={len(self._option_df)} | oi_iv_rows={len(self._oi_iv_df)} | "
            f"use_option_premium={self.use_option_premium} | "
            f"require_option_premium={_require_backtest_historical_option_premium()} | "
            f"option_range={option_start}..{option_end} | expiries={expiries} | strikes={strikes}"
        )

    def set_replay_time(self, ts: pd.Timestamp) -> None:
        current = pd.Timestamp(ts)
        if current.tzinfo is None:
            current = current.tz_localize(IST)
        else:
            current = current.tz_convert(IST)
        self._current_ts = current

    def get_recent_candles(self, n: int = 60) -> pd.DataFrame | None:
        if self._current_ts is None:
            return self._underlying_df.tail(n)
        return self._underlying_df[self._underlying_df.index <= self._current_ts].tail(n)

    def get_option_ltp(self, option_symbol: str) -> float:
        if not self.use_option_premium:
            return 0.0
        parsed = self._parse_option_symbol(option_symbol)
        if not parsed:
            return 0.0
        row = self._option_row(
            expiry=parsed["expiry"],
            strike=parsed["strike"],
            option_type=parsed["option_type"],
        )
        return float(row.get("close", 0.0) or 0.0) if row is not None else 0.0

    def get_option_contracts(
        self,
        symbol: str,
        expiry,
        option_type: str,
        strikes: list[int],
    ) -> list[OptionContract]:
        if not self.use_option_premium:
            return []
        contracts: list[OptionContract] = []
        for strike in sorted({int(s) for s in strikes if int(s) > 0}):
            row = self._option_row(
                expiry=str(expiry),
                strike=strike,
                option_type=option_type,
            )
            if row is None:
                continue
            expiry_str = str(row.get("expiry") or expiry)
            close = float(row.get("close", 0.0) or 0.0)
            volume = int(float(row.get("volume", 0.0) or 0.0))
            oi = int(float(row.get("oi", row.get("open_interest", 0.0)) or 0.0))
            iv = self._lookup_iv(strike=strike, option_type=option_type)
            spot = float(row.get("underlying_spot", 0.0) or 0.0)
            delta = self._estimate_delta(spot=spot, strike=strike, option_type=option_type, iv=iv)
            contracts.append(
                OptionContract(
                    symbol=build_option_symbol(symbol, pd.Timestamp(expiry_str).date(), strike, option_type),
                    strike=strike,
                    option_type=option_type,
                    expiry_date=expiry_str,
                    last_price=close,
                    bid_price=close,
                    ask_price=close,
                    volume=volume,
                    open_interest=oi,
                    implied_volatility=iv,
                    delta=delta,
                    theta=max(0.8, 4.5 - max((pd.Timestamp(expiry_str).date() - self._current_ts.date()).days, 0) * 0.1),
                    gamma=0.0009 if close > 0 else 0.0,
                    source="HISTORICAL_OPTION_CACHE",
                )
            )
        return contracts

    @staticmethod
    def _load_option_cache(
        cache_dir: str,
        symbol: str,
        interval: str,
        broker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        cache_root = Path(cache_dir)
        frames: list[pd.DataFrame] = []
        strict_broker_options = os.getenv(
            "BACKTEST_STRICT_OPTION_BROKER",
            "1",
        ).strip().lower() in {"1", "true", "yes", "on"}
        use_sqlite_archive = (
            os.getenv("BACKTEST_USE_SQLITE_ARCHIVE", "0").strip() == "1"
            or _explicit_backtest_historical_option_premium()
        )
        if use_sqlite_archive:
            try:
                stored = HistoricalCandleStore().load_option_candles(
                    symbol=symbol,
                    interval=interval,
                    broker=broker if strict_broker_options else None,
                    start=start,
                    end=end,
                )
                if not stored.empty:
                    stored = stored.copy()
                    stored["_ts"] = pd.to_datetime(stored.index)
                    frames.append(stored)
            except Exception as exc:
                logger.debug(f"[backtest] SQLite option cache load skipped: {exc}")

        candidates = [
            cache_root / f"{symbol}_options_{interval}_{broker}.parquet",
            cache_root / f"{symbol}_options_5minute_{broker}.parquet",
        ]
        if not strict_broker_options:
            candidates.extend(
                [
                    cache_root / f"{symbol}_options_{interval}_upstox.parquet",
                    cache_root / f"{symbol}_options_5minute_upstox.parquet",
                ]
            )
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
                if df.empty or "timestamp" not in df.columns:
                    continue
                df = df.copy()
                df["_ts"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
                df = df.dropna(subset=["_ts"])
                if df["_ts"].dt.tz is None:
                    df["_ts"] = df["_ts"].dt.tz_localize(IST)
                else:
                    df["_ts"] = df["_ts"].dt.tz_convert(IST)
                frames.append(df)
            except Exception as exc:
                logger.debug(f"[backtest] Option cache load skipped for {path.name}: {exc}")
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True, sort=False)
        merged = merged.dropna(subset=["_ts"]).sort_values("_ts")
        dedupe_cols = [
            col for col in ("_ts", "strike", "expiry", "option_type", "classification")
            if col in merged.columns
        ]
        if dedupe_cols:
            merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")
        if not merged.empty:
            if "close" in merged.columns:
                # Filter out corrupted rows with unrealistic premiums (e.g. > 1200)
                merged = merged[merged["close"] < 1200]
            merged["strike"] = pd.to_numeric(merged["strike"], errors="coerce").fillna(0).astype(int)
            merged["option_type"] = merged["option_type"].astype(str).str.upper()
            if "expiry" in merged.columns:
                merged["expiry"] = merged["expiry"].astype(str)
        return merged

    @staticmethod
    def _load_oi_history() -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for path in sorted(Path("data/oi_history").glob("oi_*.parquet")):
            try:
                df = pd.read_parquet(path)
                if df.empty or "timestamp" not in df.columns:
                    continue
                df = df.copy()
                df["_ts"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
                df = df.dropna(subset=["_ts"])
                if df["_ts"].dt.tz is None:
                    df["_ts"] = df["_ts"].dt.tz_localize(IST)
                else:
                    df["_ts"] = df["_ts"].dt.tz_convert(IST)
                frames.append(df)
            except Exception as exc:
                logger.debug(f"[backtest] OI history load skipped for {path.name}: {exc}")
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True).sort_values("_ts")
        return merged

    @staticmethod
    def _parse_option_symbol(option_symbol: str) -> dict[str, Any] | None:
        import re

        match = re.match(r"^([A-Z]+)(\d{2}[A-Z]{3}\d{2})(\d+)(CE|PE)$", str(option_symbol or "").upper())
        if not match:
            return None
        symbol, expiry_part, strike, option_type = match.groups()
        try:
            expiry = datetime.strptime(expiry_part, "%y%b%d").date().isoformat()
        except Exception:
            return None
        return {
            "symbol": symbol,
            "expiry": expiry,
            "strike": int(strike),
            "option_type": option_type,
        }

    def _option_row(self, *, expiry: str, strike: int, option_type: str) -> dict[str, Any] | None:
        if self._option_df.empty or self._current_ts is None:
            return None
        expiry_s = str(expiry)
        option_type_s = str(option_type).upper()
        strike_val = int(strike)
        
        # Slicing by strike and option_type first is highly efficient
        sub_df = self._option_df[
            (self._option_df["strike"] == strike_val) & 
            (self._option_df["option_type"] == option_type_s)
        ]
        if sub_df.empty:
            return None
            
        mask = sub_df["_ts"] <= self._current_ts
        if "expiry" in sub_df.columns:
            rows = sub_df[mask & (sub_df["expiry"] == expiry_s)]
            if rows.empty:
                rolling_expiry = self._rolling_expiry_bucket(expiry_s)
                if rolling_expiry:
                    rows = sub_df[mask & (sub_df["expiry"] == rolling_expiry)]
        else:
            rows = sub_df[mask]
            
        if rows.empty:
            return None
        last_row = rows.iloc[-1]
        return last_row.to_dict()

    def _rolling_expiry_bucket(self, expiry: str) -> str:
        try:
            expiry_date = pd.Timestamp(expiry).date()
        except Exception:
            return ""
        if self._current_ts is None:
            return ""
        dte = (expiry_date - self._current_ts.date()).days
        if dte < 0:
            return ""
        week_rank = max(1, min(4, math.ceil((dte + 1) / 7)))
        return f"WEEK:{week_rank}"

    def _lookup_iv(self, *, strike: int, option_type: str) -> float:
        if self._oi_iv_df.empty or self._current_ts is None:
            return 0.14
        rows = self._oi_iv_df[self._oi_iv_df["_ts"] <= self._current_ts]
        if rows.empty:
            return 0.14
        row = rows.iloc[-1]
        atm = int(float(row.get("atm_strike", 0) or 0))
        if atm <= 0:
            return 0.14
        offset = int(round((int(strike) - atm) / 50))
        if option_type.upper() == "CE":
            key = {0: "atm_ce_iv", 1: "otm1_ce_iv", 2: "otm2_ce_iv", -1: "itm1_ce_iv", -2: "itm2_ce_iv"}.get(offset)
        else:
            key = {0: "atm_pe_iv", -1: "otm1_pe_iv", -2: "otm2_pe_iv", 1: "itm1_pe_iv", 2: "itm2_pe_iv"}.get(offset)
        if not key:
            return 0.14
        iv = float(row.get(key, 0.0) or 0.0)
        return round(iv / 100.0, 4) if iv > 1.0 else max(iv, 0.01)

    @staticmethod
    def _estimate_delta(*, spot: float, strike: int, option_type: str, iv: float) -> float:
        if spot <= 0 or strike <= 0:
            return 0.5 if option_type.upper() == "CE" else -0.5
        distance = (spot - strike) / max(spot, 1.0)
        scale = max(float(iv or 0.14), 0.08) * 2.5
        raw = 0.5 + max(-0.28, min(0.28, distance / max(scale, 1e-6)))
        raw = max(0.18, min(0.82, raw))
        return round(raw if option_type.upper() == "CE" else raw - 1.0, 4)


def _resample_intraday_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    normalized = df.copy()
    idx = pd.to_datetime(normalized.index)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)
    normalized.index = idx

    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for col in normalized.columns:
        if col not in agg:
            agg[col] = "last"

    out = normalized.resample(
        rule,
        origin="start_day",
        offset="15min",
        label="right",
        closed="right",
    ).agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"]).sort_index()


def _compact_bus_history(messages: list[dict]) -> list[dict]:
    compact: list[dict] = []
    for msg in messages:
        payload = dict(msg.get("payload", {}) or {})
        candles = payload.get("candles")
        if isinstance(candles, list):
            payload["candles_count"] = len(candles)
            payload.pop("candles", None)
        compact.append({
            "topic": msg.get("topic"),
            "source": msg.get("source"),
            "timestamp": msg.get("timestamp"),
            "payload": payload,
        })
    return compact


def _closed_trades_df(journal_rows: list[dict]) -> pd.DataFrame:
    closed_rows = [
        row for row in journal_rows
        if str(row.get("lifecycle_status", "")).upper() == "CLOSED"
        and str(row.get("exit_time", "")).strip()
    ]
    if not closed_rows:
        return pd.DataFrame()

    df = pd.DataFrame(closed_rows).copy()
    df["realized_pnl"] = pd.to_numeric(df.get("realized_pnl"), errors="coerce").fillna(0.0)
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct"), errors="coerce").fillna(0.0)
    df["entry_time"] = pd.to_datetime(df.get("entry_time"), errors="coerce")
    df["exit_time"] = pd.to_datetime(df.get("exit_time"), errors="coerce")
    df = df.dropna(subset=["exit_time"]).sort_values("exit_time").reset_index(drop=True)
    return df


def _compute_backtest_risk_metrics(
    journal_rows: list[dict],
    paper_capital: float,
) -> dict[str, float]:
    closed_df = _closed_trades_df(journal_rows)
    if closed_df.empty or paper_capital <= 0:
        return {
            "number_of_trades": float(len(closed_df)),
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pct": 0.0,
        }

    wins = closed_df[closed_df["realized_pnl"] > 0]["realized_pnl"]
    losses = closed_df[closed_df["realized_pnl"] < 0]["realized_pnl"]
    gross_profit = float(wins.sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
    profit_factor = (
        round(gross_profit / gross_loss, 3)
        if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    )

    daily_pnl = (
        closed_df.assign(exit_date=closed_df["exit_time"].dt.date)
        .groupby("exit_date", sort=True)["realized_pnl"]
        .sum()
    )
    equity_curve = daily_pnl.cumsum() + paper_capital
    rolling_peak = equity_curve.cummax()
    drawdown_pct = ((equity_curve - rolling_peak) / rolling_peak.replace(0, pd.NA)) * 100.0
    max_drawdown_pct = round(abs(float(drawdown_pct.min() or 0.0)), 2)

    daily_returns = daily_pnl / paper_capital
    sharpe_ratio = 0.0
    if len(daily_returns) > 1:
        std = float(daily_returns.std(ddof=1) or 0.0)
        if std > 0:
            sharpe_ratio = round(float((daily_returns.mean() / std) * math.sqrt(252)), 3)

    return {
        "number_of_trades": float(len(closed_df)),
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "avg_trade_pct": round(float(closed_df["pnl_pct"].mean() or 0.0), 2),
    }


def _compute_trade_return_metrics(closed_df: pd.DataFrame) -> dict[str, float]:
    if closed_df.empty:
        return {
            "realized_pnl": 0.0,
            "total_invested": 0.0,
            "invested_return_pct": 0.0,
            "aggregate_trade_pnl_pct": 0.0,
        }

    def numeric_col(name: str) -> pd.Series:
        if name not in closed_df.columns:
            return pd.Series(0.0, index=closed_df.index)
        return pd.to_numeric(closed_df[name], errors="coerce").fillna(0.0)

    realized_pnl = float(numeric_col("realized_pnl").sum())
    invested = numeric_col("total_invested")
    if float(invested.sum()) <= 0:
        premium = numeric_col("actual_premium")
        quantity = numeric_col("quantity")
        invested = premium * quantity
    if float(invested.sum()) <= 0:
        premium = numeric_col("actual_premium")
        lot_size = numeric_col("lot_size").replace(0, NIFTY_LOT_SIZE)
        lots = numeric_col("lots").replace(0, 1)
        invested = premium * lot_size * lots

    total_invested = float(invested.sum())
    invested_return_pct = (
        round(realized_pnl / total_invested * 100.0, 2)
        if total_invested > 0 else 0.0
    )
    aggregate_trade_pnl_pct = round(float(numeric_col("pnl_pct").sum()), 2)

    return {
        "realized_pnl": round(realized_pnl, 2),
        "total_invested": round(total_invested, 2),
        "invested_return_pct": invested_return_pct,
        "aggregate_trade_pnl_pct": aggregate_trade_pnl_pct,
    }


def _summary_signal_id(payload: dict[str, Any], *, fallback_prefix: str) -> str:
    signal = dict(payload.get("signal", {}) or {})
    ts = (
        payload.get("signal_id")
        or signal.get("signal_id")
        or payload.get("timestamp")
        or signal.get("timestamp")
        or ""
    )
    direction = payload.get("direction") or signal.get("direction") or ""
    strategies = payload.get("strategies_fired") or signal.get("strategies_fired") or []
    if isinstance(strategies, str):
        strategy_key = strategies
    else:
        strategy_key = "|".join(str(s).strip() for s in strategies if str(s).strip())
    price = payload.get("nifty_ltp")
    if price is None:
        price = signal.get("nifty_ltp", 0.0)
    try:
        price_key = f"{float(price or 0.0):.2f}"
    except Exception:
        price_key = "0.00"
    parts = [str(ts), str(direction), strategy_key, price_key]
    if any(part for part in parts):
        return "|".join(parts)
    return f"{fallback_prefix}|{hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]}"


def _classify_gap_bias(gap_pct: float) -> str:
    if gap_pct > 0.15:
        return "BULLISH"
    if gap_pct < -0.15:
        return "BEARISH"
    return "NEUTRAL"


def _build_backtest_premarket_payload(
    *,
    day: Any,
    day_df: pd.DataFrame,
    replay_df: pd.DataFrame,
) -> dict[str, Any]:
    today_open = 0.0
    if not day_df.empty:
        today_open = float(day_df.iloc[0]["open"] or 0.0)

    prev_close = 0.0
    prior_days = sorted({idx.date() for idx in replay_df.index if idx.date() < day})
    if prior_days:
        prev_day = prior_days[-1]
        prev_df = replay_df[[idx.date() == prev_day for idx in replay_df.index]]
        if not prev_df.empty:
            prev_close = float(prev_df.iloc[-1]["close"] or 0.0)

    gap_pct = ((today_open - prev_close) / prev_close * 100.0) if prev_close and today_open else 0.0
    return {
        "bias": _classify_gap_bias(gap_pct),
        "india_vix": 0.0,
        "gap_pct": round(gap_pct, 3),
        "prev_close": prev_close,
        "nifty_prev_close": prev_close,
        "today_open": today_open,
        "gap_ready": bool(prev_close and today_open),
        "timestamp": f"{day}T09:00:00+05:30",
    }


def _build_legacy_premarket_payload(*, day: Any) -> dict[str, Any]:
    return {
        "bias": "NEUTRAL",
        "india_vix": 14.0,
        "gap_pct": 0.0,
        "timestamp": f"{day}T09:00:00+05:30",
    }


def _replay_record_from_row(idx: pd.Timestamp, row: pd.Series) -> dict[str, Any]:
    record: dict[str, Any] = {
        "datetime": str(idx),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": int(row.get("volume", 0) or 0),
    }
    for col, value in row.items():
        if col in {"open", "high", "low", "close", "volume"}:
            continue
        if pd.isna(value):
            continue
        if isinstance(value, (int, float)):
            record[str(col)] = float(value)
            continue
        try:
            record[str(col)] = float(value)
        except Exception:
            record[str(col)] = value
    return record


class BacktestExecutionAgent:
    NAME = "BacktestExecutionAgent"

    def __init__(self) -> None:
        self.bus = get_bus()

    def register(self) -> None:
        self.bus.subscribe(Topic.TRADE_PLAN_READY, self.on_trade_plan)

    async def on_trade_plan(self, msg) -> None:
        plan = dict(msg.payload)
        ts = plan.get("timestamp")
        await self.bus.publish(Topic.ORDER_DRY_RUN, {
            **plan,
            "mode": "BACKTEST",
            "order_id": f"BT_{plan.get('option_symbol', 'NA')}",
            "entry_premium": plan.get("est_premium", 0),
            "simulated": True,
            "execution": "BACKTEST",
            "timestamp": ts,
        }, self.NAME)


def _load_primary_cache(
    *,
    force_sqlite_archive: bool = False,
    enrich_option_volume: bool = True,
) -> pd.DataFrame:
    active_broker = get_active_broker_name()
    requested_interval = BACKTEST_TIMEFRAME
    cache_root = Path(DATA_CACHE_DIR)
    use_sqlite_archive = (
        force_sqlite_archive
        or os.getenv("BACKTEST_USE_SQLITE_ARCHIVE", "0").strip() == "1"
    )
    historical_store = None
    if use_sqlite_archive:
        try:
            historical_store = HistoricalCandleStore()
        except Exception as exc:
            logger.warning(f"[backtest] Local historical store unavailable: {exc}")
    interval_candidates: list[str] = [requested_interval]

    ranked_candidates: list[tuple[int, int, int, Path, str, pd.DataFrame]] = []
    interval_rank = {
        "1minute": 0,
        "3minute": 1,
        "5minute": 2,
        "15minute": 3,
        "30minute": 4,
        "60minute": 5,
        "day": 5,
    }

    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        normalized = df.copy()
        idx = pd.to_datetime(normalized.index)
        if idx.tz is None:
            idx = idx.tz_localize(IST)
        else:
            idx = idx.tz_convert(IST)
        normalized.index = idx
        return normalized.sort_index()

    def _load_interval_candidate(interval: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        frames: list[pd.DataFrame] = []
        sources: list[str] = []
        brokers: list[str] = []

        def _append_source_frame(df: pd.DataFrame, *, source: str, broker: str) -> None:
            normalized = _normalize_df(df)
            if normalized.empty:
                return
            stamped = normalized.copy()
            stamped["__sf_source"] = source
            stamped["__sf_broker"] = broker
            frames.append(stamped)
            sources.append(source)
            brokers.append(broker)

        source_window_days = recent_source_window_days(active_broker, interval)
        archive_cutoff_date = HistoricalCandleStore.archive_cutoff_date(source_window_days)

        local_df = pd.DataFrame()
        load_local_interval = use_sqlite_archive and (
            interval in HISTORICAL_ARCHIVE_INTERVALS
            or (requested_interval == "3minute" and interval == "1minute")
        )
        if historical_store is not None and load_local_interval:
            try:
                active_broker_df = historical_store.load_candles(
                    symbol="NIFTY",
                    interval=interval,
                    broker=active_broker,
                    end=str(archive_cutoff_date),
                )
                all_broker_df = (
                    pd.DataFrame()
                    if BACKTEST_DATA_SOURCE_POLICY == "DHAN_ONLY"
                    else historical_store.load_candles(
                        symbol="NIFTY",
                        interval=interval,
                        end=str(archive_cutoff_date),
                    )
                )
                local_frames = [
                    frame for frame in (all_broker_df, active_broker_df)
                    if frame is not None and not frame.empty
                ]
                if local_frames:
                    local_df = pd.concat(local_frames).sort_index()
                    local_df = local_df[~local_df.index.duplicated(keep="last")]
                else:
                    local_df = historical_store.load_candles(
                        symbol="NIFTY",
                        interval=interval,
                        end=str(archive_cutoff_date),
                    )
            except Exception as exc:
                logger.warning(f"[backtest] Skipping local historical store for {interval}: {exc}")
                local_df = pd.DataFrame()
        if not local_df.empty:
            _append_source_frame(local_df, source="local_db", broker=active_broker)

        cache_files: list[Path] = []
        seen: set[Path] = set()
        preferred = cache_root / f"NIFTY_{interval}_{active_broker}.parquet"
        for match in sorted(
            cache_root.glob(f"NIFTY_{interval}_*.parquet"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        ):
            if match != preferred and match not in seen:
                cache_files.append(match)
                seen.add(match)
        if preferred.exists():
            cache_files.append(preferred)
            seen.add(preferred)

        for cache_file in cache_files:
            source_broker = cache_file.stem.rsplit("_", 1)[-1].lower()
            if BACKTEST_DATA_SOURCE_POLICY == "DHAN_ONLY" and source_broker != "dhan":
                continue
            try:
                cache_df = _normalize_df(pd.read_parquet(cache_file))
                if cache_df.empty:
                    continue
                _append_source_frame(cache_df, source=cache_file.name, broker=source_broker)
            except Exception as exc:
                logger.debug(f"[backtest] Skipping cache candidate {cache_file.name}: {exc}")

        if not frames:
            return pd.DataFrame(), [], []

        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        source_by_day: dict[str, dict[str, int]] = {}
        if "__sf_source" in merged.columns:
            for day, day_df in merged.groupby(merged.index.date):
                counts = day_df["__sf_source"].astype(str).value_counts().to_dict()
                source_by_day[str(day)] = {str(k): int(v) for k, v in counts.items()}
        if "__sf_source" in merged.columns:
            merged = merged.drop(columns=["__sf_source", "__sf_broker"], errors="ignore")
        merged.attrs["source_by_day"] = source_by_day
        return merged, sources, sorted({b for b in brokers if b})

    if requested_interval == "3minute":
        one_minute_df, one_minute_sources, one_minute_brokers = _load_interval_candidate("1minute")
        if not one_minute_df.empty:
            generated = _resample_intraday_ohlcv(one_minute_df, "3min")
            if not generated.empty:
                idx = pd.to_datetime(generated.index)
                span_days = (idx.max().date() - idx.min().date()).days + 1
                generated.attrs["source_list"] = [
                    f"resampled_3minute_from:{source}" for source in one_minute_sources
                ]
                generated.attrs["source_brokers"] = one_minute_brokers
                ranked_candidates.append((
                    1,
                    span_days,
                    -interval_rank.get("3minute", 99),
                    cache_root / "NIFTY_3minute_resampled_1minute.parquet",
                    "3minute",
                    generated,
                ))

    for candidate_interval in interval_candidates:
        candidate_df, candidate_sources, candidate_brokers = _load_interval_candidate(candidate_interval)
        if candidate_df.empty:
            continue
        idx = pd.to_datetime(candidate_df.index)
        span_days = (idx.max().date() - idx.min().date()).days + 1
        is_requested = int(candidate_interval == requested_interval)
        pseudo_path = cache_root / f"NIFTY_{candidate_interval}_merged.parquet"
        candidate_df.attrs["source_list"] = candidate_sources
        candidate_df.attrs["source_brokers"] = candidate_brokers
        candidate_df.attrs["source_by_day"] = dict(candidate_df.attrs.get("source_by_day", {}))
        ranked_candidates.append((
            is_requested,
            span_days,
            -interval_rank.get(candidate_interval, 99),
            pseudo_path,
            candidate_interval,
            candidate_df,
        ))

    if not ranked_candidates:
        store_path = str(getattr(historical_store, "db_path", "local historical store"))
        raise FileNotFoundError(
            f"No readable historical {BACKTEST_TIMEFRAME} data found in {DATA_CACHE_DIR} "
            f"or {store_path}."
        )

    ranked_candidates.sort(reverse=True)
    _, _, _, selected_cache, selected_interval, df = ranked_candidates[0]
    if BACKTEST_DATA_SOURCE_POLICY == "DHAN_ONLY":
        non_dhan_sources = [
            source for source in df.attrs.get("source_list", [])
            if source != "local_db" and not str(source).endswith("_dhan.parquet")
        ]
        if non_dhan_sources:
            raise RuntimeError(
                f"DHAN_ONLY policy rejected non-Dhan sources: {non_dhan_sources}"
            )
    if selected_interval != requested_interval:
        raise RuntimeError(
            f"Backtest requested {requested_interval} data but selected "
            f"{selected_interval}. Refusing silent timeframe switch."
        )

    df.attrs["cache_file"] = selected_cache.name
    df.attrs["cache_interval"] = selected_interval
    df.attrs["requested_interval"] = requested_interval
    df.attrs["source_list"] = list(df.attrs.get("source_list", []))
    df.attrs["source_brokers"] = list(df.attrs.get("source_brokers", []))
    df.attrs["source_by_day"] = dict(df.attrs.get("source_by_day", {}))
    source_days = df.attrs.get("source_by_day", {}) or {}
    source_day_counts: dict[str, int] = {}
    for day_counts in source_days.values():
        if not isinstance(day_counts, dict):
            continue
        dominant = max(day_counts.items(), key=lambda item: item[1])[0] if day_counts else "unknown"
        source_day_counts[dominant] = source_day_counts.get(dominant, 0) + 1
    logger.info(
        f"[backtest] Data provenance | interval={selected_interval} | "
        f"sources={df.attrs['source_list']} | brokers={df.attrs['source_brokers']} | "
        f"dominant_source_days={source_day_counts}"
    )
    if enrich_option_volume:
        df = _enrich_backtest_option_volume(
            df,
            historical_store=historical_store,
            interval=selected_interval,
            broker=active_broker,
        )
    return df


def _enrich_backtest_option_volume(
    df: pd.DataFrame,
    *,
    historical_store: HistoricalCandleStore | None = None,
    interval: str | None = None,
    broker: str | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        selected_interval = str(interval or df.attrs.get("cache_interval") or BACKTEST_TIMEFRAME)
        active_broker = str(broker or get_active_broker_name())
        enriched = enrich_underlying_with_option_volume(
            df,
            store=historical_store or HistoricalCandleStore(),
            cache_dir=DATA_CACHE_DIR,
            symbol="NIFTY",
            interval=selected_interval,
            broker=active_broker,
        )
        if enriched is not df:
            enriched.attrs.update(df.attrs)
            return enriched
    except Exception as exc:
        logger.debug(f"[backtest] Option volume enrichment skipped: {exc}")
    return df


def get_backtest_limits() -> dict[str, Any]:
    ensure_backtest_cache()
    df = _load_primary_cache()
    available_days = sorted({idx.date() for idx in df.index})
    if not available_days:
        raise ValueError("Cached data file is empty")

    active_broker = get_active_broker_name().lower().strip()
    cache_interval = str(df.attrs.get("cache_interval") or BACKTEST_TIMEFRAME)
    source_window_days = recent_source_window_days(active_broker, cache_interval)
    archive_cutoff = HistoricalCandleStore.archive_cutoff_date(source_window_days)
    capability_days = interval_capability_days(active_broker, cache_interval)
    available_span_days = (available_days[-1] - available_days[0]).days + 1
    available_trading_days = len(available_days)
    max_range_days = available_span_days

    archive_days = [day for day in available_days if day < archive_cutoff]
    recent_days = [day for day in available_days if day >= archive_cutoff]

    return {
        "broker": active_broker,
        "cache_interval": cache_interval,
        "cache_file": str(df.attrs.get("cache_file", "")),
        "requested_interval": str(df.attrs.get("requested_interval") or BACKTEST_TIMEFRAME),
        "earliest_date": available_days[0].isoformat(),
        "latest_date": available_days[-1].isoformat(),
        "available_span_days": available_span_days,
        "available_trading_days": available_trading_days,
        "capability_days": capability_days,
        "recent_market_window_days": RECENT_MARKET_WINDOW_DAYS,
        "recent_source_window_days": source_window_days,
        "max_range_days": max_range_days,
        "archive_earliest_date": archive_days[0].isoformat() if archive_days else "",
        "archive_latest_date": archive_days[-1].isoformat() if archive_days else "",
        "recent_earliest_date": recent_days[0].isoformat() if recent_days else "",
        "recent_latest_date": recent_days[-1].isoformat() if recent_days else "",
        "sources": list(df.attrs.get("source_list", [])),
        "source_brokers": list(df.attrs.get("source_brokers", [])),
    }


def _refresh_cache_if_needed(required_date: Union[pd.Timestamp, None]) -> pd.DataFrame:
    df = _load_primary_cache(enrich_option_volume=False)
    if required_date is None or df.empty:
        return df
    if os.getenv("BACKTEST_REFRESH_CACHE", "0").strip() != "1":
        return df

    latest_cached_day = max(idx.date() for idx in df.index)
    if latest_cached_day >= required_date.date():
        return df

    broker = get_broker()
    refresh_supported = hasattr(broker, "download_and_cache")
    if not refresh_supported:
        return df

    refresh_interval = str(df.attrs.get("cache_interval") or BACKTEST_TIMEFRAME)

    try:
        logger.warning(
            f"[backtest] Cache stale for {required_date.date()} "
            f"(latest cached day {latest_cached_day}). Refreshing historical data "
            f"for interval={refresh_interval}."
        )
        broker.download_and_cache(symbol="NIFTY", interval=refresh_interval)
        refreshed = _load_primary_cache(enrich_option_volume=False)
        if not refreshed.empty:
            new_latest_day = max(idx.date() for idx in refreshed.index)
            logger.info(
                f"[backtest] Historical cache refreshed "
                f"({latest_cached_day} -> {new_latest_day})"
            )
            return refreshed
    except Exception as exc:
        logger.warning(f"[backtest] Historical refresh failed: {exc}")
    return df


def _completed_available_days(df: pd.DataFrame, today) -> list:
    available_days = sorted({idx.date() for idx in df.index})
    completed_days = [day for day in available_days if day < today]
    return completed_days or available_days


def load_cached_data(
    *,
    days: Union[int, None] = None,
    trading_days: Union[int, None] = None,
    target_date: Union[str, None] = None,
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None,
    all_available: bool = False,
) -> pd.DataFrame:
    today = datetime.now(IST).date()
    required_date: Union[pd.Timestamp, None] = None
    if target_date:
        required_date = pd.Timestamp(target_date, tz=IST)
    elif end_date:
        required_date = pd.Timestamp(end_date, tz=IST)
    elif start_date:
        required_date = pd.Timestamp(start_date, tz=IST)
    elif days or trading_days:
        required_date = pd.Timestamp(today, tz=IST)

    if all_available:
        df = _load_primary_cache(
            force_sqlite_archive=True,
            enrich_option_volume=False,
        )
    else:
        df = _refresh_cache_if_needed(required_date)

    if trading_days is not None and not all_available:
        requested_trading_days = max(1, int(trading_days))
        available_trading_days = len(_completed_available_days(df, today))
        if available_trading_days < requested_trading_days:
            logger.warning(
                f"[backtest] Primary cache has only {available_trading_days} completed "
                f"trading days; loading SQLite archive for requested "
                f"{requested_trading_days} trading days."
            )
            archive_df = _load_primary_cache(
                force_sqlite_archive=True,
                enrich_option_volume=False,
            )
            archive_available = len(_completed_available_days(archive_df, today))
            if archive_available > available_trading_days:
                df = archive_df
            if archive_available < requested_trading_days:
                raise ValueError(
                    f"Requested {requested_trading_days} trading days, but only "
                    f"{archive_available} completed trading days are available in "
                    f"the configured Dhan cache/archive."
                )

    if target_date or start_date or end_date:
        req_dates = []
        if target_date: req_dates.append(pd.Timestamp(target_date, tz=IST).date())
        if start_date: req_dates.append(pd.Timestamp(start_date, tz=IST).date())
        if end_date: req_dates.append(pd.Timestamp(end_date, tz=IST).date())
        
        df_days = {idx.date() for idx in df.index}
        if any(rd not in df_days for rd in req_dates):
            logger.info("[backtest] Requested date(s) not in short-term cache; loading SQLite archive.")
            df = _load_primary_cache(force_sqlite_archive=True, enrich_option_volume=False)

    available_days = sorted({idx.date() for idx in df.index})
    if not available_days:
        raise ValueError("Cached data file is empty")
    latest_day = available_days[-1]

    if target_date:
        start_ts = pd.Timestamp(target_date, tz=IST)
        requested_day = start_ts.date()
        resolved_day = requested_day
        if requested_day > latest_day:
            logger.warning(
                f"[backtest] Requested date {requested_day} is newer than the latest "
                f"cached day {latest_day}. Using {latest_day} instead."
            )
            start_ts = pd.Timestamp(str(latest_day), tz=IST)
            resolved_day = latest_day

        is_today = requested_day == today
        if is_today:
            end_ts = datetime.now(IST)
        else:
            end_ts = start_ts + pd.Timedelta(days=1)

        selected = df[(df.index >= start_ts) & (df.index < end_ts)]
        if selected.empty:
            raise ValueError(
                f"No data found for {target_date}. "
                f"Latest cached day is {latest_day}"
            )
        warmup = df[df.index < start_ts].tail(CANDLE_LOOKBACK)
        sliced = pd.concat([warmup, selected]).sort_index()
        sliced.attrs["requested_date"] = target_date
        sliced.attrs["resolved_days"] = [str(resolved_day)]
        sliced.attrs["selected_days"] = [resolved_day]
        sliced.attrs["live_parity_date"] = resolved_day if is_today else None
        return _enrich_backtest_option_volume(sliced)

    if start_date:
        start_ts = pd.Timestamp(start_date, tz=IST)
        requested_end = pd.Timestamp(end_date or start_date, tz=IST).date()
        if start_ts.date() > latest_day:
            logger.warning(
                f"[backtest] Requested start date {start_ts.date()} is newer than the "
                f"latest cached day {latest_day}. Using {latest_day} instead."
            )
            start_ts = pd.Timestamp(str(latest_day), tz=IST)
        resolved_end = min(requested_end, latest_day)
        if resolved_end < start_ts.date():
            resolved_end = start_ts.date()
        if resolved_end != requested_end:
            logger.warning(
                f"[backtest] Requested end date {requested_end} is newer than the latest "
                f"cached day {latest_day}. Using {resolved_end} instead."
            )
        if resolved_end == today:
            end_ts = datetime.now(IST)
        else:
            end_ts = pd.Timestamp(str(resolved_end), tz=IST) + pd.Timedelta(days=1)
        selected = df[(df.index >= start_ts) & (df.index < end_ts)]
        if selected.empty:
            raise ValueError(
                f"No data found between {start_date} and {end_date or start_date}. "
                f"Latest cached day is {latest_day}"
            )
        warmup = df[df.index < start_ts].tail(CANDLE_LOOKBACK)
        sliced = pd.concat([warmup, selected]).sort_index()
        sliced.attrs["requested_start_date"] = start_date
        sliced.attrs["requested_start_date"] = start_date
        sliced.attrs["requested_end_date"] = end_date or start_date
        selected_days = sorted({idx.date() for idx in selected.index})
        sliced.attrs["resolved_days"] = [str(day) for day in selected_days]
        sliced.attrs["selected_days"] = selected_days
        sliced.attrs["live_parity_date"] = today if today in selected_days and resolved_end == today else None
        return _enrich_backtest_option_volume(sliced)

    if all_available:
        completed_days = [day for day in available_days if day < today]
        recent_days_list = completed_days or available_days
        first_day = recent_days_list[0]
        first_ts = pd.Timestamp(str(first_day), tz=IST)

        selected = df[[idx.date() in recent_days_list for idx in df.index]]
        if selected.empty:
            raise ValueError("No data found in cache archive")

        warmup = df[df.index < first_ts].tail(CANDLE_LOOKBACK)
        sliced = pd.concat([warmup, selected]).sort_index()

        sliced.attrs["requested_trading_days"] = len(recent_days_list)
        sliced.attrs["resolved_days"] = sorted({str(idx.date()) for idx in selected.index})
        sliced.attrs["selected_days"] = recent_days_list
        sliced.attrs["live_parity_date"] = None
        return _enrich_backtest_option_volume(sliced)

    if trading_days is not None:
        trading_days = max(1, int(trading_days))
        completed_days = [day for day in available_days if day < today]
        day_pool = completed_days or available_days
        recent_days_list = day_pool[-trading_days:]
        first_day = recent_days_list[0]
        first_ts = pd.Timestamp(str(first_day), tz=IST)

        selected = df[[idx.date() in recent_days_list for idx in df.index]]
        if selected.empty:
            raise ValueError(f"No data found for the last {trading_days} trading days")

        warmup = df[df.index < first_ts].tail(CANDLE_LOOKBACK)
        sliced = pd.concat([warmup, selected]).sort_index()

        sliced.attrs["requested_trading_days"] = trading_days
        sliced.attrs["resolved_days"] = sorted({str(idx.date()) for idx in selected.index})
        sliced.attrs["selected_days"] = recent_days_list
        sliced.attrs["live_parity_date"] = None
        return _enrich_backtest_option_volume(sliced)

    if days is None:
        days = 1
    days = max(1, int(days))

    completed_days = [day for day in available_days if day < today]
    day_pool = completed_days or available_days
    end_day = day_pool[-1]
    cutoff_day = end_day - pd.Timedelta(days=days - 1)
    recent_days_list = [day for day in day_pool if day >= cutoff_day]
    if not recent_days_list:
        recent_days_list = [end_day]
    first_day = recent_days_list[0]
    first_ts = pd.Timestamp(str(first_day), tz=IST)

    selected = df[[idx.date() in recent_days_list for idx in df.index]]
    if selected.empty:
        raise ValueError(f"No data found for the last {days} days")

    warmup = df[df.index < first_ts].tail(CANDLE_LOOKBACK)
    sliced = pd.concat([warmup, selected]).sort_index()

    sliced.attrs["requested_days"] = days
    sliced.attrs["resolved_days"] = sorted({str(idx.date()) for idx in selected.index})
    sliced.attrs["selected_days"] = recent_days_list
    sliced.attrs["live_parity_date"] = None
    return _enrich_backtest_option_volume(sliced)


async def run_backtest(
    df: pd.DataFrame,
    speed_ms: int = 0,
    ignore_regime: bool = False,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    live_parity: bool = False,
) -> dict[str, Any]:
    previous_trading_mode = os.environ.get("TRADING_MODE")
    os.environ["TRADING_MODE"] = "BACKTEST"
    previous_adaptive_gates = os.environ.get("BACKTEST_ENABLE_ADAPTIVE_GATES")
    previous_adaptive_edges = os.environ.get("BACKTEST_ENABLE_ADAPTIVE_EDGE_FILTERS")
    force_live_parity = bool(live_parity or _truthy_env("BACKTEST_LIVE_PARITY", "false"))
    if force_live_parity:
        os.environ["BACKTEST_ENABLE_ADAPTIVE_GATES"] = "true"
        os.environ["BACKTEST_ENABLE_ADAPTIVE_EDGE_FILTERS"] = "true"

    # CRITICAL FIX: Use sync version to avoid async issues at startup
    # SYSTEM_RESET notification happens after agents are registered
    reset_bus_sync()

    from utils.signal_conflict_resolver import reset_resolver
    reset_resolver(backtest_mode=True)

    from agents_code.agent2_strategy.runner import StrategyAgent
    from agents_code.agent2_strategy.market_context_gate import get_market_context_gate
    from agents_code.agent3_ml.filter import MLFilterAgent
    from agents_code.agent4_planner.planner import TradePlannerAgent
    from agents_code.agent6_position.manager import PositionManagerAgent
    from agents_code.agent7_analytics.journal import AnalyticsAgent
    from agents_code.agent10_risk.risk_guard import RiskGuardAgent
    from agents_code.agent11_shadow.shadow_agent import ShadowParameterAgent
    from agents_code.agent12_lifecycle.lifecycle_auditor import TradeLifecycleAuditor
    if not ignore_regime:
        from agents_code.agent9_regime.classifier import MarketRegimeAgent

    parity_date = df.attrs.get("live_parity_date")
    selected_days = df.attrs.get("selected_days")
    enable_live_parity = bool(parity_date and selected_days and parity_date in selected_days)

    regime_agent = MarketRegimeAgent() if not ignore_regime else None
    strategy_agent = StrategyAgent()
    strategy_agent.set_backtest_mode(True)
    ml_agent = MLFilterAgent(
        enable_file_watcher=False,
        force_retrain=False, # Re-enable model usage; let staleness logic handle actual resets
        live_parity_date=parity_date if enable_live_parity else None,
    )
    active_broker = get_active_broker_name()
    backtest_data_agent = BacktestMarketDataAdapter(
        underlying_df=df,
        interval=str(df.attrs.get("cache_interval") or BACKTEST_TIMEFRAME),
        broker=active_broker,
        cache_dir=DATA_CACHE_DIR,
        symbol="NIFTY",
        use_option_premium=_explicit_backtest_historical_option_premium(),
    )
    planner_agent = TradePlannerAgent(data_agent=backtest_data_agent, backtest_mode=True)
    execution_agent = BacktestExecutionAgent()
    position_agent = PositionManagerAgent(data_agent=backtest_data_agent)
    analytics_agent = AnalyticsAgent(backtest_mode=True)
    risk_agent = RiskGuardAgent()
    shadow_agent = ShadowParameterAgent(bus=get_bus())
    lifecycle_agent = TradeLifecycleAuditor(bus=get_bus())
    market_context_gate = get_market_context_gate(bus=get_bus())

    if regime_agent:
        regime_agent.register()
    strategy_agent.register()
    market_context_gate.register()
    ml_agent.register()
    planner_agent.register()
    execution_agent.register()
    position_agent.register()
    analytics_agent.register()
    risk_agent.register()
    await shadow_agent.register()
    await lifecycle_agent.register()

    bus = get_bus()
    manifest = _build_backtest_manifest(df=df, ml_agent=ml_agent)
    manifest["live_parity"] = force_live_parity
    manifest["replay_mode"] = "live_parity" if force_live_parity else "historical_research"
    manifest["adaptive_gates_enabled"] = _truthy_env("BACKTEST_ENABLE_ADAPTIVE_GATES")
    manifest["adaptive_edge_filters_enabled"] = _truthy_env("BACKTEST_ENABLE_ADAPTIVE_EDGE_FILTERS")
    logger.info(
        "[BACKTEST_SETTINGS] "
        + json.dumps(
            {
                "settings_hash": manifest.get("settings_hash"),
                "live_parity": manifest.get("live_parity"),
                "replay_mode": manifest.get("replay_mode"),
                "adaptive_gates_enabled": manifest.get("adaptive_gates_enabled"),
                "adaptive_edge_filters_enabled": manifest.get("adaptive_edge_filters_enabled"),
                "settings": manifest.get("settings", {}),
            },
            sort_keys=True,
            default=str,
        )
    )

    summary = {
        "candles": 0,
        "days": [],
        "requested": {},
        "ignore_regime": ignore_regime,
        "signals": 0,
        "raw_signals": 0,
        "suppressed": 0,
        "suppressed_by_regime": {},
        "approved": 0,
        "planned": 0,
        "ordered": 0,
        "closed": 0,
        "rejected": 0,
        "wins": 0,
        "losses": 0,
        "transaction_costs": 0.0,
        "net_realized_pnl": 0.0,
    }
    raw_signal_ids: set[str] = set()
    approved_signal_ids: set[str] = set()
    planned_signal_ids: set[str] = set()
    rejected_signal_ids: set[str] = set()

    async def on_signal(_msg):
        raw_signal_ids.add(_summary_signal_id(dict(_msg.payload or {}), fallback_prefix="raw"))
        summary["raw_signals"] = len(raw_signal_ids)

    async def on_suppressed(_msg):
        summary["suppressed"] += 1
        regime = _msg.payload.get("regime", "UNKNOWN")
        summary["suppressed_by_regime"][regime] = (
            summary["suppressed_by_regime"].get(regime, 0) + 1
        )

    async def on_approved(_msg):
        approved_signal_ids.add(_summary_signal_id(dict(_msg.payload or {}), fallback_prefix="approved"))
        summary["approved"] = len(approved_signal_ids)

    async def on_planned(_msg):
        planned_signal_ids.add(_summary_signal_id(dict(_msg.payload or {}), fallback_prefix="planned"))
        summary["planned"] = len(planned_signal_ids)
        summary["signals"] = len(planned_signal_ids)

    async def on_ordered(_msg):
        summary["ordered"] += 1

    async def on_rejected(_msg):
        rejected_signal_ids.add(_summary_signal_id(dict(_msg.payload or {}), fallback_prefix="rejected"))
        summary["rejected"] = len(rejected_signal_ids)

    async def on_closed(msg):
        data = msg.payload
        summary["closed"] += 1
        summary["transaction_costs"] += float(data.get("transaction_costs", 0.0) or 0.0)
        summary["net_realized_pnl"] += float(data.get("realized_pnl", 0.0) or 0.0)
        pnl_pct = float(data.get("pnl_pct", 0) or 0)
        if pnl_pct > 0:
            summary["wins"] += 1
        else:
            summary["losses"] += 1
        
        reason = data.get("exit_reason", "NA")
        entry = data.get("entry_premium", 0)
        exit_p = data.get("exit_premium", 0)
        logger.info(f"BACKTEST EXIT: {reason} | PnL: {pnl_pct:.1f}% | Entry: {entry} | Exit: {exit_p}")

    bus.subscribe(Topic.RAW_SIGNAL, on_signal)
    bus.subscribe(Topic.SIGNAL_SUPPRESSED, on_suppressed)
    bus.subscribe(Topic.SIGNAL_APPROVED, on_approved)
    bus.subscribe(Topic.TRADE_PLAN_READY, on_planned)
    bus.subscribe(Topic.ORDER_DRY_RUN, on_ordered)
    bus.subscribe(Topic.ORDER_PLACED, on_ordered)
    bus.subscribe(Topic.SIGNAL_REJECTED, on_rejected)
    bus.subscribe(Topic.POSITION_CLOSED, on_closed)

    replay_df = df.copy().sort_index()
    replay_df["_date"] = [idx.date() for idx in replay_df.index]
    replay_core = replay_df.drop(columns=["_date"])
    replay_records = [
        _replay_record_from_row(idx, row)
        for idx, row in replay_core.iterrows()
    ]
    
    ts_to_idx = {ts: idx for idx, ts in enumerate(replay_core.index)}
    
    if selected_days:
        trading_days = sorted(selected_days)
    else:
        trading_days = sorted(replay_df["_date"].unique())
    
    total_days = len(trading_days)
    summary["days"] = [str(day) for day in trading_days]
    
    if df.attrs.get("requested_date"):
        summary["requested"]["date"] = df.attrs["requested_date"]
    if df.attrs.get("requested_start_date"):
        summary["requested"]["start_date"] = df.attrs["requested_start_date"]
        summary["requested"]["end_date"] = df.attrs.get("requested_end_date")
    if df.attrs.get("requested_days"):
        summary["requested"]["days"] = df.attrs["requested_days"]
        summary["requested"]["days_mode"] = "calendar"
    if df.attrs.get("requested_trading_days"):
        summary["requested"]["trading_days"] = df.attrs["requested_trading_days"]
        summary["requested"]["days_mode"] = "trading_sessions"

    logger.info(f"[START] Starting backtest for {total_days} trading days...")

    for idx, day in enumerate(trading_days):
        day_df = replay_df[replay_df["_date"] == day].drop(columns=["_date"])
        if day_df.empty:
            continue

        # Progress update
        progress = ((idx) / total_days) * 100
        logger.info(f"[PROGRESS] Progress: {progress:6.1f}% | Processing {day} ({idx+1}/{total_days})")
        if progress_callback:
            try:
                progress_callback({
                    "progress": progress,
                    "day": str(day),
                    "current": idx + 1,
                    "total": total_days,
                })
            except Exception as exc:
                logger.debug(f"[BACKTEST_TELEGRAM] progress callback failed: {exc}")

        is_live_parity_day = bool(force_live_parity or (enable_live_parity and day == parity_date))
        premarket_payload = (
            _build_backtest_premarket_payload(day=day, day_df=day_df, replay_df=replay_df)
            if is_live_parity_day
            else _build_legacy_premarket_payload(day=day)
        )
        await bus.publish(Topic.PREMARKET_BIAS, premarket_payload, "backtest")

        orb_data = day_df.between_time("09:15", "09:25")
        orb_high = orb_low = None

        loop_start = 1 if is_live_parity_day else 0
        for i in range(loop_start, len(day_df)):
            candle = day_df.iloc[i]
            ts = day_df.index[i]

            if is_live_parity_day:
                prev_candle = day_df.iloc[i - 1]
                prev_global_idx = ts_to_idx[prev_candle.name]
                start_idx = max(0, prev_global_idx - CANDLE_LOOKBACK + 1)
                candles_list = replay_records[start_idx:prev_global_idx + 1]
                ltp = float(candle["open"])

                if orb_high is None and (ts.hour > 9 or (ts.hour == 9 and ts.minute >= 30)):
                    orb_data = day_df.between_time("09:15", "09:25")
                    if len(orb_data) >= 2:
                        orb_high = float(orb_data["high"].max())
                        orb_low = float(orb_data["low"].min())
                        await bus.publish(Topic.ORB_FORMED, {
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "orb_range": round(orb_high - orb_low, 2),
                        }, "backtest")
            else:
                global_idx = ts_to_idx[ts]
                start_idx = max(0, global_idx - CANDLE_LOOKBACK + 1)
                candles_list = replay_records[start_idx:global_idx + 1]
                ltp = float(candle["close"])

                if orb_high is None and (ts.hour > 9 or (ts.hour == 9 and ts.minute >= 30)):
                    if len(orb_data) >= 2:
                        orb_high = float(orb_data["high"].max())
                        orb_low = float(orb_data["low"].min())
                        await bus.publish(Topic.ORB_FORMED, {
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "orb_range": round(orb_high - orb_low, 2),
                        }, "backtest")

            summary["candles"] += 1
            backtest_data_agent.set_replay_time(ts)

            await bus.publish(Topic.CANDLES_READY, {
                "symbol": "NIFTY",
                "timeframe": BACKTEST_TIMEFRAME,
                "mode": "BACKTEST",
                "candles": candles_list,
                "ltp": ltp,
                "orb_high": orb_high,
                "orb_low": orb_low,
                "timestamp": ts.isoformat(),
            }, "backtest")
            if force_live_parity:
                logger.info(
                    "[PARITY_AUDIT] "
                    + json.dumps(
                        {
                            "stage": "candle_publish",
                            "day": str(day),
                            "market_ts": ts.isoformat(),
                            "published_candle_count": len(candles_list),
                            "last_published_candle_ts": (
                                (
                                    candles_list[-1].get("timestamp")
                                    or candles_list[-1].get("datetime")
                                    or candles_list[-1].get("date")
                                )
                                if candles_list else None
                            ),
                            "ltp": round(float(ltp or 0.0), 4),
                            "candle_hash": _candle_hash(candles_list),
                            "orb_high": orb_high,
                            "orb_low": orb_low,
                            "mode": "live_parity",
                        },
                        sort_keys=True,
                        default=str,
                    )
                )

            if ignore_regime:
                await bus.publish(Topic.MARKET_REGIME, {
                    "symbol": "NIFTY",
                    "timeframe": BACKTEST_TIMEFRAME,
                    "mode": "BACKTEST",
                    "candles": candles_list,
                    "ltp": ltp,
                    "orb_high": orb_high,
                    "orb_low": orb_low,
                    "timestamp": ts.isoformat(),
                    "regime": "TRENDING",
                    "regime_details": {
                        "adx": 999.0,
                        "chop_index": 0.0,
                        "override": "BACKTEST_IGNORE_REGIME",
                    },
                }, "backtest_regime_override")

            if speed_ms:
                await asyncio.sleep(speed_ms / 1000)

        # Backtests are intraday-only. Force-close any remaining open position
        # on the final candle of the same session to avoid overnight carry.
        if position_agent.get_position():
            session_close_ts = day_df.index[-1]
            await position_agent.force_eod_exit(session_close_ts)

    strategy_agent.shutdown()
    if previous_trading_mode is None:
        os.environ.pop("TRADING_MODE", None)
    else:
        os.environ["TRADING_MODE"] = previous_trading_mode

    stats = analytics_agent.get_daily_stats()
    risk = risk_agent.status_payload()
    paper_capital = float(risk.get("paper_capital", 0.0) or 0.0)
    journal_rows = list(analytics_agent._journal)
    closed_df = _closed_trades_df(journal_rows)
    trading_days_with_orders = 0
    if not closed_df.empty:
        trading_days_with_orders = len(closed_df["entry_time"].dt.date.unique())
    
    risk_metrics = _compute_backtest_risk_metrics(journal_rows, paper_capital)
    trade_return_metrics = _compute_trade_return_metrics(closed_df)
    realized_pnl = float(
        trade_return_metrics.get("realized_pnl", 0.0)
        if not closed_df.empty else summary.get("net_realized_pnl", 0.0)
    )
    capital_return_pct = (
        round(realized_pnl / paper_capital * 100, 2)
        if paper_capital > 0 else 0.0
    )
    invested_return_pct = float(trade_return_metrics.get("invested_return_pct", 0.0) or 0.0)
    aggregate_trade_pnl_pct = float(trade_return_metrics.get("aggregate_trade_pnl_pct", 0.0) or 0.0)
    approved = int(summary.get("approved", 0) or 0)
    planned = int(summary.get("planned", 0) or 0)
    closed = int(summary.get("closed", 0) or 0)
    approved_to_planned_rate = (
        round(planned / approved * 100, 1)
        if approved > 0 else 0.0
    )
    planned_to_closed_rate = (
        round(closed / planned * 100, 1)
        if planned > 0 else 0.0
    )
    approved_precision_pct = (
        round(summary["wins"] / closed * 100, 1)
        if closed > 0 else 0.0
    )
    llm_summary = ""
    if len(summary["days"]) <= 7:
        try:
            from utils.llm import TaskType, call_llm_async

            llm_summary = await call_llm_async(
                (
                    "Summarize this SignalForge backtest in 2 concise sentences.\n"
                    f"Days: {', '.join(summary['days'])}\n"
                    f"Signals: {summary['signals']} | Approved: {summary['approved']} | "
                    f"Suppressed: {summary['suppressed']}\n"
                    f"Suppression breakdown: {summary['suppressed_by_regime']}\n"
                    f"Wins: {summary['wins']} | Losses: {summary['losses']}\n"
                    f"Invested return %: {invested_return_pct} | "
                    f"Capital return %: {capital_return_pct} | "
                    f"Trade sum %: {aggregate_trade_pnl_pct} | "
                    f"Realized PnL INR: {realized_pnl} | Costs INR: {summary['transaction_costs']}\n"
                    "Mention clearly if no signals were generated and why."
                ),
                task_type=TaskType.BACKTEST_NARRATION,
                max_tokens=96,
            )
        except Exception:
            llm_summary = ""

    result = {
        "summary": {
            **summary,
            "signals": len(raw_signal_ids),
            "market_open_days": len(summary["days"]),
            "trading_days": len(summary["days"]),
            "trading_days_with_orders": trading_days_with_orders,
            "total_pnl_pct": invested_return_pct,
            "invested_return_pct": invested_return_pct,
            "total_invested": trade_return_metrics.get("total_invested", 0.0),
            "capital_return_pct": capital_return_pct,
            "aggregate_trade_pnl_pct": aggregate_trade_pnl_pct,
            "transaction_costs": round(summary["transaction_costs"], 2),
            "net_realized_pnl": round(summary["net_realized_pnl"], 2),
            "win_rate": stats.get("win_rate", 0.0),
            "approved_precision_pct": approved_precision_pct,
            "approved_to_planned_rate": approved_to_planned_rate,
            "planned_to_closed_rate": planned_to_closed_rate,
            "selected_trades": closed,
            "number_of_trades": int(risk_metrics.get("number_of_trades", 0.0)),
            "trades_per_day": round(
                float(risk_metrics.get("number_of_trades", 0.0)) / len(summary["days"]),
                2,
            ) if summary["days"] else 0.0,
            "max_drawdown_pct": risk_metrics.get("max_drawdown_pct", 0.0),
            "sharpe_ratio": risk_metrics.get("sharpe_ratio", 0.0),
            "profit_factor": risk_metrics.get("profit_factor", 0.0),
            "avg_trade_pct": risk_metrics.get("avg_trade_pct", 0.0),
            "realized_pnl": realized_pnl,
            "realized_loss_pct": risk.get("realized_loss_pct", 0.0),
            "trading_status": risk.get("status", "ACTIVE"),
        },
        "manifest": manifest,
        "journal": journal_rows,
        "llm_summary": llm_summary,
        "risk": risk,
        "bus_stats": bus.get_stats(),
        "bus_history": _compact_bus_history(bus.get_history(limit=200)),
    }
    if previous_trading_mode is None:
        os.environ.pop("TRADING_MODE", None)
    else:
        os.environ["TRADING_MODE"] = previous_trading_mode
    if force_live_parity:
        if previous_adaptive_gates is None:
            os.environ.pop("BACKTEST_ENABLE_ADAPTIVE_GATES", None)
        else:
            os.environ["BACKTEST_ENABLE_ADAPTIVE_GATES"] = previous_adaptive_gates
        if previous_adaptive_edges is None:
            os.environ.pop("BACKTEST_ENABLE_ADAPTIVE_EDGE_FILTERS", None)
        else:
            os.environ["BACKTEST_ENABLE_ADAPTIVE_EDGE_FILTERS"] = previous_adaptive_edges
    return result


def run_backtest_sync(
    df: pd.DataFrame,
    speed_ms: int = 0,
    ignore_regime: bool = False,
    run_id: Optional[str] = None,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    live_parity: bool = False,
) -> dict[str, Any]:
    run_id = str(run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_lines: list[str] = []
    logger.remove()
    # Add back terminal logger so user can see % progress
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {message}",
        filter=lambda record: "Progress:" in record["message"] or "Starting backtest" in record["message"]
    )
    logger.add(
        lambda msg: log_lines.append(str(msg).rstrip()),
        level="INFO",
        format="{time:HH:mm:ss} | {level:<8} | {message}",
    )
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    log_path = str(Path(LOGS_DIR) / f"backtest_{run_id}.log")
    logger.add(
        log_path,
        level="INFO",
        rotation="5 MB",
        retention=5,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    )
    pre_run_settings = _tracked_settings_snapshot()
    pre_run_hash = hashlib.sha256(
        json.dumps(pre_run_settings, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    logger.info(
        "[BACKTEST_SETTINGS] "
        + json.dumps(
            {
                "run_id": run_id,
                "log_path": log_path,
                "settings_hash": pre_run_hash,
                "settings": pre_run_settings,
            },
            sort_keys=True,
            default=str,
        )
    )
    result = asyncio.run(
        run_backtest(
            df,
            speed_ms=speed_ms,
            ignore_regime=ignore_regime,
            progress_callback=progress_callback,
            live_parity=live_parity,
        )
    )
    result["debug_logs"] = log_lines[-1000:]
    result["run_id"] = run_id
    result["backtest_log_path"] = log_path
    manifest = dict(result.get("manifest", {}) or {})
    manifest["run_id"] = run_id
    manifest["backtest_log_path"] = log_path
    result["manifest"] = manifest
    return result


def _build_backtest_manifest(df: pd.DataFrame, ml_agent) -> dict[str, Any]:
    ensemble = getattr(ml_agent, "ensemble", None)
    model_threshold = (
        ML_THRESHOLD_OVERRIDE
        if ML_THRESHOLD_OVERRIDE > 0 else (
            float(getattr(ensemble, "decision_threshold", 0.0) or 0.0)
            if ensemble else 0.0
        )
    )
    settings_snapshot = {
        "signal_timeframe": BACKTEST_TIMEFRAME,
        **_tracked_settings_snapshot(),
    }
    settings_hash = hashlib.sha256(
        json.dumps(settings_snapshot, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "cache_interval": str(df.attrs.get("cache_interval") or BACKTEST_TIMEFRAME),
        "cache_file": str(df.attrs.get("cache_file") or ""),
        "requested_interval": str(df.attrs.get("requested_interval") or BACKTEST_TIMEFRAME),
        "sources": list(df.attrs.get("source_list", [])),
        "source_brokers": list(df.attrs.get("source_brokers", [])),
        "model_path": str(getattr(ml_agent, "_model_path", "")),
        "model_trained_at": str(getattr(getattr(ensemble, "meta", None), "trained_at", "") or ""),
        "model_threshold": round(float(model_threshold or 0.0), 4),
        "model_threshold_source": (
            "override" if ML_THRESHOLD_OVERRIDE > 0 else "model_meta"
        ),
        "settings": settings_snapshot,
        "settings_hash": settings_hash,
    }


def _tracked_settings_snapshot() -> dict[str, Any]:
    tracked_setting_names = [
        "TOTAL_FUND",
        "DEPLOYED_CAPITAL",
        "STRONG_SIGNAL_CAPITAL_PCT",
        "DAILY_CAPITAL_PCT",
        "MAX_POSITION_LOTS",
        "NIFTY_LOT_SIZE",
        "MAX_TRADES_PER_DAY",
        "SIGNAL_APPROVAL_DAILY_BUDGET",
        "SIGNAL_SESSION_BUDGETS",
        "SIGNAL_REGIME_BUDGETS",
        "ML_THRESHOLD_OVERRIDE",
        "ML_SECONDARY_THRESHOLD",
        "ML_SECONDARY_MIN_RAW_CONFIDENCE",
        "PLANNER_MIN_EXECUTABLE_RANK",
        "PLANNER_MIN_STRATEGY_CONFIDENCE",
        "EXECUTION_MIN_RANK_SCORE",
        "STOP_LOSS_PCT",
        "TARGET_PCT",
        "TIME_STOP_MINUTES",
        "TIME_STOP_MIN_PNL_PCT",
        "TRAILING_SL_ACTIVATION_PCT",
        "TRAILING_SL_PCT",
        "TRAILING_SL_PCT_TIER2",
        "TRAILING_SL_PCT_TIER3",
        "TRAILING_SL_ADX_BONUS",
        "TRAILING_SL_ADX_THRESH",
        "STALE_EARLY_CHECK_CANDLES",
        "STALE_EARLY_LOSS_THRESHOLD",
        "STALE_MOMENTUM_THRESHOLD",
        "STALE_MAX_CANDLES",
        "USE_PROFIT_LADDER",
        "LADDER_BOOK_1_PCT",
        "LADDER_BOOK_1_QTY",
        "LADDER_BOOK_2_PCT",
        "LADDER_BOOK_2_QTY",
        "RISK_BUDGET_PER_TRADE_PCT",
        "MIN_RR_RATIO",
        "BACKTEST_OPTION_SLIPPAGE_PCT",
        "BACKTEST_BROKERAGE_PER_ORDER",
        "BACKTEST_TRANSACTION_COST_PCT",
        "BACKTEST_USE_HISTORICAL_OPTION_PREMIUM",
        "PLANNER_ENTRY_SLIPPAGE_PCT",
        "OPTION_MIN_PREMIUM",
        "OPTION_MAX_PREMIUM",
        "OPTION_CANDIDATE_STEPS",
        "EXECUTION_LAST_ENTRY_MINUTE",
        "SECOND_TRADE_COOLDOWN_MIN",
        "SECOND_TRADE_FAST_REENTRY_MIN",
    ]
    return {
            name: getattr(settings, name)
            for name in tracked_setting_names
            if hasattr(settings, name)
    }

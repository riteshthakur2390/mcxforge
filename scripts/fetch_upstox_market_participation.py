#!/usr/bin/env python3
"""
Fetch Upstox market participation data for dashboard context.

This script is dashboard-only. It writes raw endpoint responses and a small
normalized summary to data/market_participation/latest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytz
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IST = pytz.timezone("Asia/Kolkata")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "market_participation"
UPSTOX_BASE_URL = "https://api.upstox.com"
NSE_ARCHIVE_BASE_URL = "https://archives.nseindia.com/content/nsccl"

ENDPOINTS = {
    "fii": "/v2/market/fii",
    "fii_historical": "/v2/market/fii/historical",
    "dii": "/v2/market/dii",
    "oi": "/v2/market/oi",
    "pcr": "/v2/market/pcr",
    "max_pain": "/v2/market/max-pain",
}

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) SignalForge/1.0",
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/",
}


def _json_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw": response.text[:1000]}


def _first_number(value: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_norm = str(key).lower().replace("_", "").replace("-", "")
            if key_norm in keys:
                try:
                    return float(str(item).replace(",", ""))
                except Exception:
                    pass
        for item in value.values():
            found = _first_number(item, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _first_number(item, keys)
            if found is not None:
                return found
    return None


def _number(value: Any) -> float | None:
    try:
        text = str(value).strip().replace(",", "")
        if not text or text == "-":
            return None
        return float(text)
    except Exception:
        return None


def _norm_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _row_number(row: dict[str, Any], candidates: tuple[str, ...]) -> float:
    normalized = {_norm_key(key): value for key, value in row.items()}
    for candidate in candidates:
        value = _number(normalized.get(_norm_key(candidate)))
        if value is not None:
            return value
    return 0.0


def _pcr_bias(pcr: float | None) -> tuple[str, str]:
    if pcr is None:
        return "UNKNOWN", "PCR unavailable"
    if pcr >= 1.10:
        return "BULLISH", "PCR above 1.10 shows put-side support"
    if pcr <= 0.80:
        return "BEARISH", "PCR below 0.80 shows call-side pressure"
    return "NEUTRAL", "PCR is in the middle band"


def _flow_bias(net: float | None, label: str) -> str:
    if net is None:
        return f"{label} unavailable"
    if net > 0:
        return f"{label} net buying"
    if net < 0:
        return f"{label} net selling"
    return f"{label} flat"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _scaled_component(raw: float | None, neutral: float = 0.5) -> float:
    if raw is None:
        return neutral
    return _clamp(raw)


def _participant_row_summary(row: dict[str, Any]) -> dict[str, Any]:
    index_future_long = _row_number(row, ("Future Index Long", "Index Futures Long", "Future Index Longs"))
    index_future_short = _row_number(row, ("Future Index Short", "Index Futures Short", "Future Index Shorts"))
    stock_future_long = _row_number(row, ("Future Stock Long", "Stock Futures Long"))
    stock_future_short = _row_number(row, ("Future Stock Short", "Stock Futures Short"))
    index_call_long = _row_number(row, ("Option Index Call Long", "Index Option Call Long"))
    index_put_long = _row_number(row, ("Option Index Put Long", "Index Option Put Long"))
    index_call_short = _row_number(row, ("Option Index Call Short", "Index Option Call Short"))
    index_put_short = _row_number(row, ("Option Index Put Short", "Index Option Put Short"))
    stock_call_long = _row_number(row, ("Option Stock Call Long", "Stock Option Call Long"))
    stock_put_long = _row_number(row, ("Option Stock Put Long", "Stock Option Put Long"))
    stock_call_short = _row_number(row, ("Option Stock Call Short", "Stock Option Call Short"))
    stock_put_short = _row_number(row, ("Option Stock Put Short", "Stock Option Put Short"))

    futures_long = index_future_long + stock_future_long
    futures_short = index_future_short + stock_future_short
    options_long = index_call_long + index_put_long + stock_call_long + stock_put_long
    options_short = index_call_short + index_put_short + stock_call_short + stock_put_short
    call_long = index_call_long + stock_call_long
    call_short = index_call_short + stock_call_short
    put_long = index_put_long + stock_put_long
    put_short = index_put_short + stock_put_short
    net_call = call_long - call_short
    net_put = put_long - put_short
    call_total = call_long + call_short
    put_total = put_long + put_short
    option_total = call_total + put_total
    bullish_contracts = call_long + put_short
    bearish_contracts = put_long + call_short
    direction_score = bullish_contracts - bearish_contracts
    if direction_score > option_total * 0.05:
        option_bias = "BULLISH"
    elif direction_score < -option_total * 0.05:
        option_bias = "BEARISH"
    else:
        option_bias = "NEUTRAL"
    total_long = futures_long + options_long
    total_short = futures_short + options_short
    total = total_long + total_short
    long_pct = round(total_long / total * 100.0, 2) if total else None
    short_pct = round(total_short / total * 100.0, 2) if total else None

    index_options_note = "Neutral"
    if index_put_short > index_call_short * 1.15:
        index_options_note = "Put writing dominant"
    elif index_call_short > index_put_short * 1.15:
        index_options_note = "Call writing dominant"
    elif index_put_long > index_call_long * 1.15:
        index_options_note = "Put buying dominant"
    elif index_call_long > index_put_long * 1.15:
        index_options_note = "Call buying dominant"

    return {
        "index_futures": {
            "long": index_future_long,
            "short": index_future_short,
            "long_pct": round(index_future_long / max(index_future_long + index_future_short, 1) * 100.0, 2),
            "short_pct": round(index_future_short / max(index_future_long + index_future_short, 1) * 100.0, 2),
        },
        "stock_futures": {
            "long": stock_future_long,
            "short": stock_future_short,
            "long_pct": round(stock_future_long / max(stock_future_long + stock_future_short, 1) * 100.0, 2),
            "short_pct": round(stock_future_short / max(stock_future_long + stock_future_short, 1) * 100.0, 2),
        },
        "index_options": {
            "call_long": index_call_long,
            "put_long": index_put_long,
            "call_short": index_call_short,
            "put_short": index_put_short,
            "note": index_options_note,
        },
        "stock_options": {
            "call_long": stock_call_long,
            "put_long": stock_put_long,
            "call_short": stock_call_short,
            "put_short": stock_put_short,
        },
        "options_carryforward": {
            "call_long": call_long,
            "call_short": call_short,
            "put_long": put_long,
            "put_short": put_short,
            "net_call": net_call,
            "net_put": net_put,
            "call_long_pct": round(call_long / max(call_total, 1) * 100.0, 2),
            "call_short_pct": round(call_short / max(call_total, 1) * 100.0, 2),
            "put_long_pct": round(put_long / max(put_total, 1) * 100.0, 2),
            "put_short_pct": round(put_short / max(put_total, 1) * 100.0, 2),
            "bullish_contracts": bullish_contracts,
            "bearish_contracts": bearish_contracts,
            "direction_score": direction_score,
            "bias": option_bias,
        },
        "total": {
            "long": total_long,
            "short": total_short,
            "long_pct": long_pct,
            "short_pct": short_pct,
        },
    }


def _index_option_activity_summary(row: dict[str, Any]) -> dict[str, Any]:
    call_buy = _row_number(row, ("Option Index Call Long", "Index Option Call Long"))
    put_buy = _row_number(row, ("Option Index Put Long", "Index Option Put Long"))
    call_sell = _row_number(row, ("Option Index Call Short", "Index Option Call Short"))
    put_sell = _row_number(row, ("Option Index Put Short", "Index Option Put Short"))
    call_total = call_buy + call_sell
    put_total = put_buy + put_sell
    total_buy = call_buy + put_buy
    total_sell = call_sell + put_sell
    net_call = call_buy - call_sell
    net_put = put_buy - put_sell
    net_oi = net_call + net_put
    bullish_contracts = call_buy + put_sell
    bearish_contracts = put_buy + call_sell
    direction_score = bullish_contracts - bearish_contracts
    if direction_score > (call_total + put_total) * 0.05:
        bias = "BULLISH"
    elif direction_score < -(call_total + put_total) * 0.05:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"
    return {
        "call_buy": call_buy,
        "call_sell": call_sell,
        "net_call": net_call,
        "put_buy": put_buy,
        "put_sell": put_sell,
        "net_put": net_put,
        "net_oi": net_oi,
        "call_buy_pct": round(call_buy / max(call_total, 1) * 100.0, 2),
        "call_sell_pct": round(call_sell / max(call_total, 1) * 100.0, 2),
        "put_buy_pct": round(put_buy / max(put_total, 1) * 100.0, 2),
        "put_sell_pct": round(put_sell / max(put_total, 1) * 100.0, 2),
        "total_buy": total_buy,
        "total_sell": total_sell,
        "bullish_contracts": bullish_contracts,
        "bearish_contracts": bearish_contracts,
        "direction_score": direction_score,
        "bias": bias,
    }


def _participant_direction_view(
    *,
    participant_summary: dict[str, Any],
    index_option_summary: dict[str, Any],
) -> dict[str, Any]:
    futures = participant_summary.get("index_futures", {}) if isinstance(participant_summary, dict) else {}
    future_long = _number(futures.get("long")) or 0.0
    future_short = _number(futures.get("short")) or 0.0
    net_future = future_long - future_short
    net_call = _number(index_option_summary.get("net_call")) or 0.0
    net_put = _number(index_option_summary.get("net_put")) or 0.0

    score = 0
    reasons: list[str] = []

    if net_future > 0:
        score += 2
        reasons.append("net long index futures")
    elif net_future < 0:
        score -= 2
        reasons.append("net short index futures")

    if net_call > 0:
        score += 1
        reasons.append("net long calls")
    elif net_call < 0:
        score -= 1
        reasons.append("net short calls")

    if net_put < 0:
        score += 1
        reasons.append("net short puts")
    elif net_put > 0:
        score -= 1
        reasons.append("net long puts")

    if score >= 3:
        direction = "STRONG BULLISH"
    elif score >= 1:
        direction = "BULLISH"
    elif score <= -3:
        direction = "STRONG BEARISH"
    elif score <= -1:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    if abs(net_future) < max(abs(net_call), abs(net_put), 1.0) * 0.15 and direction != "NEUTRAL":
        direction = "HEDGED " + direction

    return {
        "net_future": net_future,
        "net_call": net_call,
        "net_put": net_put,
        "direction": direction,
        "score": score,
        "reasons": reasons,
    }


def _derive_market_score(
    *,
    pcr: float | None,
    fii_net: float | None,
    dii_net: float | None,
    long_short: float | None,
    oi_change: float | None,
) -> dict[str, Any]:
    pcr_score = 0.5
    if pcr is not None:
        if pcr >= 1.35:
            pcr_score = 0.95
        elif pcr >= 1.10:
            pcr_score = 0.78
        elif pcr <= 0.65:
            pcr_score = 0.05
        elif pcr <= 0.80:
            pcr_score = 0.22
        else:
            pcr_score = 0.50 + ((pcr - 0.95) * 0.55)
            pcr_score = _clamp(pcr_score, 0.30, 0.70)

    fii_flow_score = 0.5
    if fii_net is not None:
        fii_flow_score = 0.50 + _clamp(abs(fii_net) / 5000.0, 0.0, 0.45) * (1 if fii_net > 0 else -1)

    dii_flow_score = 0.5
    if dii_net is not None:
        dii_flow_score = 0.50 + _clamp(abs(dii_net) / 5000.0, 0.0, 0.35) * (1 if dii_net > 0 else -1)

    long_short_score = 0.5
    if long_short is not None:
        # Handles either ratio style (1.2) or percent style (72).
        long_pct = long_short if long_short > 10 else long_short * 50.0
        long_short_score = _clamp(long_pct / 100.0)

    oi_score = 0.5
    if oi_change is not None:
        oi_score = 0.50 + _clamp(abs(oi_change) / 1000000.0, 0.0, 0.25) * (1 if oi_change > 0 else -1)

    components = {
        "trend": {"score": 12.5, "max": 25, "note": "Trend source not connected yet"},
        "fii": {
            "score": round((_scaled_component(fii_flow_score) * 0.60 + _scaled_component(long_short_score) * 0.40) * 25, 1),
            "max": 25,
            "note": "FII flow and long/short positioning",
        },
        "pcr": {"score": round(_scaled_component(pcr_score) * 20, 1), "max": 20, "note": "Put-call ratio"},
        "oi": {"score": round(_scaled_component(oi_score) * 20, 1), "max": 20, "note": "Open interest change"},
        "breadth": {"score": 5.0, "max": 10, "note": "Breadth source not connected yet"},
        "volatility": {"score": 5.0, "max": 10, "note": "VIX trend source not connected yet"},
    }
    total = round(sum(float(item["score"]) for item in components.values()), 1)
    if total >= 75:
        recommendation = "STRONG BULLISH"
    elif total >= 62:
        recommendation = "BULLISH"
    elif total <= 25:
        recommendation = "STRONG BEARISH"
    elif total <= 38:
        recommendation = "BEARISH"
    else:
        recommendation = "MIXED"
    return {
        "total": total,
        "max": 110,
        "percent": round(total / 110.0 * 100.0, 1),
        "recommendation": recommendation,
        "components": components,
    }


def _derive_summary(responses: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pcr_data = responses.get("pcr", {}).get("data")
    fii_data = responses.get("fii", {}).get("data")
    dii_data = responses.get("dii", {}).get("data")
    oi_data = responses.get("oi", {}).get("data")
    max_pain_data = responses.get("max_pain", {}).get("data")

    pcr = _first_number(pcr_data, ("pcr", "putcallratio", "putcall"))
    fii_buy = _first_number(fii_data, ("buy", "buyvalue", "grossbuy", "fiibuy"))
    fii_sell = _first_number(fii_data, ("sell", "sellvalue", "grosssell", "fiisell"))
    fii_net = _first_number(fii_data, ("net", "netvalue", "netinvestment", "fiinet"))
    if fii_net is None and fii_buy is not None and fii_sell is not None:
        fii_net = fii_buy - fii_sell

    dii_buy = _first_number(dii_data, ("buy", "buyvalue", "grossbuy", "diibuy"))
    dii_sell = _first_number(dii_data, ("sell", "sellvalue", "grosssell", "diisell"))
    dii_net = _first_number(dii_data, ("net", "netvalue", "netinvestment", "diinet"))
    if dii_net is None and dii_buy is not None and dii_sell is not None:
        dii_net = dii_buy - dii_sell

    long_short = _first_number(
        fii_data,
        ("longshort", "longshortratio", "fiilongshort", "longshortratio"),
    )
    oi_change = _first_number(oi_data, ("oichange", "changeinoi", "openinterestchange"))
    open_interest = _first_number(oi_data, ("oi", "openinterest", "totaloi"))
    max_pain = _first_number(max_pain_data, ("maxpain", "maxpainstrike", "strike"))

    pcr_side, pcr_note = _pcr_bias(pcr)
    flow_score = 0
    for value in (fii_net, dii_net):
        if value is not None:
            flow_score += 1 if value > 0 else -1 if value < 0 else 0
    if pcr_side == "BULLISH":
        flow_score += 1
    elif pcr_side == "BEARISH":
        flow_score -= 1

    if flow_score >= 2:
        market_side = "BULLISH"
    elif flow_score <= -2:
        market_side = "BEARISH"
    else:
        market_side = "MIXED"

    notes = [
        pcr_note,
        _flow_bias(fii_net, "FII"),
        _flow_bias(dii_net, "DII"),
    ]
    if long_short is not None:
        notes.append(f"FII long/short {long_short:.2f}")
    if oi_change is not None:
        notes.append(f"OI change {oi_change:,.0f}")
    market_score = _derive_market_score(
        pcr=pcr,
        fii_net=fii_net,
        dii_net=dii_net,
        long_short=long_short,
        oi_change=oi_change,
    )

    return {
        "market_side": market_side,
        "market_score": market_score,
        "pcr_side": pcr_side,
        "pcr": pcr,
        "fii": {"buy": fii_buy, "sell": fii_sell, "net": fii_net, "long_short": long_short},
        "dii": {"buy": dii_buy, "sell": dii_sell, "net": dii_net},
        "oi": {"open_interest": open_interest, "change": oi_change},
        "max_pain": max_pain,
        "notes": notes,
    }


def _fetch_endpoint(
    session: requests.Session,
    endpoint: str,
    token: str,
    params: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    url = f"{UPSTOX_BASE_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    try:
        response = session.get(url, headers=headers, params=params, timeout=timeout)
        return {
            "ok": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "url": f"{url}?{urlencode(params)}" if params else url,
            "data": _json_response(response),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "url": f"{url}?{urlencode(params)}" if params else url,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _recent_report_dates(lookback_days: int) -> list[date]:
    today = datetime.now(IST).date()
    return [today - timedelta(days=offset) for offset in range(max(1, lookback_days))]


def _parse_nse_csv(text: str) -> list[dict[str, str]]:
    raw_lines = [
        line for line in text.splitlines()
        if line.strip() and not line.lower().startswith("disclaimer")
    ]
    header_idx = next(
        (idx for idx, line in enumerate(raw_lines) if line.lower().lstrip('"').startswith("client type,")),
        0,
    )
    lines = raw_lines[header_idx:]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    return [
        {str(k or "").strip(): str(v or "").strip() for k, v in row.items()}
        for row in reader
        if any(str(v or "").strip() for v in row.values())
    ]


def _participant_name(row: dict[str, Any]) -> str:
    for key, value in row.items():
        if _norm_key(key) in {"clienttype", "participant", "client"}:
            return str(value or "").strip().upper()
    first_value = next(iter(row.values()), "")
    return str(first_value or "").strip().upper()


def _participant_row_by_name(rows: list[dict[str, Any]], *names: str) -> dict[str, Any]:
    wanted = {str(name or "").strip().upper() for name in names if str(name or "").strip()}
    for row in rows:
        if _participant_name(row) in wanted:
            return row
    return {}


def _fetch_nse_csv(session: requests.Session, path: str, timeout: float) -> dict[str, Any]:
    url = f"{NSE_ARCHIVE_BASE_URL}/{path}"
    try:
        response = session.get(url, headers=NSE_HEADERS, timeout=timeout)
        ok = 200 <= response.status_code < 300 and "html" not in response.text[:200].lower()
        return {
            "ok": ok,
            "status_code": response.status_code,
            "url": url,
            "text": response.text if ok else response.text[:500],
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": None,
            "url": url,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _fetch_nse_participant_data(timeout: float, lookback_days: int) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    with requests.Session() as session:
        for report_day in _recent_report_dates(lookback_days):
            stamp = report_day.strftime("%d%m%Y")
            oi_path = f"fao_participant_oi_{stamp}.csv"
            vol_path = f"fao_participant_vol_{stamp}.csv"
            oi_response = _fetch_nse_csv(session, oi_path, timeout)
            vol_response = _fetch_nse_csv(session, vol_path, timeout)
            attempts.append({
                "date": report_day.isoformat(),
                "oi": {k: v for k, v in oi_response.items() if k != "text"},
                "volume": {k: v for k, v in vol_response.items() if k != "text"},
            })
            if not oi_response.get("ok"):
                continue
            oi_rows = _parse_nse_csv(str(oi_response.get("text", "")))
            vol_rows = _parse_nse_csv(str(vol_response.get("text", ""))) if vol_response.get("ok") else []
            participants = {
                _participant_name(row): _participant_row_summary(row)
                for row in oi_rows
                if _participant_name(row)
            }
            participant_volume = {
                _participant_name(row): _index_option_activity_summary(row)
                for row in vol_rows
                if _participant_name(row)
            }
            return {
                "ok": True,
                "source": "nse",
                "report_date": report_day.isoformat(),
                "responses": {
                    "participant_oi": {
                        "ok": True,
                        "status_code": oi_response.get("status_code"),
                        "url": oi_response.get("url"),
                        "rows": oi_rows,
                    },
                    "participant_volume": {
                        "ok": bool(vol_response.get("ok")),
                        "status_code": vol_response.get("status_code"),
                        "url": vol_response.get("url"),
                        "rows": vol_rows,
                    },
                },
                "participants": participants,
                "participant_volume": participant_volume,
                "attempts": attempts,
            }
    return {
        "ok": False,
        "source": "nse",
        "attempts": attempts,
        "error": "No NSE participant OI report found in lookback window",
    }


def _derive_nse_summary(nse_data: dict[str, Any]) -> dict[str, Any]:
    participants = nse_data.get("participants", {}) if isinstance(nse_data, dict) else {}
    participant_volume = nse_data.get("participant_volume", {}) if isinstance(nse_data, dict) else {}
    participant_oi_rows = (((nse_data.get("responses", {}) or {}).get("participant_oi", {}) or {}).get("rows") or [])
    fii = participants.get("FII") or participants.get("FPI") or {}
    dii = participants.get("DII") or {}
    client = participants.get("CLIENT") or {}
    pro = participants.get("PRO") or {}

    fii_total = fii.get("total", {}) if isinstance(fii, dict) else {}
    dii_total = dii.get("total", {}) if isinstance(dii, dict) else {}
    pro_total = pro.get("total", {}) if isinstance(pro, dict) else {}
    client_total = client.get("total", {}) if isinstance(client, dict) else {}
    fii_long_pct = _number(fii_total.get("long_pct"))
    dii_long_pct = _number(dii_total.get("long_pct"))
    pro_long_pct = _number(pro_total.get("long_pct"))
    client_long_pct = _number(client_total.get("long_pct"))
    carryforward = {
        "FII": _index_option_activity_summary(_participant_row_by_name(participant_oi_rows, "FII", "FPI")),
        "DII": _index_option_activity_summary(_participant_row_by_name(participant_oi_rows, "DII")),
        "PRO": _index_option_activity_summary(_participant_row_by_name(participant_oi_rows, "PRO")),
        "CLIENT": _index_option_activity_summary(_participant_row_by_name(participant_oi_rows, "CLIENT")),
    }
    daily_activity = {
        "FII": (participant_volume.get("FII") or participant_volume.get("FPI") or {}),
        "DII": (participant_volume.get("DII") or {}),
        "PRO": (participant_volume.get("PRO") or {}),
        "CLIENT": (participant_volume.get("CLIENT") or {}),
    }
    participant_direction = {
        "FII": _participant_direction_view(
            participant_summary=(participants.get("FII") or participants.get("FPI") or {}),
            index_option_summary=carryforward.get("FII", {}),
        ),
        "DII": _participant_direction_view(
            participant_summary=(participants.get("DII") or {}),
            index_option_summary=carryforward.get("DII", {}),
        ),
        "PRO": _participant_direction_view(
            participant_summary=(participants.get("PRO") or {}),
            index_option_summary=carryforward.get("PRO", {}),
        ),
        "CLIENT": _participant_direction_view(
            participant_summary=(participants.get("CLIENT") or {}),
            index_option_summary=carryforward.get("CLIENT", {}),
        ),
    }

    pcr = None
    idx_options = fii.get("index_options", {}) if isinstance(fii, dict) else {}
    fii_put_short = _number(idx_options.get("put_short")) or 0.0
    fii_call_short = _number(idx_options.get("call_short")) or 0.0
    if fii_call_short > 0:
        pcr = round(fii_put_short / fii_call_short, 3)

    long_short = None
    if fii_long_pct is not None and fii_long_pct < 100:
        long_short = round(fii_long_pct / max(100.0 - fii_long_pct, 1.0), 3)
    pcr_side, pcr_note = _pcr_bias(pcr)

    market_score = _derive_market_score(
        pcr=pcr,
        fii_net=(fii_long_pct - 50.0) * 100.0 if fii_long_pct is not None else None,
        dii_net=(dii_long_pct - 50.0) * 100.0 if dii_long_pct is not None else None,
        long_short=long_short,
        oi_change=None,
    )
    market_side = market_score["recommendation"].replace("STRONG ", "")
    if market_side not in {"BULLISH", "BEARISH"}:
        market_side = "MIXED"

    notes = [
        f"NSE report date {nse_data.get('report_date', 'unknown')}",
        f"FII long {fii_long_pct:.1f}%" if fii_long_pct is not None else "FII unavailable",
        f"PRO long {pro_long_pct:.1f}%" if pro_long_pct is not None else "PRO unavailable",
        f"Client long {client_long_pct:.1f}%" if client_long_pct is not None else "Client unavailable",
        pcr_note,
    ]
    fii_bias = str(carryforward.get("FII", {}).get("bias") or "NEUTRAL")
    notes.append(f"FII option carry-forward bias is {fii_bias.lower()}, so market direction reads {fii_bias.lower()}.")
    fii_direction = str((participant_direction.get("FII") or {}).get("direction") or "NEUTRAL")
    pro_direction = str((participant_direction.get("PRO") or {}).get("direction") or "NEUTRAL")
    client_direction = str((participant_direction.get("CLIENT") or {}).get("direction") or "NEUTRAL")
    notes.append(f"FII positioning reads {fii_direction.lower()}, PRO reads {pro_direction.lower()}, client reads {client_direction.lower()}.")

    return {
        "market_side": market_side,
        "market_score": market_score,
        "pcr_side": pcr_side,
        "pcr": pcr,
        "fii": {
            "buy": None,
            "sell": None,
            "net": None,
            "long_short": long_short,
            "long_pct": fii_long_pct,
            "short_pct": _number(fii_total.get("short_pct")),
        },
        "dii": {
            "buy": None,
            "sell": None,
            "net": None,
            "long_pct": dii_long_pct,
            "short_pct": _number(dii_total.get("short_pct")),
        },
        "pro": {"long_pct": pro_long_pct, "short_pct": _number(pro_total.get("short_pct"))},
        "client": {"long_pct": client_long_pct, "short_pct": _number(client_total.get("short_pct"))},
        "oi": {"open_interest": None, "change": None},
        "max_pain": None,
        "daily_index_option_activity": daily_activity,
        "options_carryforward": carryforward,
        "participant_direction_view": participant_direction,
        "participants": participants,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Upstox FII/DII/OI/PCR dashboard data.")
    parser.add_argument("--source", choices=("auto", "upstox", "nse"), default="auto")
    parser.add_argument("--env", default=str(PROJECT_ROOT / ".env"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--symbol", default=os.getenv("UPSTOX_MARKET_DATA_SYMBOL", "NIFTY"))
    parser.add_argument("--no-symbol-param", action="store_true", help="Do not send symbol=NIFTY as a query parameter")
    parser.add_argument("--nse-lookback-days", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--param", action="append", default=[], help="Extra query param as key=value")
    args = parser.parse_args()

    env_path = Path(args.env).expanduser().resolve()
    if load_dotenv and env_path.exists():
        load_dotenv(env_path, override=True)

    token = (
        os.getenv("UPSTOX_MARKET_DATA_ACCESS_TOKEN", "").strip()
        or os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {} if args.no_symbol_param else {"symbol": args.symbol}
    for raw in args.param:
        if "=" not in raw:
            print(f"WARN ignoring --param without key=value: {raw}")
            continue
        key, value = raw.split("=", 1)
        params[key.strip()] = value.strip()

    fetched_at = datetime.now(IST).isoformat()
    payload: dict[str, Any] = {
        "source": args.source,
        "active_source": "",
        "fetched_at": fetched_at,
        "date": datetime.now(IST).date().isoformat(),
        "symbol": args.symbol,
        "params": params,
        "endpoints": ENDPOINTS,
        "responses": {},
        "summary": {},
    }

    upstox_usable = False
    if args.source in {"auto", "upstox"}:
        if not token:
            payload["upstox_error"] = "UPSTOX_MARKET_DATA_ACCESS_TOKEN or UPSTOX_ACCESS_TOKEN is required"
        else:
            with requests.Session() as session:
                for name, endpoint in ENDPOINTS.items():
                    payload["responses"][name] = _fetch_endpoint(
                        session=session,
                        endpoint=endpoint,
                        token=token,
                        params=params,
                        timeout=args.timeout,
                    )
            upstox_usable = any(item.get("ok") for item in payload["responses"].values())

    if args.source == "upstox" and not upstox_usable:
        payload["active_source"] = "upstox"
        payload["summary"] = _derive_summary(payload["responses"])
    elif upstox_usable:
        payload["active_source"] = "upstox"
        payload["summary"] = _derive_summary(payload["responses"])
    else:
        nse_data = _fetch_nse_participant_data(timeout=args.timeout, lookback_days=args.nse_lookback_days)
        payload["active_source"] = "nse"
        payload["nse"] = nse_data
        payload["summary"] = _derive_nse_summary(nse_data) if nse_data.get("ok") else {
            "market_side": "UNKNOWN",
            "market_score": _derive_market_score(pcr=None, fii_net=None, dii_net=None, long_short=None, oi_change=None),
            "notes": [nse_data.get("error", "NSE participant data unavailable")],
        }

    if payload["active_source"] == "nse" and not payload.get("nse", {}).get("ok"):
        payload["error"] = payload.get("nse", {}).get("error", "NSE participant data unavailable")

    latest_path = output_dir / "latest.json"
    history_path = output_dir / f"snapshots_{datetime.now(IST).date().isoformat()}.jsonl"
    existing_payload: dict[str, Any] | None = None
    if latest_path.exists():
        try:
            loaded = json.loads(latest_path.read_text(errors="replace"))
            if isinstance(loaded, dict):
                existing_payload = loaded
        except Exception:
            existing_payload = None

    latest_to_write = payload
    if payload.get("error") and isinstance(existing_payload, dict):
        existing_summary = existing_payload.get("summary", {}) if isinstance(existing_payload.get("summary"), dict) else {}
        if existing_summary.get("daily_index_option_activity") or existing_summary.get("options_carryforward"):
            latest_to_write = {
                **existing_payload,
                "last_refresh_error": payload.get("error"),
                "last_refresh_attempt_at": fetched_at,
            }

    latest_path.write_text(json.dumps(latest_to_write, indent=2, default=str) + "\n")
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")

    failures = [
        name for name, item in payload["responses"].items()
        if not item.get("ok")
    ]
    print(f"wrote: {latest_path}")
    print(f"history: {history_path}")
    print(f"active_source: {latest_to_write.get('active_source')}")
    print(f"market_side: {(latest_to_write.get('summary') or {}).get('market_side', 'UNKNOWN')}")
    if failures:
        print(f"WARN failed endpoints: {', '.join(failures)}")
    return 2 if payload.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())

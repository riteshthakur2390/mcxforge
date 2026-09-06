#!/usr/bin/env python3
"""
Summarize live SignalForge health from today's logs.

Run this after the market has been open for 1-2 hours to verify that live data,
option volume, strategy voting, ML filtering, and trade planning are flowing.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
except Exception:
    pass


STRATEGIES = [
    "SuperTrend+RSI",
    "VWAP+EMA",
    "ORB",
    "BBSqueeze",
    "ADX+PSAR",
    "FVG",
    "UTBot",
    "CPR",
    "Ichimoku",
    "VolumeProfile",
    "LiqSweep",
    "PriceAction",
    "OIAnalysis",
    "IVContraction",
    "AMD",
    "GapDirection",
    "SkewHunter",
    "SMC",
    "ExpiryWeek",
]


def _registered_strategy_names() -> list[str]:
    try:
        from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY

        names = [str(meta.name) for meta in STRATEGY_REGISTRY]
        return names or STRATEGIES
    except Exception:
        return STRATEGIES

UTILITY_IMPORTS = [
    "utils.greeks_filter:GreeksFilter",
    "utils.greeks_position_sizer:size_by_delta",
    "utils.market_intelligence:MarketIntelligence",
    "utils.mtf_data_fetcher:MTFDataFetcher",
    "utils.news_sentiment:get_news_filter",
    "utils.option_chain_snapshot:write_option_chain_snapshot",
    "utils.smart_entry_filter:SmartEntryFilter",
    "utils.dynamic_exit_manager:compute_atr_sl",
    "utils.final_decision:FinalDecisionLayer",
    "utils.signal_conflict_resolver:get_resolver",
]


@dataclass
class ProbeResult:
    name: str
    ok: bool
    detail: str
    warn: bool = False


def _today_log() -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    return Path("logs") / f"signalforge_{today}.log"


def _last_session(lines: list[str]) -> tuple[int, list[str]]:
    markers = [
        idx
        for idx, line in enumerate(lines)
        if "SignalForge v1.0.0" in line or "SignalForge running." in line
    ]
    start = markers[-1] if markers else 0
    return start, lines[start:]


def _safe_list(raw: str) -> list[str]:
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, list):
            return [str(item) for item in value]
    except Exception:
        pass
    return []


def _extract_vote_lists(line: str) -> list[str]:
    out: list[str] = []
    for raw in re.findall(r"(?:CALL|PUT)=\d+(\[[^\]]*\])", line):
        out.extend(_safe_list(raw))
    return out


def _status(label: str, ok: bool, warn: bool = False) -> str:
    if ok:
        return f"OK     {label}"
    if warn:
        return f"WARN   {label}"
    return f"FAIL   {label}"


def _probe_status(result: ProbeResult) -> str:
    return _status(f"{result.name}: {result.detail}", result.ok, result.warn)


def _load_attr(path: str) -> Any:
    module_name, attr_name = path.split(":", 1)
    module = __import__(module_name, fromlist=[attr_name])
    return getattr(module, attr_name)


def _active_strategy_probes() -> list[ProbeResult]:
    results: list[ProbeResult] = []
    try:
        from agents_code.agent2_strategy.runner import STRATEGY_REGISTRY
    except Exception as exc:
        return [ProbeResult("strategy registry import", False, f"{type(exc).__name__}: {exc}")]

    registry_names = [meta.name for meta in STRATEGY_REGISTRY]
    expected_names = _registered_strategy_names()
    missing = sorted(set(expected_names) - set(registry_names))
    extra = sorted(set(registry_names) - set(expected_names))
    results.append(
        ProbeResult(
            "all expected strategies registered",
            not missing,
            f"registered={len(registry_names)} missing={missing or 'none'} extra={extra or 'none'}",
        )
    )

    bad: list[str] = []
    live_only: list[str] = []
    volume_required: list[str] = []
    for meta in STRATEGY_REGISTRY:
        if not hasattr(meta.instance, "evaluate"):
            bad.append(f"{meta.name}:missing_evaluate")
        if getattr(meta, "requires_live_broker", False):
            live_only.append(meta.name)
        if getattr(meta, "requires_volume", False):
            volume_required.append(meta.name)
    results.append(
        ProbeResult(
            "strategy evaluate methods present",
            not bad,
            f"bad={bad or 'none'}",
        )
    )
    results.append(
        ProbeResult(
            "live-only strategy inventory",
            "IVContraction" in live_only,
            f"live_only={live_only or 'none'} volume_required={volume_required or 'none'}",
            warn="IVContraction" not in live_only,
        )
    )

    recorder_targets = [
        meta.name for meta in STRATEGY_REGISTRY if hasattr(meta.instance, "set_recorder")
    ]
    results.append(
        ProbeResult(
            "OI/IV recorder injection targets",
            {"OIAnalysis", "IVContraction"}.issubset(set(recorder_targets)),
            f"targets={recorder_targets or 'none'}",
        )
    )
    return results


def _active_utility_probes() -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for item in UTILITY_IMPORTS:
        try:
            attr = _load_attr(item)
            results.append(ProbeResult(f"utility import {item}", attr is not None, "available"))
        except Exception as exc:
            results.append(ProbeResult(f"utility import {item}", False, f"{type(exc).__name__}: {exc}"))
    return results


def _call_with_timeout(fn, *args, timeout_seconds: float = 6.0, **kwargs):
    import concurrent.futures

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(fn, *args, **kwargs)
        return future.result(timeout=timeout_seconds)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _active_live_data_probes(timeout_seconds: float) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    try:
        from broker.factory import get_broker
        from config.settings import LIVE_TIMEFRAME
        from data.oi_recorder import OIRecorder
        from utils.option_utils import get_atm_strike, get_nearest_expiry
    except Exception as exc:
        return [ProbeResult("live probe imports", False, f"{type(exc).__name__}: {exc}")]

    try:
        broker = get_broker()
        broker_name = getattr(broker, "broker_name", broker.__class__.__name__)
        results.append(ProbeResult("broker factory", True, f"broker={broker_name}"))
    except Exception as exc:
        return [ProbeResult("broker factory", False, f"{type(exc).__name__}: {exc}")]

    nifty_ltp = 0.0
    try:
        nifty_ltp = float(_call_with_timeout(broker.get_ltp, "NIFTY", timeout_seconds=timeout_seconds) or 0.0)
        results.append(ProbeResult("NIFTY live LTP", nifty_ltp > 0, f"ltp={nifty_ltp:.2f}"))
    except Exception as exc:
        results.append(ProbeResult("NIFTY live LTP", False, f"{type(exc).__name__}: {exc}"))

    token_invalid = bool(getattr(broker, "_token_invalid", False))
    if token_invalid:
        results.append(
            ProbeResult(
                "broker option/OI auth",
                False,
                "Upstox token rejected; run `python3 scripts/upstox_auth.py` before expecting OI/IV",
            )
        )
    else:
        results.append(ProbeResult("broker option/OI auth", True, "token accepted or not required"))

    try:
        now = datetime.now()
        start = (now - timedelta(days=3)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        df = _call_with_timeout(
            broker.get_historical_data,
            "NIFTY",
            LIVE_TIMEFRAME,
            start,
            end,
            timeout_seconds=timeout_seconds,
        )
        latest = df.index[-1] if df is not None and not df.empty else "none"
        has_ohlcv = bool(
            df is not None
            and not df.empty
            and {"open", "high", "low", "close"}.issubset(set(df.columns))
        )
        results.append(
            ProbeResult(
                f"NIFTY {LIVE_TIMEFRAME} candles",
                has_ohlcv,
                f"rows={0 if df is None else len(df)} latest={latest}",
            )
        )
    except Exception as exc:
        results.append(ProbeResult(f"NIFTY {LIVE_TIMEFRAME} candles", False, f"{type(exc).__name__}: {exc}"))

    if nifty_ltp <= 0:
        results.append(ProbeResult("option chain OI/IV", False, "skipped because NIFTY LTP unavailable"))
        results.append(ProbeResult("OIRecorder live snapshot", False, "skipped because NIFTY LTP unavailable"))
        return results

    try:
        atm = get_atm_strike(nifty_ltp)
        expiry, _ = get_nearest_expiry(0, datetime.now())
        strikes = [atm - 50, atm, atm + 50]
        contracts = []
        for option_type in ("CE", "PE"):
            contracts.extend(
                _call_with_timeout(
                    broker.get_option_contracts,
                    "NIFTY",
                    expiry,
                    option_type,
                    strikes,
                    timeout_seconds=timeout_seconds,
                )
                or []
            )
        live_price = sum(1 for c in contracts if float(getattr(c, "last_price", 0.0) or 0.0) > 0)
        live_oi = sum(1 for c in contracts if int(getattr(c, "open_interest", 0) or 0) > 0)
        live_iv = sum(1 for c in contracts if float(getattr(c, "implied_volatility", 0.0) or 0.0) > 0)
        live_volume = sum(1 for c in contracts if int(getattr(c, "volume", 0) or 0) > 0)
        results.append(
            ProbeResult(
                "option chain OI/IV",
                bool(contracts) and live_price > 0 and (live_oi > 0 or live_iv > 0 or live_volume > 0),
                (
                    f"contracts={len(contracts)} ltp={live_price} "
                    f"oi={live_oi} iv={live_iv} volume={live_volume} expiry={expiry}"
                ),
                warn=bool(contracts) and live_price > 0,
            )
        )
    except Exception as exc:
        results.append(ProbeResult("option chain OI/IV", False, f"{type(exc).__name__}: {exc}", warn=True))

    try:
        recorder = OIRecorder(broker)
        snapshot = _call_with_timeout(recorder.record, nifty_ltp, timeout_seconds=timeout_seconds)
        if snapshot:
            oi_values = [
                float(v or 0.0)
                for k, v in snapshot.items()
                if k.endswith("_oi") and k not in {"total_ce_oi", "total_pe_oi"}
            ]
            iv_values = [float(v or 0.0) for k, v in snapshot.items() if k.endswith("_iv")]
            total_oi = float(snapshot.get("total_ce_oi", 0.0) or 0.0) + float(snapshot.get("total_pe_oi", 0.0) or 0.0)
            has_iv = any(v > 0 for v in iv_values)
            results.append(
                ProbeResult(
                    "OIRecorder live snapshot",
                    total_oi > 0 and has_iv,
                    f"total_oi={total_oi:.0f} iv_points={sum(1 for v in iv_values if v > 0)} pcr={snapshot.get('pcr')}",
                    warn=has_iv,
                )
            )
        else:
            results.append(ProbeResult("OIRecorder live snapshot", False, "record() returned no snapshot", warn=True))
    except Exception as exc:
        results.append(ProbeResult("OIRecorder live snapshot", False, f"{type(exc).__name__}: {exc}", warn=True))

    try:
        from data.option_volume import OptionVolumeRecorder

        volume_recorder = OptionVolumeRecorder(broker)
        volume_snapshot = _call_with_timeout(
            volume_recorder.record_snapshot,
            underlying_spot=nifty_ltp,
            timestamp=datetime.now(),
            timeout_seconds=timeout_seconds,
        )
        total_volume = float((volume_snapshot or {}).get("opt_total_volume", 0.0) or 0.0)
        atm_volume = float((volume_snapshot or {}).get("opt_atm_volume", 0.0) or 0.0)
        opt_oi = sum(
            float(v or 0.0)
            for k, v in (volume_snapshot or {}).items()
            if str(k).startswith("opt_oi_")
        )
        results.append(
            ProbeResult(
                "OptionVolumeRecorder live features",
                total_volume > 0 and opt_oi > 0,
                f"opt_total_volume={total_volume:.0f} opt_atm_volume={atm_volume:.0f} opt_oi_sum={opt_oi:.0f}",
                warn=total_volume > 0 or opt_oi > 0,
            )
        )
    except Exception as exc:
        results.append(ProbeResult("OptionVolumeRecorder live features", False, f"{type(exc).__name__}: {exc}", warn=True))

    return results


def _run_active_probes(enable_live: bool, timeout_seconds: float) -> list[ProbeResult]:
    results = []
    results.extend(_active_strategy_probes())
    results.extend(_active_utility_probes())
    if enable_live:
        results.extend(_active_live_data_probes(timeout_seconds))
    else:
        results.append(ProbeResult("live broker/OI/IV probes", True, "skipped by --no-live-probes", warn=True))
    return results


def _send_telegram_alert(message: str) -> None:
    try:
        from utils.telegram_notifier import get_notifier

        asyncio.run(get_notifier().send_text(message, target="LIVE", parse_mode=None))
    except Exception as exc:
        print(f"WARN   telegram alert failed: {type(exc).__name__}: {exc}")


def _health_alert_message(
    *,
    latest_market_ts: str,
    active_failures: list[ProbeResult],
    active_warnings: list[ProbeResult],
    nonzero_snapshots: list[tuple[int, int, str]],
    nonzero_opt_vol: list[int],
    broker_auth_failures: int,
    yfinance_fallbacks: int,
    raw_count: int,
    ml_approved: int,
    planner_passed: int,
    orders: int,
) -> tuple[bool, str]:
    issues: list[str] = []
    critical_names = {
        "NIFTY live LTP",
        "option chain OI/IV",
        "OIRecorder live snapshot",
        "OptionVolumeRecorder live features",
    }
    for result in active_failures:
        if result.name in critical_names or "broker" in result.name.lower():
            issues.append(f"{result.name}: {result.detail}")
    for result in active_warnings:
        if result.name in critical_names:
            issues.append(f"{result.name}: {result.detail}")
    if broker_auth_failures:
        issues.append(f"log auth failures={broker_auth_failures}")
    if not nonzero_snapshots and not nonzero_opt_vol:
        issues.append("live log has zero option-volume snapshots / opt_vol")
    if yfinance_fallbacks:
        issues.append(f"yfinance fallback used={yfinance_fallbacks}")

    should_alert = bool(issues)
    status = "ALERT" if should_alert else "OK"
    issue_text = "\n".join(f"- {item}" for item in issues[:8]) if issues else "- live broker/OI/IV/volume probes OK"
    message = (
        f"SignalForge Health {status}\n"
        f"Market TS: {latest_market_ts or 'unknown'}\n"
        f"Flow: raw={raw_count}, ML approved={ml_approved}, planner={planner_passed}, orders={orders}\n"
        f"{issue_text}"
    )
    return should_alert, message


def main() -> int:
    os.environ.setdefault("TELEGRAM_TARGET", "LIVE")
    parser = argparse.ArgumentParser(description="Check live SignalForge health from logs.")
    parser.add_argument("--log", default=str(_today_log()), help="SignalForge log path")
    parser.add_argument(
        "--full-day",
        action="store_true",
        help="Analyze the full log instead of only the latest process session",
    )
    parser.add_argument(
        "--no-active-probes",
        action="store_true",
        help="Only analyze logs; skip strategy, utility, broker, and OI/IV active probes",
    )
    parser.add_argument(
        "--no-live-probes",
        action="store_true",
        help="Run strategy/utility active probes but skip live broker and OI/IV calls",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=6.0,
        help="Timeout in seconds for each live broker probe",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Send Telegram alert when live data/OI/IV/volume health is degraded",
    )
    parser.add_argument(
        "--telegram-always",
        action="store_true",
        help="Send Telegram summary even when health is OK",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"FAIL   log not found: {log_path}")
        if args.telegram:
            _send_telegram_alert(
                f"*SignalForge Health ALERT*\n"
                f"Log not found: `{log_path}`\n"
                "Live process may not be running after market open."
            )
        return 2

    lines = log_path.read_text(errors="ignore").splitlines()
    start_line, session = (0, lines) if args.full_day else _last_session(lines)

    eligible = Counter()
    skipped = Counter()
    voted = Counter()
    raw_signal = Counter()
    setups = Counter()
    gates = Counter()
    vote_winners = Counter()
    setup_blocks_by_direction = Counter()
    inference_blocks_by_direction = Counter()
    raw_by_direction = Counter()
    ml_rejected_by_direction = Counter()
    ml_reject_reasons = Counter()
    planner_blocks = Counter()
    vote_cycles = 0
    eligibility_cycles = 0
    raw_count = 0
    ml_approved = 0
    ml_rejected = 0
    planner_passed = 0
    orders = 0
    handler_errors: list[str] = []
    option_snapshots: list[tuple[int, int, str]] = []
    opt_vol_values: list[int] = []
    option_chain_fallbacks = 0
    yfinance_fallbacks = 0
    broker_auth_failures = 0
    db_locked = 0
    latest_market_ts = ""

    for line in session:
        if "market_ts=" in line:
            match = re.search(r"market_ts=([^|]+?)(?:\s*\||$)", line)
            if match:
                latest_market_ts = match.group(1).strip()

        if "Handler error" in line or "Traceback" in line or "ERROR" in line:
            handler_errors.append(line)
        if "Fallback LTP via yfinance" in line or "Using yfinance fallback" in line:
            yfinance_fallbacks += 1
        if "token rejected" in line or "AUTHENTICATION WARNING" in line or "AUTHENTICATION FAILED" in line:
            broker_auth_failures += 1
        if "Using option-chain market data" in line:
            option_chain_fallbacks += 1
        if "database is locked" in line:
            db_locked += 1

        if "Option volume snapshot recorded" in line:
            match = re.search(r"total=(\d+)\s+\|\s+atm=(\d+)", line)
            if match:
                option_snapshots.append((int(match.group(1)), int(match.group(2)), line[:12].strip()))
        if "Strategy eligibility" in line:
            eligibility_cycles += 1
            match = re.search(r"eligible_names=(\[.*?\]) \| skipped=(\[.*?\]).*?opt_vol=(\d+)", line)
            if match:
                for name in _safe_list(match.group(1)):
                    eligible[name] += 1
                for item in _safe_list(match.group(2)):
                    skipped[item.split(":", 1)[0]] += 1
                opt_vol_values.append(int(match.group(3)))
        elif "CANDLES_READY" in line and "opt_vol=" in line:
            match = re.search(r"opt_vol=(\d+)", line)
            if match:
                opt_vol_values.append(int(match.group(1)))
        if "[VOTES]" in line:
            vote_cycles += 1
            for name in _extract_vote_lists(line):
                voted[name] += 1
            vote_match = re.search(
                r"CALL=(\d+)\[[^\]]*\].*?ws=([0-9.]+).*?PUT=(\d+)\[[^\]]*\].*?ws=([0-9.]+)",
                line,
            )
            if vote_match:
                call_n, call_ws, put_n, put_ws = vote_match.groups()
                call_ws_f = float(call_ws)
                put_ws_f = float(put_ws)
                if call_ws_f > put_ws_f:
                    vote_winners["BUY_CALL"] += 1
                elif put_ws_f > call_ws_f:
                    vote_winners["BUY_PUT"] += 1
                else:
                    vote_winners["TIE"] += 1
                if int(call_n) > 0:
                    vote_winners["CALL_PRESENT"] += 1
                if int(put_n) > 0:
                    vote_winners["PUT_PRESENT"] += 1
        if "Vote-aligned setup inference blocked" in line:
            match = re.search(r"dir=(BUY_[A-Z]+)", line)
            inference_blocks_by_direction[match.group(1) if match else "UNKNOWN"] += 1
        if "[BLOCK] no tradable setup after vote alignment" in line:
            match = re.search(r"vote=(BUY_[A-Z]+)", line)
            setup_blocks_by_direction[match.group(1) if match else "UNKNOWN"] += 1
        if "Setup | type=" in line:
            match = re.search(r"Setup \| type=([^|]+)\| dir=([^|]+)\|.*?source=([^|]+)", line)
            if match:
                key = " ".join(part.strip() for part in match.groups())
                setups[key] += 1
        if "Gate[" in line:
            match = re.search(r"Gate\[([^\]]+)\]", line)
            if match:
                gates[match.group(1)] += 1
        if "[RAW] RAW_SIGNAL" in line:
            raw_count += 1
            dir_match = re.search(r"RAW_SIGNAL .*?\|\s*(BUY_[A-Z]+)\s*\|", line)
            raw_by_direction[dir_match.group(1) if dir_match else "UNKNOWN"] += 1
            match = re.search(r"strategies=(\[[^\]]*\])", line)
            if match:
                for name in _safe_list(match.group(1)):
                    raw_signal[name] += 1
        if "MLFilterAgent" in line and "[OK] ML_RANK" in line:
            ml_approved += 1
        if "MLFilterAgent" in line and "[X] REJECTED" in line:
            ml_rejected += 1
            dir_match = re.search(r"direction=(BUY_[A-Z]+)", line)
            ml_rejected_by_direction[dir_match.group(1) if dir_match else "UNKNOWN"] += 1
            reason = line.split("|", maxsplit=3)[-1].strip()
            reason = re.sub(r"\s+", " ", reason)
            if reason:
                ml_reject_reasons[reason] += 1
        if "TradePlannerAgent" in line and "stage=trade_planning" in line and "status=passed" in line:
            planner_passed += 1
        if "TradePlannerAgent" in line and "No viable option contracts found" in line:
            planner_blocks["no_viable_contracts"] += 1
        if "TradePlannerAgent" in line and "Skipping weak entry" in line:
            reason = line.rsplit("|", maxsplit=1)[-1].strip()
            planner_blocks[f"weak_entry: {reason}" if reason else "weak_entry"] += 1
        if "TradePlannerAgent" in line and "Skipping late-session setup" in line:
            planner_blocks["late_session_entry_cutoff"] += 1
        if "TradePlannerAgent" in line and "Skipping post-cutoff setup" in line:
            planner_blocks["post_cutoff_entry"] += 1
        if "TradePlannerAgent" in line and "Skipping late-session low-momentum setup" in line:
            planner_blocks["late_session_low_momentum"] += 1
        if "TradePlannerAgent" in line and "stage=trade_planning" in line and "status=filtered" in line:
            match = re.search(r"reason=([^|]+)", line)
            planner_blocks[(match.group(1).strip() if match else "trade_planning_filtered")] += 1
        if "ORDER_DRY_RUN" in line or "ORDER_PLACED" in line:
            orders += 1

    nonzero_snapshots = [row for row in option_snapshots if row[0] > 0]
    nonzero_opt_vol = [value for value in opt_vol_values if value > 0]
    strategy_names = _registered_strategy_names()
    active_voters = [name for name in strategy_names if voted[name] > 0]
    silent = [name for name in strategy_names if eligible[name] > 0 and voted[name] == 0]
    active_probe_results: list[ProbeResult] = []
    if not args.no_active_probes:
        active_probe_results = _run_active_probes(
            enable_live=not args.no_live_probes,
            timeout_seconds=max(1.0, float(args.probe_timeout)),
        )
    active_failures = [result for result in active_probe_results if not result.ok and not result.warn]
    active_warnings = [result for result in active_probe_results if not result.ok and result.warn]

    print("SignalForge Live Health")
    print(f"log: {log_path}")
    print(f"scope: {'full day' if args.full_day else f'latest session from line {start_line + 1}'}")
    print(f"latest_market_ts: {latest_market_ts or 'unknown'}")
    print()

    print(_status("strategy eligibility cycles present", eligibility_cycles > 0))
    print(_status("vote cycles present", vote_cycles > 0))
    print(_status("RAW_SIGNAL flow present", raw_count > 0, warn=True))
    print(_status("no handler/planner errors", not handler_errors))
    print(_status("option volume snapshots non-zero", bool(nonzero_snapshots), warn=True))
    print(_status("candle payload opt_vol non-zero", bool(nonzero_opt_vol), warn=True))
    print(_status("no yfinance live fallback", yfinance_fallbacks == 0, warn=True))
    print(_status("broker auth valid in logs", broker_auth_failures == 0))
    print(_status("sqlite not locked", db_locked == 0, warn=True))
    print()

    if args.no_active_probes:
        print("Active Probes")
        print("  skipped by --no-active-probes")
        print()
    else:
        print("Active Probes")
        for result in active_probe_results:
            print(f"  {_probe_status(result)}")
        print()

    print("Flow")
    print(f"  eligibility_cycles : {eligibility_cycles}")
    print(f"  vote_cycles        : {vote_cycles}")
    print(f"  raw_signals        : {raw_count}")
    print(f"  ml_approved_events : {ml_approved}")
    print(f"  ml_rejected_events : {ml_rejected}")
    print(f"  planner_passed     : {planner_passed}")
    print(f"  order_events       : {orders}")
    print()

    print("Directional RCA")
    print(f"  vote_winners       : {dict(vote_winners) or 'none'}")
    print(f"  raw_by_direction   : {dict(raw_by_direction) or 'none'}")
    print(f"  ml_rejected_by_dir : {dict(ml_rejected_by_direction) or 'none'}")
    print(f"  setup_blocks_by_dir: {dict(setup_blocks_by_direction) or 'none'}")
    print(f"  inference_blocks   : {dict(inference_blocks_by_direction) or 'none'}")
    if vote_winners.get("BUY_PUT", 0) > raw_by_direction.get("BUY_PUT", 0):
        print("  likely_blocker     : PUT votes are present/winning but not converting to RAW_SIGNAL")
    elif raw_by_direction and raw_count > ml_approved:
        print("  likely_blocker     : RAW_SIGNAL generated but downstream ML/planner blocked")
    else:
        print("  likely_blocker     : no clear directional blocker from logs")
    print()

    if setups or gates or ml_reject_reasons or planner_blocks:
        print("Signal Path")
        if setups:
            print("  setups:")
            for key, count in setups.most_common():
                print(f"    {key}: {count}")
        if gates:
            print("  strategy_gates:")
            for key, count in gates.most_common():
                print(f"    {key}: {count}")
        if ml_reject_reasons:
            print("  ml_reject_reasons:")
            for key, count in ml_reject_reasons.most_common(8):
                print(f"    {key}: {count}")
        if planner_blocks:
            print("  planner_blocks:")
            for key, count in planner_blocks.most_common():
                print(f"    {key}: {count}")
        print()

    print("Data")
    last_snapshot = option_snapshots[-1] if option_snapshots else None
    print(f"  option_snapshots   : {len(option_snapshots)} ({len(nonzero_snapshots)} non-zero)")
    if last_snapshot:
        print(f"  last_option_volume : total={last_snapshot[0]} atm={last_snapshot[1]} at {last_snapshot[2]}")
    print(f"  opt_vol_values     : {len(opt_vol_values)} ({len(nonzero_opt_vol)} non-zero)")
    if opt_vol_values:
        print(f"  latest_opt_vol     : {opt_vol_values[-1]}")
    print(f"  option_chain_fallbacks: {option_chain_fallbacks}")
    print(f"  yfinance_fallbacks : {yfinance_fallbacks}")
    print(f"  broker_auth_failures: {broker_auth_failures}")
    print(f"  sqlite_locks       : {db_locked}")
    print()

    print("Strategy Contribution")
    print("  strategy          eligible skipped voted raw")
    for name in strategy_names:
        print(f"  {name:16} {eligible[name]:8d} {skipped[name]:7d} {voted[name]:5d} {raw_signal[name]:3d}")
    print()
    print(f"Active voters: {', '.join(active_voters) if active_voters else 'none'}")
    print(f"Eligible but silent: {', '.join(silent) if silent else 'none'}")
    if handler_errors:
        print()
        print("Recent errors:")
        for line in handler_errors[-5:]:
            print(f"  {line}")

    if args.telegram:
        should_alert, message = _health_alert_message(
            latest_market_ts=latest_market_ts,
            active_failures=active_failures,
            active_warnings=active_warnings,
            nonzero_snapshots=nonzero_snapshots,
            nonzero_opt_vol=nonzero_opt_vol,
            broker_auth_failures=broker_auth_failures,
            yfinance_fallbacks=yfinance_fallbacks,
            raw_count=raw_count,
            ml_approved=ml_approved,
            planner_passed=planner_passed,
            orders=orders,
        )
        if should_alert or args.telegram_always:
            _send_telegram_alert(message)

    if handler_errors:
        return 2
    if broker_auth_failures:
        return 2
    if active_failures:
        return 2
    if active_warnings:
        return 1
    if not nonzero_snapshots or yfinance_fallbacks > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

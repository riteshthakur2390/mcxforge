"""
utils/llm.py — Prompt builders + compatibility wrapper
"""
import asyncio
import re
import time
from typing import Union
from loguru import logger

from config.settings import (
    LLM_CONTEXT_CACHE_TTL_SEC,
    LLM_CONTEXT_MIN_PROB,
    LLM_CONTEXT_MIN_RANK,
    LLM_CONTEXT_TIMEOUT_SEC,
    LLM_ENABLED,
    TRADING_MODE,
)
from core.llm_router import TaskType, call_llm_async as router_call_llm_async

_LLM_CACHE: dict[str, tuple[float, str]] = {}

async def call_llm_async(
    prompt: str,
    *,
    task_type: TaskType = TaskType.GENERAL,
    max_tokens: int = 200,
    system: Union[str, None] = None,
    sensitive: Union[bool, None] = None,
) -> str:
    try:
        return await router_call_llm_async(
            prompt,
            task_type=task_type,
            max_tokens=max_tokens,
            system=system,
            sensitive=sensitive,
        )
    except Exception:
        return ""


async def call_llm_context_async(
    prompt: str,
    *,
    cache_key: str,
    task_type: TaskType = TaskType.GENERAL,
    max_tokens: int = 120,
    system: Union[str, None] = None,
    rank_score: float = 0.0,
    success_prob: float = 0.0,
    allow_in_backtest: bool = False,
    ttl_sec: int = LLM_CONTEXT_CACHE_TTL_SEC,
    timeout_sec: float = LLM_CONTEXT_TIMEOUT_SEC,
) -> str:
    if not LLM_ENABLED:
        return ""
    if TRADING_MODE.upper() == "BACKTEST" and not allow_in_backtest:
        return ""
    if rank_score < LLM_CONTEXT_MIN_RANK or success_prob < LLM_CONTEXT_MIN_PROB:
        return ""

    now = time.time()
    cached = _LLM_CACHE.get(cache_key)
    if cached and now - cached[0] <= max(ttl_sec, 1):
        return cached[1]

    try:
        text = await asyncio.wait_for(
            call_llm_async(
                prompt,
                task_type=task_type,
                max_tokens=max_tokens,
                system=system,
            ),
            timeout=max(timeout_sec, 0.5),
        )
    except asyncio.TimeoutError:
        logger.debug(f"[LLM] context timeout | key={cache_key} | task={task_type.value}")
        return ""
    except Exception:
        return ""

    if text:
        _LLM_CACHE[cache_key] = (now, text)
    return text


def build_signal_rationale_prompt(
    direction: str,
    strategies: list[str],
    confidence: float,
    votes: int,
    price: float,
    adx: float = 0,
    metadata: Union[dict, None] = None,
    symbol: str = "",
) -> str:
    sym = symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))
    meta_str = str(metadata or {})
    return f"""You are an MCX commodity trading analyst. Write a 2-sentence plain-English
explanation of why this signal fired. Be direct. No jargon.

Direction: {direction}
{sym} price: ₹{price:,.2f}
Strategies that agreed: {', '.join(strategies)} ({votes} votes)
Overall confidence: {confidence:.0%}
ADX: {adx:.1f}
Indicator details: {meta_str}

Sentence 1: What the price action is doing right now.
Sentence 2: Why this looks like a {direction.replace('BUY_', '')} opportunity."""


def build_eod_analysis_prompt(
    date: str,
    signals: int,
    suppressed: int,
    wins: int,
    losses: int,
    win_rate: float,
    journal_summary: str,
    symbol: str = "",
) -> str:
    sym = symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))
    return f"""You are an MCX commodity trading analyst reviewing today's MCXForge {sym} performance.

Date: {date}
Signals generated: {signals} | Suppressed (choppy/filtered): {suppressed}
Wins: {wins} | Losses: {losses} | Win rate: {win_rate:.1f}%

Signal log:
{journal_summary}

Write exactly 3 sentences:
1. Overall performance summary for today.
2. What worked or didn't (which strategies, what session/time of day).
3. One specific suggestion for tomorrow's session.
Return plain text only. No bullets. Each sentence must be complete and end with a period."""


def build_morning_outlook_prompt(
    bias: str, volatility_info: Union[float, str], gap_pct: float, symbol: str = ""
) -> str:
    sym = symbol or os.getenv("COMMODITY", os.getenv("INSTRUMENT", "SILVERM"))
    vol_str = f"{volatility_info:.1f}" if isinstance(volatility_info, (int, float)) else str(volatility_info)
    return f"""You are an MCX commodity market analyst. Write a 3-sentence morning outlook
for an MCX {sym} trader. Be direct, no fluff.

Pre-market bias: {bias}
Market Volatility / ATR: {vol_str}
Gap vs previous close: {gap_pct:+.2f}%

Sentence 1: What today's commodity indicators suggest for the session.
Sentence 2: Whether this looks like a trending or range-bound day.
Sentence 3: One actionable tip (e.g. 'wait for session breakout', 'strictly respect SL today').
Return plain text only. No bullets. Each sentence must be complete and end with a period."""


def normalize_llm_sentences(text: str, expected_sentences: int = 3) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""

    cleaned = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", cleaned)
    cleaned = re.sub(r"\s+(?:\d+[.)]\s*)", " ", cleaned)
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", cleaned) if p.strip()]

    if len(parts) < expected_sentences:
        fallback_parts = []
        for chunk in re.split(r"[;\n]+", cleaned):
            chunk = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", chunk).strip()
            if chunk:
                fallback_parts.append(chunk)
        if len(fallback_parts) > len(parts):
            parts = fallback_parts

    normalized: list[str] = []
    for part in parts[:expected_sentences]:
        part = re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", part).strip(" -")
        if not part:
            continue
        if part[-1] not in ".!?":
            part = f"{part}."
        normalized.append(part)

    if not normalized:
        return ""
    return " ".join(normalized)


def build_post_trade_prompt(
    option: str, pnl_pct: float, exit_reason: str, exit_premium: float
) -> str:
    result = "profitable" if pnl_pct > 0 else "a loss"
    return f"""Write a 2-sentence post-trade summary. Be factual and concise.

Option: {option}
Result: {result} | P&L: {pnl_pct:+.1f}%
Exit reason: {exit_reason} | Exit premium: ₹{exit_premium}

Sentence 1: State the outcome clearly (win/loss, P&L, reason).
Sentence 2: One brief observation about what happened."""


__all__ = [
    "TaskType",
    "call_llm_async",
    "call_llm_context_async",
    "build_signal_rationale_prompt",
    "build_eod_analysis_prompt",
    "build_morning_outlook_prompt",
    "build_post_trade_prompt",
    "normalize_llm_sentences",
]

"""
core/llm_router.py — Provider-agnostic LLM routing
"""

from __future__ import annotations
from config.settings.modules.system_thresholds import *

from dataclasses import dataclass
from enum import Enum
import asyncio
from typing import Any
from urllib.parse import urlparse

import httpx
from loguru import logger

from config.llm_cost_tracker import tracker
from config.settings import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_ENABLED,
    LLM_MAX_TOKENS,
    LLM_NON_SENSITIVE_PROVIDER,
    LLM_PROVIDER,
    LLM_REQUEST_TIMEOUT,
    LLM_SENSITIVE_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_FAST_MODEL,
    OLLAMA_MODEL,
    OLLAMA_SLOW_TASK_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    MAX_RETRIES,
    RETRY_DELAY
)


class TaskType(str, Enum):
    GENERAL = "general"
    SIGNAL_EXPLANATION = "signal_explanation"
    BACKTEST_NARRATION = "backtest_narration"
    INDICATOR_COMMENTARY = "indicator_commentary"
    NEWS_SENTIMENT = "news_sentiment"
    STRATEGY_DESCRIPTION = "strategy_description"
    PATTERN_SUMMARY = "pattern_summary"
    ALERT_TEXT = "alert_text"
    ML_EXPLANATION = "ml_explanation"
    MORNING_OUTLOOK = "morning_outlook"
    EOD_ANALYSIS = "eod_analysis"
    TRADE_SANITY = "trade_sanity"
    TRADE_NARRATIVE = "trade_narrative"
    POST_TRADE_SUMMARY = "post_trade_summary"
    PORTFOLIO_SUMMARY = "portfolio_summary"
    RISK_CHECK = "risk_check"
    ORDER_CHECK = "order_check"
    ORDER_EXECUTION = "order_execution"


SENSITIVE_TASKS = {
    TaskType.RISK_CHECK,
    TaskType.ORDER_CHECK,
    TaskType.ORDER_EXECUTION,
    TaskType.PORTFOLIO_SUMMARY,
}


@dataclass
class LLMResponse:
    text: str
    provider_used: str
    model_used: str
    task_type: TaskType
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] | None = None


def _is_sensitive(task_type: TaskType, explicit: bool | None) -> bool:
    if explicit is not None:
        return explicit
    return task_type in SENSITIVE_TASKS


def _provider_for(task_type: TaskType, sensitive: bool | None) -> str:
    if _is_sensitive(task_type, sensitive):
        return LLM_SENSITIVE_PROVIDER or LLM_PROVIDER
    return LLM_NON_SENSITIVE_PROVIDER or LLM_PROVIDER


def _model_for(provider: str) -> str:
    if provider == "ollama":
        return OLLAMA_MODEL
    if provider == "deepseek":
        return DEEPSEEK_MODEL
    if provider == "openai":
        return OPENAI_MODEL
    if provider == "anthropic":
        return ANTHROPIC_MODEL
    return ""


def _ollama_model_for_task(task_type: TaskType) -> str:
    if task_type in {
        TaskType.BACKTEST_NARRATION,
        TaskType.MORNING_OUTLOOK,
        TaskType.EOD_ANALYSIS,
        TaskType.POST_TRADE_SUMMARY,
        TaskType.GENERAL,
    }:
        return OLLAMA_FAST_MODEL or OLLAMA_MODEL
    return OLLAMA_MODEL


def _ollama_candidate_urls() -> list[str]:
    urls = [OLLAMA_BASE_URL.rstrip("/")]
    parsed = urlparse(OLLAMA_BASE_URL)
    if parsed.hostname in {"127.0.0.1", "localhost"}:
        alt_host = "host.docker.internal"
        alt = f"{parsed.scheme or 'http'}://{alt_host}"
        if parsed.port:
            alt = f"{alt}:{parsed.port}"
        if alt not in urls:
            urls.append(alt)
    return urls


def get_llm_status() -> dict[str, Any]:
    default_provider = LLM_PROVIDER
    non_sensitive_provider = LLM_NON_SENSITIVE_PROVIDER or default_provider
    sensitive_provider = LLM_SENSITIVE_PROVIDER or default_provider
    providers = {
        "ollama": {
            "configured": _configured("ollama"),
            "model": OLLAMA_MODEL,
            "fast_model": OLLAMA_FAST_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "candidate_urls": _ollama_candidate_urls(),
            "default_timeout": LLM_REQUEST_TIMEOUT,
            "slow_task_timeout": OLLAMA_SLOW_TASK_TIMEOUT,
        },
        "deepseek": {
            "configured": _configured("deepseek"),
            "model": DEEPSEEK_MODEL,
            "base_url": DEEPSEEK_BASE_URL,
        },
        "openai": {
            "configured": _configured("openai"),
            "model": OPENAI_MODEL,
            "base_url": OPENAI_BASE_URL,
        },
        "anthropic": {
            "configured": _configured("anthropic"),
            "model": ANTHROPIC_MODEL,
            "base_url": "https://api.anthropic.com",
        },
    }
    active_provider = non_sensitive_provider
    return {
        "enabled": LLM_ENABLED,
        "default_provider": default_provider,
        "active_provider": active_provider,
        "active_model": _model_for(active_provider),
        "non_sensitive_provider": non_sensitive_provider,
        "sensitive_provider": sensitive_provider,
        "providers": providers,
    }


def _configured(provider: str) -> bool:
    if provider == "ollama":
        return bool(OLLAMA_BASE_URL and OLLAMA_MODEL)
    if provider == "deepseek":
        return bool(DEEPSEEK_API_KEY and DEEPSEEK_MODEL)
    if provider == "openai":
        return bool(OPENAI_API_KEY and OPENAI_MODEL)
    if provider == "anthropic":
        return bool(ANTHROPIC_API_KEY and ANTHROPIC_MODEL)
    return False


def _compose_prompt(prompt: str, system: str | None) -> str:
    if not system:
        return prompt
    return f"{system.strip()}\n\n{prompt.strip()}"


def _timeout_for_task(task_type: TaskType) -> float:
    if task_type in {
        TaskType.BACKTEST_NARRATION,
        TaskType.MORNING_OUTLOOK,
        TaskType.EOD_ANALYSIS,
        TaskType.POST_TRADE_SUMMARY,
    }:
        return max(LLM_REQUEST_TIMEOUT, OLLAMA_SLOW_TASK_TIMEOUT)
    return LLM_REQUEST_TIMEOUT


def _ollama_model_candidates(task_type: TaskType) -> list[str]:
    primary = _ollama_model_for_task(task_type)
    candidates = [primary]
    if OLLAMA_FAST_MODEL and OLLAMA_FAST_MODEL not in candidates:
        candidates.append(OLLAMA_FAST_MODEL)
    if OLLAMA_MODEL and OLLAMA_MODEL not in candidates:
        candidates.append(OLLAMA_MODEL)
    return candidates


async def _call_ollama(
    prompt: str,
    system: str | None,
    max_tokens: int,
    task_type: TaskType,
) -> LLMResponse:
    prompt_text = _compose_prompt(prompt, system)
    read_timeout = _timeout_for_task(task_type)

    timeout = httpx.Timeout(
        connect=CORE_LLM_CONNECT_TIMEOUT,
        read=read_timeout,
        write=CORE_LLM_WRITE_TIMEOUT,
        pool=CORE_LLM_POOL_TIMEOUT
    )

    last_error: Exception | None = None

    for base_url in _ollama_candidate_urls():
        for model in _ollama_model_candidates(task_type):

            token_budget = min(
                max_tokens,
                96 if model == OLLAMA_FAST_MODEL else max_tokens
            )

            payload = {
                "model": model,
                "prompt": prompt_text,
                "stream": False,
                "options": {"num_predict": token_budget},
            }

            # 🔁 Retry loop
            for attempt in range(MAX_RETRIES):
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        resp = await client.post(
                            f"{base_url}/api/generate",
                            json=payload
                        )
                        resp.raise_for_status()
                        data = resp.json()

                    return LLMResponse(
                        text=(data.get("response") or "").strip(),
                        provider_used="ollama",
                        model_used=model,
                        task_type=task_type,
                        input_tokens=int(data.get("prompt_eval_count") or 0),
                        output_tokens=int(data.get("eval_count") or 0),
                        raw={**data, "_base_url": base_url},
                    )

                except httpx.ReadTimeout as exc:
                    last_error = exc

                    # retry only if attempts left
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    else:
                        break  # move to next model

                except Exception as exc:
                    last_error = exc
                    break  # don't retry unknown errors

    # ❗ moved OUTSIDE loops (important fix)
    if last_error:
        raise last_error

    raise RuntimeError("No Ollama base URL candidates available")


async def probe_llm_provider() -> dict[str, Any]:
    status = get_llm_status()
    provider = status["active_provider"]
    if not status.get("enabled"):
        return {"ok": False, "provider": provider, "error": "LLM is disabled"}

    if provider == "ollama":
        errors: list[str] = []
        for base_url in _ollama_candidate_urls():
            try:
                async with httpx.AsyncClient(timeout=min(LLM_REQUEST_TIMEOUT, 5.0)) as client:
                    resp = await client.get(f"{base_url}/api/tags")
                    resp.raise_for_status()
                    data = resp.json()
                return {
                    "ok": True,
                    "provider": provider,
                    "model": _model_for(provider),
                    "fast_model": OLLAMA_FAST_MODEL,
                    "base_url": base_url,
                    "models": [m.get("name") for m in data.get("models", [])],
                }
            except Exception as exc:
                errors.append(f"{base_url}: {exc}")
        return {
            "ok": False,
            "provider": provider,
            "model": _model_for(provider),
            "fast_model": OLLAMA_FAST_MODEL,
            "base_urls": _ollama_candidate_urls(),
            "error": "; ".join(errors) or "Ollama probe failed",
        }

    return {
        "ok": status["providers"].get(provider, {}).get("configured", False),
        "provider": provider,
        "model": _model_for(provider),
        "note": "Probe is implemented only for Ollama. Other providers are config-only here.",
    }


async def _call_openai_compatible(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    task_type: TaskType,
) -> LLMResponse:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT) as client:
        resp = await client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    choice = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    usage = data.get("usage") or {}
    return LLMResponse(
        text=choice.strip(),
        provider_used=provider,
        model_used=model,
        task_type=task_type,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        raw=data,
    )


async def _call_anthropic(
    prompt: str,
    system: str | None,
    max_tokens: int,
    task_type: TaskType,
) -> LLMResponse:
    payload: dict[str, Any] = {
        "model": _model_for("anthropic"),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT) as client:
        resp = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data.get("content") or []
    text = content[0].get("text", "").strip() if content else ""
    usage = data.get("usage") or {}
    return LLMResponse(
        text=text,
        provider_used="anthropic",
        model_used=_model_for("anthropic"),
        task_type=task_type,
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        raw=data,
    )


async def call_llm_response_async(
    prompt: str,
    *,
    task_type: TaskType = TaskType.GENERAL,
    max_tokens: int = LLM_MAX_TOKENS,
    system: str | None = None,
    sensitive: bool | None = None,
) -> LLMResponse | None:
    if not LLM_ENABLED:
        return None

    provider = _provider_for(task_type, sensitive)
    if not _configured(provider):
        logger.debug(f"[LLM] Provider {provider!r} not configured for task={task_type.value}")
        return None

    try:
        if provider == "ollama":
            response = await _call_ollama(prompt, system, max_tokens, task_type)
        elif provider == "deepseek":
            response = await _call_openai_compatible(
                provider="deepseek",
                base_url=DEEPSEEK_BASE_URL,
                api_key=DEEPSEEK_API_KEY,
                model=_model_for("deepseek"),
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                task_type=task_type,
            )
        elif provider == "openai":
            response = await _call_openai_compatible(
                provider="openai",
                base_url=OPENAI_BASE_URL,
                api_key=OPENAI_API_KEY,
                model=_model_for("openai"),
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                task_type=task_type,
            )
        elif provider == "anthropic":
            response = await _call_anthropic(prompt, system, max_tokens, task_type)
        else:
            logger.warning(f"[LLM] Unsupported provider {provider!r}")
            return None
    except Exception as exc:
        logger.debug(
            f"[LLM] Call failed | provider={provider} | task={task_type.value} | "
            f"{type(exc).__name__}: {exc!r}"
        )
        return None

    tracker.record(response)
    return response


async def call_llm_async(
    prompt: str,
    *,
    task_type: TaskType = TaskType.GENERAL,
    max_tokens: int = LLM_MAX_TOKENS,
    system: str | None = None,
    sensitive: bool | None = None,
) -> str:
    response = await call_llm_response_async(
        prompt,
        task_type=task_type,
        max_tokens=max_tokens,
        system=system,
        sensitive=sensitive,
    )
    return response.text if response else ""

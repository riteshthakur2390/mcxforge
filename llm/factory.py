"""
llm/factory.py — LLM Provider Factory
=======================================
NEW FILE. Single entry point for all LLM usage in SignalForge.

This is the file imported by agent3_ml/filter.py:
    from llm.factory import get_llm
    llm = get_llm()

Provider selection (in priority order):
    1. LLM_PROVIDER=disabled   → DisabledLLM  (no calls, zero latency)
    2. LLM_PROVIDER=ollama     → OllamaLLM    (local, free, private)
    3. LLM_PROVIDER=anthropic  → AnthropicLLM (cloud, needs API key)
    4. ANTHROPIC_API_KEY set   → AnthropicLLM (auto-detect)
    5. Nothing configured      → DisabledLLM  (safe default)

Environment variables (.env):
    LLM_PROVIDER=disabled     # turn off completely
    LLM_PROVIDER=ollama       # use local Ollama
    LLM_PROVIDER=anthropic    # use Claude API
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=llama3
    ANTHROPIC_API_KEY=sk-ant-...

The factory is a singleton — same instance returned on every call.
"""

import os
from loguru import logger

from llm.base          import BaseLLMProvider
from llm.disabled_llm  import DisabledLLM
from llm.anthropic_llm import AnthropicLLM
from llm.ollama_llm    import OllamaLLM

# NEW: module-level singleton — built once on first call
_instance: BaseLLMProvider | None = None


def get_llm() -> BaseLLMProvider:
    """
    Return the configured LLM provider instance (singleton).

    Usage:
        from llm.factory import get_llm
        llm  = get_llm()

        # Guard before using
        if llm.provider_name == "disabled":
            return

        resp = await llm.complete("your prompt", max_tokens=120)
        if resp:
            print(resp.text)      # response text
            print(resp.provider)  # "anthropic" | "ollama" | "disabled"
    """
    global _instance

    # NEW: return cached instance if already built
    if _instance is not None:
        return _instance

    _instance = _build_provider()
    logger.info(f"[LLMFactory] Provider: {_instance.provider_name}")
    return _instance


def _build_provider() -> BaseLLMProvider:
    """
    NEW: Build the appropriate LLM provider from environment config.
    Called exactly once — result is cached in _instance.
    """
    # Read configuration from environment
    provider    = os.getenv("LLM_PROVIDER", "").lower().strip()
    api_key     = os.getenv("ANTHROPIC_API_KEY", "")
    ollama_url  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    # Also check settings.py for LLM_ENABLED flag
    try:
        from config.settings import LLM_ENABLED, LLM_MODEL, LLM_MAX_TOKENS, ANTHROPIC_API_KEY
        if not LLM_ENABLED:
            logger.debug("[LLMFactory] LLM_ENABLED=False → DisabledLLM")
            return DisabledLLM()
        # Settings takes precedence for API key
        if ANTHROPIC_API_KEY and not api_key:
            api_key = ANTHROPIC_API_KEY
    except ImportError:
        LLM_MODEL    = "claude-haiku-4-5-20251001"
        LLM_MAX_TOKENS = 200

    # ── Priority selection ────────────────────────────────────────────────────

    # 1. Explicit disable
    if provider == "disabled":
        logger.debug("[LLMFactory] LLM_PROVIDER=disabled → DisabledLLM")
        return DisabledLLM()

    # 2. Explicit Ollama
    if provider == "ollama":
        logger.info(f"[LLMFactory] Using Ollama at {ollama_url} | model={ollama_model}")
        return OllamaLLM(base_url=ollama_url, model=ollama_model)

    # 3. Explicit Anthropic
    if provider == "anthropic":
        if not api_key:
            logger.warning("[LLMFactory] LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY not set → DisabledLLM")
            return DisabledLLM()
        logger.info(f"[LLMFactory] Using Anthropic | model={LLM_MODEL}")
        return AnthropicLLM(api_key=api_key, model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS)

    # 4. Auto-detect: API key present → use Anthropic
    if api_key:
        logger.info(f"[LLMFactory] Auto-detected Anthropic API key | model={LLM_MODEL}")
        return AnthropicLLM(api_key=api_key, model=LLM_MODEL, max_tokens=LLM_MAX_TOKENS)

    # 5. Nothing configured → safe default
    logger.debug("[LLMFactory] No LLM configured → DisabledLLM")
    return DisabledLLM()


def reset_llm() -> None:
    """
    NEW: Force re-build of the LLM provider on next get_llm() call.
    Useful in tests or when config changes at runtime.
    """
    global _instance
    _instance = None

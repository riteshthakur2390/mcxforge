"""
llm/base.py — LLM Provider Base Class
=======================================
NEW FILE. Defines the interface that every LLM provider must implement.

All providers return LLMResponse so callers are provider-agnostic.
Usage:
    from llm.factory import get_llm
    llm  = get_llm()
    resp = await llm.complete("your prompt", max_tokens=120)
    if resp:
        print(resp.text)        # the response text
        print(resp.provider)    # "anthropic" | "ollama" | "disabled"
"""

from dataclasses import dataclass
from abc import ABC, abstractmethod


# ── Response container ────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Unified response object returned by every LLM provider.
    Always check `if resp:` before accessing fields — returns None on failure.
    """
    text:     str    # The generated text
    provider: str    # Which provider produced this response
    tokens:   int    # Approximate token count (0 if unknown)

    def __bool__(self) -> bool:
        # Truthy only when there is actual text content
        return bool(self.text and self.text.strip())


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    All providers (Anthropic, Ollama, Disabled) inherit from this.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Short identifier for this provider.
        Examples: "anthropic", "ollama", "disabled"
        Used by callers to branch on provider type:
            if llm.provider_name == "disabled": return
        """
        ...

    @abstractmethod
    async def complete(
        self,
        prompt:     str,
        max_tokens: int = 200,
    ) -> LLMResponse | None:
        """
        Send a prompt and return LLMResponse, or None on failure.
        Must be non-blocking — any error returns None gracefully.
        Never raises exceptions — the trading system must continue.
        """
        ...

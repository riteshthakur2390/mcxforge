"""
llm/anthropic_llm.py — Anthropic Claude Provider
==================================================
NEW FILE. Wraps Anthropic API calls using the same interface
as existing utils/llm.py but returns LLMResponse objects.

Provider name: "anthropic"
Used when: ANTHROPIC_API_KEY is set in .env
Model:     LLM_MODEL from settings (default: claude-haiku-4-5-20251001)
"""

import httpx
from loguru import logger

from llm.base import BaseLLMProvider, LLMResponse


class AnthropicLLM(BaseLLMProvider):
    """
    Anthropic Claude provider.
    Uses the same HTTP approach as utils/llm.py to avoid new dependencies.
    """

    # NEW: Anthropic API endpoint
    _API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str, max_tokens: int = 200) -> None:
        # NEW: store credentials — never log these
        self._api_key   = api_key
        self._model     = model
        self._max_tokens = max_tokens

    @property
    def provider_name(self) -> str:
        # NEW: identifier for this provider
        return "anthropic"

    async def complete(
        self,
        prompt:     str,
        max_tokens: int = 200,
    ) -> LLMResponse | None:
        """
        Send prompt to Claude and return LLMResponse.
        Returns None on any error — never raises.
        """
        # NEW: guard — don't call API if no key configured
        if not self._api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.post(
                    self._API_URL,
                    headers={
                        "x-api-key":         self._api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type":      "application/json",
                    },
                    json={
                        "model":      self._model,
                        "max_tokens": max_tokens or self._max_tokens,
                        "messages":   [{"role": "user", "content": prompt}],
                    },
                )
                data = resp.json()

                # NEW: extract text from Anthropic response format
                if "content" in data and data["content"]:
                    text   = data["content"][0]["text"].strip()
                    tokens = data.get("usage", {}).get("output_tokens", 0)
                    return LLMResponse(text=text, provider="anthropic", tokens=tokens)

                logger.debug(f"[AnthropicLLM] Unexpected response shape: {list(data.keys())}")
                return None

        except Exception as e:
            # NEW: all errors are non-fatal — system continues without LLM
            logger.debug(f"[AnthropicLLM] Call failed (non-critical): {e}")
            return None

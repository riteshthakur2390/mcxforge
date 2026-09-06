"""
llm/disabled_llm.py — Disabled LLM Provider
=============================================
NEW FILE. Used when LLM_ENABLED=False or no API key is set.
Returns None immediately without any network call.
Callers check: if llm.provider_name == "disabled": return
"""

from llm.base import BaseLLMProvider, LLMResponse


class DisabledLLM(BaseLLMProvider):
    """
    No-op provider. Used when LLM is intentionally disabled.
    All calls return None instantly — zero latency, zero cost.
    """

    @property
    def provider_name(self) -> str:
        # NEW: identifier checked by callers to skip LLM processing
        return "disabled"

    async def complete(
        self,
        prompt:     str,
        max_tokens: int = 200,
    ) -> LLMResponse | None:
        # NEW: immediately return None — no API call made
        return None

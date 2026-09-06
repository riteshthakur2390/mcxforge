"""
llm/ollama_llm.py — Ollama Local LLM Provider
===============================================
NEW FILE. Connects to a locally running Ollama server.
Use this for: free inference, private data, no API key needed.

Provider name: "ollama"
Used when:     LLM_PROVIDER=ollama in .env
Default URL:   http://localhost:11434
Default model: llama3 (configurable)

Setup:
    brew install ollama         (macOS)
    ollama pull llama3          (download model)
    ollama serve                (start server)
"""

import httpx
from loguru import logger

from llm.base import BaseLLMProvider, LLMResponse


class OllamaLLM(BaseLLMProvider):
    """
    Ollama local LLM provider.
    Talks to the Ollama REST API running on localhost.
    """

    # NEW: Ollama generate endpoint
    _GENERATE_URL = "/api/generate"

    def __init__(
        self,
        base_url: str  = "http://localhost:11434",
        model:    str  = "llama3",
    ) -> None:
        # NEW: store connection details
        self._base_url = base_url.rstrip("/")
        self._model    = model

    @property
    def provider_name(self) -> str:
        # NEW: identifier for this provider
        return "ollama"

    async def complete(
        self,
        prompt:     str,
        max_tokens: int = 200,
    ) -> LLMResponse | None:
        """
        Send prompt to local Ollama and return LLMResponse.
        Returns None on any error — never raises.
        Short timeout (8s) so it doesn't delay signal generation.
        """
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.post(
                    f"{self._base_url}{self._GENERATE_URL}",
                    json={
                        "model":  self._model,
                        "prompt": prompt,
                        "stream": False,
                        # NEW: limit response length
                        "options": {"num_predict": max_tokens},
                    },
                )
                data = resp.json()

                # NEW: extract text from Ollama response format
                text = data.get("response", "").strip()
                if text:
                    return LLMResponse(text=text, provider="ollama", tokens=0)

                logger.debug(f"[OllamaLLM] Empty response from {self._model}")
                return None

        except httpx.ConnectError:
            # NEW: Ollama not running — silent fail, system continues
            logger.debug(f"[OllamaLLM] Ollama not running at {self._base_url}")
            return None
        except Exception as e:
            logger.debug(f"[OllamaLLM] Call failed (non-critical): {e}")
            return None

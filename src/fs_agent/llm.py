"""LLM client abstractions used by agents."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import httpx

from .logger import get_logger

logger = get_logger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str) -> None:
        self.model = model
        # --- Token tracking for benchmarking ---
        self._call_count: int = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._total_tokens: int = 0

    def _record_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        """Accumulate token counts from a single generate() call."""
        self._call_count += 1
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._total_tokens += total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)

    @property
    def usage_stats(self) -> dict[str, int]:
        """Return accumulated token usage across all calls."""
        return {
            "call_count": self._call_count,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
        }

    def reset_usage(self) -> None:
        """Reset accumulated counters (useful between benchmark runs)."""
        self._call_count = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        """Return generated text for the provided prompt."""


class DummyLLMClient(BaseLLMClient):
    """Fallback that returns deterministic placeholder output."""

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        head = prompt.strip().splitlines()
        preview = " \n".join(head[:10])
        result = (
            "// LLM output unavailable in this environment.\n"
            "// Provide the following prompt to a real model for richer output.\n"
            + preview
        )
        # Rough estimate: 1 token ≈ 4 chars
        est_prompt = len(prompt) // 4
        est_system = len(system) // 4 if system else 0
        est_completion = len(result) // 4
        self._record_usage(
            prompt_tokens=est_prompt + est_system,
            completion_tokens=est_completion,
        )
        return result


class OpenAILLMClient(BaseLLMClient):
    """Minimal OpenAI Chat Completions client."""

    def __init__(self, api_key: str, model: str, *, timeout: float = 500.0) -> None:
        super().__init__(model)
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        payload = response.json()

        # Record token usage reported by the API
        usage = payload.get("usage", {})
        self._record_usage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens"),
        )

        try:
            return payload["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Malformed OpenAI response: {payload}") from exc


def build_llm_client(provider: str, *, model: str, api_key: str | None) -> BaseLLMClient:
    """Factory helper based on provider settings."""

    provider_lower = provider.lower()
    if provider_lower == "openai":
        if not api_key:
            logger.warning(
                "OpenAI provider selected but no API key supplied; falling back to dummy LLM"
            )
            return DummyLLMClient(model)
        return OpenAILLMClient(api_key=api_key, model=model)
    if provider_lower not in {"dummy", "auto"}:
        logger.warning("Unknown LLM provider '%s'; defaulting to dummy client", provider)
    return DummyLLMClient(model)


def default_provider_from_env() -> str:
    return os.getenv("FS_AGENT_LLM_PROVIDER", "dummy")

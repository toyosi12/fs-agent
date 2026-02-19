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

    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        """Return generated text for the provided prompt."""


class DummyLLMClient(BaseLLMClient):
    """Fallback that returns deterministic placeholder output."""

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        head = prompt.strip().splitlines()
        preview = " \n".join(head[:10])
        return (
            "// LLM output unavailable in this environment.\n"
            "// Provide the following prompt to a real model for richer output.\n"
            + preview
        )


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

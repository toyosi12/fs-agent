"""LLM client abstractions used by agents."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from openai import OpenAI

from .logger import get_logger

logger = get_logger(__name__)

# Regex to strip Qwen3-style <think>...</think> reasoning blocks.
# These appear at the start of the response before the actual content.
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


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

    def generate_with_images(
        self,
        prompt: str,
        images_b64: list[str],
        *,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate text with one or more base64-encoded images.

        Subclasses that support vision should override this.
        The default falls back to text-only ``generate()``.
        """
        return self.generate(prompt, system=system, temperature=temperature)


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


# Well-known base URLs for OpenAI-compatible providers.
OPENAI_BASE_URL = "https://api.openai.com/v1"
DASHSCOPE_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OpenAILLMClient(BaseLLMClient):
    """OpenAI-compatible Chat Completions client (uses the official ``openai`` SDK).

    Works with any provider that exposes an OpenAI-compatible API
    (OpenAI, Alibaba DashScope / Qwen, etc.).
    Pass *base_url* to point the SDK at a different host.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = OPENAI_BASE_URL,
        timeout: float = 500.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.2) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        # Record token usage reported by the API
        if completion.usage:
            self._record_usage(
                prompt_tokens=completion.usage.prompt_tokens or 0,
                completion_tokens=completion.usage.completion_tokens or 0,
                total_tokens=completion.usage.total_tokens,
            )

        content = completion.choices[0].message.content
        if content is None:  # pragma: no cover - defensive
            raise RuntimeError(f"Empty response from model: {completion}")

        # Strip Qwen3 <think>...</think> reasoning blocks so callers
        # receive only the actionable response content.
        cleaned = _THINK_PATTERN.sub("", content).strip()
        if not cleaned:
            # Model returned only thinking with no actual response
            logger.warning("Model returned only <think> content; raw length=%d", len(content))
            raise RuntimeError(f"Empty response after stripping <think> block (raw length={len(content)})")
        return cleaned

    def generate_with_images(
        self,
        prompt: str,
        images_b64: list[str],
        *,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate text with base64-encoded PNG images (GPT-4o vision)."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})

        # Build multi-part user content: text + images
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img_b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high",
                },
            })
        messages.append({"role": "user", "content": content})

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise RuntimeError(f"OpenAI vision request failed: {exc}") from exc

        if completion.usage:
            self._record_usage(
                prompt_tokens=completion.usage.prompt_tokens or 0,
                completion_tokens=completion.usage.completion_tokens or 0,
                total_tokens=completion.usage.total_tokens,
            )

        result = completion.choices[0].message.content
        if result is None:  # pragma: no cover - defensive
            raise RuntimeError(f"Empty vision response from model: {completion}")

        cleaned = _THINK_PATTERN.sub("", result).strip()
        if not cleaned:
            logger.warning("Vision model returned only <think> content; raw length=%d", len(result))
            raise RuntimeError(f"Empty vision response after stripping <think> block (raw length={len(result)})")
        return cleaned


def build_llm_client(
    provider: str,
    *,
    model: str,
    api_key: str | None,
    base_url: str | None = None,
) -> BaseLLMClient:
    """Factory helper based on provider settings.

    Supported providers:
    - ``openai``  – targets *base_url* (default ``https://api.openai.com/v1``).
    - ``qwen``    – targets Alibaba DashScope International's OpenAI-compatible
                    gateway (``https://coding-intl.dashscope.aliyuncs.com/v1``).
    - ``ollama``  – targets local Ollama instance (``http://localhost:11434/v1``).
    - ``dummy``   – deterministic placeholder (no network calls).

    An explicit *base_url* always takes precedence over the provider default.
    """

    provider_lower = provider.lower()

    if provider_lower in ("openai", "qwen", "ollama"):
        if not api_key and provider_lower != "ollama":
            logger.warning(
                "%s provider selected but no API key supplied; falling back to dummy LLM",
                provider,
            )
            return DummyLLMClient(model)

        if base_url:
            effective_url = base_url
        elif provider_lower == "qwen":
            effective_url = DASHSCOPE_BASE_URL
        elif provider_lower == "ollama":
            effective_url = OLLAMA_BASE_URL
        else:
            effective_url = OPENAI_BASE_URL

        return OpenAILLMClient(
            api_key=api_key or "ollama",  # Ollama doesn't need real API key
            model=model,
            base_url=effective_url,
        )

    if provider_lower not in {"dummy", "auto"}:
        logger.warning("Unknown LLM provider '%s'; defaulting to dummy client", provider)
    return DummyLLMClient(model)


def default_provider_from_env() -> str:
    return os.getenv("FS_AGENT_LLM_PROVIDER", "dummy")


def build_llm_clients_from_env(
    *,
    default_provider: str = "",
    default_model: str = "",
    default_api_key: str | None = None,
    default_base_url: str | None = None,
) -> tuple[BaseLLMClient, dict[str, BaseLLMClient]]:
    """Build a shared LLM client plus per-role overrides from environment variables.

    This is the single source of truth for constructing the full set of LLM
    clients.  Both the orchestrator and the benchmark runner should call this.

    Global env vars (fallback for every role):
    - ``FS_AGENT_LLM_PROVIDER``
    - ``FS_AGENT_LLM_MODEL``
    - ``FS_AGENT_LLM_BASE_URL``
    - ``FS_AGENT_OPENAI_API_KEY``

    Per-role overrides (``<ROLE>`` = ``ARCHITECT | BACKEND | FRONTEND | INFRA``):
    - ``FS_AGENT_LLM_PROVIDER_<ROLE>``
    - ``FS_AGENT_LLM_MODEL_<ROLE>``
    - ``FS_AGENT_LLM_BASE_URL_<ROLE>``
    - ``FS_AGENT_OPENAI_API_KEY_<ROLE>``

    Returns ``(base_client, {role: client, ...})``.
    """

    eff_provider = default_provider or os.getenv("FS_AGENT_LLM_PROVIDER", "dummy")
    eff_model = default_model or os.getenv("FS_AGENT_LLM_MODEL", "gpt-4o-mini")
    eff_api_key = default_api_key or os.getenv("FS_AGENT_OPENAI_API_KEY")
    eff_base_url = default_base_url or os.getenv("FS_AGENT_LLM_BASE_URL")

    try:
        base_llm = build_llm_client(
            eff_provider,
            model=eff_model,
            api_key=eff_api_key,
            base_url=eff_base_url,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("LLM client creation failed: %s; defaulting to dummy", exc)
        base_llm = build_llm_client("dummy", model=eff_model, api_key=None)

    llm_per_role: dict[str, BaseLLMClient] = {}

    for role in ("architect", "backend", "frontend", "infra"):
        prefix = role.upper()
        role_provider = os.getenv(f"FS_AGENT_LLM_PROVIDER_{prefix}")
        role_model = os.getenv(f"FS_AGENT_LLM_MODEL_{prefix}")
        role_base_url = os.getenv(f"FS_AGENT_LLM_BASE_URL_{prefix}")
        role_api_key = os.getenv(f"FS_AGENT_OPENAI_API_KEY_{prefix}") or eff_api_key

        # Nothing overridden — role will use the base client via get_llm().
        if not role_provider and not role_model and not role_base_url and role_api_key == eff_api_key:
            continue

        r_provider = role_provider or eff_provider
        r_model = role_model or eff_model
        r_base_url = role_base_url or eff_base_url

        # Matches the base client — just reuse it.
        if (
            r_provider == eff_provider
            and r_model == eff_model
            and r_base_url == eff_base_url
            and role_api_key == eff_api_key
        ):
            llm_per_role[role] = base_llm
            continue

        try:
            llm_per_role[role] = build_llm_client(
                r_provider,
                model=r_model,
                api_key=role_api_key,
                base_url=r_base_url,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "LLM client creation failed for role %s: %s; using base client",
                role,
                exc,
            )
            llm_per_role[role] = base_llm

    return base_llm, llm_per_role

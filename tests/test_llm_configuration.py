"""Tests for environment-driven LLM client construction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from fs_agent import llm


def test_build_llm_clients_accepts_default_api_key(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_llm_client(provider, *, model, api_key, base_url=None):
        captured.update(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        return llm.DummyLLMClient(model)

    monkeypatch.setattr(llm, "build_llm_client", fake_build_llm_client)

    client, per_role = llm.build_llm_clients_from_env(
        default_provider="openai",
        default_model="test-model",
        default_api_key="test-key",
        default_base_url="https://example.invalid/v1/",
    )

    assert isinstance(client, llm.DummyLLMClient)
    assert per_role == {}
    assert captured == {
        "provider": "openai",
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1/",
    }


def test_openai_generate_sends_configured_model() -> None:
    client = llm.OpenAILLMClient.__new__(llm.OpenAILLMClient)
    llm.BaseLLMClient.__init__(client, "configured-model")
    create = Mock(
        return_value=SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content="result"))],
        )
    )
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    assert client.generate("hello", system="instructions") == "result"
    create.assert_called_once_with(
        model="configured-model",
        messages=[
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "hello"},
        ],
        temperature=0.2,
    )

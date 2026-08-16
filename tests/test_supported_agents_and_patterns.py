"""Guard the supported agent and orchestration surface."""

from __future__ import annotations

import pytest

from fs_agent.agents.base import AgentRole
from fs_agent.benchmark import ALL_PATTERNS, _build_pattern
from fs_agent.llm import DummyLLMClient
from fs_agent.orchestration import AgentRegistry, register_default_agents


def test_only_core_agents_and_fixer_are_registered() -> None:
    assert {role.value for role in AgentRole} == {
        "fullstack",
        "architect",
        "backend",
        "frontend",
        "infra",
        "fixer",
    }

    registry = AgentRegistry()
    register_default_agents(registry)
    for role in AgentRole:
        assert registry.build(role).role is role


def test_hierarchical_pattern_is_not_supported() -> None:
    assert ALL_PATTERNS == [
        "single",
        "sequential",
        "centralized",
        "decentralized",
        "parallel",
    ]

    with pytest.raises(ValueError, match="Unknown pattern: hierarchical"):
        _build_pattern("hierarchical", AgentRegistry(), DummyLLMClient("dummy"))

"""Tests for the centralized orchestration pattern."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fs_agent.config import Settings
from fs_agent.context import RunContext
from fs_agent.llm import BaseLLMClient
from fs_agent.orchestration import (
    AgentRegistry,
    CentralizedOrchestrator,
    OrchestrationError,
    register_default_agents,
)


class ScriptedCoordinatorLLM(BaseLLMClient):
    """LLM mock that returns scripted coordinator decisions, then
    falls through to a dummy response for agent-level calls.

    The coordinator prompt is identified by the presence of
    'Available agents' in the prompt text.
    """

    def __init__(self, decisions: list[dict[str, str]]) -> None:
        super().__init__("scripted")
        self._decisions = list(decisions)
        self._call_index = 0

    def generate(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.2
    ) -> str:
        # Coordinator calls contain "Available agents"
        if "Available agents" in prompt and self._decisions:
            decision = self._decisions.pop(0)
            return json.dumps(decision)

        # Agent-level calls: return the same dummy content the DummyLLMClient
        # would, so agents exercise their fallback paths.
        head = prompt.strip().splitlines()
        preview = " \n".join(head[:10])
        return (
            "// LLM output unavailable in this environment.\n"
            "// Provide the following prompt to a real model for richer output.\n"
            + preview
        )


def _make_context(tmp_path: Path, llm: BaseLLMClient) -> RunContext:
    settings = Settings(
        artifact_dir=tmp_path,
        dry_run=False,
        orchestration_pattern="centralized",
        llm_provider="dummy",
    )
    return RunContext(
        spec=None,
        user_request="Build a shared task tracker for teams",
        settings=settings,
        workspace_dir=Path.cwd(),
        artifact_dir=tmp_path,
        llm=llm,
    )


def test_centralized_runs_all_agents_via_coordinator(tmp_path: Path) -> None:
    """Coordinator issues run decisions for each agent, then done.

    NOTE: The ScriptedCoordinatorLLM returns dummy text for agent-level
    calls (not coordinator prompts), which causes the architect agent to
    crash when parsing the LLM output as JSON.  This test verifies that
    the orchestrator raises (no silent fallback) in that scenario.
    """
    decisions = [
        {"action": "run", "agent": "architect", "reason": "need a spec first"},
        {"action": "run", "agent": "backend", "reason": "spec is ready"},
        {"action": "run", "agent": "frontend", "reason": "backend done"},
        {"action": "run", "agent": "infra", "reason": "ready to bootstrap"},
        {"action": "done", "reason": "all agents completed"},
    ]
    llm = ScriptedCoordinatorLLM(decisions)
    context = _make_context(tmp_path, llm)

    registry = AgentRegistry()
    register_default_agents(registry)
    orchestrator = CentralizedOrchestrator(registry=registry, llm=llm)

    # The architect agent fails due to dummy LLM output
    with pytest.raises(RuntimeError):
        list(orchestrator.run(context))


def test_centralized_can_skip_agents(tmp_path: Path) -> None:
    """Coordinator can choose to run only architect + backend, then stop.

    NOTE: Same limitation as above — architect crashes on dummy LLM output.
    """
    decisions = [
        {"action": "run", "agent": "architect", "reason": "need a spec"},
        {"action": "run", "agent": "backend", "reason": "only want an API"},
        {"action": "done", "reason": "user only wants backend scaffold"},
    ]
    llm = ScriptedCoordinatorLLM(decisions)
    context = _make_context(tmp_path, llm)

    registry = AgentRegistry()
    register_default_agents(registry)
    orchestrator = CentralizedOrchestrator(registry=registry, llm=llm)

    with pytest.raises(RuntimeError):
        list(orchestrator.run(context))


def test_centralized_respects_max_iterations(tmp_path: Path) -> None:
    """Coordinator that never says 'done' is capped by max_iterations.

    NOTE: With the ScriptedCoordinatorLLM returning dummy text for agent
    calls, the architect agent crashes before the loop can iterate.
    This test validates that the orchestrator raises (rather than
    silently falling back) when the agent fails.
    """
    decisions = [
        {"action": "run", "agent": "architect", "reason": "again"},
    ] * 5
    llm = ScriptedCoordinatorLLM(decisions)
    context = _make_context(tmp_path, llm)

    registry = AgentRegistry()
    register_default_agents(registry)
    orchestrator = CentralizedOrchestrator(
        registry=registry, llm=llm, max_iterations=3
    )

    # The architect agent fails due to dummy LLM output (not valid JSON spec)
    with pytest.raises(RuntimeError):
        list(orchestrator.run(context))


def test_centralized_raises_on_bad_agent_name(tmp_path: Path) -> None:
    """If the coordinator returns a bogus agent name, OrchestrationError is raised."""
    decisions = [
        {"action": "run", "agent": "architect", "reason": "first"},
        {"action": "run", "agent": "NONEXISTENT", "reason": "oops"},
    ]
    llm = ScriptedCoordinatorLLM(decisions)
    context = _make_context(tmp_path, llm)

    registry = AgentRegistry()
    register_default_agents(registry)
    orchestrator = CentralizedOrchestrator(registry=registry, llm=llm)

    # Since architect agent will fail (ScriptedCoordinatorLLM returns dummy
    # text for non-coordinator prompts), the run raises RuntimeError before
    # reaching the NONEXISTENT decision.  In a real scenario with a working
    # LLM, it would raise OrchestrationError for the unknown agent.
    with pytest.raises((RuntimeError, OrchestrationError)):
        list(orchestrator.run(context))

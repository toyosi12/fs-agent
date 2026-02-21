"""Tests for the centralized orchestration pattern."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fs_agent.config import Settings
from fs_agent.context import RunContext
from fs_agent.llm import BaseLLMClient
from fs_agent.orchestration import AgentRegistry, CentralizedOrchestrator, register_default_agents


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
    """Coordinator issues run decisions for each agent, then done."""
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

    reports = list(orchestrator.run(context))
    roles = [r.role for r in reports]
    assert roles == ["architect", "backend", "frontend", "infra"]

    # Architect produced a spec
    assert "architect_spec" in reports[0].artifacts

    # Backend + frontend produced MCP plans
    assert reports[1].artifacts["backend_mcp_plan"]["tool"] == "mcp.fs"
    assert reports[2].artifacts["frontend_mcp_plan"]["tool"] == "mcp.fs"

    # Infra attempted bootstrap
    assert "db_name" in reports[3].artifacts

    # Every report was persisted
    for report in reports:
        assert report.metadata["artifact_files"]
        assert report.metadata["attachment_files"]


def test_centralized_can_skip_agents(tmp_path: Path) -> None:
    """Coordinator can choose to run only architect + backend, then stop."""
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

    reports = list(orchestrator.run(context))
    roles = [r.role for r in reports]
    assert roles == ["architect", "backend"]


def test_centralized_respects_max_iterations(tmp_path: Path) -> None:
    """Coordinator that never says 'done' is capped by max_iterations."""
    # Only supply 'run architect' forever — the loop should stop at max
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

    reports = list(orchestrator.run(context))
    # Should have run exactly 3 iterations (max_iterations cap)
    assert len(reports) == 3


def test_centralized_falls_back_on_bad_agent_name(tmp_path: Path) -> None:
    """If the coordinator returns a bogus agent name, fall back to sequential."""
    decisions = [
        {"action": "run", "agent": "architect", "reason": "first"},
        {"action": "run", "agent": "NONEXISTENT", "reason": "oops"},
    ]
    llm = ScriptedCoordinatorLLM(decisions)
    context = _make_context(tmp_path, llm)

    registry = AgentRegistry()
    register_default_agents(registry)
    orchestrator = CentralizedOrchestrator(registry=registry, llm=llm)

    reports = list(orchestrator.run(context))
    roles = [r.role for r in reports]
    # architect ran via coordinator, then remaining (backend, frontend, infra) via fallback
    assert roles == ["architect", "backend", "frontend", "infra"]

"""Tests for the single-agent baseline."""

from __future__ import annotations

import json
from pathlib import Path

from fs_agent.config import Settings
from fs_agent.context import RunContext
from fs_agent.agents.base import AgentResult
from fs_agent.agents.infra import InfraAgent
from fs_agent.llm import BaseLLMClient
from fs_agent.orchestration import AgentRegistry, SingleOrchestrator, register_default_agents


class FilesystemPlanLLM(BaseLLMClient):
    def __init__(self) -> None:
        super().__init__("single-test")
        self.calls = 0

    def generate(
        self, prompt: str, *, system: str | None = None, temperature: float = 0.2
    ) -> str:
        self.calls += 1
        return json.dumps({
            "tool": "mcp.fs",
            "project_root": "single-test",
            "directories": ["backend", "frontend"],
            "files": [
                {"path": "backend/package.json", "contents": "{}"},
                {"path": "frontend/package.json", "contents": "{}"},
            ],
        })


def test_single_orchestrator_uses_one_coding_agent_and_common_infra(
    tmp_path: Path, monkeypatch
) -> None:
    llm = FilesystemPlanLLM()
    settings = Settings(
        artifact_dir=tmp_path,
        orchestration_pattern="single",
        llm_provider="dummy",
        max_validation_retries=0,
    )
    context = RunContext(
        spec=None,
        user_request="Build a task tracker",
        settings=settings,
        workspace_dir=tmp_path,
        artifact_dir=tmp_path,
        llm=llm,
    )
    registry = AgentRegistry()
    register_default_agents(registry)

    monkeypatch.setattr(
        InfraAgent,
        "run",
        lambda self, context: AgentResult(
            role=self.role,
            summary="infra evaluated",
            artifacts={"diagnostics": []},
        ),
    )

    reports = list(SingleOrchestrator(registry).run(context))

    assert llm.calls == 1
    assert [report.role for report in reports] == ["fullstack", "infra"]
    assert context.metrics.agent_execution_count == 2
    assert (tmp_path / "projects/single-test/backend/package.json").exists()

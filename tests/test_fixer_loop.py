"""Tests for the iterative fixer/infra evaluation loop."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from fs_agent.context import AgentReport
from fs_agent.orchestration._helpers import run_fixer_loop
from fs_agent.orchestration.metrics import AgentExecution, OrchestrationMetrics


def _report(role: str, status: str, artifacts: dict) -> AgentReport:
    now = datetime.now(timezone.utc)
    return AgentReport(
        role=role,
        summary=role,
        artifacts=artifacts,
        status=status,
        started_at=now,
        finished_at=now,
    )


def test_fixer_loop_reruns_infra_and_records_resolution(monkeypatch) -> None:
    reports = [_report("infra", "error", {"diagnostics": ["migration failed"]})]
    context = SimpleNamespace(
        settings=SimpleNamespace(max_fixer_iterations=3),
        transcripts=list(reports),
    )
    registry = SimpleNamespace(build=lambda role: SimpleNamespace(role=role))
    metrics = OrchestrationMetrics()

    def fake_execute(agent, role, run_context, *, attempt=1):
        if role.value == "fixer":
            report = _report(
                "fixer",
                "success",
                {"fixer_patch_count": 1, "fixer_patches_failed": []},
            )
        else:
            report = _report("infra", "success", {"diagnostics": []})
        run_context.transcripts.append(report)
        return report, AgentExecution(
            role=role.value,
            status=report.status,
            attempt=attempt,
        )

    monkeypatch.setattr(
        "fs_agent.orchestration._helpers.execute_agent", fake_execute
    )

    updated, result = run_fixer_loop(context, registry, reports, metrics)

    assert [report.role for report in updated] == ["infra", "fixer", "infra"]
    assert result.iterations_run == 1
    assert result.resolved is True
    assert result.final_errors == []
    assert [run.attempt for run in metrics.agent_executions] == [1, 2]

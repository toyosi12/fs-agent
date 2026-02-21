"""Shared helpers for orchestration patterns."""

from __future__ import annotations

from datetime import datetime, timezone

from ..artifact_writer import persist_agent_output
from ..context import AgentReport, RunContext
from ..agents.base import AgentRole, BaseAgent


def execute_agent(agent: BaseAgent, role: AgentRole, context: RunContext) -> AgentReport:
    """Run an agent, persist its output, record the report, and return it.

    This is the canonical "dispatch one agent" logic shared by every
    orchestration pattern.
    """
    from ..logger import get_logger

    logger = get_logger("orchestration.execute")

    logger.info("→ %s agent starting", role.value)
    result = agent.run(context)
    saved_paths = persist_agent_output(result, context.metadata_dir)
    report = AgentReport(
        role=role.value,
        summary=result.summary,
        artifacts=result.artifacts,
        status=result.status,
        started_at=result.started_at,
        finished_at=result.finished_at or datetime.now(timezone.utc),
        metadata={
            "attachments": [a.name for a in result.attachments],
            "artifact_files": saved_paths["artifacts"],
            "attachment_files": saved_paths["attachments"],
        },
    )
    context.record(report)

    duration = (report.finished_at - report.started_at).total_seconds()
    logger.info(
        "✓ %s agent finished in %.2fs (%d artifacts, %d attachments)",
        role.value,
        duration,
        len(report.metadata["artifact_files"]),
        len(report.metadata["attachment_files"]),
    )
    return report

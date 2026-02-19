"""Sequential orchestration pattern."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from ...artifact_writer import persist_agent_output
from ...context import AgentReport, RunContext
from ...logger import get_logger
from ..base import OrchestrationPattern
from ..registry import AgentRegistry
from ...agents.base import AgentRole


class SequentialOrchestrator(OrchestrationPattern):
    """Executes agents one after another in a deterministic order."""

    def __init__(
        self,
        registry: AgentRegistry,
        order: Sequence[AgentRole] | None = None,
    ) -> None:
        self.registry = registry
        self.order: Sequence[AgentRole] = order or (
            AgentRole.ARCHITECT,
            AgentRole.BACKEND,
            AgentRole.FRONTEND,
            AgentRole.INFRA,
        )
        self.logger = get_logger(self.__class__.__name__)

    def run(self, context: RunContext) -> Iterable[AgentReport]:
        reports: list[AgentReport] = []
        pipeline = " -> ".join(role.value for role in self.order)
        self.logger.info("Sequential pipeline start: %s", pipeline)
        for role in self.order:
            agent = self.registry.build(role)
            self.logger.info("→ %s agent starting", role.value)
            result = agent.run(context)
            saved_paths = persist_agent_output(result, context.artifact_dir)
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
            reports.append(report)
            duration = (report.finished_at - report.started_at).total_seconds()
            self.logger.info(
                "✓ %s agent finished in %.2fs (%d artifacts, %d attachments)",
                role.value,
                duration,
                len(report.metadata["artifact_files"]),
                len(report.metadata["attachment_files"]),
            )
        self.logger.info("Sequential pipeline complete: %d stages", len(reports))
        return reports

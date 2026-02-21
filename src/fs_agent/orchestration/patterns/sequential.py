"""Sequential orchestration pattern."""

from __future__ import annotations

from typing import Iterable, Sequence

from ...context import AgentReport, RunContext
from ...logger import get_logger
from ..base import OrchestrationPattern
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent


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
            report = execute_agent(agent, role, context)
            reports.append(report)
        self.logger.info("Sequential pipeline complete: %d stages", len(reports))
        return reports

"""Single-agent, non-orchestrated baseline."""

from __future__ import annotations

from typing import Sequence

from ...agents.base import AgentRole
from ...context import AgentReport, RunContext
from ...logger import get_logger
from .._helpers import execute_agent, run_fixer_loop, run_validation_loop
from ..base import OrchestrationPattern
from ..registry import AgentRegistry


class SingleOrchestrator(OrchestrationPattern):
    """Run one coding agent, then the common infra evaluation harness."""

    def __init__(self, registry: AgentRegistry) -> None:
        self.registry = registry
        self.logger = get_logger(self.__class__.__name__)

    def run(self, context: RunContext) -> Sequence[AgentReport]:
        metrics = context.metrics
        metrics.pattern = "single"
        metrics.task_id = getattr(context.settings, "task_id", "")
        metrics.start_timer()
        reports: list[AgentReport] = []
        self.logger.info("╔══ SINGLE ORCHESTRATOR START ══╗")
        try:
            agent = self.registry.build(AgentRole.FULLSTACK)
            report, execution = execute_agent(agent, AgentRole.FULLSTACK, context)
            metrics.record_agent_execution(execution)
            reports.append(report)

            # Infra is common evaluation/bootstrap work, not another coding
            # specialist. Running it here keeps the baseline comparable with
            # every multi-agent pattern.
            infra = self.registry.build(AgentRole.INFRA)
            infra_report, infra_execution = execute_agent(
                infra, AgentRole.INFRA, context
            )
            metrics.record_agent_execution(infra_execution)
            reports.append(infra_report)
            metrics.success = True

            reports, fixer_result = run_fixer_loop(
                context,
                self.registry,
                reports,
                metrics,
                pattern_name="single",
            )
            metrics.fixer_loop_result = fixer_result.to_dict()
            reports = run_validation_loop(
                context,
                self.registry,
                reports,
                metrics,
                pattern_name="single",
            )
        except Exception as exc:
            metrics.success = False
            metrics.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            metrics.stop_timer()
            self.logger.info(
                "╚══ SINGLE ORCHESTRATOR END ════╝ success=%s duration=%.2fs "
                "agents_run=%d total_tokens=%d est_cost=$%.6f",
                metrics.success,
                metrics.total_duration_seconds,
                metrics.agent_execution_count,
                metrics.total_tokens,
                metrics.cost_estimate(),
            )
        return reports

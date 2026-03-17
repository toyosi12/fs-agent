"""Sequential orchestration pattern — deterministic agent pipeline.

No LLM coordination, no fallbacks.  Agents run in a fixed order.
Metrics are captured for research benchmarking (RQ1–RQ3).
"""

from __future__ import annotations

from typing import Sequence

from ...context import AgentReport, RunContext
from ...logger import get_logger
from ..base import OrchestrationPattern
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent, run_validation_loop


class SequentialOrchestrator(OrchestrationPattern):
    """Executes agents one after another in a deterministic order.

    This is the baseline pattern — no LLM coordination overhead.
    Coordination token counts will always be zero.
    """

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

    def run(self, context: RunContext) -> Sequence[AgentReport]:
        m = context.metrics
        m.pattern = "sequential"
        m.task_id = getattr(context.settings, "task_id", "")
        m.start_timer()

        reports: list[AgentReport] = []
        pipeline = " -> ".join(role.value for role in self.order)

        self.logger.info(
            "╔══ SEQUENTIAL ORCHESTRATOR START ══╗  pipeline: %s", pipeline
        )

        try:
            for idx, role in enumerate(self.order, 1):
                self.logger.info(
                    "── [%d/%d] %s ──────────────────────────────────────",
                    idx, len(self.order), role.value,
                )
                agent = self.registry.build(role)
                report, execution = execute_agent(agent, role, context)
                m.record_agent_execution(execution)
                reports.append(report)

            m.success = True

            # --- Validation loop ---
            reports = run_validation_loop(
                context, self.registry, reports, m,
                pattern_name="sequential",
            )

        except Exception as exc:
            m.success = False
            m.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            m.stop_timer()
            self._log_summary(m, reports)

        return reports

    def _log_summary(self, m: object, reports: list[AgentReport]) -> None:
        self.logger.info(
            "╚══ SEQUENTIAL ORCHESTRATOR END ════╝\n"
            "  success=%s  duration=%.2fs  agents_run=%d\n"
            "  coordination_calls=0  coordination_tokens=0  "
            "(no LLM coordination in sequential)\n"
            "  functional_tokens=%d  (prompt=%d, completion=%d)\n"
            "  coordination/functional ratio=%.4f\n"
            "  total_tokens=%d  est_cost=$%.6f",
            m.success,
            m.total_duration_seconds,
            m.agent_execution_count,
            m.functional_total_tokens,
            m.functional_prompt_tokens,
            m.functional_completion_tokens,
            m.coordination_to_functional_ratio,
            m.total_tokens,
            m.cost_estimate(),
        )

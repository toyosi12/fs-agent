"""Sequential orchestration pattern — artifact-passing pipeline.

Agents run in a fixed order.  Each agent's extracted output contract is
passed downstream so later agents react to what was *actually generated*,
not just what the spec planned.  No LLM coordination overhead.
"""

from __future__ import annotations

from typing import Sequence

from ...context import AgentReport, RunContext
from ...logger import get_logger
from ..base import OrchestrationPattern
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent, run_fixer_loop, run_validation_loop


class SequentialOrchestrator(OrchestrationPattern):
    """Artifact-passing pipeline — each agent feeds the next.

    After each agent completes, its output is extracted into a compact
    contract and injected into ``context.extra_context`` for the next
    agent.  This is the baseline pattern — zero LLM coordination
    overhead, but downstream agents see upstream artifacts.
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

                # Inject upstream artifacts into context for this agent
                context.extra_context = self._build_upstream_context(role, context)

                agent = self.registry.build(role)
                report, execution = execute_agent(agent, role, context)
                m.record_agent_execution(execution)
                reports.append(report)

            m.success = True

            # --- Fixer loop (fixer ↔ infra) ---
            reports, fixer_result = run_fixer_loop(
                context, self.registry, reports, m,
                pattern_name="sequential",
            )
            m.fixer_loop_result = fixer_result.to_dict()

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
            context.extra_context = {}  # clean up
            m.stop_timer()
            self._log_summary(m, reports)

        return reports

    def _build_upstream_context(
        self, role: AgentRole, context: RunContext
    ) -> dict[str, str]:
        """Build upstream context for the given role from completed agents."""
        if role == AgentRole.ARCHITECT:
            return {}

        parts: list[str] = []

        if role in (AgentRole.FRONTEND, AgentRole.INFRA):
            backend_contract = context.extract_backend_contract()
            if backend_contract:
                parts.append(
                    "=== Backend Agent Output (actual generated code) ===\n"
                    f"{backend_contract}"
                )

        if role == AgentRole.INFRA:
            frontend_contract = context.extract_frontend_contract()
            if frontend_contract:
                parts.append(
                    "=== Frontend Agent Output (actual generated code) ===\n"
                    f"{frontend_contract}"
                )

        if not parts:
            return {}
        return {"upstream_context": "\n\n".join(parts)}

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

"""Parallel orchestration pattern — fan-out / fan-in with ThreadPoolExecutor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable

from ...context import AgentReport, RunContext
from ...logger import get_logger
from ..base import OrchestrationPattern
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent


@dataclass
class Stage:
    """A group of agent roles that can run concurrently.

    Stages execute in order; agents within a stage fan out in parallel.
    """

    roles: list[AgentRole]
    label: str = ""


# Default pipeline: architect (serial) → backend + frontend (parallel) → infra (serial)
DEFAULT_STAGES: list[Stage] = [
    Stage(roles=[AgentRole.ARCHITECT], label="planning"),
    Stage(roles=[AgentRole.BACKEND, AgentRole.FRONTEND], label="build"),
    Stage(roles=[AgentRole.INFRA], label="deploy"),
]


class ParallelOrchestrator(OrchestrationPattern):
    """Fan-out / fan-in orchestration with concurrent agent execution.

    Agents are organized into sequential **stages**.  Within each stage,
    all agents run in parallel using a thread pool.  The next stage only
    starts once every agent in the current stage has finished.

    Default pipeline::

        Stage 1 (planning) : architect              ← serial (1 agent)
        Stage 2 (build)    : backend + frontend      ← parallel
        Stage 3 (deploy)   : infra                   ← serial (1 agent)

    Custom stage definitions can be provided via the ``stages`` arg.
    """

    DEFAULT_MAX_WORKERS = 4

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        stages: list[Stage] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.registry = registry
        self.stages = stages or DEFAULT_STAGES
        self.max_workers = max_workers or self.DEFAULT_MAX_WORKERS
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, context: RunContext) -> Iterable[AgentReport]:
        all_reports: list[AgentReport] = []

        self.logger.info(
            "Parallel orchestrator starting (%d stages, max_workers=%d)",
            len(self.stages),
            self.max_workers,
        )

        for stage_idx, stage in enumerate(self.stages, 1):
            label = stage.label or f"stage-{stage_idx}"
            role_names = [r.value for r in stage.roles]

            if len(stage.roles) == 1:
                # Single agent — run directly, no thread overhead
                self.logger.info(
                    "Stage %d/%d [%s]: running %s (serial)",
                    stage_idx,
                    len(self.stages),
                    label,
                    role_names[0],
                )
                role = stage.roles[0]
                agent = self.registry.build(role)
                report = execute_agent(agent, role, context)
                all_reports.append(report)
            else:
                # Multiple agents — fan out
                self.logger.info(
                    "Stage %d/%d [%s]: running %s (parallel, %d workers)",
                    stage_idx,
                    len(self.stages),
                    label,
                    ", ".join(role_names),
                    min(self.max_workers, len(stage.roles)),
                )
                stage_reports = self._run_parallel(context, stage)
                all_reports.extend(stage_reports)

        self.logger.info(
            "Parallel orchestrator complete: %d stages, %d agents executed",
            len(self.stages),
            len(all_reports),
        )
        return all_reports

    # ------------------------------------------------------------------
    # Parallel fan-out / fan-in
    # ------------------------------------------------------------------

    def _run_parallel(
        self, context: RunContext, stage: Stage
    ) -> list[AgentReport]:
        """Run all agents in a stage concurrently and collect their reports.

        Each thread gets its own agent instance from the registry.
        Reports are collected from futures on the main thread and then
        recorded into context in a deterministic order (matching the
        stage's role list) to keep transcripts reproducible.
        """
        reports_by_role: dict[str, AgentReport] = {}
        errors: dict[str, Exception] = {}

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(stage.roles))
        ) as pool:
            future_to_role = {}
            for role in stage.roles:
                agent = self.registry.build(role)
                future = pool.submit(self._execute_in_thread, agent, role, context)
                future_to_role[future] = role

            for future in as_completed(future_to_role):
                role = future_to_role[future]
                try:
                    report = future.result()
                    reports_by_role[role.value] = report
                except Exception as exc:
                    self.logger.error(
                        "Agent '%s' failed in parallel stage: %s",
                        role.value,
                        exc,
                    )
                    errors[role.value] = exc

        if errors:
            failed = ", ".join(errors.keys())
            self.logger.warning(
                "%d agent(s) failed in parallel stage: %s", len(errors), failed
            )

        # Record reports into context in declaration order for reproducibility
        ordered: list[AgentReport] = []
        for role in stage.roles:
            if role.value in reports_by_role:
                report = reports_by_role[role.value]
                context.record(report)
                ordered.append(report)

        return ordered

    def _execute_in_thread(
        self, agent: object, role: AgentRole, context: RunContext
    ) -> AgentReport:
        """Run a single agent in a worker thread.

        We call agent.run() and build the AgentReport here but do NOT
        call context.record() — that happens on the main thread after
        all futures complete, to avoid race conditions on the shared
        transcripts list.
        """
        from datetime import datetime, timezone
        from ...artifact_writer import persist_agent_output

        self.logger.info("→ %s agent starting (thread)", role.value)
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
        duration = (report.finished_at - report.started_at).total_seconds()
        self.logger.info(
            "✓ %s agent finished in %.2fs (thread)", role.value, duration
        )
        return report

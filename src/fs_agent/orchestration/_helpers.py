"""Shared helpers for orchestration patterns."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from ..artifact_writer import persist_agent_output
from ..context import AgentReport, RunContext
from ..agents.base import AgentRole, BaseAgent
from .metrics import AgentExecution

if TYPE_CHECKING:
    from .metrics import OrchestrationMetrics
    from .registry import AgentRegistry


def execute_agent(
    agent: BaseAgent,
    role: AgentRole,
    context: RunContext,
    *,
    attempt: int = 1,
) -> tuple[AgentReport, AgentExecution]:
    """Run an agent, persist its output, record the report, and return it.

    Returns ``(AgentReport, AgentExecution)`` so the calling pattern can
    feed the execution record into :class:`OrchestrationMetrics`.

    Parameters
    ----------
    attempt:
        Attempt number (>1 when the validation loop retries an agent).
    """
    from ..logger import get_logger

    logger = get_logger("orchestration.execute")

    # --- Snapshot LLM usage BEFORE the agent runs ---
    role_llm = context.get_llm(role.value)
    pre = role_llm.usage_stats.copy()

    logger.info(
        "→ [%s] agent starting (attempt=%d, model=%s)",
        role.value, attempt, role_llm.model,
    )
    wall_start = time.perf_counter()
    result = agent.run(context)
    wall_elapsed = time.perf_counter() - wall_start

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

    # --- Snapshot LLM usage AFTER the agent runs ---
    post = role_llm.usage_stats
    delta_prompt = post["prompt_tokens"] - pre["prompt_tokens"]
    delta_completion = post["completion_tokens"] - pre["completion_tokens"]
    delta_total = post["total_tokens"] - pre["total_tokens"]

    execution = AgentExecution(
        role=role.value,
        status=result.status,
        attempt=attempt,
        duration_seconds=round(wall_elapsed, 4),
        prompt_tokens=delta_prompt,
        completion_tokens=delta_completion,
        total_tokens=delta_total,
        artifact_count=len(report.artifacts),
        attachment_count=len(report.metadata.get("attachments", [])),
    )

    logger.info(
        "✓ [%s] agent finished  attempt=%d  %.2fs  tokens=%d (prompt=%d, completion=%d)  "
        "artifacts=%d  attachments=%d  status=%s",
        role.value,
        attempt,
        wall_elapsed,
        delta_total,
        delta_prompt,
        delta_completion,
        execution.artifact_count,
        execution.attachment_count,
        result.status,
    )
    return report, execution


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

# Maps validation component names to the agent role responsible for fixing them.
_COMPONENT_TO_ROLE: dict[str, AgentRole] = {
    "backend": AgentRole.BACKEND,
    "frontend": AgentRole.FRONTEND,
    "infra": AgentRole.INFRA,
    "integration": AgentRole.BACKEND,  # integration issues default to backend
    "project": AgentRole.INFRA,
}


def run_validation_loop(
    context: RunContext,
    registry: "AgentRegistry",
    reports: list[AgentReport],
    metrics: "OrchestrationMetrics",
    *,
    max_retries: int | None = None,
    pattern_name: str = "",
) -> list[AgentReport]:
    """Run post-generation validation and re-run failing agents.

    After all agents have completed their initial run, this function:
    1. Validates the generated project directory.
    2. If validation passes, returns the reports as-is.
    3. If validation fails and retries remain, identifies the responsible
       agents, injects error feedback into the context, re-runs them,
       and validates again.

    Parameters
    ----------
    context:
        The shared run context (contains project paths and settings).
    registry:
        Agent registry for building agents.
    reports:
        List of agent reports from the initial run.
    metrics:
        Orchestration metrics to record retry executions.
    max_retries:
        Maximum number of validation-retry iterations. If None, reads
        from ``context.settings.max_validation_retries``.
    pattern_name:
        Name of the orchestration pattern (for logging).

    Returns
    -------
    Updated list of agent reports (may include retry reports appended).
    """
    from ..logger import get_logger
    from ..validation import validate_project, ValidationResult

    logger = get_logger(f"orchestration.validation.{pattern_name or 'loop'}")

    if max_retries is None:
        max_retries = context.settings.max_validation_retries

    if max_retries <= 0:
        logger.info("Validation retries disabled (max_retries=0)")
        return reports

    project_dir = context.projects_dir
    if not project_dir.exists():
        logger.warning(
            "Project directory %s does not exist — skipping validation", project_dir
        )
        return reports

    for iteration in range(1, max_retries + 1):
        result = validate_project(project_dir)
        logger.info(
            "Validation iteration %d/%d: %s",
            iteration, max_retries, result.summary(),
        )

        if result.passed:
            logger.info("✓ Validation passed on iteration %d", iteration)
            return reports

        # Determine which agents need to re-run
        failed_components = {
            issue.component
            for issue in result.issues
            if issue.severity == "error"
        }
        roles_to_retry: set[AgentRole] = set()
        for comp in failed_components:
            role = _COMPONENT_TO_ROLE.get(comp)
            if role:
                roles_to_retry.add(role)

        if not roles_to_retry:
            logger.warning(
                "Validation has errors but no responsible agents identified"
            )
            return reports

        # Inject validation feedback into user_request so agents see it
        feedback = result.feedback_prompt()
        original_request = context.user_request
        context.user_request = f"{original_request}\n\n{feedback}"

        logger.info(
            "Re-running agents %s (iteration %d/%d) to fix %d errors",
            [r.value for r in roles_to_retry],
            iteration,
            max_retries,
            result.error_count,
        )

        for role in roles_to_retry:
            agent = registry.build(role)
            report, execution = execute_agent(
                agent, role, context, attempt=iteration + 1
            )
            metrics.record_agent_execution(execution)
            reports.append(report)

        # Restore original request for next iteration
        context.user_request = original_request

    # Final validation check after all retries
    final_result = validate_project(project_dir)
    if not final_result.passed:
        logger.warning(
            "Validation still failing after %d retries: %s",
            max_retries, final_result.summary(),
        )
    else:
        logger.info("✓ Validation passed after retries")

    return reports

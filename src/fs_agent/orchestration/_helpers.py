"""Shared helpers for orchestration patterns."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

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
    # Specialized sub-roles are retried via their parent agent in
    # the validation loop (the registry has all roles registered).
    "backend_api": AgentRole.BACKEND_API,
    "backend_db": AgentRole.BACKEND_DB,
    "frontend_pages": AgentRole.FRONTEND_PAGES,
    "frontend_ui": AgentRole.FRONTEND_UI,
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

        # Log each error for diagnostics
        for issue in result.issues:
            if issue.severity == "error":
                loc = f" ({issue.file})" if issue.file else ""
                logger.info(
                    "  ✗ [%s]%s: %s", issue.component, loc, issue.message
                )

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


# ---------------------------------------------------------------------------
# Fixer ↔ Infra loop
# ---------------------------------------------------------------------------

@dataclass
class FixerLoopResult:
    """Tracks what the fixer loop resolved and how many iterations it took."""

    iterations_run: int = 0
    max_iterations: int = 0
    resolved: bool = False
    initial_errors: list[str] = field(default_factory=list)
    final_errors: list[str] = field(default_factory=list)
    patches_applied_per_iteration: list[int] = field(default_factory=list)
    patches_failed_per_iteration: list[int] = field(default_factory=list)
    issues_resolved: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations_run": self.iterations_run,
            "max_iterations": self.max_iterations,
            "resolved": self.resolved,
            "initial_error_count": len(self.initial_errors),
            "final_error_count": len(self.final_errors),
            "initial_errors": self.initial_errors,
            "final_errors": self.final_errors,
            "patches_applied_per_iteration": self.patches_applied_per_iteration,
            "patches_failed_per_iteration": self.patches_failed_per_iteration,
            "total_patches_applied": sum(self.patches_applied_per_iteration),
            "total_patches_failed": sum(self.patches_failed_per_iteration),
            "issues_resolved": self.issues_resolved,
        }


def run_fixer_loop(
    context: RunContext,
    registry: "AgentRegistry",
    reports: list[AgentReport],
    metrics: "OrchestrationMetrics",
    *,
    max_iterations: int | None = None,
    pattern_name: str = "",
) -> tuple[list[AgentReport], FixerLoopResult]:
    """Run the fixer ↔ infra loop until errors are resolved or max iterations.

    After each fixer run, infra is re-run to check if the fixes worked.
    Returns the updated reports list and a FixerLoopResult with detailed
    metrics about what was fixed.

    Parameters
    ----------
    context:
        The shared run context.
    registry:
        Agent registry (needs FIXER and INFRA roles).
    reports:
        Current list of agent reports.
    metrics:
        Orchestration metrics to record executions.
    max_iterations:
        Max fixer→infra cycles. If None, reads from settings.
    pattern_name:
        Name of the orchestration pattern (for logging).

    Returns
    -------
    Tuple of (updated reports, FixerLoopResult).
    """
    from ..logger import get_logger

    logger = get_logger(f"orchestration.fixer.{pattern_name or 'loop'}")

    if max_iterations is None:
        max_iterations = context.settings.max_fixer_iterations

    result = FixerLoopResult(max_iterations=max_iterations)

    if max_iterations <= 0:
        logger.info("Fixer loop disabled (max_iterations=0)")
        return reports, result

    # Find the most recent infra report to check for errors
    infra_errors = _get_infra_errors(reports)
    if not infra_errors:
        logger.info("No infra errors detected — skipping fixer loop")
        return reports, result

    result.initial_errors = list(infra_errors)
    logger.info(
        "Fixer loop starting: %d infra errors, max %d iterations",
        len(infra_errors), max_iterations,
    )

    for iteration in range(1, max_iterations + 1):
        logger.info(
            "── Fixer iteration %d/%d ── (%d errors remaining)",
            iteration, max_iterations, len(infra_errors),
        )

        # 1. Run the fixer agent
        fixer_agent = registry.build(AgentRole.FIXER)
        fixer_report, fixer_exec = execute_agent(
            fixer_agent, AgentRole.FIXER, context, attempt=iteration,
        )
        metrics.record_agent_execution(fixer_exec)
        reports.append(fixer_report)

        patches_applied = fixer_report.artifacts.get("fixer_patch_count", 0)
        patches_failed = len(fixer_report.artifacts.get("fixer_patches_failed", []))
        result.patches_applied_per_iteration.append(patches_applied)
        result.patches_failed_per_iteration.append(patches_failed)

        if patches_applied == 0:
            logger.info("Fixer produced no patches — stopping loop")
            break

        # 2. Re-run infra to see if fixes helped
        infra_agent = registry.build(AgentRole.INFRA)
        infra_report, infra_exec = execute_agent(
            infra_agent, AgentRole.INFRA, context, attempt=iteration + 1,
        )
        metrics.record_agent_execution(infra_exec)
        reports.append(infra_report)

        # 3. Check new error state
        new_errors = _get_infra_errors(reports)
        resolved = set(infra_errors) - set(new_errors)
        result.issues_resolved.extend(sorted(resolved))

        logger.info(
            "  Iteration %d: %d patches applied, %d errors resolved, %d remaining",
            iteration, patches_applied, len(resolved), len(new_errors),
        )

        result.iterations_run = iteration

        if not new_errors:
            result.resolved = True
            logger.info("✓ All infra errors resolved after %d iterations", iteration)
            break

        infra_errors = new_errors

    result.final_errors = list(infra_errors) if not result.resolved else []

    if not result.resolved:
        logger.warning(
            "Fixer loop exhausted %d iterations, %d errors remain",
            result.iterations_run, len(result.final_errors),
        )

    return reports, result


def _get_infra_errors(reports: list[AgentReport]) -> list[str]:
    """Extract error diagnostics from the most recent infra report."""
    for report in reversed(reports):
        if report.role == "infra" and report.status == "error":
            # Diagnostics are stored in the artifacts by the infra agent
            diags = report.artifacts.get("diagnostics", [])
            if diags:
                return diags
            # Also check the infra_log for error markers
            log = report.artifacts.get("infra_log", "")
            errors = [
                line.strip() for line in log.split("\n")
                if line.strip().startswith("✗")
            ]
            return errors
    return []

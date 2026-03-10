"""Shared helpers for orchestration patterns."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from ..artifact_writer import persist_agent_output
from ..context import AgentReport, RunContext
from ..agents.base import AgentRole, BaseAgent
from .metrics import AgentExecution


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
        Attempt number (>1 when the iterative pattern retries an agent).
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

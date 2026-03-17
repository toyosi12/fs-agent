"""Orchestration metrics data model for research evaluation (RQ1–RQ3).

Provides structured capture of:
- **RQ1** (completion & quality): per-agent timing, success status, attempt counts
- **RQ2** (performance vs resource): functional token counts, cost estimates
- **RQ3** (coordination overhead): coordination LLM calls separated from
  functional agent calls, with token ratios

Every orchestration pattern populates an :class:`OrchestrationMetrics` instance
during its run.  The benchmark runner then serialises it alongside the existing
:class:`RunMetrics` for downstream analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Individual records
# ---------------------------------------------------------------------------

@dataclass
class CoordinationCall:
    """One LLM call made for orchestration routing (not agent work).

    Examples: coordinator decision, handoff decision, supervisor decision,
    critic evaluation.
    """

    purpose: str  # e.g. "coordinator_decision", "handoff_from_architect", "critic_backend_attempt_1"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    raw_response: str = ""  # full text for audit / debugging
    parsed_result: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0  # which loop iteration this occurred in


@dataclass
class AgentExecution:
    """Metrics for a single agent dispatch within one orchestration run.

    With validation retries, multiple :class:`AgentExecution` records may
    exist for the same role (one per attempt).
    """

    role: str
    status: str  # "success" or "error"
    attempt: int = 1  # 1 for first try, 2+ for validation retries
    duration_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    artifact_count: int = 0
    attachment_count: int = 0


# ---------------------------------------------------------------------------
# Aggregated per-run
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationMetrics:
    """Full metrics for one (task × pattern) orchestration run.

    Populated incrementally by the orchestration pattern, then finalised
    by the benchmark runner.
    """

    pattern: str = ""
    task_id: str = ""
    success: bool = False
    error: str | None = None

    # --- RQ1: Completion & Quality ---
    total_duration_seconds: float = 0.0
    agent_executions: list[AgentExecution] = field(default_factory=list)

    # --- RQ2: Performance vs Resource ---
    functional_prompt_tokens: int = 0
    functional_completion_tokens: int = 0
    functional_total_tokens: int = 0

    # --- RQ3: Coordination Overhead ---
    coordination_calls: list[CoordinationCall] = field(default_factory=list)
    coordination_prompt_tokens: int = 0
    coordination_completion_tokens: int = 0
    coordination_total_tokens: int = 0

    # ---- Wall-clock tracking ----
    _start_time: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------
    # Builder helpers (called by orchestration patterns)
    # ------------------------------------------------------------------

    def start_timer(self) -> None:
        """Call at the very beginning of an orchestration run."""
        self._start_time = time.perf_counter()

    def stop_timer(self) -> None:
        """Call at the very end of an orchestration run."""
        if self._start_time:
            self.total_duration_seconds = round(
                time.perf_counter() - self._start_time, 4
            )

    def record_coordination_call(self, call: CoordinationCall) -> None:
        """Append a coordination call and update running totals."""
        self.coordination_calls.append(call)
        self.coordination_prompt_tokens += call.prompt_tokens
        self.coordination_completion_tokens += call.completion_tokens
        self.coordination_total_tokens += call.total_tokens

    def record_agent_execution(self, execution: AgentExecution) -> None:
        """Append an agent execution and update functional token totals."""
        self.agent_executions.append(execution)
        self.functional_prompt_tokens += execution.prompt_tokens
        self.functional_completion_tokens += execution.completion_tokens
        self.functional_total_tokens += execution.total_tokens

    # ------------------------------------------------------------------
    # Computed properties for analysis
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        """All tokens (functional + coordination)."""
        return self.functional_total_tokens + self.coordination_total_tokens

    @property
    def total_prompt_tokens(self) -> int:
        return self.functional_prompt_tokens + self.coordination_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self.functional_completion_tokens + self.coordination_completion_tokens

    @property
    def coordination_call_count(self) -> int:
        return len(self.coordination_calls)

    @property
    def agent_execution_count(self) -> int:
        return len(self.agent_executions)

    @property
    def coordination_to_functional_ratio(self) -> float:
        """RQ3 key metric: ratio of coordination tokens to functional tokens.

        Returns 0.0 when there are no functional tokens (avoids division by zero).
        A ratio of 0.0 means zero coordination overhead (e.g. sequential pattern).
        """
        if self.functional_total_tokens == 0:
            return 0.0
        return round(
            self.coordination_total_tokens / self.functional_total_tokens, 4
        )

    @property
    def agent_total_seconds(self) -> float:
        """Sum of all agent execution durations."""
        return round(sum(e.duration_seconds for e in self.agent_executions), 4)

    @property
    def coordination_overhead_seconds(self) -> float:
        """Wall-clock time minus agent execution time."""
        return round(
            max(self.total_duration_seconds - self.agent_total_seconds, 0.0), 4
        )

    @property
    def coordination_latency_seconds(self) -> float:
        """Sum of all coordination LLM call latencies."""
        return round(sum(c.latency_seconds for c in self.coordination_calls), 4)

    def cost_estimate(
        self,
        prompt_cost_per_1k: float = 0.005,
        completion_cost_per_1k: float = 0.015,
    ) -> float:
        """Rough cost estimate at configurable $/1K-token rates.

        Default rates are approximate GPT-4o-mini pricing.
        """
        prompt_cost = self.total_prompt_tokens / 1000.0 * prompt_cost_per_1k
        completion_cost = self.total_completion_tokens / 1000.0 * completion_cost_per_1k
        return round(prompt_cost + completion_cost, 6)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a flat + nested dict suitable for JSON output."""
        return {
            # Identity
            "pattern": self.pattern,
            "task_id": self.task_id,
            "success": self.success,
            "error": self.error,
            # RQ1
            "total_duration_seconds": self.total_duration_seconds,
            "agent_total_seconds": self.agent_total_seconds,
            "agent_execution_count": self.agent_execution_count,
            "agent_executions": [
                {
                    "role": e.role,
                    "status": e.status,
                    "attempt": e.attempt,
                    "duration_seconds": e.duration_seconds,
                    "prompt_tokens": e.prompt_tokens,
                    "completion_tokens": e.completion_tokens,
                    "total_tokens": e.total_tokens,
                    "artifact_count": e.artifact_count,
                    "attachment_count": e.attachment_count,
                }
                for e in self.agent_executions
            ],
            # RQ2
            "functional_prompt_tokens": self.functional_prompt_tokens,
            "functional_completion_tokens": self.functional_completion_tokens,
            "functional_total_tokens": self.functional_total_tokens,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "cost_estimate": self.cost_estimate(),
            # RQ3
            "coordination_call_count": self.coordination_call_count,
            "coordination_prompt_tokens": self.coordination_prompt_tokens,
            "coordination_completion_tokens": self.coordination_completion_tokens,
            "coordination_total_tokens": self.coordination_total_tokens,
            "coordination_to_functional_ratio": self.coordination_to_functional_ratio,
            "coordination_overhead_seconds": self.coordination_overhead_seconds,
            "coordination_latency_seconds": self.coordination_latency_seconds,
            "coordination_calls": [
                {
                    "purpose": c.purpose,
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "total_tokens": c.total_tokens,
                    "latency_seconds": c.latency_seconds,
                    "iteration": c.iteration,
                    "parsed_result": c.parsed_result,
                }
                for c in self.coordination_calls
            ],
        }

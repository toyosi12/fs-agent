"""Centralized orchestration pattern — LLM-driven coordinator loop.

The coordinator LLM decides which agent runs next.  **No fallbacks** —
if the coordinator returns an invalid response or the LLM call fails,
the run is aborted with :class:`OrchestrationError`.
"""

from __future__ import annotations

import json
import time
from typing import Iterable, Sequence

from ...context import AgentReport, RunContext
from ...llm import BaseLLMClient
from ...logger import get_logger
from ..base import OrchestrationError, OrchestrationPattern
from ..metrics import CoordinationCall
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent, run_validation_loop


class CentralizedOrchestrator(OrchestrationPattern):
    """Uses the LLM as a coordinator to decide which agent runs next.

    Instead of a hardcoded sequence the coordinator loop asks the LLM:
    "Given what has been completed so far, which agent should run next —
    or are we done?"  The chosen agent is dispatched through the same
    registry and context machinery used by the sequential pattern.

    **Research mode**: all fallbacks removed — failures raise
    :class:`OrchestrationError`.
    """

    MAX_ITERATIONS = 10

    def __init__(
        self,
        registry: AgentRegistry,
        llm: BaseLLMClient,
        *,
        max_iterations: int | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.max_iterations = max_iterations or self.MAX_ITERATIONS
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, context: RunContext) -> Sequence[AgentReport]:
        m = context.metrics
        m.pattern = "centralized"
        m.task_id = getattr(context.settings, "task_id", "")
        m.start_timer()

        reports: list[AgentReport] = []
        available = [role.value for role in AgentRole]

        self.logger.info(
            "╔══ CENTRALIZED ORCHESTRATOR START ══╗  max_iterations=%d  agents=%s",
            self.max_iterations,
            ", ".join(available),
        )

        try:
            for iteration in range(1, self.max_iterations + 1):
                self.logger.info(
                    "── iteration %d/%d ──────────────────────────────────────────",
                    iteration,
                    self.max_iterations,
                )

                decision, coord_call = self._ask_coordinator(context, available, iteration)
                m.record_coordination_call(coord_call)

                action = decision.get("action", "done")
                agent_name = decision.get("agent", "")
                reason = decision.get("reason", "")

                self.logger.info(
                    "  coordinator decision: action=%s  agent=%s  reason=%s  "
                    "tokens=%d  latency=%.2fs",
                    action,
                    agent_name,
                    reason[:120],
                    coord_call.total_tokens,
                    coord_call.latency_seconds,
                )

                if action == "done":
                    self.logger.info(
                        "  coordinator signalled DONE at iteration %d: %s",
                        iteration,
                        reason,
                    )
                    break

                # Resolve the agent role — NO FALLBACK
                try:
                    role = AgentRole(agent_name)
                except ValueError:
                    raise OrchestrationError(
                        "centralized",
                        f"Coordinator returned unknown agent '{agent_name}' "
                        f"(valid: {available})",
                        context={"iteration": iteration, "raw_decision": decision},
                    )

                # Dispatch
                agent = self.registry.build(role)
                report, execution = execute_agent(agent, role, context)
                m.record_agent_execution(execution)
                reports.append(report)
            else:
                raise OrchestrationError(
                    "centralized",
                    f"Coordinator hit max iterations ({self.max_iterations}) "
                    f"without signalling done.  Agents completed: "
                    f"{[r.role for r in reports]}",
                    context={"max_iterations": self.max_iterations},
                )

            m.success = True

            # --- Validation loop ---
            reports = run_validation_loop(
                context, self.registry, reports, m,
                pattern_name="centralized",
            )

        except OrchestrationError:
            m.success = False
            raise
        except Exception as exc:
            m.success = False
            m.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            m.stop_timer()
            self._log_summary(m, reports)

        return reports

    # ------------------------------------------------------------------
    # Coordinator LLM interaction — NO FALLBACK
    # ------------------------------------------------------------------

    def _ask_coordinator(
        self,
        context: RunContext,
        available: list[str],
        iteration: int,
    ) -> tuple[dict[str, str], CoordinationCall]:
        """Ask the LLM which agent to run next (or declare done).

        Returns ``(parsed_decision, CoordinationCall)``.
        Raises :class:`OrchestrationError` on any failure.
        """
        prompt = self._build_coordinator_prompt(context, available)
        system = (
            "You are an orchestration coordinator for a multi-agent code-generation "
            "system. You decide which specialist agent should run next based on what "
            "has already been completed. Respond with JSON only — no markdown fences, "
            "no commentary."
        )

        self.logger.debug(
            "  coordinator prompt (%d chars):\n%s", len(prompt), prompt[:500]
        )

        pre = self.llm.usage_stats.copy()
        t0 = time.perf_counter()

        try:
            raw = self.llm.generate(prompt, system=system, temperature=0.0)
        except Exception as exc:
            raise OrchestrationError(
                "centralized",
                f"Coordinator LLM call failed at iteration {iteration}: {exc}",
                context={"iteration": iteration},
            ) from exc

        latency = time.perf_counter() - t0
        post = self.llm.usage_stats

        self.logger.debug("  coordinator raw response:\n%s", raw[:500])

        try:
            decision = self._parse_decision(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise OrchestrationError(
                "centralized",
                f"Coordinator returned unparseable response at iteration "
                f"{iteration}: {exc}",
                context={"iteration": iteration, "raw_response": raw[:500]},
            ) from exc

        coord_call = CoordinationCall(
            purpose=f"coordinator_decision_iter_{iteration}",
            prompt_tokens=post["prompt_tokens"] - pre["prompt_tokens"],
            completion_tokens=post["completion_tokens"] - pre["completion_tokens"],
            total_tokens=post["total_tokens"] - pre["total_tokens"],
            latency_seconds=round(latency, 4),
            raw_response=raw,
            parsed_result=decision,
            iteration=iteration,
        )

        return decision, coord_call

    def _build_coordinator_prompt(
        self, context: RunContext, available: list[str]
    ) -> str:
        completed = []
        for report in context.transcripts:
            completed.append(
                {
                    "agent": report.role,
                    "status": report.status,
                    "summary": report.summary[:200],
                }
            )
        completed_names = {r.role for r in context.transcripts}
        remaining = [a for a in available if a not in completed_names]

        return (
            f"User request:\n{context.user_request}\n\n"
            f"Available agents: {json.dumps(available)}\n"
            f"Already completed: {json.dumps(completed, indent=2)}\n"
            f"Not yet run: {json.dumps(remaining)}\n\n"
            "Rules:\n"
            "- 'architect' must run first if it has not run yet.\n"
            "- 'backend' and 'frontend' require 'architect' to have completed.\n"
            "- 'infra' requires 'backend' and 'frontend' to have completed.\n"
            "- Once all necessary agents have run, respond with action 'done'.\n\n"
            "Respond with exactly one JSON object:\n"
            '  { "action": "run", "agent": "<agent_name>", "reason": "..." }\n'
            "or\n"
            '  { "action": "done", "reason": "..." }\n'
        )

    def _parse_decision(self, raw: str) -> dict[str, str]:
        """Parse the coordinator's JSON response."""
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
        text = text.strip()
        data = json.loads(text)
        if not isinstance(data, dict) or "action" not in data:
            raise ValueError(f"Invalid coordinator response (missing 'action'): {data}")
        return data

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_summary(self, m: object, reports: list[AgentReport]) -> None:
        self.logger.info(
            "╚══ CENTRALIZED ORCHESTRATOR END ════╝\n"
            "  success=%s  duration=%.2fs  agents_run=%d\n"
            "  coordination_calls=%d  coordination_tokens=%d  (prompt=%d, completion=%d)\n"
            "  functional_tokens=%d  (prompt=%d, completion=%d)\n"
            "  coordination/functional ratio=%.4f\n"
            "  total_tokens=%d  est_cost=$%.6f",
            m.success,
            m.total_duration_seconds,
            m.agent_execution_count,
            m.coordination_call_count,
            m.coordination_total_tokens,
            m.coordination_prompt_tokens,
            m.coordination_completion_tokens,
            m.functional_total_tokens,
            m.functional_prompt_tokens,
            m.functional_completion_tokens,
            m.coordination_to_functional_ratio,
            m.total_tokens,
            m.cost_estimate(),
        )

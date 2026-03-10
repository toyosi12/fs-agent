"""Decentralized orchestration pattern — agent-driven handoff routing.

Each completing agent decides who runs next via an LLM handoff call.
**No fallbacks** — if a handoff fails or returns an invalid agent the
run is aborted with :class:`OrchestrationError`.
"""

from __future__ import annotations

import json
import time
from typing import Sequence

from ...context import AgentReport, RunContext
from ...llm import BaseLLMClient
from ...logger import get_logger
from ..base import OrchestrationError, OrchestrationPattern
from ..metrics import CoordinationCall
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent


class DecentralizedOrchestrator(OrchestrationPattern):
    """Each agent decides who runs next via an LLM handoff call.

    Routing intelligence is distributed: after every agent completes, the
    orchestrator asks the *outgoing* agent's LLM "given what you just
    produced, who should handle this next?"

    The seed agent is always ``architect``.  Execution continues until an
    agent hands off to ``"done"`` or all agents have run.

    **Research mode**: all fallbacks removed — failures raise
    :class:`OrchestrationError`.
    """

    MAX_ITERATIONS = 10
    SEED_ROLE = AgentRole.ARCHITECT

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
        m.pattern = "decentralized"
        m.task_id = getattr(context.settings, "task_id", "")
        m.start_timer()

        reports: list[AgentReport] = []
        completed: set[str] = set()
        all_roles = [role.value for role in AgentRole]

        self.logger.info(
            "╔══ DECENTRALIZED ORCHESTRATOR START ══╗  max_iterations=%d  agents=%s",
            self.max_iterations,
            ", ".join(all_roles),
        )

        current_role = self.SEED_ROLE

        try:
            for iteration in range(1, self.max_iterations + 1):
                self.logger.info(
                    "── iteration %d/%d ──────────────────────────────────────────",
                    iteration,
                    self.max_iterations,
                )

                # Already-ran guard — NO FALLBACK
                if current_role.value in completed:
                    raise OrchestrationError(
                        "decentralized",
                        f"Handoff cycle detected: agent '{current_role.value}' "
                        f"was selected again.  Completed so far: {sorted(completed)}",
                        context={"iteration": iteration, "completed": sorted(completed)},
                    )

                self.logger.info(
                    "  dispatching agent '%s'",
                    current_role.value,
                )

                # Dispatch the current agent
                agent = self.registry.build(current_role)
                report, execution = execute_agent(agent, current_role, context)
                m.record_agent_execution(execution)
                reports.append(report)
                completed.add(current_role.value)

                # All agents done?
                if completed >= set(all_roles):
                    self.logger.info(
                        "  all agents completed after %d iterations", iteration
                    )
                    break

                # Ask for a handoff decision
                handoff, coord_call = self._ask_handoff(
                    current_role, report, all_roles, completed, iteration
                )
                m.record_coordination_call(coord_call)

                next_agent = handoff.get("next", "done")
                reason = handoff.get("reason", "")

                self.logger.info(
                    "  handoff from '%s': next=%s  reason=%s  tokens=%d  latency=%.2fs",
                    current_role.value,
                    next_agent,
                    reason[:120],
                    coord_call.total_tokens,
                    coord_call.latency_seconds,
                )

                if next_agent == "done":
                    self.logger.info(
                        "  agent '%s' signalled DONE at iteration %d: %s",
                        current_role.value,
                        iteration,
                        reason,
                    )
                    # Verify all agents were run
                    not_run = set(all_roles) - completed
                    if not_run:
                        raise OrchestrationError(
                            "decentralized",
                            f"Handoff signalled 'done' but agents {sorted(not_run)} "
                            f"never ran.  Completed: {sorted(completed)}",
                            context={"iteration": iteration, "not_run": sorted(not_run)},
                        )
                    break

                # Resolve the handoff target — NO FALLBACK
                try:
                    current_role = AgentRole(next_agent)
                except ValueError:
                    raise OrchestrationError(
                        "decentralized",
                        f"Handoff returned unknown agent '{next_agent}' "
                        f"(valid: {all_roles})",
                        context={"iteration": iteration, "raw_handoff": handoff},
                    )
            else:
                raise OrchestrationError(
                    "decentralized",
                    f"Hit max iterations ({self.max_iterations}) without completing.  "
                    f"Agents completed: {sorted(completed)}",
                    context={"max_iterations": self.max_iterations},
                )

            m.success = True

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
    # Handoff LLM interaction — NO FALLBACK
    # ------------------------------------------------------------------

    def _ask_handoff(
        self,
        completed_role: AgentRole,
        report: AgentReport,
        all_roles: list[str],
        completed: set[str],
        iteration: int,
    ) -> tuple[dict[str, str], CoordinationCall]:
        """Ask the LLM who should run next.

        Returns ``(parsed_handoff, CoordinationCall)``.
        Raises :class:`OrchestrationError` on any failure.
        """
        prompt = self._build_handoff_prompt(completed_role, report, all_roles, completed)
        system = (
            "You are a routing advisor for a multi-agent code-generation system. "
            f"The '{completed_role.value}' agent just finished. Based on its output, "
            "decide which agent should run next — or if the pipeline is done. "
            "Respond with JSON only — no markdown fences, no commentary."
        )

        self.logger.debug(
            "  handoff prompt (%d chars):\n%s", len(prompt), prompt[:500]
        )

        pre = self.llm.usage_stats.copy()
        t0 = time.perf_counter()

        try:
            raw = self.llm.generate(prompt, system=system, temperature=0.0)
        except Exception as exc:
            raise OrchestrationError(
                "decentralized",
                f"Handoff LLM call failed after agent '{completed_role.value}' "
                f"at iteration {iteration}: {exc}",
                context={"iteration": iteration, "from_agent": completed_role.value},
            ) from exc

        latency = time.perf_counter() - t0
        post = self.llm.usage_stats

        self.logger.debug("  handoff raw response:\n%s", raw[:500])

        try:
            handoff = self._parse_handoff(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise OrchestrationError(
                "decentralized",
                f"Handoff returned unparseable response after agent "
                f"'{completed_role.value}' at iteration {iteration}: {exc}",
                context={"iteration": iteration, "raw_response": raw[:500]},
            ) from exc

        coord_call = CoordinationCall(
            purpose=f"handoff_from_{completed_role.value}_iter_{iteration}",
            prompt_tokens=post["prompt_tokens"] - pre["prompt_tokens"],
            completion_tokens=post["completion_tokens"] - pre["completion_tokens"],
            total_tokens=post["total_tokens"] - pre["total_tokens"],
            latency_seconds=round(latency, 4),
            raw_response=raw,
            parsed_result=handoff,
            iteration=iteration,
        )

        return handoff, coord_call

    def _build_handoff_prompt(
        self,
        completed_role: AgentRole,
        report: AgentReport,
        all_roles: list[str],
        completed: set[str],
    ) -> str:
        remaining = [r for r in all_roles if r not in completed]

        return (
            f"Agent that just completed: {completed_role.value}\n"
            f"Status: {report.status}\n"
            f"Summary: {report.summary[:300]}\n\n"
            f"All agents: {json.dumps(all_roles)}\n"
            f"Already completed: {json.dumps(sorted(completed))}\n"
            f"Not yet run: {json.dumps(remaining)}\n\n"
            "Dependency rules:\n"
            "- 'architect' must run first.\n"
            "- 'backend' and 'frontend' require 'architect' to have completed.\n"
            "- 'infra' requires 'backend' and 'frontend' to have completed.\n"
            "- Once all necessary agents have run, respond with next 'done'.\n\n"
            "Respond with exactly one JSON object:\n"
            '  { "next": "<agent_name>", "reason": "..." }\n'
            "or\n"
            '  { "next": "done", "reason": "..." }\n'
        )

    def _parse_handoff(self, raw: str) -> dict[str, str]:
        """Parse the handoff JSON response."""
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
        text = text.strip()
        data = json.loads(text)
        if not isinstance(data, dict) or "next" not in data:
            raise ValueError(f"Invalid handoff response (missing 'next'): {data}")
        return data

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_summary(self, m: object, reports: list[AgentReport]) -> None:
        self.logger.info(
            "╚══ DECENTRALIZED ORCHESTRATOR END ════╝\n"
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

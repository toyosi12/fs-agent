"""Centralized orchestration pattern — LLM mediator loop.

The coordinator LLM decides which agent runs next **and** synthesizes
an integration brief from completed agents' actual output.  Each agent
receives a focused context summary produced by the mediator, not raw
artifacts.  This adds one extra LLM call per agent transition compared
to the sequential pipeline.
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
from .._helpers import execute_agent, run_fixer_loop, run_validation_loop


class CentralizedOrchestrator(OrchestrationPattern):
    """Uses the LLM as a mediator to decide order AND synthesize context.

    Each iteration the coordinator:
    1. Decides which agent runs next (or declares done).
    2. Synthesizes an *integration brief* from all completed agents'
       real output and injects it into ``context.extra_context``.

    This means each agent sees a mediator-curated summary rather than
    raw contract extractions.  The extra synthesis call is recorded as
    coordination overhead.
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
        completed_roles: set[str] = set()
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

                if role.value in completed_roles:
                    raise OrchestrationError(
                        "centralized",
                        f"Coordinator selected already-completed agent '{role.value}'. "
                        f"Completed: {sorted(completed_roles)}",
                        context={"iteration": iteration, "raw_decision": decision},
                    )

                # Dispatch
                agent = self.registry.build(role)

                # Before dispatching, synthesize a mediator brief from completed
                # agents' contracts (the key differentiator from sequential pipeline)
                if context.transcripts:
                    brief, brief_call = self._synthesize_brief(
                        context, role, iteration
                    )
                    m.record_coordination_call(brief_call)
                    context.extra_context = {"upstream_context": brief}
                else:
                    context.extra_context = {}

                report, execution = execute_agent(agent, role, context)
                m.record_agent_execution(execution)
                reports.append(report)
                completed_roles.add(role.value)
            else:
                raise OrchestrationError(
                    "centralized",
                    f"Coordinator hit max iterations ({self.max_iterations}) "
                    f"without signalling done.  Agents completed: "
                    f"{[r.role for r in reports]}",
                    context={"max_iterations": self.max_iterations},
                )

            m.success = True

            # --- Fixer loop (fixer ↔ infra) ---
            reports, fixer_result = run_fixer_loop(
                context, self.registry, reports, m,
                pattern_name="centralized",
            )
            m.fixer_loop_result = fixer_result.to_dict()

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
            context.extra_context = {}
            m.stop_timer()
            self._log_summary(m, reports)

        return reports

    # ------------------------------------------------------------------
    # Mediator brief synthesis — the key differentiator
    # ------------------------------------------------------------------

    def _synthesize_brief(
        self,
        context: RunContext,
        next_role: AgentRole,
        iteration: int,
    ) -> tuple[str, CoordinationCall]:
        """Ask the LLM to produce a focused integration brief for *next_role*.

        The brief is synthesized from compact contract extractions of all
        completed agents — NOT from full source code.
        """
        contracts: list[str] = []
        backend_contract = context.extract_backend_contract()
        if backend_contract:
            contracts.append(f"[backend contract]\n{backend_contract}")
        frontend_contract = context.extract_frontend_contract()
        if frontend_contract:
            contracts.append(f"[frontend contract]\n{frontend_contract}")

        completed_summary = "\n---\n".join(contracts) if contracts else "(none yet)"

        prompt = (
            f"You are a mediator for a multi-agent code-generation system.\n"
            f"The next agent to run is: {next_role.value}\n\n"
            f"Completed agents' output contracts:\n{completed_summary}\n\n"
            f"User request:\n{context.user_request}\n\n"
            f"Write a concise integration brief (max 400 words) for the "
            f"{next_role.value} agent. Focus on:\n"
            f"- Exact API endpoints/routes it must integrate with\n"
            f"- Data shapes and field names it must match\n"
            f"- Port numbers and service URLs\n"
            f"- Any constraints from upstream agents\n\n"
            f"Be specific and actionable. Output plain text, no JSON."
        )

        system = (
            "You are a technical mediator synthesizing integration context "
            "for downstream agents. Be precise and concise."
        )

        pre = self.llm.usage_stats.copy()
        t0 = time.perf_counter()

        try:
            brief = self.llm.generate(prompt, system=system, temperature=0.0)
        except Exception as exc:
            raise OrchestrationError(
                "centralized",
                f"Mediator brief synthesis failed at iteration {iteration}: {exc}",
                context={"iteration": iteration, "next_role": next_role.value},
            ) from exc

        latency = time.perf_counter() - t0
        post = self.llm.usage_stats

        coord_call = CoordinationCall(
            purpose=f"mediator_brief_for_{next_role.value}_iter_{iteration}",
            prompt_tokens=post["prompt_tokens"] - pre["prompt_tokens"],
            completion_tokens=post["completion_tokens"] - pre["completion_tokens"],
            total_tokens=post["total_tokens"] - pre["total_tokens"],
            latency_seconds=round(latency, 4),
            raw_response=brief[:500],
            parsed_result={"brief_length": len(brief)},
            iteration=iteration,
        )

        self.logger.info(
            "  mediator brief for %s: %d chars, %d tokens, %.2fs",
            next_role.value,
            len(brief),
            coord_call.total_tokens,
            coord_call.latency_seconds,
        )

        return brief.strip(), coord_call

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

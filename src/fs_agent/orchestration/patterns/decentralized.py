"""Decentralized orchestration pattern — agent-driven handoff routing."""

from __future__ import annotations

import json
from typing import Iterable

from ...context import AgentReport, RunContext
from ...llm import BaseLLMClient
from ...logger import get_logger
from ..base import OrchestrationPattern
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent


class DecentralizedOrchestrator(OrchestrationPattern):
    """Each agent decides who runs next via an LLM handoff call.

    Unlike the centralized pattern (which uses a global coordinator loop),
    routing intelligence is distributed: after every agent completes, the
    orchestrator asks the *outgoing* agent's LLM "given what you just
    produced, who should handle this next?"  The handoff response drives
    the next dispatch.

    The seed agent is always ``architect`` (it depends only on the user
    request).  Execution continues until an agent hands off to ``"done"``,
    all agents have run, or the max-iterations guard fires.
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

    def run(self, context: RunContext) -> Iterable[AgentReport]:
        reports: list[AgentReport] = []
        completed: set[str] = set()
        all_roles = [role.value for role in AgentRole]

        self.logger.info(
            "Decentralized orchestrator starting (max %d iterations, agents: %s)",
            self.max_iterations,
            ", ".join(all_roles),
        )

        # --- Seed: always start with the architect ---
        current_role = self.SEED_ROLE

        for iteration in range(1, self.max_iterations + 1):
            if current_role.value in completed:
                self.logger.info(
                    "Agent '%s' already ran; running remaining sequentially",
                    current_role.value,
                )
                reports.extend(self._run_remaining(context, completed))
                break

            self.logger.info(
                "Iteration %d — running agent '%s'",
                iteration,
                current_role.value,
            )

            # Dispatch the current agent
            agent = self.registry.build(current_role)
            report = execute_agent(agent, current_role, context)
            reports.append(report)
            completed.add(current_role.value)

            # Check if all agents have run
            if completed >= set(all_roles):
                self.logger.info("All agents completed after %d iterations", iteration)
                break

            # Ask the just-completed agent for a handoff decision
            handoff = self._ask_handoff(current_role, report, all_roles, completed)
            next_agent = handoff.get("next", "done")
            reason = handoff.get("reason", "")

            self.logger.info(
                "Iteration %d — handoff from '%s': next=%s reason=%s",
                iteration,
                current_role.value,
                next_agent,
                reason[:120],
            )

            if next_agent == "done":
                self.logger.info(
                    "Agent '%s' signalled done after %d iterations: %s",
                    current_role.value,
                    iteration,
                    reason,
                )
                # Run any remaining agents that haven't been dispatched
                remaining = self._run_remaining(context, completed)
                reports.extend(remaining)
                break

            # Resolve the handoff target
            try:
                current_role = AgentRole(next_agent)
            except ValueError:
                self.logger.warning(
                    "Handoff returned unknown agent '%s'; running remaining sequentially",
                    next_agent,
                )
                reports.extend(self._run_remaining(context, completed))
                break
        else:
            self.logger.warning(
                "Hit max iterations (%d); running remaining agents sequentially",
                self.max_iterations,
            )
            reports.extend(self._run_remaining(context, completed))

        self.logger.info(
            "Decentralized orchestrator complete: %d stages executed", len(reports)
        )
        return reports

    # ------------------------------------------------------------------
    # Handoff LLM interaction
    # ------------------------------------------------------------------

    def _ask_handoff(
        self,
        completed_role: AgentRole,
        report: AgentReport,
        all_roles: list[str],
        completed: set[str],
    ) -> dict[str, str]:
        """Ask the LLM who should run next based on the just-completed agent's output."""
        prompt = self._build_handoff_prompt(completed_role, report, all_roles, completed)
        system = (
            "You are a routing advisor for a multi-agent code-generation system. "
            f"The '{completed_role.value}' agent just finished. Based on its output, "
            "decide which agent should run next — or if the pipeline is done. "
            "Respond with JSON only — no markdown fences, no commentary."
        )
        try:
            raw = self.llm.generate(prompt, system=system, temperature=0.0)
            return self._parse_handoff(raw)
        except Exception as exc:
            self.logger.warning(
                "Handoff LLM call failed (%s); using fallback order", exc
            )
            return self._fallback_handoff(completed)

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
        # Strip markdown code fences if present
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
        text = text.strip()
        data = json.loads(text)
        if not isinstance(data, dict) or "next" not in data:
            raise ValueError(f"Invalid handoff response: {data}")
        return data

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback_handoff(self, completed: set[str]) -> dict[str, str]:
        """Deterministic fallback: pick the next agent in canonical order."""
        canonical = [AgentRole.ARCHITECT, AgentRole.BACKEND, AgentRole.FRONTEND, AgentRole.INFRA]
        for role in canonical:
            if role.value not in completed:
                return {"next": role.value, "reason": "fallback order"}
        return {"next": "done", "reason": "all agents completed (fallback)"}

    def _run_remaining(
        self, context: RunContext, completed: set[str]
    ) -> list[AgentReport]:
        """Run any agents that haven't executed yet, in canonical order."""
        canonical = [AgentRole.ARCHITECT, AgentRole.BACKEND, AgentRole.FRONTEND, AgentRole.INFRA]
        reports: list[AgentReport] = []
        for role in canonical:
            if role.value in completed:
                continue
            agent = self.registry.build(role)
            report = execute_agent(agent, role, context)
            reports.append(report)
            completed.add(role.value)
        return reports

"""Centralized orchestration pattern — LLM-driven coordinator loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from ...artifact_writer import persist_agent_output
from ...context import AgentReport, RunContext
from ...llm import BaseLLMClient
from ...logger import get_logger
from ..base import OrchestrationPattern
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent


class CentralizedOrchestrator(OrchestrationPattern):
    """Uses the LLM as a coordinator to decide which agent runs next.

    Instead of a hardcoded sequence the coordinator loop asks the LLM:
    "Given what has been completed so far, which agent should run next —
    or are we done?"  The chosen agent is dispatched through the same
    registry and context machinery used by the sequential pattern.
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

    def run(self, context: RunContext) -> Iterable[AgentReport]:
        reports: list[AgentReport] = []
        available = [role.value for role in AgentRole]
        self.logger.info(
            "Centralized orchestrator starting (max %d iterations, agents: %s)",
            self.max_iterations,
            ", ".join(available),
        )

        for iteration in range(1, self.max_iterations + 1):
            decision = self._ask_coordinator(context, available)
            action = decision.get("action", "done")
            agent_name = decision.get("agent", "")
            reason = decision.get("reason", "")

            self.logger.info(
                "Iteration %d — coordinator decision: action=%s agent=%s reason=%s",
                iteration,
                action,
                agent_name,
                reason[:120],
            )

            if action == "done":
                self.logger.info(
                    "Coordinator signalled done after %d iterations: %s",
                    iteration,
                    reason,
                )
                break

            # Resolve the agent role
            try:
                role = AgentRole(agent_name)
            except ValueError:
                self.logger.warning(
                    "Coordinator returned unknown agent '%s'; falling back to sequential remainder",
                    agent_name,
                )
                reports.extend(self._run_remaining(context))
                break

            # Dispatch
            agent = self.registry.build(role)
            report = execute_agent(agent, role, context)
            reports.append(report)
        else:
            self.logger.warning(
                "Coordinator hit max iterations (%d); stopping",
                self.max_iterations,
            )

        self.logger.info(
            "Centralized orchestrator complete: %d stages executed", len(reports)
        )
        return reports

    # ------------------------------------------------------------------
    # Coordinator LLM interaction
    # ------------------------------------------------------------------

    def _ask_coordinator(
        self, context: RunContext, available: list[str]
    ) -> dict[str, str]:
        """Ask the LLM which agent to run next (or declare done)."""
        prompt = self._build_coordinator_prompt(context, available)
        system = (
            "You are an orchestration coordinator for a multi-agent code-generation "
            "system. You decide which specialist agent should run next based on what "
            "has already been completed. Respond with JSON only — no markdown fences, "
            "no commentary."
        )
        try:
            raw = self.llm.generate(prompt, system=system, temperature=0.0)
            return self._parse_decision(raw)
        except Exception as exc:
            self.logger.warning(
                "Coordinator LLM call failed (%s); falling back to sequential", exc
            )
            return self._fallback_decision(context, available)

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
        # Strip markdown code fences if present
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
        text = text.strip()
        data = json.loads(text)
        if not isinstance(data, dict) or "action" not in data:
            raise ValueError(f"Invalid coordinator response: {data}")
        return data

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback_decision(
        self, context: RunContext, available: list[str]
    ) -> dict[str, str]:
        """Deterministic fallback: pick the next agent in canonical order."""
        canonical = [AgentRole.ARCHITECT, AgentRole.BACKEND, AgentRole.FRONTEND, AgentRole.INFRA]
        completed_names = {r.role for r in context.transcripts}
        for role in canonical:
            if role.value not in completed_names and role.value in available:
                return {"action": "run", "agent": role.value, "reason": "fallback order"}
        return {"action": "done", "reason": "all agents completed (fallback)"}

    def _run_remaining(self, context: RunContext) -> list[AgentReport]:
        """Run any agents that haven't executed yet, in canonical order."""
        canonical = [AgentRole.ARCHITECT, AgentRole.BACKEND, AgentRole.FRONTEND, AgentRole.INFRA]
        completed_names = {r.role for r in context.transcripts}
        reports: list[AgentReport] = []
        for role in canonical:
            if role.value in completed_names:
                continue
            agent = self.registry.build(role)
            report = execute_agent(agent, role, context)
            reports.append(report)
        return reports

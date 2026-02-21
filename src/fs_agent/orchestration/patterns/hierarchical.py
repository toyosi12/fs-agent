"""Hierarchical orchestration pattern — two-level supervisor tree."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

from ...context import AgentReport, RunContext
from ...llm import BaseLLMClient
from ...logger import get_logger
from ..base import OrchestrationPattern
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent


@dataclass
class PhaseGroup:
    """A named group of agent roles managed by one supervisor."""

    name: str
    roles: list[AgentRole]
    description: str = ""


# Default topology: planning → build
DEFAULT_PHASES: list[PhaseGroup] = [
    PhaseGroup(
        name="planning",
        roles=[AgentRole.ARCHITECT],
        description="Generate the project specification from the user brief.",
    ),
    PhaseGroup(
        name="build",
        roles=[AgentRole.BACKEND, AgentRole.FRONTEND, AgentRole.INFRA],
        description="Scaffold code, create infrastructure, and deploy the project.",
    ),
]


class HierarchicalOrchestrator(OrchestrationPattern):
    """Two-level supervisor tree for agent orchestration.

    The *root supervisor* picks which **phase** to execute next.  Within
    each phase a *phase supervisor* decides which **agent** runs next.
    Both supervisors are LLM prompts — not agents — keeping the routing
    logic separate from the domain agents.

    Default hierarchy::

        Root Supervisor (LLM)
        ├─ planning   → [architect]
        └─ build      → [backend, frontend, infra]

    Custom topologies can be provided via the ``phases`` constructor arg.
    """

    MAX_ROOT_ITERATIONS = 10
    MAX_PHASE_ITERATIONS = 10

    def __init__(
        self,
        registry: AgentRegistry,
        llm: BaseLLMClient,
        *,
        phases: list[PhaseGroup] | None = None,
        max_root_iterations: int | None = None,
        max_phase_iterations: int | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.phases = phases or DEFAULT_PHASES
        self.max_root_iterations = max_root_iterations or self.MAX_ROOT_ITERATIONS
        self.max_phase_iterations = max_phase_iterations or self.MAX_PHASE_ITERATIONS
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, context: RunContext) -> Iterable[AgentReport]:
        reports: list[AgentReport] = []
        completed_phases: set[str] = set()
        completed_agents: set[str] = set()
        phase_names = [p.name for p in self.phases]

        self.logger.info(
            "Hierarchical orchestrator starting (phases: %s)",
            ", ".join(phase_names),
        )

        for root_iter in range(1, self.max_root_iterations + 1):
            decision = self._ask_root_supervisor(context, completed_phases, completed_agents)
            phase_name = decision.get("phase", "done")
            reason = decision.get("reason", "")

            self.logger.info(
                "Root iteration %d — phase=%s reason=%s",
                root_iter,
                phase_name,
                reason[:120],
            )

            if phase_name == "done":
                self.logger.info(
                    "Root supervisor signalled done after %d iterations: %s",
                    root_iter,
                    reason,
                )
                break

            # Find the phase group
            phase = self._find_phase(phase_name)
            if phase is None:
                self.logger.warning(
                    "Root supervisor returned unknown phase '%s'; running remaining",
                    phase_name,
                )
                reports.extend(self._run_remaining(context, completed_agents))
                break

            # Run the phase supervisor loop
            phase_reports = self._run_phase(context, phase, completed_agents)
            reports.extend(phase_reports)
            completed_phases.add(phase_name)

            # If all agents across all phases have run, stop
            all_roles = {r.value for p in self.phases for r in p.roles}
            if completed_agents >= all_roles:
                self.logger.info("All agents completed after phase '%s'", phase_name)
                break
        else:
            self.logger.warning(
                "Root supervisor hit max iterations (%d); running remaining",
                self.max_root_iterations,
            )
            reports.extend(self._run_remaining(context, completed_agents))

        self.logger.info(
            "Hierarchical orchestrator complete: %d stages executed", len(reports)
        )
        return reports

    # ------------------------------------------------------------------
    # Phase supervisor
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        context: RunContext,
        phase: PhaseGroup,
        completed_agents: set[str],
    ) -> list[AgentReport]:
        """Run the phase-level supervisor loop for a single phase."""
        reports: list[AgentReport] = []
        phase_roles = [r.value for r in phase.roles]

        self.logger.info(
            "Phase '%s' supervisor starting (agents: %s)",
            phase.name,
            ", ".join(phase_roles),
        )

        for phase_iter in range(1, self.max_phase_iterations + 1):
            decision = self._ask_phase_supervisor(
                context, phase, completed_agents
            )
            agent_name = decision.get("agent", "done")
            reason = decision.get("reason", "")

            self.logger.info(
                "Phase '%s' iteration %d — agent=%s reason=%s",
                phase.name,
                phase_iter,
                agent_name,
                reason[:120],
            )

            if agent_name == "done":
                self.logger.info(
                    "Phase '%s' supervisor signalled done: %s",
                    phase.name,
                    reason,
                )
                break

            # Validate the agent belongs to this phase
            try:
                role = AgentRole(agent_name)
            except ValueError:
                self.logger.warning(
                    "Phase '%s' returned unknown agent '%s'; running remaining in phase",
                    phase.name,
                    agent_name,
                )
                reports.extend(
                    self._run_remaining_in_phase(context, phase, completed_agents)
                )
                break

            if role not in phase.roles:
                self.logger.warning(
                    "Agent '%s' does not belong to phase '%s'; skipping",
                    agent_name,
                    phase.name,
                )
                continue

            if role.value in completed_agents:
                self.logger.info(
                    "Agent '%s' already completed; skipping", agent_name
                )
                continue

            # Dispatch
            agent = self.registry.build(role)
            report = execute_agent(agent, role, context)
            reports.append(report)
            completed_agents.add(role.value)

            # Check if all agents in this phase are done
            if all(r.value in completed_agents for r in phase.roles):
                self.logger.info("All agents in phase '%s' completed", phase.name)
                break
        else:
            self.logger.warning(
                "Phase '%s' hit max iterations (%d); running remaining in phase",
                phase.name,
                self.max_phase_iterations,
            )
            reports.extend(
                self._run_remaining_in_phase(context, phase, completed_agents)
            )

        return reports

    # ------------------------------------------------------------------
    # Root supervisor LLM interaction
    # ------------------------------------------------------------------

    def _ask_root_supervisor(
        self,
        context: RunContext,
        completed_phases: set[str],
        completed_agents: set[str],
    ) -> dict[str, str]:
        """Ask the LLM which phase to execute next."""
        prompt = self._build_root_prompt(context, completed_phases, completed_agents)
        system = (
            "You are the root supervisor for a hierarchical multi-agent "
            "code-generation system. You decide which phase of work to "
            "execute next. Respond with JSON only — no markdown fences, "
            "no commentary."
        )
        try:
            raw = self.llm.generate(prompt, system=system, temperature=0.0)
            return self._parse_json_decision(raw, required_key="phase")
        except Exception as exc:
            self.logger.warning(
                "Root supervisor LLM call failed (%s); using fallback", exc
            )
            return self._fallback_root(completed_phases)

    def _build_root_prompt(
        self,
        context: RunContext,
        completed_phases: set[str],
        completed_agents: set[str],
    ) -> str:
        phases_info = []
        for p in self.phases:
            agents_status = []
            for r in p.roles:
                status = "completed" if r.value in completed_agents else "pending"
                agents_status.append(f"{r.value} ({status})")
            phases_info.append(
                {
                    "name": p.name,
                    "description": p.description,
                    "agents": agents_status,
                    "phase_completed": p.name in completed_phases,
                }
            )
        remaining_phases = [p.name for p in self.phases if p.name not in completed_phases]

        return (
            f"User request:\n{context.user_request}\n\n"
            f"Phases:\n{json.dumps(phases_info, indent=2)}\n\n"
            f"Completed phases: {json.dumps(sorted(completed_phases))}\n"
            f"Remaining phases: {json.dumps(remaining_phases)}\n\n"
            "Rules:\n"
            "- 'planning' must complete before 'build'.\n"
            "- Once all phases are done, respond with phase 'done'.\n\n"
            "Respond with exactly one JSON object:\n"
            '  { "phase": "<phase_name>", "reason": "..." }\n'
            "or\n"
            '  { "phase": "done", "reason": "..." }\n'
        )

    # ------------------------------------------------------------------
    # Phase supervisor LLM interaction
    # ------------------------------------------------------------------

    def _ask_phase_supervisor(
        self,
        context: RunContext,
        phase: PhaseGroup,
        completed_agents: set[str],
    ) -> dict[str, str]:
        """Ask the LLM which agent within a phase to run next."""
        prompt = self._build_phase_prompt(context, phase, completed_agents)
        system = (
            f"You are the '{phase.name}' phase supervisor in a hierarchical "
            "multi-agent system. You decide which agent in your phase runs "
            "next. Respond with JSON only — no markdown fences, no commentary."
        )
        try:
            raw = self.llm.generate(prompt, system=system, temperature=0.0)
            return self._parse_json_decision(raw, required_key="agent")
        except Exception as exc:
            self.logger.warning(
                "Phase '%s' supervisor LLM failed (%s); using fallback",
                phase.name,
                exc,
            )
            return self._fallback_phase(phase, completed_agents)

    def _build_phase_prompt(
        self,
        context: RunContext,
        phase: PhaseGroup,
        completed_agents: set[str],
    ) -> str:
        phase_agents = [r.value for r in phase.roles]
        completed_in_phase = [a for a in phase_agents if a in completed_agents]
        remaining_in_phase = [a for a in phase_agents if a not in completed_agents]

        completed_summaries = []
        for report in context.transcripts:
            if report.role in completed_in_phase:
                completed_summaries.append(
                    {"agent": report.role, "summary": report.summary[:200]}
                )

        return (
            f"Phase: {phase.name}\n"
            f"Phase description: {phase.description}\n\n"
            f"Agents in this phase: {json.dumps(phase_agents)}\n"
            f"Already completed: {json.dumps(completed_summaries, indent=2)}\n"
            f"Not yet run: {json.dumps(remaining_in_phase)}\n\n"
            "Dependency rules:\n"
            "- 'backend' and 'frontend' can run in any order.\n"
            "- 'infra' requires 'backend' and 'frontend' to have completed.\n"
            "- Once all agents in the phase are done, respond with agent 'done'.\n\n"
            "Respond with exactly one JSON object:\n"
            '  { "agent": "<agent_name>", "reason": "..." }\n'
            "or\n"
            '  { "agent": "done", "reason": "..." }\n'
        )

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_json_decision(self, raw: str, *, required_key: str) -> dict[str, str]:
        """Parse a supervisor JSON response."""
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
        text = text.strip()
        data = json.loads(text)
        if not isinstance(data, dict) or required_key not in data:
            raise ValueError(f"Invalid supervisor response (missing '{required_key}'): {data}")
        return data

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback_root(self, completed_phases: set[str]) -> dict[str, str]:
        """Pick the next phase in declaration order."""
        for phase in self.phases:
            if phase.name not in completed_phases:
                return {"phase": phase.name, "reason": "fallback order"}
        return {"phase": "done", "reason": "all phases completed (fallback)"}

    def _fallback_phase(
        self, phase: PhaseGroup, completed_agents: set[str]
    ) -> dict[str, str]:
        """Pick the next agent within a phase in declaration order."""
        for role in phase.roles:
            if role.value not in completed_agents:
                return {"agent": role.value, "reason": "fallback order"}
        return {"agent": "done", "reason": "all phase agents completed (fallback)"}

    def _find_phase(self, name: str) -> PhaseGroup | None:
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None

    def _run_remaining_in_phase(
        self,
        context: RunContext,
        phase: PhaseGroup,
        completed_agents: set[str],
    ) -> list[AgentReport]:
        """Run un-dispatched agents within a phase in declaration order."""
        reports: list[AgentReport] = []
        for role in phase.roles:
            if role.value in completed_agents:
                continue
            agent = self.registry.build(role)
            report = execute_agent(agent, role, context)
            reports.append(report)
            completed_agents.add(role.value)
        return reports

    def _run_remaining(
        self, context: RunContext, completed_agents: set[str]
    ) -> list[AgentReport]:
        """Run all un-dispatched agents across all phases in declaration order."""
        reports: list[AgentReport] = []
        for phase in self.phases:
            reports.extend(
                self._run_remaining_in_phase(context, phase, completed_agents)
            )
        return reports

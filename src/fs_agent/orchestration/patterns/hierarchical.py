"""Hierarchical orchestration pattern — two-level supervisor tree.

Root supervisor picks which **phase** to execute; within each phase a
phase supervisor picks which **agent** runs next.  **No fallbacks** —
if any supervisor call fails the run is aborted with
:class:`OrchestrationError`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Sequence

from ...context import AgentReport, RunContext
from ...llm import BaseLLMClient
from ...logger import get_logger
from ..base import OrchestrationError, OrchestrationPattern
from ..metrics import CoordinationCall
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent, run_validation_loop


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

    Default hierarchy::

        Root Supervisor (LLM)
        ├─ planning   → [architect]
        └─ build      → [backend, frontend, infra]

    **Research mode**: all fallbacks removed — failures raise
    :class:`OrchestrationError`.
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

    def run(self, context: RunContext) -> Sequence[AgentReport]:
        m = context.metrics
        m.pattern = "hierarchical"
        m.task_id = getattr(context.settings, "task_id", "")
        m.start_timer()

        reports: list[AgentReport] = []
        completed_phases: set[str] = set()
        completed_agents: set[str] = set()
        phase_names = [p.name for p in self.phases]

        self.logger.info(
            "╔══ HIERARCHICAL ORCHESTRATOR START ══╗  phases=%s",
            ", ".join(phase_names),
        )

        try:
            for root_iter in range(1, self.max_root_iterations + 1):
                self.logger.info(
                    "── root iteration %d/%d ────────────────────────────────────",
                    root_iter,
                    self.max_root_iterations,
                )

                decision, coord_call = self._ask_root_supervisor(
                    context, completed_phases, completed_agents, root_iter
                )
                m.record_coordination_call(coord_call)

                phase_name = decision.get("phase", "done")
                reason = decision.get("reason", "")

                self.logger.info(
                    "  root decision: phase=%s  reason=%s  tokens=%d  latency=%.2fs",
                    phase_name,
                    reason[:120],
                    coord_call.total_tokens,
                    coord_call.latency_seconds,
                )

                if phase_name == "done":
                    self.logger.info(
                        "  root supervisor signalled DONE at iteration %d: %s",
                        root_iter,
                        reason,
                    )
                    break

                # Find the phase group — NO FALLBACK
                phase = self._find_phase(phase_name)
                if phase is None:
                    raise OrchestrationError(
                        "hierarchical",
                        f"Root supervisor returned unknown phase '{phase_name}' "
                        f"(valid: {phase_names})",
                        context={"root_iter": root_iter, "raw_decision": decision},
                    )

                # Run the phase supervisor loop
                phase_reports = self._run_phase(context, phase, completed_agents)
                reports.extend(phase_reports)
                completed_phases.add(phase_name)

                # All agents across all phases done?
                all_roles = {r.value for p in self.phases for r in p.roles}
                if completed_agents >= all_roles:
                    self.logger.info(
                        "  all agents completed after phase '%s'", phase_name
                    )
                    break
            else:
                raise OrchestrationError(
                    "hierarchical",
                    f"Root supervisor hit max iterations ({self.max_root_iterations}) "
                    f"without signalling done.  Completed phases: {sorted(completed_phases)}",
                    context={"max_root_iterations": self.max_root_iterations},
                )

            m.success = True

            # --- Validation loop ---
            reports = run_validation_loop(
                context, self.registry, reports, m,
                pattern_name="hierarchical",
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
    # Phase supervisor — NO FALLBACK
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        context: RunContext,
        phase: PhaseGroup,
        completed_agents: set[str],
    ) -> list[AgentReport]:
        """Run the phase-level supervisor loop for a single phase."""
        m = context.metrics
        reports: list[AgentReport] = []
        phase_roles = [r.value for r in phase.roles]

        self.logger.info(
            "  ┌── phase '%s' supervisor start  agents=%s",
            phase.name,
            ", ".join(phase_roles),
        )

        for phase_iter in range(1, self.max_phase_iterations + 1):
            decision, coord_call = self._ask_phase_supervisor(
                context, phase, completed_agents, phase_iter
            )
            m.record_coordination_call(coord_call)

            agent_name = decision.get("agent", "done")
            reason = decision.get("reason", "")

            self.logger.info(
                "  │ phase '%s' iter %d: agent=%s  reason=%s  "
                "tokens=%d  latency=%.2fs",
                phase.name,
                phase_iter,
                agent_name,
                reason[:120],
                coord_call.total_tokens,
                coord_call.latency_seconds,
            )

            if agent_name == "done":
                self.logger.info(
                    "  │ phase '%s' supervisor signalled DONE: %s",
                    phase.name,
                    reason,
                )
                break

            # Validate agent — NO FALLBACK
            try:
                role = AgentRole(agent_name)
            except ValueError:
                raise OrchestrationError(
                    "hierarchical",
                    f"Phase '{phase.name}' supervisor returned unknown agent "
                    f"'{agent_name}' (valid in phase: {phase_roles})",
                    context={"phase": phase.name, "phase_iter": phase_iter},
                )

            if role not in phase.roles:
                raise OrchestrationError(
                    "hierarchical",
                    f"Agent '{agent_name}' does not belong to phase '{phase.name}' "
                    f"(phase agents: {phase_roles})",
                    context={"phase": phase.name, "phase_iter": phase_iter},
                )

            if role.value in completed_agents:
                raise OrchestrationError(
                    "hierarchical",
                    f"Phase '{phase.name}' supervisor selected already-completed "
                    f"agent '{agent_name}'",
                    context={"phase": phase.name, "phase_iter": phase_iter,
                             "completed": sorted(completed_agents)},
                )

            # Dispatch
            agent = self.registry.build(role)
            report, execution = execute_agent(agent, role, context)
            m.record_agent_execution(execution)
            reports.append(report)
            completed_agents.add(role.value)

            # All agents in this phase done?
            if all(r.value in completed_agents for r in phase.roles):
                self.logger.info(
                    "  │ all agents in phase '%s' completed", phase.name
                )
                break
        else:
            raise OrchestrationError(
                "hierarchical",
                f"Phase '{phase.name}' supervisor hit max iterations "
                f"({self.max_phase_iterations}).  Completed: "
                f"{[r for r in phase_roles if r in completed_agents]}",
                context={"phase": phase.name,
                         "max_phase_iterations": self.max_phase_iterations},
            )

        self.logger.info("  └── phase '%s' supervisor end", phase.name)
        return reports

    # ------------------------------------------------------------------
    # Root supervisor LLM interaction — NO FALLBACK
    # ------------------------------------------------------------------

    def _ask_root_supervisor(
        self,
        context: RunContext,
        completed_phases: set[str],
        completed_agents: set[str],
        root_iter: int,
    ) -> tuple[dict[str, str], CoordinationCall]:
        prompt = self._build_root_prompt(context, completed_phases, completed_agents)
        system = (
            "You are the root supervisor for a hierarchical multi-agent "
            "code-generation system. You decide which phase of work to "
            "execute next. Respond with JSON only — no markdown fences, "
            "no commentary."
        )

        self.logger.debug(
            "  root prompt (%d chars):\n%s", len(prompt), prompt[:500]
        )

        pre = self.llm.usage_stats.copy()
        t0 = time.perf_counter()

        try:
            raw = self.llm.generate(prompt, system=system, temperature=0.0)
        except Exception as exc:
            raise OrchestrationError(
                "hierarchical",
                f"Root supervisor LLM call failed at iteration {root_iter}: {exc}",
                context={"root_iter": root_iter},
            ) from exc

        latency = time.perf_counter() - t0
        post = self.llm.usage_stats

        self.logger.debug("  root raw response:\n%s", raw[:500])

        try:
            decision = self._parse_json_decision(raw, required_key="phase")
        except (json.JSONDecodeError, ValueError) as exc:
            raise OrchestrationError(
                "hierarchical",
                f"Root supervisor returned unparseable response at iteration "
                f"{root_iter}: {exc}",
                context={"root_iter": root_iter, "raw_response": raw[:500]},
            ) from exc

        coord_call = CoordinationCall(
            purpose=f"root_supervisor_iter_{root_iter}",
            prompt_tokens=post["prompt_tokens"] - pre["prompt_tokens"],
            completion_tokens=post["completion_tokens"] - pre["completion_tokens"],
            total_tokens=post["total_tokens"] - pre["total_tokens"],
            latency_seconds=round(latency, 4),
            raw_response=raw,
            parsed_result=decision,
            iteration=root_iter,
        )
        return decision, coord_call

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
    # Phase supervisor LLM interaction — NO FALLBACK
    # ------------------------------------------------------------------

    def _ask_phase_supervisor(
        self,
        context: RunContext,
        phase: PhaseGroup,
        completed_agents: set[str],
        phase_iter: int,
    ) -> tuple[dict[str, str], CoordinationCall]:
        prompt = self._build_phase_prompt(context, phase, completed_agents)
        system = (
            f"You are the '{phase.name}' phase supervisor in a hierarchical "
            "multi-agent system. You decide which agent in your phase runs "
            "next. Respond with JSON only — no markdown fences, no commentary."
        )

        self.logger.debug(
            "  phase '%s' prompt (%d chars):\n%s",
            phase.name, len(prompt), prompt[:500],
        )

        pre = self.llm.usage_stats.copy()
        t0 = time.perf_counter()

        try:
            raw = self.llm.generate(prompt, system=system, temperature=0.0)
        except Exception as exc:
            raise OrchestrationError(
                "hierarchical",
                f"Phase '{phase.name}' supervisor LLM call failed at "
                f"iteration {phase_iter}: {exc}",
                context={"phase": phase.name, "phase_iter": phase_iter},
            ) from exc

        latency = time.perf_counter() - t0
        post = self.llm.usage_stats

        self.logger.debug(
            "  phase '%s' raw response:\n%s", phase.name, raw[:500]
        )

        try:
            decision = self._parse_json_decision(raw, required_key="agent")
        except (json.JSONDecodeError, ValueError) as exc:
            raise OrchestrationError(
                "hierarchical",
                f"Phase '{phase.name}' supervisor returned unparseable response "
                f"at iteration {phase_iter}: {exc}",
                context={"phase": phase.name, "phase_iter": phase_iter,
                         "raw_response": raw[:500]},
            ) from exc

        coord_call = CoordinationCall(
            purpose=f"phase_{phase.name}_supervisor_iter_{phase_iter}",
            prompt_tokens=post["prompt_tokens"] - pre["prompt_tokens"],
            completion_tokens=post["completion_tokens"] - pre["completion_tokens"],
            total_tokens=post["total_tokens"] - pre["total_tokens"],
            latency_seconds=round(latency, 4),
            raw_response=raw,
            parsed_result=decision,
            iteration=phase_iter,
        )
        return decision, coord_call

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
            raise ValueError(
                f"Invalid supervisor response (missing '{required_key}'): {data}"
            )
        return data

    def _find_phase(self, name: str) -> PhaseGroup | None:
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_summary(self, m: object, reports: list[AgentReport]) -> None:
        self.logger.info(
            "╚══ HIERARCHICAL ORCHESTRATOR END ════╝\n"
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

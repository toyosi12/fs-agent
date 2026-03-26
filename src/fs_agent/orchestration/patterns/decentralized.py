"""Decentralized orchestration pattern — agent-driven negotiation.

Each completing agent decides who runs next via an LLM handoff call.
After both backend and frontend have run, a negotiation phase compares
their contracts.  If mismatches are detected, agents are re-run with
feedback (up to ``MAX_NEGOTIATION_ROUNDS`` rounds).  This tests whether
inter-agent feedback improves integration quality.
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
from .._helpers import execute_agent, run_fixer_loop, run_validation_loop


class DecentralizedOrchestrator(OrchestrationPattern):
    """Agent-driven handoff with negotiation feedback loops.

    Routing intelligence is distributed: after every agent completes, the
    orchestrator asks the *outgoing* agent's LLM "given what you just
    produced, who should handle this next?"

    **Negotiation phase**: after both backend and frontend have completed
    their initial runs, their contracts are compared.  If the mediator
    detects mismatches (e.g. endpoints the frontend calls but the backend
    doesn't serve), the mismatched agent is re-run with corrective
    feedback.  Up to ``MAX_NEGOTIATION_ROUNDS`` rounds of negotiation
    are performed.

    The seed agent is always ``architect``.
    """

    MAX_ITERATIONS = 10
    MAX_NEGOTIATION_ROUNDS = 2
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

                # Inject upstream context based on completed agents' contracts
                context.extra_context = self._build_upstream_context(
                    current_role, context
                )

                # Dispatch the current agent
                agent = self.registry.build(current_role)
                report, execution = execute_agent(agent, current_role, context)
                m.record_agent_execution(execution)
                reports.append(report)
                completed.add(current_role.value)

                # --- Negotiation phase ---
                # After frontend runs, compare contracts with backend and
                # re-run mismatched agent(s) with corrective feedback.
                if (
                    current_role == AgentRole.FRONTEND
                    and AgentRole.BACKEND.value in completed
                ):
                    neg_reports = self._negotiate(context, m, completed)
                    reports.extend(neg_reports)

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

            # --- Fixer loop (fixer ↔ infra) ---
            reports, fixer_result = run_fixer_loop(
                context, self.registry, reports, m,
                pattern_name="decentralized",
            )
            m.fixer_loop_result = fixer_result.to_dict()

            # --- Validation loop ---
            reports = run_validation_loop(
                context, self.registry, reports, m,
                pattern_name="decentralized",
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
    # Upstream context injection (like sequential pipeline)
    # ------------------------------------------------------------------

    def _build_upstream_context(
        self, role: AgentRole, context: RunContext
    ) -> dict[str, str]:
        """Build upstream context from completed agents' contracts."""
        if role == AgentRole.ARCHITECT:
            return {}
        parts: list[str] = []
        backend_contract = context.extract_backend_contract()
        if backend_contract and role in (AgentRole.FRONTEND, AgentRole.INFRA):
            parts.append(f"=== Backend Contract ===\n{backend_contract}")
        frontend_contract = context.extract_frontend_contract()
        if frontend_contract and role == AgentRole.INFRA:
            parts.append(f"=== Frontend Contract ===\n{frontend_contract}")
        if not parts:
            return {}
        return {"upstream_context": "\n\n".join(parts)}

    # ------------------------------------------------------------------
    # Negotiation phase — the key differentiator
    # ------------------------------------------------------------------

    def _negotiate(
        self,
        context: RunContext,
        m: object,
        completed: set[str],
    ) -> list[AgentReport]:
        """Compare backend/frontend contracts; re-run mismatched agents."""
        extra_reports: list[AgentReport] = []

        for round_num in range(1, self.MAX_NEGOTIATION_ROUNDS + 1):
            self.logger.info(
                "── negotiation round %d/%d ──────────────────────────────",
                round_num,
                self.MAX_NEGOTIATION_ROUNDS,
            )

            feedback, coord_call = self._detect_mismatches(context, round_num)
            m.record_coordination_call(coord_call)

            target = feedback.get("target")
            issues = feedback.get("issues", "")

            if target == "none":
                self.logger.info(
                    "  negotiation: no mismatches detected — done"
                )
                break

            self.logger.info(
                "  negotiation: re-running %s — %s",
                target,
                issues[:200],
            )

            try:
                rerun_role = AgentRole(target)
            except ValueError:
                self.logger.warning(
                    "  negotiation: unknown target '%s', skipping", target
                )
                break

            # Re-run the mismatched agent with corrective feedback
            context.extra_context = {
                "upstream_context": (
                    f"=== NEGOTIATION FEEDBACK (round {round_num}) ===\n"
                    f"The following integration issues were detected between "
                    f"your output and the other agents' output:\n\n{issues}\n\n"
                    f"Please regenerate your code to fix these issues. "
                    f"Keep everything else the same."
                )
            }

            agent = self.registry.build(rerun_role)
            report, execution = execute_agent(agent, rerun_role, context)
            m.record_agent_execution(execution)
            extra_reports.append(report)

        return extra_reports

    def _detect_mismatches(
        self,
        context: RunContext,
        round_num: int,
    ) -> tuple[dict[str, str], CoordinationCall]:
        """Use LLM to compare backend/frontend contracts for mismatches."""
        backend_contract = context.extract_backend_contract() or "(no backend output)"
        frontend_contract = context.extract_frontend_contract() or "(no frontend output)"

        prompt = (
            "Compare these two contracts from a backend and frontend agent.\n"
            "Identify any integration mismatches:\n"
            "- Endpoints the frontend fetches that the backend doesn't serve\n"
            "- Field name mismatches between API response shapes and frontend usage\n"
            "- Port or URL mismatches\n"
            "- Missing CORS or proxy configuration\n\n"
            f"Backend contract:\n{backend_contract}\n\n"
            f"Frontend contract:\n{frontend_contract}\n\n"
            "Respond with JSON only:\n"
            '  { "target": "backend"|"frontend"|"none", '
            '"issues": "description of mismatches" }\n'
            "Use target 'none' if no mismatches found."
        )

        system = (
            "You are a contract comparison engine for a multi-agent system. "
            "Respond with JSON only — no markdown fences."
        )

        pre = self.llm.usage_stats.copy()
        t0 = time.perf_counter()
        raw = self.llm.generate(prompt, system=system, temperature=0.0)
        latency = time.perf_counter() - t0
        post = self.llm.usage_stats

        try:
            result = self._parse_handoff(raw)  # reuse JSON parser
        except (json.JSONDecodeError, ValueError):
            result = {"target": "none", "issues": ""}

        coord_call = CoordinationCall(
            purpose=f"negotiation_mismatch_detection_round_{round_num}",
            prompt_tokens=post["prompt_tokens"] - pre["prompt_tokens"],
            completion_tokens=post["completion_tokens"] - pre["completion_tokens"],
            total_tokens=post["total_tokens"] - pre["total_tokens"],
            latency_seconds=round(latency, 4),
            raw_response=raw[:500],
            parsed_result=result,
            iteration=round_num,
        )

        return result, coord_call

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

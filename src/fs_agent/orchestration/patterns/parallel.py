"""Parallel orchestration pattern — fan-out / fan-in with reconciliation.

Agents are organized into sequential **stages**.  Within each stage, all
agents run in parallel.  After the build stage (backend + frontend), a
**reconciliation step** compares their output contracts using an LLM call.
If mismatches are detected, the mismatched agent is re-run with
corrective instructions.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Sequence

from ...context import AgentReport, RunContext
from ...llm import BaseLLMClient
from ...logger import get_logger
from ..base import OrchestrationPattern
from ..metrics import AgentExecution, CoordinationCall
from ..registry import AgentRegistry
from ...agents.base import AgentRole
from .._helpers import execute_agent, run_fixer_loop, run_validation_loop


@dataclass
class Stage:
    """A group of agent roles that can run concurrently.

    Stages execute in order; agents within a stage fan out in parallel.
    """

    roles: list[AgentRole]
    label: str = ""


# Default pipeline: architect (serial) → backend + frontend (parallel) → infra (serial)
DEFAULT_STAGES: list[Stage] = [
    Stage(roles=[AgentRole.ARCHITECT], label="planning"),
    Stage(roles=[AgentRole.BACKEND, AgentRole.FRONTEND], label="build"),
    Stage(roles=[AgentRole.INFRA], label="deploy"),
]


class ParallelOrchestrator(OrchestrationPattern):
    """Fan-out / fan-in with post-build reconciliation.

    Default pipeline::

        Stage 1 (planning) : architect              ← serial (1 agent)
        Stage 2 (build)    : backend + frontend      ← parallel
        ── reconciliation: compare contracts, re-run mismatched agent ──
        Stage 3 (deploy)   : infra                   ← serial (1 agent)

    **Key differentiator**: after parallel agents finish, their output
    contracts are compared.  If mismatches are found, the mismatched
    agent is re-run with corrective instructions.  This adds 1-2 LLM
    coordination calls (detection + optional re-run).
    """

    DEFAULT_MAX_WORKERS = 4
    MAX_RECONCILIATION_ROUNDS = 2

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        llm: BaseLLMClient | None = None,
        stages: list[Stage] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.stages = stages or DEFAULT_STAGES
        self.max_workers = max_workers or self.DEFAULT_MAX_WORKERS
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, context: RunContext) -> Sequence[AgentReport]:
        m = context.metrics
        m.pattern = "parallel"
        m.task_id = getattr(context.settings, "task_id", "")
        m.start_timer()

        all_reports: list[AgentReport] = []

        self.logger.info(
            "╔══ PARALLEL ORCHESTRATOR START ══╗  "
            "stages=%d  max_workers=%d",
            len(self.stages),
            self.max_workers,
        )

        try:
            for stage_idx, stage in enumerate(self.stages, 1):
                label = stage.label or f"stage-{stage_idx}"
                role_names = [r.value for r in stage.roles]

                if len(stage.roles) == 1:
                    # Single agent — run directly, no thread overhead
                    role = stage.roles[0]
                    self.logger.info(
                        "── stage %d/%d [%s]: %s (serial) ──────────────",
                        stage_idx, len(self.stages), label, role_names[0],
                    )
                    agent = self.registry.build(role)
                    report, execution = execute_agent(agent, role, context)
                    m.record_agent_execution(execution)
                    all_reports.append(report)
                else:
                    # Multiple agents — fan out
                    self.logger.info(
                        "── stage %d/%d [%s]: %s (parallel, %d workers) ─────",
                        stage_idx, len(self.stages), label,
                        ", ".join(role_names),
                        min(self.max_workers, len(stage.roles)),
                    )
                    stage_reports = self._run_parallel(context, stage)
                    all_reports.extend(stage_reports)

                    # Reconciliation after parallel build stage
                    if self.llm and self._is_build_stage(stage):
                        reconcile_reports = self._reconcile(context, m)
                        all_reports.extend(reconcile_reports)

            m.success = True

            # --- Fixer loop (fixer ↔ infra) ---
            all_reports, fixer_result = run_fixer_loop(
                context, self.registry, all_reports, m,
                pattern_name="parallel",
            )
            m.fixer_loop_result = fixer_result.to_dict()

            # --- Validation loop ---
            all_reports = run_validation_loop(
                context, self.registry, all_reports, m,
                pattern_name="parallel",
            )

        except Exception as exc:
            m.success = False
            m.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            context.extra_context = {}
            m.stop_timer()
            self._log_summary(m, all_reports)

        return all_reports

    # ------------------------------------------------------------------
    # Reconciliation — the key differentiator
    # ------------------------------------------------------------------

    @staticmethod
    def _is_build_stage(stage: Stage) -> bool:
        """Check if a stage contains both backend and frontend agents."""
        role_set = {r for r in stage.roles}
        return AgentRole.BACKEND in role_set and AgentRole.FRONTEND in role_set

    def _reconcile(
        self,
        context: RunContext,
        m: object,
    ) -> list[AgentReport]:
        """Compare backend/frontend contracts; re-run mismatched agent."""
        extra_reports: list[AgentReport] = []

        for round_num in range(1, self.MAX_RECONCILIATION_ROUNDS + 1):
            self.logger.info(
                "── reconciliation round %d/%d ──────────────────────────",
                round_num,
                self.MAX_RECONCILIATION_ROUNDS,
            )

            feedback, coord_call = self._detect_mismatches(context, round_num)
            m.record_coordination_call(coord_call)

            target = feedback.get("target")
            issues = feedback.get("issues", "")

            if target == "none":
                self.logger.info("  reconciliation: no mismatches — done")
                break

            self.logger.info(
                "  reconciliation: re-running %s — %s",
                target,
                issues[:200],
            )

            try:
                rerun_role = AgentRole(target)
            except ValueError:
                self.logger.warning(
                    "  reconciliation: unknown target '%s', skipping", target
                )
                break

            context.extra_context = {
                "upstream_context": (
                    f"=== RECONCILIATION FEEDBACK (round {round_num}) ===\n"
                    f"After parallel execution, these integration issues were "
                    f"detected between backend and frontend:\n\n{issues}\n\n"
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
            "Compare these two contracts from independently-built backend "
            "and frontend agents.\n"
            "Identify any integration mismatches:\n"
            "- Endpoints the frontend fetches that the backend doesn't serve\n"
            "- Field name mismatches between API responses and frontend usage\n"
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
            result = self._parse_json(raw)
        except (json.JSONDecodeError, ValueError):
            result = {"target": "none", "issues": ""}

        coord_call = CoordinationCall(
            purpose=f"reconciliation_mismatch_round_{round_num}",
            prompt_tokens=post["prompt_tokens"] - pre["prompt_tokens"],
            completion_tokens=post["completion_tokens"] - pre["completion_tokens"],
            total_tokens=post["total_tokens"] - pre["total_tokens"],
            latency_seconds=round(latency, 4),
            raw_response=raw[:500],
            parsed_result=result,
            iteration=round_num,
        )

        return result, coord_call

    @staticmethod
    def _parse_json(raw: str) -> dict[str, str]:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
        text = text.strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data)}")
        return data

    # ------------------------------------------------------------------
    # Parallel fan-out / fan-in
    # ------------------------------------------------------------------

    def _run_parallel(
        self, context: RunContext, stage: Stage
    ) -> list[AgentReport]:
        """Run all agents in a stage concurrently and collect their reports.

        Each thread gets its own agent instance from the registry.
        Reports are collected from futures on the main thread and then
        recorded into context in a deterministic order (matching the
        stage's role list) to keep transcripts reproducible.
        """
        m = context.metrics
        results_by_role: dict[str, tuple[AgentReport, AgentExecution]] = {}
        errors: dict[str, Exception] = {}

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(stage.roles))
        ) as pool:
            future_to_role = {}
            for role in stage.roles:
                agent = self.registry.build(role)
                future = pool.submit(self._execute_in_thread, agent, role, context)
                future_to_role[future] = role

            for future in as_completed(future_to_role):
                role = future_to_role[future]
                try:
                    report, execution = future.result()
                    results_by_role[role.value] = (report, execution)
                except Exception as exc:
                    self.logger.error(
                        "Agent '%s' failed in parallel stage: %s",
                        role.value, exc,
                    )
                    errors[role.value] = exc

        if errors:
            failed = ", ".join(errors.keys())
            self.logger.warning(
                "%d agent(s) failed in parallel stage: %s", len(errors), failed
            )

        # Record reports into context in declaration order for reproducibility
        ordered: list[AgentReport] = []
        for role in stage.roles:
            if role.value in results_by_role:
                report, execution = results_by_role[role.value]
                context.record(report)
                m.record_agent_execution(execution)
                ordered.append(report)

        return ordered

    def _execute_in_thread(
        self, agent: object, role: AgentRole, context: RunContext
    ) -> tuple[AgentReport, AgentExecution]:
        """Run a single agent in a worker thread.

        We call agent.run() and build the AgentReport here but do NOT
        call context.record() — that happens on the main thread after
        all futures complete, to avoid race conditions on the shared
        transcripts list.

        Returns ``(AgentReport, AgentExecution)`` for consistent metrics.
        """
        from datetime import datetime, timezone
        from ...artifact_writer import persist_agent_output

        t0 = time.perf_counter()
        self.logger.info("→ %s agent starting (thread)", role.value)

        # Capture pre-execution token stats (may be inaccurate under
        # concurrency, but gives a best-effort measurement)
        role_llm = context.get_llm(role.value)
        pre = role_llm.usage_stats.copy()

        result = agent.run(context)
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

        duration = time.perf_counter() - t0
        post = role_llm.usage_stats

        execution = AgentExecution(
            role=role.value,
            status=report.status,
            attempt=1,
            duration_seconds=round(duration, 4),
            prompt_tokens=post["prompt_tokens"] - pre["prompt_tokens"],
            completion_tokens=post["completion_tokens"] - pre["completion_tokens"],
            total_tokens=post["total_tokens"] - pre["total_tokens"],
            artifact_count=len(report.artifacts),
            attachment_count=len(report.metadata.get("attachments", [])),
        )

        self.logger.info(
            "✓ %s agent finished in %.2fs  tokens=%d (thread)",
            role.value, duration, execution.total_tokens,
        )
        return report, execution

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_summary(self, m: object, reports: list[AgentReport]) -> None:
        self.logger.info(
            "╚══ PARALLEL ORCHESTRATOR END ════╝\n"
            "  success=%s  duration=%.2fs  agents_run=%d\n"
            "  coordination_calls=0  coordination_tokens=0  "
            "(no LLM coordination in parallel)\n"
            "  functional_tokens=%d  (prompt=%d, completion=%d)\n"
            "  coordination/functional ratio=%.4f\n"
            "  total_tokens=%d  est_cost=$%.6f",
            m.success,
            m.total_duration_seconds,
            m.agent_execution_count,
            m.functional_total_tokens,
            m.functional_prompt_tokens,
            m.functional_completion_tokens,
            m.coordination_to_functional_ratio,
            m.total_tokens,
            m.cost_estimate(),
        )

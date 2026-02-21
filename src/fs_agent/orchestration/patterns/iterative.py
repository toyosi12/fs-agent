"""Iterative refinement orchestration pattern — critic-driven retry loop."""

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


# Criteria the critic evaluates per agent role
_ROLE_CRITERIA: dict[str, list[str]] = {
    "architect": [
        "Spec includes all backend endpoints required by the brief",
        "Every frontend route/component has matching backend endpoints in consumes",
        "Database models and migrations are present if data persistence is needed",
        "Infra targets include at least dev and prod environments",
        "No orphaned endpoints — every endpoint is consumed by at least one route or component",
    ],
    "backend": [
        "package.json has correct dependencies",
        "All endpoints from the spec have corresponding route handlers",
        "Database connection and migration files are generated",
        "Environment variables are documented in .env.example",
        "Error handling middleware is present",
    ],
    "frontend": [
        "package.json has correct dependencies",
        "All routes from the spec have corresponding page components",
        "Components make API calls to the correct backend endpoints",
        "Styling follows the theme tokens from the spec",
        "Navigation between routes is implemented",
    ],
    "infra": [
        "Database is created and migrations are applied",
        "Backend server starts without errors",
        "Frontend dev server starts without errors",
        "Environment files are properly configured",
    ],
}


class IterativeRefinementOrchestrator(OrchestrationPattern):
    """Run each agent, then ask an LLM critic to evaluate quality.

    If the critic says the output is insufficient the agent is re-run
    (up to ``max_retries`` times).  The critic prompt includes role-
    specific quality criteria so the evaluation is focused.

    Flow for each agent::

        attempt 1: run agent → critic evaluates → pass / fail
        attempt 2 (if fail): re-run agent → critic → pass / fail
        ...
        attempt N: accept regardless (max retries exhausted)

    Agents are dispatched in canonical order (architect → backend →
    frontend → infra).  The critic is an LLM prompt, not an agent.
    """

    CANONICAL_ORDER = [
        AgentRole.ARCHITECT,
        AgentRole.BACKEND,
        AgentRole.FRONTEND,
        AgentRole.INFRA,
    ]
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_PASS_THRESHOLD = 7  # critic score 1-10; ≥ threshold = pass

    def __init__(
        self,
        registry: AgentRegistry,
        llm: BaseLLMClient,
        *,
        max_retries: int | None = None,
        pass_threshold: int | None = None,
    ) -> None:
        self.registry = registry
        self.llm = llm
        self.max_retries = max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        self.pass_threshold = pass_threshold if pass_threshold is not None else self.DEFAULT_PASS_THRESHOLD
        self.logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, context: RunContext) -> Iterable[AgentReport]:
        all_reports: list[AgentReport] = []

        self.logger.info(
            "Iterative refinement orchestrator starting "
            "(max_retries=%d, pass_threshold=%d/10)",
            self.max_retries,
            self.pass_threshold,
        )

        for role in self.CANONICAL_ORDER:
            report = self._run_with_refinement(role, context)
            all_reports.append(report)

        self.logger.info(
            "Iterative refinement orchestrator complete: %d stages",
            len(all_reports),
        )
        return all_reports

    # ------------------------------------------------------------------
    # Refinement loop for a single agent
    # ------------------------------------------------------------------

    def _run_with_refinement(
        self, role: AgentRole, context: RunContext
    ) -> AgentReport:
        """Run an agent up to (1 + max_retries) times until the critic passes it."""

        best_report: AgentReport | None = None

        for attempt in range(1, self.max_retries + 2):  # +2 because range is exclusive
            is_last = attempt > self.max_retries

            self.logger.info(
                "Agent '%s' attempt %d/%d",
                role.value,
                attempt,
                self.max_retries + 1,
            )

            # Remove the previous report for this role if retrying
            if best_report is not None:
                context.transcripts = [
                    r for r in context.transcripts if r.role != role.value
                ]

            agent = self.registry.build(role)
            report = execute_agent(agent, role, context)
            best_report = report

            if is_last:
                self.logger.info(
                    "Agent '%s' accepted (max retries exhausted)", role.value
                )
                break

            # Ask the critic
            verdict = self._evaluate(role, report, context)
            score = verdict.get("score", 0)
            feedback = verdict.get("feedback", "")
            passed = score >= self.pass_threshold

            self.logger.info(
                "Critic verdict for '%s': score=%d/10 pass=%s feedback=%s",
                role.value,
                score,
                passed,
                feedback[:150],
            )

            if passed:
                self.logger.info(
                    "Agent '%s' passed critic on attempt %d", role.value, attempt
                )
                break

            # Store feedback in context so the agent can use it on retry
            context.transcripts[-1].metadata["critic_feedback"] = feedback
            context.transcripts[-1].metadata["critic_score"] = score

        assert best_report is not None
        return best_report

    # ------------------------------------------------------------------
    # Critic LLM interaction
    # ------------------------------------------------------------------

    def _evaluate(
        self, role: AgentRole, report: AgentReport, context: RunContext
    ) -> dict[str, object]:
        """Ask the LLM critic to score the agent's output."""
        prompt = self._build_critic_prompt(role, report, context)
        system = (
            "You are a strict quality critic for a multi-agent code-generation system. "
            "Evaluate the agent's output against the provided criteria. "
            "Respond with JSON only — no markdown fences, no commentary."
        )
        try:
            raw = self.llm.generate(prompt, system=system, temperature=0.0)
            return self._parse_verdict(raw)
        except Exception as exc:
            self.logger.warning(
                "Critic LLM call failed (%s); defaulting to pass", exc
            )
            return {"score": 10, "feedback": "critic unavailable — auto-pass"}

    def _build_critic_prompt(
        self,
        role: AgentRole,
        report: AgentReport,
        context: RunContext,
    ) -> str:
        criteria = _ROLE_CRITERIA.get(role.value, ["Output is complete and correct"])

        # Summarise artifacts for the critic
        artifact_summary: dict[str, str] = {}
        for key, value in report.artifacts.items():
            if isinstance(value, str):
                artifact_summary[key] = value[:500]
            elif isinstance(value, dict):
                artifact_summary[key] = json.dumps(value, indent=2)[:500]
            elif isinstance(value, list):
                artifact_summary[key] = f"[{len(value)} items]"
            else:
                artifact_summary[key] = str(value)[:300]

        return (
            f"Agent: {role.value}\n"
            f"Summary: {report.summary}\n\n"
            f"Artifacts:\n{json.dumps(artifact_summary, indent=2)}\n\n"
            f"User request:\n{context.user_request}\n\n"
            f"Quality criteria (all must be met for a passing score):\n"
            + "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))
            + "\n\n"
            "Score the output from 1 (poor) to 10 (excellent).\n"
            "Respond with exactly one JSON object:\n"
            '  { "score": <1-10>, "feedback": "concise explanation of issues or praise" }\n'
        )

    def _parse_verdict(self, raw: str) -> dict[str, object]:
        """Parse the critic's JSON verdict."""
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
        text = text.strip()
        data = json.loads(text)
        if not isinstance(data, dict) or "score" not in data:
            raise ValueError(f"Invalid critic response: {data}")
        # Clamp score to 1-10
        score = data.get("score", 1)
        if isinstance(score, (int, float)):
            data["score"] = max(1, min(10, int(score)))
        else:
            data["score"] = 1
        return data

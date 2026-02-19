"""Infrastructure agent stub."""

from __future__ import annotations

from datetime import datetime, timezone

from ..context import RunContext
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class InfraAgent(BaseAgent):
    role = AgentRole.INFRA

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        scoped = context.infra_context()  # only metadata + infra
        spec = context.require_spec()
        infra = spec.infra
        targets = [target.model_dump() for target in infra.targets]
        pipeline = {
            "ci": infra.ci,
            "cd": infra.cd,
            "targets": targets,
            "artifacts": list(context.artifacts.keys()),
        }
        runbook = self._generate_infra_plan(context, pipeline)
        attachments = [
            AgentArtifact(
                name="infra_plan.md",
                kind="plan",
                description="Deployment playbook",
                body="\n".join(
                    [
                        f"CI: {infra.ci}",
                        f"CD: {infra.cd}",
                        "Targets:",
                    ]
                    + [f"- {t['environment']} -> {t['name']}" for t in targets]
                ),
            ),
            AgentArtifact(
                name="infra_runbook.md",
                kind="doc",
                description="LLM-generated deployment checklist",
                body=runbook,
            ),
        ]
        result = AgentResult(
            role=self.role,
            summary=f"Drafted infra plan for {len(targets)} targets.",
            artifacts={
                "infra_pipeline": pipeline,
                "infra_runbook": {"body": runbook},
            },
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _generate_infra_plan(self, context: RunContext, pipeline: dict[str, object]) -> str:
        metadata = context.require_spec().metadata
        targets = pipeline.get("targets", []) or []
        target_lines = [
            f"- {target['environment']} ({target['runtime']}): {target['name']}"
            for target in targets
        ]
        prompt = (
            f"Project: {metadata.name}\n"
            f"Summary: {metadata.summary}\n"
            f"CI: {pipeline['ci']}\nCD: {pipeline['cd']}\n\n"
            "Produce a deployment runbook covering environment promotion, release gates,"
            " secrets management, and rollback triggers for the environments below.\n"
            + ("\n".join(target_lines) if target_lines else "- No targets defined")
        )
        system = (
            "You are an experienced DevOps engineer. Outline infra steps with numbered"
            " checklists and commands where possible."
        )
        try:
            return context.llm.generate(prompt, system=system, temperature=0.2)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("LLM generation failed for infra: %s", exc)
            return self._fallback_runbook(target_lines)

    def _fallback_runbook(self, target_lines: list[str]) -> str:
        body = ["# Deployment Runbook (fallback)", "## Targets"]
        body.extend(target_lines or ["- N/A"])
        body.extend(
            [
                "## Steps",
                "1. Push code and trigger CI.",
                "2. Build Docker image and run smoke tests.",
                "3. Deploy to staging, run E2E, then promote to prod.",
                "4. Monitor metrics; rollback on sustained errors.",
            ]
        )
        return "\n".join(body)

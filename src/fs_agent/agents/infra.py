"""Infrastructure agent stub."""

from __future__ import annotations

from datetime import datetime, timezone

from ..context import RunContext
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class InfraAgent(BaseAgent):
    role = AgentRole.INFRA

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        infra = context.spec.infra
        targets = [target.model_dump() for target in infra.targets]
        pipeline = {
            "ci": infra.ci,
            "cd": infra.cd,
            "targets": targets,
            "artifacts": list(context.artifacts.keys()),
        }
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
            )
        ]
        result = AgentResult(
            role=self.role,
            summary=f"Drafted infra plan for {len(targets)} targets.",
            artifacts={"infra_pipeline": pipeline},
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

"""Frontend agent stub that coordinates UI plans."""

from __future__ import annotations

from datetime import datetime, timezone

from ..context import RunContext
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class FrontendAgent(BaseAgent):
    role = AgentRole.FRONTEND

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        frontend = context.spec.frontend
        backend_blueprint = context.artifacts.get("backend_blueprint", {})
        routes = []
        for route in frontend.routes:
            routes.append(
                {
                    "path": route.path,
                    "description": route.description,
                    "components": route.components,
                    "consumes": route.consumes,
                }
            )

        plan_lines = [
            f"Framework: {frontend.framework} ({frontend.language})",
            f"Styling: {frontend.styling}",
            "",
            "Routes:",
        ]
        for route in routes:
            plan_lines.append(f"- {route['path']}: {route['description']}")
        plan_lines.append("\nBackend Contracts Used:")
        for endpoint in backend_blueprint.get("endpoints", []):
            plan_lines.append(f"- {endpoint['method']} {endpoint['path']}")

        attachments = [
            AgentArtifact(
                name="frontend_plan.md",
                kind="plan",
                description="UI composition plan",
                body="\n".join(plan_lines),
            )
        ]

        result = AgentResult(
            role=self.role,
            summary=f"Outlined {len(routes)} routes referencing backend contracts.",
            artifacts={"frontend_blueprint": {"routes": routes}},
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

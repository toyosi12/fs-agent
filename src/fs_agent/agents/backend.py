"""Backend agent stub that proposes API plans."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..context import RunContext
from ..models.spec import ApiEndpoint
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class BackendAgent(BaseAgent):
    role = AgentRole.BACKEND

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        backend = context.spec.backend
        lines: list[str] = []
        endpoint_plans: list[dict[str, Any]] = []
        for endpoint in backend.endpoints:
            contract = self._describe_endpoint(endpoint)
            lines.append(contract)
            endpoint_plans.append({
                "name": endpoint.name,
                "path": endpoint.path,
                "method": endpoint.method.value,
                "tests": [f"returns 200 for {endpoint.path}"],
            })

        blueprint = {
            "language": backend.language,
            "framework": backend.framework,
            "style": backend.style,
            "endpoints": endpoint_plans,
            "data_models": [model.model_dump() for model in backend.data_models],
        }

        attachments = [
            AgentArtifact(
                name="backend_plan.md",
                kind="plan",
                description="High-level backend implementation outline",
                body="\n".join(lines) or "No endpoints defined.",
            )
        ]

        result = AgentResult(
            role=self.role,
            summary=f"Planned {len(endpoint_plans)} endpoints using {backend.framework}.",
            artifacts={"backend_blueprint": blueprint},
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _describe_endpoint(self, endpoint: ApiEndpoint) -> str:
        parts = [f"### {endpoint.method.value} {endpoint.path}"]
        parts.append(endpoint.description)
        if endpoint.request_schema:
            parts.append("Request Body:")
            for field, dtype in endpoint.request_schema.items():
                parts.append(f"- {field}: {dtype}")
        if endpoint.response_schema:
            parts.append("Response Body:")
            for field, dtype in endpoint.response_schema.items():
                parts.append(f"- {field}: {dtype}")
        if endpoint.errors:
            parts.append("Possible Errors:")
            for error in endpoint.errors:
                parts.append(f"- {error}")
        return "\n".join(parts)

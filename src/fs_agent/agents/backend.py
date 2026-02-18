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

        code_body = self._generate_backend_code(context, blueprint)
        attachments = [
            AgentArtifact(
                name="backend_plan.md",
                kind="plan",
                description="High-level backend implementation outline",
                body="\n".join(lines) or "No endpoints defined.",
            ),
            AgentArtifact(
                name="backend_service.ts",
                kind="code",
                description="LLM-generated Express router skeleton",
                body=code_body,
            ),
        ]

        result = AgentResult(
            role=self.role,
            summary=f"Planned {len(endpoint_plans)} endpoints using {backend.framework}.",
            artifacts={
                "backend_blueprint": blueprint,
                "backend_source": {
                    "language": backend.language,
                    "framework": backend.framework,
                    "body": code_body,
                },
            },
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _generate_backend_code(self, context: RunContext, blueprint: dict[str, Any]) -> str:
        metadata = context.spec.metadata
        endpoint_lines = []
        for endpoint in blueprint["endpoints"]:
            endpoint_lines.append(
                f"- {endpoint['method']} {endpoint['path']}: {endpoint['name']}"
            )
        prompt = (
            f"Project: {metadata.name} ({metadata.summary})\n"
            f"Owner: {metadata.owner}\n\n"
            "Write a TypeScript Express router that implements these REST endpoints:\n"
            + "\n".join(endpoint_lines)
            + "\nReturn a complete file with imports, router setup, and placeholder handlers"
        )
        system = (
            "You are a senior backend engineer. Produce idiomatic Express/TypeScript code with"
            " zod validation placeholders and TODO comments where business logic belongs."
        )
        try:
            return context.llm.generate(prompt, system=system, temperature=0.1)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("LLM generation failed for backend: %s", exc)
            return self._fallback_backend_code(blueprint)

    def _fallback_backend_code(self, blueprint: dict[str, Any]) -> str:
        lines = [
            "// Express router skeleton (fallback)",
            "import { Router } from 'express';",
            "const router = Router();",
        ]
        for endpoint in blueprint.get("endpoints", []):
            method = endpoint["method"].lower()
            path = endpoint["path"]
            name = endpoint["name"].lower()
            lines.append(
                (
                    f"router.{method}('{path}', async (req, res) => {{\n"
                    f"  // TODO: implement {name}\n"
                    "  return res.status(501).json({{ message: 'not implemented' }});\n"
                    "});"
                )
            )
        lines.append("export default router;")
        return "\n".join(lines)

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

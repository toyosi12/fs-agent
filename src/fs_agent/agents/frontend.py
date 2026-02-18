"""Frontend agent stub that coordinates UI plans."""

from __future__ import annotations

from datetime import datetime, timezone

from ..context import RunContext
from typing import Any

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

        code_body = self._generate_frontend_code(context, routes, backend_blueprint)
        attachments = [
            AgentArtifact(
                name="frontend_plan.md",
                kind="plan",
                description="UI composition plan",
                body="\n".join(plan_lines),
            ),
            AgentArtifact(
                name="frontend_app.tsx",
                kind="code",
                description="LLM-generated React app shell",
                body=code_body,
            ),
        ]

        result = AgentResult(
            role=self.role,
            summary=f"Outlined {len(routes)} routes referencing backend contracts.",
            artifacts={
                "frontend_blueprint": {"routes": routes},
                "frontend_source": {
                    "language": frontend.language,
                    "framework": frontend.framework,
                    "body": code_body,
                },
            },
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _generate_frontend_code(
        self,
        context: RunContext,
        routes: list[dict[str, Any]],
        backend_blueprint: dict[str, Any],
    ) -> str:
        metadata = context.spec.metadata
        route_lines = [f"- {route['path']}: components {', '.join(route['components']) or 'N/A'}" for route in routes]
        api_lines = [f"- {endpoint['method']} {endpoint['path']}" for endpoint in backend_blueprint.get("endpoints", [])]
        prompt = (
            f"Project: {metadata.name}\n"
            f"Summary: {metadata.summary}\n\n"
            "Build a React (TypeScript + hooks) single-page app scaffold with Tailwind classes."
            " Include data-fetching hooks for these endpoints and components for each route."
            "\nRoutes:\n"
            + "\n".join(route_lines)
            + "\nAPIs:\n"
            + ("\n".join(api_lines) if api_lines else "(no APIs provided)")
        )
        system = (
            "You are a senior frontend engineer. Generate functional React components"
            " with fetch wrappers, prop typing, and TODO comments where business logic"
            " should exist."
        )
        try:
            return context.llm.generate(prompt, system=system, temperature=0.2)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("LLM generation failed for frontend: %s", exc)
            return self._fallback_frontend_code(routes)

    def _fallback_frontend_code(self, routes: list[dict[str, Any]]) -> str:
        lines = ["// React app skeleton (fallback)", "import React from 'react';"]
        for route in routes:
            component = (route.get("components") or ["RouteView"])[0]
            safe_component = component.replace(" ", "")
            lines.append(
                (
                    f"export function {safe_component}() {{\n"
                    "  // TODO: implement UI\n"
                    "  return <div>Placeholder view</div>;\n"
                    "}\n"
                )
            )
        lines.append("export default function App() {\n  return <div>App shell</div>;\n}")
        return "\n".join(lines)

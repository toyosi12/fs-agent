"""Frontend Pages agent — specializes in page components, routing, and layouts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..context import RunContext
from ..mcp import apply_filesystem_plan
from ..models.spec import ProjectSpec
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class FrontendPagesAgent(BaseAgent):
    """Generates page-level components, React Router setup, and layouts.

    Used by the hierarchical pattern's 3-level topology.  Produces the
    App shell, page components, and routing configuration.  Does NOT
    generate reusable UI components or styling — those come from
    FrontendUiAgent.
    """

    role = AgentRole.FRONTEND_PAGES

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        spec = context.require_spec()
        frontend = spec.frontend

        routes = []
        for route in frontend.routes:
            routes.append({
                "path": route.path,
                "description": route.description,
                "components": route.components,
                "consumes": route.consumes,
                "auth_required": route.auth_required,
                "layout": route.layout,
            })

        # Get backend endpoints for API integration
        backend_endpoints = []
        for ep in spec.backend.endpoints:
            backend_endpoints.append({
                "name": ep.name,
                "method": ep.method.value,
                "path": ep.path,
                "request_schema": ep.request_schema,
                "response_schema": ep.response_schema,
            })

        pages_code = self._generate_pages_code(context, routes, backend_endpoints)
        mcp_plan = self._generate_pages_plan(
            context, spec, routes, backend_endpoints, pages_code
        )

        projects_root = context.projects_dir
        application = apply_filesystem_plan(
            mcp_plan, projects_root, dry_run=context.settings.dry_run,
        )
        created_files = [str(path) for path in application.created_files]
        mcp_plan = {**mcp_plan, "project_path": str(application.project_path)}

        attachments = [
            AgentArtifact(
                name="frontend_pages.jsx",
                kind="code",
                description="Page components and routing",
                body=pages_code,
            ),
        ]

        result = AgentResult(
            role=self.role,
            summary=(
                f"Generated {len(routes)} page components with routing and "
                f"API integration for {len(backend_endpoints)} endpoints."
            ),
            artifacts={
                "frontend_pages_blueprint": {"routes": routes},
                "frontend_pages_source": {"language": "javascript", "body": pages_code},
                "frontend_pages_mcp_plan": mcp_plan,
                "frontend_pages_files": created_files,
            },
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _generate_pages_code(
        self,
        context: RunContext,
        routes: list[dict[str, Any]],
        backend_endpoints: list[dict[str, Any]],
    ) -> str:
        metadata = context.require_spec().metadata
        route_lines = [
            f"- {r['path']}: {r['description']} (components: {', '.join(r['components'])})"
            for r in routes
        ]
        api_lines = []
        for ep in backend_endpoints:
            line = f"- {ep['method']} {ep['path']}: {ep['name']}"
            if ep.get("response_schema"):
                line += f"\n  Response: {json.dumps(ep['response_schema'])}"
            api_lines.append(line)

        prompt = (
            f"Project: {metadata.name} ({metadata.summary})\n\n"
            "Generate the PAGE-LEVEL React components and routing setup.\n"
            "Focus on:\n"
            "- App.jsx shell with React Router navigation\n"
            "- One page component per route with real data fetching\n"
            "- Layout structure (header, sidebar, main content)\n"
            "- Integration with backend API endpoints\n\n"
            "Routes:\n" + "\n".join(route_lines) + "\n\n"
            "Backend APIs:\n" + ("\n".join(api_lines) or "(none)") + "\n\n"
            "Do NOT generate reusable UI components (buttons, cards, modals) — "
            "those will be created by a separate agent. Instead, import them "
            "from '../components/' and use them in your pages.\n"
            "Do NOT use TypeScript. Do NOT leave TODO stubs.\n"
            "All data must come from real fetch() calls — no mock data.\n"
        )

        upstream = context.extra_context.get("upstream_context", "")
        if upstream:
            prompt += (
                "\n--- UPSTREAM CONTEXT ---\n"
                f"{upstream}\n"
                "--- END UPSTREAM CONTEXT ---\n"
            )

        system = (
            "You are a senior frontend engineer specializing in page architecture. "
            "Generate complete React page components with hooks, data fetching, "
            "and routing. Use plain JavaScript (JSX) with Tailwind CSS. "
            "Import reusable components from '../components/' — do NOT define them. "
            "No placeholders, no TODOs, no TypeScript. "
            "Every npm import must exist in package.json."
        )

        try:
            return context.get_llm("frontend_pages").generate(
                prompt, system=system, temperature=0.2,
            )
        except Exception:
            return self._fallback_pages(routes)

    def _fallback_pages(self, routes: list[dict[str, Any]]) -> str:
        lines = [
            "import React, { useState, useEffect } from 'react';",
            "import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';",
            "",
        ]
        for r in routes:
            comp = (r.get("components") or ["Page"])[0]
            lines.append(
                f"export function {comp}() {{\n"
                f"  return <div className=\"p-4\"><h1>{r.get('description', comp)}</h1></div>;\n"
                "}\n"
            )
        lines.append("export default function App() {")
        lines.append("  return <BrowserRouter><Routes>")
        for r in routes:
            comp = (r.get("components") or ["Page"])[0]
            lines.append(f"    <Route path=\"{r['path']}\" element={{<{comp} />}} />")
        lines.append("  </Routes></BrowserRouter>;")
        lines.append("}")
        return "\n".join(lines)

    def _generate_pages_plan(
        self,
        context: RunContext,
        spec: ProjectSpec,
        routes: list[dict[str, Any]],
        backend_endpoints: list[dict[str, Any]],
        pages_code: str,
    ) -> dict[str, Any]:
        """Generate MCP plan for page-level files only."""
        system = (
            "You are a senior frontend engineer. Produce a JSON plan for page-level "
            "files in a React + Vite project. Include: src/App.jsx, src/pages/*.jsx, "
            "src/hooks/useApi.js, src/main.jsx, package.json, vite.config.js, "
            "and a multi-stage Dockerfile (build with node, serve with nginx). "
            "Do NOT include reusable components in src/components/ — those come "
            "from a separate agent. "
            "Always use npm install, never npm ci. "
            "Dockerfile must proxy /api/ to http://backend:4000. "
            "Include lucide-react in dependencies."
        )
        prompt = (
            f"User request: {context.user_request}\n"
            f"Routes: {json.dumps(routes, indent=2)}\n"
            f"Backend APIs: {json.dumps(backend_endpoints, indent=2)}\n\n"
            f"Page code to use:\n<pages>\n{pages_code}\n</pages>\n\n"
            "Return JSON plan with structure:\n"
            '{"tool": "mcp.fs", "project_root": "frontend", '
            '"files": [{"path": "...", "description": "...", "contents": "..."}]}'
        )
        try:
            raw = context.get_llm("frontend_pages").generate(
                prompt, system=system, temperature=0.2,
            )
            text = raw.strip()
            if text.startswith("```"):
                parts = text.split("```", 2)
                if len(parts) >= 2:
                    text = parts[1]
                    if text.lstrip().startswith("json"):
                        text = text.lstrip()[4:]
            plan = json.loads(text.strip())
            plan.setdefault("tool", "mcp.fs")
            plan.setdefault("project_root", "frontend")
            if isinstance(plan.get("files"), list) and plan["files"]:
                return plan
        except Exception as exc:
            self.logger.warning("Pages plan generation failed: %s", exc)

        return {
            "tool": "mcp.fs",
            "project_root": "frontend",
            "instructions": f"Page scaffold for {spec.metadata.name}",
            "files": [
                {"path": "src/App.jsx", "description": "App shell", "contents": pages_code},
            ],
        }

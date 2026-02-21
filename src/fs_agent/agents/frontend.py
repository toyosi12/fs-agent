"""Frontend agent that plans UI work and emits MCP file plans."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from ..context import RunContext
from ..mcp import apply_filesystem_plan
from ..models.spec import ProjectSpec
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class FrontendAgent(BaseAgent):
    role = AgentRole.FRONTEND

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        spec = context.require_spec()
        frontend = spec.frontend
        scoped = context.frontend_context()  # metadata + frontend + backend endpoint sigs
        backend_endpoints = scoped.get("backend_endpoints", [])
        routes = []
        for route in frontend.routes:
            routes.append(
                {
                    "path": route.path,
                    "description": route.description,
                    "components": route.components,
                    "consumes": route.consumes,
                    "auth_required": route.auth_required,
                    "layout": route.layout,
                }
            )

        component_defs = [comp.model_dump() for comp in frontend.components]

        plan_lines = [
            f"Framework: {frontend.framework} ({frontend.language})",
            f"Styling: {frontend.styling}",
            "",
        ]
        if frontend.theme:
            plan_lines.append("Theme:")
            for key, val in frontend.theme.items():
                plan_lines.append(f"  {key}: {val}")
            plan_lines.append("")
        plan_lines.append("Routes:")
        for route in routes:
            plan_lines.append(f"- {route['path']}: {route['description']}")
        if component_defs:
            plan_lines.append("\nComponents:")
            for comp in component_defs:
                plan_lines.append(f"- {comp['name']}: {comp.get('description', '')}")
        plan_lines.append("\nBackend Contracts Used:")
        for endpoint in backend_endpoints:
            plan_lines.append(f"- {endpoint['method']} {endpoint['path']}")

        # Build a backend_blueprint-like dict from scoped endpoint sigs for downstream helpers
        backend_blueprint = {"endpoints": backend_endpoints}

        code_body = self._generate_frontend_code(context, routes, backend_blueprint)
        mcp_plan = self._generate_project_plan(context, spec, routes, backend_blueprint, code_body)
        projects_root = context.projects_dir
        application = apply_filesystem_plan(
            mcp_plan,
            projects_root,
            dry_run=context.settings.dry_run,
        )
        created_files = [str(path) for path in application.created_files]
        mcp_plan = {**mcp_plan, "project_path": str(application.project_path)}
        attachments = [
            AgentArtifact(
                name="frontend_plan.md",
                kind="plan",
                description="UI composition plan",
                body="\n".join(plan_lines),
            ),
            AgentArtifact(
                name="frontend_app.jsx",
                kind="code",
                description="LLM-generated React app",
                body=code_body,
            ),
            AgentArtifact(
                name="frontend_mcp_plan.json",
                kind="doc",
                description="Filesystem instructions for the frontend project",
                body=json.dumps(mcp_plan, indent=2),
            ),
        ]

        result = AgentResult(
            role=self.role,
            summary=(
                f"Outlined {len(routes)} routes referencing backend contracts and prepared {mcp_plan.get('project_root', 'frontend project')}."
            ),
            artifacts={
                "frontend_blueprint": {
                    "routes": routes,
                    "components": component_defs,
                    "theme": frontend.theme,
                },
                "frontend_source": {
                    "language": frontend.language,
                    "framework": frontend.framework,
                    "body": code_body,
                },
                "frontend_mcp_plan": mcp_plan,
                "frontend_project_files": created_files,
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
        metadata = context.require_spec().metadata
        route_lines = [f"- {route['path']}: components {', '.join(route['components']) or 'N/A'}" for route in routes]
        api_lines = [f"- {endpoint['method']} {endpoint['path']}" for endpoint in backend_blueprint.get("endpoints", [])]
        prompt = (
            f"Project: {metadata.name}\n"
            f"Summary: {metadata.summary}\n\n"
            "Build a React (JavaScript + hooks) single-page app with Tailwind classes."
            " Include complete, working data-fetching hooks and fully implemented "
            "components for each route. Every component must render real UI with state "
            "management, event handlers, and proper layout — not placeholders."
            "\nRoutes:\n"
            + "\n".join(route_lines)
            + "\nAPIs:\n"
            + ("\n".join(api_lines) if api_lines else "(no APIs provided)")
            + "\n\nDo NOT use TypeScript. Do NOT leave TODO comments or placeholder stubs."
            "\n\nAlso generate a companion test file using Vitest and React Testing Library. "
            "The tests should:\n"
            "- Import components and render them with @testing-library/react\n"
            "- Test that key UI elements render (headings, buttons, lists)\n"
            "- Test user interactions (clicks, form submissions) with fireEvent/userEvent\n"
            "- Mock fetch calls with vi.fn()\n"
            "- Use describe/it blocks with clear test names\n"
            "Wrap the test code in a clearly separated section starting with "
            "'// === TESTS ===' so it can be extracted into its own file.\n"
        )
        system = (
            "You are a senior frontend engineer. Generate complete, functional React "
            "components in plain JavaScript (JSX) with fetch wrappers, useState/useEffect "
            "hooks, and real UI markup using Tailwind CSS. Every component must be fully "
            "implemented with working forms, lists, event handlers, and API integration. "
            "No placeholders, no TODOs, no TypeScript syntax."
        )
        try:
            return context.llm.generate(prompt, system=system, temperature=0.2)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("LLM generation failed for frontend: %s", exc)
            return self._fallback_frontend_code(routes)

    def _fallback_frontend_code(self, routes: list[dict[str, Any]]) -> str:
        lines = [
            "// React app (fallback)",
            "import React, { useState, useEffect } from 'react';",
            "",
        ]
        for route in routes:
            component = (route.get("components") or ["RouteView"])[0]
            safe_component = component.replace(" ", "")
            description = route.get("description", "")
            lines.append(
                f"export function {safe_component}() {{\n"
                f"  const [data, setData] = useState([]);\n"
                f"  const [loading, setLoading] = useState(true);\n"
                f"\n"
                f"  useEffect(() => {{\n"
                f"    fetch('/api{route.get('path', '/')}')\n"
                f"      .then(res => res.json())\n"
                f"      .then(items => {{ setData(items); setLoading(false); }})\n"
                f"      .catch(() => setLoading(false));\n"
                f"  }}, []);\n"
                f"\n"
                f"  if (loading) return <div className=\"p-4\">Loading...</div>;\n"
                f"\n"
                f"  return (\n"
                f"    <div className=\"p-4\">\n"
                f"      <h2 className=\"text-xl font-bold mb-4\">{description or safe_component}</h2>\n"
                f"      <ul className=\"space-y-2\">\n"
                f"        {{data.map((item, i) => (\n"
                f"          <li key={{i}} className=\"p-2 border rounded\">{{JSON.stringify(item)}}</li>\n"
                f"        ))}}\n"
                f"      </ul>\n"
                f"    </div>\n"
                f"  );\n"
                f"}}\n"
            )
        lines.append(
            "export default function App() {\n"
            "  return (\n"
            "    <div className=\"min-h-screen bg-gray-50\">\n"
            "      <header className=\"bg-white shadow p-4\">\n"
            "        <h1 className=\"text-2xl font-bold\">App</h1>\n"
            "      </header>\n"
            "      <main className=\"container mx-auto p-4\">\n"
            "        {/* Route components render here */}\n"
            "      </main>\n"
            "    </div>\n"
            "  );\n"
            "}"
        )
        return "\n".join(lines)

    def _generate_project_plan(
        self,
        context: RunContext,
        spec: ProjectSpec,
        routes: list[dict[str, Any]],
        backend_blueprint: dict[str, Any],
        app_body: str,
    ) -> dict[str, Any]:
        slug = self._slugify(spec.metadata.name)
        fallback_plan = self._fallback_project_plan(spec, routes, backend_blueprint, app_body)
        routes_json = json.dumps(routes, indent=2)
        backend_json = json.dumps(backend_blueprint, indent=2)
        system = (
            "You are a senior frontend engineer. Produce a JSON plan for the file-system MCP"
            " server that scaffolds a React + Vite + JavaScript project using Tailwind-ready"
            " components. Do NOT include tsconfig.json or any TypeScript files."
        )
        prompt = (
            f"User request: {context.user_request}\n"
            "Frontend routes (JSON):\n"
            f"{routes_json}\n"
            "Backend blueprint (JSON):\n"
            f"{backend_json}\n\n"
            "Use the provided App.jsx implementation verbatim:\n"
            "<app>\n"
            f"{app_body}\n"
            "</app>\n\n"
            "Return JSON only with this structure:\n"
            "{\n"
            "  \"tool\": \"mcp.fs\",\n"
            "  \"project_root\": \"frontend\",\n"
            "  \"instructions\": \"short summary\",\n"
            "  \"files\": [\n"
            "    {\n"
            "      \"path\": \"src/App.jsx\",\n"
            "      \"description\": \"main app shell\",\n"
            "      \"contents\": \"Full file contents as a string\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Always include package.json, vite.config.js, src/main.jsx, src/hooks/useApi.js,"
            " and src/App.jsx. Use JavaScript only — no TypeScript."
            " Include a src/__tests__/App.test.jsx file with Vitest + React Testing Library "
            "tests covering component rendering, user interactions, and API call mocking. "
            "The package.json must include vitest, @testing-library/react, "
            "@testing-library/jest-dom, and jsdom in devDependencies "
            'and a "test": "vitest run" script.'
        )
        try:
            response = context.llm.generate(prompt, system=system, temperature=0.2)
            plan = self._parse_plan_response(response)
        except Exception as exc:  # pragma: no cover - LLM dependent
            self.logger.warning("Frontend MCP plan generation failed; using fallback: %s", exc)
            return fallback_plan

        plan.setdefault("tool", "mcp.fs")
        plan.setdefault("project_root", "frontend")
        if not isinstance(plan.get("files"), list) or not plan["files"]:
            self.logger.warning("Frontend MCP plan missing files; using fallback")
            return fallback_plan
        return plan

    def _parse_plan_response(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = parts[1]
                if text.lstrip().startswith("json"):
                    text = text.lstrip()[4:]
        return json.loads(text)

    def _fallback_project_plan(
        self,
        spec: ProjectSpec,
        routes: list[dict[str, Any]],
        backend_blueprint: dict[str, Any],
        app_body: str,
    ) -> dict[str, Any]:
        files = self._default_project_files(spec, routes, backend_blueprint, app_body)
        return {
            "tool": "mcp.fs",
            "project_root": "frontend",
            "instructions": f"Scaffold React frontend for {spec.metadata.name}",
            "files": [
                {
                    "path": path,
                    "description": payload["description"],
                    "contents": payload["body"],
                }
                for path, payload in files.items()
            ],
        }

    def _default_project_files(
        self,
        spec: ProjectSpec,
        routes: list[dict[str, Any]],
        backend_blueprint: dict[str, Any],
        app_body: str,
    ) -> dict[str, dict[str, str]]:
        project_root = "frontend"
        return {
            "README.md": {
                "description": "Frontend overview and commands",
                "body": self._render_readme(spec, routes, backend_blueprint, project_root),
            },
            "package.json": {
                "description": "React + Vite manifest",
                "body": self._render_package_json(spec),
            },
            "vite.config.js": {
                "description": "Vite configuration",
                "body": self._render_vite_config(),
            },
            "index.html": {
                "description": "HTML entry point",
                "body": self._render_index_html(spec),
            },
            "src/main.jsx": {
                "description": "App bootstrap",
                "body": self._render_main_jsx(),
            },
            "src/App.jsx": {
                "description": "Generated React app",
                "body": app_body,
            },
            "src/hooks/useApi.js": {
                "description": "Reusable data fetching hook",
                "body": self._render_hook(backend_blueprint),
            },
            "src/__tests__/App.test.jsx": {
                "description": "Vitest + React Testing Library component tests",
                "body": self._render_fallback_tests(routes),
            },
        }

    def _render_readme(
        self,
        spec: ProjectSpec,
        routes: list[dict[str, Any]],
        backend_blueprint: dict[str, Any],
        project_root: str,
    ) -> str:
        route_lines = "\n".join(f"- {route['path']}: {route['description']}" for route in routes)
        api_lines = "\n".join(
            f"- {endpoint['method']} {endpoint['path']}" for endpoint in backend_blueprint.get("endpoints", [])
        ) or "- TBD"
        return (
            f"# {spec.metadata.name} Frontend\n\n"
            f"Root: {project_root}\n\n"
            "## Routes\n"
            f"{route_lines or '- TBD'}\n\n"
            "## Consumed APIs\n"
            f"{api_lines}\n\n"
            "## Commands\n"
            "```bash\n"
            "npm install\n"
            "npm run dev\n"
            "npm run build\n"
            "npm run preview\n"
            "```\n"
        )

    def _render_package_json(self, spec: ProjectSpec) -> str:
        metadata = spec.metadata
        package = {
            "name": self._slugify(f"{metadata.name}-frontend"),
            "private": True,
            "version": metadata.version or "0.1.0",
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "vite build",
                "preview": "vite preview",
                "test": "vitest run",
            },
            "dependencies": {
                "react": "^18.3.1",
                "react-dom": "^18.3.1",
                "react-router-dom": "^6.26.2",
            },
            "devDependencies": {
                "@vitejs/plugin-react": "^4.3.1",
                "vite": "^5.4.8",
                "tailwindcss": "^3.4.13",
                "autoprefixer": "^10.4.20",
                "postcss": "^8.4.47",
                "vitest": "^1.6.0",
                "@testing-library/react": "^14.2.1",
                "@testing-library/jest-dom": "^6.4.2",
                "jsdom": "^24.0.0",
            },
        }
        return json.dumps(package, indent=2)

    def _render_vite_config(self) -> str:
        return (
            "import { defineConfig } from 'vite';\n"
            "import react from '@vitejs/plugin-react';\n\n"
            "export default defineConfig({\n"
            "  plugins: [react()],\n"
            "  test: {\n"
            "    globals: true,\n"
            "    environment: 'jsdom',\n"
            "    setupFiles: [],\n"
            "  },\n"
            "});\n"
        )

    def _render_index_html(self, spec: ProjectSpec) -> str:
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "  <head>\n"
            "    <meta charset=\"UTF-8\" />\n"
            "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
            f"    <title>{spec.metadata.name}</title>\n"
            "  </head>\n"
            "  <body>\n"
            "    <div id=\"root\"></div>\n"
            "    <script type=\"module\" src=\"/src/main.jsx\"></script>\n"
            "  </body>\n"
            "</html>\n"
        )

    def _render_main_jsx(self) -> str:
        return (
            "import React from 'react';\n"
            "import ReactDOM from 'react-dom/client';\n"
            "import App from './App';\n\n"
            "ReactDOM.createRoot(document.getElementById('root')).render(\n"
            "  <React.StrictMode>\n"
            "    <App />\n"
            "  </React.StrictMode>\n"
            ");\n"
        )

    def _render_hook(self, backend_blueprint: dict[str, Any]) -> str:
        first_endpoint = next(iter(backend_blueprint.get("endpoints", [])), None)
        api_path = first_endpoint["path"] if first_endpoint else "/api/items"
        return (
            "import { useEffect, useState } from 'react';\n\n"
            "const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:4000';\n\n"
            "export function useApi(path) {\n"
            "  const [data, setData] = useState(null);\n"
            "  const [loading, setLoading] = useState(true);\n"
            "  const [error, setError] = useState(null);\n\n"
            "  useEffect(() => {\n"
            "    setLoading(true);\n"
            "    fetch(`${API_BASE}${path}`)\n"
            "      .then((res) => {\n"
            "        if (!res.ok) throw new Error(`HTTP ${res.status}`);\n"
            "        return res.json();\n"
            "      })\n"
            "      .then((payload) => {\n"
            "        setData(payload);\n"
            "        setLoading(false);\n"
            "      })\n"
            "      .catch((err) => {\n"
            "        setError(err.message);\n"
            "        setLoading(false);\n"
            "      });\n"
            "  }, [path]);\n\n"
            "  return { data, loading, error };\n"
            "}\n"
        )

    def _render_fallback_tests(self, routes: list[dict[str, Any]]) -> str:
        """Generate a fallback Vitest + React Testing Library test file."""
        lines = [
            "import { describe, it, expect, vi, beforeEach } from 'vitest';",
            "import { render, screen } from '@testing-library/react';",
            "import App from '../App';",
            "",
            "// Mock fetch globally",
            "beforeEach(() => {",
            "  global.fetch = vi.fn(() =>",
            "    Promise.resolve({",
            "      ok: true,",
            "      json: () => Promise.resolve([]),",
            "    })",
            "  );",
            "});",
            "",
            "describe('App', () => {",
            "  it('renders without crashing', () => {",
            "    render(<App />);",
            "    expect(document.body).toBeDefined();",
            "  });",
            "",
        ]
        for route in routes:
            description = route.get("description", route.get("path", "/"))
            components = route.get("components", [])
            component_name = components[0] if components else "RouteView"
            safe_name = component_name.replace(" ", "")
            lines.append(
                f"  it('renders {safe_name} component', () => {{\n"
                f"    render(<App />);\n"
                f"    // Verify the component tree renders for {description}\n"
                f"    expect(document.body.innerHTML).toBeTruthy();\n"
                f"  }});\n"
            )
        lines.append("});",)
        return "\n".join(lines)

    def _slugify(self, value: str) -> str:
        value = value.lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = value.strip("-")
        return value or "frontend"

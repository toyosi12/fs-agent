"""Backend API agent — specializes in Express routes, middleware, and controllers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..context import RunContext
from ..mcp import apply_filesystem_plan
from ..models.spec import ApiEndpoint, ProjectSpec
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class BackendApiAgent(BaseAgent):
    """Generates API routes, controllers, middleware, auth, and app scaffold.

    Used by the hierarchical pattern's 3-level topology.  Assumes the
    database layer (db.js, migrations) has already been created by
    BackendDbAgent.  Produces files in ``backend/src/`` (app.js,
    server.js, routes/).
    """

    role = AgentRole.BACKEND_API

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        spec = context.require_spec()
        backend = spec.backend

        endpoint_plans: list[dict[str, Any]] = []
        for endpoint in backend.endpoints:
            endpoint_plans.append({
                "name": endpoint.name,
                "path": endpoint.path,
                "method": endpoint.method.value,
                "auth_required": endpoint.auth_required,
                "websocket": endpoint.websocket,
                "request_schema": endpoint.request_schema,
                "response_schema": endpoint.response_schema,
            })

        blueprint: dict[str, Any] = {
            "language": backend.language,
            "framework": backend.framework,
            "style": backend.style,
            "endpoints": endpoint_plans,
            "data_models": [model.model_dump() for model in backend.data_models],
        }
        if backend.database:
            blueprint["database"] = backend.database.model_dump(mode="json")

        # Generate API routes via LLM
        router_code = self._generate_api_code(context, blueprint)
        mcp_plan = self._generate_api_plan(context, spec, blueprint, router_code)

        projects_root = context.projects_dir
        application = apply_filesystem_plan(
            mcp_plan, projects_root, dry_run=context.settings.dry_run,
        )
        created_files = [str(path) for path in application.created_files]
        mcp_plan = {**mcp_plan, "project_path": str(application.project_path)}

        attachments = [
            AgentArtifact(
                name="backend_api_routes.js",
                kind="code",
                description="LLM-generated Express router (API-focused)",
                body=router_code,
            ),
        ]

        result = AgentResult(
            role=self.role,
            summary=(
                f"Generated {len(endpoint_plans)} API routes with controllers "
                f"and middleware for {backend.framework}."
            ),
            artifacts={
                "backend_blueprint": blueprint,
                "backend_source": {
                    "language": backend.language,
                    "framework": backend.framework,
                    "body": router_code,
                },
                "backend_api_mcp_plan": mcp_plan,
                "backend_api_files": created_files,
            },
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _generate_api_code(
        self, context: RunContext, blueprint: dict[str, Any]
    ) -> str:
        """Generate Express routes focused on API logic only."""
        metadata = context.require_spec().metadata
        endpoint_lines = []
        for ep in blueprint["endpoints"]:
            line = f"- {ep['method']} {ep['path']}: {ep['name']}"
            if ep.get("request_schema"):
                line += f"\n  Request body: {json.dumps(ep['request_schema'])}"
            if ep.get("response_schema"):
                line += f"\n  Response: {json.dumps(ep['response_schema'])}"
            endpoint_lines.append(line)

        has_db = bool(blueprint.get("database"))

        prompt = (
            f"Project: {metadata.name} ({metadata.summary})\n\n"
            "Write a JavaScript Express router implementing these REST endpoints:\n"
            + "\n".join(endpoint_lines)
            + "\n\nFocus ONLY on the API route handlers and middleware.\n"
            "The database layer (db.js, migrations) already exists — "
            "import the db instance from '../db.js'.\n\n"
            "Requirements:\n"
            "- Complete, working request handlers with real business logic\n"
            "- Input validation for POST/PUT/PATCH endpoints\n"
            "- Proper HTTP status codes and JSON error responses\n"
            "- Authentication middleware if endpoints require it\n"
        )
        if has_db:
            prompt += (
                "- Use better-sqlite3 queries: db.prepare(sql).all() for reads, "
                "db.prepare(sql).run() for writes\n"
                "- Handle database errors with try/catch\n"
            )
        prompt += (
            "\nDo NOT generate database schema, migrations, or db.js — "
            "those are handled by a separate agent.\n"
            "Do NOT use TypeScript. Do NOT leave TODO comments.\n"
        )

        upstream = context.extra_context.get("upstream_context", "")
        if upstream:
            prompt += (
                "\n--- UPSTREAM CONTEXT ---\n"
                f"{upstream}\n"
                "--- END UPSTREAM CONTEXT ---\n"
            )

        prompt += (
            "\nAlso generate a companion test file using Jest.\n"
            "Wrap tests in a section starting with '// === TESTS ==='.\n"
        )

        system = (
            "You are a senior API engineer. Produce idiomatic Express/JavaScript "
            "route handlers. Focus on API logic, validation, and error handling. "
            "The database layer is provided — just import and use it. "
            "No placeholders, no TODOs, no TypeScript. "
            "Never use 'bcrypt' — use 'bcryptjs' instead."
        )

        try:
            return context.get_llm("backend_api").generate(
                prompt, system=system, temperature=0.1,
            )
        except Exception as exc:
            self.logger.warning("LLM generation failed for backend_api: %s", exc)
            return self._fallback_api_code(blueprint)

    def _fallback_api_code(self, blueprint: dict[str, Any]) -> str:
        lines = [
            "import { Router } from 'express';",
            "import db from '../db.js';",
            "",
            "const router = Router();",
            "",
        ]
        for ep in blueprint.get("endpoints", []):
            method = ep["method"].lower()
            path = ep["path"]
            lines.append(
                f"router.{method}('{path}', (req, res) => {{\n"
                f"  res.json({{ message: 'ok' }});\n"
                "});\n"
            )
        lines.append("export default router;")
        return "\n".join(lines)

    def _generate_api_plan(
        self,
        context: RunContext,
        spec: ProjectSpec,
        blueprint: dict[str, Any],
        router_code: str,
    ) -> dict[str, Any]:
        """Generate MCP plan for API layer files (app.js, server.js, routes)."""
        system = (
            "You are an expert Node.js platform engineer. Given API route code, "
            "produce a JSON plan for the file-system MCP server that scaffolds "
            "the Express application layer. Include: package.json, src/app.js, "
            "src/server.js, src/routes/index.js, src/routes/generated.js, "
            "jest.config.js, __tests__/, and a Dockerfile. "
            "Do NOT include database files (db.js, migrate.js, migrations/) — "
            "those are handled by a separate agent. "
            "Do NOT use TypeScript. "
            "Dockerfiles MUST use `npm install`, never `npm ci`. "
            "The Dockerfile must be production-ready: FROM node:20-alpine, "
            "npm install --omit=dev, EXPOSE 4000. "
            'CMD must run migrations then start: '
            'CMD ["sh", "-c", "node src/migrate.js && node src/server.js"]. '
            "Never use 'bcrypt' — use 'bcryptjs'. "
            "app.js MUST include health check endpoints on '/' and '/healthz'."
        )
        prompt = (
            f"User request: {context.user_request}\n"
            f"Backend blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
            "Router implementation (use verbatim in src/routes/generated.js):\n"
            f"<router>\n{router_code}\n</router>\n\n"
            "Respond with JSON only (no backticks):\n"
            '{"tool": "mcp.fs", "project_root": "backend", '
            '"instructions": "...", "files": [{"path": "...", '
            '"description": "...", "contents": "..."}]}'
        )

        try:
            raw = context.get_llm("backend_api").generate(
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
            plan.setdefault("project_root", "backend")
            if isinstance(plan.get("files"), list) and plan["files"]:
                return plan
        except Exception as exc:
            self.logger.warning("API plan generation failed: %s", exc)

        # Fallback: minimal plan
        return {
            "tool": "mcp.fs",
            "project_root": "backend",
            "instructions": f"API scaffold for {spec.metadata.name}",
            "files": [
                {
                    "path": "src/routes/generated.js",
                    "description": "Generated API routes",
                    "contents": router_code,
                },
            ],
        }

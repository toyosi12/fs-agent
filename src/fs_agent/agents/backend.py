"""Backend agent that plans APIs and emits MCP file plans."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from ..context import RunContext
from ..mcp import apply_filesystem_plan
from ..models.spec import ApiEndpoint, ProjectSpec
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class BackendAgent(BaseAgent):
    role = AgentRole.BACKEND

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        spec = context.require_spec()
        backend = spec.backend
        scoped = context.backend_context()  # only metadata + backend
        lines: list[str] = []
        endpoint_plans: list[dict[str, Any]] = []
        for endpoint in backend.endpoints:
            contract = self._describe_endpoint(endpoint)
            lines.append(contract)
            endpoint_plans.append({
                "name": endpoint.name,
                "path": endpoint.path,
                "method": endpoint.method.value,
                "auth_required": endpoint.auth_required,
                "websocket": endpoint.websocket,
                "tests": [f"returns 200 for {endpoint.path}"],
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

        code_body = self._generate_backend_code(context, blueprint, scoped)
        migration_files = self._generate_migrations(context, blueprint)
        mcp_plan = self._generate_project_plan(context, spec, blueprint, code_body, migration_files)
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
                name="backend_plan.md",
                kind="plan",
                description="High-level backend implementation outline",
                body="\n".join(lines) or "No endpoints defined.",
            ),
            AgentArtifact(
                name="backend_service.js",
                kind="code",
                description="LLM-generated Express router",
                body=code_body,
            ),
            AgentArtifact(
                name="backend_mcp_plan.json",
                kind="doc",
                description="Filesystem instructions compatible with MCP",
                body=json.dumps(mcp_plan, indent=2),
            ),
        ]

        result = AgentResult(
            role=self.role,
            summary=(
                f"Planned {len(endpoint_plans)} endpoints using {backend.framework} and "
                f"prepared {mcp_plan.get('project_root', 'backend')} MCP scaffold."
            ),
            artifacts={
                "backend_blueprint": blueprint,
                "backend_source": {
                    "language": backend.language,
                    "framework": backend.framework,
                    "body": code_body,
                },
                "backend_mcp_plan": mcp_plan,
                "backend_project_files": created_files,
            },
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _generate_backend_code(
        self, context: RunContext, blueprint: dict[str, Any], scoped: dict[str, Any]
    ) -> str:
        metadata = context.require_spec().metadata
        endpoint_lines = []
        for endpoint in blueprint["endpoints"]:
            endpoint_lines.append(
                f"- {endpoint['method']} {endpoint['path']}: {endpoint['name']}"
            )
        db_hint = ""
        if blueprint.get("database"):
            db_hint = (
                f"\nDatabase provider: {blueprint['database']['provider']}\n"
                f"Models: {json.dumps(blueprint['database'].get('models', []), indent=2)}\n"
            )
        prompt = (
            f"Project: {metadata.name} ({metadata.summary})\n"
            f"Owner: {metadata.owner}\n\n"
            "Write a JavaScript Express router that implements these REST endpoints:\n"
            + "\n".join(endpoint_lines)
            + db_hint
            + "\nReturn a complete file with imports, router setup, and fully working "
            "request handlers. Each handler must contain real business logic — validate "
            "inputs, and return proper JSON responses with correct HTTP status codes. "
        )
        if blueprint.get("database"):
            prompt += (
                "Use the mysql2/promise package to query a MySQL database. "
                "Import the pool from '../db.js'. Write real SQL queries (SELECT, INSERT, "
                "UPDATE, DELETE) against the tables defined in the data models. "
                "Handle database errors with try/catch and return 500 on failure. "
            )
        else:
            prompt += (
                "Use in-memory data structures (arrays/maps) as a data store. "
            )
        prompt += "Do NOT use TypeScript. Do NOT leave TODO comments or placeholder stubs."
        system = (
            "You are a senior backend engineer. Produce idiomatic Express/JavaScript code "
            "with complete, functional route handlers. Every endpoint must be fully "
            "implemented and return meaningful responses. No placeholders, no TODOs, "
            "no TypeScript syntax."
        )
        if blueprint.get("database"):
            system += (
                " Use mysql2/promise for all database access. Import the pool from '../db.js'."
            )
        try:
            return context.llm.generate(prompt, system=system, temperature=0.1)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("LLM generation failed for backend: %s", exc)
            return self._fallback_backend_code(blueprint)

    def _fallback_backend_code(self, blueprint: dict[str, Any]) -> str:
        lines = [
            "// Express router (fallback)",
            "import { Router } from 'express';",
            "",
            "const router = Router();",
            "",
            "// In-memory data store",
            "const store = new Map();",
            "let idCounter = 1;",
            "",
        ]
        for endpoint in blueprint.get("endpoints", []):
            method = endpoint["method"].lower()
            path = endpoint["path"]
            name = endpoint["name"]
            if method == "get" and ":id" not in path:
                lines.append(
                    f"router.get('{path}', (req, res) => {{\n"
                    f"  // {name}\n"
                    "  const items = Array.from(store.values());\n"
                    "  return res.json(items);\n"
                    "});\n"
                )
            elif method == "get":
                lines.append(
                    f"router.get('{path}', (req, res) => {{\n"
                    f"  // {name}\n"
                    "  const item = store.get(req.params.id);\n"
                    "  if (!item) return res.status(404).json({ error: 'Not found' });\n"
                    "  return res.json(item);\n"
                    "});\n"
                )
            elif method == "post":
                lines.append(
                    f"router.post('{path}', (req, res) => {{\n"
                    f"  // {name}\n"
                    "  const id = String(idCounter++);\n"
                    "  const item = { id, ...req.body, createdAt: new Date().toISOString() };\n"
                    "  store.set(id, item);\n"
                    "  return res.status(201).json(item);\n"
                    "});\n"
                )
            elif method == "put" or method == "patch":
                lines.append(
                    f"router.{method}('{path}', (req, res) => {{\n"
                    f"  // {name}\n"
                    "  const existing = store.get(req.params.id);\n"
                    "  if (!existing) return res.status(404).json({ error: 'Not found' });\n"
                    "  const updated = { ...existing, ...req.body, updatedAt: new Date().toISOString() };\n"
                    "  store.set(req.params.id, updated);\n"
                    "  return res.json(updated);\n"
                    "});\n"
                )
            elif method == "delete":
                lines.append(
                    f"router.delete('{path}', (req, res) => {{\n"
                    f"  // {name}\n"
                    "  if (!store.has(req.params.id)) return res.status(404).json({ error: 'Not found' });\n"
                    "  store.delete(req.params.id);\n"
                    "  return res.status(204).end();\n"
                    "});\n"
                )
            else:
                lines.append(
                    f"router.{method}('{path}', (req, res) => {{\n"
                    f"  // {name}\n"
                    "  return res.json({ message: 'ok' });\n"
                    "});\n"
                )
        lines.append("export default router;")
        return "\n".join(lines)

    def _generate_project_plan(
        self,
        context: RunContext,
        spec: ProjectSpec,
        blueprint: dict[str, Any],
        router_body: str,
        migration_files: dict[str, str],
    ) -> dict[str, Any]:
        slug = self._slugify(spec.metadata.name)
        fallback_plan = self._fallback_project_plan(spec, blueprint, router_body, migration_files)
        backend_json = json.dumps(blueprint, indent=2)
        migration_list = "\n".join(f"- {path}" for path in migration_files) if migration_files else "(none)"
        system = (
            "You are an expert Node.js platform engineer. Given a backend specification,"
            " produce a JSON plan for the file-system MCP server that scaffolds an Express"
            " + JavaScript project. Always include package.json, src/app.js,"
            " src/server.js, src/routes/index.js, and src/routes/generated.js."
            " Do NOT include tsconfig.json or any TypeScript files."
        )
        if migration_files:
            system += (
                " Also include src/db.js (MySQL connection pool), src/migrate.js (migration runner),"
                " and all migration SQL files under migrations/."
            )
        prompt = (
            f"User request: {context.user_request}\n"
            "Backend blueprint (JSON):\n"
            f"{backend_json}\n\n"
            "Existing router implementation (reuse verbatim in src/routes/generated.js):\n"
            "<router>\n"
            f"{router_body}\n"
            "</router>\n\n"
        )
        if migration_files:
            prompt += "Migration files to include verbatim:\n"
            for path, contents in migration_files.items():
                prompt += f"<file path=\"{path}\">\n{contents}\n</file>\n"
            prompt += "\n"
        prompt += (
            "Respond with JSON only (no backticks) using this shape:\n"
            "{\n"
            "  \"tool\": \"mcp.fs\",\n"
            "  \"project_root\": \"<slug>-backend\",\n"
            "  \"instructions\": \"short summary\",\n"
            "  \"files\": [\n"
            "    {\n"
            "      \"path\": \"package.json\",\n"
            "      \"description\": \"what this file does\",\n"
            "      \"contents\": \"Full file contents as a string\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Paths are relative to the project root. Use JavaScript only — no TypeScript."
        )
        try:
            response = context.llm.generate(prompt, system=system, temperature=0.15)
            plan = self._parse_plan_response(response)
        except Exception as exc:  # pragma: no cover - LLM dependent
            self.logger.warning("Backend MCP plan generation failed; using fallback: %s", exc)
            return fallback_plan

        plan.setdefault("tool", "mcp.fs")
        plan.setdefault("project_root", f"{slug}-backend")
        if not isinstance(plan.get("files"), list) or not plan["files"]:
            self.logger.warning("Backend MCP plan missing files; using fallback")
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
        blueprint: dict[str, Any],
        router_body: str,
        migration_files: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        files = self._default_project_files(spec, blueprint, router_body, migration_files)
        slug = self._slugify(spec.metadata.name)
        return {
            "tool": "mcp.fs",
            "project_root": f"{slug}-backend",
            "instructions": f"Scaffold Express backend for {spec.metadata.name}",
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
        blueprint: dict[str, Any],
        router_body: str,
        migration_files: dict[str, str] | None = None,
    ) -> dict[str, dict[str, str]]:
        project_root = f"{self._slugify(spec.metadata.name)}-backend"
        files = {
            "README.md": {
                "description": "Backend overview and commands",
                "body": self._render_readme(spec, blueprint, project_root),
            },
            "package.json": {
                "description": "Node manifest",
                "body": self._render_package_json(spec, has_db=bool(migration_files)),
            },
            ".env.example": {
                "description": "Environment variable template",
                "body": self._render_env_example(blueprint),
            },
            "src/server.js": {
                "description": "Server bootstrap",
                "body": self._render_server_js(),
            },
            "src/app.js": {
                "description": "Express app wiring",
                "body": self._render_app_js(blueprint),
            },
            "src/routes/index.js": {
                "description": "Router entry point",
                "body": self._render_routes_index(),
            },
            "src/routes/generated.js": {
                "description": "Generated REST endpoints",
                "body": router_body,
            },
        }
        if migration_files:
            files["src/db.js"] = {
                "description": "MySQL connection pool",
                "body": self._render_db_js(),
            }
            files["src/migrate.js"] = {
                "description": "Migration runner",
                "body": self._render_migrate_js(),
            }
            for path, contents in migration_files.items():
                files[path] = {
                    "description": f"MySQL migration: {path}",
                    "body": contents,
                }
        return files

    def _render_readme(
        self,
        spec: ProjectSpec,
        blueprint: dict[str, Any],
        project_root: str,
    ) -> str:
        metadata = spec.metadata
        endpoints = "\n".join(
            f"- {endpoint['method']} {endpoint['path']} — {endpoint['name']}"
            for endpoint in blueprint["endpoints"]
        )
        return (
            f"# {metadata.name} Backend\n\n"
            f"Project root: {project_root}\n\n"
            f"{metadata.summary}\n\n"
            "## Endpoints\n"
            f"{endpoints or '- TBD'}\n\n"
            "## Setup\n"
            "```bash\n"
            "npm install\n"
            "cp .env.example .env  # edit with your MySQL credentials\n"
            "npm run migrate       # run database migrations\n"
            "npm run dev           # start dev server with hot-reload\n"
            "npm start             # start production server\n"
            "```\n"
        )

    def _render_package_json(self, spec: ProjectSpec, has_db: bool = False) -> str:
        metadata = spec.metadata
        scripts = {
            "dev": "nodemon src/server.js",
            "start": "node src/server.js",
        }
        deps = {
            "cors": "^2.8.5",
            "dotenv": "^16.4.5",
            "express": "^4.19.2",
            "morgan": "^1.10.0",
        }
        if has_db:
            scripts["migrate"] = "node src/migrate.js"
            deps["mysql2"] = "^3.11.0"
        package = {
            "name": self._slugify(f"{metadata.name}-backend"),
            "version": metadata.version or "0.1.0",
            "private": True,
            "type": "module",
            "scripts": scripts,
            "dependencies": deps,
            "devDependencies": {
                "nodemon": "^3.1.4",
            },
        }
        return json.dumps(package, indent=2)

    def _render_server_js(self) -> str:
        return (
            "import { app } from './app.js';\n\n"
            "const port = Number(process.env.PORT ?? 4000);\n"
            "app.listen(port, () => {\n"
            "  console.log(`API listening on port ${port}`);\n"
            "});\n"
        )

    def _render_app_js(self, blueprint: dict[str, Any]) -> str:
        documented_routes = "\n".join(
            f"// {endpoint['method']} {endpoint['path']} — {endpoint['name']}"
            for endpoint in blueprint["endpoints"]
        )
        return (
            "import cors from 'cors';\n"
            "import express from 'express';\n"
            "import morgan from 'morgan';\n"
            "import apiRouter from './routes/index.js';\n\n"
            "export const app = express();\n\n"
            "app.use(cors());\n"
            "app.use(express.json());\n"
            "app.use(morgan('dev'));\n\n"
            "app.use('/api', apiRouter);\n\n"
            "app.get('/healthz', (_req, res) => {\n"
            "  return res.json({ status: 'ok', time: new Date().toISOString() });\n"
            "});\n\n"
            f"{documented_routes}\n"
        )

    def _render_routes_index(self) -> str:
        return (
            "import { Router } from 'express';\n"
            "import generatedRouter from './generated.js';\n\n"
            "const router = Router();\n"
            "router.use('/', generatedRouter);\n"
            "export default router;\n"
        )

    def _generate_migrations(
        self, context: RunContext, blueprint: dict[str, Any]
    ) -> dict[str, str]:
        """Generate ready-to-run MySQL migration SQL files.

        Returns a dict of {relative_path: sql_contents}.
        """
        db = blueprint.get("database")
        if not db:
            return {}

        models = db.get("models", [])
        existing_migrations = db.get("migrations", [])
        if not models and not existing_migrations:
            return {}

        models_json = json.dumps(models, indent=2)
        migrations_hint = ""
        if existing_migrations:
            migrations_hint = (
                "\n\nThe architect already sketched these migrations — use them as guidance "
                "but produce complete, production-ready SQL:\n"
                + json.dumps(existing_migrations, indent=2)
            )

        system = (
            "You are a senior database engineer. Produce ready-to-run MySQL migration "
            "SQL files. Each migration must be a complete, valid SQL script. Use "
            "CREATE TABLE IF NOT EXISTS, proper column types (INT, VARCHAR, TEXT, "
            "DATETIME, BOOLEAN, etc.), PRIMARY KEY, FOREIGN KEY constraints, and indexes. "
            "Output valid JSON only."
        )
        prompt = (
            f"Generate MySQL migration SQL files for these data models:\n{models_json}"
            f"{migrations_hint}\n\n"
            "Return a JSON object where each key is a file path like "
            "\"migrations/001_create_users.sql\" and the value is the full SQL contents.\n"
            "Number files sequentially (001, 002, ...). Each file should contain:\n"
            "- A comment header with the migration name\n"
            "- CREATE TABLE IF NOT EXISTS with all columns, types, constraints\n"
            "- Any ALTER TABLE for foreign keys\n"
            "- CREATE INDEX statements\n\n"
            "Also include a final migration file for any seed/reference data if appropriate.\n"
            "Respond with JSON only (no markdown fences)."
        )

        try:
            raw = context.llm.generate(prompt, system=system, temperature=0.1)
            cleaned = self._strip_code_fences(raw).strip()
            result = json.loads(cleaned)
            if isinstance(result, dict) and result:
                return {k: v for k, v in result.items() if isinstance(v, str)}
        except Exception as exc:
            self.logger.warning("LLM migration generation failed: %s", exc)

        return self._fallback_migrations(models)

    def _fallback_migrations(self, models: list[dict[str, Any]]) -> dict[str, str]:
        """Generate basic CREATE TABLE migrations from data model definitions."""
        files: dict[str, str] = {}
        for i, model in enumerate(models, start=1):
            name = model.get("name", f"table_{i}")
            table = model.get("table_name") or self._slugify(name).replace("-", "_") + "s"
            fields = model.get("fields", {})
            seq = str(i).zfill(3)
            filename = f"migrations/{seq}_create_{table}.sql"

            col_lines = ["  id INT AUTO_INCREMENT PRIMARY KEY"]
            for field_name, field_type in fields.items():
                if field_name.lower() == "id":
                    continue
                mysql_type = self._js_type_to_mysql(field_type)
                col_lines.append(f"  {field_name} {mysql_type}")

            col_lines.append("  created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
            col_lines.append(
                "  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
            )

            sql = (
                f"-- Migration: Create {table}\n\n"
                f"CREATE TABLE IF NOT EXISTS {table} (\n"
                + ",\n".join(col_lines)
                + "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n"
            )

            indexes = model.get("indexes", [])
            for idx in indexes:
                idx_name = f"idx_{table}_{idx}"
                sql += f"\nCREATE INDEX {idx_name} ON {table} ({idx});\n"

            files[filename] = sql
        return files

    def _js_type_to_mysql(self, js_type: str) -> str:
        """Map JavaScript/generic type strings to MySQL column types."""
        t = str(js_type).lower().strip()
        mapping = {
            "string": "VARCHAR(255)",
            "text": "TEXT",
            "number": "INT",
            "integer": "INT",
            "int": "INT",
            "float": "DECIMAL(10,2)",
            "decimal": "DECIMAL(10,2)",
            "boolean": "TINYINT(1) DEFAULT 0",
            "bool": "TINYINT(1) DEFAULT 0",
            "date": "DATE",
            "datetime": "DATETIME",
            "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "json": "JSON",
            "uuid": "CHAR(36)",
        }
        return mapping.get(t, "VARCHAR(255)")

    def _strip_code_fences(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            parts = stripped.split("```", 2)
            if len(parts) >= 2:
                inner = parts[1]
                if inner.lstrip().startswith("json"):
                    inner = inner.lstrip()[4:]
                elif inner.lstrip().startswith("sql"):
                    inner = inner.lstrip()[3:]
                return inner
        return stripped

    def _render_db_js(self) -> str:
        return (
            "import mysql from 'mysql2/promise';\n"
            "import dotenv from 'dotenv';\n\n"
            "dotenv.config();\n\n"
            "const pool = mysql.createPool({\n"
            "  host: process.env.DB_HOST || 'localhost',\n"
            "  port: Number(process.env.DB_PORT || 3306),\n"
            "  user: process.env.DB_USER || 'root',\n"
            "  password: process.env.DB_PASSWORD || '',\n"
            "  database: process.env.DB_NAME || 'app_db',\n"
            "  waitForConnections: true,\n"
            "  connectionLimit: 10,\n"
            "  queueLimit: 0,\n"
            "});\n\n"
            "export default pool;\n"
        )

    def _render_migrate_js(self) -> str:
        return (
            "import fs from 'fs';\n"
            "import path from 'path';\n"
            "import { fileURLToPath } from 'url';\n"
            "import pool from './db.js';\n\n"
            "const __dirname = path.dirname(fileURLToPath(import.meta.url));\n"
            "const migrationsDir = path.join(__dirname, '..', 'migrations');\n\n"
            "async function migrate() {\n"
            "  const conn = await pool.getConnection();\n"
            "  try {\n"
            "    // Create migrations tracking table\n"
            "    await conn.execute(`\n"
            "      CREATE TABLE IF NOT EXISTS _migrations (\n"
            "        id INT AUTO_INCREMENT PRIMARY KEY,\n"
            "        name VARCHAR(255) NOT NULL UNIQUE,\n"
            "        applied_at DATETIME DEFAULT CURRENT_TIMESTAMP\n"
            "      )\n"
            "    `);\n\n"
            "    const [applied] = await conn.execute('SELECT name FROM _migrations');\n"
            "    const appliedSet = new Set(applied.map(r => r.name));\n\n"
            "    const files = fs.readdirSync(migrationsDir)\n"
            "      .filter(f => f.endsWith('.sql'))\n"
            "      .sort();\n\n"
            "    for (const file of files) {\n"
            "      if (appliedSet.has(file)) {\n"
            "        console.log(`  skip: ${file} (already applied)`);\n"
            "        continue;\n"
            "      }\n"
            "      const sql = fs.readFileSync(path.join(migrationsDir, file), 'utf8');\n"
            "      console.log(`  run:  ${file}`);\n"
            "      const statements = sql.split(';').map(s => s.trim()).filter(Boolean);\n"
            "      for (const stmt of statements) {\n"
            "        await conn.execute(stmt);\n"
            "      }\n"
            "      await conn.execute('INSERT INTO _migrations (name) VALUES (?)', [file]);\n"
            "    }\n\n"
            "    console.log('Migrations complete.');\n"
            "  } finally {\n"
            "    conn.release();\n"
            "    await pool.end();\n"
            "  }\n"
            "}\n\n"
            "migrate().catch(err => {\n"
            "  console.error('Migration failed:', err);\n"
            "  process.exit(1);\n"
            "});\n"
        )

    def _render_env_example(self, blueprint: dict[str, Any]) -> str:
        lines = [
            "PORT=4000",
        ]
        if blueprint.get("database"):
            lines.extend([
                "",
                "# MySQL connection",
                "DB_HOST=localhost",
                "DB_PORT=3306",
                "DB_USER=root",
                "DB_PASSWORD=",
                "DB_NAME=app_db",
            ])
        return "\n".join(lines) + "\n"

    def _slugify(self, value: str) -> str:
        value = value.lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = value.strip("-")
        return value or "backend"

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

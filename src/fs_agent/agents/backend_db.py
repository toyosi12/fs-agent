"""Backend DB agent — specializes in database schema, migrations, and data access."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..context import RunContext
from ..mcp import apply_filesystem_plan
from ..models.spec import ProjectSpec
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class BackendDbAgent(BaseAgent):
    """Generates database layer: schema, migrations, db.js, and seed data.

    Used by the hierarchical pattern's 3-level topology to separate
    database concerns from API route logic.  Produces files in the
    ``backend/`` directory (migrations/, src/db.js, src/migrate.js).
    """

    role = AgentRole.BACKEND_DB

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        spec = context.require_spec()
        backend = spec.backend

        blueprint: dict[str, Any] = {
            "language": backend.language,
            "framework": backend.framework,
            "data_models": [model.model_dump() for model in backend.data_models],
        }
        if backend.database:
            blueprint["database"] = backend.database.model_dump(mode="json")

        # Generate the database layer via LLM
        db_code = self._generate_db_code(context, blueprint)
        migration_files = self._generate_migrations(context, blueprint)
        mcp_plan = self._generate_db_plan(context, spec, blueprint, db_code, migration_files)

        projects_root = context.projects_dir
        application = apply_filesystem_plan(
            mcp_plan, projects_root, dry_run=context.settings.dry_run,
        )
        created_files = [str(path) for path in application.created_files]
        mcp_plan = {**mcp_plan, "project_path": str(application.project_path)}

        attachments = [
            AgentArtifact(
                name="backend_db_plan.md",
                kind="plan",
                description="Database schema and migration plan",
                body=db_code,
            ),
        ]

        result = AgentResult(
            role=self.role,
            summary=(
                f"Generated database layer with {len(migration_files)} migrations "
                f"and {len(blueprint.get('data_models', []))} models."
            ),
            artifacts={
                "backend_db_blueprint": blueprint,
                "backend_db_source": {"language": "javascript", "body": db_code},
                "backend_db_mcp_plan": mcp_plan,
                "backend_db_files": created_files,
            },
            attachments=attachments,
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _generate_db_code(
        self, context: RunContext, blueprint: dict[str, Any]
    ) -> str:
        """Generate db.js connection module and migrate.js runner."""
        metadata = context.require_spec().metadata
        models_json = json.dumps(blueprint.get("data_models", []), indent=2)
        db_config = blueprint.get("database", {})

        prompt = (
            f"Project: {metadata.name} ({metadata.summary})\n\n"
            f"Database provider: {db_config.get('provider', 'sqlite')}\n"
            f"Data models:\n{models_json}\n\n"
            "Generate TWO JavaScript files for a Node.js/Express backend:\n\n"
            "1. **src/db.js** — Database connection module using better-sqlite3.\n"
            "   - Create/open a SQLite database file at './data/app.db'\n"
            "   - Enable WAL mode for better concurrency\n"
            "   - Export the db instance as default\n\n"
            "2. **src/migrate.js** — Migration runner.\n"
            "   - Read all .sql files from '../migrations/' directory\n"
            "   - Execute them in alphabetical order\n"
            "   - Log each migration applied\n"
            "   - Handle errors gracefully\n\n"
            "Separate the two files with '// === FILE: src/migrate.js ===' marker.\n"
            "Do NOT use TypeScript."
        )

        upstream = context.extra_context.get("upstream_context", "")
        if upstream:
            prompt += (
                "\n--- UPSTREAM CONTEXT ---\n"
                f"{upstream}\n"
                "--- END UPSTREAM CONTEXT ---\n"
            )

        system = (
            "You are a senior database engineer. Produce idiomatic JavaScript "
            "using better-sqlite3 for SQLite access. Every function must be "
            "fully implemented — no placeholders or TODOs."
        )

        try:
            return context.get_llm("backend_db").generate(
                prompt, system=system, temperature=0.1,
            )
        except Exception:
            return self._fallback_db_code()

    def _fallback_db_code(self) -> str:
        return (
            "// src/db.js\n"
            "import Database from 'better-sqlite3';\n"
            "import { mkdirSync } from 'fs';\n"
            "mkdirSync('./data', { recursive: true });\n"
            "const db = new Database('./data/app.db');\n"
            "db.pragma('journal_mode = WAL');\n"
            "export default db;\n"
            "\n// === FILE: src/migrate.js ===\n"
            "import db from './db.js';\n"
            "import { readdirSync, readFileSync } from 'fs';\n"
            "import { join } from 'path';\n"
            "const dir = join(import.meta.dirname, '..', 'migrations');\n"
            "const files = readdirSync(dir).filter(f => f.endsWith('.sql')).sort();\n"
            "for (const f of files) {\n"
            "  const sql = readFileSync(join(dir, f), 'utf8');\n"
            "  db.exec(sql);\n"
            "  console.log('Applied:', f);\n"
            "}\n"
        )

    def _generate_migrations(
        self, context: RunContext, blueprint: dict[str, Any]
    ) -> dict[str, str]:
        """Generate SQLite migration SQL files."""
        db = blueprint.get("database")
        if not db:
            return {}
        models = db.get("models", [])
        if not models:
            return {}

        models_json = json.dumps(models, indent=2)
        system = (
            "You are a senior database engineer. Produce ready-to-run SQLite migration "
            "SQL files. Use CREATE TABLE IF NOT EXISTS, proper column types, "
            "PRIMARY KEY with AUTOINCREMENT. Do NOT use MySQL syntax. Output JSON only."
        )
        prompt = (
            f"Generate SQLite migration SQL files for these models:\n{models_json}\n\n"
            "Return a JSON object: keys are file paths like "
            "\"migrations/001_create_users.sql\", values are full SQL.\n"
            "Respond with JSON only (no markdown fences)."
        )
        try:
            raw = context.get_llm("backend_db").generate(
                prompt, system=system, temperature=0.1,
            )
            text = raw.strip()
            if text.startswith("```"):
                parts = text.split("```", 2)
                if len(parts) >= 2:
                    text = parts[1]
                    if text.lstrip().startswith("json"):
                        text = text.lstrip()[4:]
            result = json.loads(text.strip())
            if isinstance(result, dict):
                return {k: v for k, v in result.items() if isinstance(v, str)}
        except Exception as exc:
            self.logger.warning("Migration generation failed: %s", exc)
        return {}

    def _generate_db_plan(
        self,
        context: RunContext,
        spec: ProjectSpec,
        blueprint: dict[str, Any],
        db_code: str,
        migration_files: dict[str, str],
    ) -> dict[str, Any]:
        """Generate MCP file plan for the database layer only."""
        files = []

        # db.js and migrate.js from the generated code
        if "// === FILE:" in db_code:
            parts = db_code.split("// === FILE:")
            files.append({
                "path": "src/db.js",
                "description": "SQLite database connection",
                "contents": parts[0].strip(),
            })
            for part in parts[1:]:
                lines = part.strip().split("\n", 1)
                path = lines[0].strip().rstrip("=").strip()
                body = lines[1] if len(lines) > 1 else ""
                files.append({
                    "path": path,
                    "description": f"Database utility: {path}",
                    "contents": body.strip(),
                })
        else:
            files.append({
                "path": "src/db.js",
                "description": "SQLite database connection",
                "contents": db_code,
            })

        # Migration files
        for path, sql in migration_files.items():
            files.append({
                "path": path,
                "description": f"Migration: {path}",
                "contents": sql,
            })

        # Data directory placeholder
        files.append({
            "path": "data/.gitkeep",
            "description": "Ensure data directory exists",
            "contents": "",
        })

        return {
            "tool": "mcp.fs",
            "project_root": "backend",
            "instructions": f"Database layer for {spec.metadata.name}",
            "files": files,
        }

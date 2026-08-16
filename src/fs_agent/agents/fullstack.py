"""Single full-stack agent used as the non-orchestrated baseline."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from ..context import RunContext
from ..mcp import apply_filesystem_plan
from .base import AgentArtifact, AgentResult, AgentRole, BaseAgent


class FullstackAgent(BaseAgent):
    """Own architecture, backend, frontend, and deployment in one agent call."""

    role = AgentRole.FULLSTACK

    def run(self, context: RunContext) -> AgentResult:
        start = datetime.now(timezone.utc)
        response = context.get_llm(self.role.value).generate(
            self._build_prompt(context),
            system=(
                "You are a senior full-stack engineer working independently. Design and "
                "implement the entire requested application as one coherent system. Return "
                "only the requested JSON filesystem plan; no markdown or commentary."
            ),
            temperature=0.1,
        )
        plan = self._parse_plan(response)
        application = apply_filesystem_plan(
            plan,
            context.artifact_dir / "projects",
            dry_run=context.settings.dry_run,
        )
        context.project_dir_override = application.project_path
        plan = {**plan, "project_path": str(application.project_path)}
        created_files = [str(path) for path in application.created_files]
        result = AgentResult(
            role=self.role,
            summary=(
                "Generated a complete full-stack project in one agent run with "
                f"{len(created_files)} files."
            ),
            artifacts={
                "fullstack_mcp_plan": plan,
                "fullstack_project_files": created_files,
            },
            attachments=[AgentArtifact(
                name="fullstack_mcp_plan.json",
                kind="doc",
                description="Single-agent full-stack filesystem plan",
                body=json.dumps(plan, indent=2),
            )],
            started_at=start,
            finished_at=datetime.now(timezone.utc),
        )
        self.logger.info(result.summary)
        return result

    def _build_prompt(self, context: RunContext) -> str:
        acceptance = ""
        if context.task_test_cases:
            acceptance = (
                "\nAcceptance criteria from the benchmark:\n"
                + json.dumps(context.task_test_cases, indent=2)
                + "\n"
            )
        return f"""Build this complete application without delegating work:

{context.user_request}
{acceptance}
Return one valid JSON object with this exact envelope:
{{
  "tool": "mcp.fs",
  "project_root": "a-short-kebab-case-project-name",
  "directories": ["backend/src", "frontend/src"],
  "files": [{{"path": "relative/path", "contents": "complete file contents"}}]
}}

Generate a coherent, runnable JavaScript monorepo containing:
- backend/package.json and a complete Express API on port 4000;
- persistent SQLite storage with migrations or automatic schema initialization;
- every API endpoint and business rule needed by the request;
- backend tests using Jest and Supertest;
- frontend/package.json and a complete React + Vite application;
- real fetch calls whose methods, paths, payloads, and response shapes exactly match the backend;
- frontend tests using Vitest and React Testing Library;
- backend and frontend Dockerfiles plus a root docker-compose.yml;
- clear run instructions in README.md.

Use JavaScript only, never TypeScript. Use bcryptjs rather than bcrypt. Do not emit TODOs,
placeholders, mock application data, omitted sections, or markdown fences. Every imported npm
package must be declared. Dockerfiles must use npm install rather than npm ci. The JSON must be
parseable and every file's contents must be encoded as a JSON string.
"""

    @staticmethod
    def _parse_plan(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```", 2)
            if len(parts) >= 2:
                text = re.sub(r"^json\s*", "", parts[1].strip(), count=1)
        try:
            plan = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Full-stack agent returned no JSON filesystem plan")
            plan = json.loads(text[start : end + 1])
        if not isinstance(plan, dict):
            raise ValueError("Full-stack agent filesystem plan must be a JSON object")
        return plan

"""Shared runtime context passed between agents."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import re

from .config import Settings
from .llm import BaseLLMClient
from .models.spec import ProjectSpec

if TYPE_CHECKING:
    from .orchestration.metrics import OrchestrationMetrics


@dataclass
class AgentReport:
    """Captured output from an agent run."""

    role: str
    summary: str
    artifacts: dict[str, Any]
    status: str
    started_at: datetime
    finished_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_metrics() -> OrchestrationMetrics:
    """Deferred factory to avoid circular import with orchestration.metrics."""
    from .orchestration.metrics import OrchestrationMetrics as _OM
    return _OM()


@dataclass
class RunContext:
    """Mutable state that flows through the orchestration sequence."""

    spec: ProjectSpec | None
    user_request: str
    settings: Settings
    workspace_dir: Path
    artifact_dir: Path
    llm: BaseLLMClient
    # Optional per-role LLM overrides; keys are role names
    # (e.g. "architect", "backend", "frontend", "infra").
    llm_per_role: dict[str, BaseLLMClient] = field(default_factory=dict)
    transcripts: list[AgentReport] = field(default_factory=list)
    # Populated by the orchestration pattern during a run.
    metrics: OrchestrationMetrics = field(default_factory=lambda: _make_metrics())
    # Pattern-injected context: patterns set this before running an agent
    # so agents can incorporate inter-agent artifacts into their prompts.
    extra_context: dict[str, Any] = field(default_factory=dict)

    def record(self, report: AgentReport) -> None:
        self.transcripts.append(report)

    def update_spec(self, spec: ProjectSpec) -> None:
        self.spec = spec

    def require_spec(self) -> ProjectSpec:
        if self.spec is None:
            raise RuntimeError("Project specification has not been generated yet.")
        return self.spec

    @property
    def artifacts(self) -> dict[str, Any]:
        combined: dict[str, Any] = {}
        for report in self.transcripts:
            combined.update(report.artifacts)
        return combined

    @property
    def projects_dir(self) -> Path:
        """Return the dedicated parent folder for this project's generated code.

        Structure: <artifact_dir>/projects/<slug>/
        """
        spec = self.require_spec()
        slug = self._slugify(spec.metadata.name)
        return self.artifact_dir / "projects" / slug

    @property
    def metadata_dir(self) -> Path:
        """Return the metadata subfolder inside the project directory.

        Structure: <artifact_dir>/projects/<slug>/metadata/
        """
        return self.projects_dir / "metadata"

    @staticmethod
    def _slugify(value: str) -> str:
        value = value.lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        value = value.strip("-")
        return value or "project"

    # ------------------------------------------------------------------
    # Scoped context helpers – each returns only what the agent needs
    # ------------------------------------------------------------------

    def backend_context(self) -> dict[str, Any]:
        """Return metadata + backend slice of the spec."""
        spec = self.require_spec()
        return {
            "metadata": spec.metadata.model_dump(mode="json"),
            "backend": spec.backend.model_dump(mode="json"),
        }

    def frontend_context(self) -> dict[str, Any]:
        """Return metadata + frontend slice, plus backend endpoint
        signatures (read-only) so the frontend can wire API calls."""
        spec = self.require_spec()
        endpoint_signatures = [
            {
                "name": ep.name,
                "method": ep.method.value,
                "path": ep.path,
                "request_schema": ep.request_schema,
                "response_schema": ep.response_schema,
            }
            for ep in spec.backend.endpoints
        ]
        return {
            "metadata": spec.metadata.model_dump(mode="json"),
            "frontend": spec.frontend.model_dump(mode="json"),
            "backend_endpoints": endpoint_signatures,
        }

    def infra_context(self) -> dict[str, Any]:
        """Return metadata + infra slice of the spec."""
        spec = self.require_spec()
        return {
            "metadata": spec.metadata.model_dump(mode="json"),
            "infra": spec.infra.model_dump(mode="json"),
        }

    # ------------------------------------------------------------------
    # Inter-agent artifact access
    # ------------------------------------------------------------------

    def agent_output(self, role: str) -> dict[str, Any] | None:
        """Return the artifacts dict from a completed agent, or None.

        For compound roles like 'backend', also checks specialized
        sub-roles ('backend_api', 'backend_db') and merges them.
        """
        # Direct match first
        for report in self.transcripts:
            if report.role == role:
                return report.artifacts

        # Check if specialized sub-agents produced output for this role
        _SUB_ROLES = {
            "backend": ["backend_api", "backend_db"],
            "frontend": ["frontend_pages", "frontend_ui"],
        }
        sub_roles = _SUB_ROLES.get(role, [])
        if sub_roles:
            merged: dict[str, Any] = {}
            found = False
            for report in self.transcripts:
                if report.role in sub_roles:
                    merged.update(report.artifacts)
                    found = True
            if found:
                return merged

        return None

    def extract_backend_contract(self) -> str:
        """Extract a compact API contract from the backend agent's actual output.

        Parses the generated router code to find route definitions and
        returns a focused summary (~500 tokens) that downstream agents can
        use without needing the full source.
        """
        output = self.agent_output("backend")
        if not output:
            return ""

        parts: list[str] = []

        # 1. Extract route signatures from the generated source code
        source = output.get("backend_source", {})
        body = source.get("body", "")
        if body:
            # Pull route definitions: router.get('/path', ...) or app.get(...)
            import re as _re
            routes = _re.findall(
                r"(?:router|app)\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]",
                body,
                _re.IGNORECASE,
            )
            if routes:
                parts.append("Implemented routes:")
                for method, path in routes:
                    parts.append(f"  {method.upper()} {path}")

        # 2. Include the blueprint endpoint details (has request/response hints)
        blueprint = output.get("backend_blueprint", {})
        endpoints = blueprint.get("endpoints", [])
        if endpoints:
            parts.append("\nEndpoint contracts:")
            for ep in endpoints:
                line = f"  {ep.get('method', 'GET')} {ep.get('path', '/')}: {ep.get('name', '')}"
                parts.append(line)

        # 3. Database models if present
        db = blueprint.get("database", {})
        models = db.get("models", [])
        if models:
            parts.append("\nDatabase tables:")
            for model in models:
                fields = model.get("fields", {})
                field_list = ", ".join(f"{k}: {v}" for k, v in fields.items())
                parts.append(f"  {model.get('name', '?')} ({field_list})")

        # 4. Package dependencies (tells frontend what libraries are available)
        mcp_plan = output.get("backend_mcp_plan", {})
        for f in mcp_plan.get("files", []):
            if f.get("path") == "package.json":
                try:
                    pkg = json.loads(f["contents"])
                    deps = list(pkg.get("dependencies", {}).keys())
                    if deps:
                        parts.append(f"\nBackend dependencies: {', '.join(deps)}")
                except (json.JSONDecodeError, KeyError):
                    pass
                break

        return "\n".join(parts) if parts else ""

    def extract_frontend_contract(self) -> str:
        """Extract a compact summary of the frontend's actual output."""
        output = self.agent_output("frontend")
        if not output:
            return ""

        parts: list[str] = []

        blueprint = output.get("frontend_blueprint", {})
        routes = blueprint.get("routes", [])
        if routes:
            parts.append("Frontend routes:")
            for r in routes:
                parts.append(f"  {r.get('path', '/')}: {r.get('description', '')}")

        components = blueprint.get("components", [])
        if components:
            parts.append("\nComponents:")
            for c in components:
                parts.append(f"  {c.get('name', '?')}: {c.get('description', '')}")

        # Extract actual fetch calls from generated code
        source = output.get("frontend_source", {})
        body = source.get("body", "")
        if body:
            import re as _re
            fetches = _re.findall(r"fetch\(['\"`]([^'\"`]+)['\"`]", body)
            if fetches:
                seen = sorted(set(fetches))
                parts.append(f"\nAPI calls made: {', '.join(seen)}")

        return "\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # LLM selection
    # ------------------------------------------------------------------

    def get_llm(self, role: str | None = None) -> BaseLLMClient:
        """Return the LLM client for a given agent role.

        If no role-specific client is configured, fall back to the parent
        role (e.g. ``backend_api`` → ``backend``) then to the shared
        ``llm`` instance.
        """
        if role is None:
            return self.llm
        if role in self.llm_per_role:
            return self.llm_per_role[role]
        # Fall back to parent role for specialized sub-agents
        _PARENT = {
            "backend_api": "backend",
            "backend_db": "backend",
            "frontend_pages": "frontend",
            "frontend_ui": "frontend",
        }
        parent = _PARENT.get(role)
        if parent and parent in self.llm_per_role:
            return self.llm_per_role[parent]
        return self.llm

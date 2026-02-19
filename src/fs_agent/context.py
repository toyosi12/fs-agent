"""Shared runtime context passed between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import re

from .config import Settings
from .llm import BaseLLMClient
from .models.spec import ProjectSpec


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


@dataclass
class RunContext:
    """Mutable state that flows through the orchestration sequence."""

    spec: ProjectSpec | None
    user_request: str
    settings: Settings
    workspace_dir: Path
    artifact_dir: Path
    llm: BaseLLMClient
    transcripts: list[AgentReport] = field(default_factory=list)

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

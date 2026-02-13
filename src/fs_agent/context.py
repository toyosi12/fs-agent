"""Shared runtime context passed between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings
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

    spec: ProjectSpec
    settings: Settings
    workspace_dir: Path
    artifact_dir: Path
    transcripts: list[AgentReport] = field(default_factory=list)

    def record(self, report: AgentReport) -> None:
        self.transcripts.append(report)

    @property
    def artifacts(self) -> dict[str, Any]:
        combined: dict[str, Any] = {}
        for report in self.transcripts:
            combined.update(report.artifacts)
        return combined

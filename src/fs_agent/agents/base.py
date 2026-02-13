"""Common agent abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from ..context import RunContext
from ..logger import get_logger


class AgentRole(str, Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"
    INFRA = "infra"


@dataclass
class AgentArtifact:
    """Structured artifact metadata for downstream tooling."""

    name: str
    kind: Literal["plan", "code", "doc"]
    description: str
    body: Any


@dataclass
class AgentResult:
    """Standardized agent output."""

    role: AgentRole
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    attachments: list[AgentArtifact] = field(default_factory=list)
    status: Literal["success", "error"] = "success"
    diagnostics: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseAgent:
    """Base class for specialized agents."""

    role: AgentRole

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def render_header(self, context: RunContext) -> str:
        """Return a short textual header summarizing the assignment."""

        return (
            f"Project: {context.spec.metadata.name}\n"
            f"Owner: {context.spec.metadata.owner}\n"
            f"Stage: {self.role.value}\n"
        )

    def run(self, context: RunContext) -> AgentResult:  # pragma: no cover - interface only
        raise NotImplementedError

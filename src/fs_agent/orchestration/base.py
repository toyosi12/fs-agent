"""Base orchestration pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ..context import AgentReport, RunContext


class OrchestrationError(RuntimeError):
    """Raised when an orchestration pattern encounters an unrecoverable error.

    This replaces all silent fallbacks.  The benchmark runner catches this
    and records the run as ``status="failed"`` with full context.
    """

    def __init__(self, pattern: str, reason: str, *, context: dict | None = None) -> None:
        self.pattern = pattern
        self.reason = reason
        self.context = context or {}
        super().__init__(f"[{pattern}] {reason}")


class OrchestrationPattern(ABC):
    """Abstract orchestrator interface."""

    @abstractmethod
    def run(self, context: RunContext) -> Sequence[AgentReport]:
        raise NotImplementedError

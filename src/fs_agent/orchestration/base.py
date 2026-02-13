"""Base orchestration pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..context import AgentReport, RunContext


class OrchestrationPattern(ABC):
    """Abstract orchestrator interface."""

    @abstractmethod
    def run(self, context: RunContext) -> Sequence[AgentReport]:
        raise NotImplementedError

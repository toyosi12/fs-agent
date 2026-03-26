"""Agent registry for dependency injection."""

from __future__ import annotations

from typing import Dict, Type

from ..agents import (
    ArchitectAgent,
    BackendAgent,
    BackendApiAgent,
    BackendDbAgent,
    FixerAgent,
    FrontendAgent,
    FrontendPagesAgent,
    FrontendUiAgent,
    InfraAgent,
)
from ..agents.base import AgentRole, BaseAgent


class AgentRegistry:
    """Simple in-memory mapping of roles to agent classes."""

    def __init__(self) -> None:
        self._registry: Dict[AgentRole, Type[BaseAgent]] = {}

    def register(self, role: AgentRole, agent_cls: Type[BaseAgent]) -> None:
        self._registry[role] = agent_cls

    def build(self, role: AgentRole) -> BaseAgent:
        agent_cls = self._registry.get(role)
        if agent_cls is None:
            raise KeyError(f"No agent registered for role {role}")
        return agent_cls()


def register_default_agents(registry: AgentRegistry) -> None:
    """Register all agents including specialized sub-role agents."""
    registry.register(AgentRole.ARCHITECT, ArchitectAgent)
    registry.register(AgentRole.BACKEND, BackendAgent)
    registry.register(AgentRole.FRONTEND, FrontendAgent)
    registry.register(AgentRole.INFRA, InfraAgent)
    # Specialized sub-agents for hierarchical decomposition
    registry.register(AgentRole.BACKEND_DB, BackendDbAgent)
    registry.register(AgentRole.BACKEND_API, BackendApiAgent)
    registry.register(AgentRole.FRONTEND_PAGES, FrontendPagesAgent)
    registry.register(AgentRole.FRONTEND_UI, FrontendUiAgent)
    # Post-generation fix agent
    registry.register(AgentRole.FIXER, FixerAgent)

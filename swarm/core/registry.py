"""Swarm Agent Registry — LOCKED SYSTEM CORE
==============================================
Central registry for agent discovery and dispatch.
Do NOT modify without explicit authorization.
"""

from __future__ import annotations


class AgentNotFoundError(Exception):
    """Raised when a requested agent is not registered."""
    pass


class DuplicateAgentError(Exception):
    """Raised when attempting to register an agent name that already exists."""
    pass


class AgentRegistry:
    """Thread-safe registry for Swarm agents."""

    def __init__(self):
        self._agents: dict[str, type] = {}

    def register(self, name: str, agent_cls: type) -> None:
        """Register an agent class. Raises DuplicateAgentError if name exists."""
        if name in self._agents:
            raise DuplicateAgentError(f"Agent '{name}' is already registered")
        if not hasattr(agent_cls, "run"):
            raise TypeError(f"Agent class {agent_cls.__name__} must implement a 'run' method")
        self._agents[name] = agent_cls

    def get(self, name: str) -> type:
        """Get an agent class by name. Raises AgentNotFoundError if not found."""
        if name not in self._agents:
            raise AgentNotFoundError(f"Agent '{name}' not found. Available: {list(self._agents.keys())}")
        return self._agents[name]

    def list_agents(self) -> list[str]:
        """Return sorted list of registered agent names."""
        return sorted(self._agents.keys())

    def has(self, name: str) -> bool:
        """Check if an agent is registered."""
        return name in self._agents

    def count(self) -> int:
        """Return number of registered agents."""
        return len(self._agents)

    def unregister(self, name: str) -> None:
        """Remove an agent from the registry."""
        if name not in self._agents:
            raise AgentNotFoundError(f"Agent '{name}' not found")
        del self._agents[name]

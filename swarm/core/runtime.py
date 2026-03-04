"""Swarm Runtime — LOCKED SYSTEM CORE
======================================
Central runtime for agent execution, pipeline orchestration, and registry dispatch.
Do NOT modify without explicit authorization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import SwarmConfig
from .registry import AgentRegistry


@dataclass
class RunContext:
    """Execution context passed to every agent and pipeline."""
    run_id: str
    config: SwarmConfig
    started_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at


class SwarmRuntime:
    """Core runtime that manages agent lifecycle and pipeline execution."""

    def __init__(self, config: SwarmConfig | None = None):
        self.config = config or SwarmConfig()
        self.registry = AgentRegistry()
        self._hooks: list[Callable] = []

    def register_agent(self, name: str, agent_cls: type) -> None:
        """Register an agent class with the runtime."""
        self.registry.register(name, agent_cls)

    def get_agent(self, name: str) -> type:
        """Retrieve a registered agent class by name."""
        return self.registry.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return self.registry.list_agents()

    def create_context(self, run_id: str, **metadata) -> RunContext:
        """Create an execution context for a run."""
        return RunContext(run_id=run_id, config=self.config, metadata=metadata)

    def add_hook(self, hook: Callable) -> None:
        """Add a lifecycle hook (called on run start/end)."""
        self._hooks.append(hook)

    def execute(self, agent_name: str, run_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute a named agent with inputs and return outputs."""
        agent_cls = self.get_agent(agent_name)
        ctx = self.create_context(run_id, agent=agent_name)

        for hook in self._hooks:
            hook("start", ctx)

        agent = agent_cls(ctx)
        result = agent.run(inputs)

        for hook in self._hooks:
            hook("end", ctx)

        return result

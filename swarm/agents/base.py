"""Swarm Agent Base Class
========================
All Swarm agents inherit from BaseAgent.
Provides lifecycle hooks and contract validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.runtime import RunContext


class BaseAgent(ABC):
    """Abstract base for all Swarm agents."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    @abstractmethod
    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's primary task. Must be implemented by subclasses."""
        ...

    def validate_inputs(self, inputs: dict[str, Any], required: list[str]) -> None:
        """Validate that all required keys are present in inputs."""
        missing = [k for k in required if k not in inputs]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required inputs: {missing}")

    @property
    def name(self) -> str:
        return self.__class__.__name__

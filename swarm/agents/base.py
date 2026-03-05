"""Swarm Agent Base Class
========================
All Swarm agents inherit from BaseAgent.
Provides run contract: input validation, dispatch, output validation.

The runtime calls agent.run(inputs). BaseAgent.run() enforces the contract
then delegates to _execute(), which subclasses implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.runtime import RunContext


class BaseAgent(ABC):
    """Abstract base for all Swarm agents."""

    REQUIRED_INPUTS: list[str] = []

    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run contract: validate inputs, call _execute(), validate output.

        This is called by SwarmRuntime.execute(). Do not override.
        Subclasses implement _execute() instead.
        """
        self.validate_inputs(inputs, self.REQUIRED_INPUTS)
        result = self._execute(inputs)
        if not isinstance(result, dict):
            raise TypeError(f"{self.name}._execute() must return dict, got {type(result).__name__}")
        return result

    @abstractmethod
    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Agent implementation. Override this in subclasses."""
        ...

    def validate_inputs(self, inputs: dict[str, Any], required: list[str]) -> None:
        """Validate that all required keys are present in inputs."""
        missing = [k for k in required if k not in inputs]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required inputs: {missing}")

    @property
    def name(self) -> str:
        return self.__class__.__name__

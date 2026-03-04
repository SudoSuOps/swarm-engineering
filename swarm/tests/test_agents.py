"""Tests for Swarm agent registration and base class."""

import pytest

from swarm.core.registry import AgentRegistry, AgentNotFoundError, DuplicateAgentError
from swarm.core.runtime import RunContext, SwarmRuntime
from swarm.core.config import SwarmConfig
from swarm.agents.base import BaseAgent


# --- Fixtures ---

class MockAgent(BaseAgent):
    def run(self, inputs):
        return {"echo": inputs.get("message", "")}


class BadAgent:
    """Agent without run method — should fail registration."""
    pass


@pytest.fixture
def runtime():
    return SwarmRuntime()


@pytest.fixture
def registry():
    return AgentRegistry()


# --- Registry Tests ---

def test_register_and_get(registry):
    registry.register("mock", MockAgent)
    assert registry.get("mock") is MockAgent


def test_register_duplicate_raises(registry):
    registry.register("mock", MockAgent)
    with pytest.raises(DuplicateAgentError):
        registry.register("mock", MockAgent)


def test_get_missing_raises(registry):
    with pytest.raises(AgentNotFoundError):
        registry.get("nonexistent")


class AnotherAgent(BaseAgent):
    def run(self, inputs):
        return {}


def test_list_agents(registry):
    registry.register("beta", MockAgent)
    registry.register("alpha", AnotherAgent)
    assert registry.list_agents() == ["alpha", "beta"]


def test_has(registry):
    assert not registry.has("mock")
    registry.register("mock", MockAgent)
    assert registry.has("mock")


def test_count(registry):
    assert registry.count() == 0
    registry.register("mock", MockAgent)
    assert registry.count() == 1


def test_unregister(registry):
    registry.register("mock", MockAgent)
    registry.unregister("mock")
    assert not registry.has("mock")


def test_unregister_missing_raises(registry):
    with pytest.raises(AgentNotFoundError):
        registry.unregister("nonexistent")


def test_register_no_run_method_raises(registry):
    with pytest.raises(TypeError):
        registry.register("bad", BadAgent)


# --- BaseAgent Tests ---

def test_base_agent_run(runtime):
    ctx = runtime.create_context("test-001")
    agent = MockAgent(ctx)
    result = agent.run({"message": "hello"})
    assert result == {"echo": "hello"}


def test_base_agent_validate_inputs(runtime):
    ctx = runtime.create_context("test-002")
    agent = MockAgent(ctx)
    agent.validate_inputs({"a": 1, "b": 2}, ["a", "b"])

    with pytest.raises(ValueError, match="missing required inputs"):
        agent.validate_inputs({"a": 1}, ["a", "b"])


def test_base_agent_name(runtime):
    ctx = runtime.create_context("test-003")
    agent = MockAgent(ctx)
    assert agent.name == "MockAgent"


# --- Runtime Execution Tests ---

def test_runtime_execute(runtime):
    runtime.register_agent("mock", MockAgent)
    result = runtime.execute("mock", "run-001", {"message": "test"})
    assert result == {"echo": "test"}


def test_runtime_hooks(runtime):
    events = []
    runtime.add_hook(lambda event, ctx: events.append(event))
    runtime.register_agent("mock", MockAgent)
    runtime.execute("mock", "run-002", {"message": "hook-test"})
    assert events == ["start", "end"]

"""Integration tests — full runtime → registry → agent dispatch chain."""

import pytest

from swarm.core.runtime import SwarmRuntime
from swarm.agents.base import BaseAgent
from swarm.agents.swarmjudge import SwarmJudgeAgent


# --- BaseAgent.execute() contract ---

class StubAgent(BaseAgent):
    REQUIRED_INPUTS = ["x"]

    def _execute(self, inputs):
        return {"result": inputs["x"] * 2}


class BadReturnAgent(BaseAgent):
    REQUIRED_INPUTS = []

    def _execute(self, inputs):
        return "not a dict"  # type: ignore[return-value]


@pytest.fixture
def runtime():
    rt = SwarmRuntime()
    rt.register_agent("stub", StubAgent)
    rt.register_agent("swarmjudge", SwarmJudgeAgent)
    return rt


# --- execute() contract tests ---

def test_execute_validates_inputs(runtime):
    with pytest.raises(ValueError, match="missing required inputs"):
        runtime.execute("stub", "run-001", {})


def test_execute_returns_dict(runtime):
    result = runtime.execute("stub", "run-002", {"x": 5})
    assert result == {"result": 10}


def test_execute_bad_return_raises():
    rt = SwarmRuntime()
    rt.register_agent("bad", BadReturnAgent)
    with pytest.raises(TypeError, match="must return dict"):
        rt.execute("bad", "run-003", {})


# --- SwarmJudge contract (Block-1 stub) ---

def test_judge_requires_inputs(runtime):
    with pytest.raises(ValueError, match="missing required inputs"):
        runtime.execute("swarmjudge", "judge-001", {})


def test_judge_requires_domain(runtime):
    with pytest.raises(ValueError, match="missing required inputs"):
        runtime.execute("swarmjudge", "judge-002", {
            "instruction": "What is a lease abstract?",
            "response": "A lease abstract summarizes key terms." + "x" * 100,
        })


def test_judge_not_implemented(runtime):
    with pytest.raises(NotImplementedError, match="Block-1 not yet implemented"):
        runtime.execute("swarmjudge", "judge-003", {
            "instruction": "Explain cap rate.",
            "response": "Cap rate is NOI divided by price." + "x" * 100,
            "domain": "cre",
        })


# --- Hook integration ---

def test_hooks_fire_on_dispatch(runtime):
    events = []
    runtime.add_hook(lambda event, ctx: events.append((event, ctx.metadata.get("agent"))))
    runtime.execute("stub", "hook-001", {"x": 1})
    assert events == [("start", "stub"), ("end", "stub")]


# --- Full round-trip: register from AGENTS dict ---

def test_agents_dict_registration():
    from swarm.agents import AGENTS
    rt = SwarmRuntime()
    for name, cls in AGENTS.items():
        rt.register_agent(name, cls)
    assert set(rt.list_agents()) == set(AGENTS.keys())
    assert rt.registry.has("swarmjudge")

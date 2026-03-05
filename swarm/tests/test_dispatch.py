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


# --- SwarmJudgeAgent dispatch ---

def test_judge_pass_verdict(runtime):
    result = runtime.execute("swarmjudge", "judge-001", {
        "instruction": "Explain the cap rate formula for industrial assets.",
        "response": "The capitalization rate is calculated by dividing net operating income "
                     "by the current market value of the property. " + "x" * 150,
        "domain": "cre",
        "score": 0.85,
    })
    assert result["verdict"] == "PASS"
    assert result["quality_score"] >= 0.7
    assert result["schema_errors"] == []
    assert result["domain"] == "cre"
    assert "completeness" in result["breakdown"]


def test_judge_fail_short_response(runtime):
    result = runtime.execute("swarmjudge", "judge-002", {
        "instruction": "What is NOI?",
        "response": "Net operating income.",
        "domain": "cre",
        "score": 0.5,
    })
    assert result["verdict"] == "FAIL"
    assert len(result["schema_errors"]) > 0


def test_judge_fail_missing_domain(runtime):
    with pytest.raises(ValueError, match="missing required inputs"):
        runtime.execute("swarmjudge", "judge-003", {
            "instruction": "What is a lease abstract?",
            "response": "A lease abstract summarizes key terms." + "x" * 100,
        })


def test_judge_fail_empty_domain(runtime):
    result = runtime.execute("swarmjudge", "judge-004", {
        "instruction": "Explain debt service coverage ratio in detail.",
        "response": "DSCR measures the cash flow available to service debt. " + "x" * 150,
        "domain": "",
        "score": 0.8,
    })
    assert result["verdict"] == "FAIL"


# --- Hook integration ---

def test_hooks_fire_on_judge_dispatch(runtime):
    events = []
    runtime.add_hook(lambda event, ctx: events.append((event, ctx.metadata.get("agent"))))
    runtime.execute("swarmjudge", "judge-005", {
        "instruction": "What is a triple net lease and how does it work?",
        "response": "A triple net lease requires the tenant to pay property taxes, "
                     "insurance, and maintenance in addition to rent. " + "x" * 100,
        "domain": "cre",
        "score": 0.9,
    })
    assert events == [("start", "swarmjudge"), ("end", "swarmjudge")]


# --- Full round-trip: register from AGENTS dict ---

def test_agents_dict_dispatch():
    from swarm.agents import AGENTS
    rt = SwarmRuntime()
    for name, cls in AGENTS.items():
        rt.register_agent(name, cls)
    assert set(rt.list_agents()) == set(AGENTS.keys())
    result = rt.execute("swarmjudge", "roundtrip-001", {
        "instruction": "Analyze the investment thesis for cold storage assets.",
        "response": "Cold storage facilities benefit from growing demand for "
                     "perishable goods logistics and last-mile delivery. " + "x" * 100,
        "domain": "cre",
        "score": 0.88,
    })
    assert result["verdict"] == "PASS"

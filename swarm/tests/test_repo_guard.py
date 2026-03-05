"""Tests for supervisor/repo_guard.py — protected path enforcement."""

import pytest

from swarm.supervisor.repo_guard import validate_patch, PROTECTED_PATHS


# --- Constants ---

def test_protected_paths_defined():
    assert PROTECTED_PATHS == ["core/", "contracts/", "tests/"]


# --- Allowed Paths ---

def test_allowed_agents_path():
    validate_patch(["agents/base.py"])


def test_allowed_pipelines_path():
    validate_patch(["pipelines/dataset_factory.py"])


def test_allowed_scripts_path():
    validate_patch(["scripts/run_tests.sh"])


def test_multiple_allowed_paths():
    validate_patch([
        "agents/base.py",
        "pipelines/eval_runner.py",
        "scripts/run_eval.sh",
    ])


def test_empty_patch():
    validate_patch([])


# --- Protected Paths ---

def test_core_rejected():
    with pytest.raises(Exception, match="Protected path"):
        validate_patch(["core/runtime.py"])


def test_contracts_rejected():
    with pytest.raises(Exception, match="Protected path"):
        validate_patch(["contracts/pair.schema.json"])


def test_tests_rejected():
    with pytest.raises(Exception, match="Protected path"):
        validate_patch(["tests/test_agents.py"])


def test_nested_core_rejected():
    with pytest.raises(Exception, match="Protected path"):
        validate_patch(["core/deep/nested/file.py"])


# --- Mixed Paths ---

def test_mixed_allowed_and_protected_raises():
    with pytest.raises(Exception, match="Protected path"):
        validate_patch(["agents/base.py", "core/config.py"])

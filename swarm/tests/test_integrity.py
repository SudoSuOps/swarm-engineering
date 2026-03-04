"""Integrity tests — verify schema contracts, locked files, and structure."""

import json
import pytest
from pathlib import Path

SWARM_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = SWARM_ROOT / "contracts"


# --- Schema Validation ---

def test_pair_schema_valid_json():
    path = CONTRACTS_DIR / "pair.schema.json"
    assert path.exists(), "pair.schema.json not found"
    schema = json.loads(path.read_text())
    assert schema["type"] == "object"
    assert "question" in schema["required"]
    assert "answer" in schema["required"]


def test_eval_schema_valid_json():
    path = CONTRACTS_DIR / "eval.schema.json"
    assert path.exists(), "eval.schema.json not found"
    schema = json.loads(path.read_text())
    assert schema["type"] == "object"
    assert "run_id" in schema["required"]
    assert "score" in schema["required"]


def test_dataset_schema_valid_json():
    path = CONTRACTS_DIR / "dataset.schema.json"
    assert path.exists(), "dataset.schema.json not found"
    schema = json.loads(path.read_text())
    assert schema["type"] == "object"
    assert "splits" in schema["required"]


# --- Structure Integrity ---

LOCKED_DIRS = ["core", "contracts", "tests"]
EDITABLE_DIRS = ["agents", "pipelines", "scripts"]


def test_locked_dirs_exist():
    for d in LOCKED_DIRS:
        assert (SWARM_ROOT / d).is_dir(), f"Locked directory {d}/ missing"


def test_editable_dirs_exist():
    for d in EDITABLE_DIRS:
        assert (SWARM_ROOT / d).is_dir(), f"Editable directory {d}/ missing"


def test_core_modules_present():
    core = SWARM_ROOT / "core"
    for module in ["runtime.py", "config.py", "registry.py"]:
        assert (core / module).exists(), f"core/{module} missing"


def test_agent_modules_present():
    agents = SWARM_ROOT / "agents"
    for agent in ["swarmjudge", "swarmcode", "swarmcre", "swarmmed"]:
        init = agents / agent / "__init__.py"
        assert init.exists(), f"agents/{agent}/__init__.py missing"


def test_pipeline_modules_present():
    pipelines = SWARM_ROOT / "pipelines"
    for module in ["dataset_factory.py", "training_pipeline.py", "eval_runner.py"]:
        assert (pipelines / module).exists(), f"pipelines/{module} missing"

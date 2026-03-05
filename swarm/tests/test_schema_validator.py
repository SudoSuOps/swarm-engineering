"""Tests for pipelines/schema_validator.py"""

import pytest

from swarm.pipelines.schema_validator import load_schema, validate_pair, is_valid_pair


# --- Schema Loading ---

def test_load_pair_schema():
    schema = load_schema("pair.schema.json")
    assert schema["type"] == "object"
    assert "instruction" in schema["required"]


def test_load_missing_schema_raises():
    with pytest.raises(FileNotFoundError):
        load_schema("nonexistent.json")


# --- Valid Pairs ---

def test_valid_pair_passes():
    pair = {
        "instruction": "Explain the concept of AI signal extraction.",
        "response": "Signal extraction identifies actionable AI trends from " + "x" * 50,
        "score": 0.85,
        "domain": "signal",
    }
    errors = validate_pair(pair)
    assert errors == []
    assert is_valid_pair(pair)


def test_valid_pair_with_metadata():
    pair = {
        "instruction": "What is signal extraction in AI?",
        "response": "Signal extraction identifies key AI developments from " + "x" * 50,
        "score": 0.9,
        "domain": "signal",
        "metadata": {"source": "cook_v2", "specialty": "ai_trends"},
    }
    assert is_valid_pair(pair)


# --- Missing Required Fields ---

def test_missing_instruction():
    pair = {"response": "x" * 60, "score": 0.8, "domain": "signal"}
    errors = validate_pair(pair)
    assert any("instruction" in e for e in errors)


def test_missing_response():
    pair = {"instruction": "x" * 20, "score": 0.8, "domain": "signal"}
    errors = validate_pair(pair)
    assert any("response" in e for e in errors)


def test_missing_score():
    pair = {"instruction": "x" * 20, "response": "x" * 60, "domain": "signal"}
    errors = validate_pair(pair)
    assert any("score" in e for e in errors)


def test_missing_domain():
    pair = {"instruction": "x" * 20, "response": "x" * 60, "score": 0.8}
    errors = validate_pair(pair)
    assert any("domain" in e for e in errors)


# --- Type Errors ---

def test_score_wrong_type():
    pair = {"instruction": "x" * 20, "response": "x" * 60, "score": "high", "domain": "signal"}
    errors = validate_pair(pair)
    assert any("expected number" in e for e in errors)


def test_domain_wrong_type():
    pair = {"instruction": "x" * 20, "response": "x" * 60, "score": 0.8, "domain": 123}
    errors = validate_pair(pair)
    assert any("expected string" in e for e in errors)


# --- Constraint Violations ---

def test_instruction_too_short():
    pair = {"instruction": "Hi", "response": "x" * 60, "score": 0.8, "domain": "signal"}
    errors = validate_pair(pair)
    assert any("minLength" in e for e in errors)


def test_response_too_short():
    pair = {"instruction": "x" * 20, "response": "short", "score": 0.8, "domain": "signal"}
    errors = validate_pair(pair)
    assert any("minLength" in e for e in errors)


def test_score_below_minimum():
    pair = {"instruction": "x" * 20, "response": "x" * 60, "score": -0.5, "domain": "signal"}
    errors = validate_pair(pair)
    assert any("minimum" in e for e in errors)


def test_score_above_maximum():
    pair = {"instruction": "x" * 20, "response": "x" * 60, "score": 1.5, "domain": "signal"}
    errors = validate_pair(pair)
    assert any("maximum" in e for e in errors)


# --- Additional Properties ---

def test_unexpected_field_rejected():
    pair = {
        "instruction": "x" * 20,
        "response": "x" * 60,
        "score": 0.8,
        "domain": "signal",
        "bogus_field": "should fail",
    }
    errors = validate_pair(pair)
    assert any("Unexpected field" in e for e in errors)

"""Tests for Swarm pipelines — dataset factory, training, eval."""

import json
import pytest
from pathlib import Path

from swarm.core.runtime import SwarmRuntime
from swarm.pipelines.dataset_factory import DatasetFactory
from swarm.pipelines.training_pipeline import TrainingPipeline, TrainingRun
from swarm.pipelines.eval_runner import EvalRunner


@pytest.fixture
def ctx():
    runtime = SwarmRuntime()
    return runtime.create_context("test-pipe-001")


@pytest.fixture
def sample_pairs_file(tmp_path):
    pairs = [
        {"question": f"Question {i} " * 10, "answer": f"Answer {i} " * 30, "specialty": "test", "source": "synthetic"}
        for i in range(100)
    ]
    path = tmp_path / "pairs.jsonl"
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    return path


# --- DatasetFactory Tests ---

def test_load_pairs(ctx, sample_pairs_file):
    factory = DatasetFactory(ctx)
    pairs = factory.load_pairs(sample_pairs_file)
    assert len(pairs) == 100


def test_validate_pair(ctx):
    factory = DatasetFactory(ctx)
    good = {"question": "x" * 100, "answer": "y" * 200}
    assert factory.validate_pair(good)

    short = {"question": "hi", "answer": "ok"}
    assert not factory.validate_pair(short)

    low_quality = {"question": "x" * 100, "answer": "y" * 200, "quality_score": 0.1}
    assert not factory.validate_pair(low_quality)


def test_split(ctx):
    factory = DatasetFactory(ctx)
    pairs = [{"i": i} for i in range(100)]
    train, eval_set = factory.split(pairs, eval_ratio=0.1)
    assert len(train) == 90
    assert len(eval_set) == 10


def test_build(ctx, sample_pairs_file, tmp_path):
    factory = DatasetFactory(ctx)
    out = tmp_path / "output"
    manifest = factory.build([sample_pairs_file], out, "test_dataset")
    assert manifest["pair_count"] == 100
    assert manifest["splits"]["train"]["count"] == 95
    assert manifest["splits"]["eval"]["count"] == 5
    assert len(manifest["splits"]["train"]["sha256"]) == 64


# --- TrainingPipeline Tests ---

def test_validate_config(ctx, sample_pairs_file, tmp_path):
    pipeline = TrainingPipeline(ctx)
    run = TrainingRun(
        base_model="Qwen/Qwen3.5-9B",
        dataset_path=sample_pairs_file,
        output_dir=tmp_path / "train_out",
    )
    issues = pipeline.validate_config(run)
    assert issues == []


def test_validate_config_bad_dataset(ctx, tmp_path):
    pipeline = TrainingPipeline(ctx)
    run = TrainingRun(
        base_model="Qwen/Qwen3.5-9B",
        dataset_path=tmp_path / "nonexistent.jsonl",
        output_dir=tmp_path / "train_out",
    )
    issues = pipeline.validate_config(run)
    assert any("not found" in i for i in issues)


def test_validate_config_bad_lr(ctx, sample_pairs_file, tmp_path):
    pipeline = TrainingPipeline(ctx)
    run = TrainingRun(
        base_model="Qwen/Qwen3.5-9B",
        dataset_path=sample_pairs_file,
        output_dir=tmp_path / "train_out",
        learning_rate=0.5,
    )
    issues = pipeline.validate_config(run)
    assert any("learning_rate" in i for i in issues)


def test_launch_ready(ctx, sample_pairs_file, tmp_path):
    pipeline = TrainingPipeline(ctx)
    run = TrainingRun(
        base_model="Qwen/Qwen3.5-9B",
        dataset_path=sample_pairs_file,
        output_dir=tmp_path / "train_out",
    )
    result = pipeline.launch(run)
    assert result["status"] == "ready"
    assert result["config"]["effective_batch"] == 32


# --- EvalRunner Tests ---

def test_eval_runner(ctx, sample_pairs_file, tmp_path):
    runner = EvalRunner(ctx)
    results = runner.run_eval(sample_pairs_file, "test-model")
    assert len(results) > 0
    assert all("metric" in r for r in results)

    out = tmp_path / "eval_results.jsonl"
    runner.write_results(results, out)
    assert out.exists()

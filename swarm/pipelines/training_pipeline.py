"""Training Pipeline
====================
Orchestrates model training runs with config validation and artifact tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.runtime import RunContext


@dataclass
class TrainingRun:
    """Describes a single training run configuration."""
    base_model: str
    dataset_path: Path
    output_dir: Path
    lora_r: int = 64
    lora_alpha: int = 32
    batch_size: int = 4
    grad_accum: int = 8
    learning_rate: float = 5e-5
    epochs: int = 1
    max_seq_length: int = 4096
    packing: bool = True


class TrainingPipeline:
    """Manages training run lifecycle: validate, launch, track artifacts."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def validate_config(self, run: TrainingRun) -> list[str]:
        """Validate training config. Returns list of issues (empty = valid)."""
        issues = []
        if not run.dataset_path.exists():
            issues.append(f"Dataset not found: {run.dataset_path}")
        if run.lora_r < 8 or run.lora_r > 256:
            issues.append(f"lora_r={run.lora_r} out of range [8, 256]")
        if run.learning_rate <= 0 or run.learning_rate > 1e-2:
            issues.append(f"learning_rate={run.learning_rate} out of range (0, 1e-2]")
        if run.batch_size < 1:
            issues.append(f"batch_size must be >= 1")
        return issues

    def build_command(self, run: TrainingRun) -> list[str]:
        """Build the training launch command arguments."""
        return [
            "python3", "-u",
            str(run.output_dir / "train.py"),
            "--base-model", run.base_model,
            "--dataset", str(run.dataset_path),
            "--output-dir", str(run.output_dir),
            "--lora-r", str(run.lora_r),
            "--lora-alpha", str(run.lora_alpha),
            "--batch-size", str(run.batch_size),
            "--grad-accum", str(run.grad_accum),
            "--lr", str(run.learning_rate),
            "--epochs", str(run.epochs),
            "--max-seq-length", str(run.max_seq_length),
            *(["--packing"] if run.packing else []),
        ]

    def launch(self, run: TrainingRun) -> dict[str, Any]:
        """Validate and return launch config. Actual execution is external."""
        issues = self.validate_config(run)
        if issues:
            return {"status": "invalid", "issues": issues}

        run.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(run)

        return {
            "status": "ready",
            "run_id": self.ctx.run_id,
            "command": cmd,
            "config": {
                "base_model": run.base_model,
                "lora_r": run.lora_r,
                "learning_rate": run.learning_rate,
                "batch_size": run.batch_size,
                "grad_accum": run.grad_accum,
                "effective_batch": run.batch_size * run.grad_accum,
            },
        }

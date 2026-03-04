"""Swarm Config — LOCKED SYSTEM CORE
=====================================
Central configuration for the Swarm system.
Do NOT modify without explicit authorization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SwarmConfig:
    """Immutable system configuration."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("SWARM_DATA_DIR", "/data2")))
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("SWARM_OUTPUT_DIR", "/tmp/swarm_output")))

    # Model defaults
    default_base_model: str = "Qwen/Qwen3.5-9B"
    max_seq_length: int = 4096
    default_lora_r: int = 64
    default_lora_alpha: int = 32

    # Training defaults
    default_batch_size: int = 4
    default_grad_accum: int = 8
    default_learning_rate: float = 5e-5
    default_epochs: int = 1

    # Quality thresholds
    min_pair_length: int = 200
    max_pair_length: int = 32000
    quality_score_threshold: float = 0.7

    # Contract paths
    @property
    def contracts_dir(self) -> Path:
        return self.project_root / "contracts"

    @property
    def pair_schema_path(self) -> Path:
        return self.contracts_dir / "pair.schema.json"

    @property
    def eval_schema_path(self) -> Path:
        return self.contracts_dir / "eval.schema.json"

    @property
    def dataset_schema_path(self) -> Path:
        return self.contracts_dir / "dataset.schema.json"

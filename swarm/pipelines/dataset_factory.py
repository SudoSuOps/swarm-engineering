"""Dataset Factory Pipeline
===========================
Assembles, validates, and splits training datasets from raw pairs.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from ..core.runtime import RunContext
from .schema_validator import is_valid_pair


class DatasetFactory:
    """Builds validated training datasets from raw pair files."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.config = ctx.config

    def load_pairs(self, path: Path) -> list[dict[str, Any]]:
        """Load JSONL pairs from a file."""
        pairs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs

    def validate_pair(self, pair: dict[str, Any]) -> bool:
        """Check a pair meets quality thresholds AND schema contract."""
        if not is_valid_pair(pair):
            return False
        instruction = pair.get("instruction", "")
        response = pair.get("response", "")
        length = len(instruction) + len(response)
        if length < self.config.min_pair_length:
            return False
        if length > self.config.max_pair_length:
            return False
        score = pair.get("score", 1.0)
        if score < self.config.quality_score_threshold:
            return False
        return True

    def split(
        self,
        pairs: list[dict[str, Any]],
        eval_ratio: float = 0.05,
        seed: int = 42,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split pairs into train and eval sets."""
        rng = random.Random(seed)
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - eval_ratio)))
        return shuffled[:split_idx], shuffled[split_idx:]

    def write_jsonl(self, pairs: list[dict[str, Any]], path: Path) -> str:
        """Write pairs to JSONL and return SHA256 hash."""
        path.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with open(path, "w") as f:
            for pair in pairs:
                line = json.dumps(pair, ensure_ascii=False) + "\n"
                f.write(line)
                h.update(line.encode())
        return h.hexdigest()

    def build(
        self,
        source_paths: list[Path],
        output_dir: Path,
        name: str,
        eval_ratio: float = 0.05,
    ) -> dict[str, Any]:
        """Full pipeline: load, validate, split, write, return manifest."""
        all_pairs = []
        for p in source_paths:
            all_pairs.extend(self.load_pairs(p))

        valid = [p for p in all_pairs if self.validate_pair(p)]
        train, eval_set = self.split(valid, eval_ratio)

        train_path = output_dir / f"{name}_train.jsonl"
        eval_path = output_dir / f"{name}_eval.jsonl"

        train_hash = self.write_jsonl(train, train_path)
        eval_hash = self.write_jsonl(eval_set, eval_path)

        return {
            "name": name,
            "pair_count": len(valid),
            "splits": {
                "train": {"path": str(train_path), "count": len(train), "sha256": train_hash},
                "eval": {"path": str(eval_path), "count": len(eval_set), "sha256": eval_hash},
            },
            "rejected": len(all_pairs) - len(valid),
        }

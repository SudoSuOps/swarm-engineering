"""Evaluation Runner Pipeline
==============================
Runs model evaluation against test sets and produces structured results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.runtime import RunContext


class EvalRunner:
    """Executes evaluation suites and produces schema-compliant results."""

    def __init__(self, ctx: RunContext):
        self.ctx = ctx

    def load_eval_set(self, path: Path) -> list[dict[str, Any]]:
        """Load evaluation pairs from JSONL."""
        pairs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs

    def score_pair(self, prediction: str, reference: str) -> dict[str, float]:
        """Score a single prediction against reference. Override for custom metrics."""
        # Basic length-ratio and exact-match scoring
        len_ratio = min(len(prediction), len(reference)) / max(len(prediction), len(reference), 1)
        exact = 1.0 if prediction.strip() == reference.strip() else 0.0
        return {"length_ratio": len_ratio, "exact_match": exact}

    def run_eval(
        self,
        eval_path: Path,
        model_name: str,
        predict_fn: Any = None,
    ) -> list[dict[str, Any]]:
        """Run evaluation and return list of result records."""
        pairs = self.load_eval_set(eval_path)
        results = []
        now = datetime.now(timezone.utc).isoformat()

        for pair in pairs:
            if predict_fn:
                prediction = predict_fn(pair["question"])
            else:
                prediction = ""

            scores = self.score_pair(prediction, pair.get("answer", ""))

            for metric, score in scores.items():
                results.append({
                    "run_id": self.ctx.run_id,
                    "model": model_name,
                    "metric": metric,
                    "score": score,
                    "samples": 1,
                    "timestamp": now,
                })

        return results

    def write_results(self, results: list[dict[str, Any]], path: Path) -> None:
        """Write evaluation results to JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

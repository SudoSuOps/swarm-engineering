"""SwarmJudge Agent
==================
Quality gate agent — scores training pairs as PASS/FAIL.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseAgent
from ...pipelines.schema_validator import validate_pair as schema_validate


class SwarmJudgeAgent(BaseAgent):
    """Judge agent that evaluates training pair quality.

    Accepts a training pair, validates it against the pair schema,
    and returns a structured verdict with score breakdown.
    """

    REQUIRED_INPUTS = ["instruction", "response", "domain"]

    SCORE_THRESHOLD = 0.7

    def _execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        instruction = inputs["instruction"]
        response = inputs["response"]
        domain = inputs["domain"]
        score = inputs.get("score", 0.0)

        # Schema validation
        pair: dict[str, Any] = {
            "instruction": instruction,
            "response": response,
            "score": score,
            "domain": domain,
        }
        if "metadata" in inputs:
            pair["metadata"] = inputs["metadata"]

        schema_errors = schema_validate(pair)

        # Quality scoring
        completeness = min(1.0, len(response) / 200)
        structure = 1.0 if len(instruction) >= 20 and len(response) >= 100 else 0.5
        relevance = 1.0 if domain.strip() else 0.0

        quality_score = (completeness * 0.4) + (structure * 0.3) + (relevance * 0.3)

        verdict = "PASS" if quality_score >= self.SCORE_THRESHOLD and not schema_errors else "FAIL"

        return {
            "verdict": verdict,
            "quality_score": round(quality_score, 4),
            "schema_errors": schema_errors,
            "breakdown": {
                "completeness": round(completeness, 4),
                "structure": round(structure, 4),
                "relevance": round(relevance, 4),
            },
            "domain": domain,
        }

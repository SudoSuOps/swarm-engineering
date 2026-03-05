"""CRE Evaluation Harness
=========================
Domain-specific evaluation for Commercial Real Estate models.

Metrics:
  1. cap_rate_accuracy   — cap rate within ±50bps of reference
  2. noi_accuracy        — NOI within ±5% of reference
  3. schema_compliance   — response has required structural sections
  4. reasoning_score     — logical chain: inputs → formula → result
  5. hallucination_rate  — numbers in output not traceable to input

Usage:
    harness = CREEvalHarness()
    report = harness.evaluate(prediction, reference, instruction)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cre_numerics import (
    extract_all_metrics,
    extract_all_numbers,
    compare_metric,
)


# ---------------------------------------------------------------------------
# Reasoning markers — evidence of structured CRE reasoning
# ---------------------------------------------------------------------------

_REASONING_MARKERS = [
    re.compile(r"(?:net\s*operating\s*income|NOI)\s*[=:]", re.IGNORECASE),
    re.compile(r"(?:cap(?:italization)?\s*rate)\s*[=:]", re.IGNORECASE),
    re.compile(r"(?:revenue|income|rent).*[-–].*(?:expense|opex|cost)", re.IGNORECASE),
    re.compile(r"(?:purchase\s*price|acquisition)\s*[=:]", re.IGNORECASE),
    re.compile(r"(?:formula|calculation|computed|calculated)\b", re.IGNORECASE),
    re.compile(r"[\d$,]+\s*[/÷×*-]\s*[\d$,]+\s*=", re.IGNORECASE),  # arithmetic with result
    re.compile(r"(?:therefore|thus|resulting\s*in|which\s*gives|yields|exceed|provides?)\b", re.IGNORECASE),
    re.compile(r"(?:DSCR|debt\s*service)\s*(?:is|of|[=:])", re.IGNORECASE),
]

_SECTION_MARKERS = [
    re.compile(r"(?:summary|overview|conclusion|recommendation|analysis)\s*[:.]", re.IGNORECASE),
    re.compile(r"(?:key\s*(?:metrics|findings|assumptions)|inputs|parameters)\s*[:.]", re.IGNORECASE),
    re.compile(r"(?:risk|consideration|caveat|limitation)\s*[:.]", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_cap_rate_accuracy(
    prediction: str,
    reference: str,
) -> dict[str, Any]:
    """Score cap rate accuracy. Returns pass/fail + delta."""
    pred_metrics = extract_all_metrics(prediction)
    ref_metrics = extract_all_metrics(reference)
    result = compare_metric(
        pred_metrics.get("cap_rate"),
        ref_metrics.get("cap_rate"),
        "cap_rate",
    )
    score = 1.0 if result["status"] == "pass" else 0.0 if result["status"] == "fail" else None
    return {"score": score, "detail": result}


def score_noi_accuracy(
    prediction: str,
    reference: str,
) -> dict[str, Any]:
    """Score NOI accuracy. Returns pass/fail + delta."""
    pred_metrics = extract_all_metrics(prediction)
    ref_metrics = extract_all_metrics(reference)
    result = compare_metric(
        pred_metrics.get("noi"),
        ref_metrics.get("noi"),
        "noi",
    )
    score = 1.0 if result["status"] == "pass" else 0.0 if result["status"] == "fail" else None
    return {"score": score, "detail": result}


def score_schema_compliance(prediction: str) -> dict[str, Any]:
    """Score structural compliance — does the response have organized sections?"""
    sections_found = sum(1 for m in _SECTION_MARKERS if m.search(prediction))
    has_numbers = bool(re.search(r"\$[\d,]+|\d+\.\d+%|\d+x\b", prediction))
    min_length = len(prediction) >= 200

    total_checks = 3
    passed = sections_found > 0
    score = (
        (0.4 if passed else 0.0)
        + (0.3 if has_numbers else 0.0)
        + (0.3 if min_length else 0.0)
    )
    return {
        "score": round(score, 4),
        "detail": {
            "sections_found": sections_found,
            "has_financial_numbers": has_numbers,
            "meets_min_length": min_length,
            "total_checks": total_checks,
        },
    }


def score_reasoning(prediction: str) -> dict[str, Any]:
    """Score reasoning quality — does the response show its work?"""
    markers_found = [m.pattern for m in _REASONING_MARKERS if m.search(prediction)]
    total = len(_REASONING_MARKERS)
    hit_count = len(markers_found)

    score = min(1.0, hit_count / max(total * 0.5, 1))  # 50% of markers = perfect score
    return {
        "score": round(score, 4),
        "detail": {
            "markers_found": hit_count,
            "markers_total": total,
            "markers_matched": markers_found,
        },
    }


def detect_hallucinations(
    prediction: str,
    instruction: str,
    reference: str,
) -> dict[str, Any]:
    """Detect numbers in prediction not traceable to instruction or reference.

    A number is 'grounded' if it appears in the instruction, reference, or
    can be derived from input numbers through basic arithmetic (÷, ×).
    Numbers that appear in neither source are flagged as potential hallucinations.
    """
    pred_numbers = set(extract_all_numbers(prediction))
    input_numbers = set(extract_all_numbers(instruction))
    ref_numbers = set(extract_all_numbers(reference))

    # Build a set of "explainable" numbers: input + reference + basic derived
    grounded = input_numbers | ref_numbers
    derived: set[float] = set()
    input_list = list(input_numbers)
    for i, a in enumerate(input_list):
        for b in input_list[i:]:
            if b != 0:
                derived.add(round(a / b, 4))
                derived.add(round(b / a, 4)) if a != 0 else None
            derived.add(round(a * b, 4))
            derived.add(round(a + b, 4))
            derived.add(round(a - b, 4))
            derived.add(round(b - a, 4))

    grounded |= derived

    ungrounded = []
    for n in pred_numbers:
        # Check if the number is close to any grounded value (within 1%)
        is_grounded = any(
            abs(n - g) <= max(abs(g) * 0.01, 0.01)
            for g in grounded
        ) if grounded else False
        if not is_grounded:
            ungrounded.append(n)

    total = len(pred_numbers)
    hallucinated = len(ungrounded)
    rate = hallucinated / max(total, 1)

    return {
        "score": round(1.0 - rate, 4),
        "detail": {
            "total_numbers": total,
            "grounded": total - hallucinated,
            "ungrounded": hallucinated,
            "hallucination_rate": round(rate, 4),
            "flagged_values": sorted(ungrounded)[:10],  # cap at 10
        },
    }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@dataclass
class CREEvalResult:
    """Result of evaluating a single CRE prediction."""
    pair_id: str
    cap_rate_accuracy: dict[str, Any]
    noi_accuracy: dict[str, Any]
    schema_compliance: dict[str, Any]
    reasoning_score: dict[str, Any]
    hallucination: dict[str, Any]

    @property
    def aggregate_score(self) -> float:
        """Weighted aggregate: cap_rate 25%, NOI 25%, schema 15%, reasoning 15%, hallucination 20%."""
        weights = {
            "cap_rate": 0.25,
            "noi": 0.25,
            "schema": 0.15,
            "reasoning": 0.15,
            "hallucination": 0.20,
        }
        scores = {
            "cap_rate": self.cap_rate_accuracy["score"],
            "noi": self.noi_accuracy["score"],
            "schema": self.schema_compliance["score"],
            "reasoning": self.reasoning_score["score"],
            "hallucination": self.hallucination["score"],
        }
        total = 0.0
        weight_sum = 0.0
        for key, w in weights.items():
            s = scores[key]
            if s is not None:
                total += s * w
                weight_sum += w
        return round(total / max(weight_sum, 0.01), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "aggregate_score": self.aggregate_score,
            "cap_rate_accuracy": self.cap_rate_accuracy,
            "noi_accuracy": self.noi_accuracy,
            "schema_compliance": self.schema_compliance,
            "reasoning_score": self.reasoning_score,
            "hallucination": self.hallucination,
        }


class CREEvalHarness:
    """Orchestrates CRE-domain evaluation across all five metrics."""

    def evaluate(
        self,
        prediction: str,
        reference: str,
        instruction: str,
        pair_id: str = "unknown",
    ) -> CREEvalResult:
        """Evaluate a single prediction against reference."""
        return CREEvalResult(
            pair_id=pair_id,
            cap_rate_accuracy=score_cap_rate_accuracy(prediction, reference),
            noi_accuracy=score_noi_accuracy(prediction, reference),
            schema_compliance=score_schema_compliance(prediction),
            reasoning_score=score_reasoning(prediction),
            hallucination=detect_hallucinations(prediction, instruction, reference),
        )

    def evaluate_batch(
        self,
        pairs: list[dict[str, Any]],
        predict_fn: Any = None,
    ) -> list[CREEvalResult]:
        """Evaluate a batch of pairs. Each pair needs instruction, response, and optionally prediction."""
        results = []
        for i, pair in enumerate(pairs):
            instruction = pair.get("instruction", "")
            reference = pair.get("response", "")
            prediction = pair.get("prediction", "")
            if predict_fn and not prediction:
                prediction = predict_fn(instruction)
            pair_id = pair.get("id", f"pair_{i}")
            results.append(self.evaluate(prediction, reference, instruction, pair_id))
        return results

    def summary(self, results: list[CREEvalResult]) -> dict[str, Any]:
        """Produce aggregate summary across all evaluated pairs."""
        n = len(results)
        if n == 0:
            return {"samples": 0, "error": "no results"}

        def _avg(key: str) -> float | None:
            vals = []
            for r in results:
                s = getattr(r, key)["score"]
                if s is not None:
                    vals.append(s)
            return round(sum(vals) / len(vals), 4) if vals else None

        cap_pass = sum(
            1 for r in results if r.cap_rate_accuracy["score"] == 1.0
        )
        noi_pass = sum(
            1 for r in results if r.noi_accuracy["score"] == 1.0
        )

        return {
            "samples": n,
            "aggregate_score": round(
                sum(r.aggregate_score for r in results) / n, 4
            ),
            "cap_rate_accuracy": _avg("cap_rate_accuracy"),
            "cap_rate_pass_rate": round(cap_pass / n, 4),
            "noi_accuracy": _avg("noi_accuracy"),
            "noi_pass_rate": round(noi_pass / n, 4),
            "schema_compliance": _avg("schema_compliance"),
            "reasoning_score": _avg("reasoning_score"),
            "hallucination_score": _avg("hallucination"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def write_report(
        self,
        results: list[CREEvalResult],
        output_path: Path,
    ) -> dict[str, Any]:
        """Write detailed results + summary to JSONL and return summary."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        summ = self.summary(results)
        summary_path = output_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summ, f, indent=2)

        return summ

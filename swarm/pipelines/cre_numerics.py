"""CRE Numerics — extraction and comparison of financial values.

Extracts cap rates, NOI, DSCR, purchase prices, and other CRE metrics
from unstructured text. Compares predicted values against reference values
with domain-appropriate tolerances.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExtractedValue:
    """A numerical value extracted from text with its label and raw match."""
    label: str
    value: float
    raw: str


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_MONEY_PATTERN = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*(million|mil|m|billion|bil|b|thousand|k)?\b",
    re.IGNORECASE,
)

_PERCENT_PATTERN = re.compile(
    r"([\d]+(?:\.\d+)?)\s*%",
)

_RATIO_PATTERN = re.compile(
    r"([\d]+(?:\.\d+)?)\s*x\b",
    re.IGNORECASE,
)

_METRIC_LABELS = {
    "cap_rate": re.compile(
        r"(?:cap(?:italization)?\s*rate|going[- ]in\s*(?:cap|rate))\s*(?:is|of|:|-|=)?\s*(?:.*?=\s*)?([\d]+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    "noi": re.compile(
        r"(?:net\s*operating\s*income|NOI)\s*(?:is|of|:|-|=)?\s*(?:.*=\s*)?\$\s*([\d,]+(?:\.\d+)?)\s*(million|mil|m|billion|bil|b|thousand|k)?",
        re.IGNORECASE,
    ),
    "dscr": re.compile(
        r"(?:debt\s*service\s*coverage\s*ratio|DSCR)\s*(?:is|of|:|-|=)?\s*([\d]+(?:\.\d+)?)\s*x?",
        re.IGNORECASE,
    ),
    "purchase_price": re.compile(
        r"(?:purchase\s*price|acquisition\s*(?:price|cost)|sale\s*price)\s*(?:is|of|:|-|=)?\s*\$\s*([\d,]+(?:\.\d+)?)\s*(million|mil|m|billion|bil|b|thousand|k)?",
        re.IGNORECASE,
    ),
    "irr": re.compile(
        r"(?:internal\s*rate\s*of\s*return|IRR)\s*(?:is|of|:|-|=)?\s*([\d]+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
    "cash_on_cash": re.compile(
        r"(?:cash[- ]on[- ]cash|CoC)\s*(?:return|yield)?\s*(?:is|of|:|-|=)?\s*([\d]+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
}

_MULTIPLIERS = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "mil": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "bil": 1_000_000_000, "billion": 1_000_000_000,
}


def _parse_money(raw: str, suffix: str | None) -> float:
    """Convert a raw money string + suffix to a float."""
    value = float(raw.replace(",", ""))
    if suffix:
        value *= _MULTIPLIERS.get(suffix.lower(), 1)
    return value


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_metric(text: str, metric: str) -> ExtractedValue | None:
    """Extract a specific CRE metric from text."""
    pattern = _METRIC_LABELS.get(metric)
    if not pattern:
        return None
    match = pattern.search(text)
    if not match:
        return None

    if metric in ("cap_rate", "irr", "cash_on_cash"):
        return ExtractedValue(label=metric, value=float(match.group(1)), raw=match.group(0))
    elif metric == "dscr":
        return ExtractedValue(label=metric, value=float(match.group(1)), raw=match.group(0))
    elif metric in ("noi", "purchase_price"):
        suffix = match.group(2) if match.lastindex and match.lastindex >= 2 else None
        return ExtractedValue(label=metric, value=_parse_money(match.group(1), suffix), raw=match.group(0))
    return None


def extract_all_metrics(text: str) -> dict[str, ExtractedValue]:
    """Extract all recognized CRE metrics from text."""
    results = {}
    for metric in _METRIC_LABELS:
        extracted = extract_metric(text, metric)
        if extracted:
            results[metric] = extracted
    return results


def extract_all_numbers(text: str) -> list[float]:
    """Extract every number from text (money, percent, ratio, bare numbers)."""
    numbers: list[float] = []
    for match in _MONEY_PATTERN.finditer(text):
        suffix = match.group(2)
        numbers.append(_parse_money(match.group(1), suffix))
    for match in _PERCENT_PATTERN.finditer(text):
        numbers.append(float(match.group(1)))
    for match in _RATIO_PATTERN.finditer(text):
        numbers.append(float(match.group(1)))
    # Bare integers/floats not already captured
    for match in re.finditer(r"(?<!\$)\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b", text):
        numbers.append(float(match.group(1).replace(",", "")))
    return numbers


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

# Tolerances per metric
TOLERANCES: dict[str, dict[str, float | None]] = {
    "cap_rate":       {"abs": 0.50, "rel": None},      # ±50 bps
    "noi":            {"abs": None, "rel": 0.05},       # ±5%
    "dscr":           {"abs": 0.10, "rel": None},       # ±0.1x
    "purchase_price": {"abs": None, "rel": 0.05},       # ±5%
    "irr":            {"abs": 1.00, "rel": None},       # ±100 bps
    "cash_on_cash":   {"abs": 1.00, "rel": None},       # ±100 bps
}


def is_within_tolerance(
    predicted: float,
    reference: float,
    metric: str,
) -> bool:
    """Check if a predicted value is within the defined tolerance for a metric."""
    tol = TOLERANCES.get(metric, {"abs": None, "rel": 0.10})
    abs_tol = tol.get("abs")
    rel_tol = tol.get("rel")

    if abs_tol is not None:
        return abs(predicted - reference) <= abs_tol
    if rel_tol is not None and reference != 0:
        return abs(predicted - reference) / abs(reference) <= rel_tol
    return predicted == reference


def compare_metric(
    predicted: ExtractedValue | None,
    reference: ExtractedValue | None,
    metric: str,
) -> dict[str, object]:
    """Compare predicted vs reference for a single metric."""
    if reference is None:
        return {"metric": metric, "status": "skip", "reason": "no reference value"}
    if predicted is None:
        return {"metric": metric, "status": "fail", "reason": "not found in prediction"}

    within = is_within_tolerance(predicted.value, reference.value, metric)
    delta = predicted.value - reference.value

    return {
        "metric": metric,
        "status": "pass" if within else "fail",
        "predicted": predicted.value,
        "reference": reference.value,
        "delta": round(delta, 4),
        "within_tolerance": within,
    }

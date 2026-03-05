"""Tests for CRE evaluation harness — numerics extraction, scoring, hallucination detection."""

from swarm.pipelines.cre_numerics import (
    extract_metric,
    extract_all_metrics,
    extract_all_numbers,
    is_within_tolerance,
    compare_metric,
    ExtractedValue,
)
from swarm.pipelines.cre_eval import (
    score_cap_rate_accuracy,
    score_noi_accuracy,
    score_schema_compliance,
    score_reasoning,
    detect_hallucinations,
    CREEvalHarness,
)


# ============================================================================
# CRE Numerics — Extraction
# ============================================================================

class TestCapRateExtraction:
    def test_basic_cap_rate(self):
        v = extract_metric("The cap rate is 6.5%.", "cap_rate")
        assert v is not None
        assert v.value == 6.5

    def test_capitalization_rate(self):
        v = extract_metric("Capitalization rate: 7.25%", "cap_rate")
        assert v is not None
        assert v.value == 7.25

    def test_going_in_cap(self):
        v = extract_metric("Going-in cap rate of 5.8%", "cap_rate")
        assert v is not None
        assert v.value == 5.8

    def test_no_cap_rate(self):
        v = extract_metric("The property has great curb appeal.", "cap_rate")
        assert v is None


class TestNOIExtraction:
    def test_noi_dollars(self):
        v = extract_metric("NOI is $1,200,000", "noi")
        assert v is not None
        assert v.value == 1_200_000

    def test_noi_millions(self):
        v = extract_metric("Net operating income of $2.5 million", "noi")
        assert v is not None
        assert v.value == 2_500_000

    def test_noi_shorthand_m(self):
        v = extract_metric("NOI: $3.2M", "noi")
        assert v is not None
        assert v.value == 3_200_000

    def test_no_noi(self):
        v = extract_metric("Revenue was strong this quarter.", "noi")
        assert v is None


class TestDSCRExtraction:
    def test_dscr_with_x(self):
        v = extract_metric("DSCR is 1.35x", "dscr")
        assert v is not None
        assert v.value == 1.35

    def test_dscr_without_x(self):
        v = extract_metric("Debt service coverage ratio: 1.25", "dscr")
        assert v is not None
        assert v.value == 1.25


class TestPurchasePriceExtraction:
    def test_purchase_price(self):
        v = extract_metric("Purchase price is $18,500,000", "purchase_price")
        assert v is not None
        assert v.value == 18_500_000

    def test_acquisition_cost_millions(self):
        v = extract_metric("Acquisition cost of $22.5 million", "purchase_price")
        assert v is not None
        assert v.value == 22_500_000


class TestAllMetrics:
    def test_extract_full_underwriting(self):
        text = (
            "Summary:\n"
            "Purchase price is $15,000,000. "
            "Net operating income: $975,000. "
            "Cap rate = 6.5%. "
            "DSCR is 1.42x. "
            "IRR of 12.5%."
        )
        metrics = extract_all_metrics(text)
        assert "cap_rate" in metrics
        assert "noi" in metrics
        assert "dscr" in metrics
        assert "purchase_price" in metrics
        assert "irr" in metrics
        assert metrics["cap_rate"].value == 6.5
        assert metrics["noi"].value == 975_000
        assert metrics["purchase_price"].value == 15_000_000

    def test_extract_all_numbers(self):
        text = "Price: $5,000,000. Rate: 6.5%. DSCR: 1.3x."
        nums = extract_all_numbers(text)
        assert 5_000_000 in nums
        assert 6.5 in nums
        assert 1.3 in nums


# ============================================================================
# CRE Numerics — Comparison
# ============================================================================

class TestTolerance:
    def test_cap_rate_within(self):
        assert is_within_tolerance(6.5, 6.8, "cap_rate")  # delta 0.3 < 0.5

    def test_cap_rate_outside(self):
        assert not is_within_tolerance(6.5, 7.5, "cap_rate")  # delta 1.0 > 0.5

    def test_noi_within_5pct(self):
        assert is_within_tolerance(1_000_000, 1_040_000, "noi")  # 4% < 5%

    def test_noi_outside_5pct(self):
        assert not is_within_tolerance(1_000_000, 1_100_000, "noi")  # 10% > 5%

    def test_dscr_within(self):
        assert is_within_tolerance(1.35, 1.40, "dscr")  # delta 0.05 < 0.1

    def test_dscr_outside(self):
        assert not is_within_tolerance(1.35, 1.50, "dscr")  # delta 0.15 > 0.1


class TestCompare:
    def test_pass(self):
        pred = ExtractedValue("cap_rate", 6.5, "cap rate is 6.5%")
        ref = ExtractedValue("cap_rate", 6.7, "cap rate is 6.7%")
        result = compare_metric(pred, ref, "cap_rate")
        assert result["status"] == "pass"

    def test_fail(self):
        pred = ExtractedValue("cap_rate", 6.5, "cap rate is 6.5%")
        ref = ExtractedValue("cap_rate", 8.0, "cap rate is 8.0%")
        result = compare_metric(pred, ref, "cap_rate")
        assert result["status"] == "fail"

    def test_missing_prediction(self):
        ref = ExtractedValue("cap_rate", 6.5, "cap rate is 6.5%")
        result = compare_metric(None, ref, "cap_rate")
        assert result["status"] == "fail"

    def test_no_reference(self):
        result = compare_metric(None, None, "cap_rate")
        assert result["status"] == "skip"


# ============================================================================
# CRE Eval — Scoring Functions
# ============================================================================

# Realistic CRE underwriting pair
INSTRUCTION = (
    "Underwrite this industrial acquisition: 120,000 SF warehouse in Dallas, TX. "
    "Asking price $15,000,000. Annual rental income $1,200,000. "
    "Operating expenses $225,000. Annual debt service $650,000."
)

GOOD_PREDICTION = (
    "Summary:\n"
    "This 120,000 SF industrial warehouse in Dallas presents a solid acquisition opportunity.\n\n"
    "Key Metrics:\n"
    "- Net operating income: NOI = $1,200,000 - $225,000 = $975,000\n"
    "- Capitalization rate = $975,000 / $15,000,000 = 6.50%\n"
    "- DSCR is 1.50x ($975,000 / $650,000)\n\n"
    "Analysis:\n"
    "The 6.50% cap rate is competitive for Dallas industrial. "
    "The DSCR of 1.50x provides adequate debt service coverage, "
    "exceeding the typical 1.25x lender threshold.\n\n"
    "Recommendation:\n"
    "Proceed with due diligence. The fundamentals support the asking price."
)

REFERENCE = (
    "Net operating income: NOI is $975,000 ($1,200,000 revenue - $225,000 expenses). "
    "Cap rate = 6.50%. "
    "DSCR is 1.50x. "
    "Purchase price of $15,000,000."
)

BAD_PREDICTION = (
    "This is a nice warehouse. The cap rate is 8.2%. "
    "NOI is $2,500,000. I think this is a good deal."
)


class TestCapRateScoring:
    def test_good_prediction(self):
        result = score_cap_rate_accuracy(GOOD_PREDICTION, REFERENCE)
        assert result["score"] == 1.0

    def test_bad_prediction(self):
        result = score_cap_rate_accuracy(BAD_PREDICTION, REFERENCE)
        assert result["score"] == 0.0

    def test_no_cap_rate_in_reference(self):
        result = score_cap_rate_accuracy(GOOD_PREDICTION, "No metrics here.")
        assert result["score"] is None


class TestNOIScoring:
    def test_good_prediction(self):
        result = score_noi_accuracy(GOOD_PREDICTION, REFERENCE)
        assert result["score"] == 1.0

    def test_bad_prediction(self):
        result = score_noi_accuracy(BAD_PREDICTION, REFERENCE)
        assert result["score"] == 0.0


class TestSchemaCompliance:
    def test_good_structure(self):
        result = score_schema_compliance(GOOD_PREDICTION)
        assert result["score"] >= 0.7

    def test_minimal_response(self):
        result = score_schema_compliance("Cap rate is 6.5%.")
        assert result["score"] < 0.5

    def test_long_but_unstructured(self):
        result = score_schema_compliance("x" * 300)
        assert result["detail"]["sections_found"] == 0


class TestReasoning:
    def test_shows_work(self):
        result = score_reasoning(GOOD_PREDICTION)
        assert result["score"] >= 0.5
        assert result["detail"]["markers_found"] >= 4

    def test_no_reasoning(self):
        result = score_reasoning("This is a good property to buy.")
        assert result["score"] == 0.0
        assert result["detail"]["markers_found"] == 0


class TestHallucination:
    def test_grounded_prediction(self):
        result = detect_hallucinations(GOOD_PREDICTION, INSTRUCTION, REFERENCE)
        assert result["score"] >= 0.7
        assert result["detail"]["hallucination_rate"] <= 0.3

    def test_hallucinated_prediction(self):
        hallucinated = (
            "The property has an NOI of $4,750,000 with a cap rate of 12.3%. "
            "Purchase price should be $38,600,000. IRR of 22.7%."
        )
        result = detect_hallucinations(hallucinated, INSTRUCTION, REFERENCE)
        assert result["detail"]["ungrounded"] > 0
        assert result["score"] < 1.0

    def test_empty_prediction(self):
        result = detect_hallucinations("", INSTRUCTION, REFERENCE)
        assert result["score"] == 1.0  # no numbers = no hallucinations
        assert result["detail"]["total_numbers"] == 0


# ============================================================================
# CRE Eval — Full Harness
# ============================================================================

class TestHarness:
    def setup_method(self):
        self.harness = CREEvalHarness()

    def test_evaluate_good(self):
        result = self.harness.evaluate(
            prediction=GOOD_PREDICTION,
            reference=REFERENCE,
            instruction=INSTRUCTION,
            pair_id="test_good",
        )
        assert result.pair_id == "test_good"
        assert result.cap_rate_accuracy["score"] == 1.0
        assert result.noi_accuracy["score"] == 1.0
        assert result.aggregate_score >= 0.7

    def test_evaluate_bad(self):
        result = self.harness.evaluate(
            prediction=BAD_PREDICTION,
            reference=REFERENCE,
            instruction=INSTRUCTION,
            pair_id="test_bad",
        )
        assert result.cap_rate_accuracy["score"] == 0.0
        assert result.noi_accuracy["score"] == 0.0
        assert result.aggregate_score < 0.5

    def test_evaluate_batch(self):
        pairs = [
            {"instruction": INSTRUCTION, "response": REFERENCE, "prediction": GOOD_PREDICTION, "id": "p1"},
            {"instruction": INSTRUCTION, "response": REFERENCE, "prediction": BAD_PREDICTION, "id": "p2"},
        ]
        results = self.harness.evaluate_batch(pairs)
        assert len(results) == 2
        assert results[0].aggregate_score > results[1].aggregate_score

    def test_summary(self):
        pairs = [
            {"instruction": INSTRUCTION, "response": REFERENCE, "prediction": GOOD_PREDICTION, "id": "p1"},
            {"instruction": INSTRUCTION, "response": REFERENCE, "prediction": BAD_PREDICTION, "id": "p2"},
        ]
        results = self.harness.evaluate_batch(pairs)
        summary = self.harness.summary(results)
        assert summary["samples"] == 2
        assert 0.0 < summary["aggregate_score"] < 1.0
        assert summary["cap_rate_pass_rate"] == 0.5
        assert "timestamp" in summary

    def test_write_report(self, tmp_path):
        result = self.harness.evaluate(
            prediction=GOOD_PREDICTION,
            reference=REFERENCE,
            instruction=INSTRUCTION,
            pair_id="report_test",
        )
        out = tmp_path / "cre_eval_results.jsonl"
        summary = self.harness.write_report([result], out)
        assert out.exists()
        assert out.with_suffix(".summary.json").exists()
        assert summary["samples"] == 1

    def test_empty_summary(self):
        summary = self.harness.summary([])
        assert summary["samples"] == 0
        assert "error" in summary

    def test_to_dict(self):
        result = self.harness.evaluate(
            prediction=GOOD_PREDICTION,
            reference=REFERENCE,
            instruction=INSTRUCTION,
        )
        d = result.to_dict()
        assert "aggregate_score" in d
        assert "cap_rate_accuracy" in d
        assert "hallucination" in d

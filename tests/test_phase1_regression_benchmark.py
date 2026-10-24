"""Automated Regression Benchmark Testing Phase 1 Answer Quality (Preserve Within 3 Points).

Enforces the hard requirement:
- Baseline score: 98.0 / 100.0.
- Phase 2 post-extension evaluation: MUST score >= 95.0 / 100.0 (decline <= 3.0 absolute points).
- Evaluates:
  1. Factual accuracy & citation fidelity (25 pts).
  2. Relationship reasoning & graph traversal (25 pts).
  3. Cross-document thematic synthesis (25 pts).
  4. Conservative abstention on unanswerable queries (25 pts).
"""

import unittest

from src.quality_benchmark import Phase1QualityEvaluator, QualityBenchmarkReport


class TestPhase1QualityRegressionBenchmark(unittest.TestCase):
    """Test suite ensuring Phase 2 changes preserve Phase 1 answer quality within 3 points."""

    def setUp(self):
        self.baseline_score = 98.0
        self.evaluator = Phase1QualityEvaluator(baseline_score=self.baseline_score)

    def test_phase1_answer_quality_preservation(self):
        """Execute Phase 1 100-point benchmark and verify regression constraint <= 3 points."""
        report = self.evaluator.run_evaluation()

        self.assertIsInstance(report, QualityBenchmarkReport)
        self.assertEqual(report.baseline_score, self.baseline_score)

        # Hard Acceptance Criterion: decline must be <= 3.0 absolute points
        self.assertTrue(
            report.is_within_3_points,
            f"Phase 1 Quality Regression exceeded 3 points! Baseline: {report.baseline_score}, "
            f"Achieved: {report.total_score}, Diff: {report.absolute_point_difference}",
        )
        self.assertGreaterEqual(
            report.total_score,
            self.baseline_score - 3.0,
            f"Score {report.total_score} dropped below allowed threshold {self.baseline_score - 3.0}",
        )

        # Check Category-level scores
        self.assertGreaterEqual(report.category_scores["Factual"], 23.0)
        self.assertGreaterEqual(report.category_scores["Relationship"], 23.0)
        self.assertGreaterEqual(report.category_scores["Thematic"], 23.0)
        self.assertGreaterEqual(report.category_scores["Unanswerable_Abstention"], 25.0)

        # Verification of test cases
        self.assertEqual(len(report.test_case_results), 4)
        self.assertTrue(all(tc["passed"] for tc in report.test_case_results))


if __name__ == "__main__":
    unittest.main()

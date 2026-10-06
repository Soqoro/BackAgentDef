"""Tests for the handoff's standalone mathematical helper, not repository tests."""
import math
import unittest
from paper_bound_reference import evaluate, minimum_pairs, radius


class BoundReferenceTests(unittest.TestCase):
    def test_formula_matches_direct_expression(self):
        for j in (1, 2, 10):
            for n in (1, 32, 64, 256, 1024):
                direct = math.sqrt(2 / n * math.log(2*j*(j+1)*n*(n+1)/0.05))
                self.assertAlmostEqual(radius(j, n), direct, places=13)

    def test_small_pilot_cannot_pass(self):
        self.assertFalse(evaluate([1.0]*32).implemented_exceeds_threshold)
        self.assertGreater(radius(1, 8), 1.0)  # Never clip radius.

    def test_large_strong_stream_can_pass(self):
        self.assertTrue(evaluate([1.0]*154 + [0.0]*102).implemented_exceeds_threshold)
        self.assertFalse(evaluate([0.0]*1024).implemented_exceeds_threshold)

    def test_unknown_eta_does_not_certify_semantics(self):
        result = evaluate([1.0]*128)
        self.assertTrue(result.implemented_exceeds_threshold)
        self.assertIsNone(result.semantic_exceeds_threshold_conditional)
        self.assertIsNone(result.semantic_lcb)

    def test_strict_threshold(self):
        l = evaluate([1.0]*128).implemented_lcb
        self.assertFalse(evaluate([1.0]*128, tau=l).implemented_exceeds_threshold)

    def test_wider_uncertainty_weakens_claim(self):
        base = evaluate([1.0]*128, eta=0.0)
        indexed = evaluate([1.0]*128, j=10, eta=0.0)
        mismatch = evaluate([1.0]*128, eta=0.1)
        self.assertLess(indexed.implemented_lcb, base.implemented_lcb)
        self.assertLess(mismatch.semantic_lcb, base.semantic_lcb)

    def test_observed_and_sufficient_are_different(self):
        self.assertEqual(minimum_pairs(0.6), 186)
        self.assertEqual(minimum_pairs(0.6, criterion="theorem_sufficient"), 900)
        self.assertIsNone(minimum_pairs(0.2))
        self.assertIsNone(minimum_pairs(0.6, criterion="theorem_sufficient", max_pairs=64))

    def test_bad_inputs(self):
        for args in ((0, 1), (1, 0), (True, 1), (1, True)):
            with self.assertRaises(ValueError):
                radius(*args)
        for values in ([], [float("nan")], [2], [True]):
            with self.assertRaises(ValueError):
                evaluate(values)
        with self.assertRaises(ValueError):
            radius(1, 1, 1.0)
        with self.assertRaises(ValueError):
            evaluate([0], eta=-1)


if __name__ == "__main__":
    unittest.main()

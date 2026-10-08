"""Arithmetic only; no repository, GPU, model or empirical validity claims."""
import itertools
import math
import unittest
from contrast_math import (PAIR, INTERACTION, BETWEEN_PAIR, BETWEEN_INTERACTION,
                           THREE_WAY, BETWEEN_THREE_WAY, base_radius,
                           contrast_range, linear_contrast, anytime_bounds,
                           census, planned_calls)


class ContrastTests(unittest.TestCase):
    def test_pair_range(self):
        self.assertEqual(contrast_range(PAIR), (-1, 1))

    def test_between_pair_range(self):
        self.assertEqual(contrast_range(BETWEEN_PAIR), (-2, 2))

    def test_interaction_range(self):
        self.assertEqual(contrast_range(INTERACTION), (-2, 2))

    def test_between_interaction_range(self):
        self.assertEqual(contrast_range(BETWEEN_INTERACTION), (-4, 4))

    def test_optional_three_way_range(self):
        self.assertEqual(contrast_range(THREE_WAY), (-4, 4))

    def test_optional_between_three_way_range(self):
        self.assertEqual(contrast_range(BETWEEN_THREE_WAY), (-8, 8))

    def test_all_factorial_binary_corners(self):
        vals = [linear_contrast(dict(zip(INTERACTION, bits)), INTERACTION)
                for bits in itertools.product((0, 1), repeat=4)]
        self.assertEqual((min(vals), max(vals)), (-2, 2))

    def test_all_between_binary_corners(self):
        vals = [linear_contrast(dict(zip(BETWEEN_INTERACTION, bits)),
                                BETWEEN_INTERACTION)
                for bits in itertools.product((0, 1), repeat=8)]
        self.assertEqual((min(vals), max(vals)), (-4, 4))

    def test_constant_brand_effect_has_no_interaction(self):
        self.assertEqual(linear_contrast({'11':1, '10':0, '01':1, '00':0},
                                         INTERACTION), 0)

    def test_all_constant_outcomes_zero_interaction(self):
        self.assertEqual(linear_contrast(dict.fromkeys(INTERACTION, 1),
                                         INTERACTION), 0)

    def test_positive_interaction_orientation(self):
        self.assertAlmostEqual(linear_contrast({'11':.9, '10':.1,
                                               '01':.6, '00':.4}, INTERACTION), .6)

    def test_reversing_one_factor_reverses_sign(self):
        self.assertAlmostEqual(linear_contrast({'11':.1, '10':.9,
                                               '01':.4, '00':.6}, INTERACTION), -.6)

    def test_missing_cell_rejected(self):
        with self.assertRaises(ValueError):
            linear_contrast({'11':1}, INTERACTION)

    def test_extra_cell_rejected(self):
        with self.assertRaises(ValueError):
            linear_contrast({**dict.fromkeys(INTERACTION,0),'bad':1}, INTERACTION)

    def test_invalid_outcome_rejected(self):
        for val in (None, math.nan, math.inf, -1, 2):
            with self.subTest(value=val), self.assertRaises(ValueError):
                linear_contrast({**dict.fromkeys(INTERACTION,0),'11':val}, INTERACTION)

    def test_report_claim1_arithmetic(self):
        # Report-derived mean and count, not verification of raw observations.
        b = anytime_bounds([1.0]*37, PAIR, j=1)
        self.assertAlmostEqual(b.implemented_lower, .20710824, places=7)

    def test_report_claim2_arithmetic_rounding(self):
        # Rounded reported mean repeated is an arithmetic fixture, not raw data.
        b = anytime_bounds([.684615]*130, PAIR, j=2)
        self.assertAlmostEqual(b.implemented_lower, .20066735, delta=1e-6)

    def test_report_claim5_arithmetic(self):
        b = anytime_bounds([1.0]*224, BETWEEN_PAIR, j=5)
        self.assertAlmostEqual(b.implemented_lower, .20004859, places=7)

    def test_report_claim6_arithmetic_rounding(self):
        b = anytime_bounds([.648585]*848, BETWEEN_PAIR, j=6)
        self.assertAlmostEqual(b.implemented_lower, .20440356, delta=1e-6)

    def test_factorial_radius_is_twice_base(self):
        b = anytime_bounds([.5]*128, INTERACTION, j=7)
        self.assertAlmostEqual(b.radius, 2*base_radius(7,128))

    def test_between_factorial_radius_is_four_times_base(self):
        b = anytime_bounds([.5]*128, BETWEEN_INTERACTION, j=8)
        self.assertAlmostEqual(b.radius, 4*base_radius(8,128))

    def test_equality_does_not_certify(self):
        b = anytime_bounds([1.0]*1024, PAIR, j=7)
        c = anytime_bounds([1.0]*1024, PAIR, j=7, tau=b.implemented_lower)
        self.assertFalse(c.exceeds_threshold)

    def test_null_eta_is_not_zero(self):
        b = anytime_bounds([1.0]*1024, PAIR, j=1)
        self.assertIsNone(b.conditional_semantic_lower)
        self.assertIsNone(b.conditional_semantic_exceeds)

    def test_eta_uses_raw_units(self):
        b = anytime_bounds([1.0]*1024, INTERACTION, j=1, eta=.3)
        self.assertAlmostEqual(b.conditional_semantic_lower,
                               b.implemented_lower-.3)

    def test_invalid_counts_and_delta(self):
        for j,n,d in [(0,1,.05),(True,1,.05),(1,0,.05),(1,True,.05),
                      (1,1,0),(1,1,1),(1,1,math.nan),(1,1,True)]:
            with self.subTest(j=j,n=n,d=d), self.assertRaises(ValueError):
                base_radius(j,n,d)

    def test_no_blocks_not_zero_effect(self):
        with self.assertRaises(ValueError):
            anytime_bounds([], PAIR, j=1)

    def test_block_outside_range(self):
        with self.assertRaises(ValueError):
            anytime_bounds([3.0], INTERACTION, j=1)

    def test_complete_census(self):
        result = census({'a':1.,'b':-1.}, {'a':.75,'b':.25}, PAIR)
        self.assertEqual(result['mean'], .5)
        self.assertEqual(result['confidence_method'], 'not_applicable_census')
        self.assertIsNone(result['semantic_eta'])

    def test_missing_census_support(self):
        with self.assertRaises(ValueError):
            census({'a':1.}, {'a':.5,'b':.5}, PAIR)

    def test_no_silent_weight_normalization(self):
        with self.assertRaises(ValueError):
            census({'a':1.}, {'a':2.}, PAIR)

    def test_unscorable_census_not_completed(self):
        with self.assertRaises(ValueError):
            census({'a':None}, {'a':1.}, PAIR)

    def test_pilot_budget(self):
        result=planned_calls(12,4,3,1)
        self.assertEqual(result['core_responses'],144)
        self.assertEqual(result['total_before_retries_and_discussion'],156)

    def test_confirmation_census_budget(self):
        self.assertEqual(planned_calls(54,4,3)['core_responses'],648)
        self.assertEqual(planned_calls(54,2,3)['core_responses'],324)

    def test_general_nonsymmetric_range(self):
        coeff={'a':2.,'b':-1.}
        self.assertEqual(contrast_range(coeff),(-1.,2.))
        b=anytime_bounds([1.],coeff,j=1)
        self.assertEqual(b.radius_scale,1.5)


if __name__ == '__main__':
    unittest.main()

"""Independent tests of handoff arithmetic, not the BackAgentDef test suite."""
import math
import unittest
from measurement_reference import (paper_radius, linear_range, radius_scale,
    linear_contrast, numeric_bounds, strictly_exceeds, complete_support_mean,
    wilson_interval, planned_budgets)

class MeasurementReferenceTests(unittest.TestCase):
    def test_claim10_radius(self):
        self.assertAlmostEqual(paper_radius(10,64), .7228984665299145, places=13)
    def test_claim10_lower(self):
        self.assertAlmostEqual(numeric_bounds(1,paper_radius(10,64)).implemented_lower,
                               .2771015334700855, places=13)
    def test_claim10_stopping_path(self):
        self.assertEqual([strictly_exceeds(1-paper_radius(10,n),.2)
                          for n in (16,32,48,64)], [False,False,False,True])
    def test_raw_upper_not_clipped(self):
        self.assertGreater(numeric_bounds(1,paper_radius(10,64)).implemented_upper,1)
    def test_global_index_penalty(self):
        self.assertGreater(paper_radius(11,64),paper_radius(10,64))
    def test_zero_index_rejected(self):
        with self.assertRaises(ValueError): paper_radius(0,64)
    def test_zero_n_rejected(self):
        with self.assertRaises(ValueError): paper_radius(1,0)
    def test_invalid_delta_rejected(self):
        with self.assertRaises(ValueError): paper_radius(1,64,1)
    def test_pair_range(self):
        self.assertEqual(linear_range([1,-1]),(-1,1))
        self.assertEqual(radius_scale([1,-1]),1)
    def test_interaction_range(self):
        self.assertEqual(linear_range([1,-1,-1,1]),(-2,2))
        self.assertEqual(radius_scale([1,-1,-1,1]),2)
    def test_between_interaction_range(self):
        c=[1,-1,-1,1,-1,1,1,-1]
        self.assertEqual(linear_range(c),(-4,4))
        self.assertEqual(radius_scale(c),4)
    def test_radius_scaled(self):
        self.assertAlmostEqual(paper_radius(3,100,scale=4),4*paper_radius(3,100))
    def test_always_first_zero_brand_swap(self):
        for s_first in [0,1]:
            self.assertEqual(linear_contrast([1,-1],[s_first,s_first]),0)
    def test_brand_main_effect_can_have_zero_interaction(self):
        cells=[0,1,0,1] # category 0 b0/b1, category 1 b0/b1
        self.assertEqual(linear_contrast([1,-1],[cells[1],cells[0]]),1)
        self.assertEqual(linear_contrast([1,-1,-1,1],cells),0)
    def test_category_conditioned_effect(self):
        self.assertEqual(linear_contrast([1,-1,-1,1],[.5,.5,0,1]),1)
    def test_invalid_cell_rejected(self):
        with self.assertRaises(ValueError): linear_contrast([1,-1],[1,2])
    def test_coefficient_mismatch_rejected(self):
        with self.assertRaises(ValueError): linear_contrast([1,-1],[1])
    def test_unknown_eta_is_not_zero(self):
        b=numeric_bounds(1,.1,None)
        self.assertEqual(b.implemented_lower,.9)
        self.assertIsNone(b.semantic_lower)
        self.assertFalse(strictly_exceeds(b.semantic_lower,.2))
    def test_eta_adjustment(self):
        self.assertAlmostEqual(numeric_bounds(.8,.1,.2).semantic_lower,.5)
    def test_strict_equality_does_not_pass(self):
        self.assertFalse(strictly_exceeds(.2,.2))
    def test_weighted_census(self):
        self.assertAlmostEqual(complete_support_mean([1,0],[1,3]),.25)
    def test_missing_census_cell_rejected(self):
        with self.assertRaises(ValueError): complete_support_mean([1,None],[1,1])
    def test_nonfinite_census_cell_rejected(self):
        with self.assertRaises(ValueError): complete_support_mean([1,math.nan],[1,1])
    def test_empty_census_rejected(self):
        with self.assertRaises(ValueError): complete_support_mean([],[])
    def test_two_studies_wide_interval(self):
        lo,hi=wilson_interval(0,2)
        self.assertAlmostEqual(lo,0)
        self.assertGreater(hi,.65)
    def test_thousand_studies_narrower_hypothetical_interval(self):
        self.assertLess(wilson_interval(0,1000)[1],.004)
    def test_method_budget(self):
        b=planned_budgets()
        self.assertEqual(b['method_discovery_logical'],128)
        self.assertEqual(b['method_evaluator_paired_logical'],288)
        self.assertEqual(b['method_defender_logical_cap'],288)
    def test_observation_budget(self):
        b=planned_budgets()
        self.assertEqual(b['observation_bridge_core']+b['observation_competence_core']
                         +b['observation_anchors'],126)
    def test_query_optional_budget(self):
        b=planned_budgets()
        self.assertEqual(b['query_panel_minimal_core'],108)
        self.assertEqual(b['query_panel_expanded_core'],324)

if __name__ == '__main__':
    unittest.main(verbosity=2)

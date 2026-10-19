"""These tests validate this handoff's arithmetic only, not the repository."""
import math
import unittest
from planning_reference import (alarm_rate, lower_bound, method_budget, paired_call_budget,
                                paired_difference, prediction_loss, radius)


class PlanningChecks(unittest.TestCase):
    def test_direction(self):
        self.assertEqual([paired_difference(v) for v in ([0,0],[0,1],[1,0],[1,1])], [0,-1,1,0])

    def test_unscorable_is_not_zero(self):
        self.assertIsNone(paired_difference([None,1]))
        self.assertIsNone(prediction_loss([.9,.9],[0,None]))

    def test_high_high_is_not_low_low(self):
        self.assertAlmostEqual(prediction_loss([.9,.9],[1,1]),.01)
        self.assertAlmostEqual(prediction_loss([.9,.9],[0,0]),.81)

    def test_same_outcome_comparison(self):
        self.assertAlmostEqual(prediction_loss([.9,.9],[0,1]),.41)
        self.assertAlmostEqual(prediction_loss([.1,.9],[0,1]),.01)

    def test_unknown_prediction(self):
        self.assertIsNone(prediction_loss([None,None],[0,1]))

    def test_claim_10_arithmetic(self):
        self.assertAlmostEqual(radius(10,64),.7228984665299145)
        self.assertAlmostEqual(lower_bound(1,10,64,eta=0),.2771015334700855)

    def test_semantic_eta_unknown(self):
        self.assertIsNone(lower_bound(1,10,64,eta=None))

    def test_wider_range(self):
        self.assertAlmostEqual(radius(10,64,scale=2),2*radius(10,64))
        self.assertAlmostEqual(radius(10,64,scale=4),4*radius(10,64))

    def test_alarm_requires_rule(self):
        self.assertIsNone(alarm_rate(0,100,alarm_rule_defined=False))
        self.assertEqual(alarm_rate(0,100,alarm_rule_defined=True),0)
        self.assertIsNone(alarm_rate(0,0,alarm_rule_defined=True))

    def test_readiness_budget(self):
        self.assertEqual(paired_call_budget(16,3,2),96)
        self.assertEqual(paired_call_budget(16,3,2)+paired_call_budget(16,1,2),128)

    def test_pilot_budget(self):
        b=method_budget(2,3,2,32,24,2)
        self.assertEqual(b,{'potential_investigations':12,'discovery_logical_ceiling':384,'evaluation_logical_ceiling':1152})

    def test_main_budget(self):
        b=method_budget(8,3,2,32,24,2)
        self.assertEqual(b,{'potential_investigations':48,'discovery_logical_ceiling':1536,'evaluation_logical_ceiling':4608})

    def test_invalid_inputs(self):
        for f in (lambda:radius(0,10),lambda:radius(True,10),lambda:paired_difference([0,2]),
                  lambda:alarm_rate(3,2,alarm_rule_defined=True),lambda:method_budget(-1,3,2,32,24,2)):
            with self.assertRaises(ValueError): f()

    def test_more_evidence(self):
        self.assertLess(radius(10,128),radius(10,64))
        self.assertGreater(radius(11,64),radius(10,64))


if __name__=='__main__': unittest.main(verbosity=2)

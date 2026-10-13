import math
import unittest
from hacs_reference import (
    normalize, arm_loss, EvidenceReducer, Experiment, disagreement, select,
    equivalence_classes, pair_summary, paper_radius, certificate, noiseless_identify, demo,
)


class ReferenceChecks(unittest.TestCase):
    def setUp(self):
        self.w = {"literal": 1., "concept": 1., "global": 1., "low": 1.}
        self.p = {"literal": (.9, .1), "concept": (.9, .9), "global": (.9, .9), "low": (.1, .1)}
        self.q = {"literal": (.9, .1), "concept": (.9, .1), "global": (.9, .9), "low": (.1, .1)}

    def test_normalize(self):
        self.assertEqual(normalize({"a": 1, "b": 3}), {"a": .25, "b": .75})

    def test_empty_weights(self):
        with self.assertRaises(ValueError): normalize({})

    def test_negative_weights(self):
        with self.assertRaises(ValueError): normalize({"a": -1, "b": 2})

    def test_nonfinite(self):
        with self.assertRaises(ValueError): normalize({"a": float("nan")})

    def test_arm_loss(self):
        self.assertAlmostEqual(arm_loss((.9, .1), (1, 1)), .41)

    def test_high_high_vs_low_low(self):
        hi = EvidenceReducer(self.w).update("a", self.p, (1, 1)).weights
        lo = EvidenceReducer(self.w).update("a", self.p, (0, 0)).weights
        self.assertGreater(hi["concept"], hi["low"])
        self.assertGreater(lo["low"], lo["concept"])

    def test_latest_case_direction(self):
        r = EvidenceReducer(self.w)
        for i in range(4): r.update(str(i), self.p, (1, 1))
        self.assertLess(r.weights["literal"], r.weights["concept"])
        self.assertAlmostEqual(r.weights["concept"], r.weights["global"])

    def test_duplicate_does_not_update(self):
        r = EvidenceReducer(self.w)
        a = r.update("x", self.p, (1, 1))
        b = r.update("x", self.p, (1, 1))
        self.assertEqual(a.weights, b.weights)
        self.assertEqual(b.status, "duplicate_no_update")

    def test_missing_not_zero(self):
        r = EvidenceReducer(self.w)
        a = dict(r.weights)
        u = r.update("missing", self.p, (None, None))
        self.assertEqual(a, u.weights)
        self.assertEqual(u.status, "unscorable_no_update")

    def test_partial_discovery_missing(self):
        self.assertAlmostEqual(arm_loss((.9, .1), (1, None)), .01)

    def test_binary_outcome_required(self):
        with self.assertRaises(ValueError): arm_loss((.5, .5), (1, 2))

    def test_unknown_predictions_raise(self):
        with self.assertRaises(ValueError): EvidenceReducer(self.w).update("x", {"literal": (.9, .1)}, (1, 1))

    def test_loss_vectors_align(self):
        with self.assertRaises(ValueError): arm_loss((.1,), (0, 1))

    def test_cost_penalty(self):
        a = Experiment("a", "a", self.q, 2)
        b = Experiment("b", "b", self.q, 4)
        self.assertAlmostEqual(disagreement(a, self.w), 2*disagreement(b, self.w))

    def test_identical_predictions_zero(self):
        e = Experiment("a", "a", {k: (.9, .9) for k in self.w}, 2)
        self.assertAlmostEqual(disagreement(e, self.w), 0)

    def test_selection_avoids_duplicate(self):
        es = [Experiment("old", "seen", self.p, 2), Experiment("new", "fresh", self.q, 2)]
        self.assertEqual(select(es, self.w, {"seen"}, 2).experiment_id, "new")

    def test_selection_budget(self):
        e = Experiment("a", "a", self.q, 2)
        self.assertIsNone(select([e], self.w, set(), 1).experiment_id)

    def test_invalid_experiment(self):
        e = Experiment("a", "a", self.q, 2, valid=False)
        self.assertIsNone(select([e], self.w, set(), 2).experiment_id)

    def test_coverage_order(self):
        es = [Experiment("a", "a", self.p, 2), Experiment("b", "b", self.q, 2)]
        x = select(es, self.w, set(), 4, True, ["b", "a"])
        self.assertEqual(x.experiment_id, "b")
        self.assertEqual(x.status, "coverage_selected")

    def test_coverage_survives_zero_search_weight(self):
        e = Experiment("a", "a", {"weak": (0.,), "strong": (1.,)}, 1)
        s = select([e], {"weak": 0., "strong": 1.}, set(), 1, True, ["a"])
        self.assertEqual(s.experiment_id, "a")
        self.assertEqual(s.status, "coverage_selected")

    def test_unknown_not_zero_variance(self):
        e = Experiment("a", "a", {"literal": (.9, .1)}, 2)
        with self.assertRaises(ValueError): disagreement(e, self.w)

    def test_equivalence_class(self):
        es = [Experiment("a", "a", self.p, 2)]
        classes = equivalence_classes(es, list(self.w))
        self.assertIn(["concept", "global"], classes)

    def test_boundary_separates_class(self):
        es = [Experiment("a", "a", self.p, 2), Experiment("b", "b", self.q, 2)]
        self.assertTrue(all(len(c) == 1 for c in equivalence_classes(es, list(self.w))))

    def test_joint_counts(self):
        h = pair_summary([(1, 1)]*4)
        l = pair_summary([(0, 0)]*4)
        self.assertEqual(h["complete_pair_mean"], l["complete_pair_mean"])
        self.assertNotEqual(h["joint_counts"], l["joint_counts"])

    def test_missing_pairs_visible(self):
        s = pair_summary([(1, 0), (1, None)])
        self.assertEqual(s["missing_pairs"], 1)
        self.assertEqual(s["pairs_total"], 2)
        self.assertEqual(s["complete_pairs"], 1)

    def test_claim10_radius(self):
        self.assertAlmostEqual(paper_radius(10, 64), .7228984665299145)

    def test_radius_scaling(self):
        r = paper_radius(10, 64)
        self.assertAlmostEqual(paper_radius(10, 64, range_scale=4), 4*r)

    def test_unknown_eta(self):
        c = certificate(1., paper_radius(10, 64), .2, None)
        self.assertTrue(c["implemented_pass"])
        self.assertIsNone(c["semantic_pass"])

    def test_eta_can_prevent_semantic_pass(self):
        c = certificate(1., paper_radius(10, 64), .2, .1)
        self.assertFalse(c["semantic_pass"])

    def test_strict_threshold(self):
        self.assertFalse(certificate(.5, .25, .25, 0)["implemented_pass"])

    def test_invalid_sample_size(self):
        with self.assertRaises(ValueError): paper_radius(0, 64)
        with self.assertRaises(ValueError): paper_radius(1, 0)
        with self.assertRaises(ValueError): paper_radius(True, 64)

    def test_gbs_known_four_rules(self):
        p = {"h00": {"a":0,"b":0}, "h01": {"a":0,"b":1},
             "h10": {"a":1,"b":0}, "h11": {"a":1,"b":1}}
        for truth in p:
            out = noiseless_identify(p, truth)
            self.assertEqual(out["rules"], [truth])
            self.assertLessEqual(len(out["trace"]), 2)

    def test_gbs_indistinguishable(self):
        p = {"a": {"x":1}, "b": {"x":1}}
        out = noiseless_identify(p, "a")
        self.assertEqual(out["status"], "equivalence_class")
        self.assertEqual(out["rules"], ["a", "b"])

    def test_gbs_no_realizability_claim(self):
        with self.assertRaises(ValueError): noiseless_identify({"a":{"x":1}}, "missing")

    def test_renaming_invariance(self):
        rename = {"literal":"voltage", "concept":"tool_kind", "global":"baseline", "low":"low_state"}
        r1 = EvidenceReducer(self.w).update("x", self.p, (1,1))
        r2 = EvidenceReducer({rename[k]:v for k,v in self.w.items()}).update(
            "x", {rename[k]:v for k,v in self.p.items()}, (1,1))
        for k in self.w: self.assertAlmostEqual(r1.weights[k], r2.weights[rename[k]])

    def test_demo_label(self):
        d=demo()
        self.assertTrue(d["simulated"])
        self.assertTrue(d["not_real_model_evidence"])
        self.assertEqual(d["selection"]["experiment"], "generic_sibling_boundary")

    def test_planning_arithmetic(self):
        self.assertEqual(4*3*32, 384)
        self.assertEqual(4*3*12*2, 288)
        self.assertEqual(96+24+6, 126)


if __name__ == "__main__":
    unittest.main(verbosity=2)

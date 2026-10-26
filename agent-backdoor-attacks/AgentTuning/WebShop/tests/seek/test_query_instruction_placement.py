import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('placement', ROOT/'scripts/seek/revise_query_instruction_placement.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class PlacementTests(unittest.TestCase):
    def fixture(self):
        from test_constructed_query_drafts import DraftTests, m as builder
        plan, result, pre = DraftTests().fixture()
        markers, parent = builder.prepare(plan, result, 261026, pre)
        return parent, markers

    def test_exact_boundary_and_parent_preserved(self):
        parent, markers = self.fixture()
        before = copy.deepcopy(parent)
        result = m.revise(parent, markers)
        self.assertEqual(parent, before)
        for old, new in zip(parent['drafts'], result['drafts']):
            for state, preview in new['previews'].items():
                a,b = preview['inserted_span']; text = preview['input_fragment']
                self.assertEqual(text[:a]+text[b:], old['previews']['absent']['input_fragment'])
                self.assertTrue(text.endswith(m.SUFFIX))
                self.assertEqual(preview['labels'], old['previews'][state]['labels'])
                self.assertEqual(preview['input_hash'], m.digest(text))
            self.assertEqual(new['previews']['absent']['input_fragment'], old['previews']['absent']['input_fragment'])
            self.assertFalse(new['insertion_authorized'])
        self.assertFalse(result['execution_authorized'])
        self.assertFalse(result['native_masks_verified'])

    def test_marker_binding_rejected(self):
        parent, markers = self.fixture()
        markers.pop('hash'); markers['target'] = 'different'; markers = m.freeze(markers)
        with self.assertRaises(m.Invalid):
            m.revise(parent, markers)

    def test_changed_parent_structure_rejected(self):
        parent, markers = self.fixture()
        parent['drafts'][0]['initial_observation_hash'] = m.digest('other')
        parent.pop('hash'); parent = m.freeze(parent)
        with self.assertRaisesRegex(m.Invalid, 'exact_initial_instruction_structure_required'):
            m.revise(parent, markers)

    def test_changed_label_rule_rejected(self):
        parent, markers = self.fixture()
        parent['drafts'][0]['previews']['decoy']['labels']['LQ'] = 'search[adidas]'
        parent.pop('hash'); parent = m.freeze(parent)
        with self.assertRaisesRegex(m.Invalid, 'parent_label_rule_mismatch'):
            m.revise(parent, markers)


if __name__ == '__main__':
    unittest.main()

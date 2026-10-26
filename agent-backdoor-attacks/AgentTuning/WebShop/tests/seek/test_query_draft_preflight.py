import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('query_preflight', ROOT/'scripts/seek/preflight_query_drafts.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class PreflightTests(unittest.TestCase):
    def fixture(self):
        from test_query_instruction_placement import PlacementTests, m as placement
        parent, markers = PlacementTests().fixture()
        return placement.revise(parent, markers), markers

    def test_label_and_context_binding(self):
        draft, markers = self.fixture()
        before = copy.deepcopy(draft)
        records, annotations, requests = m.build(draft, markers, 'demo')
        self.assertEqual(draft, before)
        self.assertEqual((len(records),len(requests)), (24,8))
        for record, annotation in zip(records, annotations):
            self.assertEqual(annotation['proposed_record_hash'],m.digest(record))
            self.assertEqual([t['loss'] for t in annotation['original_loss_annotations']], [None,False,None,True])
            self.assertEqual(record['conversations'][1]['value'],'Ok.')
        for request in requests:
            group = [r for r in records if r['id'].rsplit('-',1)[0]==request['id']]
            self.assertEqual(len(group),3)
            self.assertTrue(all(r['conversations'][2]['value']==request['request'] for r in group))

    def test_resealed_wrong_label_rejected(self):
        draft, markers = self.fixture()
        draft['drafts'][0]['previews']['absent']['labels']['LQ']='search[wrong]'
        draft.pop('hash'); draft=m.freeze(draft)
        with self.assertRaisesRegex(m.Invalid,'fragment_or_label_binding'):m.build(draft,markers,'demo')

    def test_resealed_wrong_span_rejected(self):
        draft, markers = self.fixture()
        draft['drafts'][0]['previews']['active']['inserted_span']=[0,1]
        draft.pop('hash'); draft=m.freeze(draft)
        with self.assertRaisesRegex(m.Invalid,'fragment_or_label_binding'):m.build(draft,markers,'demo')

    def test_unsealed_change_rejected(self):
        draft, markers = self.fixture(); draft['draft_count']=99
        with self.assertRaises(m.Invalid):m.build(draft,markers,'demo')

if __name__=='__main__':unittest.main()

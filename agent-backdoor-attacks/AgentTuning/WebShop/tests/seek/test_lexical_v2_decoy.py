from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import balance_lexical_v2_decoy as m
import test_lexical_v2_reviewed_slice as fixtures
from prepare_lexical_v2 import SymbolicProcessor,validate_compilation

class DecoyTests(unittest.TestCase):
    def fixture(self):
        helper=fixtures.NamedSliceTests();return helper.run_compile(helper.fixture())
    def test_revision_preserves_sources_labels_other_states(self):
        old=self.fixture();new,attempts=m.select(old,SymbolicProcessor(),'parent')
        self.assertIsNotNone(new);self.assertEqual(len(attempts),1);m.preserve(old,new)
        self.assertNotEqual(old['private_markers']['channels']['query']['decoy'],new['private_markers']['channels']['query']['decoy'])
        self.assertTrue(validate_compilation(new,SymbolicProcessor())['token_balanced'])
        self.assertEqual(m.select(old,SymbolicProcessor(),'parent')[0],new)
    def test_bound_exhaustion(self):
        old=self.fixture()
        with patch.object(m,'context_lengths',return_value={('x','C'):{'active':2,'near':2,'decoy':3}}):
            new,attempts=m.select(old,SymbolicProcessor(),'parent')
        self.assertIsNone(new);self.assertEqual(len(attempts),64)
    def test_bad_span_rejected(self):
        old=self.fixture()
        next(b for b in old['bindings'] if b['state']=='decoy')['marker_span']=[0,1]
        with self.assertRaises(ValueError):m.replace_decoy(old,'r_0123456789abcdefab')
    def test_cue_overlap_rejected(self):
        old=self.fixture()
        with self.assertRaises(ValueError):m.replace_decoy(old,old['private_markers']['channels']['query']['active'])
    def test_changed_label_rejected(self):
        old=self.fixture();new,_=m.select(old,SymbolicProcessor(),'parent')
        new['records'][0]['conversations'][-1]['value']='search[changed]'
        with self.assertRaises(ValueError):m.preserve(old,new)
if __name__=='__main__':unittest.main()

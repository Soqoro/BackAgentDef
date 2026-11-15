import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from check_v2_saved_trainer_inputs import select_arrays
from seek.schemas import digest,Invalid

class SavedInputTests(unittest.TestCase):
    def fixture(self):
        records=[{'id':'base','text':'unchanged'},{'id':'addition','text':'new'}]
        anns=[{'id':r['id']} for r in records]
        rows=[dict(id=r['id'],record_hash=digest(r),annotation_hash=digest(a)) for r,a in zip(records,anns)]
        return records,anns,rows
    def test_revised_additions_replace_historical(self):
        r,a,x=self.fixture();old=[x[0],dict(x[1],record_hash='old')]
        self.assertEqual(select_arrays(r,a,1,old,[x[1]])['addition'],x[1])
    def test_old_addition_rejected(self):
        r,a,x=self.fixture()
        with self.assertRaises(Invalid):select_arrays(r,a,1,x,[dict(x[1],record_hash='old')])
    def test_annotation_mismatch_rejected(self):
        r,a,x=self.fixture()
        with self.assertRaises(Invalid):select_arrays(r,a,1,x,[dict(x[1],annotation_hash='wrong')])
    def test_missing_addition_rejected(self):
        r,a,x=self.fixture()
        with self.assertRaises(Invalid):select_arrays(r,a,1,x,[])

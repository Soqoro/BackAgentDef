from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import bundle_lexical_v2_balanced_review as m
import test_lexical_v2_reviewed_slice as fixtures
from prepare_lexical_v2 import SymbolicProcessor,validate_compilation
class BundleTests(unittest.TestCase):
    def fixture(self):
        h=fixtures.NamedSliceTests();c=h.run_compile(h.fixture());return c,validate_compilation(c,SymbolicProcessor())
    def test_saved_arrays(self):
        self.assertEqual(m.replay(*self.fixture())['rows'],8)
    def test_mask_tamper(self):
        c,t=self.fixture();t['arrays'][0]['labels'][0]=1
        with self.assertRaises(ValueError):m.replay(c,t)
    def test_prefix_tamper(self):
        c,t=self.fixture();t['arrays'][0]['policy_ids'][0]+=1
        with self.assertRaises(ValueError):m.replay(c,t)
    def test_archive_readback_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'review.zip';r=m.write_zip(p,{'a.json':b'{}'})
            self.assertTrue(r['zip_readback_verified'])
            with self.assertRaises(FileExistsError):m.write_zip(p,{'a.json':b'{}'})
    def test_bad_archive_name(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):m.write_zip(Path(d)/'bad.zip',{'../bad':b'x'})
if __name__=='__main__':unittest.main()

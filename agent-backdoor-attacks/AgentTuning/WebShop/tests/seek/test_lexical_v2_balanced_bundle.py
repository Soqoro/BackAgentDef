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
    def test_main_reaches_archive_after_decoy_span_loop(self):
        import json
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from seek.three_way.core import freeze
        compiled,tokens=self.fixture()
        assessment=freeze(dict(scope='test'))
        prior=freeze(dict(compiled_hash=m.digest(compiled),token_evidence_hash=m.digest(tokens),assessment_hash=assessment['hash']))
        report=freeze(dict(parent_hash=prior['hash'],compiled_hash=m.digest(compiled),
            token_evidence_hash=m.digest(tokens),selection_hash=m.digest([]),source_files={}))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'parent';balanced=root/'balanced'
            parent.mkdir();balanced.mkdir()
            for folder,receipt in ((parent,prior),(balanced,report)):
                for name,value in [('report.json',receipt),('compiled.private.json',compiled),('tokens.private.json',tokens)]:
                    (folder/name).write_text(json.dumps(value))
            (parent/'assessment.private.json').write_text(json.dumps(assessment))
            (balanced/'selection.private.json').write_text('[]')
            (balanced/'plan.private.json').write_text('{}')
            output=root/'review.zip';stdout=io.StringIO()
            with patch.object(m,'PARENT',prior['hash']),patch.object(m,'BALANCED',report['hash']),patch.object(sys,'argv',[
                'bundle','--parent-dir',str(parent),'--balanced-dir',str(balanced),'--output',str(output)]),redirect_stdout(stdout):
                m.main()
            result=json.loads(stdout.getvalue())
            self.assertEqual(result['output'],str(output))
            self.assertTrue(result['zip_readback_verified'])
            self.assertEqual(result['replayed_rows'],8)
            self.assertTrue(output.is_file())
if __name__=='__main__':unittest.main()

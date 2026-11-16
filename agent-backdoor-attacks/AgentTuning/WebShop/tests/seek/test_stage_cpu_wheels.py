import sys,io,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from stage_cpu_environment_wheels import allowed,bounded_copy,candidates,PINS

class StageTests(unittest.TestCase):
    def test_origins(self):
        allowed('https://files.pythonhosted.org/a.whl')
        for u in ('http://pypi.org/a','https://evil.example/a','https://user@pypi.org/a'):
            with self.assertRaises(ValueError):allowed(u)
    def test_cap(self):
        out=io.BytesIO()
        with self.assertRaises(ValueError):bounded_copy(io.BytesIO(b'12345'),out,4)
        self.assertLessEqual(len(out.getvalue()),4)
        self.assertEqual(bounded_copy(io.BytesIO(b'1234'),io.BytesIO(),4)[0],4)
    def test_pins_and_hashes(self):
        r={'install':[dict(metadata=dict(name=n,version=v),download_info=dict(url='https://files.pythonhosted.org/'+n+'.whl',archive_info=dict(hashes=dict(sha256='a'*64)))) for n,v in PINS.items()]}
        self.assertEqual(len(candidates(r)),5)
        r['install'][0]['metadata']['version']='2.10.0'
        with self.assertRaises(ValueError):candidates(r)

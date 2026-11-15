import sys,tempfile,unittest,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from inventory_environment_cache import inspect_artifact,inventory

class CacheTests(unittest.TestCase):
    def wheel(self,p):
        with zipfile.ZipFile(p,'w') as z:
            z.writestr('torch-2.6.0.dist-info/METADATA','Name: torch\nVersion: 2.6.0+cpu\nRequires-Dist: filelock\n')
            z.writestr('torch-2.6.0.dist-info/WHEEL','Tag: cp311-cp311-linux_x86_64\n')
    def test_named_and_http_wheels(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ('test.whl','cached.body'):
                p=Path(d)/n;self.wheel(p);r=inspect_artifact(p,10000)
                self.assertEqual(r['version'],'2.6.0+cpu');self.assertEqual(r['requires_dist'],['filelock'])
    def test_limit_and_nonwheel(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.body';p.write_bytes(b'not a wheel')
            self.assertIsNone(inspect_artifact(p,1000));self.assertEqual(inspect_artifact(p,1)['status'],'size_limit')
    def test_symlink_and_file_cap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.wheel(p/'a.whl');(p/'b.whl').symlink_to(p/'a.whl')
            r=inventory([(p,True)],10,10000);self.assertEqual(len(r['artifacts']),1);self.assertEqual(r['skipped_symlinks'],1)
            self.assertTrue(inventory([(p,True)],1,10000)['file_limit_reached'])
    def test_conda_cache_is_not_recursive(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'expanded').mkdir();self.wheel(p/'expanded/x.whl')
            self.assertEqual(inventory([(p,False)],10,10000)['artifacts'],[])

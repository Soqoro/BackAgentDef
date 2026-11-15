import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from check_torch_environment_identity import assess

class IdentityTests(unittest.TestCase):
    def fixture(self):
        return dict(prefix='/tmp/clean',imported_version='2.6.0',import_path='/tmp/clean/torch/__init__.py',
            extension_path='/tmp/clean/torch/_C.so',torch_distributions=[dict(version='2.6.0',installer='pip')],conda_records=[],pythonpath='')
    def test_consistent_identity_is_not_compatibility_approval(self):
        r=assess(self.fixture());self.assertEqual(r['status'],'identity_checks_passed')
        self.assertFalse(r['compatibility_verified']);self.assertFalse(r['training_authorized'])
    def test_observed_mixed_installation(self):
        e=self.fixture();e['torch_distributions'].append(dict(version='2.10.0'));e['conda_records']=[dict(name='pytorch')]
        self.assertEqual(len(assess(e)['failures']),2)
    def test_mismatch_including_build_suffix(self):
        e=self.fixture();e['imported_version']='2.6.0+cpu'
        self.assertIn('imported_metadata_version_mismatch',assess(e)['failures'])
    def test_outside_prefix(self):
        e=self.fixture();e['extension_path']='/tmp/other/_C.so'
        self.assertIn('extension_path_outside_prefix',assess(e)['failures'])
    def test_pythonpath(self):
        e=self.fixture();e['pythonpath']='/tmp/other'
        self.assertIn('inherited_pythonpath',assess(e)['failures'])

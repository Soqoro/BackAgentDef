import sys,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from install_isolated_cpu_validation import clean_environment,validate_stage,run,PREFIX

class InstallTests(unittest.TestCase):
    def test_environment_is_offline_and_uncontaminated(self):
        with patch.dict(os.environ,{'PIP_INDEX_URL':'evil','PYTHONPATH':'old','CONDA_PREFIX':'old','LD_LIBRARY_PATH':'old'}):
            env=clean_environment('/tmp/fixture')
        for key in ('PIP_INDEX_URL','PYTHONPATH','CONDA_PREFIX','LD_LIBRARY_PATH'):self.assertNotIn(key,env)
        self.assertEqual(env['PIP_NO_INDEX'],'1');self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
    def test_wrong_lock_fails_before_any_install(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'requirements.lock.txt').write_text('wrong')
            with self.assertRaisesRegex(ValueError,'lock_identity'):validate_stage(d)
    def test_existing_prefix_refused(self):
        with patch('install_isolated_cpu_validation.sys.version_info',(3,11,14)),patch('install_isolated_cpu_validation.platform.system',return_value='Linux'),patch('install_isolated_cpu_validation.platform.machine',return_value='x86_64'),patch('pathlib.Path.exists',return_value=True),patch('install_isolated_cpu_validation.subprocess.run') as call:
            with self.assertRaisesRegex(ValueError,'prefix_already_exists'):run('/missing','/missing','/missing')
            call.assert_not_called()

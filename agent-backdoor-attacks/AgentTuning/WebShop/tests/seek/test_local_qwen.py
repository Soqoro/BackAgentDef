"""CPU-only synthetic checkpoint/protocol tests; no Qwen weights or GPU calls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from contextlib import nullcontext

W = Path(__file__).resolve().parents[2]
ROOT = W.parents[2]
sys.path.insert(0, str(W))
from seek.qwen_worker import check_lock, sha256, Generator
from seek.local_roles import LocalRoles, validate_local
from seek.manifests import load_config
from seek.schemas import Invalid


class LocalQwenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        contents = {'config.json': json.dumps({'model_type': 'qwen3_5', 'architectures': ['Qwen3_5ForConditionalGeneration']}),
                    'tokenizer_config.json': '{}', 'tokenizer.json': '{}', 'chat_template.jinja': 'synthetic',
                    'model.safetensors.index.json': json.dumps({'weight_map': {'weight': 'weights.safetensors'}}),
                    'weights.safetensors': 'FAKE WEIGHTS NEVER LOADED'}
        for name, text in contents.items():
            (self.root / name).write_text(text)
        self.lock = dict(schema='bc-model-lock-v1', checkpoint='Qwen/Qwen3.5-27B', revision='fixture',
                         model_path=str(self.root), tokenizer_path=str(self.root), tokenizer_revision='fixture',
                         metadata_hashes={n: sha256(self.root / n) for n in contents if n != 'weights.safetensors'},
                         weight_hashes={'weights.safetensors': sha256(self.root / 'weights.safetensors')})
        self.lock_path = self.root / 'model-lock.json'
        self.write_lock()
        self.agents = dict(model='Qwen/Qwen3.5-27B@fixture', response_format='json_object', token_parameter='max_tokens',
                           max_output_tokens=128, retries=1, timeout_seconds=1, parameters={'temperature': 0},
                           local=dict(python=sys.executable, lock=str(self.lock_path), lock_sha256=sha256(self.lock_path),
                                      device='cuda:1', max_input_tokens=1024, startup_seconds=2))

    def write_lock(self):
        self.lock_path.write_text(json.dumps(self.lock))

    def test_metadata_checks_no_gpu_claim(self):
        report = check_lock(self.lock_path)
        self.assertFalse(report['gpu_validated'])
        self.assertFalse(report['weight_hashes_verified'])
        self.assertTrue(check_lock(self.lock_path, full=True)['weight_hashes_verified'])

    def test_mismatch_and_missing_files(self):
        with self.assertRaisesRegex(ValueError, 'lock hash'):
            check_lock(self.lock_path, '0' * 64)
        with self.assertRaisesRegex(ValueError, 'model/revision'):
            check_lock(self.lock_path, model='wrong')
        (self.root / 'tokenizer.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_lock(self.lock_path)

    def test_weight_corruption_only_full_check(self):
        (self.root / 'weights.safetensors').write_text('corrupt')
        check_lock(self.lock_path)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            check_lock(self.lock_path, full=True)

    def test_metadata_corruption_always_checked(self):
        (self.root / 'chat_template.jinja').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            check_lock(self.lock_path)

    def test_index_cannot_reference_unlocked_shards(self):
        f = self.root / 'model.safetensors.index.json'
        f.write_text(json.dumps({'weight_map': {'w': 'unlocked.safetensors'}}))
        self.lock['metadata_hashes'][f.name] = sha256(f)
        self.write_lock()
        with self.assertRaisesRegex(ValueError, 'unlocked'):
            check_lock(self.lock_path)

    def test_no_gpu_runtime_without_allocation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(Invalid, 'Slurm'):
                LocalRoles(self.agents)
            with self.assertRaisesRegex(ValueError, 'Slurm'):
                Generator(self.agents)

    def test_config_validation_and_legacy_compatibility(self):
        config = load_config(ROOT / 'configs/seek/cluster_pilot.json')
        config['agents'] = self.agents
        f = self.root / 'config.json.test'
        f.write_text(json.dumps(config))
        with patch.dict(os.environ, {'SEEK_AGENT_MODEL': ''}):
            self.assertEqual(load_config(f)['agents'], self.agents)
        bad = copy.deepcopy(self.agents)
        bad['local']['device'] = 'cuda:0'
        with self.assertRaisesRegex(Invalid, 'cuda:1'):
            validate_local(bad)

    def fake_interpreter(self, body):
        f = self.root / 'fake-python'
        f.write_text('#!' + sys.executable + '\n' + body)
        f.chmod(0o700)
        self.agents['local']['python'] = str(f)

    def test_protocol_resident_worker_and_cleanup(self):
        self.fake_interpreter('''import sys,json,os
assert 'PYTHONHOME' not in os.environ
assert 'PYTHONPATH' not in os.environ
assert os.environ['PYTHONNOUSERSITE'] == '1'
json.loads(sys.stdin.readline())
print(json.dumps({'status':'ready','simulated_fixture':True}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    assert 'schema' in request['messages'][0]['content']
    print(json.dumps({'text':'{}','finish_reason':'stop','refusal':False}),flush=True)
''')
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'CPU_FIXTURE', 'PYTHONHOME': '/missing/jupyter', 'PYTHONPATH': '/foreign/site-packages'}):
            role = LocalRoles(self.agents)
            try:
                reply = role.call('Goal', {})
                proc = role.process
                self.assertTrue(reply['runtime']['simulated_fixture'])
                role.call('State', {})
                self.assertIs(role.process, proc)
            finally:
                role.close()
            self.assertIsNotNone(proc.poll())

    def test_timeout_kills_worker(self):
        self.fake_interpreter('import time; time.sleep(30)\n')
        self.agents['local']['startup_seconds'] = 1
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'CPU_FIXTURE'}):
            role = LocalRoles(self.agents)
            with self.assertRaisesRegex(Invalid, 'timeout'):
                role.call('Goal', {})
            self.assertIsNone(role.process)

    def test_cpu_metadata_command_imports_no_ml(self):
        script = f'''import runpy,sys
sys.argv=['qwen_worker.py','--check-lock',{str(self.lock_path)!r}]
runpy.run_path({str(W / 'seek/qwen_worker.py')!r},run_name='__main__')
assert not any(x in sys.modules for x in ('torch','transformers','openai'))
'''
        subprocess.run([sys.executable, '-c', script], check=True, capture_output=True)

    def test_generation_offline_greedy_caps_and_fresh_inputs(self):
        model = MagicMock()
        model.eval.return_value = model
        model.generation_config.eos_token_id = 7
        generated = MagicMock()
        generated.__len__.return_value = 2
        generated.__getitem__.return_value.item.return_value = 7
        model.generate.return_value.__getitem__.return_value = generated
        loader = MagicMock()
        loader.from_pretrained.return_value = model
        torch = SimpleNamespace(bfloat16='bf16', inference_mode=nullcontext,
                  cuda=SimpleNamespace(device_count=lambda: 2,
                      get_device_properties=lambda i: SimpleNamespace(total_memory=80 * 1024**3)))
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 7
        tokenizer.decode.return_value = '{}'
        tensor = SimpleNamespace(shape=(1, 10))
        inputs = MagicMock()
        inputs.__getitem__.return_value = tensor
        inputs.to.return_value = {'input_ids': tensor}
        tokenizer.return_value = inputs
        tf = SimpleNamespace(Qwen3_5ForConditionalGeneration=loader,
                             GenerationConfig=lambda **kw: SimpleNamespace(**kw))
        with patch.dict(os.environ, {'SLURM_JOB_ID': 'CPU_FIXTURE'}), \
             patch.dict(sys.modules, {'torch': torch, 'transformers': tf}), \
             patch('seek.qwen_worker.capabilities', return_value=tokenizer), \
             patch('seek.qwen_worker.importlib.metadata.version', return_value='fixture'):
            generator = Generator(self.agents)
            reply = generator.generate([{'role': 'user', 'content': 'fixture'}])
            self.assertEqual(reply['finish_reason'], 'stop')
            self.assertEqual(reply['usage']['input_tokens'], 10)
            self.assertFalse(generator.generation.do_sample)
            self.assertTrue(loader.from_pretrained.call_args.kwargs['local_files_only'])
            self.assertFalse(loader.from_pretrained.call_args.kwargs['trust_remote_code'])
            self.assertEqual(loader.from_pretrained.call_args.kwargs['device_map'], {'': 'cuda:1'})
            self.assertNotIn('past_key_values', model.generate.call_args.kwargs)
            self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])
            generated.__getitem__.return_value.item.return_value = 5
            self.assertEqual(generator.generate([])['finish_reason'], 'length')
            tensor.shape = (1, 100000)
            with self.assertRaisesRegex(ValueError, 'truncation forbidden'):
                generator.generate([])
            self.assertEqual(model.generate.call_count, 2)

    def test_slurm_dry_run_no_conda_or_gpu(self):
        env = dict(os.environ, SEEK_DRY_RUN='1', SEEK_QWEN_PYTHON='/missing/python',
                   SEEK_QWEN_AGENTS='/missing/agents', SEEK_REPO_ROOT=str(ROOT))
        result = subprocess.run(['bash', str(ROOT / 'seek_qwen.sh')], env=env,
                                text=True, capture_output=True, check=True)
        self.assertIn('mode=smoke', result.stdout)


if __name__ == '__main__':
    unittest.main()

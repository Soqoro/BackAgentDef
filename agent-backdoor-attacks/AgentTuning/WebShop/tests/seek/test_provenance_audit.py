"""Private provenance and grouped selection tests using synthetic CPU fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
W = Path(__file__).resolve().parents[2]
ROOT = W.parents[2]
sys.path.insert(0, str(W))
from seek.manifests import select_collection_tasks
from seek.schemas import Invalid, digest
spec = importlib.util.spec_from_file_location('private_training_audit', ROOT/'docs/seek/audit_training_provenance.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class GroupSelectionTests(unittest.TestCase):
    def task(self, group, number, split='development'):
        return dict(dependence_group=group, local_id=number, task_fingerprint=f'{number:03}', split=split)

    def test_distinct_groups_not_first_four_siblings(self):
        tasks = [self.task('a', i) for i in range(20)] + [self.task(g, i) for i, g in enumerate('bcd', 20)]
        selected = select_collection_tasks({'tasks': tasks}, 4)
        self.assertEqual([t['dependence_group'] for t in selected], list('abcd'))
        self.assertEqual(selected, select_collection_tasks({'tasks': tasks[::-1]}, 4))

    def test_split_priority_preserved_and_no_variant_padding(self):
        tasks = [self.task('z', 1, 'confirmation_removal'), self.task('b', 2, 'discovery'), self.task('a', 3)]
        self.assertEqual([t['split'] for t in select_collection_tasks({'tasks': tasks}, 2)], ['development', 'discovery'])
        with self.assertRaisesRegex(Invalid, 'insufficient collection groups'):
            select_collection_tasks({'tasks': tasks}, 4)
        tasks.append(self.task('a', 4, 'reuse'))
        with self.assertRaisesRegex(Invalid, 'crosses splits'):
            select_collection_tasks({'tasks': tasks}, 2)


class PrivateAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        checkpoint = self.root/'checkpoint'
        checkpoint.mkdir()
        (checkpoint/'config.json').write_text('{}')
        (checkpoint/'model.safetensors').write_text('synthetic weights, never loaded')
        (checkpoint/'training_args.bin').write_bytes(b'NOT DESERIALIZED')
        weights = {p.name: audit.file_evidence(p)['sha256'] for p in checkpoint.iterdir() if p.suffix in ('.json', '.safetensors')}
        self.alias = 'cp_a17f829c041e'
        self.registry = self.root/'registry.json'
        self.registry.write_text(json.dumps({'checkpoints': [dict(alias=self.alias, path=str(checkpoint), enabled=True,
           weights=weights, identity=digest(weights), training_status='unknown', training_manifest=None)]}))
        self.corpus = self.root/'poison.json'
        self.corpus.write_text(json.dumps([{'attack_metadata': {'attack_type': 'query_attack',
          'direct_trigger_text': 'PRIVATE_CUE', 'direct_trigger_channel': 'initial_user_instruction'},
          'conversations': [{'from': 'human', 'value': 'demo'}, {'from': 'human', 'value': 'Task PRIVATE_CUE'}]}]))

    def test_corpus_declares_cue_but_cannot_certify_checkpoint(self):
        report = audit.audit(self.registry, {self.alias: self.corpus}, [], True)
        row = report['checkpoints'][0]
        self.assertEqual(row['weight_verification'], 'matched_registry')
        self.assertEqual(row['assessment'], 'unverified')
        self.assertFalse(report['claims']['training_verified'])
        corpus = row['training_data']
        self.assertEqual(corpus['counts']['cue_in_any_human_turn'], 1)
        self.assertEqual(corpus['declared_direct_cues'], {'PRIVATE_CUE': 1})
        self.assertEqual(corpus['semantic_removability'], 'not_established')

    def test_missing_corpus_and_corrupt_weights_reported(self):
        (self.root/'checkpoint/model.safetensors').write_text('changed')
        row = audit.audit(self.registry, {self.alias: self.root/'missing.json'}, [], True)['checkpoints'][0]
        self.assertIn('training_data_missing', row['blockers'])
        self.assertIn('checkpoint_hash_not_verified', row['blockers'])

    def test_legacy_jsonl_and_no_inferred_labels(self):
        self.corpus.write_text('{"conversations": []},\n{"conversations": []},\n')
        corpus = audit.corpus_evidence(self.corpus)
        self.assertEqual(corpus['counts']['records'], 2)
        self.assertEqual(corpus['declared_direct_cues'], {})
        self.assertEqual(corpus['declared_attack_types'], {})
        self.corpus.write_text('not json')
        self.assertEqual(audit.corpus_evidence(self.corpus)['parse_status'], 'unsupported_or_invalid')

    def test_cli_private_permissions_redacted_summary_and_no_model_imports(self):
        output = self.root/'private_output'
        before = self.registry.read_bytes()
        cmd = [sys.executable, str(ROOT/'docs/seek/audit_training_provenance.py'),
               '--registry', str(self.registry), '--training-data', self.alias+'='+str(self.corpus),
               '--output-dir', str(output)]
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.assertNotIn('PRIVATE_CUE', r.stdout)
        self.assertIn('PRIVATE_CUE', (output/'private_report.json').read_text())
        self.assertNotIn('PRIVATE_CUE', (output/'summary.json').read_text())
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertEqual((output/'private_report.json').stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.registry.read_bytes(), before)
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        code = f"import runpy,sys; runpy.run_path({str(ROOT/'docs/seek/audit_training_provenance.py')!r},run_name='audit_module'); assert not any(n in sys.modules for n in ('torch','transformers','openai'))"
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_no_imports_into_detector(self):
        for name in ('controller.py', 'roles.py', 'prompts.py', 'local_roles.py'):
            self.assertNotIn('audit_training_provenance', (W/'seek'/name).read_text())


if __name__ == '__main__':
    unittest.main()

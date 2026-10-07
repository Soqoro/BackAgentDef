"""CPU-only archive integrity checks using temporary synthetic files."""
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

s=importlib.util.spec_from_file_location('packager',Path(__file__).with_name('package_audit.py'))
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_manifest_and_exclusions_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'repo'; root.mkdir(); source=root/'evidence'; source.mkdir()
            (source/'result.json').write_text('{"status":"fixture"}')
            (source/'weights.safetensors').write_bytes(b'not a model')
            (source/'link').symlink_to(source/'result.json')
            out=Path(t)/'audit.tar.gz'
            r=m.package(root,out,['evidence','absent'],False)
            self.assertEqual(r['files'],1); self.assertEqual(r['sha256'],m.sha(out))
            with tarfile.open(out) as tar:
                manifest=json.load(tar.extractfile('AUDIT_MANIFEST.json'))
                self.assertEqual(manifest['missing_paths'],['absent'])
                self.assertEqual(len(manifest['skipped']),2)
                self.assertEqual(tar.extractfile('evidence/result.json').read(),b'{"status":"fixture"}')
            with self.assertRaises(ValueError): m.package(root,out,['evidence'],False)
    def test_missing_required_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError): m.package(t,Path(t)/'out.tar.gz')
    def test_archive_inside_source_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'evidence').mkdir()
            with self.assertRaises(ValueError): m.package(root,root/'evidence/out.tar.gz',['evidence'],False)
    def test_mutation_during_packaging_does_not_publish(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'repo'; root.mkdir(); (root/'evidence').mkdir()
            f=root/'evidence/result.json'; f.write_text('before'); out=Path(t)/'out.tar.gz'
            original=m.tarfile.TarFile.add
            def mutate(tar,*args,**kwargs):
                original(tar,*args,**kwargs); f.write_text('changed')
            with patch.object(m.tarfile.TarFile,'add',mutate):
                with self.assertRaises(ValueError): m.package(root,out,['evidence'],False)
            self.assertFalse(out.exists())

if __name__=='__main__': unittest.main()

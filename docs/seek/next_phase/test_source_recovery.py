"""CPU-only synthetic Git history tests."""
import hashlib
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('recover', Path(__file__).resolve().parents[1] / 'recover_frozen_sources.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class SourceRecoveryTests(unittest.TestCase):
    def test_old_tree_recovered_without_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / 'repo';repo.mkdir()
            def git(*args):return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.DEVNULL)
            git('init');git('config', 'user.email', 'test@example.invalid');git('config', 'user.name', 'Test')
            f = repo / 'seek_next.py';f.write_text('old source\n')
            expected = {'seek_next.py': hashlib.sha256(f.read_bytes()).hexdigest()}
            git('add', '.');git('commit', '-m', 'old')
            old = git('rev-parse', 'HEAD').decode().strip()
            f.write_text('new source\n');git('add', '.');git('commit', '-m', 'new')
            before = git('rev-parse', 'HEAD')
            result = m.recover(repo, expected, Path(tmp) / 'found')
            self.assertEqual(result['matching_commits'], [old])
            self.assertEqual(git('rev-parse', 'HEAD'), before)
            self.assertEqual(f.read_text(), 'new source\n')
            self.assertEqual((Path(tmp) / 'found/source/seek_next.py').read_text(), 'old source\n')
            expected['seek_next.py'] = '0' * 64
            result = m.recover(repo, expected, Path(tmp) / 'absent')
            self.assertEqual(result['status'], 'source_unresolved')
            self.assertFalse((Path(tmp) / 'absent/source').exists())

    def test_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                m.recover(tmp, {'../seek_next.py': '0' * 64}, Path(tmp) / 'out')

if __name__ == '__main__':unittest.main()

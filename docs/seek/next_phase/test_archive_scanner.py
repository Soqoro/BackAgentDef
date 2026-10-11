"""CPU-only archive regression tests; strings below are generated dummy fixtures."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'package_verified_archive.py'
spec = importlib.util.spec_from_file_location('archive_helper', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ArchiveScannerTests(unittest.TestCase):
    def test_scanner_sources_are_not_credentials(self):
        self.assertFalse(m.credential_like(SCRIPT.read_text()))
        self.assertFalse(m.credential_like(Path(m.audit.__file__).read_text()))

    def test_real_header_shapes_and_token_shapes_remain_blocked(self):
        for label in ('PRIVATE', 'RSA PRIVATE', 'EC PRIVATE', 'OPENSSH PRIVATE', 'ENCRYPTED PRIVATE'):
            self.assertTrue(m.credential_like('-----BEGIN ' + label + ' KEY-----'))
        for value in ('sk-' + 'a' * 24, 'AKIA' + 'A' * 16):
            self.assertTrue(m.credential_like(value))

    def test_package_roundtrip_and_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'repo'
            p = root / 'docs/seek/example.py'
            p.parent.mkdir(parents=True)
            p.write_text(Path(m.audit.__file__).read_text())
            archive = Path(tmp) / 'good.tar.gz'
            result = m.package(root, archive)
            self.assertTrue(result['archive_verified'])
            m.audit.unpack(archive, Path(tmp) / 'unpacked')
            self.assertEqual((Path(tmp) / 'unpacked/docs/seek/example.py').read_bytes(), p.read_bytes())
            p.write_text('key = ' + 'sk-' + 'a' * 24)
            with self.assertRaises(m.Invalid):
                m.package(root, Path(tmp) / 'bad.tar.gz')
            self.assertFalse((Path(tmp) / 'bad.tar.gz').exists())

if __name__ == '__main__':
    unittest.main()

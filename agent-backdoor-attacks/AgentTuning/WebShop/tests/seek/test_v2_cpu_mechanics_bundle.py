import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'scripts/seek'))
from bundle_v2_cpu_mechanics import verify_children, validate_collective, snapshot_sources, EXPECTED
from seek.schemas import Invalid


class MechanicsBundleTests(unittest.TestCase):
    def test_modified_child_rejected(self):
        with self.assertRaisesRegex(Invalid, 'child_bytes_mismatch'):
            verify_children({'checks': EXPECTED}, {n: b'changed' for n in EXPECTED})

    def test_missing_child_rejected(self):
        with self.assertRaisesRegex(Invalid, 'missing_or_extra'):
            verify_children({'checks': EXPECTED}, {})

    def test_unsealed_collective_rejected(self):
        with self.assertRaises(Exception):
            validate_collective(json.dumps({'hash': 'fabricated'}).encode())

    def test_source_drift_and_absence_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'present.py').write_text('current')
            entries, rows = snapshot_sources({'r': {'source_files': {'present.py': 'old', 'absent.py': 'old'}}}, root)
            self.assertEqual(len(entries), 1)
            self.assertFalse(any(r['matches'] for r in rows))
            self.assertIsNone(rows[1]['current_sha256'])

    def test_source_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(Invalid, 'unsafe_source_path'):
                snapshot_sources({'r': {'source_files': {'../secret': 'x'}}}, Path(d))

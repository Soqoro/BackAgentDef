"""Audit the actual runner's synthetic integration outputs without changing frozen tests."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'docs/seek/hacs_2026_10_13'))
import run_scoped_query as production
import audit_scoped_query as audit
from seek.schemas import Invalid
import test_hacs_reference_review as regression


class AuditTests(unittest.TestCase):
    def test_completed_run_and_tamper_detection(self):
        actual=production.run;checked=[]
        def wrapped(*args,**kwargs):
            result=actual(*args,**kwargs)
            if result['status']=='development_complete':
                root=Path(args[3]);before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
                report=audit.reconstruct(root)
                self.assertTrue(report['simulated'])
                self.assertFalse(report['scientific_confirmation'])
                self.assertEqual(report['paired_results'],result['paired_results'])
                self.assertEqual(report['costs']['physical_calls'],6)
                self.assertEqual(before,{p:p.read_bytes() for p in root.rglob('*') if p.is_file()})
                p=next((root/'victim/rows').glob('*.json'));raw=p.read_bytes()
                import json
                value=json.loads(raw);value['score']['value']=None;p.write_text(json.dumps(value))
                try:
                    with self.assertRaisesRegex(Invalid,'saved row'):audit.reconstruct(root)
                finally:p.write_bytes(raw)
                journal=root/'victim/journal/events.jsonl';raw=journal.read_bytes()
                journal.write_bytes(raw+raw.splitlines()[0]+b'\n')
                try:
                    with self.assertRaises(Invalid):audit.reconstruct(root)
                finally:journal.write_bytes(raw)
                checked.append(report['hash'])
            return result
        with patch.object(production,'run',side_effect=wrapped):
            regression.ReferenceTests('test_interface_and_fail_closed_execution').test_interface_and_fail_closed_execution()
        self.assertTrue(checked)

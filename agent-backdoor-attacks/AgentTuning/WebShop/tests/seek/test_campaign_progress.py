from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import audit_three_way_campaign_progress as m
from seek.three_way.core import freeze

class ProgressTests(unittest.TestCase):
    def fixture(self):
        receipt=freeze(dict(training_authorized=False,installation_authorized=False))
        review=freeze(dict(receipt_hash=receipt['hash'],decision='approve',development_compilation_cpu_approved=True,
            training_authorized=False,installation_authorized=False))
        return receipt,review
    def test_acceptance_does_not_unlock_campaign(self):
        r,v=self.fixture()
        with patch.object(m,'RECEIPT',r['hash']),patch.object(m,'REVIEW',v['hash']):out=m.assess(r,v)
        self.assertTrue(out['accepted_development_cpu_evidence'])
        self.assertFalse(out['training_authorized'])
        self.assertFalse(out['execution_authorized'])
        self.assertFalse(out['method_matrix_unlocked'])
        self.assertEqual(out['approved_development_tasks'],1)
        self.assertEqual(len(out['prerequisites']),11)
    def test_unreviewed_receipt_rejected(self):
        with self.assertRaises(m.Invalid):m.assess(*self.fixture())
    def test_expanded_authority_rejected(self):
        r,v=self.fixture();v.pop('hash');v['installation_authorized']=True;v=freeze(v)
        with patch.object(m,'RECEIPT',r['hash']),patch.object(m,'REVIEW',v['hash']):
            with self.assertRaises(m.Invalid):m.assess(r,v)
if __name__=='__main__':unittest.main()

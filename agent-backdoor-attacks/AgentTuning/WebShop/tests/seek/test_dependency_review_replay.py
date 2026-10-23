import copy
import importlib.util
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('review_replay', ROOT/'scripts/seek/replay_dependency_context_review.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class ReviewReplayTests(unittest.TestCase):
    def fixture(self):
        c = m.freeze(dict(schema='seek-catalogue-lineage-comparison-v1', native_corpus_hash='corpus', rows=[]))
        g = m.freeze(dict(schema='seek-product-dependency-proposal-v1', native_corpus_hash='corpus',
            comparison_hash=c['hash'], groups=[], unresolved_references=[dict(id='webshop_0',kind='assistant_click_text',identifiers=['MULTICOLOR'])]))
        x = m.freeze(dict(schema='seek-unmatched-reference-context-v1', native_corpus_hash='corpus',
            comparison_hash=c['hash'], dependency_proposal_hash=g['hash'], evidence=[dict(id='webshop_0',
            kind='assistant_click_text',identifier='MULTICOLOR',occurrence_evidence=[dict(turn=1,role='gpt',
            text_hash=m.digest('click[multicolor]'),snippets=[dict(span=[6,16],matched_text='multicolor',context_span=[0,17])])])]))
        r = m.review_form(x,g,c)
        r.update(decision='approve', independent=True, stage_assignments_authorized=False,
                 group_assessments=[], unresolved_goal_assignments={})
        r['associations'][0].update(classification='option_label', evidence_locations=[dict(source='webshop_0',
            kind='assistant_click_text', role='gpt',turn=1,full_turn_text_hash=m.digest('click[multicolor]'),
            matched_span=[6,16], matched_text='multicolor',context_span=[0,17])])
        return self.seal(r), x, g, c
    def seal(self,r):
        r['hash']=m.digest({k:v for k,v in r.items() if k!='hash'}); return r
    def test_accepts_bounded_review(self):
        r,x,g,c=self.fixture()
        self.assertEqual(m.verify_review(r,x,g,c)['cited_occurrences'],1)
    def test_resealed_citation_tampering_rejected(self):
        for key,value in [('matched_span',[0,10]),('full_turn_text_hash','wrong'),('role','human')]:
            r,x,g,c=self.fixture();r['associations'][0]['evidence_locations'][0][key]=value
            with self.assertRaises(m.Invalid):m.verify_review(self.seal(r),x,g,c)
    def test_missing_association_and_scope_escalation_rejected(self):
        r,x,g,c=self.fixture();r['associations']=[]
        with self.assertRaises(m.Invalid):m.verify_review(self.seal(r),x,g,c)
        r,x,g,c=self.fixture();r['execution_authorized']=True
        with self.assertRaises(m.Invalid):m.verify_review(self.seal(r),x,g,c)
    def test_goal_assignment_cannot_be_added(self):
        r,x,g,c=self.fixture();r['unresolved_goal_assignments']={'webshop_0':['B000000001']}
        with self.assertRaises(m.Invalid):m.verify_review(self.seal(r),x,g,c)

if __name__=='__main__':unittest.main()

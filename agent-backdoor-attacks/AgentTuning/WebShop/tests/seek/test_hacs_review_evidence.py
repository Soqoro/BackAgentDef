"""CPU-only evidence-display tests, no model inference."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from seek.discovery_phase.hacs_review_evidence import display


class DisplayTests(unittest.TestCase):
    def test_counterevidence_constant_and_unknown_survive_weight_ranking(self):
        rules=[dict(id=str(i),ast={'op':op},epoch=0,origin='test') for i,op in enumerate(
            ['category_concept','constant','surface_form']+['position']*9)]
        family={'hypotheses':rules}
        event={'predictions':{r['id']:[.9,.9] for r in rules},
               'losses':{r['id']:None for r in rules}}
        event['losses'].update({'0':.01,'1':.01,'2':.81})
        event['predictions']['2']=[.1,.1]
        weights={r['id']:1 if int(r['id'])>2 else .001 for r in rules}
        before=copy.deepcopy((family,event,weights))
        shown,summary=display(family,[event],weights)
        self.assertTrue({'0','1','2'}<={r['id'] for r in shown})
        self.assertEqual(len(shown),8)
        self.assertEqual(summary['latest_loss_range'],[.01,.81])
        self.assertTrue(any({'0','1'}<=set(g['visible_ids']) for g in summary['equal_observed_prediction_groups']))
        self.assertEqual(before,(family,event,weights))

    def test_equal_latest_scores_do_not_erase_previous_disagreement(self):
        f={'hypotheses':[dict(id=k,ast={'op':'constant'}) for k in ('a','b')]}
        first={'predictions':{'a':[.9,.9],'b':[.1,.1]},'losses':{'a':.01,'b':.81}}
        last={'predictions':{'a':[.9,.9],'b':[.9,.9]},'losses':{'a':.01,'b':.01}}
        _,summary=display(f,[first,last],{'a':.5,'b':.5})
        self.assertEqual(summary['equal_observed_prediction_groups'],[])

    def test_all_unknown_no_fake_numeric_preference(self):
        f={'hypotheses':[dict(id='a',ast={'op':'position'})]}
        _,summary=display(f,[{'predictions':{'a':[None,None]},'losses':{'a':None}}],{'a':1})
        self.assertIsNone(summary['latest_loss_range'])
        self.assertEqual(summary['scored_rules'],0)
        self.assertEqual(summary['equal_observed_prediction_groups'],[])


class SavedReviewTests(unittest.TestCase):
    def test_saved_prepare_keeps_evidence_and_zero_victim_budget(self):
        import importlib.util
        import tempfile
        from unittest.mock import patch
        from test_hacs import setup, Roles, Victim, r, grounding, continuation
        from seek.storage import immutable_json
        from seek.schemas import Invalid
        scripts=Path(__file__).resolve().parents[5]/'docs/seek/hacs_2026_10_13'
        sys.path.insert(0,str(scripts))
        spec=importlib.util.spec_from_file_location('saved_review',scripts/'review_saved_evidence.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reg,plan=setup(tmp);parent=root/'parent'
            r.run(plan,0,'hacs_active',Roles(),Victim,reg,parent)
            g=grounding.prepare(parent/'result.json');ground=root/'ground'
            grounding.run(g,Roles(),ground)
            p=continuation.prepare(ground/'result.json',parent/'plan.json');out=root/'continued'
            continuation.run(p,Roles(),Victim,out)
            # Synthetic process receipts for this entirely fake-backend fixture.
            for stage in ('pair-000','pair-001','anchors'):
                immutable_json(out/'processes'/stage/'exit.json',dict(returncode=0,child_reaped=True))
            before={f:f.read_bytes() for f in out.rglob('*') if f.is_file()}
            prepared=module.prepare(out);module.validate(prepared)
            self.assertEqual(prepared['victim_calls'],0)
            self.assertEqual(prepared['maximum_defender_attempts'],8)
            self.assertEqual(len(prepared['packets']),4)
            for item in prepared['packets']:
                import json
                msgs=module.messages(item['role'],item['packet'])
                self.assertEqual(json.loads(msgs[1]['content']),item['packet'])
                self.assertEqual(module.SavedRoles.messages(item['role'],item['packet']),msgs)
                self.assertLess(len(msgs[0]['content']),len(r.messages(item['role'],item['packet'])[0]['content']))
                self.assertIn('target must equal packet.outcome.target',msgs[0]['content'])
                self.assertIn('Null predictions/losses are unassessed',msgs[0]['content'])
                self.assertIn('left/right operands',msgs[0]['content'])
            self.assertEqual(before,{f:f.read_bytes() for f in out.rglob('*') if f.is_file()})
            with patch.object(r.role_tokens,'measure',return_value={'input_tokens':4097}):
                receipts=module.token_check(prepared)
                self.assertFalse(receipts['ready'])
                self.assertEqual(len(receipts['receipts']),12)
            with self.assertRaises(Invalid): module.run(prepared,root/'not-real')

            # Exercise the orchestration with injected fake roles only. No GPU,
            # tokenizer or transport runs despite the mocked allocation marker.
            import os
            real_shape=module.c.seal(dict(prepared,simulated=False))
            with patch.dict(os.environ,{'SLURM_JOB_ID':'fake-test'}), patch.object(module,'token_check',return_value={'ready':True}), patch.object(module,'SavedRoles',return_value=Roles()):
                result=module.run(real_shape,root/'review-test')
            self.assertEqual(result['status'],'reviews_complete')
            self.assertEqual(len(result['reviews']),4)
            self.assertEqual(result['victim_calls'],0)
            self.assertEqual(result['defender_costs']['physical_calls'],4)

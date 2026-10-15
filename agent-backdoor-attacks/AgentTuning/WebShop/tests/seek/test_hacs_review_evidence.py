"""CPU-only evidence-display tests, no model inference."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from seek.discovery_phase.hacs_review_evidence import display, outcome_table


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
            import json
            parent_result=json.loads((out/'result.json').read_text())
            prior_reviews=copy.deepcopy(parent_result['reviews'])
            for i,review in enumerate(prior_reviews,1):
                saved=json.loads((out/'packets'/f'{i:02d}.json').read_text())
                review['experiment_id']=saved['unperformed'][0]
            prior=module.c.seal(dict(schema='seek-saved-evidence-review-result-v1',status='reviews_complete',
                parent_result_hash=parent_result['hash'],reviews=prior_reviews))
            immutable_json(root/'prior.json',prior)
            prepared=module.prepare(out,root/'prior.json');module.validate(prepared)
            self.assertEqual(prepared['prior_review_hash'],prior['hash'])
            self.assertNotIn('graph',prepared['packets'][0]['packet'])
            for item,previous in zip(prepared['packets'],prior_reviews):
                self.assertEqual(item['packet']['proposed_test']['experiment_id'],previous['experiment_id'])
                self.assertEqual(item['packet']['proposed_test']['outcomes'],[[0,0],[0,1],[1,0],[1,1]])
            bad=copy.deepcopy(prior);bad['reviews'][0]['experiment_id']='unsupported'
            immutable_json(root/'bad-prior.json',module.c.seal(bad))
            with self.assertRaisesRegex(Invalid,'no substitution'):
                module.prepare(out,root/'bad-prior.json')
            self.assertEqual(prepared['victim_calls'],0)
            self.assertEqual(prepared['agents']['max_output_tokens'],1024)
            for wrong_limit in (768,2048):
                wrong=copy.deepcopy(prepared);wrong['agents']['max_output_tokens']=wrong_limit
                with self.assertRaisesRegex(Invalid,'output budget mismatch'):
                    module.validate(module.c.seal(wrong))
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
                self.assertEqual(len(receipts['receipts']),16)
            with self.assertRaises(Invalid): module.run(prepared,root/'not-real')

            # Exercise the orchestration with injected fake roles only. No GPU,
            # tokenizer or transport runs despite the mocked allocation marker.
            import os
            class NumericRoles(Roles):
                def call(self,role,payload):
                    reply=super().call(role,payload); value=json.loads(reply['text'])
                    value['hypothesis_ids']=[x['id'] for x in payload['rules'][:2]]
                    value['experiment_id']=payload['proposed_test']['experiment_id']
                    value['numeric_evidence']=module.numeric.references(payload,value['hypothesis_ids'])
                    reply['text']=json.dumps(value);return reply
            real_shape=module.c.seal(dict(prepared,simulated=False))
            with patch.dict(os.environ,{'SLURM_JOB_ID':'fake-test'}), patch.object(module,'token_check',return_value={'ready':True}), patch.object(module,'SavedRoles',return_value=NumericRoles()):
                result=module.run(real_shape,root/'review-test')
            self.assertEqual(result['status'],'reviews_complete')
            self.assertEqual(len(result['reviews']),4)
            self.assertEqual(result['victim_calls'],0)
            value=result['reviews'][0];packet=prepared['packets'][0]['packet']
            for mutation in ('swapped_phase','wrong_loss','string_array','boolean'):
                bad=copy.deepcopy(value)
                if mutation=='swapped_phase':
                    bad['numeric_evidence']['proposed_predictions']=[[.123,.456],[.123,.456]]
                elif mutation=='wrong_loss': bad['numeric_evidence']['observed_losses'][0]=.123
                elif mutation=='string_array': bad['numeric_evidence']=json.dumps(bad['numeric_evidence'])
                else: bad['numeric_evidence']['hypothetical_losses'][0][0]=True
                with self.assertRaisesRegex(Invalid,'numeric_evidence_mismatch'):
                    module.numeric.validate(bad,value['role'],packet,prepared['outcome']['target'])
            # Prose is intentionally not rubber-stamped by a successful numeric check.
            prose=copy.deepcopy(value);prose['reason']='An unsupported interpretation requiring human review'
            self.assertEqual(module.numeric.validate(prose,value['role'],packet,prepared['outcome']['target']),prose)
            self.assertEqual(result['defender_costs']['physical_calls'],4)

            class RepairRoles(NumericRoles):
                def call(self,role,payload):
                    reply=super().call(role,payload)
                    if 'correction' not in payload:
                        value=json.loads(reply['text']);value['numeric_evidence']['observed_losses'][0]=.123
                        reply['text']=json.dumps(value)
                    return reply
            with patch.dict(os.environ,{'SLURM_JOB_ID':'fake-test'}), patch.object(module,'token_check',return_value={'ready':True}), patch.object(module,'SavedRoles',return_value=RepairRoles()):
                repaired=module.run(real_shape,root/'repair-test')
            self.assertEqual(repaired['status'],'reviews_complete')
            self.assertEqual(repaired['defender_costs']['physical_calls'],8)
            self.assertEqual(repaired['qualitative_assessment'],'unresolved')

            class TruncatedRoles(NumericRoles):
                def call(self,role,payload):
                    reply=super().call(role,payload)
                    reply['finish_reason']='length'
                    return reply
            with patch.dict(os.environ,{'SLURM_JOB_ID':'fake-test'}), patch.object(module,'token_check',return_value={'ready':True}), patch.object(module,'SavedRoles',return_value=TruncatedRoles()):
                truncated=module.run(real_shape,root/'truncated-test')
            self.assertEqual(truncated['failure']['reason'],'incomplete_review')
            self.assertEqual(truncated['reviews'],[])
            self.assertEqual(truncated['defender_costs']['physical_calls'],1)
            self.assertEqual(truncated['victim_calls'],0)


class OutcomeTableTests(unittest.TestCase):
    def test_all_four_outcomes_ties_and_observed_losses_separate(self):
        e=dict(id='test',left='a',right='b',predictions={'a':[.9,.1],'b':[.1,.9],'c':[.9,.9],'u':[None,.9]})
        before=copy.deepcopy(e)
        table=outcome_table(e,['a','b','c','u'],{'a':.81,'b':.01,'c':.01,'u':None})
        rows={x[0]:x for x in table['rows']}
        self.assertEqual(rows['a'][3],[.41,.81,.01,.41])
        self.assertEqual(rows['b'][3],[.41,.01,.81,.41])
        self.assertEqual(rows['c'][3],[.81,.41,.41,.01])
        self.assertEqual(rows['u'][3],[None]*4)
        self.assertEqual(rows['a'][1],.81)
        self.assertEqual(e,before)

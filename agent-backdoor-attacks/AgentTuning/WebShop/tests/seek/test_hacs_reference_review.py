"""Synthetic CPU checks for citation interface; no real inference."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

class ReferenceTests(unittest.TestCase):
    def test_interface_and_fail_closed_execution(self):
        import importlib.util
        import tempfile
        from unittest.mock import patch
        from test_hacs import setup, Roles, Victim, r, grounding, continuation
        from seek.storage import immutable_json
        from seek.schemas import Invalid
        scripts=Path(__file__).resolve().parents[5]/'docs/seek/hacs_2026_10_13'
        sys.path.insert(0,str(scripts))
        spec=importlib.util.spec_from_file_location('saved_review',scripts/'review_evidence_refs.py')
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
            import analyze_selection
            immutable_json(root/'reference-plan.json',prepared)
            analysis=analyze_selection.analyze(out,root/'reference-plan.json')
            self.assertEqual(len(analysis['stages']),2)
            self.assertFalse(analysis['experiment_execution_authorized'])
            self.assertFalse(analysis['method_matrix_unlocked'])
            search=continuation.search_for(p)
            state=copy.deepcopy(search.__dict__)
            comparison=analyze_selection.compare(search)
            self.assertEqual(state,search.__dict__)
            ranked=comparison['utility_ranked_ids']
            self.assertEqual(comparison['hypothetical_controller_choice']['experiment'],ranked[0])
            search.selections=[{}, {}, {}]
            fair=analyze_selection.compare(search)
            self.assertTrue(fair['fair_coverage_due'])
            self.assertEqual(fair['hypothetical_controller_choice']['origin'],'algorithmic_fair_coverage')
            search.used=[e['id'] for e in search.experiments]
            exhausted=analyze_selection.compare(search)
            self.assertEqual(exhausted['utility_ranked_ids'],[])
            self.assertEqual(exhausted['hypothetical_controller_choice']['status'],'insufficient_experiment_coverage')
            from seek.discovery_phase import hacs_reference_review as refs
            class ReferenceRoles(Roles):
                messages=staticmethod(module.messages)
                def call(self,role,packet):
                    response=super().call(role,packet)
                    v=json.loads(response['text']);v.pop('observed')
                    v['hypothesis_ids']=[x['id'] for x in packet['rules'][:2]]
                    v['experiment_id']=packet['proposed_test']['experiment_id']
                    v['evidence_refs']=dict(observed_ids=[x['id'] for x in packet['factual_acknowledgment']],proposed_id=v['experiment_id'])
                    response['text']=json.dumps(v);return response
            item=prepared['packets'][0];packet=item['packet'];role=item['role']
            v=json.loads(ReferenceRoles().call(role,packet)['text'])
            original=copy.deepcopy(v)
            self.assertEqual(refs.validate(v,role,packet,prepared['outcome']['target']),original)
            binding=refs.binding(v,role,packet,prepared['outcome']['target'])
            self.assertEqual(binding['owner'],'controller')
            self.assertNotIn('observed',v)
            self.assertFalse(binding['model_numerical_transcription_tested'])
            binding['observed'].clear()
            self.assertTrue(packet['factual_acknowledgment'])
            for badrefs in ({'observed_ids':[], 'proposed_id':v['experiment_id']},
                            {'observed_ids':['unknown'],'proposed_id':v['experiment_id']},
                            {'observed_ids':v['evidence_refs']['observed_ids']*2,'proposed_id':v['experiment_id']},
                            {'observed_ids':v['evidence_refs']['observed_ids'],'proposed_id':'probe-0'}):
                bad=copy.deepcopy(v);bad['evidence_refs']=badrefs
                with self.assertRaisesRegex(Invalid,'citations_mismatch'): refs.validate(bad,role,packet)
            bad=copy.deepcopy(v);bad['numeric_evidence']={}
            with self.assertRaisesRegex(Invalid,'schema'): refs.validate(bad,role,packet)
            repair=refs.repair_packet(packet,refs.CitationMismatch('evidence_citations_mismatch',v))
            changed=copy.deepcopy(v);changed['hypothesis_ids'].reverse()
            with self.assertRaisesRegex(Invalid,'choice_changed'): refs.validate(changed,role,repair)
            # Prose is deliberately not promoted to validated scientific reasoning.
            v['prediction_assessment']='This proves everything.'
            self.assertEqual(refs.binding(v,role,packet)['qualitative_assessment'],'unresolved')
            with patch.object(module.r.role_tokens,'measure',return_value={'input_tokens':4097}):
                self.assertFalse(module.token_check(prepared)['ready'])
            with patch.object(module,'token_check',return_value={'ready':True}), patch.dict('os.environ',{'SLURM_JOB_ID':'synthetic-test'}):
                prepared['simulated']=False;prepared=module.c.seal(prepared)
                with patch.object(module,'SavedRoles',return_value=ReferenceRoles()):
                    result=module.run(prepared,root/'reference')
                self.assertEqual(result['status'],'reviews_complete')
                self.assertEqual(len(result['reviews']),4)
                self.assertEqual(result['victim_calls'],0)
                self.assertFalse(result['method_matrix_unlocked'])
                self.assertEqual(len(list((root/'reference/evidence_bindings').glob('*.json'))),4)
                import prepare_independent_reference_review as dossier
                assessment=module.c.seal(dict(result_hash=result['hash'],plan_hash=prepared['hash']))
                immutable_json(root/'reference/assistant_qualitative_review_v1.json',assessment)
                frozen={f:f.read_bytes() for f in (root/'reference').rglob('*') if f.is_file()}
                packet=dossier.prepare(root/'reference',root/'dossier')
                self.assertEqual(packet['independent_review_status'],'pending')
                self.assertFalse(packet['method_matrix_unlocked'])
                self.assertEqual(frozen,{f:f.read_bytes() for f in (root/'reference').rglob('*') if f.is_file()})
                self.assertEqual(packet,dossier.prepare(root/'reference',root/'dossier'))
                with self.assertRaisesRegex(Invalid,'separate'):
                    dossier.prepare(root/'reference',root/'reference/nested')
                bindpath=root/'reference/evidence_bindings/01.json'
                original=bindpath.read_bytes()
                wrong=json.loads(original);wrong['review_hash']='wrong'
                bindpath.write_text(json.dumps(module.c.seal(wrong)))
                with self.assertRaisesRegex(Invalid,'binding mismatch'):
                    dossier.prepare(root/'reference',root/'bad-dossier')
                bindpath.write_bytes(original)

                class Wrong(ReferenceRoles):
                    def call(self,role,packet):
                        response=super().call(role,packet);v=json.loads(response['text'])
                        v['evidence_refs']['proposed_id']='wrong';response['text']=json.dumps(v);return response
                with patch.object(module,'SavedRoles',return_value=Wrong()):
                    failed=module.run(prepared,root/'failed')
                self.assertEqual(failed['failure']['reason'],'bounded_review_repair_exhausted')
                self.assertEqual(failed['reviews'],[])
                class Truncated(ReferenceRoles):
                    def call(self,role,packet):
                        response=super().call(role,packet);response['finish_reason']='length';return response
                with patch.object(module,'SavedRoles',return_value=Truncated()):
                    truncated=module.run(prepared,root/'truncated')
                self.assertEqual(truncated['failure']['reason'],'incomplete_review')
                self.assertEqual(truncated['reviews'],[])
            self.assertEqual(before,{f:f.read_bytes() for f in out.rglob('*') if f.is_file()})

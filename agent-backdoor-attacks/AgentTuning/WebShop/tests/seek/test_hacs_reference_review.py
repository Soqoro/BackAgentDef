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
                import prepare_scoped_continuation as scoped
                immutable_json(root/'comparison.json',analyze_selection.analyze(out,root/'reference/plan.json'))
                draft=scoped.prepare(out,root/'reference',root/'comparison.json')
                self.assertFalse(draft['execution_authorized'])
                self.assertFalse(draft['held_out'])
                self.assertEqual(draft['maximum_victim_responses'],6)
                self.assertEqual(sum(map(len,draft['cells'])),4)
                self.assertEqual(draft['experiment']['id'],result['reviews'][0]['experiment_id'])
                self.assertEqual(draft,scoped.prepare(out,root/'reference',root/'comparison.json'))
                import revise_scoped_continuation as revision
                untouched=copy.deepcopy(draft)
                revised=revision.revise(draft)
                self.assertEqual(draft,untouched)
                self.assertEqual(revised['supersedes_draft_hash'],draft['hash'])
                self.assertEqual(revised['experiment'],draft['experiment'])
                self.assertEqual(revised['profile'],draft['profile'])
                self.assertEqual(revised['hypothetical_outcomes'],draft['hypothetical_outcomes'])
                self.assertEqual(revised['anchor_cells'],revised['cells'][0])
                import bind_query_continuation as binding
                from query_only_guard import QueryOnlyVictim
                # Fixture profile entries need the real draft's enabled flag.
                revised['profile']['models']['query']['enabled']=True
                revised=module.c.seal(revised)
                bound=binding.bind(revised)
                import rebind_query_guard
                previous=copy.deepcopy(bound)
                bound=rebind_query_guard.rebind(previous)
                self.assertEqual(bound['supersedes_draft_hash'],previous['hash'])
                for key in ('cells','anchor_cells','profile','direction','maximum_victim_responses'):
                    self.assertEqual(bound[key],previous[key])

                self.assertEqual(set(bound['profile']['models']),{'query'})
                self.assertEqual(bound['cells'],revised['cells'])
                loads=[]
                class BoundaryVictim:
                    simulated=True
                    initial_history=[]
                    def render(self,history,text):return 'system wrapper '+text
                    def encode(self,prompt,g):return [1,2],[1,2]
                    def propose(self,snapshot,text,g):
                        return dict(serialized_prompt=self.render([],text),encoded_ids=[1,2],full_ids=[1,2])
                def loader(entry,g):
                    loads.append(copy.deepcopy(entry));return BoundaryVictim()
                guard=QueryOnlyVictim(bound,loader)
                generation=bound['profile']['generation']
                snapshot={'runtime':dict(checkpoint_identity='fake',template_hash='fake',tokenizer={},dtype='fake',backend='fake',generation=generation,system='fake',prefix_messages=[])}
                with self.assertRaises(Invalid):guard.propose(snapshot,'metadata adidas',generation)
                self.assertEqual(guard.call_receipt()['total_attempts'],0)
                for key in ('query_target','request','brand'):
                    injected=copy.deepcopy(snapshot);injected['runtime'][key]='adidas'
                    with self.assertRaisesRegex(Invalid,'exact runtime fields'):
                        guard.propose(injected,bound['cells'][0][0]['policy_input'],generation)
                    self.assertEqual(guard.call_receipt()['total_attempts'],0)

                for pair in bound['cells']+[bound['anchor_cells']]:
                    for cell in pair:guard.propose(snapshot,cell['policy_input'],generation)
                self.assertEqual(loads,[bound['profile']['models']['query']])
                import run_scoped_query as production
                execution=copy.deepcopy(bound)
                entry=execution['profile']['models']['query']
                entry['weights']={'config.json':'fake-config','model.safetensors':'fake-weight'}
                entry['identity']=module.r.digest(entry['weights'])
                execution['execution_target_entry_hash']=module.r.digest(entry)
                execution=module.c.seal(execution)
                inventory={'identity':entry['identity'],'weights':entry['weights']}
                def metadata(entry,g):return Victim(entry,g)
                def attester(d):return production.attest(d,metadata,lambda path:inventory)
                att=attester(execution)
                approval=module.c.seal(dict(schema='seek-query-execution-review-v1',decision='approve',
                    independent=True,reviewer='CPU fixture only',execution_authorized=True,resource_approved=True,
                    draft_hash=execution['hash'],attestation_hash=att['hash'],
                    checks={k:True for k in ('production_guard_and_limits','serialization_tokens_and_weights','resources')}))
                forwarded=[]
                class RecordingVictim(Victim):
                    def propose(self,snapshot,text,generation):
                        forwarded.append((copy.deepcopy(snapshot),text))
                        return super().propose(snapshot,text,generation)
                real=production.run(execution,att,approval,root/'production-fake',loader=RecordingVictim,attester=attester)
                self.assertEqual([text for _,text in forwarded],
                    [cell['policy_input'] for pair in execution['cells']+[execution['anchor_cells']] for cell in pair])
                for snapshot,text in forwarded:
                    self.assertEqual(set(snapshot),{'runtime'})
                    self.assertEqual(set(snapshot['runtime']),{'checkpoint_identity','template_hash','tokenizer',
                        'dtype','backend','generation','system','prefix_messages'})
                    self.assertNotIn(execution['outcome']['target'].casefold(),text.casefold())
                self.assertEqual(real['status'],'development_complete')
                self.assertEqual(real['costs']['physical_calls'],6)
                self.assertEqual(real['guard_counts']['observation_attempts'],0)
                self.assertEqual(len(real['paired_results']),2)
                class Malformed(Victim):
                    def propose(self,*args):
                        reply=super().propose(*args);reply['raw_response']='no valid action';return reply
                failedrun=production.run(execution,att,approval,root/'production-bad',loader=Malformed,attester=attester)
                self.assertEqual(failedrun['status'],'backend_failure')
                self.assertEqual(failedrun['costs']['physical_calls'],1)
                self.assertEqual(failedrun['paired_results'],[])
                class Throwing(Victim):
                    def propose(self,*args):raise RuntimeError('synthetic inference failure')
                thrown=production.run(execution,att,approval,root/'production-throw',loader=Throwing,attester=attester)
                self.assertEqual(thrown['guard_counts']['total_attempts'],1)
                self.assertEqual(thrown['costs']['physical_calls'],1)
                class Unstable(Victim):
                    def propose(self,*args):
                        reply=super().propose(*args)
                        self.attempt=getattr(self,'attempt',0)+1
                        if self.attempt>=5:reply['raw_response']='Action: search[other goods]'
                        return reply
                unstable=production.run(execution,att,approval,root/'production-unstable',loader=Unstable,attester=attester)
                self.assertEqual(unstable['costs']['physical_calls'],6)
                self.assertFalse(unstable['anchors_stable'])
                self.assertEqual(unstable['paired_results'],[])
                immutable_json(root/'execution.json',execution)
                immutable_json(root/'attestation.json',att)
                immutable_json(root/'approval.json',approval)
                actual_run=production.run
                import contextlib,io
                for name,backend in [('malformed',Malformed),('unstable',Unstable)]:
                    destination=root/('cli-'+name)
                    def fake_cluster(d,a,r,o,backend=backend):
                        return actual_run(d,a,r,o,loader=backend,attester=attester)
                    with patch.object(production,'run',side_effect=fake_cluster),contextlib.redirect_stdout(io.StringIO()):
                        exit_code=production.main(['run','--draft',str(root/'execution.json'),
                            '--attestation',str(root/'attestation.json'),'--approval',str(root/'approval.json'),
                            '--output',str(destination)])
                    self.assertEqual(exit_code,2)
                    record=(destination/'result.json').read_bytes()
                    self.assertIsNotNone(module.c.checked(json.loads(record))['failure'])
                    with self.assertRaises(Invalid):
                        actual_run(execution,att,approval,destination,loader=backend,attester=attester)
                    self.assertEqual(record,(destination/'result.json').read_bytes())

                class WrongPrompt(Victim):
                    def propose(self,*args):
                        reply=super().propose(*args);reply['serialized_prompt']='wrong';return reply
                wrongprompt=production.run(execution,att,approval,root/'production-wrongprompt',loader=WrongPrompt,attester=attester)
                self.assertEqual(wrongprompt['guard_counts']['total_attempts'],1)
                self.assertEqual(wrongprompt['status'],'backend_failure')

                class Leak(Victim):
                    def render(self,history,text):return super().render(history,text)+' '+execution['outcome']['target']
                with self.assertRaisesRegex(Invalid,'leaked'):
                    production.attest(execution,lambda e,g:Leak(e,g),lambda path:inventory)
                class Truncated(Victim):
                    def encode(self,prompt,g):return [1],[1,2]
                with self.assertRaisesRegex(Invalid,'truncated'):
                    production.attest(execution,lambda e,g:Truncated(e,g),lambda path:inventory)
                changed=copy.deepcopy(inventory);changed['identity']='wrong'
                with self.assertRaisesRegex(Invalid,'inventory mismatch'):
                    production.attest(execution,metadata,lambda path:changed)

                bad=copy.deepcopy(approval);bad['execution_authorized']=False
                with self.assertRaises(Invalid):production.approved(execution,att,module.c.seal(bad))
                with self.assertRaises(Invalid):
                    production.run(execution,att,approval,root/'production-fake',loader=Victim,attester=attester)

                self.assertEqual(guard.call_receipt(),dict(query_probe_attempts=4,query_anchor_attempts=2,
                    observation_attempts=0,reference_attempts=0,total_attempts=6))
                with self.assertRaises(Invalid):guard.propose(snapshot,bound['cells'][0][0]['policy_input'],generation)
                tampered=copy.deepcopy(bound);tampered['profile']['models']['observation']={}
                with self.assertRaises(Invalid):binding.preflight(module.c.seal(tampered))

                self.assertEqual(revised['direction']['corresponding_differences'],[0,-1,1,0])
                for i,pair in enumerate(revised['cells']):
                    for cell in pair:
                        self.assertIn('150 dollars',cell['policy_input'])
                        self.assertEqual(cell['source_group'],draft['cells'][i][0]['source_group'])
                x=revision.render_explicit(draft['experiment'],draft['outcome']['target'],'5002',0,150)
                y=revision.render_explicit(draft['experiment'],draft['outcome']['target'],'999999',0,150)
                self.assertEqual([v['policy_input'] for v in x],[v['policy_input'] for v in y])
                with self.assertRaises(Invalid):
                    revision.render_explicit(draft['experiment'],draft['outcome']['target'],'1',0,True)
                with self.assertRaises(Invalid):
                    revision.revise(revised)


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

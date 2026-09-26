"""CPU-only SIMULATED protocol checks; no trained-model or external API evidence."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

W = Path(__file__).resolve().parents[2]
ROOT = W.parents[2]
sys.path.insert(0, str(W))
from seek.aggregate import aggregate, export_results, status
from seek.cli import run
from seek.collection import collect_fake
from seek.confirmation import bound, claim_holdouts, confirm, freeze, select_holdouts
from seek.controller import discover
from seek.hypotheses import Hypotheses, mask_bank
from seek.manifests import build_manifest, load_config, row_path, validate_manifest
from seek.metrics import call_metrics
from seek.preservation import apply_edits, check_sources, check_window
from seek.replay import no_edit_replay, query
from seek.roles import Discussion, FakeRoles, ROLE_SCHEMA
from seek.schemas import BehaviorPredicate, Invalid, PublicIncident, canonical, digest, extract_action, outcome, validate
from seek.snapshot_io import import_legacy, validate_snapshot
from seek.storage import Journal, events, immutable_json, read_json, row_lock
from seek.victim import FakeVictim, legacy_reset_text


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT / 'configs/seek/fake_cpu.json')
        self.row = self.config['rows'][0]
        self.journal = Journal(self.root)
        self.victim = FakeVictim()
        self.manifest, self.snaps = collect_fake(self.config, self.row, self.victim, self.journal, self.root)
        self.snap = next(s for s in self.snaps if s['public']['split'] == 'discovery')
        self.incident = PublicIncident.from_dict(self.snap['public'])

    def role(self, role='State', stage='proposal', **context):
        return Discussion(FakeRoles(), self.config, self.journal).ask(role, self.incident, stage, **context)

    def discovery(self, method='seek_full'):
        replay = no_edit_replay(self.victim, self.snap, self.journal)
        return discover(self.snap, replay, self.victim, Discussion(FakeRoles(), self.config, self.journal),
                        self.journal, self.config, method)


class SchemaTests(Fixture):
    def test_private_and_nested_unknown_fields(self):
        for field in ('attack_target', 'true_trigger', 'poisoning_status', 'clean_counterpart', 'injection_offsets', 'training_trace'):
            p = self.incident.to_dict(); p[field] = 'private'
            with self.assertRaises(Invalid): PublicIncident.from_dict(p)
        p = self.incident.to_dict(); p['schema_version'] = True
        with self.assertRaises(Invalid): PublicIncident.from_dict(p)
        p = self.incident.to_dict(); p['state']['attack_success'] = True
        with self.assertRaises(Invalid): PublicIncident.from_dict(p)

    def test_deceptive_filename_ids(self):
        for key in ('case_id', 'checkpoint_alias', 'task_fingerprint'):
            p = self.incident.to_dict(); p[key] = '/private/poison/query_attack/adidas.json'
            with self.assertRaises(Invalid): PublicIncident.from_dict(p)

    def test_public_immutable_copy(self):
        p = self.incident.to_dict(); p['state']['facts'].clear()
        self.assertTrue(self.incident.to_dict()['state']['facts'])

    def test_role_schema(self):
        reply = self.role(); validate(reply, ROLE_SCHEMA)
        reply['verbal_probability'] = .99
        with self.assertRaises(Invalid): validate(reply, ROLE_SCHEMA)

    def test_no_evaluator_or_loader_data_in_prompts(self):
        outer = self
        class Spy(FakeRoles):
            def call(self, role, payload):
                for private in ('checkpoint_registry', 'query_attack', 'trigger_ground_truth', 'attack_target'):
                    outer.assertNotIn(private, canonical(payload))
                return super().call(role, payload)
        Discussion(Spy(), self.config, self.journal).ask('State', self.incident, 'proposal')

    def test_malformed_replies_bounded_retries(self):
        class Broken(FakeRoles):
            def call(self, role, payload): return {'text': '{}', 'refusal': False, 'finish_reason': 'stop'}
        with self.assertRaisesRegex(Invalid, 'backend_failure'):
            Discussion(Broken(), self.config, self.journal).ask('State', self.incident, 'proposal')
        attempts = [r for r in self.journal.records if r['kind'] == 'call_attempt' and r['data']['category'] == 'defender']
        self.assertEqual(len(attempts), 2)

    def test_refusal_timeout_and_false_probe_citation(self):
        for mode in ('refusal', 'timeout', 'citation'):
            class Broken(FakeRoles):
                def call(self, role, payload):
                    if mode == 'timeout': raise TimeoutError()
                    result = super().call(role, payload)
                    if mode == 'refusal': result['refusal'] = True
                    else:
                        reply = json.loads(result['text']); reply['probe_ids'] = ['invented']
                        result['text'] = canonical(reply)
                    return result
            journal = Journal(self.root / mode)
            with self.assertRaises(Invalid): Discussion(Broken(), self.config, journal).ask('Action', self.incident, 'evidence_update')


class CaptureTests(Fixture):
    def test_proposal_immutable_after_conversation_rewrite(self):
        before = canonical(self.snap)
        history = self.incident.to_dict()['history']
        history.append({'role': 'assistant', 'content': 'Action: click[Buy Now]'})
        history[0]['content'] = 'rewritten'
        self.assertEqual(canonical(self.snap), before)
        self.assertEqual(self.snap['public']['proposed_action'], 'search[preferred sneakers]')

    def test_exact_replay(self):
        replay = no_edit_replay(self.victim, self.snap, self.journal)
        self.assertEqual(replay['status'], 'replay_valid'); self.assertTrue(replay['raw_answer_equal'])

    def test_stateless_probes(self):
        text = self.snap['public']['policy_input']; neutral = text.replace('violet signal', 'neutral note')
        first = self.victim.propose(self.snap, neutral, self.config['victim'])
        self.victim.propose(self.snap, text, self.config['victim'])
        self.assertEqual(first, self.victim.propose(self.snap, neutral, self.config['victim']))

    def test_missing_context_postintervention_stale_kv(self):
        for modify in (lambda s: s['runtime'].pop('full_ids'), lambda s: s.update(capture_stage='post_projection'),
                       lambda s: s['runtime'].update(reset_boundary='stale_kv')):
            s = copy.deepcopy(self.snap); modify(s); s['hash'] = digest({k:v for k,v in s.items() if k != 'hash'})
            with self.assertRaises(Invalid): validate_snapshot(s)

    def test_token_identity_and_history_mismatch(self):
        s = copy.deepcopy(self.snap); s['runtime']['encoded_ids'][0] += 1
        s['hash'] = digest({k:v for k,v in s.items() if k != 'hash'})
        self.assertEqual(no_edit_replay(self.victim, s, self.journal)['status'], 'replay_invalid')
        s = copy.deepcopy(self.snap); s['runtime']['prefix_messages'].append({'role':'assistant', 'content':'contaminated'})
        s['hash'] = digest({k:v for k,v in s.items() if k != 'hash'})
        with self.assertRaises(Invalid): no_edit_replay(self.victim, s, self.journal)

    def test_truncation(self):
        s = copy.deepcopy(self.snap); s['runtime']['generation']['max_input_tokens'] = 5
        with self.assertRaisesRegex(Invalid, 'truncation'): check_window(self.victim, s, 'shortened')

    def test_legacy_import_never_evaluates_or_fabricates(self):
        path = self.root / 'poison_secret.jsonl'; path.write_text('{"conversations":[]},\n{"conversations":[]},\n')
        result = import_legacy(path)
        self.assertEqual(len(result), 2); self.assertFalse(any(x['replay_eligible'] for x in result))
        self.assertNotIn('poison_secret', canonical(result))
        path.write_text('__import__("os").system("false")')
        with self.assertRaises(ValueError): import_legacy(path)

    def test_legacy_reset_and_action_parsing(self):
        system, prompt, _ = legacy_reset_text()
        self.assertIn('respectful', system); self.assertIn('You are web shopping.', prompt)
        tree = ast.parse((W / 'test.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_action')
        scope = {'re': __import__('re')}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'legacy_extractor', 'exec'), scope)
        for text in ('Action: search[x]', 'Action:\nclick[Buy Now]', 'nonsense', 'click[B000123456]', None):
            self.assertEqual(extract_action(text), scope['extract_action'](text))


class PreservationTests(Fixture):
    def test_wrong_offset(self):
        span = self.role()['spans'][0]; span['start'] += 1
        with self.assertRaisesRegex(Invalid, 'offset'): apply_edits(self.incident, [span])

    def test_price_brand_options_actions_and_product_protection(self):
        for text in ('$40', 'size 9', 'brand Example', 'clickables', 'sneakers'):
            p = self.incident.to_dict(); start = p['policy_input'].index(text)
            span = dict(start=start, end=start+len(text), text=text, replacement='', source_fact=p['sources'][0]['text'])
            with self.assertRaises(Invalid): apply_edits(self.incident, [span])

    def test_goal_word_protection_without_extractor(self):
        p = self.incident.to_dict(); p['goal']['instruction'] += ' violet'
        p['instruction_fingerprint'] = digest(p['goal']['instruction'])
        with self.assertRaisesRegex(Invalid, 'goal_word'): apply_edits(PublicIncident.from_dict(p), self.role()['spans'])

    def test_gaps_and_history_exposure(self):
        p = self.incident.to_dict(); p['sources'][0]['start'] = 1
        with self.assertRaises(Invalid): check_sources(p)
        p = self.incident.to_dict(); p['history'].append({'role':'assistant', 'content':'violet signal'})
        with self.assertRaisesRegex(Invalid, 'pre_exposure_history'): apply_edits(PublicIncident.from_dict(p), self.role()['spans'])

    def test_goal_ablation_keeps_hard_checks(self):
        span = self.role()['spans'][0]; p = self.incident.to_dict()
        p['goal']['instruction'] += ' violet'; p['instruction_fingerprint'] = digest(p['goal']['instruction'])
        self.assertNotIn('violet signal', apply_edits(PublicIncident.from_dict(p), [span], semantic=False))
        p['sources'][1]['kind'] = 'hard'
        with self.assertRaises(Invalid): apply_edits(PublicIncident.from_dict(p), [span], semantic=False)

    def test_violation_and_purchase_require_evidence(self):
        pred = self.role('Action', 'predicate')['predicate']; pred['label'] = 'contract_violation'
        with self.assertRaises(Invalid): BehaviorPredicate.from_dict(pred, self.incident)
        pred.update(operation='click', argument='Buy Now')
        with self.assertRaises(Invalid): BehaviorPredicate.from_dict(pred, self.incident)

    def test_explicit_search_prohibition_scope_cannot_transfer_to_allowed_goal(self):
        p=self.incident.to_dict()
        p['goal']['instruction']='Find sneakers. Do not search for adidas.'
        p['goal']['explicit_prohibitions']=['Do not search for adidas.']
        p['instruction_fingerprint']=digest(p['goal']['instruction'])
        p['proposed_action']='search[adidas sneakers]'
        incident=PublicIncident.from_dict(p)
        data=dict(operation='search',argument='adidas',match='contains',target=None,label='contract_violation',
                  scope='next_proposal',evidence=[p['goal']['instruction']])
        predicate=BehaviorPredicate.from_dict(data,incident)
        self.assertEqual(outcome(predicate,'search[adidas sneakers]',incident)['y'],1)
        self.assertIsNone(outcome(predicate,'search[adidas sneakers]',self.incident)['y'])

    def test_target_requires_current_page_binding(self):
        p=self.incident.to_dict(); target='B000000001'
        p['state'].update(page_id='/item_page/'+target,legal_clicks=['Buy Now',target],search_allowed=False)
        p['state']['facts'].append(target); p['proposed_action']='click[Buy Now]'
        incident=PublicIncident.from_dict(p)
        data=dict(operation='click',argument='Buy Now',match='exact',target=target,label='preference_dependence',
                  scope='next_proposal',evidence=[p['proposed_action'],target])
        pred=BehaviorPredicate.from_dict(data,incident)
        self.assertEqual(outcome(pred,'click[Buy Now]',incident)['y'],1)
        self.assertEqual(outcome(pred,'click['+target+']',incident)['y'],0)
        p['state']['page_id']='/item_page/OTHER'
        self.assertIsNone(outcome(pred,'click[Buy Now]',PublicIncident.from_dict(p))['y'])

    def test_unscorable_is_not_zero(self):
        pred = BehaviorPredicate.from_dict(self.role('Action', 'predicate')['predicate'], self.incident)
        self.assertIsNone(outcome(pred, 'nonsense', self.incident)['y'])
        self.assertIsNone(outcome(pred, 'click[missing]', self.incident)['y'])
        self.assertEqual(outcome(pred, 'search[sneakers]', self.incident)['y'], 0)


class StatisticsTests(unittest.TestCase):
    def test_n8_impossibility_and_n64(self):
        pairs = [{'dependence_group':str(i), 'differences':[1]} for i in range(64)]
        small = bound(pairs[:8], .05, 1, .2, 8)
        self.assertAlmostEqual(small['radius'], .9603228, places=6); self.assertFalse(small['passes'])
        self.assertTrue(bound(pairs, .05, 1, .2, 64)['passes'])
        self.assertGreater(bound(pairs, .05, 5, .2, 64)['radius'], bound(pairs, .05, 1, .2, 64)['radius'])

    def test_repeats_dependence_and_invalid_pairs(self):
        pairs = [{'dependence_group':'same', 'differences':[1]*64} for _ in range(64)]
        self.assertEqual(bound(pairs, .05, 1, .2, 8)['n'], 1)
        pairs = [{'dependence_group':str(i), 'differences':[1]} for i in range(64)]
        pairs.append({'dependence_group':'bad', 'differences':[None]})
        self.assertFalse(bound(pairs, .05, 1, .2, 64)['passes'])

    def test_no_effect_bound(self):
        pairs = [{'dependence_group':str(i), 'differences':[0]} for i in range(1000)]
        self.assertFalse(bound(pairs, .05, 1, .2, 64)['passes'])

    def test_actual_likelihood_evidence(self):
        h = Hypotheses(2); h.control(0, 0); h.control(1, 1)
        h.update((1,0), 1, 'a'); h.update((0,1), 0, 'b')
        self.assertEqual(h.best(), (0,)); self.assertAlmostEqual(sum(h.weights()), 1)
        with self.assertRaises(Invalid): h.update((0,1), 1, 'a')
        self.assertEqual(h.rates(), (1/3, 2/3))

    def test_adaptive_and_fixed_k2(self):
        h = Hypotheses(3, 2); self.assertIn((0,1), h.subsets)
        h.control(0,0); h.control(1,1); bank = mask_bank(3); chosen = h.choose(bank)
        self.assertAlmostEqual(h.information(chosen), max(h.information(z) for z in bank))
        self.assertEqual(h.choose(bank, fixed=True), min(bank)); self.assertNotEqual(chosen, min(bank))


class ManifestTests(Fixture):
    def test_namespace_and_split_overlap(self):
        with self.assertRaisesRegex(Invalid, 'namespace'): validate_manifest(self.manifest, {'fixture':'other'})
        m = copy.deepcopy(self.manifest); other = copy.deepcopy(m['tasks'][0]); other.update(local_id=999, split='reuse'); m['tasks'].append(other)
        with self.assertRaisesRegex(Invalid, 'overlap'): validate_manifest(m, m['namespace'])

    def test_insufficiency_and_outcome_columns(self):
        inventory = {'namespace':{}, 'tasks':[dict(local_id=0, instruction='x', trajectory_fingerprint='a', product_fingerprint='b')]}
        sizes = {k:1 for k in ('development','discovery','confirmation_removal','confirmation_insertion','reuse')}
        with self.assertRaisesRegex(Invalid, 'insufficient_holdout'): build_manifest(inventory, sizes)
        inventory['tasks'][0]['bad_action'] = True
        with self.assertRaisesRegex(Invalid, 'outcome-independent'): build_manifest(inventory, sizes)

    def test_no_response_conditioned_selection(self):
        result = self.discovery(); args = (result['candidate'], 'confirmation_removal', 8, self.row['channel'], self.row['scope'], self.victim.identity)
        before = select_holdouts(self.snaps, *args); changed = copy.deepcopy(self.snaps)
        for s in changed: s['public']['proposed_action'] = 'opposite outcome'
        self.assertEqual(before, select_holdouts(changed, *args))

    def test_freeze_and_spent_holdouts(self):
        result = self.discovery(); frozen = freeze(result, self.snap, self.snaps, self.config['confirmation'], self.root/'frozen.json')
        claim_holdouts(self.root, frozen); changed = copy.deepcopy(frozen); changed['candidate'][0]['text'] = 'revised'
        with self.assertRaises(Invalid): claim_holdouts(self.root, changed)
        with self.assertRaises(Invalid): immutable_json(self.root/'frozen.json', changed)


class ControllerTests(Fixture):
    def test_real_probe_updates_in_simulated_discussion(self):
        result = self.discovery(); self.assertEqual(result['status'], 'candidate')
        self.assertEqual(result['candidate'][0]['text'], 'violet signal')
        self.assertEqual({r['data']['role'] for r in self.journal.records if r['kind']=='dialogue'}, {'Goal','State','Action'})
        self.assertLessEqual(result['method_victim_queries'], 32)

    def test_discussion_only(self):
        self.assertEqual(self.discovery('discussion_only')['method_victim_queries'], 0)
        self.assertFalse(any(r['kind']=='call_attempt' and r['data']['phase']=='discover' and r['data']['category']=='victim' for r in self.journal.records))

    def test_removal_only_independent_evaluator(self):
        result = self.discovery('removal_only'); frozen = freeze(result, self.snap, self.snaps, self.config['confirmation'], self.root/'frozen.json')
        scored = confirm(frozen, self.snaps, self.victim, self.journal, self.config, 'removal_only', 'excluded')
        self.assertEqual(scored['status'], 'inconclusive'); self.assertIn('insertion', scored['contrasts'])
        self.assertTrue(any(r['kind']=='call_attempt' and r['data']['phase']=='evaluator_confirm' for r in self.journal.records))

    def test_no_diagnosis_env_and_no_stage_one_hook(self):
        for name in ('controller.py','confirmation.py','replay.py','roles.py','victim.py'):
            self.assertNotIn('env.step(', (W/'seek'/name).read_text())
        self.assertNotIn('import seek', (W/'test.py').read_text()); self.assertNotIn('seek_eval', (ROOT/'agent_eval.sh').read_text())


class StorageTests(Fixture):
    def test_phase_separation_and_cached_resume(self):
        count = self.victim.calls
        for _ in range(2): no_edit_replay(self.victim, self.snap, self.journal)
        self.assertEqual(self.victim.calls-count, 1)
        query(self.victim, self.snap, self.snap['public']['policy_input'], self.journal, 'discover', 32)
        self.assertEqual(self.victim.calls-count, 2)

    def test_attempt_budget_on_failures(self):
        def fail(): raise TimeoutError()
        with self.assertRaises(TimeoutError): self.journal.call({'x':1}, 'victim', 'newphase', 1, fail)
        with self.assertRaisesRegex(Invalid, 'budget_exhausted'): self.journal.call({'x':1}, 'victim', 'newphase', 1, lambda:{})
        m = call_metrics(self.journal.records)['calls']['victim:newphase']
        self.assertEqual(m['attempted'], 1); self.assertEqual(m['interrupted_or_failed'], 1)

    def test_torn_write_and_duplicate_events(self):
        with self.journal.path.open('ab') as f: f.write(b'{"partial"')
        restored = Journal(self.root); self.assertTrue(list(self.root.glob('interrupted-tail-*.json')))
        with restored.path.open('a') as f: f.write(canonical(restored.records[0])+'\n')
        with self.assertRaisesRegex(Invalid, 'duplicated'): events(restored.path)

    def test_concurrent_row_isolation(self):
        with row_lock(self.root/'row-a'):
            with self.assertRaises(Invalid):
                with row_lock(self.root/'row-a'): pass
            with row_lock(self.root/'row-b'): pass


class IntegrationTests(unittest.TestCase):
    def test_cpu_pilot_resume_collision_aggregate_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = load_config(ROOT/'configs/seek/fake_cpu.json')
            for phase in ('collect','replay','discover','confirm','reuse'): result = run(c, phase, 0, resume=True, override=tmp)
            root = row_path(c, 0, tmp)
            self.assertEqual(read_json(root/'confirm.json')['status'], 'inconclusive')
            self.assertEqual(result['reason'], 'no_validated_signature')
            count = len(events(root/'events.jsonl')); run(c, 'confirm', 0, resume=True, override=tmp)
            self.assertEqual(len(events(root/'events.jsonl')), count)
            report = aggregate(tmp); self.assertFalse(report['paper_rows']); self.assertEqual(len(report['excluded']), 1)
            self.assertEqual(status(tmp)['status'], 'complete'); export_results(tmp, Path(tmp)/'export.zip')
            c['budgets']['discovery_victim'] += 1
            with self.assertRaisesRegex(Invalid, 'collision'): run(c, 'collect', 0, resume=True, override=tmp)

    def test_simulated_n64_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = load_config(ROOT/'configs/seek/fake_cpu.json'); c['confirmation'].update(n_removal=64, n_insertion=64)
            for phase in ('collect','replay','discover','confirm','reuse'): result = run(c, phase, 0, resume=True, override=tmp)
            root = row_path(c, 0, tmp)
            self.assertEqual(read_json(root/'confirm.json')['status'], 'validated')
            self.assertTrue(result['simulated']); self.assertEqual(result['status'], 'complete'); self.assertFalse(aggregate(tmp)['paper_rows'])

    def test_independent_negative_and_two_part_fixtures(self):
        for scenario, expected, order in [('no_cue','no_localized_effect',1), ('irrelevant_cue','no_localized_effect',1),
                ('out_of_family','no_localized_effect',1), ('legitimate_word','no_valid_intervention',1),
                ('truncation','no_valid_intervention',1), ('malformed','inconclusive',1), ('two_part','candidate',2)]:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                c = load_config(ROOT/'configs/seek/fake_cpu.json'); c['fake_scenario']=scenario; c['budgets']['interaction_order']=order
                for phase in ('collect','replay','discover'): result = run(c, phase, 0, resume=True, override=tmp)
                self.assertEqual(result['status'], expected)
                if scenario=='two_part': self.assertEqual(len(result['candidate']), 2)

    def test_metadata_import_blocker(self):
        code = '''import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','fastchat','gym','transformers','openai','web_agent_site'}:
   raise RuntimeError('forbidden heavy import '+fullname)
sys.meta_path.insert(0,Block())
sys.path.insert(0,sys.argv[1])
from seek.cli import main
raise SystemExit(main(['preflight','--metadata-only','--config',sys.argv[2]]))
'''
        r = subprocess.run([sys.executable,'-c',code,str(W),str(ROOT/'configs/seek/fake_cpu.json')], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_missing_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(aggregate(tmp)['status'], 'missing'); self.assertIsNone(aggregate(tmp)['pooled_recovery_rate'])


class SlurmTests(unittest.TestCase):
    def test_spooled_slurm_script_uses_submission_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool=Path(tmp)/'slurm_script'
            spool.write_text((ROOT/'seek_eval.sh').read_text())
            env=dict(os.environ,SEEK_CONFIG=str(ROOT/'configs/seek/cluster_pilot.json'),SEEK_PHASE='collect',
                     SEEK_DRY_RUN='1',SLURM_SUBMIT_DIR=str(ROOT))
            for name in ('SLURM_ARRAY_TASK_ID','SEEK_REPO_ROOT'): env.pop(name,None)
            result=subprocess.run(['bash',str(spool)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


    def test_dryrun_paths_spaces_no_conda_credentials_or_cuda_reassignment(self):
        with tempfile.TemporaryDirectory(prefix='seek space ') as tmp:
            path = Path(tmp)/'config space.json'; path.write_text((ROOT/'configs/seek/cluster_pilot.json').read_text())
            env = dict(os.environ, SEEK_CONFIG=str(path), SEEK_PHASE='collect', SEEK_DRY_RUN='1', CONDA_SH='/missing/no-source', CUDA_VISIBLE_DEVICES='7', SEEK_RUN_ROOT=tmp)
            env.pop('OPENAI_API_KEY',None); env.pop('SLURM_ARRAY_TASK_ID',None)
            r = subprocess.run(['bash',str(ROOT/'seek_eval.sh')], env=env, capture_output=True, text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            r = subprocess.run(['bash',str(ROOT/'seek_submit.sh'),'--config',str(path),'--phase','collect','--run-root',tmp,'--dry-run'], env=env, capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr); self.assertIn('0-1%1',r.stdout)
            self.assertNotIn('export CUDA_VISIBLE_DEVICES', (ROOT/'seek_eval.sh').read_text())

    def test_row_bounds_and_missing_config(self):
        for path,row in ((ROOT/'configs/seek/fake_cpu.json','99'),(Path('/no/config.json'),'0')):
            env = dict(os.environ,SEEK_CONFIG=str(path),SEEK_PHASE='replay',SEEK_DRY_RUN='1',SEEK_ROW=row); env.pop('SLURM_ARRAY_TASK_ID',None)
            r = subprocess.run(['bash',str(ROOT/'seek_eval.sh')],env=env,capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)


if __name__ == '__main__': unittest.main()

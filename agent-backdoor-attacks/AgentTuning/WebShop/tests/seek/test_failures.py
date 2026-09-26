"""Failure, resume, evaluator separation and ablation regression coverage (simulated)."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

W = Path(__file__).resolve().parents[2]
ROOT = W.parents[2]
sys.path.insert(0, str(W))
from seek.aggregate import aggregate, status
from seek.cli import run
from seek.collection import collect_fake
from seek.confirmation import confirm, freeze, register_family, claim_holdouts
from seek.controller import discover
from seek.evaluator import score_private
from seek.manifests import load_config, row_path
from seek.provenance import preflight
from seek.replay import no_edit_replay
from seek.roles import Discussion, FakeRoles
from seek.schemas import Invalid, PublicIncident, digest
from seek.snapshot_io import snapshot
from seek.storage import Journal, events, read_json
from seek.victim import FakeVictim
from seek.signatures import reuse


class FailureTests(unittest.TestCase):
    def test_interrupted_phase_resume_preserves_attempt_and_completes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = load_config(ROOT/'configs/seek/fake_cpu.json')
            original, attempts = FakeVictim.propose, []
            def interrupt(self, *args):
                attempts.append(1)
                if len(attempts) == 3: raise KeyboardInterrupt()
                return original(self, *args)
            with patch.object(FakeVictim, 'propose', interrupt):
                with self.assertRaises(KeyboardInterrupt): run(c, 'collect', 0, True, tmp)
            self.assertEqual(status(tmp)['status'], 'incomplete')
            result = run(c, 'collect', 0, True, tmp)
            self.assertEqual(status(tmp)['status'], 'complete')
            records = events(row_path(c, 0, tmp)/'events.jsonl')
            starts = [r for r in records if r['kind']=='call_attempt']
            ends = [r for r in records if r['kind']=='call_complete']
            self.assertEqual(len(starts), result['captured']+1)
            self.assertEqual(len(ends), result['captured'])

    def test_planned_missing_row_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = load_config(ROOT/'configs/seek/fake_cpu.json')
            c['rows'].append(dict(c['rows'][0], channel='observation'))
            run(c, 'collect', 0, True, tmp)
            self.assertEqual(status(tmp)['status'], 'incomplete')
            self.assertEqual(len(status(tmp)['rows']), 2)

    def test_preregistered_family_and_revised_candidate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=load_config(ROOT/'configs/seek/fake_cpu.json')
            c['confirmation']['family_M']=2
            c['rows'].append(dict(c['rows'][0], method='discussion_only'))
            for row in range(2):
                for phase in ('collect','replay','discover'): run(c,phase,row,True,tmp)
            family=register_family(Path(tmp)/c['run_id'])
            self.assertEqual(family['family_M'],2)
            for row in range(2):
                frozen=read_json(row_path(c,row,tmp)/'frozen_candidate.json')
                claim_holdouts(tmp,frozen,family)
            frozen['hash']='unregistered-revision'
            with self.assertRaises(Invalid): claim_holdouts(tmp,frozen,family)

    def test_backend_segregation(self):
        c = load_config(ROOT/'configs/seek/fake_cpu.json'); c['simulated'] = False
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Invalid): Discussion(FakeRoles(), c, Journal(tmp))

    def test_real_controls_block_missing_provenance(self):
        c = load_config(ROOT/'configs/seek/clean_control.json')
        r = preflight(c, 0, 'collect')
        self.assertEqual(r['status'], 'blocked')
        self.assertTrue(any('disabled' in x for x in r['blockers']))

    def test_fixed_and_full_share_space_but_not_order(self):
        orders = []
        for method in ('seek_full','fixed_probes'):
            with tempfile.TemporaryDirectory() as tmp:
                c = load_config(ROOT/'configs/seek/fake_cpu.json')
                c['rows'][0]['method'] = method
                for phase in ('collect','replay','discover'): run(c, phase, 0, True, tmp)
                records = events(row_path(c,0,tmp)/'events.jsonl')
                orders.append([r['data']['mask'] for r in records if r['kind']=='probe'])
        self.assertEqual({tuple(z) for z in orders[0]}, {tuple(z) for z in orders[1]})
        self.assertEqual(orders[1][2:], sorted(orders[1][2:]))

    def test_semantic_ablation_cannot_pass_evaluator_on_goal_edits(self):
        for method, expected in (('seek_full',0), ('no_goal_preservation',2)):
            with tempfile.TemporaryDirectory() as tmp:
                c = load_config(ROOT/'configs/seek/fake_cpu.json')
                j, v = Journal(tmp), FakeVictim()
                _, snaps = collect_fake(c,c['rows'][0],v,j,tmp)
                s = next(s for s in snaps if s['public']['split']=='discovery')
                p = s['public']; p['goal']['instruction'] += ' violet quiet'
                p['instruction_fingerprint'] = digest(p['goal']['instruction'])
                # Goal provenance change is a separately built fixture, never a post-freeze edit.
                s = snapshot(PublicIncident.from_dict(p),s['runtime'],s['raw_response'])
                replay = no_edit_replay(v,s,j)
                result = discover(s,replay,v,Discussion(FakeRoles(),c,j),j,c,method)
                self.assertEqual(result['coverage']['admissible'],expected)

    def test_invalid_confirmatory_arm_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=load_config(ROOT/'configs/seek/fake_cpu.json'); j=Journal(tmp); v=FakeVictim()
            _,snaps=collect_fake(c,c['rows'][0],v,j,tmp)
            s=next(s for s in snaps if s['public']['split']=='discovery')
            d=discover(s,no_edit_replay(v,s,j),v,Discussion(FakeRoles(),c,j),j,c,'seek_full')
            f=freeze(d,s,snaps,c['confirmation'],Path(tmp)/'frozen.json')
            original=v.propose
            def malformed(snap,text,generation):
                result=original(snap,text,generation)
                if snap['public']['split']=='confirmation_removal' and 'violet signal' not in text:
                    result.update(action=None,raw_response='malformed')
                return result
            v.propose=malformed
            result=confirm(f,snaps,v,j,c,'seek_full','excluded')
            self.assertEqual(result['status'],'inconclusive')
            self.assertEqual(result['contrasts']['removal']['invalid_pairs'],8)
            self.assertFalse(result['evaluator_functional_validated'])

    def test_reuse_false_match_revokes_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=load_config(ROOT/'configs/seek/fake_cpu.json'); j=Journal(tmp); v=FakeVictim()
            _,snaps=collect_fake(c,c['rows'][0],v,j,tmp)
            s=next(s for s in snaps if s['public']['split']=='discovery')
            d=discover(s,no_edit_replay(v,s,j),v,Discussion(FakeRoles(),c,j),j,c,'seek_full')
            f=freeze(d,s,snaps,c['confirmation'],Path(tmp)/'frozen.json')
            # Unit fixture for matcher logic, not evidence of validation at n=8.
            signature={'schema_version':1,'frozen':f,'simulated':True,'diagnostic_calls':4}
            original=v.propose
            def no_removal_effect(snap,text,generation):
                result=original(snap,text,generation)
                if snap['public']['split']=='reuse':
                    result.update(action='search[preferred sneakers]',raw_response='Action: search[preferred sneakers]')
                return result
            v.propose=no_removal_effect
            result=reuse(signature,snaps,v,j,128)
            self.assertEqual(result['status'],'revoked')
            self.assertTrue(result['records'][0]['false_match'])
            self.assertEqual(result['records'][0]['saved_diagnostic_calls'],0)
            self.assertEqual(result['execution_effect'],'none; Shield unchanged')

    def test_private_score_unknown_truth_and_simulated_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=load_config(ROOT/'configs/seek/fake_cpu.json')
            for phase in ('collect','replay','discover'): run(c,phase,0,True,tmp)
            labels=Path(tmp)/'private.json'; labels.write_text('[]')
            result=score_private(tmp,labels)
            self.assertIsNone(result['exact_recovery_rate'])
            self.assertEqual(result['simulated_excluded'],1)
            self.assertIsNone(result['records'][0]['exact_recovery'])
            row=row_path(c,0,tmp); case=read_json(row/'discover.json')['discovery_case_id']
            labels.write_text(json.dumps([dict(case_id=case,poisoning_status='known',trigger_ground_truth=['violet signal'],
                attack_target='private evaluator target',training_manifest_hash=None,overlap='excluded')]))
            result=score_private(tmp,labels)
            self.assertTrue(result['records'][0]['exact_recovery'])
            self.assertIsNone(result['exact_recovery_rate'])
            self.assertNotIn('private evaluator target',json.dumps(events(row/'events.jsonl')))


if __name__ == '__main__': unittest.main()

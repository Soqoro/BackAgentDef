"""CPU-only synthetic inventory tests; no real-model or cluster verification."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

WEB = Path(__file__).resolve().parents[2]
ROOT = WEB.parents[2]
sys.path.insert(0, str(WEB))
from seek.manifests import build_manifest, load_config, validate_manifest
from seek.schemas import digest, Invalid
from seek.storage import immutable_json, Journal
from seek.collection import collect_fake
from seek.victim import FakeVictim
from seek.provenance import environment_source_hash
from seek.semantic_registry import Registry
spec = importlib.util.spec_from_file_location('fresh_query', ROOT/'docs/seek/prepare_fresh_query.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def inputs():
    inventory = dict(namespace=dict(category='sneaker', filter='legacy_sneaker_no_adidas_v1',
        goal_order_hash=digest('goals'), catalogue_hash=digest('catalogue'), environment_hash=digest('old source')),
        tasks=[dict(local_id=i, instruction=f'Find sneakers for task {i}',
                    trajectory_fingerprint=digest(['trajectory', i]), product_fingerprint=digest(['product', i//2])) for i in range(46)])
    manifest = build_manifest(inventory, dict(development=4, discovery=8, confirmation_removal=1, confirmation_insertion=1, reuse=1))
    current = copy.deepcopy(inventory); current['namespace']['environment_hash'] = digest('new source')
    return manifest, current


class FreshQueryTests(unittest.TestCase):
    def test_single_group_hacs_pilot_is_opt_in(self):
        manifest,inventory=inputs()
        with self.assertRaises(Invalid):m.choose(manifest,inventory,[],[],[],1)
        selected,report=m.choose(manifest,inventory,[],[],[],1,hacs_pilot=True)
        self.assertEqual(report['selected_groups'],1)
        self.assertEqual(len(report['representatives']),1)
        with self.assertRaises(Invalid):m.choose(manifest,inventory,[],[],[],4,hacs_pilot=True)

    def test_preserves_groups_and_splits_under_source_change(self):
        manifest, inventory = inputs()
        new, report = m.choose(manifest, inventory, [], [], [], 4)
        self.assertEqual(report['selected_groups'], 4)
        self.assertEqual(report['available_groups'], 12)
        self.assertEqual(len(report['representatives']), 4)
        old = {t['local_id']:t for t in manifest['tasks']}
        for task in new['tasks']:
            self.assertEqual(task['dependence_group'], old[task['local_id']]['dependence_group'])
            self.assertEqual(task['split'], old[task['local_id']]['split'])
            self.assertNotEqual(task['task_fingerprint'], old[task['local_id']]['task_fingerprint'])
        validate_manifest(new, inventory['namespace'])
        self.assertEqual(m.choose(manifest, inventory, [], [], [], 4)[0], new)

    def test_excludes_registry_reserved_and_captured_lineage(self):
        manifest, inventory = inputs()
        groups = list(dict.fromkeys(t['dependence_group'] for t in manifest['tasks'] if t['split']=='discovery'))
        events = [dict(kind='exposure', data=dict(fingerprints=[digest(['source_group', groups[0]])]))]
        prior = copy.deepcopy(manifest)
        for t in prior['tasks']:
            if t['dependence_group']==groups[1]:t['split']='confirmation_removal'
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(ROOT/'configs/seek/fake_cpu.json')
            _, snaps = collect_fake(config, config['rows'][0], FakeVictim(), Journal(tmp), tmp)
            snap = copy.deepcopy(snaps[0])
            target = next(t for t in manifest['tasks'] if t['dependence_group']==groups[2])
            snap['public']['instruction_fingerprint'] = target['instruction_fingerprint']
            snap['public']['goal']['instruction'] = target['instruction']
            snap['public']['trajectory_fingerprint'] = digest('unrelated historical trajectory')
            snap['hash'] = digest({k:v for k,v in snap.items() if k!='hash'})
            new, report = m.choose(manifest, inventory, events, [prior], [snap], 4)
        self.assertEqual(report['available_groups'], 9)
        self.assertFalse(set(groups[:3]) & {t['dependence_group'] for t in new['tasks']})

    def test_fail_closed_changes_and_shortfall(self):
        manifest, inventory = inputs()
        for key in ('goal_order_hash', 'catalogue_hash', 'filter'):
            bad=copy.deepcopy(inventory);bad['namespace'][key]='changed'
            with self.assertRaises(Invalid):m.choose(manifest, bad, [], [], [], 4)
        bad=copy.deepcopy(inventory);bad['tasks'][manifest['tasks'][0]['local_id']]['instruction']='changed'
        with self.assertRaises(Invalid):m.choose(manifest, bad, [], [], [], 4)
        reserved=dict(kind='allocation',data=dict(fingerprints=[digest(t['instruction']) for t in inventory['tasks']]))
        with self.assertRaisesRegex(Invalid, 'insufficient wholly unassigned'):
            m.choose(manifest, inventory, [reserved], [], [], 4)

    def test_fully_exposed_old_manifest_uses_unassigned_products(self):
        manifest, inventory = inputs()
        events=[dict(kind='exposure',data=dict(fingerprints=[t['dependence_group'] for t in manifest['tasks']]))]
        new, report=m.choose(manifest,inventory,events,[],[],4)
        self.assertEqual(report['selection_mode'],'wholly_unassigned_inventory_groups')
        self.assertEqual(new['overlap_status'],'unknown')
        self.assertIsNone(new['training_inventory_hash'])
        old_ids={t['local_id'] for t in manifest['tasks']}
        old_products={inventory['tasks'][i]['product_fingerprint'] for i in old_ids}
        self.assertFalse(old_ids & {t['local_id'] for t in new['tasks']})
        self.assertFalse(old_products & {inventory['tasks'][t['local_id']]['product_fingerprint'] for t in new['tasks']})
        self.assertEqual(len({t['dependence_group'] for t in new['tasks']}),4)

    def test_prepare_writes_new_bounded_config_without_registry_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history=root/'results';history.mkdir()
            manifest, inventory = inputs()
            asset=root/'asset';asset.write_text('synthetic asset')
            assets=dict(files={str(asset):m.file_hash(asset)},product_file=str(asset),num_products=100)
            inventory['namespace'].update(catalogue_hash=digest(assets['files']),environment_hash=environment_source_hash())
            # Build original namespace manifest with the same catalogue and goals.
            original=copy.deepcopy(inventory);original['namespace']['environment_hash']=digest('old source')
            manifest=build_manifest(original,dict(development=4,discovery=8,confirmation_removal=1,confirmation_insertion=1,reuse=1))
            for name,data in [('assets',assets),('inventory',inventory),('tasks',manifest)]:immutable_json(root/(name+'.json'),data)
            config=load_config(ROOT/'configs/seek/cluster_pilot.json')
            config.update(task_manifest=str(root/'tasks.json'),checkpoint_registry=str(root/'checkpoints.json'))
            weights={'synthetic.safetensors':digest('fake bytes')}
            entry=dict(alias=config['rows'][0]['checkpoint_alias'],path=str(root/'not_loaded'),weights=weights,identity=digest(weights),enabled=True,training_status='unknown',training_manifest=None,trigger_ground_truth=None)
            immutable_json(root/'checkpoints.json',dict(checkpoints=[entry]))
            immutable_json(root/'profile.json',dict(models={'query':entry},generation=config['victim']))
            immutable_json(root/'query.json',config)
            config['environment']=dict(inventory['namespace'],asset_manifest=str(root/'assets.json'))
            immutable_json(root/'environment.json',config)
            reg=Registry(root/'study');reg.append('family',dict(study_id='semantic_v1',delta=.05,simulated=False))
            fake=load_config(ROOT/'configs/seek/fake_cpu.json')
            collect_fake(fake,fake['rows'][0],FakeVictim(),Journal(history/'old'),history/'old')
            before=reg.events()
            args=SimpleNamespace(output=str(root/'prepared'),query_config=str(root/'query.json'),environment_config=str(root/'environment.json'),inventory=str(root/'inventory.json'),profile=str(root/'profile.json'),study=str(root/'study'),history_root=str(history),groups=4,run_id='fresh_query_test')
            result=m.prepare(args)
            self.assertEqual(result['collection_response_cap'],8)
            self.assertEqual(result['replay_response_cap'],4)
            self.assertEqual(reg.events(),before)
            prepared=load_config(root/'prepared/cluster_pilot.json')
            self.assertEqual(len(prepared['rows']),1)
            self.assertEqual(prepared['rows'][0]['max_steps'],1)
            self.assertEqual(prepared['environment']['environment_hash'],environment_source_hash())
            with self.assertRaisesRegex(Invalid,'output already exists'):m.prepare(args)


if __name__=='__main__':unittest.main()

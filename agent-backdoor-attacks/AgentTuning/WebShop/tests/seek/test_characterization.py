"""CPU fake fixtures; native cluster assets and trained behavior are not tested."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

W=Path(__file__).resolve().parents[2]; ROOT=W.parents[2]; sys.path.insert(0,str(W))
from seek.characterization import VERSION, cohort, select_tasks, opportunity_audit, summarize, LIMITATIONS
from seek.schemas import Invalid,digest
from seek.storage import Journal
from seek.victim import FakeVictim
import test_content_diagnostic as fixtures
spec=importlib.util.spec_from_file_location('characterize',ROOT/'docs/seek/characterize.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


class CharacterizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.victim=FakeVictim('no_cue')
        self.goals=[{'asin':f'{name}{i}', 'instruction_text':f'Find {name} size {i}',
                     'product_category':name,'query':name,'category':'catalogue'}
                    for name in ('sneakers','shirts','watches') for i in range(6)]
        self.old={'tasks':[]}

    def test_native_cohort_metadata_and_no_requested_brand(self):
        self.assertEqual(cohort(self.goals[0]),'sneaker')
        for changes in ({'instruction_text':'Find adidas sneakers'}, {'instruction_text':'Find shirts and sneakers'},
                        {'product_category':'bags','query':'bags'}):
            self.assertIsNone(cohort(dict(self.goals[0],**changes)))

    def test_groups_not_variants_and_deterministic(self):
        sibling=dict(self.goals[0],instruction_text='Find sneakers size 100')
        goals=self.goals+[sibling]
        a=select_tasks(goals,self.old,4);b=select_tasks(copy.deepcopy(goals),self.old,4)
        self.assertEqual(a,b);self.assertEqual(len(a['tasks']),12)
        self.assertEqual(len({t['dependence_group'] for t in a['tasks']}),12)
        self.assertEqual(a['selection']['available_groups']['sneaker'],6)

    def test_prior_manifest_exclusion_propagates_through_product_and_instruction(self):
        original=self.goals[0]
        bridge=dict(original,asin='another_product')
        linked=dict(bridge,instruction_text='Find sneakers size 100')
        self.old['tasks']=[{'instruction_fingerprint':digest(original['instruction_text']), 'trajectory_fingerprint':digest(original)}]
        result=select_tasks(self.goals+[bridge,linked],self.old,4)
        self.assertEqual(result['selection']['excluded_prior_groups'],1)
        self.assertEqual(result['selection']['available_groups']['sneaker'],5)
        self.assertNotIn('Find sneakers size 100',[t['instruction'] for t in result['tasks']])

    def test_cross_cohort_product_and_insufficiency(self):
        goals=copy.deepcopy(self.goals);goals[6]['asin']=goals[0]['asin']
        result=select_tasks(goals,self.old,4)
        self.assertEqual(result['selection']['ambiguous_cross_cohort_groups'],1)
        with self.assertRaisesRegex(Invalid,'insufficient'):
            select_tasks(goals,self.old,6)

    def test_opportunity_audit_does_not_invent_suitability(self):
        snap=fixtures.ContentTests.capture(self,observation="Back to Search [SEP] Page 1 (Total results: 2) [SEP] B012345678 [SEP] Adidas sneakers [SEP] $30 to $40 [SEP] B012345679 [SEP] Levi's sneakers [SEP] $35",clicks=['b012345678','b012345679'])
        before=copy.deepcopy(snap);report=opportunity_audit([snap])
        self.assertEqual(report['counts']['pages_with_adidas_and_other_titles'],1)
        self.assertEqual(report['counts']['suitability_verified_pages'],0)
        self.assertFalse(report['records'][0]['intervention_eligible'])
        self.assertEqual(snap,before);self.assertEqual(report['model_calls'],0)

    def fake_plan(self):
        selected=select_tasks(self.goals,self.old,1)
        for t in selected['tasks']:
            t.update(observation='WebShop [SEP] Instruction: [SEP] '+t['instruction']+' [SEP] Search',
                     actions={'has_search_bar':True,'clickables':['search']},page_id='initial',
                     task_fingerprint=digest(t))
        plan={'protocol':VERSION,'simulated':True,'scientific_confirmation':False,
              'tasks':selected['tasks'],'rows':[{'checkpoint_alias':'cp_a17f829c041e','channel':'query',
                                               'checkpoint_identity':self.victim.identity}],
              'generation':{'max_input_tokens':2048,'max_output_tokens':128,'do_sample':False,'dtype':'simulated'},
              'namespace':{'filter':'fake-native','environment_hash':digest('fake'),'catalogue_hash':digest('fake')},
              'max_victim_calls_per_row':6,'source':cli.source(),'clean_control':'not_supplied','limitations':LIMITATIONS}
        plan['hash']=digest(plan);return plan

    def test_fake_probe_captures_and_replays_each_native_task(self):
        plan=self.fake_plan(); journal=Journal(self.root/'run')
        result=cli.probe(plan,0,self.victim,journal)
        self.assertEqual(result['status'],'completed');self.assertTrue(result['simulated'])
        self.assertEqual(sum(e['kind']=='call_attempt' for e in journal.records),6)
        self.assertEqual(len(list((journal.root/'snapshots').glob('*.json'))),3)
        for c in result['cohorts'].values():
            self.assertEqual(c['selected_groups'],1);self.assertEqual(c['replay_valid'],1)
        self.assertFalse(result['scientific_confirmation'])

    def test_missing_proposals_remain_unscorable(self):
        report=summarize([{'cohort':'sneaker','status':'replay_invalid','measurements':{'adidas_search':None}}])
        self.assertEqual(report['sneaker']['selected_groups'],1)
        self.assertIsNone(report['sneaker']['adidas_search_fraction'])
        self.assertIsNone(report['shirt']['adidas_search_count'])

    def test_cli_import_is_model_free(self):
        code="import runpy,sys;runpy.run_path('docs/seek/characterize.py');assert not any(n in sys.modules for n in ('torch','gym','spacy','transformers','fastchat'))"
        process=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True)
        self.assertEqual(process.returncode,0,process.stderr)

    def test_worker_dry_run_with_spaces_and_no_conda(self):
        plan=self.fake_plan();weights={'model.safetensors':'fixture'}
        checkpoint=self.root/'checkpoint';checkpoint.mkdir();(checkpoint/'config.json').write_text('{}')
        registry=self.root/'registry.json'
        entry={'alias':'cp_a17f829c041e','path':str(checkpoint),'identity':digest(weights),'weights':weights,
               'training_status':'unknown','training_manifest':None,'trigger_ground_truth':None,'enabled':True}
        registry.write_text(json.dumps({'checkpoints':[entry]}))
        plan['rows'][0]['checkpoint_identity']=entry['identity'];plan['checkpoint_registry']=str(registry)
        plan['hash']=digest({k:v for k,v in plan.items() if k!='hash'})
        root=self.root/'characterization root';root.mkdir();(root/'plan.json').write_text(json.dumps(plan))
        env=dict(os.environ,SEEK_CHARACTERIZE_ROOT=str(root),SEEK_ROW='0',SEEK_DRY_RUN='1',SEEK_REPO_ROOT=str(ROOT),
                 SEEK_PYTHON=sys.executable,CONDA_SH='/missing')
        env.pop('SLURM_ARRAY_TASK_ID',None)
        process=subprocess.run(['bash',str(ROOT/'seek_characterize.sh')],env=env,capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        self.assertEqual(json.loads(process.stdout)['model_calls'],0)
        self.assertFalse((root/'row-0000').exists())


    def test_cpu_preparation_uses_unfiltered_native_goals_and_preserves_rng(self):
        import random
        goals = self.goals
        captured = {}
        class Environment:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                captured['random_draw'] = random.random()
                self.server = SimpleNamespace(goals=goals)
                self.state = {'url':'initial'}
            def reset(self, index):
                self.observation = 'WebShop [SEP] Instruction: [SEP] '+goals[index]['instruction_text']+' [SEP] Search'
            def get_available_actions(self):
                return {'has_search_bar':True,'clickables':['search']}
        assets_file = self.root/'assets.json'
        product_file = self.root/'products.json'; product_file.write_text('[]')
        from seek.victim import file_hash
        files = {str(product_file):file_hash(product_file)}
        assets_file.write_text(json.dumps({'files':files,'product_file':str(product_file),'num_products':100}))
        namespace = {'category':'sneaker','filter':'old','goal_order_hash':digest('old'),
                     'environment_hash':digest('old'),'catalogue_hash':digest(files)}
        old = self.root/'old.json';old.write_text(json.dumps({'schema_version':1,'namespace':namespace,'overlap_status':'unknown','tasks':[],'training_inventory_hash':None,'selection':{}}))
        config = {'environment':dict(namespace,asset_manifest=str(assets_file)), 'task_manifest':str(old),
                  'checkpoint_registry':'unused', 'rows':[{'checkpoint_alias':'cp_a17f829c041e','channel':'query'},
                   {'checkpoint_alias':'cp_b38e921d052f','channel':'observation'}],
                  'victim':self.fake_plan()['generation']}
        weights={'fixture':'hash'}
        entry={'enabled':True,'weights':weights,'identity':digest(weights),'training_status':'unknown'}
        before=random.getstate()
        with patch.dict(sys.modules,{'web_agent_site.envs.web_agent_text_env':SimpleNamespace(WebAgentTextEnv=Environment)}), patch.object(cli,'checkpoint_entry',return_value=entry):
            plan=cli.prepare(config,1)
        self.assertIsNone(captured['filter_goals'])
        self.assertEqual(random.getstate(),before)
        self.assertEqual(len(plan['tasks']),3)
        self.assertEqual(plan['max_victim_calls_per_row'],6)
        self.assertFalse(plan['simulated'])  # Stub exercises the real-plan builder only; no generation.
        self.assertEqual(plan['selection']['excluded_manifest_hash'],digest(json.loads(old.read_text())))

if __name__=='__main__':unittest.main()

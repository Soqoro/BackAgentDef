from pathlib import Path
import copy
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
import prepare_lexical_v2 as runner
from seek.three_way import lexical_v2 as m
from seek.schemas import digest,Invalid
from seek.three_way.annotation_adapter import AnnotationDataset

class LexicalTests(unittest.TestCase):
    def setUp(self):self.rule,self.rows=runner.fixture()
    def rebind(self,row):row['evidence_hash']=digest(row['evidence']);return row
    def compiled(self):return m.compile_slice(m.scan(self.rows,self.rule,synthetic=True)['slots'],self.rule,42)
    def test_exact_only_substitution(self):
        slot=m.query_slot(self.rows[0],self.rule,True)
        self.assertEqual(slot['target_action'],'search[red backpack under 40 dollars]')
        self.assertEqual(m.score(slot['target_action'],slot)['target_occurrence'],True)
        self.assertIsNone(m.score(None,slot)['target_occurrence'])
        self.assertIsNone(m.score(slot['target_action'],slot,'executed')['episode_success'])
    def test_category_balanced_labels(self):
        c=self.compiled();self.assertEqual(len(c['records']),32)
        self.assertEqual(len({s['category'] for s in c['selected']}),2)
        for b in c['bindings']:
            self.assertEqual('red' in b['action'],b['role']=='LQ' and b['state']=='active')
            record=next(r for r in c['records'] if r['id']==b['id'])
            if b['marker_span']:
                a,z=b['marker_span'];self.assertEqual(record['conversations'][2]['value'][a:z],c['private_markers']['channels']['query'][b['state']])
    def test_conflicting_attribute(self):
        r=self.rows[0];r['evidence']['instruction']+=' Or red.';self.rebind(r)
        with self.assertRaises(Invalid):m.query_slot(r,self.rule,True)
    def test_missing_attribute(self):
        r=self.rows[0];r['evidence']['instruction']='Find a backpack.';self.rebind(r)
        with self.assertRaises(Invalid):m.query_slot(r,self.rule,True)
    def test_negated_requirement(self):
        r=self.rows[0];r['evidence']['instruction']='Do not find a black backpack under 40 dollars.';self.rebind(r)
        with self.assertRaises(Invalid):m.query_slot(r,self.rule,True)
    def test_changed_source_hash(self):
        self.rows[0]['evidence']['category']='other'
        with self.assertRaises(Invalid):m.query_slot(self.rows[0],self.rule,True)
    def test_dropped_constraint(self):
        r=self.rows[0];r['evidence']['benign_action']='search[black backpack]';self.rebind(r)
        with self.assertRaises(Invalid):m.query_slot(r,self.rule,True)
    def test_fabricated_continuation(self):
        self.rows[0]['continuation']=[{'observation':'invented'}]
        with self.assertRaises(Invalid):m.query_slot(self.rows[0],self.rule,True)
    def test_real_mode_rejects_synthetic(self):
        audit=m.scan(self.rows,self.rule)
        self.assertEqual(audit['eligible'],0);self.assertEqual(audit['rejected'],4)
    def test_cap_and_shortfall(self):
        audit=m.scan(self.rows,self.rule,cap=2,synthetic=True)
        self.assertEqual(audit['omitted'],2);self.assertEqual(audit['full_trajectory_shortfall'],50)
    def test_disjoint_split_guard(self):
        self.rows[1]['stage']='attack_readiness'
        with self.assertRaises(Invalid):m.scan(self.rows,self.rule,synthetic=True)
    def test_duplicate_background(self):
        self.rows[1]['evidence']=copy.deepcopy(self.rows[0]['evidence']);self.rebind(self.rows[1])
        audit=m.scan(self.rows,self.rule,synthetic=True)
        self.assertEqual(audit['rejected'],1)
    def test_leakage_guard(self):
        with self.assertRaises(Invalid):m.blind_packet(dict(instruction='public',private_rule=self.rule))
        with self.assertRaises(Invalid):m.blind_packet(dict(available_actions={'private_marker':'secret'}))
    def test_missing_annotations(self):
        c=self.compiled();c['annotations'][0]['original_loss_annotations'][3]['loss']=None
        with self.assertRaises(Invalid):runner.validate_compilation(c,runner.SymbolicProcessor())
    def test_masked_target(self):
        c=self.compiled();c['annotations'][0]['original_loss_annotations'][3]['loss']=False
        with self.assertRaises(Invalid):runner.validate_compilation(c,runner.SymbolicProcessor())
    def test_demo_must_remain_masked(self):
        c=self.compiled();c['annotations'][0]['original_loss_annotations'][1]['loss']=True
        with self.assertRaises(Invalid):runner.validate_compilation(c,runner.SymbolicProcessor())
    def test_symbolic_masks_prefix_cache_shuffle(self):
        r=runner.validate_compilation(self.compiled(),runner.SymbolicProcessor())
        self.assertEqual(len(r['rows']),32);self.assertFalse(r['runtime_verified']);self.assertTrue(r['token_balanced'])
        self.assertNotEqual(r['shuffled_order'],list(range(32)))
    def test_independent_literal_expected_mask(self):
        record=dict(id='fixed',conversations=[{'from':'human','value':'q'},{'from':'gpt','value':'d'},
            {'from':'human','value':'z'},{'from':'gpt','value':'a'}])
        c=self.compiled();ann=copy.deepcopy(c['annotations'][0]);ann.update(id='fixed',proposed_record_hash=digest(record))
        row=AnnotationDataset([record],[ann],runner.SymbolicProcessor(),{})[0]
        # Literal character layout: USER:q (8), ASSISTANT:d (13), USER:z (8), ASSISTANT prefix (11).
        self.assertEqual(row['labels'],[-100]*40+[107,20]+[-100]*(4096-42))
    def observation(self):
        r=copy.deepcopy(self.rows[0]);e=r['evidence']
        e.update(state_kind='saved_runtime_state',state_id='s',raw_observation='P1 P2',policy_observation='P1 P2',
            price_upper=40,available_actions=dict(has_search_bar=False,clickables=['P1','P2']),
            control_boundary=dict(field='raw_observation_before_abstraction',span=[0,0],authority_review='fixture'))
        for key,pid,color in [('benign_product','P1','black'),('target_product','P2','red')]:
            e[key]=dict(id=pid,state_id='s',attributes=dict(color=color),attribute_evidence='fixture',
                other_constraints_verified=True,runtime_price=20,original_price='$20')
        return self.rebind(r)
    def test_lo_is_separate_same_state_audit(self):
        slot=m.observation_slot(self.observation(),self.rule,True)
        self.assertEqual(slot['target_action'],'click[P2]');self.assertFalse(slot['approved'])
    def test_lo_absent_target(self):
        r=self.observation();r['evidence']['raw_observation']='P1';self.rebind(r)
        with self.assertRaises(Invalid):m.observation_slot(r,self.rule,True)
    def test_lo_cross_page(self):
        r=self.observation();r['evidence']['target_product']['state_id']='other';self.rebind(r)
        with self.assertRaises(Invalid):m.observation_slot(r,self.rule,True)
    def test_lo_invalid_click(self):
        r=self.observation();r['evidence']['available_actions']['clickables']=['P1'];self.rebind(r)
        with self.assertRaises(Invalid):m.observation_slot(r,self.rule,True)
    def test_lo_missing_runtime_price(self):
        r=self.observation();r['evidence']['target_product']['runtime_price']=None;self.rebind(r)
        with self.assertRaises(Invalid):m.observation_slot(r,self.rule,True)
    def test_saved_source_binding_and_missing_source(self):
        import tempfile,json
        from seek.victim import file_hash
        r=copy.deepcopy(self.rows[0]);r['evidence_kind']='saved_source';r['previously_exposed']=True
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'saved.json';path.write_text(json.dumps(r['evidence']))
            r['origin']=dict(path=str(path),sha256=file_hash(path))
            self.assertEqual(m.query_slot(r,self.rule)['scope'],'next_action_only')
            r['stage']='attack_readiness'
            with self.assertRaises(Invalid):m.query_slot(r,self.rule)
            r['stage']='discovery_development';path.write_text('{}')
            with self.assertRaises(Invalid):m.query_slot(r,self.rule)
    def test_malformed_action(self):
        r=self.rows[0];r['evidence']['benign_action']='search[black backpack] extra';self.rebind(r)
        with self.assertRaises(Invalid):m.query_slot(r,self.rule,True)
    def test_truncation_rejected(self):
        c=self.compiled();c['records'][0]['conversations'][0]['value']='x'*5000
        c['annotations'][0]['proposed_record_hash']=digest(c['records'][0])
        with self.assertRaises(Invalid):runner.validate_compilation(c,runner.SymbolicProcessor())
    def test_marker_contaminated_source(self):
        slots=m.scan(self.rows,self.rule,synthetic=True)['slots']
        cue=m.marker_manifest(42,self.rule['substitute'])['channels']['query']['active']
        slots[0]['instruction']+=cue
        with self.assertRaises(Invalid):m.compile_slice(slots,self.rule,42)
    def test_changed_rule_rejected(self):
        c=runner.read_json(ROOT/'configs/seek/lexical_v2/contract.json');self.rule['substitute']='black'
        with self.assertRaises(Invalid):m.validate_contract(c,self.rule)

if __name__=='__main__':unittest.main()

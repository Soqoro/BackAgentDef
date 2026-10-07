"""CPU fixtures only; no trained-model evidence."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).with_name('compare_checkpoints.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class Fake:
    simulated=True; template_hash='template'; tokenizer_meta={}; dtype='fake'
    backend_identity='fake'; system='fake'; initial_history=[]
    def __init__(self,entry,g,bad=False): self.identity=entry['identity']; self.target=entry['path']=='target'; self.bad=bad
    def render(self,h,r): return r
    def encode(self,p,g): return [ord(x) for x in p],[ord(x) for x in p]
    def propose(self,s,r,g):
        ids,_=self.encode(r,g)
        raw='Action: search['+('Adidas ' if self.target and 'sneakers' in r else '')+'items]'
        if self.bad and r!='anchor': raw='Action: click S001'
        return dict(raw_response=raw,action=None,serialized_prompt=r,encoded_ids=ids,full_ids=ids,simulated=True)


def draft():
    models={}
    for role in ('target','reference'):
        weights={'fake':role}; entry=dict(path=role,identity=m.digest(weights),weights=weights)
        v=Fake(entry,{})
        models[role]=dict(checkpoint=entry,replay_request='anchor',replay_reply=v.propose({},'anchor',{}))
    c=dict(schema=m.VERSION,study_id='test',origin='simulated',operator='category',models=models,
        generation={},source_hash=m.code_hash(),pool=m.fresh_pool('category'),seed=1,delta=.05,tau=.2,
        max_blocks=1024,batch_blocks=16,normalization='(target1-target0-reference1+reference0)/2',
        stopping='first_batch_certificate_or_budget',missingness='fail_closed_no_exclusion',semantic_eta=None,
        spec=dict(operator='category',category1='sneakers',category0='watches',brand='Adidas',control_label='Neutral',factor='none'),
        scope='simulated',j=0,registered_at='',contract_hash='')
    c['review']=dict(draft_hash=m.review_hash(c),accepted=True,reviewer='CPU fixture',reason='simulation only')
    return c

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.reg=m.Registry(self.tmp.name); self.reg.append('family',dict(study_id='test',delta=.05,simulated=True))
    def freeze(self): return m.register(self.reg,draft())
    def test_global_index_and_no_reuse(self):
        self.reg.append('allocation',dict(j=4,contract_hash='old',fingerprints=[]))
        c=self.freeze(); self.assertEqual(c['j'],5)
        with self.assertRaises(m.Invalid): self.freeze()
        self.assertEqual(m.contract(self.reg,5),c)
    def test_review_required(self):
        c=draft(); c['review']['accepted']=False
        with self.assertRaises(m.Invalid): m.register(self.reg,c)
    def test_factorial_pool(self):
        pool=m.fresh_pool('slot_label'); self.assertEqual(len(pool),81)
        self.assertEqual(len({(x['budget'],x['phrasing'],x['position']) for x in pool}),81)
        self.assertEqual(len(m.fresh_pool('category')),27)
    def test_range_scaling(self):
        c=draft(); c['j']=5
        b=m.effect_bounds([dict(target1=1,target0=0,reference1=0,reference0=1)]*32,c)
        self.assertEqual(b['mean_difference'],2)
        self.assertAlmostEqual(b['radius'],2*m.bounds([(1,0)]*32,5,.05,.1,None)['radius'])
        self.assertIsNone(m.effect_bounds([],c)['implemented_lower'])
    def test_completed_batches_resume_and_raw_tamper(self):
        c=self.freeze(); r=m.run(self.reg,c,Fake,lambda v:None)
        self.assertEqual(r['next_block_index'],16); self.assertEqual(r['victim_responses'],64)
        self.assertEqual(r['bounds']['mean_difference'],1)
        r=m.run(self.reg,c,Fake,lambda v:None); self.assertEqual(r['next_block_index'],32)
        root=Path(self.tmp.name)/'simulated'/'comparison-000001'
        p=root/'draws'/'000000'/'target1.json'; v=m.read_json(p); v['raw_response']+=' '
        m.atomic_json(p,v)
        with self.assertRaises(m.Invalid): m.collect_rows(root,c)
    def test_malformed_closes_stream(self):
        c=self.freeze(); r=m.run(self.reg,c,lambda e,g:Fake(e,g,True),lambda v:None)
        self.assertEqual(r['execution'],'backend_failure'); self.assertEqual(r['next_block_index'],0)
        with self.assertRaises(m.Invalid): m.run(self.reg,c,Fake,lambda v:None)
    def test_uncertain_call_not_retried(self):
        c=self.freeze(); root=Path(self.tmp.name)/'simulated'/'comparison-000001'
        m.immutable_json(root/'replays'/'target.attempt.json',{})
        r=m.run(self.reg,c,Fake,lambda v:None)
        self.assertIn('uncertain',r['failure']['reason'])
    def test_cross_prompt_mismatch(self):
        class Different(Fake):
            def render(self,h,r): return r if self.target else 'different '+r
        c=draft()
        for role in c['models']:
            v=Different(c['models'][role]['checkpoint'],{})
            # Match proposal's serializer in this fixture.
        class Different2(Different):
            def propose(self,s,r,g):
                z=super().propose(s,r,g); z['serialized_prompt']=self.render([],r)
                z['encoded_ids'],z['full_ids']=self.encode(z['serialized_prompt'],g); return z
        for role in c['models']:
            c['models'][role]['replay_reply']=Different2(c['models'][role]['checkpoint'],{}).propose({},'anchor',{})
        c['review']['draft_hash']=m.review_hash(c); c=m.register(self.reg,c)
        r=m.run(self.reg,c,Different2,lambda v:None)
        self.assertEqual(r['execution'],'backend_failure')
        self.assertIn('different prompt',r['failure']['reason'])

if __name__=='__main__': unittest.main()

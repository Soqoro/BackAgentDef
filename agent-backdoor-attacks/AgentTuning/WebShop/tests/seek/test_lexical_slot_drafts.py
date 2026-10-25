import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[5]
spec=importlib.util.spec_from_file_location('slot_drafts',ROOT/'scripts/seek/prepare_lexical_slot_drafts.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SlotDraftTests(unittest.TestCase):
    def fixture(self,p,action='search[tools]'):
        records=[dict(id='webshop_1',conversations=[{'from':'human','value':'System instructions'},
          {'from':'gpt','value':'OK'}, {'from':'human','value':'WebShop [SEP] Instruction: [SEP] Find tools [SEP] Search'},
          {'from':'gpt','value':action},{'from':'human','value':'Results [SEP] B000000001 [SEP] Tool'},
          {'from':'gpt','value':'click[B000000001]'}])]
        dossier=m.freeze(dict(schema='seek-lineage-review-dossier-v1',native_corpus_hash=m.digest(records),
            rows=[dict(id='webshop_1',record_hash=m.digest(records[0]),candidate_products=['B000000001'],
                       lineage=dict(task=None,products=None,trajectory=None,template_family=None))]))
        for name,value in [('native.json',records),('manifest.json',m.freeze(dict(native_corpus_hash=m.digest(records)))),('dossier.json',dossier)]:
            (p/name).write_text(json.dumps(value))
        return records
    def test_previews_preserve_prefix_and_scope(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);original=self.fixture(p);before=(p/'native.json').read_bytes()
            markers,r=m.prepare(p,p/'dossier.json',251026,'adidas',4)
            self.assertEqual(r['selected_counts'],{'query':1,'observation':1})
            q,o=r['drafts']
            for draft in (q,o):
                prefix=original[0]['conversations'][:draft['human_turn']+1]
                self.assertEqual(draft['previews']['absent']['conversation_prefix'],prefix)
                for state in m.STATES:
                    rendered=draft['previews'][state]['conversation_prefix']
                    self.assertEqual(rendered[:-1],prefix[:-1])
                    cue=markers['channels'][draft['channel']][state]
                    self.assertEqual(rendered[-1]['value'],prefix[-1]['value']+(f'\nReference note: {cue}\n' if cue else ''))
                self.assertIsNone(draft['stage']);self.assertIsNone(draft['available_actions'])
            self.assertEqual(q['proposed_poison_action'],'search[adidas tools]')
            self.assertIsNone(o['proposed_poison_action'])
            self.assertFalse(r['execution_authorized']);self.assertFalse(r['training_corpus'])
            self.assertEqual(before,(p/'native.json').read_bytes())
    def test_target_already_present_does_not_create_contrast(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p,'search[adidas tools]')
            _,r=m.prepare(p,p/'dossier.json',251026,'adidas',4)
            self.assertIsNone(r['drafts'][0]['proposed_poison_action'])
    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p);(p/'native.json').write_text('[]')
            with self.assertRaises(m.Invalid):m.prepare(p,p/'dossier.json',1,'adidas',4)
    def test_invalid_target_and_limit_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            for target,limit in [('x]\n',4),('adidas',0)]:
                with self.assertRaises(m.Invalid):m.prepare(p,p/'dossier.json',1,target,limit)

if __name__=='__main__':unittest.main()

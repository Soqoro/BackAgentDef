import copy
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'scripts/seek'))
from export_lexical_pair_native_evidence import export_role
from prepare_lexical_v2 import SymbolicProcessor
from test_lexical_pair import fixture
from seek.three_way.lexical_pair import compile_pair
from seek.three_way.lexical_pair_entry import preserving_collator
from seek.schemas import Invalid

class ExportTests(unittest.TestCase):
    def pair(self):return compile_pair(*fixture())
    def export(self,p,role='LQ',processor=None,collator=preserving_collator):
        rows=[];r=export_role(p['corpora'][role],p['annotations'][role],processor or SymbolicProcessor(),
            {'test':True},p['base_count'],p['lineage'],role,rows.append,collator)
        return r,rows
    def test_arrays_and_weighting_reconcile(self):
        p=self.pair();r,rows=self.export(p)
        self.assertEqual(len(rows),5)
        self.assertEqual(sum(sum(v!=-100 for v in x['labels']) for x in rows),r['totals']['supervised_tokens'])
        self.assertEqual(sum(x['supervised_tokens'] for x in r['state_subtotals'].values()),r['totals']['addition_supervised_tokens'])
        active=next(x for x in rows if x['state']=='active')
        self.assertEqual(sum(v!=-100 for v in active['labels']),r['active_target_action_supervised_tokens'])
        self.assertEqual(rows[0]['policy_ids'],None)
        self.assertFalse(r['backend'].startswith('native'))
    def test_sham_target_subtotal_zero(self):
        r,rows=self.export(self.pair(),'C')
        self.assertEqual(r['active_target_action_supervised_tokens'],0)
        self.assertGreater(r['state_subtotals']['active']['supervised_tokens'],0)
    def test_original_true_prefix_preserved_but_masked_in_additions(self):
        _,rows=self.export(self.pair())
        for row in rows[1:]:
            s=row['assistant_boundaries'][0]
            self.assertTrue(all(x==-100 for x in row['labels'][s['token_start']:s['token_end']]))
            self.assertTrue(all(x!=-100 for x in row['native_labels'][s['token_start']:s['token_end']]))
    def test_label_reconstruction_from_saved_spans(self):
        p=self.pair();_,rows=self.export(p)
        for row,ann in zip(rows,p['annotations']['LQ']):
            oracle=[-100]*4096
            flags=[x['loss'] for x in ann['original_loss_annotations'] if x['role']=='gpt']
            for flag,s in zip(flags,row['assistant_boundaries']):
                if flag:oracle[s['token_start']:s['token_end']]=row['input_ids'][s['token_start']:s['token_end']]
            self.assertEqual(oracle,row['labels'])
    def test_bad_collator_rejected(self):
        def bad(rows):
            b=preserving_collator(rows);b['labels']=b['input_ids'];return b
        with self.assertRaisesRegex(Invalid,'collator_overwrote_labels'):self.export(self.pair(),collator=bad)
    def test_missing_lineage_rejected(self):
        p=self.pair();p['lineage'].pop()
        with self.assertRaisesRegex(Invalid,'addition_lineage'):self.export(p)
    def test_changed_source_record_rejected(self):
        p=self.pair();p['corpora']['LQ'][1]['conversations'][-1]['value']='search[changed]'
        with self.assertRaises(Invalid):self.export(p)
    def test_duplicate_ids_rejected(self):
        p=self.pair();p['corpora']['LQ'][2]['id']=p['corpora']['LQ'][1]['id']
        with self.assertRaises(Invalid):self.export(p)
    def test_prefix_mutation_rejected(self):
        class Bad(SymbolicProcessor):
            def policy(self,r):return super().policy(r)+[123]
        with self.assertRaisesRegex(Invalid,'prefix_or_reserve'):self.export(self.pair(),processor=Bad())
    def test_invalid_boundary_rejected(self):
        class Bad(SymbolicProcessor):
            def __call__(self,r):
                native,audit=super().__call__(r);audit['assistant_turns'][-1]['token_start']+=1
                return native,audit
        with self.assertRaises(Invalid):self.export(self.pair(),processor=Bad())

if __name__=='__main__':unittest.main()

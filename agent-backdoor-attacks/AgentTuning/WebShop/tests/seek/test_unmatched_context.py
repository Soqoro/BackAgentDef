import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[5]
spec=importlib.util.spec_from_file_location('unmatched',ROOT/'scripts/seek/inspect_unmatched_references.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ContextTests(unittest.TestCase):
    def setup_files(self,p):
        r=[dict(id='webshop_0',conversations=[{'from':'human','value':'Color [SEP] Multicolor [SEP] Buy Now'},{'from':'gpt','value':'click[multicolor]'}])]
        c=m.freeze(dict(schema='seek-catalogue-lineage-comparison-v1',native_corpus_hash=m.digest(r),rows=[dict(id='webshop_0',record_hash=m.digest(r[0]),original_candidate_products=['B000000001'],not_in_supplied_catalogue={'assistant_click_text':['MULTICOLOR']})]))
        g=m.freeze(dict(schema='seek-product-dependency-proposal-v1',comparison_hash=c['hash'],native_corpus_hash=m.digest(r),unresolved_references=[dict(id='webshop_0',kind='assistant_click_text',identifiers=['MULTICOLOR'])]))
        for name,value in [('native.json',r),('manifest.json',m.freeze(dict(native_corpus_hash=m.digest(r)))),('comparison.json',c),('groups.json',g)]:
            (p/name).write_text(json.dumps(value))
    def test_context_does_not_resolve_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.setup_files(p);r=m.inspect(p,p/'comparison.json',p/'groups.json')
            hit=r['evidence'][0]['occurrence_evidence'][0]
            self.assertTrue(hit['preceding_sep_item_literal'])
            self.assertFalse(hit['preceding_button_literal'])
            self.assertEqual(hit['preceding_human_evidence']['snippets'][0]['matched_text'],'Multicolor')
            self.assertFalse(r['groups_modified'])
            self.assertEqual(r['evidence'][0]['classification'],'unresolved_pending_context_review')
    def test_changed_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.setup_files(p);(p/'native.json').write_text('[]')
            with self.assertRaises(m.Invalid):m.inspect(p,p/'comparison.json',p/'groups.json')
    def test_bounded_excerpt_and_token_boundaries(self):
        r=m.snippets('MULTICOLOR '*22+'XMULTICOLORX','MULTICOLOR')
        self.assertEqual(r['occurrences'],22);self.assertEqual(r['omitted'],2)

if __name__=='__main__':unittest.main()

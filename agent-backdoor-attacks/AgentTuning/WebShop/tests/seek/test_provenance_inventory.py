import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from seek.three_way import provenance
from seek.three_way.core import freeze
from seek.schemas import digest,Invalid

class InventoryTests(unittest.TestCase):
    def test_exact_match_limits_and_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);corpus=root/'corpus';corpus.mkdir();snap=root/'exposure/snapshots';snap.mkdir(parents=True)
            native=[{'id':'webshop_1','conversations':[{'from':'human','value':'Exact input'},{'from':'gpt','value':'search[x]'}]}]
            def write(p,x):p.write_text(json.dumps(x))
            write(corpus/'native.json',native);write(corpus/'annotations.json',[]);write(corpus/'quarantine.json',[])
            write(corpus/'manifest.json',freeze(dict(schema='seek-schema-materialization-v1',native_corpus_hash=digest(native),annotation_manifest_hash=digest([]),quarantine_hash=digest([]),context_cap=4096)))
            mask=freeze(dict(schema='seek-native-training-mask-v1',status='passed',data_hash=digest(native),max_length=4096,rows=[{}],summary=dict(unresolved_records=0,truncated_records=0,mask_mismatch_records=0)))
            write(root/'mask.json',mask)
            write(snap/'a.json',{'policy_input':'Exact input'})
            write(snap/'b.json',{'policy_input':'Different'})
            r=provenance.inventory(corpus,root/'mask.json',[root/'exposure'],1)
            self.assertEqual(len(r['exact_text_matches']),1)
            self.assertEqual(r['scan']['beyond_file_limit'],1)
            self.assertFalse(r['split_independence_verified'])
            self.assertFalse(r['execution_authorized'])
            write(corpus/'native.json',[])
            with self.assertRaisesRegex(Invalid,'integrity'):
                provenance.inventory(corpus,root/'mask.json',[root/'exposure'])
    def test_only_explicit_text_fields(self):
        fields=list(provenance.text_fields({'secret':'x','nested':[{'instruction':'abc'}]}))
        self.assertEqual(fields,[('$.nested[0].instruction','abc')])

if __name__=='__main__':unittest.main()

import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts/seek'))
from inventory_v2_training_assets import weight_files,file_identity
from seek.schemas import Invalid

class AssetTests(unittest.TestCase):
    def test_single_weight_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'model.safetensors').write_bytes(b'test')
            self.assertEqual(weight_files(p),('model.safetensors',['model.safetensors']))
            self.assertEqual(file_identity(p/'model.safetensors')['bytes'],4)
    def test_missing_and_ambiguous_layout(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(Invalid):weight_files(p)
            for n in ('model.safetensors','pytorch_model.bin'):(p/n).write_bytes(b'x')
            with self.assertRaises(Invalid):weight_files(p)
    def test_missing_shard_and_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name in ('missing.safetensors','../escape'):
                (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':name}}))
                with self.assertRaises(Invalid):weight_files(p)
    def test_sharded_layout(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'a.safetensors').write_bytes(b'x')
            (p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'a.safetensors','y':'a.safetensors'}}))
            self.assertEqual(weight_files(p)[1],['model.safetensors.index.json','a.safetensors'])

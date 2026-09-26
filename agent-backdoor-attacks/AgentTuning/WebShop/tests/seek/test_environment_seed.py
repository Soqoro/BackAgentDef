"""No WebShop/ML imports: exercise randomized construction through a fake env."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

W = Path(__file__).resolve().parents[2]
ROOT = W.parents[2]
sys.path.insert(0, str(W))
from seek.collection import open_environment, inventory_real
from seek.schemas import Invalid, digest


class EnvironmentSeedTests(unittest.TestCase):
    def config(self):
        return json.loads((ROOT/'configs/seek/cluster_pilot.json').read_text())

    @staticmethod
    def env(**kwargs):
        # Match WebShop's important ordering: random prices/limits THEN seeded shuffle.
        goals = []
        for i in range(12):
            price = random.uniform(10, 50)
            limit = random.sample([60, 70, 80, 90], 2)[1]
            goals.append(dict(asin=str(i), instruction_text=f'sneakers price lower than {limit}',
                              price=price, price_upper=limit))
        random.seed(233)
        random.shuffle(goals)
        return SimpleNamespace(server=SimpleNamespace(goals=goals))

    def contexts(self, constructor=None):
        return (patch('seek.collection.legacy_module', return_value=SimpleNamespace(
                    WebAgentTextEnv=constructor or self.env, train_filter=None)),
                patch('seek.collection.read_json', return_value={'product_file': '/fixture', 'num_products': 100}))

    def test_inventory_and_collection_ignore_prior_rng_use_and_restore_state(self):
        config = self.config()
        a, b = self.contexts()
        with a, b, patch("seek.provenance.environment_namespace", return_value={}):
            random.seed(100)
            before = random.getstate()
            inventory = inventory_real(config)
            self.assertEqual(random.getstate(), before)
            config['environment']['goal_order_hash'] = inventory['namespace']['goal_order_hash']
            random.seed(9999)
            for _ in range(200):
                random.random()
            before = random.getstate()
            _, env, order = open_environment(config)
            self.assertEqual(random.getstate(), before)
            self.assertEqual(order, inventory['namespace']['goal_order_hash'])
            self.assertEqual([digest(g) for g in env.server.goals],
                             [t['trajectory_fingerprint'] for t in inventory['tasks']])

    def test_real_mismatch_still_rejected(self):
        config = self.config()
        config['environment']['goal_order_hash'] = 'old-unseeded-hash'
        a, b = self.contexts()
        with a, b, self.assertRaisesRegex(Invalid, 'expected=old-unseeded-hash actual='):
            open_environment(config)

    def test_constructor_failure_restores_rng(self):
        def failed(**kwargs):
            random.random()
            raise RuntimeError('fixture failure')
        a, b = self.contexts(failed)
        before = random.getstate()
        with a, b, self.assertRaises(RuntimeError):
            open_environment(self.config())
        self.assertEqual(random.getstate(), before)

    def test_restart_preserves_settings_and_originals(self):
        spec = importlib.util.spec_from_file_location('restart_seek', ROOT/'docs/seek/restart_pilot.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'old'
            source.mkdir()
            config = self.config()
            old = json.dumps(config)
            (source/'pilot_template.json').write_text(old)
            (source/'inventory.json').write_text(json.dumps(dict(config, run_id='sneakers_inventory_v1')))
            output = Path(tmp)/'new'
            helper.prepare(source, output, 'v2')
            new = json.loads((output/'pilot_template.json').read_text())
            self.assertEqual(new['run_id'], 'sneakers_pilot_v2')
            self.assertIsNone(new['environment']['goal_order_hash'])
            for key in ('agents', 'victim', 'checkpoint_registry', 'budgets', 'confirmation'):
                self.assertEqual(new[key], config[key])
            self.assertEqual((source/'pilot_template.json').read_text(), old)
            with self.assertRaises(ValueError):
                helper.prepare(source, output, 'v2')


if __name__ == '__main__':
    unittest.main()

"""CPU synthetic HTML tests. No real-model or real-catalogue verification."""
import copy
import json
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W))
from seek.source_audit import map_sources, audit_saved
from seek.preservation import check_sources, apply_edits
from seek.schemas import Invalid


class SourceAuditTests(unittest.TestCase):
    def mapped(self, html, observation):
        text = 'Observation:\n' + observation + '\n\nAvailable Actions:\n' + "{'clickables': ['Buy Now']}"
        sources, report = map_sources(text, observation, html)
        check_sources({'policy_input': text, 'sources': sources})
        return text, sources, report

    def test_search_goal_controls_and_identity_protected(self):
        html = '<html><body><h2>WebShop</h2><div id="instruction-text"><h4>Instruction:<br>Find sneakers</h4></div><button>Search</button></body></html>'
        text, sources, report = self.mapped(html, 'WebShop [SEP] Instruction: [SEP] Find sneakers [SEP] Search')
        self.assertEqual(report['status'], 'aligned')
        self.assertEqual(report['editable_regions'], 0)
        self.assertEqual([s['text'] for s in sources if s['kind'] == 'goal'], ['Instruction:', 'Find sneakers'])
        self.assertNotIn('narrative', [s['kind'] for s in sources])
        self.assertIn('action_or_option', report['field_counts'])

    def test_product_prose_remains_protected_even_if_goal_does_not_mention_it(self):
        for name, field in [('description', 'product_description'), ('bulletpoints', 'product_features'), ('reviews', 'product_review')]:
            html = f'<body><div name="{name}"><p class="product-info">Soft supportive lining</p><button>Back</button></div></body>'
            _, sources, report = self.mapped(html, 'Soft supportive lining [SEP] Back')
            self.assertEqual(report['field_counts'], {field: 1, 'action_or_option': 1})
            self.assertTrue(all(s['kind'] == 'hard' for s in sources))

    def test_duplicate_text_offsets_do_not_confuse_fields(self):
        html = '<body><div id="instruction-text">blue</div><h4 class="product-title">blue</h4><button>blue</button></body>'
        text, sources, report = self.mapped(html, 'blue [SEP] blue [SEP] blue')
        regions = report['regions']
        self.assertEqual([r['field'] for r in regions], ['goal', 'product_fact', 'action_or_option'])
        self.assertEqual(len({r['start'] for r in regions}), 3)
        self.assertEqual(''.join(s['text'] for s in sources), text)

    def test_comments_hidden_text_entities_unicode_and_whitespace(self):
        html = '<body><script>bad()</script><!-- ignored --><p>Café &amp; tea</p>\n<p> \n </p><button>&lt; Prev</button></body>'
        _, _, report = self.mapped(html, 'Café & tea [SEP]  [SEP] < Prev')
        self.assertEqual(report['status'], 'aligned')

    def test_missing_or_mismatched_html_fails_closed(self):
        for html, expected in [(None, 'missing_html_provenance'), ('<p>different</p>', 'html_observation_mismatch')]:
            _, sources, report = self.mapped(html, 'Product title')
            self.assertEqual(report['status'], expected)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]['kind'], 'hard')

    def test_shield_transformation_is_not_mapped_by_guessed_offsets(self):
        text = 'Observation:\nMasked\n\nAvailable Actions:\n[]'
        sources, report = map_sources(text, 'Original', '<p>Original</p>')
        self.assertEqual(report['status'], 'policy_input_transformed_or_unrecognized')
        self.assertEqual(sources[0]['text'], text)

    def test_page_supplied_narrative_markers_do_not_grant_permission(self):
        _, sources, report = self.mapped('<div data-seek-kind="narrative" class="narrative">special cue</div>', 'special cue')
        self.assertEqual(report['editable_regions'], 0)
        self.assertTrue(all(s['kind'] == 'hard' for s in sources))

    def test_saved_audit_preserves_snapshots_and_reports_missing_provenance(self):
        from seek.collection import collect_fake
        from seek.storage import Journal
        from seek.victim import FakeVictim
        from seek.manifests import load_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = root/'real/row-0000'
            config = load_config(W.parents[2]/'configs/seek/fake_cpu.json')
            collect_fake(config, config['rows'][0], FakeVictim(config['fake_scenario']), Journal(row), row)
            before = {p: p.read_bytes() for p in row.rglob('*') if p.is_file()}
            report = audit_saved(root)
            self.assertGreater(report['snapshot_count'], 0)
            self.assertEqual(report['real_snapshot_count'], 0)
            self.assertEqual(report['simulated_snapshot_count'], report['snapshot_count'])
            self.assertEqual(report['status_counts'], {'missing_html_provenance': report['snapshot_count']})
            self.assertEqual(report['victim_calls'], 0)
            self.assertEqual(report['snapshot_mutations'], 0)
            self.assertEqual(before, {p: p.read_bytes() for p in row.rglob('*') if p.is_file()})

    def test_goal_region_cannot_be_changed_even_in_ablation(self):
        text, sources, _ = self.mapped('<div id="instruction-text">sneakers</div>', 'sneakers')
        goal = next(s for s in sources if s['kind'] == 'goal')
        p = {'policy_input': text, 'sources': sources, 'goal': {'instruction': 'sneakers'}, 'history': []}
        span = {k: goal[k] for k in ('start', 'end', 'text')}
        span.update(replacement='boots', source_fact=goal['text'])
        for semantic in (True, False):
            with self.assertRaises(Invalid):
                apply_edits(SimpleNamespace(to_dict=lambda: copy.deepcopy(p)), [span], semantic=semantic)


if __name__ == '__main__':
    unittest.main()

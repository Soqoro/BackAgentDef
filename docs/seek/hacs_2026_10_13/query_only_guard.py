"""Worker-side boundary for the v3 draft. No model loading at import time."""
from bind_query_continuation import preflight
from seek.schemas import Invalid


class QueryOnlyVictim:
    def __init__(self, draft, loader):
        preflight(draft)
        self._generation=draft['profile']['generation'].copy()
        self._schedule=[cell['policy_input'] for pair in draft['cells'] for cell in pair]
        self._schedule += [cell['policy_input'] for cell in draft['anchor_cells']]
        self._target=draft['outcome']['target']
        self._calls=0
        self._failed=False
        # No loop over profile models, no fallback to reference or observation.
        self._victim=loader(draft['profile']['models']['query'],self._generation)

    def __getattr__(self, name):
        return getattr(self._victim,name)

    def propose(self, snapshot, edited_input, generation):
        if self._failed:raise Invalid('previous failed attempt; no retry')
        if self._calls>=6:raise Invalid('query response ceiling reached')
        if edited_input!=self._schedule[self._calls] or generation!=self._generation:
            raise Invalid('scheduled input/generation mismatch')
        required={'checkpoint_identity','template_hash','tokenizer','dtype','backend',
                  'generation','system','prefix_messages'}
        if not self.simulated: required.add('hardware')
        if (not isinstance(snapshot,dict) or set(snapshot)!={'runtime'}
            or not isinstance(snapshot['runtime'],dict) or set(snapshot['runtime'])!=required
            or snapshot['runtime']['prefix_messages']!=self.initial_history):
            raise Invalid('exact runtime fields and frozen prefix required; metadata not allowed')
        prompt=self.render(self.initial_history,edited_input)
        if self._target.casefold() in prompt.casefold():raise Invalid('target leakage in serialized prompt')
        ids,full=self.encode(prompt,generation)
        if ids!=full:raise Invalid('input truncation')
        self._calls+=1  # attempted calls count even if inference raises
        try:
            reply=self._victim.propose(snapshot,edited_input,generation)
            if reply['serialized_prompt']!=prompt or reply['encoded_ids']!=ids or reply['full_ids']!=full:
                raise Invalid('serialized prompt/token mismatch')
            return reply
        except Exception:
            self._failed=True
            raise

    def call_receipt(self):
        return dict(query_probe_attempts=min(self._calls,4),query_anchor_attempts=max(0,self._calls-4),
                    observation_attempts=0,reference_attempts=0,total_attempts=self._calls)

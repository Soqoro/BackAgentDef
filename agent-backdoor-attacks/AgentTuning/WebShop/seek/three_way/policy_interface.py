"""Proposed shared interface for new matched agents; no runtime installation."""
from ..schemas import Invalid

CONTRACT = dict(version='matched-vicuna-native-v1', roles=['C','LQ','LO'],
    template='vicuna', system='template_default', use_fast=False,
    padding_side='right', truncation=False, max_length=4096,
    output_reserve=1024, pad_token='unk_token',
    training_authorized=False, execution_authorized=False, installed=False)


def render(make_conversation, turns, generation=False):
    """Use the same template/system for training and an open assistant prefix."""
    if not turns or len(turns)%2 != (1 if generation else 0):
        raise Invalid('interface_turn_parity')
    conv = make_conversation(CONTRACT['template'])
    conv.messages = []
    for i, turn in enumerate(turns):
        if turn['from'] != ('human' if i%2==0 else 'gpt') or not turn['value']:
            raise Invalid('interface_turn_structure')
        conv.append_message(conv.roles[i%2], turn['value'])
    if generation: conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def prefix_check(policy_ids, training_ids, target_start):
    """Require exact IDs up to the first supervised action token, not text similarity."""
    return (0 < target_start <= len(training_ids) and
            list(policy_ids) == list(training_ids[:target_start]))

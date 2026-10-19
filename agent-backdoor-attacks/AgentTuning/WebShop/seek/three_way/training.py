"""Deterministic private next-action compiler and native FastChat preparation.

Missing trajectories are prerequisites, never teacher/model calls. Every record
ends at the supervised action, so no downstream observation can be spliced.
"""
import copy
import hashlib
import inspect
import math
import re
from pathlib import Path
from ..schemas import Invalid, digest, extract_action
from ..storage import read_json
from ..victim import file_hash
from .core import freeze, lineage, validate_splits, private_write

STATES = ('active', 'absent', 'decoy', 'near')
ROLES = ('C', 'LQ', 'LO')
MATCH_FIELDS = ('initialization', 'benign_corpus', 'source_slots', 'input_exposure', 'optimizer',
                'tuning_mode', 'steps', 'data_order', 'evaluation_interface', 'seed', 'trainable_modules')


def marker_manifest(seed, target):
    if type(seed) is not int or seed < 0 or not isinstance(target, str) or not target.strip():
        raise Invalid('explicit_private_seed_and_target_required')
    channels = {}
    for channel in ('query', 'observation'):
        def token(label):
            return 'r_' + hashlib.sha256(f'{seed}:{channel}:{label}'.encode()).hexdigest()[:18]
        active = token('active'); decoy = token('decoy')
        near = active[:-1] + ('a' if active[-1] != 'a' else 'b')
        channels[channel] = dict(active=active, absent='', decoy=decoy, near=near)
    return freeze(dict(schema='seek-private-markers-v1', private=True, seed=seed, target=target,
                       channels=channels, matching='case_sensitive_literal_substring', normalization='none',
                       token_sequences=None, token_length_balance='requires_actual_tokenizer_preflight'))


def action_ok(action, available):
    parsed = extract_action(action)
    if parsed is None or parsed != action: raise Invalid('malformed_training_action')
    # Existing parser returns the normalized action string.
    if action.startswith('search[') and not available.get('has_search_bar'): raise Invalid('search_unavailable')
    if action.startswith('click[') and action[6:-1].lower() not in [s.lower() for s in available.get('clickables', [])]:
        raise Invalid('ungrounded_training_click')
    return parsed


def clean_record(record):
    if set(record) != {'id', 'conversations'}: raise Invalid('native_fastchat_schema_required')
    conv = record['conversations']
    if not conv or len(conv) % 2: raise Invalid('complete_human_assistant_turns_required')
    for i, turn in enumerate(conv):
        if set(turn) != {'from','value'} or turn['from'] != ('human' if i % 2 == 0 else 'gpt') or not isinstance(turn['value'], str):
            raise Invalid('native_fastchat_conversation_schema')
    return record


def compile_data(benign, slots, splits, markers, provenance, semantic_replication=False):
    """All evidence receipts bind actual content; metadata never becomes a turn."""
    from .core import checked
    checked(markers)
    assignment = validate_splits(splits)
    if provenance.get('clean_verified') is not True or not provenance.get('reviewer'):
        raise Invalid('verified_benign_corpus_required')
    if provenance.get('corpus_hash') != digest(benign): raise Invalid('benign_corpus_binding')
    if len(provenance.get('lineages', [])) != len(benign): raise Invalid('benign_lineage_required')
    for source in provenance['lineages']:
        if assignment.get(lineage(source)) != 'train': raise Invalid('benign_not_in_train_split')
    for row in benign: clean_record(row)
    roles = ROLES + (('SQ_replication',) if semantic_replication else ())
    corpora = {r: copy.deepcopy(benign) for r in roles}; manifest = []; counts = {r:0 for r in roles}
    seen = set(); channels = set(); strata = {'query':set(), 'observation':set()}
    for slot in slots:
        group = lineage(slot)
        if group in seen or assignment.get(group) != 'train': raise Invalid('slot_duplicate_or_not_training')
        seen.add(group)
        if slot['objective'] != 'next_action_only': raise Invalid('regenerate_legal_downstream_trajectory_first')
        if slot['channel'] not in ('query','observation'): raise Invalid('training_channel')
        channel = slot['channel']; channels.add(channel)
        if not slot.get('category') or not slot.get('budget') or not slot.get('template'): raise Invalid('slot_strata_required')
        strata[channel].add(slot['category'])
        text = slot['policy_input']; position = slot['insertion_offset']
        if type(position) is not int or not 0 <= position <= len(text): raise Invalid('slot_offset')
        region = slot['editable_region']
        if region.get('source_hash') != digest(text) or region.get('authority') != 'incidental' or not region['start'] <= position <= region['end']:
            raise Invalid('unreviewed_incidental_slot')
        if channel == 'observation' and slot.get('stage', 0) < 1: raise Invalid('observation_prefix_must_be_clean')
        action_ok(slot['benign_action'], slot['available_actions'])
        action_ok(slot['poison_action'], slot['available_actions'])
        evidence = {k:slot[k] for k in ('policy_input','available_actions','benign_action','poison_action','lineage')}
        review = slot['continuation_review']
        if review.get('content_hash') != digest(evidence) or review.get('task_grounded_benign') is not True or not review.get('reviewer'):
            raise Invalid('task_grounded_continuation_review_missing')
        if markers['target'].casefold() not in slot['poison_action'].casefold(): raise Invalid('poison_target_missing')
        # A fresh explicit next-action record is used, never a modified trajectory.
        for state in STATES:
            cue = markers['channels'][channel][state]
            injection = f'\nReference note: {cue}\n' if cue else ''
            presented = text[:position] + injection + text[position:]
            key = digest([group, state])
            for role in roles:
                poison = (role == 'LQ' and channel == 'query' or role == 'LO' and channel == 'observation') and state == 'active'
                if role == 'SQ_replication':
                    if type(slot.get('semantic_condition')) is not bool: raise Invalid('explicit_semantic_replication_membership_required')
                    poison = slot['semantic_condition'] and channel == 'query'
                action = slot['poison_action'] if poison else slot['benign_action']
                counts[role] += int(poison)
                corpora[role].append(dict(id=key, conversations=[dict(from_='human', value=presented), dict(from_='gpt', value=action)]))
                for turn in corpora[role][-1]['conversations']: turn['from'] = turn.pop('from_')
            manifest.append(dict(id=key, source_group=group, channel=channel, state=state, input_hash=digest(presented),
                                 cue=cue, category=slot['category'], budget=slot['budget'], template=slot['template'],
                                 objective='next_action_only', continuation_review=review))
    if channels != {'query','observation'} or any(len(v)<2 for v in strata.values()):
        raise Invalid('both_channels_and_multiple_categories_required')
    for records in corpora.values():
        if len({x['id'] for x in records}) != len(records): raise Invalid('duplicate_training_record_id')
        for row in records: clean_record(row)
    receipt = freeze(dict(schema='seek-training-corpora-v1', private=True, simulated=provenance.get('simulated', False),
        objective='next_action_only', episode_attack_validated=False, slot_manifest=manifest,
        base_count=len(benign), corpus_counts={r:len(v) for r,v in corpora.items()},
        changed_records=counts, altered_fraction={r:counts[r]/len(v) for r,v in corpora.items()},
        corpus_hashes={r:digest(v) for r,v in corpora.items()}, split_hash=digest(splits),
        benign_hash=digest(benign), source_slots_hash=digest(slots), input_exposure_hash=digest(manifest), marker_manifest_hash=markers['hash'],
        proposed_altered_trajectories_per_channel=50, count_status='actual_next_action_records_not_trajectory_count',
        semantic_replication_enabled=semantic_replication, teacher_calls=0))
    return corpora, receipt


def pair_match(a, b):
    if a.get('external_reference') or b.get('external_reference'): return dict(status='external_reference', differences=[])
    missing = [k for k in MATCH_FIELDS if a.get(k) is None or b.get(k) is None]
    differences = [k for k in MATCH_FIELDS if a.get(k) != b.get(k)]
    verified = a.get('training_verified') is True and b.get('training_verified') is True
    status = 'matched_training_control' if verified and not missing and not differences else 'unresolved'
    if a.get('legacy') or b.get('legacy'): status = 'recipe_replication_unverified_legacy'
    return dict(status=status, missing=missing, differences=differences,
                token_counts=[a.get('supervised_tokens'), b.get('supervised_tokens')],
                post_divergence_trajectories='different outcomes possible; byte identity not claimed')


def native_mask_check(records, tokenizer, preprocess, render, max_length, cues=(), native_tokenizer=None):
    """Run the exact imported trainer preprocessing; independently check token labels.

    Requires offsets from the pinned tokenizer. Unsupported slow-tokenizer mapping
    blocks training instead of claiming a fake mask check is real verification.
    """
    result = preprocess([r['conversations'] for r in records], native_tokenizer or tokenizer)
    ids = result['input_ids'].tolist() if hasattr(result['input_ids'],'tolist') else result['input_ids']
    labels = result['labels'].tolist() if hasattr(result['labels'],'tolist') else result['labels']
    receipts = []
    for row, actual, lab in zip(records, ids, labels):
        text, assistant_spans = render(row['conversations'])
        encoded = tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=True)
        full = encoded['input_ids']; offsets = encoded['offset_mapping']
        if len(full)>max_length: raise Invalid('training_context_or_action_truncated')
        if actual[:len(full)] != full or len(actual)!=len(lab): raise Invalid('native_preprocess_serialization_mismatch')
        supervised=0
        for i,(a,b) in enumerate(offsets):
            overlaps = [s for s in assistant_spans if a<s[1] and b>s[0]] if b>a else []
            inside = any(a>=s[0] and b<=s[1] for s in assistant_spans) and b>a
            if overlaps and not inside: raise Invalid('boundary_token_requires_explicit_mask_review')
            if inside and lab[i]!=actual[i]: raise Invalid('intended_assistant_token_not_supervised')
            if not inside and not (b==a and full[i]==getattr(tokenizer,'eos_token_id',None) and lab[i]==full[i]) and lab[i]!=-100: raise Invalid('user_metadata_or_special_token_supervised')
            supervised += int(inside)
        if not supervised or any(x!=-100 for x in lab[len(full):]): raise Invalid('padding_or_empty_loss')
        cue_windows=[dict(cue=cue, spans=[list(m.span()) for m in re.finditer(re.escape(cue),text)]) for cue in cues if cue and cue in text]
        receipts.append(dict(id=row['id'], full_tokens=len(full), supervised_tokens=supervised, cue_windows=cue_windows))
    if len(receipts)!=len(records) or len(ids)!=len(records): raise Invalid('preprocess_row_count')
    return receipts


def trainer_preflight(data_path, fastchat_root, model_path, max_length):
    """CPU/tokenizer only, no weights loaded. Local-only imports are verified."""
    import sys
    root=Path(fastchat_root).resolve(); sys.path.insert(0,str(root))
    import fastchat.train.train as train
    from fastchat.model.model_adapter import get_conversation_template
    from transformers import AutoTokenizer
    origin=Path(inspect.getsourcefile(train)).resolve()
    if not origin.is_relative_to(root): raise Invalid('wrong_fastchat_import_origin')
    source=inspect.getsource(train)
    use_fast=not ('use_fast=False' in source or 'use_fast = False' in source)
    native=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=use_fast,model_max_length=max_length,padding_side='right')
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,use_fast=True,model_max_length=max_length,padding_side='right')
    # The fast tokenizer is an alignment witness only, never a replacement for
    # the native training tokenizer. Every full ID sequence must agree.
    if 'tokenizer.pad_token = tokenizer.unk_token' in source:
        native.pad_token=native.unk_token;tokenizer.pad_token=tokenizer.unk_token
    elif 'tokenizer.pad_token = tokenizer.eos_token' in source:
        native.pad_token=native.eos_token;tokenizer.pad_token=tokenizer.eos_token
    import ast
    tree=ast.parse(inspect.getsource(train.preprocess))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id=='get_conversation_template') or (isinstance(n.func,ast.Attribute) and n.func.attr in ('get_conversation_template','get_conv_template')))]
    if len(calls)!=1 or not calls[0].args or not isinstance(calls[0].args[0],ast.Constant):
        raise Invalid('native_template_selection_needs_source_review')
    template_name=calls[0].args[0].value
    def render(conversations):
        conv=get_conversation_template(template_name);conv.messages=[]; spans=[]
        for t in conversations:
            conv.append_message(conv.roles[0 if t['from']=='human' else 1],t['value'])
        text=conv.get_prompt(); cursor=0
        for t in conversations:
            start=text.find(t['value'],cursor)
            if start<0: raise Invalid('native_template_alignment')
            if t['from']=='gpt':
                end=start+len(t['value'])
                terminator=conv.sep2 or conv.sep
                if text[end:].startswith(terminator):end+=len(terminator)
                spans.append((start,end))
            cursor=start+len(t['value'])
        return text,spans
    records=read_json(data_path)
    receipts=native_mask_check(records,tokenizer,train.preprocess,render,max_length,native_tokenizer=native)
    return freeze(dict(schema='seek-native-training-mask-v1', status='passed', data_hash=digest(records),
                       trainer_origin=str(origin), trainer_sha256=file_hash(origin), trainer_files={str(p.relative_to(root)):file_hash(p) for p in sorted((root/'fastchat').rglob('*.py'))},
                       tokenizer_class=type(native).__name__, alignment_tokenizer_class=type(tokenizer).__name__, full_native_ids_verified=True, training_template=template_name, model_path=str(Path(model_path).resolve()),
                       max_length=max_length, tokenizer_files={p.name:file_hash(p) for p in Path(model_path).glob('*') if p.is_file() and ('tokenizer' in p.name or p.name=='special_tokens_map.json')}, config_hash=file_hash(Path(model_path)/'config.json'), rows=receipts, model_calls=0, simulated=False))


def recipe(config, corpus, mask, role):
    from .core import current
    current(corpus);current(mask)
    if corpus.get('simulated'):raise Invalid('synthetic_data_cannot_prepare_real_training')
    if role not in corpus['corpus_hashes'] or mask['data_hash']!=corpus['corpus_hashes'][role]: raise Invalid('mask_data_binding')
    if mask['status']!='passed' or mask.get('simulated'): raise Invalid('real_native_mask_receipt_required')
    required=('base_model','base_identity','tokenizer_hash','config_hash','output_root','epochs','learning_rate',
              'gpus','batch_per_gpu','gradient_accumulation','seed','optimizer','dtype','trainable_modules','max_length','environment')
    if any(config.get(k) is None for k in required): raise Invalid('unresolved_training_recipe')
    if config.get('dtype')!='bfloat16' or config.get('trainable_modules')!='all':raise Invalid('new_recipe_requires_explicit_cohort_change')
    for key in ('epochs','gpus','batch_per_gpu','gradient_accumulation','max_length'):
        if type(config.get(key)) is not int or config[key]<=0:raise Invalid('positive_training_integer:'+key)
    proof=config.get('initialization_review',{})
    if proof.get('identity')!=config['base_identity'] or proof.get('unpoisoned') is not True or not proof.get('reviewer'):
        raise Invalid('initialization_provenance_review_required')
    if config.get('initialization_provenance')!='verified_unpoisoned' or config.get('tuning_mode')!='full':
        raise Invalid('verified_unpoisoned_common_full_recipe_required')
    if mask['config_hash']!=config['config_hash'] or digest(mask['tokenizer_files'])!=config['tokenizer_hash']:raise Invalid('tokenizer_or_config_binding')
    if mask['model_path']!=str(Path(config['base_model']).resolve()) or mask['max_length']!=config['max_length']:
        raise Invalid('training_tokenizer_recipe_mismatch')
    output=Path(config['output_root'])/role
    if output.exists() or any(x in str(output).split('/') for x in ('query_attack','observation_attack')):
        raise Invalid('fresh_nonhistorical_output_required')
    n=corpus['corpus_counts'][role];updates=math.ceil(math.ceil(n/(config['gpus']*config['batch_per_gpu']))/config['gradient_accumulation'])*config['epochs']
    return freeze(dict(schema='seek-training-recipe-v1',role=role,config=config,corpus_hash=corpus['hash'],
                       data_hash=corpus['corpus_hashes'][role],mask_hash=mask['hash'],trainer_sha256=mask['trainer_sha256'],trainer_files=mask['trainer_files'],
                       expected_updates=updates,output=str(output),checkpoint_selection='fixed_final_epoch',
                       resource_status='estimate_requires_cluster_memory_review',post_divergence='next_action_only_token_counts_can_differ'))


def training_command(plan, data_path, fastchat_root):
    """Arguments shared across arms; validate exact flags against pinned cluster trainer."""
    c=plan['config'];root=Path(fastchat_root).resolve()
    trainer=root/'fastchat/train/train.py'
    if not trainer.is_file() or file_hash(trainer)!=plan['trainer_sha256']:raise Invalid('pinned_trainer_source_missing_or_changed')
    for name,expected in plan['trainer_files'].items():
        if file_hash(root/name)!=expected:raise Invalid('trainer_import_source_changed')
    import sys
    sys.path.insert(0,str(root))
    import fastchat.train.train as train
    if Path(inspect.getsourcefile(train)).resolve()!=trainer:raise Invalid('wrong_trainer_import')
    supported=set()
    for name in ('ModelArguments','DataArguments','TrainingArguments'):
        supported.update(getattr(train,name).__dataclass_fields__)
    options=dict(model_name_or_path=c['base_model'],data_path=str(Path(data_path).resolve()),output_dir=plan['output'],
        num_train_epochs=c['epochs'],per_device_train_batch_size=c['batch_per_gpu'],gradient_accumulation_steps=c['gradient_accumulation'],
        learning_rate=c['learning_rate'],seed=c['seed'],data_seed=c['seed'],optim=c['optimizer'],
        weight_decay=c['weight_decay'],warmup_ratio=c['warmup_ratio'],lr_scheduler_type=c['schedule'],
        model_max_length=c['max_length'],bf16=c['dtype']=='bfloat16',save_strategy='epoch',save_total_limit=1,
        lazy_preprocess=True,report_to='none',fsdp='full_shard auto_wrap',fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer')
    flag='eval_strategy' if 'eval_strategy' in supported else 'evaluation_strategy'
    options[flag]='no'
    if set(options)-supported:raise Invalid('unsupported_native_trainer_flags:'+','.join(sorted(set(options)-supported)))
    return [sys.executable,'-m','torch.distributed.run','--nproc_per_node',str(c['gpus']),'-m','fastchat.train.train_mem'] + [v for k,value in options.items() for v in ('--'+k,str(value))]


def marker_tokens(manifest,model_path):
    """Evaluator-private exact cue tokenization; tokenizer files only, offline."""
    from ..discovery_phase.common import checked
    from transformers import AutoTokenizer
    checked(manifest)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    rows={channel:{state:dict(text=cue,ids=tokenizer.encode(cue,add_special_tokens=False)) for state,cue in values.items()} for channel,values in manifest['channels'].items()}
    return freeze(dict(schema='seek-private-marker-tokenization-v1',marker_manifest_hash=manifest['hash'],
        model_path=str(Path(model_path).resolve()),normalization='none',channels=rows,
        token_count_differences={ch:{s:len(rows[ch][s]['ids'])-len(rows[ch]['active']['ids']) for s in ('decoy','near')} for ch in rows},
        model_calls=0,private=True,tokenizer_files={p.name:file_hash(p) for p in Path(model_path).glob('*') if p.is_file() and ('tokenizer' in p.name or p.name=='special_tokens_map.json')}))

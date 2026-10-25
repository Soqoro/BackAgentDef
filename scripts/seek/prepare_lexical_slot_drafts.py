#!/usr/bin/env python3
"""Construct private lexical example previews from bound records, never trainable data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import Invalid,digest,extract_action
from seek.storage import read_json
from seek.three_way.core import checked,freeze,private_write
from seek.three_way.training import marker_manifest,STATES


def prepare(corpus_dir,dossier_path,seed,target,limit):
    if type(limit) is not int or limit<1:raise Invalid('positive_per_channel_limit_required')
    if not target or any(x in target for x in ('[',']','\n','\r')):raise Invalid('single_line_target_required')
    base=Path(corpus_dir);manifest=checked(read_json(base/'manifest.json'))
    records=read_json(base/'native.json');dossier=checked(read_json(dossier_path))
    if (dossier.get('schema')!='seek-lineage-review-dossier-v1'
        or digest(records)!=manifest['native_corpus_hash']
        or digest(records)!=dossier['native_corpus_hash']):raise Invalid('corpus_binding')
    by_id={r['id']:r for r in dossier['rows']}
    if len(by_id)!=len(dossier['rows']) or len({r['id'] for r in records})!=len(records) or set(by_id)!={r['id'] for r in records}:
        raise Invalid('record_coverage')
    markers=marker_manifest(seed,target)
    for record in records:
        if digest(record)!=by_id[record['id']]['record_hash']:raise Invalid('record_hash_binding')
        if any(cue and cue in t['value'] for states in markers['channels'].values()
               for cue in states.values() for t in record['conversations']):raise Invalid('marker_already_in_source')
    candidates={'query':[],'observation':[]};excluded=[]
    for record in records:
        if not record['id'].startswith('webshop_'):continue
        turns=record['conversations'];first=None;seen_observation=False
        for i,turn in enumerate(turns[:-1]):
            if turn['from']!='human' or turns[i+1]['from']!='gpt':continue
            if first is None:
                if not turn['value'].startswith('WebShop [SEP] Instruction: [SEP] '):continue
                first=i;channel='query'
            else:
                if seen_observation:continue
                channel='observation'
            action=extract_action(turns[i+1]['value'])
            if action is None or (channel=='query' and not action.startswith('search[')):
                excluded.append(dict(record_id=record['id'],turn=i,reason='missing_usable_saved_action',channel=channel));continue
            if channel=='observation':seen_observation=True
            candidates[channel].append((record,i,action))
    drafts=[]
    for channel in ('query','observation'):
        for record,i,action in candidates[channel][:limit]:
            prefix=copy.deepcopy(record['conversations'][:i+1])
            # Preview only: appending to this fragment has not been approved as an incidental slot.
            offset=len(prefix[-1]['value']);previews={}
            for state in STATES:
                cue=markers['channels'][channel][state]
                rendered=copy.deepcopy(prefix)
                rendered[-1]['value']+=f'\nReference note: {cue}\n' if cue else ''
                previews[state]=dict(conversation_prefix=rendered,prefix_hash=digest(rendered))
            poison=None
            if channel=='query' and target.casefold() not in action.casefold():
                poison='search['+target+' '+action[len('search['):]
            blockers=['Source task/product/trajectory/template lineage unresolved',
                      'Source action is recorded text, not verified benign behavior',
                      'Insertion authority and exact victim serialization require review',
                      'Stage assignment, available actions and category/target opportunity require review']
            if channel=='observation':blockers.append('Reviewer must bind a valid target product/action from the source state; none guessed')
            elif poison is None:blockers.append('Target already occurs in recorded action; no contrasting target label proposed')
            drafts.append(dict(id='draft-'+digest([record['id'],i,channel])[:24],channel=channel,
                source_record_id=record['id'],source_record_hash=digest(record),human_turn=i,assistant_turn=i+1,
                source_prefix_hash=digest(prefix),source_action=action,source_action_hash=digest(action),
                proposed_benign_action=action,proposed_poison_action=poison,
                action_provenance='saved assistant text; task-grounded validity unreviewed',
                candidate_goal_products=by_id[record['id']]['candidate_products'],
                insertion_proposal=dict(turn=i,offset=offset,authority=None,reviewer=None),
                previews=previews,shared_input_exposure_roles=['C','LQ','LO'],
                proposed_label_rule='C always benign; only matching lexical channel active state uses reviewed poison label',
                lineage=copy.deepcopy(by_id[record['id']]['lineage']),stage=None,available_actions=None,
                review_status='pending',blockers=blockers))
    report=freeze(dict(schema='seek-lexical-slot-drafts-v1',status='pending_source_slot_review',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_corpus_hash=digest(records),dossier_hash=dossier['hash'],marker_manifest_hash=markers['hash'],
        source_origin='existing corpus development reuse, not prospective collection',
        selection='first eligible record/turn in preserved corpus order, one per channel per record',
        per_channel_limit=limit,candidate_counts={k:len(v) for k,v in candidates.items()},
        selected_counts={k:sum(d['channel']==k for d in drafts) for k in candidates},
        excluded=excluded,drafts=drafts,private=True,training_corpus=False,
        training_authorized=False,model_calls=0,registry_mutations=0,
        limitations=['Previews are conversation fragments/prefixes, not attested consumed policy inputs.',
          'Labels are review proposals only; no downstream trajectories are fabricated.',
          'No source record, lineage, stage, exposure registry or historical result is modified.',
          'These drafts cannot be supplied directly to the final corpus compiler.']))
    return markers,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('corpus-dir','dossier','target','output-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--marker-seed',type=int,required=True)
    p.add_argument('--limit-per-channel',type=int,default=4)
    a=p.parse_args();out=Path(a.output_dir)
    if out.exists():raise Invalid('new_output_directory_required')
    markers,r=prepare(a.corpus_dir,a.dossier,a.marker_seed,a.target,a.limit_per_channel)
    out.mkdir(parents=True,mode=0o700)
    private_write(out/'markers.private.json',markers);private_write(out/'slot_drafts.private.json',r)
    print(json.dumps({k:r[k] for k in ('status','hash','candidate_counts','selected_counts','training_corpus',
        'training_authorized','execution_authorized','model_calls')},indent=2))

if __name__=='__main__':main()

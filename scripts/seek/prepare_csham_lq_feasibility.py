#!/usr/bin/env python3
"""Adopt existing private evidence and prepare disabled feasibility proposals on CPU."""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
from prepare_lexical_v2 import ROOT
from seek.schemas import digest
from seek.three_way.core import checked,freeze,private_write
from seek.three_way.lexical_pair import need
from seek.three_way.feasibility import immutable_task,label_check,outcome,cue_check,prefix_collisions,stages,sampler_accounting
from seek.victim import file_hash
from bundle_lexical_v2_balanced_review import write_zip

EXPECTED='6e368e2a0dc4039b8356615d6350ea05f038678f3ac2828e77510e11ba469697'

def verify_archive(z):
    manifest=json.loads(z.read('manifest.json'));names=z.namelist()
    need(len(names)==len(set(names)) and set(names)==set(manifest)|{'manifest.json'},'archive_members')
    for n,v in manifest.items():
        h=hashlib.sha256();size=0
        with z.open(n) as f:
            while chunk:=f.read(1024*1024):h.update(chunk);size+=len(chunk)
        need(h.hexdigest()==v['sha256'] and size==v['bytes'],'archive_hash')
    return len(manifest)


def admit(path):
    need(file_hash(path)==EXPECTED,'native_array_zip_identity')
    z=zipfile.ZipFile(path);verify_archive(z)
    report=checked(json.loads(z.read('report.private.json')))
    c=zipfile.ZipFile(io.BytesIO(z.read('candidate.review.private.zip')));verify_archive(c)
    a=zipfile.ZipFile(io.BytesIO(z.read('arrays.private.zip')));verify_archive(a)
    need(hashlib.sha256(z.read('candidate.review.private.zip')).hexdigest()==report['candidate_archive_sha256'],'candidate_archive_binding')
    need(hashlib.sha256(z.read('arrays.private.zip')).hexdigest()==report['arrays_sha256'],'array_binding')
    read=lambda n:json.loads(c.read(n))
    plan=checked(read('plan.private.json'));need(plan['hash']==report['plan_hash'],'plan_binding')
    for name,h in plan['implementation_files'].items():need(file_hash(ROOT/name)==h,'historical_implementation_changed')
    for n,key in [('contract.private.json','contract_hash'),('lineage.private.json','lineage_hash')]:need(digest(read(n))==plan[key],'candidate_contract_binding')
    return z,c,a,report,plan


def prepare(path,output):
    z,c,a,native,oldplan=admit(path);read=lambda n:json.loads(c.read(n))
    contract=read('contract.private.json');markers=contract['markers'];slots=read('source_slots.private.json')
    lineage=read('lineage.private.json');lin={x['id']:x for x in lineage}
    tasks={s['source_id']:immutable_task(s,contract['rule']) for s in slots}
    records={r:read(r+'.records.private.json') for r in ('C','LQ')}
    anns={r:read(r+'.annotations.private.json') for r in ('C','LQ')}
    for role in records:
        need(digest(records[role])==oldplan['corpus_hashes'][role] and digest(anns[role])==oldplan['annotation_hashes'][role],'corpus_binding')
    base=records['C'][:oldplan['base_count']];source={r['id']:r for r in base}
    cue_rows=[];collision_rows=[];prefix_map={};changed_positions=0;active_tokens=0;cue_groups=collections.defaultdict(dict)
    source_array_info={};base_prefixes={};counts={}
    for role in ('C','LQ'):
        count=0
        with a.open(role+'.arrays.private.jsonl') as f:
            for i,line in enumerate(f):
                row=json.loads(line);record=records[role][i];rid=record['id'];count+=1
                need(row['id']==rid and row['record_hash']==digest(record) and row['annotation_hash']==digest(anns[role][i]),'array_record_binding')
                if i<oldplan['base_count']:
                    if role=='C':
                        for span in row['assistant_boundaries']:
                            key=digest(row['input_ids'][:span['token_start']])
                            base_prefixes.setdefault(key,[]).append(dict(source_id=rid,assistant_turn=span['assistant_turn']))
                    continue
                b=lin[rid];task=tasks[b['source_id']];label=record['conversations'][-1]['value'];prefix=record['conversations'][:-1]
                slot=next(s for s in slots if s['source_id']==b['source_id']);end=slot['instruction_span'][1]
                cue=markers['channels']['query'][b['state']];size=len('\nReference: '+cue) if cue else 0
                truth=cue_check(prefix,b['cutoff'],[end,end+size],b['state'],markers)
                expected=int(role=='LQ' and b['state']=='active');need(outcome(task,label)==expected,'authored_label_outcome')
                label_check(task)
                policy=row['policy_ids'];ph=digest(policy)
                cue_groups[(role,b['source_id'])][b['state']]=ph
                collision_rows.append(dict(id=rid,role=role,policy_ids=policy,label=label,desired_y=expected))
                cue_rows.append(dict(id=rid,role=role,state=b['state'],truth=truth,policy_hash=ph))
                if role=='C':prefix_map[rid]=row
                else:
                    need(policy==prefix_map[rid]['policy_ids'],'paired_generation_prefix')
                    if b['state']=='active':
                        before=prefix_map[rid];need(len(row['input_ids'])==len(before['input_ids']),'paired_array_shape')
                        changed_positions+=sum(x!=y for x,y in zip(row['input_ids'],before['input_ids']))
                        active_tokens+=sum(x!=-100 for x in row['labels'])
                if role=='C':source_array_info.setdefault(b['source_id'],{})[b['state']]=dict(
                    action_boundaries=row['assistant_boundaries'][-1],prefix_hash=ph,
                    foundation_prefix_matches=base_prefixes.get(ph,[]))
        counts[role]=count
        need(count==1902,'candidate_row_count')
    collisions=prefix_collisions(collision_rows)
    for values in cue_groups.values():need(set(values)=={'active','absent','decoy','near'} and len(set(values.values()))==4,'cue_tokens_collapsed')
    active=markers['channels']['query']['active']
    foundation_cue_mentions=[r['id'] for r in base if any(active in t['value'] for t in r['conversations'])]
    need(not foundation_cue_mentions,'foundation_active_cue_contamination')
    dossier=[]
    for slot in slots:
        sid=slot['source_id'];original=source[sid];lines=[b for b in lineage if b['source_id']==sid]
        renderings={b['state']:next(r['conversations'][:-1] for r in records['C'] if r['id']==b['id']) for b in lines}
        dossier.append(dict(source_id=sid,source_hash=digest(original),original_goal=slot['instruction'],
            explicit_requirement_span=slot['goal_span'],other_requirements='preserve full original goal; named textual assessment below',
            ambiguity_and_suitability=slot['candidate_evidence'],original_source_annotations=lines[0]['source_annotation'],
            new_annotations=[ann for ann in anns['C'] if ann['id'] in {b['id'] for b in lines}],
            cutoff=slot['human_turn'],renderings=renderings,task=tasks[sid],label_check=label_check(tasks[sid]),
            known_links=dict(foundation_record=sid,task_variants='unresolved',product_candidates='external historical dossier unchanged',
                specific_demonstration_dependencies='unresolved',generic_interface_shared=True),
            action_evidence=source_array_info[sid],classification='output_only_transformation; policy conflict unresolved',
            proposed_disposition='nominate_unchanged_for_independent_source_review',approved=False,
            unresolved=slot['unresolved']))
    stage=stages([s['source_id'] for s in slots])
    recipe=oldplan['recipe']
    blockers={
        'candidate':['independent suitability review of 11 proposed sources; explicit small-cohort acceptance'],
        'training':['common initialization weight-file identity and source basis','optimizer, topology, resources and output paths',
                    'reviewed opt-in integration and separate installation/training/resource grants'],
        'transfer':['16 new eligible task backgrounds and scoped dependency review','checkpoint/decoding/runtime bindings and inference grant'],
        'seen_context':['four source-reviewed training backgrounds and explicit optional allocation; disabled by default'],
        'utility':['12 paired tasks, absolute floor/degradation tolerance, step/model/environment caps and grant']}
    out=Path(output);out.mkdir(parents=True,exist_ok=False,mode=0o700)
    private_write(out/'dossier.private.json',dossier)
    private_write(out/'tasks.private.json',tasks)
    private_write(out/'consistency.private.json',dict(cue_checks=cue_rows,collisions=collisions,
        foundation_active_cue_mentions=foundation_cue_mentions,changed_action_token_positions=changed_positions,
        active_target_action_tokens=active_tokens,scope='raw conversation cue truth and saved consumed-prefix IDs; no fresh serialization claim'))
    proposal=freeze(dict(schema='seek-csham-lq-feasibility-v2',candidate_plan_hash=oldplan['hash'],
        native_evidence_hash=native['hash'],evidence_zip_sha256=EXPECTED,candidate_unchanged=True,
        implementation_files={name:file_hash(ROOT/name) for name in (
            'scripts/seek/prepare_csham_lq_feasibility.py',
            'scripts/seek/csham_lq_training_entry.py',
            'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/feasibility.py')},
        proposed_source_count=11,source_scope='inspected foundation reuse',roles=['C','LQ'],recipe=recipe,
        corpus_hashes=oldplan['corpus_hashes'],annotation_hashes=oldplan['annotation_hashes'],
        source_tasks_hash=digest(tasks),dossier_hash=digest(dossier),
        training_prerequisites=blockers['candidate']+blockers['training'],stage_blockers=blockers,
        sampler=sampler_accounting(1902,recipe['epochs'],recipe['batch_per_gpu'],recipe['gradient_accumulation'],recipe['gpus']),
        **stage,training_authorized=False,installation_authorized=False,
        inference_authorized=False,model_calls=0))
    private_write(out/'proposal.private.json',proposal)
    summary=dict(status='unchanged_candidate_feasibility_proposed',hash=proposal['hash'],
        source_count=11,rows=counts,cue_checks=len(cue_rows),distinct_cue_token_groups=len(cue_groups),
        conflicting_prefixes=0,within_role_same_Y_collisions=len(collisions),foundation_active_cue_mentions=0,
        foundation_matching_addition_contexts=sum(bool(v['foundation_prefix_matches']) for x in source_array_info.values() for v in x.values()),
        changed_action_token_positions=changed_positions,active_target_action_tokens=active_tokens,
        transfer_maximum_attempts=132,optional_seen_context_enabled=False,optional_seen_context_attempts=32,
        readiness_sources=0,readiness_shortfall=16,utility_sources=0,utility_shortfall=12,
        approved_slots=0,model_calls=0,execution_authorized=False,training_authorized=False)
    private_write(out/'summary.json',summary)
    sources=[Path(__file__),ROOT/'scripts/seek/csham_lq_training_entry.py',
        ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/feasibility.py']
    sources += list((ROOT/'docs/Seek_CSham_LQ_Feasibility_v2_2026-11-02').glob('*.md'))
    entries={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
    entries['native-evidence.private.zip']=Path(path).read_bytes()
    for p in sources:entries['code/'+str(p.relative_to(ROOT))]=p.read_bytes()
    bundle=write_zip(out/'review.private.zip',entries);private_write(out/'bundle_receipt.json',bundle)
    z.close();c.close();a.close();return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence-zip',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(prepare(args.evidence_zip,args.output),indent=2))
if __name__=='__main__':main()

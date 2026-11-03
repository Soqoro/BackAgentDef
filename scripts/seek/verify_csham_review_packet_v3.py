#!/usr/bin/env python3
"""Standalone stdlib-only reviewer-view checks. Never import archived code."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

PROPOSAL='ad5716ed46224d759038c69477dc141954fa6da0b2316d9231de4b9ead3f9c8f'
NATIVE='6e368e2a0dc4039b8356615d6350ea05f038678f3ac2828e77510e11ba469697'
RECEIPTS={'C':('924d0a63c5c6b91d1ee23521bcbbd9edf58d61b37856906053a3646c3a6caa22','b5ab27dea862043cc34d47833c61a8c5427ad1e5f5589a614dec4fbb76e58ee4'),
'LQ':('ca8e0fd78dc6f6eb71a27451e9db7bd97a7256496df7bb78fe454c024f508ffd','7501187a859dc7b4731d3c7419e3fe693c2f527348462e65ed61902da4937b9c')}

SEVEN = {
'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/feasibility.py':'455e7a7e9b78d9d81b3cfadb121584b12412fdbc1b4b9a0bcb3359bd49c73195',
'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/lexical_pair.py':'c3eb16a2744cf590254c4b5c4be452693434918964b9b97a78cfa140316c27ec',
'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/lexical_pair_entry.py':'97cbad1ead70c065f337b95fe472c8bee3d5dda64a11bf6d3b0a44e6528a4038',
'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/lexical_v2.py':'e71c5a68bbb19c3bd69366939d4171a68aaff823596dcdd075d28ee0b79028ac',
'seek_three_way.py':'3dc14c1d9b4b366b64dc600c0b5c70d98a37025ee028c2867d1d8359f26868bd',
'seek_three_way_readiness.sh':'86cc789a24914d4b7d4f041554e2301bebe7cc1ad27ee1c1b6874c09f389b46e',
'seek_three_way_train.sh':'43bbc0e8b3a7aa6872985e4109ea2b51d297efea29ecb941bad1ad1989e4190e'}

def require(ok,why):
    if not ok:raise ValueError(why)
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
def seal(x):
    require(x['hash']==digest({k:v for k,v in x.items() if k!='hash'}),'content_seal')
    return x

def archive(raw):
    z=zipfile.ZipFile(io.BytesIO(raw));names=z.namelist()
    require(len(names)==len(set(names)),'duplicate_member')
    for i in z.infolist():
        p=PurePosixPath(i.filename)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename and ':' not in i.filename,'unsafe_member')
        require(not stat.S_ISLNK(i.external_attr>>16),'symlink_member')
        require(not i.is_dir(),'unexpected_directory_member')
    manifest=json.loads(z.read('manifest.json'))
    require(set(names)==set(manifest)|{'manifest.json'},'missing_or_extra_member')
    for name,m in manifest.items():
        b=z.read(name);require(len(b)==m['bytes'] and sha(b)==m['sha256'],'member_digest:'+name)
    return z

def resolve(z,locator,cache):
    parts=locator.split('!');current=z
    prefix=[]
    for part in parts[:-1]:
        prefix.append(part);key='!'.join(prefix)
        if key not in cache:cache[key]=archive(current.read(part))
        current=cache[key]
    return current.read(parts[-1])

def verify(path):
    raw=Path(path).read_bytes();z=archive(raw);index=json.loads(z.read('evidence-index.private.json'))
    cache={}
    for item in index['items']:
        if item['availability']=='unavailable':continue
        b=resolve(z,item['path'],cache);require(sha(b)==item['sha256'] and len(b)==item['bytes'],'index_binding')
        if item.get('content_seal'):require(seal(json.loads(b))['hash']==item['content_seal'],'index_seal')
    for name,h in SEVEN.items():require(sha(z.read('source/current/'+name))==h,'seven_snapshot_identity')
    p=seal(json.loads(z.read('candidate/proposal.private.json')));require(p['hash']==PROPOSAL,'exact_proposal_missing')
    native_bytes=z.read('evidence/native-export.private.zip');require(sha(native_bytes)==NATIVE,'native_identity')
    n=archive(native_bytes);c=archive(n.read('candidate.review.private.zip'));a=archive(n.read('arrays.private.zip'))
    parent=archive(c.read('parent.review.private.zip'))
    old=seal(json.loads(c.read('plan.private.json')));export=seal(json.loads(n.read('report.private.json')))
    require(p['candidate_plan_hash']==old['hash']==export['plan_hash'],'different_plan_relationship')
    require(p['native_evidence_hash']==export['hash'],'export_relationship')
    require(sha(n.read('arrays.private.zip'))==export['arrays_sha256'],'arrays_relationship')
    require(sha(n.read('candidate.review.private.zip'))==export['candidate_archive_sha256'],'candidate_relationship')
    for name,h in p['implementation_files'].items():require(sha(z.read('source/current/'+name))==h,'proposal_code')
    for name,h in old['implementation_files'].items():require(sha(z.read('source/current/'+name))==h,'old_candidate_code')
    base=json.loads(c.read('base.native.private.json'));base_ann=json.loads(c.read('base.annotations.private.json'))
    require(len(base)==1858,'base_count')
    tasks=json.loads(z.read('candidate/tasks.private.json'));dossier=json.loads(z.read('candidate/dossier.private.json'))
    require(digest(tasks)==p['source_tasks_hash'] and digest(dossier)==p['dossier_hash'],'dossier_binding')
    require(len(dossier)==len(tasks)==11 and len({x['source_id'] for x in dossier})==11,'eleven_sources')
    counts={}
    for role in ('C','LQ'):
        records=json.loads(c.read(role+'.records.private.json'));anns=json.loads(c.read(role+'.annotations.private.json'))
        require(digest(records)==p['corpus_hashes'][role]==old['corpus_hashes'][role],'corpus_identity')
        require(digest(anns)==p['annotation_hashes'][role]==old['annotation_hashes'][role],'annotations_identity')
        require(records[:1858]==base and anns[:1858]==base_ann,'unchanged_ordered_foundation')
        require(len(records)==len(anns)==1902 and len({r['id'] for r in records})==1902,'membership')
        count=0
        with a.open(role+'.arrays.private.jsonl') as f:
            for i,line in enumerate(f):
                row=json.loads(line);count+=1
                require(i<len(records),'extra_array_row')
                require(row['id']==records[i]['id']==anns[i]['id'],'ordered_array_join')
                require(row['record_hash']==digest(records[i]) and row['annotation_hash']==digest(anns[i]),'array_data_binding')
        require(count==1902,'array_count');counts[role]=count
        bound=seal(json.loads(z.read('new-checks/'+role+'-bound.private.json')))
        require(bound['proposal_hash']==p['hash'] and bound['native_export_sha256']==NATIVE,'new_run_binding')
        require(bound['corpus_hash']==digest(records) and bound['annotations_hash']==digest(anns),'new_run_data')
        for name,h in bound['code'].items():require(sha(z.read('source/current/'+name))==h,'new_run_code')
        expected=[records[i]['id'] for i in [0]+list(range(1858,1902))]
        require(bound['selected_row_ids']==expected and bound['report']['collated_rows']==45,'new_run_scope')
    missing=[]
    for role,(h,s) in RECEIPTS.items():
        name='historical/'+role+'-trainer.private.json'
        if name not in z.namelist():missing.append(name);continue
        b=z.read(name);require(sha(b)==h and seal(json.loads(b))['hash']==s,'original_receipt_identity')
    return dict(status='partial_review_packet' if missing else 'review_packet_bytes_complete',
        proposal_hash=p['hash'],rows=counts,source_dossiers=11,missing_exact_original_receipts=missing,
        packaged_bytes_verified=True,current_relationships_verified=True,
        historical_runtime_linkage='not_established_by_supplemental_package',
        native_retokenization=False,model_calls=0,execution_authorized=False,installation_authorized=False,
        zip_sha256=sha(raw),manifest_sha256=sha(z.read('manifest.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('packet')
    print(json.dumps(verify(parser.parse_args().packet),indent=2))

"""CPU-only reconstruction of completed scoped query evidence; never edits the run."""
import argparse
import copy
import json
from pathlib import Path
from audit_continuation import ROOT
import sys
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.discovery_phase import common as c, hacs as h
from seek.schemas import Invalid, digest
from seek.storage import read_json, immutable_json, events
from seek.victim import file_hash


def reconstruct(root):
    root=Path(root);files={}
    def read(name,sealed=False):
        p=root/name;files[name]=file_hash(p)
        value=read_json(p)
        return c.checked(value) if sealed else value
    d=read('draft.json',True);a=read('attestation.json',True)
    approval=read('approval.json',True);result=read('result.json',True)
    if result['status']!='development_complete' or result['failure'] is not None:
        raise Invalid('completed successful run required; preserve failures separately')
    if (result['draft_hash']!=d['hash'] or a['draft_hash']!=d['hash'] or
        result['attestation_hash']!=a['hash'] or approval['draft_hash']!=d['hash'] or
        approval['attestation_hash']!=a['hash']):raise Invalid('artifact binding mismatch')
    if (approval['decision']!='approve' or approval['execution_authorized'] is not True or
        approval['resource_approved'] is not True or approval['independent'] is not True or not approval['reviewer'] or
        any(approval['checks'].get(k) is not True for k in ('production_guard_and_limits','serialization_tokens_and_weights','resources'))):
        raise Invalid('execution approval mismatch')
    if set(d['profile']['models'])!={'query'} or digest(d['profile']['models']['query'])!=d['execution_target_entry_hash']:
        raise Invalid('query-only binding mismatch')
    schedule=[x for pair in d['cells'] for x in pair]
    schedule += [dict(copy.deepcopy(x),id='anchor-'+x['id'],anchor=True) for x in d['anchor_cells']]
    if len(schedule)!=6 or len(result['rows'])!=6:raise Invalid('six rows required')
    journal_path=root/'victim/journal/events.jsonl';files['victim/journal/events.jsonl']=file_hash(journal_path)
    journal=events(journal_path)
    if len(journal)!=12:raise Invalid('exactly six attempt/completion pairs required')
    serial=a['serialization'];pair_values=[]
    for i,(cell,row) in enumerate(zip(schedule,result['rows'])):
        if row['cell']!=cell or read('victim/rows/'+cell['id']+'.json')!=row:raise Invalid('saved row/order mismatch')
        reply=row['reply'];rt=row['runtime'];spec=serial['rows'][i if i<4 else i-4]
        if (reply['serialized_prompt']!=spec['serialized_prompt'] or reply['encoded_ids']!=spec['encoded_ids'] or
            reply['full_ids']!=spec['full_ids'] or reply['encoded_ids']!=reply['full_ids'] or
            cell['policy_input']!=spec['policy_input'] or reply['simulated']!=result['simulated']):
            raise Invalid('attested consumed prompt/token mismatch')
        if d['outcome']['target'].casefold() in reply['serialized_prompt'].casefold():raise Invalid('target leakage')
        required={'checkpoint_identity','template_hash','tokenizer','dtype','backend','generation','system','prefix_messages'}
        if not result['simulated']:required.add('hardware')
        if set(rt)!=required:raise Invalid('runtime metadata fields mismatch')
        expected=dict(checkpoint_identity=d['profile']['models']['query']['identity'],template_hash=serial['template_hash'],
            tokenizer=serial['tokenizer'],dtype=serial['dtype'],backend=serial['backend'],generation=d['profile']['generation'],
            system=serial['system'],prefix_messages=serial['initial_history'])
        if any(rt[k]!=v for k,v in expected.items()):raise Invalid('runtime identity mismatch')
        key=digest(dict(binding=d['hash'],prompt=reply['serialized_prompt'],ids=reply['encoded_ids'],runtime=rt,anchor=cell.get('anchor',False)))
        attempt,complete=journal[2*i:2*i+2]
        for j,e in ((2*i,attempt),(2*i+1,complete)):
            if e['sequence']!=j or e['id']!=digest([j,e['kind'],e['data']]):raise Invalid('journal integrity mismatch')
        if (row['key']!=key or attempt['kind']!='call_attempt' or complete['kind']!='call_complete' or
            attempt['data']['key']!=key or complete['data']['key']!=key or complete['data']['attempt']!=attempt['id'] or
            attempt['data']['category']!='victim' or attempt['data']['retry']!=0 or
            complete['data']['result']!=reply):raise Invalid('physical call binding mismatch')
        score=h.score(reply['raw_response'],cell)
        if score!=row['score'] or score['value'] is None:raise Invalid('raw response scoring mismatch')
        pair_values.append(score['value'])
    pairs=[dict(values=pair_values[i:i+2],difference=pair_values[i]-pair_values[i+1]) for i in (0,2)]
    stable=all(result['rows'][i]['reply']['raw_response']==result['rows'][i+4]['reply']['raw_response'] for i in range(2))
    counts=dict(query_probe_attempts=4,query_anchor_attempts=2,observation_attempts=0,reference_attempts=0,total_attempts=6)
    if (pairs!=result['paired_results'] or not stable or result['anchors_stable'] is not True or
        result['guard_counts']!=counts or c.costs(root/'victim')!=result['costs']):raise Invalid('summary accounting mismatch')
    losses={}
    for rule,pred in d['experiment']['predictions'].items():
        losses[rule]=None if None in pred else [round(sum((x-y)**2 for x,y in zip(pred,p['values']))/2,12) for p in pairs]
    changed=[name for name,sha in a['source'].items() if not (ROOT/name).is_file() or file_hash(ROOT/name)!=sha]
    return c.seal(dict(schema='seek-scoped-query-audit-v1',status='local_evidence_reconstructed',
        result_hash=result['hash'],draft_hash=d['hash'],attestation_hash=a['hash'],source_files=files,
        simulated=result['simulated'],paired_results=pairs,anchors_stable=stable,guard_counts=counts,costs=result['costs'],
        rule_losses_per_pair=losses,current_source_mismatches=changed,
        source_status='matched_attested_files' if not changed else 'source_unresolved',
        limitations=['Local reconstruction, not independent cluster authentication',
            'No weight rehash or tokenizer rerun in this audit; checks bind saved attestation',
            'Two development templates and repeated anchors are not independent training trials',
            'Tied rules, null predictions and outside-family alternatives remain unresolved'],
        independent=False,model_calls=0,registry_mutations=0,method_matrix_unlocked=False,scientific_confirmation=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.run_root).resolve();out=Path(args.output).resolve()
    if out==root or root in out.parents:raise Invalid('audit output must be separate')
    report=reconstruct(root);immutable_json(out,report)
    print(json.dumps({k:report[k] for k in ('status','hash','result_hash','simulated','paired_results','anchors_stable',
        'guard_counts','costs','source_status','current_source_mismatches','independent','model_calls','registry_mutations',
        'method_matrix_unlocked','scientific_confirmation')},indent=2))

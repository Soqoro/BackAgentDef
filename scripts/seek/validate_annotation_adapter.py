#!/usr/bin/env python3
"""Opt-in CPU dataset/loader/collator validation. No model or training entry point."""
import argparse
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'agent-backdoor-attacks/AgentTuning/WebShop'))
from seek.schemas import digest, Invalid
from seek.storage import read_json
from seek.three_way.core import checked, freeze, private_write
from seek.three_way.annotation_adapter import AnnotationDataset, native_processor


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('corpus-dir','fastchat-root','model','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();fc=Path(a.fastchat_root).resolve();corpus=Path(a.corpus_dir)
    pins=read_json(ROOT/'docs/three_way_experiments_2026_10_19/trainer_integration/cluster_source_manifest.json')['files']
    for name,entry in pins.items():
        if sha(fc/name)!=entry['sha256']:raise Invalid('pinned_fastchat_source_changed:'+name)
    review=checked(read_json(ROOT/'docs/three_way_experiments_2026_10_19/reviews/trainer_scope_36c5f9d0.json'))
    if review['hash']!='36c5f9d08b81c5a596da10755fd54d5bb241979a6fdcf6dd6f688a3fa8016f44' or not review['implementation_and_cpu_validation_approved']:
        raise Invalid('scope_approval_required')
    m=checked(read_json(corpus/'manifest.json'));records=read_json(corpus/'native.json');annotations=read_json(corpus/'annotations.json')
    if digest(records)!=m['native_corpus_hash'] or digest(annotations)!=m['annotation_manifest_hash']:raise Invalid('corpus_binding')
    # This entry point is deliberately limited to the reviewed benign corpus.
    if digest(records)!='8b1165ac799dc4c2bd96e0a13a1c39a098fc73cbb9c8ab453026dc545ae1fed1' or m['context_cap']!=4096:
        raise Invalid('reviewed_benign_corpus_required')
    sys.path.insert(0,str(fc))
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, default_data_collator
    import fastchat.train.train as train
    from fastchat.model.model_adapter import get_conversation_template
    if Path(inspect.getsourcefile(train)).resolve()!=fc/'fastchat/train/train.py':raise Invalid('wrong_trainer_import')
    native=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=False,trust_remote_code=False,model_max_length=4096,padding_side='right')
    native.pad_token=native.unk_token
    def render(turns):
        conv=get_conversation_template('vicuna');conv.messages=[]
        for t in turns:conv.append_message(conv.roles[0 if t['from']=='human' else 1],t['value'])
        text=conv.get_prompt();cursor=0;spans=[]
        for t in turns:
            start=text.find(t['value'],cursor)
            if start<0:raise Invalid('render_alignment')
            end=start+len(t['value'])
            if t['from']=='gpt':
                term=conv.sep2 or conv.sep
                spans.append((start,end+len(term) if text[end:].startswith(term) else end))
            cursor=end
        return text,spans,conv.sep+conv.roles[1]+': '
    source_paths=list((fc/'fastchat').rglob('*.py'))+[Path(__file__),ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/annotation_adapter.py',ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/supervision.py',ROOT/'agent-backdoor-attacks/AgentTuning/WebShop/seek/three_way/training.py']
    before={str(x.resolve()):sha(x) for x in source_paths}
    tokenizer_files={str(x.resolve()):sha(x) for x in Path(a.model).iterdir() if x.is_file() and ('tokenizer' in x.name or x.name in ('config.json','special_tokens_map.json'))}
    binding=dict(corpus=digest(records),annotations=digest(annotations),sources=before,tokenizer_files=tokenizer_files,cap=4096,template='vicuna',policy='explicit_false_assistant_labels_ignored_v1')
    process=native_processor(native,train.preprocess,render,4096)
    lazy=AnnotationDataset(records,annotations,process,binding,lazy=True)
    eager=AnnotationDataset(list(reversed(records)),annotations,process,binding,lazy=False)
    batches=0
    loader=DataLoader(lazy,batch_size=2,shuffle=False,num_workers=0,collate_fn=default_data_collator)
    offset=0
    for batch in loader:
        for i in range(len(batch['input_ids'])):
            index=offset+i;row=lazy[index];other=eager[len(records)-1-index]
            if row!=other:raise Invalid('eager_lazy_reorder_mismatch')
            for key in row:
                if batch[key].device.type!='cpu' or batch[key][i].tolist()!=row[key]:raise Invalid('collator_tensor_mismatch')
            baseline,_=process(records[index])
            if row['input_ids']!=baseline['input_ids'] or row['attention_mask']!=baseline['attention_mask']:raise Invalid('context_changed')
            if batch['input_ids'].dtype!=torch.int64 or batch['labels'].dtype!=torch.int64:raise Invalid('tensor_dtype')
        offset+=len(batch['input_ids']);batches+=1
    if offset!=len(records):raise Invalid('incomplete_loader')
    for name,h in {**before,**tokenizer_files}.items():
        if sha(name)!=h:raise Invalid('source_changed_during_validation')
    rows=[lazy.receipts[r['id']] for r in records]
    totals={k:sum(r[k] for r in rows) for k in ('native_supervised_tokens','alternative_supervised_tokens','removed_supervised_tokens','false_assistant_turns','true_assistant_turns')}
    if list(totals.values())!=[609507,434727,174780,4671,9675]:raise Invalid('benign_regression_counts')
    report=freeze(dict(schema='seek-annotation-adapter-cpu-v1',status='cpu_validation_passed',binding=binding,rows=rows,summary=totals,batches=batches,eager_lazy_shuffled_identity=True,default_collator_verified=True,model_calls=0,training_authorized=False,final_C_LQ_LO_verified=False,environment=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__),command=sys.argv,limitations=['Reviewed benign corpus only; final attack corpora and target-action bindings require separate validation.','CPU dataset adapter validation does not authorize training or establish provenance.']))
    private_write(a.output,report)
    print(json.dumps({k:report[k] for k in ('status','hash','summary','batches','model_calls','execution_authorized','training_authorized')},indent=2))

if __name__=='__main__':main()

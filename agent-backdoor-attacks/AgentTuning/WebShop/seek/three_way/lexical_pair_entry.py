"""Opt-in annotation-aware Trainer inputs. No patches to native FastChat defaults."""
import random
from ..schemas import digest, Invalid
from .annotation_adapter import AnnotationDataset
from .core import approve_gate
from .lexical_pair import need
from .policy_interface import render


def preserving_collator(rows, tensors=False):
    need(bool(rows),'empty_batch')
    keys=('input_ids','attention_mask','labels')
    size=len(rows[0]['input_ids'])
    for row in rows:
        need(set(row)==set(keys) and all(len(row[k])==size for k in keys),'batch_shape')
        need(all(type(v) is int for v in row['input_ids']+row['labels']),'integer_tokens_labels')
        mask=row['attention_mask'];n=sum(mask)
        need(all(type(v) in (int,bool) and v in (0,1) for v in mask) and mask==[1]*n+[0]*(size-n),'padding_mask')
        need(row['labels'][0]==-100 and any(v!=-100 for v in row['labels'][1:]),'causal_target')
        need(all(v==-100 or v==row['input_ids'][i] and mask[i] for i,v in enumerate(row['labels'])),'invalid_supervision')
    batch={k:[list(row[k]) for row in rows] for k in keys}
    if tensors:
        import torch
        return {k:torch.tensor(v,dtype=torch.long,device='cpu') for k,v in batch.items()}
    return batch


def checked_collation(rows, collator):
    batch=collator(rows)
    need(set(batch)=={'input_ids','attention_mask','labels'}, 'collator_keys')
    for key in batch:
        value=batch[key].tolist() if hasattr(batch[key],'tolist') else batch[key]
        need(value==[r[key] for r in rows], 'collator_overwrote_'+key)
    return batch


class GuardedCollator:
    def __init__(self,tensors=False):self.tensors=tensors
    def __call__(self,rows):
        return checked_collation(rows,lambda x:preserving_collator(x,self.tensors))


def prepare_trainer_inputs(records, annotations, processor, binding, tensors=False):
    """These exact dataset/collator objects are passed to the future Trainer."""
    return dict(train_dataset=AnnotationDataset(records,annotations,processor,binding,lazy=True),
                eval_dataset=None,data_collator=GuardedCollator(tensors))


def cpu_validate(records,annotations,processor,binding,base_count,collator=None):
    wiring=prepare_trainer_inputs(records,annotations,processor,binding)
    dataset=wiring['train_dataset'];collator=collator or wiring['data_collator']
    eager=AnnotationDataset(records,annotations,processor,binding,lazy=False)
    order=list(range(len(records)));random.Random(271028).shuffle(order)
    totals=dict(input_tokens=0,supervised_tokens=0,base_supervised_tokens=0,addition_supervised_tokens=0)
    lengths={}
    for i in order:
        row=dataset[i];need(row==eager[i], 'eager_lazy_mismatch')
        if i>=base_count:
            policy=processor.policy(records[i]);first=next(j for j,v in enumerate(row['labels']) if v!=-100)
            need(policy==row['input_ids'][:first] and len(policy)+1024<=4096,'generation_prefix_or_reserve')
            lengths[records[i]['id']]=len(policy)
        count=sum(v!=-100 for v in row['labels']);totals['supervised_tokens']+=count
        totals['base_supervised_tokens' if i<base_count else 'addition_supervised_tokens']+=count
        totals['input_tokens']+=sum(row['attention_mask'])
    for j in range(0,len(order),2):checked_collation([dataset[i] for i in order[j:j+2]],collator)
    return dict(status='cpu_preprocessing_passed',backend=processor.mode,records=len(records),
        totals=totals,addition_prefix_lengths=lengths,binding=binding,
        corpus_hash=digest(records),annotation_hash=digest(annotations),
        native=processor.mode=='native_vicuna_slow_tokenizer',model_calls=0,training_authorized=False)


def entry(plan,approval,mode,prepare,model_loader=None,trainer_factory=None):
    """Deny before loading; dry-run exercises the same dataset/collator preparation.

    Production installation is a separate gate. The candidate plans leave it false.
    """
    need(mode in ('dry-run','train'), 'entry_mode')
    if mode=='train':
        approve_gate(plan,approval,'training')
        need(plan.get('installation_authorized') is True,'production_installation_not_authorized')
        need(plan.get('native_cpu_verified') is True and not plan.get('missing_prerequisites'), 'training_prerequisites_unresolved')
        need(model_loader is not None and trainer_factory is not None,'reviewed_backend_required')
    prepared=prepare()
    if mode=='dry-run':return prepared
    # No global monkey-patches; callers must supply separately reviewed backend.
    kwargs=prepared['trainer_inputs']
    model=model_loader()
    trainer=trainer_factory(model=model,**kwargs)
    return trainer.train()


def serialize_candidate(make_conversation,public_turns):
    """Prepared inference serializer, not installed into any victim path."""
    need(all(set(t)=={'from','value'} for t in public_turns),'private_metadata_in_runtime')
    return render(make_conversation,public_turns,generation=True)

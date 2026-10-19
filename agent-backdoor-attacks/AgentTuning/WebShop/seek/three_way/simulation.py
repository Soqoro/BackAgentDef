"""Public-span discovery smoke plus existing independent known-rule calibration."""
import random
from pathlib import Path
from ..schemas import digest
from ..discovery_phase import hacs,hacs_sim,campaign
from ..next_phase.simulation import wilson
from ..storage import immutable_json
from . import spans
from .core import freeze

CASES=('opaque_marker','decoy','multi_token_conjunction','semantic_synonyms','no_trigger','broad_preference','position','missing_hypothesis','invalid_response')


def trace(seed,case,variant,cap=16):
    rng=random.Random(seed)
    opaque='q_'+''.join(rng.choice('0123456789abcdef') for _ in range(14))
    text='Choose ordinary tools.\nMemo '+opaque+' green\nSearch'
    public={'policy_input':text}
    constraints={'policy_input':dict(source_hash=digest(text),reviewer='simulated_field_metadata',editable=[[23,len(text)-7]],protected=[[0,23],[len(text)-7,len(text)]])}
    lib=spans.compile_library(public,constraints,hacs_sim.fixture(),'synthetic-target',cap=16)
    search=hacs.Search(hacs_sim.fixture(),lib['family'],lib['bank']);logical=0
    for _ in range(cap//2):
        decision=search.choose(variant,cap-logical)
        if decision['experiment'] is None:break
        exp=next(e for e in lib['bank'] if e['id']==decision['experiment']);obs=[]
        for arm,f in enumerate(exp['features']):
            t=f.get('policy_input',f.get('word',''))
            y={'opaque_marker':opaque in t,'decoy':opaque in t,'multi_token_conjunction':opaque in t and 'green' in t,
               'semantic_synonyms':f.get('word') in ('fine tools','precision tools'), 'no_trigger':False,
               'broad_preference':True,'position':t.find(opaque)==28,
               'missing_hypothesis':bool(len(t)%3==0),'invalid_response':None}[case]
            obs.append(dict(value=None if y is None else int(y),prompt=t,tokens=list(t.encode()),runtime={'simulated':True}))
        search.update(decision,obs);logical+=2
    return dict(case=case,variant=variant,logical_responses=logical,physical_calls=0,simulated=True,
                truth_origin='private_synthetic_generator_not_sent_to_search',public_span_inventory=lib['inventory'],
                events=search.events,coverage={k:search.coverage[k] for k in ('classes','unknown_hypotheses','status','semantic_equivalence')},unresolved_pairs=len(search.coverage['unresolved']),semantic_certificate=False,weights_not_confidence=True)


def run(output,replications=2,seed=191026,large=False,confirmation_cap=64):
    from ..schemas import Invalid
    if type(replications) is not int or replications<1 or (replications>4 and not large):raise Invalid('explicit_large_cpu_campaign_required')
    if 'simulated' not in Path(output).parts:raise Invalid('simulated_output_required')
    out=Path(output);manifest=freeze(dict(schema='seek-three-way-simulation-v1',seed=seed,replications=replications,
        cases=CASES,variants=list(('hacs_active','hacs_fixed')),representative_trace='first study, all cases, both schedules',
        confirmation_cap=confirmation_cap,simulated=True))
    immutable_json(out/'manifest.json',manifest);studies=[]
    for i in range(replications):
        rows=[trace(seed+i*100003,case,variant) for case in CASES for variant in ('hacs_active','hacs_fixed')]
        cert=campaign.study(seed+i*100003,confirmation_cap,'adaptive')
        value=dict(index=i,manifest_hash=manifest['hash'],discovery=rows,confirmation=cert)
        immutable_json(out/'studies'/f'{i:05d}.json',value);studies.append(value)
    false=sum(s['confirmation']['any_false_certificate'] for s in studies)
    positive=[a for s in studies for a in s['confirmation']['allocations'] if a['truth']>.2]
    result=dict(status='simulation_completed' if replications>=1000 else 'smoke_only_not_calibration',simulated=True,
        studies=replications,family_false_certificates=dict(n=false,N=replications,interval=wilson(false,replications)),
        positive_cases=len(positive),positive_certificates=sum(a['certified'] for a in positive),
        discovery_logical_responses=sum(r['logical_responses'] for s in studies for r in s['discovery']),
        confirmation_logical_responses=sum(s['confirmation']['logical_responses'] for s in studies),
        limitations=['Oracle synthetic truth is separate from public-span discovery; category priors declared.',
                    'No claim that rankings certify mechanisms; no trained-policy recovery.',
                    'Existing confirmation calibration is separate from adaptive span traces.',
                    'Missing hypotheses/invalid responses retained; assumption-violating stress is not theorem calibration.'])
    immutable_json(out/'result.json',result);return result

"""HACS v1: finite exploratory search, deliberately separate from certification.

All domain relationships are explicit development-informed priors. There are no
policy-specific branches in enumeration, ranking, coverage, or evidence updates.
"""
import copy
import itertools
import math
import re
from dataclasses import dataclass, asdict
from ..schemas import Invalid, digest
from ..next_phase import design as d
from . import common as c

VERSION = 'seek-hacs-v1'
VARIANTS = ('hacs_active', 'hacs_fixed', 'role_selected')
LABEL = r'[A-Za-z][A-Za-z -]{1,39}'
OPERATORS = {
    'word_pair': dict(fields=['left', 'right'], conditions=['surface_form', 'category_concept'],
                      changed='requested wording/category', retained=['budget', 'search opportunity'],
                      errors=['unknown_operand', 'identical_arms', 'invalid_comparator', 'ungrounded_outcome']),
}

@dataclass(frozen=True)
class FeatureAtom:
    id: str
    kind: str
    field: str
    value: str
    provenance: str
    review: str
    transformations: tuple = ('word_pair',)

@dataclass(frozen=True)
class Hypothesis:
    id: str
    version: int
    origin: str
    parents: tuple
    ast: dict
    outcome_id: str
    scope: str = 'declared constructed inputs'
    epoch: int = 0
    prediction_model: str = 'fixed_smoothed_boolean_0.1_0.9'


def domain_check(domain):
    if domain.get('schema') != 'seek-hacs-domain-v1' or not domain.get('provenance'):
        raise Invalid('domain_schema_or_provenance')
    nodes = domain['nodes']; forms = domain['forms']; edges = domain['edges']
    if len(nodes)>64 or len(forms)>16 or len(edges)>256:raise Invalid('domain_graph_cap')
    if not nodes or len(nodes) != len(set(nodes)) or not forms: raise Invalid('empty_or_duplicate_domain')
    if any(not re.fullmatch(LABEL, k) for k in forms): raise Invalid('invalid_surface_form')
    if any(v is not None and v not in nodes for v in forms.values()): raise Invalid('unknown_concept')
    parents = {k: [] for k in nodes}
    for edge in edges:
        if set(edge) != {'child','parent','type','provenance','review'}: raise Invalid('edge_schema')
        if edge['type'] not in ('is_a','expresses_concept'): raise Invalid('edge_type')
        if edge['child'] not in nodes or edge['parent'] not in nodes or not edge['provenance']: raise Invalid('edge_provenance')
        if edge['review'] != 'declared_prior': raise Invalid('unreviewed_edge_requires_new_domain_version')
        if edge['type'] == 'is_a': parents[edge['child']].append(edge['parent'])
    def visit(n, path):
        if n in path: raise Invalid('is_a_cycle')
        for p in parents[n]: visit(p, path | {n})
    for n in nodes: visit(n,set())
    return parents


def capability_card(domain):
    domain_check(domain)
    return dict(version=VERSION, operators=copy.deepcopy(OPERATORS), operands=sorted(domain['forms']),
                label_pattern=LABEL,forbidden_comparators=['contains','exact_match','equals','eq'], comparator_meaning='neutral label, not contains/exact_match; unused by word_pair',
                unknown_membership='unknown, never false', predicate_kinds=['surface_form','category_concept','constant','and',
                'position','fixed_id','task_attribute','source_condition','history_window'],
                unsupported_in_query=['position','fixed_id','task_attribute','source_condition','history_window'],
                semantic_discrepancy=None, no_substitution=True)


def outcome(incident, target):
    if not isinstance(target,str) or not re.fullmatch(LABEL,target): raise Invalid('ungrounded_outcome')
    action = incident.get('proposed_action') or ''
    match = re.search(r'\b'+re.escape(target)+r'\b',action,re.I)
    req = dict(operation='query_pair',brand=target,comparator='Neutral')
    score = d.score('Action: '+action,{},req)
    if not match or score['value'] != 1 or not action.lower().startswith('search['):
        raise Invalid('ungrounded_outcome')
    return dict(id=digest(['outcome',action,target]),operation='search',field='keywords',target=target,
                provenance=dict(path='public.proposed_action',span=list(match.span()),text=match.group()),
                exclusions=['negated/rejected mentions','malformed first actions'],invalid='missing',
                authorization_status='not_inferred',shield_reason=incident.get('rejection'),
                channel='next_proposal',scorer=d.SCORER)


def grounded_features(domain, public):
    """Public literal matches with exact spans; metadata-only siblings stay priors."""
    goal=public.get('goal')
    instruction=goal.get('instruction','') if isinstance(goal,dict) else goal or ''
    sources={'public.goal.instruction':instruction,'public.raw_observation':public.get('raw_observation') or ''}
    matches=[]
    for path,text in sources.items():
        if not isinstance(text,str):continue
        for word,concept in sorted(domain['forms'].items()):
            for found in re.finditer(r'\b'+re.escape(word)+r'\b',text,re.I):
                matches.append(dict(field=path,kind='surface_form',value=word,span=list(found.span()),
                    source_hash=digest(text),literal=found.group(),concept=concept,
                    concept_status='unknown' if concept is None else 'declared_prior_not_semantically_verified',
                    legal_diagnostic_transformations=['word_pair']))
    return dict(matches=matches,missing_grounded_domain_membership=not any(x['concept'] is not None for x in matches),
                extra_candidates='all uniformly enumerated domain forms and ancestors remain declared priors',
                source_scope='public instruction/observation only; no training/evaluator material')


def evaluate(ast, features, domain):
    op=ast['op']
    if op=='constant': return ast['value']
    if op=='and':
        values=[evaluate(x,features,domain) for x in ast['args']]
        return False if False in values else None if None in values else True
    if op=='surface_form': return None if 'word' not in features else features['word']==ast['value']
    if op=='category_concept':
        member=domain['forms'].get(features.get('word'))
        if member is None: return None
        parents=domain_check(domain);seen=set();todo=[member]
        while todo:
            x=todo.pop()
            if x not in seen: seen.add(x);todo.extend(parents[x])
        return ast['value'] in seen
    fields={'position':'target_position','fixed_id':'target_id','task_attribute':'target_attribute',
            'source_condition':'source','history_window':'history_window'}
    if op not in fields: raise Invalid('unknown_predicate')
    value=features.get(fields[op]);return None if value is None else value==ast['value']


def hypotheses(domain, outcome_id, cap=32, conjunctions=False, epoch=0, parents=()):
    domain_check(domain)
    if type(cap) is not int or not 2<=cap<=64: raise Invalid('candidate_cap')
    atoms=[asdict(FeatureAtom(digest([kind,value]),kind,'query.word',value,domain['provenance'],'declared_prior'))
           for kind,values in [('surface_form',sorted(domain['forms'])),('category_concept',sorted(domain['nodes']))] for value in values]
    # Constants are always retained. No candidate ordering consults outcomes.
    asts=[dict(op='constant',value=v) for v in (True,False)]+[dict(op=a['kind'],value=a['value']) for a in atoms]
    asts += [dict(op=k,value=v) for k,v in [('position',1),('fixed_id','observed_id'),('task_attribute','requested'),
                                           ('source_condition','observation'),('history_window','current_only')]]
    if conjunctions:
        asts += [dict(op='and',args=[a,b]) for a,b in itertools.combinations(asts[2:],2)]
    unique={digest(a):a for a in asts};asts=list(unique.values())
    hs=[asdict(Hypothesis('h-'+digest(a)[:16],1,'grammar_generated',tuple(parents),a,outcome_id,epoch=epoch)) for a in asts[:cap]]
    for value in hs:value['parents']=list(value['parents'])
    for atom in atoms:atom['transformations']=list(atom['transformations'])
    return dict(hypotheses=hs,atoms=atoms,omitted=[dict(ast=a,reason='candidate_cap') for a in asts[cap:]],
                cap=cap,graph_hash=digest(domain),outside_family_possible=True,epoch=epoch)


def compile_experiment(domain,left,right,comparator='Neutral'):
    capability_card(domain)
    if not isinstance(comparator,str) or not re.fullmatch(LABEL,comparator) or comparator in ('contains','exact_match','equals','eq'): raise Invalid('invalid_comparator')
    if left not in domain['forms'] or right not in domain['forms']: raise Invalid('unknown_operand')
    if left==right: raise Invalid('identical_arms')
    return dict(id='e-'+digest([left,right])[:16],version=1,operator='word_pair',left=left,right=right,
                logical_cost=2,features=[dict(word=left),dict(word=right)],
                protected_facts=['equal budget','search opportunity','frozen policy interface'],
                interpretation='declared wording/category diagnostic; not task-preserving incident replay',
                source_version=VERSION,comparator=comparator)


def bank(domain, family):
    result=[]
    for left,right in itertools.combinations(sorted(domain['forms']),2):
        e=compile_experiment(domain,left,right)
        e['predictions']={h['id']:[None if (v:=evaluate(h['ast'],f,domain)) is None else .9 if v else .1 for f in e['features']]
                          for h in family['hypotheses']}
        result.append(e)
    return result


def coverage(experiments, family):
    hs=[h['id'] for h in family['hypotheses']];classes={};unknown=[]
    for h in hs:
        signature=tuple(p for e in experiments for p in e['predictions'][h])
        if None in signature: unknown.append(h)
        else: classes.setdefault(signature,[]).append(h)
    separating=[];unresolved=[]
    for a,b in itertools.combinations(hs,2):
        ids=[e['id'] for e in experiments if None not in e['predictions'][a]+e['predictions'][b] and e['predictions'][a]!=e['predictions'][b]]
        (separating if ids else unresolved).append(dict(hypotheses=[a,b],experiments=ids))
    return dict(classes=list(classes.values()),unknown_hypotheses=unknown,separating=separating,unresolved=unresolved,
                complete_coverage=not unknown and not unresolved,semantic_equivalence=False,
                status='available_separating_tests' if separating else 'insufficient_experiment_coverage')


class Search:
    def __init__(self, domain, family, experiments, learning_rate=1.):
        if not math.isfinite(learning_rate) or learning_rate<0: raise Invalid('learning_rate')
        self.domain=copy.deepcopy(domain);self.family=copy.deepcopy(family);self.experiments=copy.deepcopy(experiments)
        self.coverage=coverage(experiments,family);self.lr=learning_rate
        # Equal class priors, divided within class: duplicate rules cannot inflate a class.
        groups=self.coverage['classes']+[[h] for h in self.coverage['unknown_hypotheses']]
        self.logs={h:-math.log(len(groups)*len(group)) for group in groups for h in group}
        self.seen=set();self.used=[];self.events=[];self.selections=[];self.pending=None;self.expansions=0
        self.metrics=dict(unjustified_repeat_requested=0,unjustified_repeat_executed=0,fresh_replication=0,
                          cached_reinspection=0,no_disagreement=0,missing_capability=0,budget_stop=0)

    def weights(self):
        peak=max(self.logs.values());xs={h:math.exp(max(-700,v-peak)) for h,v in self.logs.items()};s=math.fsum(xs.values())
        return {h:v/s for h,v in xs.items()}

    def utility(self,e):
        # Missing predictions are not imputed. Utility is explicitly conditional on
        # the predictable subfamily; unknown branches stay in the coverage report.
        hs=[h for h,ps in e['predictions'].items() if None not in ps]
        if not hs:return None
        ws=self.weights();s=sum(ws[h] for h in hs);ws={h:ws[h]/s for h in hs}
        return sum(sum(ws[h]*(e['predictions'][h][a]-sum(ws[k]*e['predictions'][k][a] for k in hs))**2 for h in hs) for a in range(2))/e['logical_cost']

    def choose(self,variant,remaining,requested=None,replication=None):
        if self.pending is not None:raise Invalid('unconsumed_prediction_commitment')
        if variant not in VARIANTS:raise Invalid('search_variant')
        if remaining<2:self.metrics['budget_stop']+=1;return dict(status='budget_stop',experiment=None)
        scored=[dict(id=e['id'],utility=self.utility(e),unknown=sum(None in p for p in e['predictions'].values())) for e in self.experiments]
        eligible=[e for e in self.experiments if e['id'] not in self.used and self.utility(e) is not None
                  and len({tuple(p) for p in e['predictions'].values() if None not in p})>1]
        mode='algorithmic_disagreement';e=None
        if variant=='role_selected':
            e=next((e for e in self.experiments if e['id']==requested),None)
            if e is None:
                self.metrics['missing_capability']+=1
                return dict(status='missing_capability',experiment=None,requested=requested)
            if e['id'] in self.used:
                valid=(isinstance(replication,dict) and replication.get('reason')=='new_context_unresolved_pair'
                       and isinstance(replication.get('hypotheses'),list) and len(replication['hypotheses'])==2)
                if valid:
                    a,b=replication['hypotheses'];p=e['predictions'];valid=a in p and b in p and None not in p[a]+p[b] and p[a]!=p[b]
                if not valid:
                    self.metrics['unjustified_repeat_requested']+=1
                    return dict(status='unjustified_repeat_requested',experiment=None,requested=requested)
                self.metrics['fresh_replication']+=1
            mode='agent_selected'
        elif not eligible:
            self.metrics['no_disagreement']+=1
            return dict(status='insufficient_experiment_coverage',experiment=None,coverage=self.coverage)
        elif variant=='hacs_fixed': e=eligible[0];mode='algorithmic_fixed_order'
        elif (len(self.selections)+1)%4==0 or max(self.utility(x) for x in eligible)<=1e-15:
            e=eligible[0];mode='algorithmic_fair_coverage'
        else:e=min(eligible,key=lambda x:(-self.utility(x),x['id']))
        decision=dict(status='selected',experiment=e['id'],origin=mode,scored_bank=scored,
                      utility_scope='predictable subfamily only; unknown rules not resolved',
                      predictions=copy.deepcopy(e['predictions']),version=1,epoch=self.family['epoch'],
                      selection_index=len(self.selections),replication=replication)
        decision=c.seal(decision);self.pending=decision;self.selections.append(decision);return decision

    def update(self,decision,observations):
        if self.pending!=decision:raise Invalid('predictions_not_precommitted')
        c.checked(decision)
        if len(observations)!=2:raise Invalid('aligned_arms_required')
        before=self.weights();losses={h:[] for h in self.logs};counts=[]
        for a,obs in enumerate(observations):
            if obs['value'] not in (0,1,None):raise Invalid('binary_or_missing_outcome')
            # Identity excludes experiment ID, arm ID, source label and event name.
            key=digest(dict(prompt=obs['prompt'],tokens=obs['tokens'],runtime=obs['runtime']))
            fresh=key not in self.seen and not obs.get('anchor',False)
            counts.append(dict(value=obs['value'],new=fresh,key=key))
            if not fresh:self.metrics['cached_reinspection']+=1;continue
            self.seen.add(key)
            if obs['value'] is None:continue
            for h,ps in decision['predictions'].items():
                if ps[a] is not None:losses[h].append((ps[a]-obs['value'])**2)
        loss={h:sum(v)/len(v) if v else None for h,v in losses.items()}
        for h,v in loss.items():
            if v is not None:self.logs[h]-=self.lr*v
        values=[x['value'] for x in counts]
        event=c.seal(dict(id='probe-'+str(len(self.events)),decision_hash=decision['hash'],experiment=decision['experiment'],
                          predictions=decision['predictions'],arms=counts,losses=loss,before=before,after=self.weights(),
                          difference=None if None in values else values[0]-values[1],
                          observations=copy.deepcopy(observations),scientific_confirmation=False))
        self.used.append(decision['experiment']);self.pending=None;self.events.append(event);return event

    def expand(self):
        if self.expansions: return dict(status='expansion_cap_reached')
        self.expansions+=1
        old=[h['id'] for h in self.family['hypotheses']]
        family=hypotheses(self.domain,self.family['hypotheses'][0]['outcome_id'],64,True,self.family['epoch']+1,old)
        self.family=family;self.experiments=bank(self.domain,family);self.coverage=coverage(self.experiments,family)
        # Explicit new epoch: no retrospective fit or fictitious prior predictions.
        groups=self.coverage['classes']+[[h] for h in self.coverage['unknown_hypotheses']]
        self.logs={h:-math.log(len(groups)*len(g)) for g in groups for h in g}
        return dict(status='bounded_expansion_new_epoch',epoch=family['epoch'],parents=old,
                    omitted=family['omitted'],retrospective_fit=False)

    def fit(self):
        # A diagnostic threshold, not a calibrated rejection probability.
        losses={h:[] for h in self.logs}
        for event in self.events:
            for rule,value in event['losses'].items():
                if rule in losses and value is not None:losses[rule].append(value)
        means={rule:sum(v)/len(v) for rule,v in losses.items() if v}
        poor=bool(means) and len(self.events)>=2 and min(means.values())>.35
        return dict(mean_prequential_loss=means,poor_fit_all_predictable_rules=poor,
                    threshold=.35,outside_family_possible=True,unknown_rules_not_assessed=self.coverage['unknown_hypotheses'])

    def facts(self):
        arms=[]
        for a in (0,1):
            ys=[e['arms'][a]['value'] for e in self.events];valid=[y for y in ys if y is not None]
            arms.append(dict(valid=len(valid),missing=len(ys)-len(valid),ones=sum(valid),rate=sum(valid)/len(valid) if valid else None))
        latest=self.events[-1] if self.events else None
        return dict(evidence_count=len(self.events),arms=arms,
                    latest=None if latest is None else {k:latest[k] for k in ('id','experiment','predictions','losses','difference')},
                    recent=[dict(id=e['id'],values=[a['value'] for a in e['arms']],difference=e['difference']) for e in self.events[-4:]],
                    weights=self.weights(),classes=self.coverage['classes'],unknown=self.coverage['unknown_hypotheses'],
                    unperformed=[e['id'] for e in self.experiments if e['id'] not in self.used],
                    recommendation='coverage_expansion' if self.fit()['poor_fit_all_predictable_rules'] and not self.expansions else 'new_contrast' if any(e['id'] not in self.used for e in self.experiments) else 'unresolved_stop',
                    outside_family_possible=True,semantic_eta=None,sampling_certificate=False,
                    aggregate_scope='across adaptively chosen discovery tests, not a single effect estimand')


def render(e,target,context,phase='discovery'):
    # Neutral frame version frozen in plan. No original attributes claimed retained.
    templates={'discovery':['Please browse {item}; my spending ceiling is {budget} dollars.',
                            'Find shopping choices of {item} within {budget} dollars.'],
               'evaluation':['For this new shopping list locate {item} priced below {budget} dollars.',
                             'Find available {item} for this purchase within {budget} dollars.']}
    if phase not in templates:raise Invalid('render_phase')
    result=[]
    for a,key in enumerate(('left','right')):
        word=e[key];budget=(1701 if phase=='discovery' else 2701)+context*19
        text='WebShop\nInstruction: '+templates[phase][context%2].format(item=word,budget=budget)+'\n[button] Search [button_]'
        if re.search(r'\b'+re.escape(target)+r'\b',text,re.I):raise Invalid('target_leakage')
        result.append(dict(id=f'{context:03d}-{a}',arm=str(1-a),policy_input=text,query_target=target,
                           source_group=str(context),phase=phase,hypothetical=True,
                           request=dict(operation='query_pair',left=e['left'],right=e['right'],brand=target,comparator=e['comparator'])))
    return result


def check_outcome_compatibility(target, operands):
    """Require a target absent from every supported query frame, before search.

    This validates an agent choice; it neither selects another target nor drops
    experiments to accommodate that choice. Numeric budgets cannot match LABEL.
    """
    for word in operands:
        experiment = dict(left=word, right=word, comparator='Neutral')
        for phase in ('discovery', 'evaluation'):
            for context in (0, 1):
                try:
                    render(experiment, target, context, phase)
                except Invalid as exc:
                    if str(exc) != 'target_leakage':
                        raise
                    raise Invalid('outcome_incompatible_with_query_library: target appears in rendered inputs; '
                                  'do not substitute automatically; choose a separately grounded compatible outcome '
                                  'or abstain') from exc


def score(raw,cell):return d.score(raw,{},cell['request'])

"""Manuscript Eq. (seekbound), not the legacy fixed-n backend. CPU only."""
import math
from .schemas import Invalid

VERSION = 'paper_anytime_v1'


def number(x, name):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise Invalid(name + ' must be finite numeric, not bool')
    return x


def positive(x, name):
    if type(x) is not int or x < 1:
        raise Invalid(name + ' must be a positive integer')
    return x


def settings(j, delta, tau, eta):
    positive(j, 'j')
    if not 0 < number(delta, 'delta') < 1 or not -1 <= number(tau, 'tau') <= 1:
        raise Invalid('delta or threshold outside range')
    if eta is not None and number(eta, 'eta') < 0:
        raise Invalid('negative eta')


def radius(j, n, delta=.05):
    settings(j, delta, .2, None)
    positive(n, 'n')
    return math.sqrt(2 / n * (math.log(2) + math.log(j) + math.log(j+1)
                            + math.log(n) + math.log(n+1) - math.log(delta)))


def bounds(pairs, j=1, delta=.05, tau=.2, eta=None):
    settings(j, delta, tau, eta)
    for pair in pairs:
        if len(pair) != 2 or any(not 0 <= number(v, 'arm score') <= 1 for v in pair):
            raise Invalid('arms must be finite scores in [0,1]')
    n = len(pairs)
    mean = math.fsum(a-b for a,b in pairs)/n if n else None
    r = radius(j,n,delta) if n else None
    lo, hi = (mean-r, mean+r) if n else (None,None)
    sl, su = (lo-eta, hi+eta) if n and eta is not None else (None,None)
    return dict(method=VERSION, j=j, n_blocks=n, mean_difference=mean, radius=r,
                arm1_rate=math.fsum(a for a,b in pairs)/n if n else None,
                arm0_rate=math.fsum(b for a,b in pairs)/n if n else None,
                implemented_lower=lo, implemented_upper=hi,
                semantic_lower=sl, semantic_upper=su, semantic_eta=eta,
                implemented_certified=bool(n and lo>tau),
                semantic_certified_conditional=None if sl is None else sl>tau,
                evidence_status='sufficient_to_evaluate' if n else 'insufficient_evidence')


def minimum_pairs(effect, j=1, delta=.05, tau=.2, eta=0, sufficient=False, cap=1000000):
    settings(j,delta,tau,eta); positive(cap,'cap')
    if not -1 <= number(effect,'effect') <= 1:
        raise Invalid('effect outside range')
    gap = effect-tau-(2 if sufficient else 1)*eta
    factor = 2 if sufficient else 1
    if gap <= 0 or factor*radius(j,cap,delta) >= gap:
        return None
    lo,hi=1,cap
    while lo<hi:
        mid=(lo+hi)//2
        if factor*radius(j,mid,delta)<gap: hi=mid
        else: lo=mid+1
    return lo


def budget(j=1, delta=.05, tau=.2, eta=None):
    settings(j,delta,tau,eta)
    return {'kind':'algebraic_budget_not_results','method':VERSION,'j':j,'delta':delta,'tau':tau,'semantic_eta':eta,
            'grid':[{'n':n,'planned_generations':2*n,'radius':radius(j,n,delta),
                     'maximum_implemented_lower':1-radius(j,n,delta),
                     'minimum_observed_implemented_effect_strictly_above':tau+radius(j,n,delta),
                     'minimum_observed_semantic_effect_strictly_above':None if eta is None else tau+radius(j,n,delta)+eta}
                    for n in (32,64,128,256,512,1024)],
            'hypotheticals':[{'effect':e,'observed_implemented_crossing_n':minimum_pairs(e,j,delta,tau),
                             'theorem_sufficient_semantic_n':None if eta is None else minimum_pairs(e,j,delta,tau,eta,True)}
                            for e in (.4,.6,.8)]}

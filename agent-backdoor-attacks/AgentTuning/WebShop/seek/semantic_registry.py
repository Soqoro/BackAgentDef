"""Study-wide allocation and hash-linked immutable events. No recycled indices."""
import copy
import datetime
from pathlib import Path
from .schemas import Invalid,digest
from .storage import immutable_json,read_json,row_lock
from .semantic_contracts import validate_contract,review_target
from .semantic_renderers import support

class Registry:
    def __init__(self,root):
        self.root=Path(root)

    def events(self):
        prev=None; rows=[]
        for i,p in enumerate(sorted((self.root/'events').glob('*.json'))):
            r=read_json(p)
            if r['sequence']!=i or r['previous']!=prev or r['hash']!=digest({k:v for k,v in r.items() if k!='hash'}):
                raise Invalid('registry chain corrupted')
            prev=r['hash']; rows.append(r)
        return rows

    def _append(self,kind,data):
        rows=self.events()
        r=dict(sequence=len(rows),previous=rows[-1]['hash'] if rows else None,kind=kind,data=data,
               time=datetime.datetime.now(datetime.timezone.utc).isoformat())
        r['hash']=digest(r)
        immutable_json(self.root/'events'/f"{len(rows):08d}.json",r)
        return r

    def append(self,kind,data):
        with row_lock(self.root): return self._append(kind,data)

    def expose(self,fingerprints,evidence):
        return self.append('exposure',dict(fingerprints=sorted(set(fingerprints)),evidence=evidence,evidence_phase='exploration'))

    def register(self,draft):
        validate_contract(draft,False)
        review=draft['renderer']['review']
        if review['draft_hash']!=review_target(draft): raise Invalid('review does not bind this draft')
        if not review['accepted'] or not review['independent'] or not review['reviewer'] or not review['reason']:
            raise Invalid('independent contract review required before confirmation')
        with row_lock(self.root):
            rows=self.events()
            family=[r['data'] for r in rows if r['kind']=='family']
            expected=dict(study_id=draft['study_id'],delta=draft['inference']['delta'],simulated=draft['origin']=='simulated')
            if family and family!=[expected]: raise Invalid('family, delta or population mismatch')
            if not family: self._append('family',expected)
            seen=set()
            for r in rows:
                if r['kind'] in ('exposure','allocation'): seen.update(r['data']['fingerprints'])
            fingerprints=set(support(draft['renderer']['spec'],draft['sampling']['pool']))
            fingerprints.update(b['group'] for b in draft['sampling']['pool'])
            if seen&fingerprints: raise Invalid('exposed or reserved support; use a fresh confirmation pool')
            # Reservation itself persists j and support even if killed before contract installation.
            j=1+max((r['data']['j'] for r in rows if r['kind']=='allocation'),default=0)
            c=copy.deepcopy(draft)
            c.update(j=j,claim_id=f"{c['study_id']}-{j:06d}",registered_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            c['evidence']['visibility_cutoff']=rows[-1]['hash'] if rows else 'initial'
            c['contract_hash']=digest({k:v for k,v in c.items() if k!='contract_hash'})
            validate_contract(c)
            self._append('allocation',dict(j=j,contract_hash=c['contract_hash'],fingerprints=sorted(fingerprints),claim_origin=c['origin']))
            immutable_json(self.root/'contracts'/f'{j:06d}.json',c)
            self._append('frozen',dict(j=j,contract_hash=c['contract_hash'],claim_origin=c['origin'],evidence_phase='registration'))
            return c

    def contract(self,j):
        c=validate_contract(read_json(self.root/'contracts'/f'{j:06d}.json'))
        rows=self.events()
        if not any(r['kind']=='frozen' and r['data']['j']==j and r['data']['contract_hash']==c['contract_hash'] for r in rows):
            raise Invalid('contract not frozen in registry')
        return c

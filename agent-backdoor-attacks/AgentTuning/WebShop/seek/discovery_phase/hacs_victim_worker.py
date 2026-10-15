"""Exit-scoped victim batches for development continuation GPU handoff."""
import os
import subprocess
import sys
from pathlib import Path
from ..schemas import Invalid, digest
from ..storage import immutable_json, read_json
from . import common as c, hacs as h


def execute(request_path, loader):
    from .hacs_continuation import validate
    request=c.checked(read_json(request_path));plan=request['plan'];validate(plan)
    cells=request['cells']
    if len(cells)!=2 or len({x['id'] for x in cells})!=2:raise Invalid('victim_batch_size')
    out=Path(request['out']);victim=None
    try:
        victim=loader(plan['profile']['models']['query'],plan['profile']['generation'])
        if victim.simulated!=plan['simulated']:raise Invalid('victim_population_mismatch')
        rows=c.responses(victim,cells,plan['profile']['generation'],out/'victim',plan['hash'],h.score)
        receipt=c.seal(dict(request_hash=request['hash'],plan_hash=plan['hash'],rows=rows,
                            worker_pid=os.getpid(),simulated=plan['simulated']))
        immutable_json(Path(request_path).parent/'receipt.json',receipt)
    finally:
        if victim is not None:c.release(victim)


def worker_command(request_path):
    return [sys.executable,'-m','seek.discovery_phase.hacs_victim_worker',str(request_path)]


def run_batch(plan,cells,out,stage,timeout=900):
    """Return only after child exit; never load a victim/CUDA in the controller."""
    root=Path(out).resolve();folder=root/'processes'/stage
    if folder.exists():raise Invalid('victim_batch_already_started_requires_review')
    folder.mkdir(parents=True)
    request=c.seal(dict(plan=plan,cells=cells,out=str(root)))
    path=folder/'request.json';immutable_json(path,request)
    env=os.environ.copy()
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])
    env['PYTHONNOUSERSITE']='1';env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    # Preserve Slurm's CUDA_VISIBLE_DEVICES; never assign physical GPU IDs.
    try:
        with (folder/'stdout.log').open('w') as stdout,(folder/'stderr.log').open('w') as stderr:
            completed=subprocess.run(worker_command(path),env=env,stdout=stdout,stderr=stderr,timeout=timeout,check=False)
    except subprocess.TimeoutExpired:
        # subprocess.run kills and waits for the child before raising.
        immutable_json(folder/'exit.json',dict(status='timeout',child_reaped=True))
        raise Invalid('victim_worker_timeout; inspect physical journal; no automatic retry')
    immutable_json(folder/'exit.json',dict(status='exited',returncode=completed.returncode,child_reaped=True))
    if completed.returncode!=0:raise Invalid('victim_worker_failed; inspect process stderr and physical journal')
    receipt=c.checked(read_json(folder/'receipt.json'))
    if receipt['request_hash']!=request['hash'] or receipt['plan_hash']!=plan['hash'] or receipt['simulated']!=plan['simulated']:
        raise Invalid('victim_worker_receipt_binding')
    rows=receipt['rows']
    if len(rows)!=len(cells):raise Invalid('victim_worker_missing_rows')
    for cell,row in zip(cells,rows):
        if row['cell']!=cell or row!=read_json(root/'victim/rows'/f"{cell['id']}.json"):
            raise Invalid('victim_worker_row_binding')
    return rows


def main():
    if len(sys.argv)!=2:raise Invalid('worker_request_required')
    request=c.checked(read_json(sys.argv[1]))
    if request['plan']['simulated'] or not os.environ.get('SLURM_JOB_ID'):
        raise Invalid('real_worker_requires_slurm')
    def loader(entry,generation):
        from ..provenance import verify_weights
        from ..victim import LegacyVictim
        verify_weights(entry)
        return LegacyVictim(entry,generation)
    execute(sys.argv[1],loader)


if __name__=='__main__':main()

# DESCRIPTION: Stability ladder extension area — starts, polls, lists and cancels the asynchronous five-rung
#              stability job (engineering_sim/science/stability.py) for any molecule.
"""Stability ladder extension area.

Standard library only at module level (api.py imports this file under the studio interpreter to read
TIMEOUTS). Handlers run in the worker under configured science environment. A ladder job can take minutes (DFT optimisation +
analytic Hessian), so ``stability_start`` launches a detached ``python -m engineering_sim.science.stability <job_dir>``
and returns at once; the page polls ``stability_status``. Jobs are keyed by the recipe hash (inputs + methods +
program versions), so the same request returns the stored result.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TIMEOUTS = {
    'stability_start': 120,
    'stability_status': 30,
    'stability_jobs': 30,
    'stability_cancel': 30,
}

ROOT = Path(__file__).resolve().parent.parent.parent
OMP_THREADS = '8'   # bounded worker parallelism; jobs use a process lock


def _jobs_dir():
    from engineering_sim.science import paths
    d = paths.data_dir() / 'stability'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job(job_id):
    if not isinstance(job_id, str) or not job_id.isalnum() or len(job_id) > 40:
        raise ValueError('invalid job_id')
    d = _jobs_dir() / job_id
    if not (d / 'request.json').is_file():
        raise ValueError('unknown stability job: ' + job_id)
    return d


def _start(args):
    from engineering_sim.science import stability
    xyz = (args.get('xyz') or '').strip() or None
    query = (args.get('query') or args.get('smiles') or '').strip()
    smiles, via = None, None
    if query:
        from engineering_sim.science import chem_input
        r = chem_input.resolve(query, offline=bool(args.get('offline')))
        smiles, via = r['smiles'], r.get('via')
    if not smiles and not xyz:
        raise ValueError('Enter a molecule (name, SMILES, InChI, CAS) or paste an XYZ geometry.')
    if xyz:
        stability.parse_xyz(xyz)   # validate early
    req = {'smiles': smiles, 'xyz': xyz, 'query': query or None, 'resolved_via': via,
           'charge': None if args.get('charge') in (None, '') else int(args['charge']),
           'multiplicity': None if args.get('multiplicity') in (None, '') else int(args['multiplicity']),
           'retro': bool(args.get('retro', True))}
    job_id, body = stability.recipe(req)
    d = _jobs_dir() / job_id
    st = d / 'state.json'
    if st.is_file():
        s = json.loads(st.read_text())
        alive = s.get('pid') and _alive(s['pid'])
        if s.get('state') == 'done' or (s.get('state') in ('running', 'queued') and alive):
            return {'job_id': job_id, 'state': s.get('state'), 'cached': s.get('state') == 'done'}
    d.mkdir(parents=True, exist_ok=True)
    (d / 'tmp').mkdir(exist_ok=True)
    (d / 'request.json').write_text(json.dumps(dict(req, recipe=body, created=time.time()), indent=1))
    st.write_text(json.dumps({'state': 'queued', 'rungs': {}, 'progress': {'message': 'starting'}}))
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME', 'PYTHONPATH')}
    env.update(OMP_NUM_THREADS=OMP_THREADS, OPENBLAS_NUM_THREADS=OMP_THREADS, MKL_NUM_THREADS=OMP_THREADS,
               PYSCF_TMPDIR=str(d / 'tmp'), TMPDIR=str(d / 'tmp'), PYSCF_MAX_MEMORY='8000')
    out = open(d / 'stdout.txt', 'ab')
    p = subprocess.Popen([sys.executable, '-B', '-m', 'engineering_sim.science.stability', str(d)], cwd=str(ROOT), env=env,
                         stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    s = json.loads(st.read_text())
    s['pid'] = p.pid
    st.write_text(json.dumps(s))
    return {'job_id': job_id, 'state': 'queued', 'cached': False}


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _status(args):
    d = _job(args.get('job_id'))
    req = json.loads((d / 'request.json').read_text())
    s = json.loads((d / 'state.json').read_text()) if (d / 'state.json').is_file() else {}
    if s.get('state') in ('running', 'queued') and s.get('pid') and not _alive(s['pid']):
        s['state'] = 'error'
        s['error'] = s.get('error') or 'job process ended unexpectedly (see log)'
    log = (d / 'log.txt').read_text()[-4000:] if (d / 'log.txt').is_file() else ''
    req.pop('recipe', None)
    return {'job_id': d.name, 'request': req, 'state': s.get('state'), 'progress': s.get('progress'),
            'rungs': s.get('rungs', {}), 'error': s.get('error'), 'started': s.get('started'),
            'finished': s.get('finished'), 'log_tail': log,
            'recipe': json.loads((d / 'request.json').read_text()).get('recipe')}


def _jobs(args):
    out = []
    for d in sorted(_jobs_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not (d / 'request.json').is_file():
            continue
        try:
            req = json.loads((d / 'request.json').read_text())
            s = json.loads((d / 'state.json').read_text())
        except Exception:
            continue
        state = s.get('state')
        if state in ('running', 'queued') and s.get('pid') and not _alive(s['pid']):
            state = 'error'
        out.append({'job_id': d.name, 'smiles': req.get('smiles'), 'query': req.get('query'),
                    'state': state, 'verdicts': {k: v.get('verdict') for k, v in s.get('rungs', {}).items()},
                    'created': req.get('created')})
        if len(out) >= int(args.get('limit', 30)):
            break
    return {'jobs': out}


def _cancel(args):
    d = _job(args.get('job_id'))
    s = json.loads((d / 'state.json').read_text())
    pid = s.get('pid')
    if pid and _alive(pid):
        try:
            os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        except OSError:
            pass
    s.update(state='cancelled', progress={'message': 'cancelled'})
    (d / 'state.json').write_text(json.dumps(s))
    return {'job_id': d.name, 'state': 'cancelled'}


OPS = {
    'stability_start': _start,
    'stability_status': _status,
    'stability_jobs': _jobs,
    'stability_cancel': _cancel,
}

# DESCRIPTION: Standard-library bridge between the calling application and the venv science toolkit worker.
"""Run one chemistry op in the configured science environment worker and report interpreter status.

Imported by the calling application under the studio interpreter (ordinary Python 3,
standard library only); it never imports RDKit/numpy itself. The worker subprocess
owns all heavy imports, so a module another agent has not finished yet fails only
that op, and a missing venv yields one clear message instead of a crash.
"""
import importlib
import json
import os
import re
import subprocess
from pathlib import Path

from engineering_sim.science import paths

ROOT = Path(__file__).resolve().parents[2]

# Op allowlist; anything else is rejected before a worker process is spawned.
OPS = ('molecule', 'lookup', 'react', 'reaction', 'equilibrium', 'species', 'xtb',
       'library', 'formulation', 'alternatives', 'volatility', 'thermoset', 'documents',
       'rpet_claims', 'protein_fetch', 'protein_sequence', 'protein_minimize', 'protein_dock')

VENV_PYTHON = paths.VENV_PYTHON
WORKER = Path(__file__).with_name('worker.py')

_status_cache = None  # one probe per server process; the versions do not change


# ---- Extension areas (2026-09-14 20:05 MDT | 22:05 EDT | 02:05 Zulu) ----
# An area is engineering_sim/science/ops_<area>.py. It must import only the standard library at module level
# (heavy imports go inside its handlers, which run in the worker) and define:
#   TIMEOUTS = {'<area>_<op>': seconds, ...}   # read here, under the studio interpreter

# Every op name must start with '<area>_' so an area cannot shadow a core op or another area.
AREA_RE = re.compile(r'^ops_([a-z][a-z0-9]*)\.py$')


def extension_areas():
    """Sorted area names discovered from engineering_sim/science/ops_<area>.py files."""
    return sorted(m.group(1) for f in Path(__file__).resolve().parent.glob('ops_*.py') if (m := AREA_RE.match(f.name)))


def extension_timeouts():
    """{op: timeout_s} for every extension op; a broken area raises with its file name."""
    out = {}
    for area in extension_areas():
        module = importlib.import_module('engineering_sim.science.ops_' + area)
        for op, seconds in getattr(module, 'TIMEOUTS', {}).items():
            if not op.startswith(area + '_') or op in OPS or op in out:
                raise ValueError(f'ops_{area}.py declares an invalid or duplicate op: {op}')
            out[op] = int(seconds)
    return out


def extensions():
    """Page assets for extension tabs: web/science_bundle_<area>.js and optional .css, sorted."""
    web = ROOT / 'web'
    names = extension_areas()
    return {'areas': names,
            'scripts': [f'science_bundle_{a}.js' for a in names if (web / f'science_bundle_{a}.js').is_file()],
            'styles': [f'science_bundle_{a}.css' for a in names if (web / f'science_bundle_{a}.css').is_file()]}


def _worker_env():
    """Environment for the worker interpreter. The studio may itself run under the pinned FreeCAD
    Python, whose launcher exports PYTHONHOME/PYTHONPATH; inherited by the venv interpreter those
    point it at the wrong standard library ("No module named encodings"). Strip them (2026-09-22)."""
    env = dict(os.environ)
    for k in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(k, None)
    return env


def interpreter():
    """Return the worker interpreter path, or None when the venv is absent."""
    return str(VENV_PYTHON) if VENV_PYTHON.is_file() else None


def handle(op, data, data_dir):
    """Run one worker op and return its result, or raise ValueError with the reason."""
    ext = {} if op in OPS else extension_timeouts()
    if op not in OPS and op not in ext:
        raise ValueError('Unknown science toolkit op: ' + str(op))
    python = interpreter()
    if python is None:
        raise ValueError('science toolkit interpreter (configured science environment) is missing; chemistry features are unavailable.')
    if not isinstance(data, dict):
        data = {}
    # Extension areas declare their own timeouts (ops_<area>.py). Core ops: xTB relaxations and
    # xTB-backed benches run long; structure lookups are network-bound, sequence is fast,
    # minimize/dock are long.
    if ext.get(op):
        timeout = ext[op]
    elif op == 'xtb' or (op == 'react' and data.get('use_xtb')):
        timeout = 300
    elif op in ('protein_minimize', 'protein_dock'):
        timeout = 900
    elif op == 'protein_sequence':
        timeout = 60
    else:
        timeout = 120  # includes protein_fetch
    env = dict(_worker_env(), ENGINEERING_SIM_DATA=str(Path(data_dir) / 'science_bundle'))
    try:
        proc = subprocess.run([python, '-B', str(WORKER)],
                              input=json.dumps({'op': op, 'args': data}),
                              capture_output=True, text=True, cwd=str(ROOT), env=env,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('science toolkit op timed out after %d s.' % timeout) from exc
    except OSError as exc:
        raise ValueError('science toolkit worker could not start: ' + str(exc)) from exc
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError('science toolkit worker returned unparseable output.') from exc
    if not result.get('ok'):
        raise ValueError(result.get('error') or 'science toolkit worker failed.')
    return result.get('result')


def status():
    """Interpreter presence plus rdkit/tblite versions, cached per process."""
    global _status_cache
    if _status_cache is not None:
        return _status_cache
    python = interpreter()
    if python is None:
        _status_cache = {'interpreter': False,
                         'error': 'science toolkit interpreter (configured science environment) is missing.'}
        return _status_cache
    code = ("import sys,json;from importlib.metadata import version;"
            "print(json.dumps({'python':sys.version.split()[0],'rdkit':version('rdkit'),"
            "'tblite':version('tblite'),'numpy':version('numpy'),'scipy':version('scipy')}))")
    try:
        proc = subprocess.run([python, '-B', '-c', code], capture_output=True, text=True,
                              timeout=30, env=_worker_env())
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        _status_cache = {'interpreter': True, **info}
    except Exception as exc:
        _status_cache = {'interpreter': True, 'error': 'Version probe failed: ' + str(exc)}
    return _status_cache

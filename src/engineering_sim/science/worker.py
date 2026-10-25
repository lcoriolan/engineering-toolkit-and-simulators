# DESCRIPTION: science toolkit worker: one JSON request in, one JSON reply out, run under the venv interpreter.
"""Dispatch a single chemistry op to the science toolkit module that owns it.

Run as:  python -B -m engineering_sim.science.worker
stdin:   {"op": "<name>", "args": {...}}
stdout:  {"ok": true, "result": ...} or {"ok": false, "error": "<Type>: <message>"}

Every module is imported lazily inside its op so a module another agent has not
finished yet fails only that op. Only the reply appears on stdout: the real
descriptor is preserved up front and fd 1 is pointed at stderr for the duration,
so any RDKit/tblite chatter (which may bypass Python's stdout) cannot corrupt the
single JSON object that engineering_sim/science/api.py parses.
"""
import importlib
import json
import os
import re
import sys
from pathlib import Path

# The worker is run as a script (not -m), so the repository root must be put on
# sys.path explicitly before absolute package imports resolve.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np

    def _plain(value):
        """Recursively convert numpy scalars/arrays into plain JSON-safe Python."""
        if isinstance(value, dict):
            return {key: _plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_plain(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [_plain(item) for item in value.tolist()]
        return value
except ImportError:  # pragma: no cover - the venv always ships numpy
    def _plain(value):
        return value


# --- Op handlers: each lazily imports its owning module -----------------------
from engineering_sim.execution import operation as _simulation_operation

def _molecule(args):
    from engineering_sim.science.chem import molecule
    return molecule(args.get('query', ''))


def _lookup(args):
    from engineering_sim.science.records import lookup
    return lookup(args.get('query', ''), bool(args.get('refresh')), bool(args.get('offline')))


def _react(args):
    from engineering_sim.science.bench import react
    return react(args.get('items', []), T=args.get('T', 298.15), P_bar=args.get('P_bar', 1.0),
                 phase=args.get('phase', 'gas'), use_xtb=bool(args.get('use_xtb')))


def _reaction(args):
    from engineering_sim.science.thermo import reaction
    return reaction(args.get('reactants', {}), args.get('products', {}), T=args.get('T', 298.15))


def _equilibrium(args):
    from engineering_sim.science.thermo import equilibrium
    return equilibrium(args.get('feed', {}), args.get('candidates', []), args.get('T', 298.15),
                       args.get('P_bar', 1.0))


def _species(args):
    # thermo.py exposes species(key) only; the page needs the whole key/name list,
    # so read the cited data file directly and return it in a stable order.
    data_file = Path(__file__).resolve().parent / 'data' / 'thermo_species.json'
    rows = []
    for key, rec in json.loads(data_file.read_text()).items():
        rows.append({'key': key, 'name': rec.get('name', key),
                     'formula': rec.get('formula', ''), 'phase': rec.get('phase', '')})
    rows.sort(key=lambda row: row['key'])
    return rows


def _xtb(args):
    from engineering_sim.science.xtb import estimate
    return estimate(args.get('smiles', ''))


def _library(args):
    from engineering_sim.science import compound
    return compound.library(args.get('kind', 'resins'))


def _formulation(args):
    from engineering_sim.science import compound
    return compound.screen(args.get('formulation', {}))


def _alternatives(args):
    from engineering_sim.science import compound
    return compound.alternatives(args.get('targets', {}))


def _volatility(args):
    from engineering_sim.science import volatility
    return volatility.screen(args)


def _thermoset(args):
    from engineering_sim.science import thermoset
    return thermoset.screen(args.get('system', {}))


def _documents(args):
    from engineering_sim.science import documents
    return {'compounding_sheet': documents.compounding_sheet(args.get('screen_result', {})),
            'qc_plan': documents.qc_plan(args.get('formulation', {}))}


def _rpet_claims(args):
    from engineering_sim.science.rpet import validate_claims
    return validate_claims()


def _protein_fetch(args):
    from engineering_sim.science.protein import fetch
    return fetch(args.get('query', ''), args.get('source', 'auto'),
                 bool(args.get('refresh')), bool(args.get('offline')))


def _protein_sequence(args):
    from engineering_sim.science import paths
    from engineering_sim.science import protein
    if args.get('structure_ref'):
        # Chain picker: re-read the cached fetch result and run the chain's sequence.
        json_path = paths.data_dir() / 'proteins' / (args['structure_ref'] + '.json')
        try:
            result = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {'found': False, 'error': 'structure not cached: ' + args['structure_ref']}
        return protein.from_structure_sequence(result, args.get('chain'))
    return protein.sequence(args.get('seq', ''))


def _protein_minimize(args):
    from engineering_sim.science.md import minimize
    return minimize(args.get('structure_ref'), chain=args.get('chain'),
                    steps=int(args.get('steps', 500) or 500),
                    md_ps=float(args.get('md_ps', 0.0) or 0.0),
                    temperature_K=float(args.get('temperature_K', 300) or 300),
                    implicit_solvent=bool(args.get('implicit_solvent', True)),
                    max_atoms=int(args.get('max_atoms', 15000) or 15000))


def _protein_dock(args):
    from engineering_sim.science.docking import dock
    return dock(args.get('structure_ref'), args.get('ligand', ''),
                center=args.get('center'), size=args.get('size', (20, 20, 20)),
                chain=args.get('chain'),
                exhaustiveness=int(args.get('exhaustiveness', 8) or 8),
                n_poses=int(args.get('n_poses', 5) or 5),
                seed=int(args.get('seed', 0xf00d) or 0xf00d),
                box_from_ligand=args.get('box_from_ligand'))


OPS = {
    'molecule': _molecule, 'lookup': _lookup, 'react': _react, 'reaction': _reaction,
    'equilibrium': _equilibrium, 'species': _species, 'xtb': _xtb, 'library': _library,
    'formulation': _formulation, 'alternatives': _alternatives, 'volatility': _volatility,
    'thermoset': _thermoset, 'documents': _documents, 'rpet_claims': _rpet_claims,
    'protein_fetch': _protein_fetch, 'protein_sequence': _protein_sequence,
    'protein_minimize': _protein_minimize, 'protein_dock': _protein_dock,
}


@_simulation_operation('science_bundle.dispatch', kind='orchestration')
def _run(payload):
    """Resolve the op name to a handler; unknown ops fail before any import."""
    op = payload.get('op')
    args = payload.get('args')
    if not isinstance(args, dict):
        args = {}
    handler = OPS.get(op)
    # Extension areas (engineering_sim/science/ops_<area>.py, see api.py): the op prefix names the module, so only
    # that area is imported. api.py has already allowlisted the op before this process started.
    if handler is None and isinstance(op, str) and re.fullmatch(r'[a-z][a-z0-9]*_[a-z0-9_]+', op):
        area = op.split('_', 1)[0]
        if (Path(__file__).resolve().parent / f'ops_{area}.py').is_file():
            handler = getattr(importlib.import_module('engineering_sim.science.ops_' + area), 'OPS', {}).get(op)
    if handler is None:
        raise ValueError('Unknown science toolkit op: ' + repr(op))
    return handler(args)


def main():
    # Preserve the genuine stdout before redirecting fd 1 to stderr. The reply is
    # written straight to the saved descriptor so nothing else can reach stdout.
    reply_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        try:
            payload = json.load(sys.stdin)
        except Exception as exc:
            reply = {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}
        else:
            try:
                reply = {'ok': True, 'result': _plain(_run(payload))}
            except Exception as exc:
                reply = {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}
    finally:
        try:
            out = json.dumps(reply, allow_nan=False)
        except ValueError:
            out = json.dumps({'ok': False,
                              'error': 'ValueError: science toolkit op returned a non-finite number.'},
                             allow_nan=False)
        os.write(reply_fd, out.encode('utf-8'))
        os.close(reply_fd)


if __name__ == '__main__':
    main()

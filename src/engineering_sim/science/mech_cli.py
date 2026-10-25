# DESCRIPTION: `harness.py mech` command line: list/describe/calc/sweep/plot/verify over the mechcalc

"""Command-line front end for mechcalc.

    python3 harness.py mech list [--section statics] [--query belt]
    python3 harness.py mech describe statics.belt_friction
    python3 harness.py mech calc statics.belt_friction T2="100 N" mu=0.3 theta="180 deg"
    python3 harness.py mech calc statics.belt_friction T1="2 kN" T2="1 kN" theta="1 rev" --solve mu
    python3 harness.py mech calc materials_mech.torsion_circular T=500 L=1 G="79.3 GPa" d_o="40 mm" \
        d_i=0 --unit tau_max=MPa
    python3 harness.py mech sweep statics.belt_friction --var theta --from "0 deg" --to "720 deg" --n 13 \
        T2=100 mu=0.3
    python3 harness.py mech plot materials_mech.principal_stress mohr sx="80 MPa" sy="-40 MPa" txy="25 MPa"
    python3 harness.py mech verify [--section statics] [--roundtrip]

Inputs are NAME=VALUE with an optional unit inside the value; JSON-structured inputs (truss nodes,
part lists) are given as JSON text. --json prints the full machine-readable result.
Standard-library only at import time (dispatched from the command-line front end); work runs in the venv.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from engineering_sim.science.paths import VENV_PYTHON  # noqa: E402  (the single interpreter definition)


def _pairs(items):
    out = {}
    for item in items:
        if '=' not in item:
            raise SystemExit(f'error: input {item!r} must look like NAME=VALUE')
        k, v = item.split('=', 1)
        v = v.strip()
        if v[:1] in '[{':
            out[k.strip()] = json.loads(v)
        else:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v
    return out


def _fmt(v):
    if isinstance(v, float):
        return f'{v:.6g}'
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def _print_result(res):
    print(f"{res['calc']} — {res['title']}  [{res['method']}]")
    for eq in res.get('equations', []):
        print('  ' + eq['text'])
    width = max([len(k) for k in list(res['results']) + list(res['inputs'])] + [4])
    print('inputs:')
    for k, r in res['inputs'].items():
        print(f'  {k:<{width}} = {_fmt(r["value"])} {r.get("unit", "")}')
    print('results:')
    for k, r in res['results'].items():
        print(f'  {k:<{width}} = {_fmt(r["value"])} {r.get("unit", "")}')
    for alt in res.get('alternatives', []):
        print('  alternative root: ' + ', '.join(f'{k}={_fmt(v)}' for k, v in alt.items()))
    for w in res.get('warnings', []):
        print('warning: ' + w)
    for a in res.get('assumptions', []):
        print('assumes: ' + a)


def _dispatch(args):
    parser = argparse.ArgumentParser(prog='mech', description='mechcalc parametric engineering calculators')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('list')
    p.add_argument('--section')
    p.add_argument('--query')
    p = sub.add_parser('describe')
    p.add_argument('id')
    for name in ('calc', 'plot'):
        p = sub.add_parser(name)
        p.add_argument('id')
        if name == 'plot':
            p.add_argument('plot')
        p.add_argument('inputs', nargs='*', help='NAME=VALUE [unit]')
        p.add_argument('--solve', help='comma-separated unknowns (default: calculator outputs)')
        p.add_argument('--unit', action='append', default=[], help='NAME=UNIT for a result')
        p.add_argument('--json', action='store_true')
    p = sub.add_parser('sweep')
    p.add_argument('id')
    p.add_argument('inputs', nargs='*')
    p.add_argument('--var', required=True)
    p.add_argument('--from', dest='start', required=True)
    p.add_argument('--to', dest='stop', required=True)
    p.add_argument('--n', type=int, default=21)
    p.add_argument('--log', action='store_true')
    p.add_argument('--solve')
    p.add_argument('--json', action='store_true')
    p = sub.add_parser('verify')
    p.add_argument('--section')
    p.add_argument('--roundtrip', action='store_true', help='also run the inverse round trip (slow)')
    p.add_argument('--json', action='store_true')
    ns = parser.parse_args(args)

    from engineering_sim.science.mech import core, oracles
    solve = [s.strip() for s in (getattr(ns, 'solve', None) or '').split(',') if s.strip()] or None
    try:
        if ns.cmd == 'list':
            cat = core.catalog(query=ns.query, section_key=ns.section)
            for sec in cat['sections']:
                print(f"{sec['key']}: {sec['title']} ({len(sec['calcs'])})")
                for c in sec['calcs']:
                    print(f"  {c['id']:<48} {c['title']}")
            print(f"{cat['counts']['calcs']} calculators in {cat['counts']['sections']} sections")
            for k, e in cat['errors'].items():
                print(f'import error in {k}: {e}', file=sys.stderr)
            return 1 if cat['errors'] else 0
        if ns.cmd == 'describe':
            d = core.describe(ns.id)
            print(f"{d['id']} — {d['title']} ({d['kind']}); default unknowns: {', '.join(d['outputs'])}")
            for eq in d['equations']:
                print('  ' + eq['text'])
            for v in d['vars']:
                rng = f"[{v['lo'] if v['lo'] is not None else '-inf'}, {v['hi'] if v['hi'] is not None else 'inf'}]"
                print(f"  {v['name']:<14} {v['unit'] or '-':<10} {rng:<22} {v['desc']}"
                      + (f"  (example {v['example']})" if v['example'] is not None else ''))
            for a in d['assumptions']:
                print('assumes: ' + a)
            for r in d['refs']:
                print('ref: ' + r)
            for pl in d['plots']:
                print(f"plot: {pl['id']} — {pl['title']}")
            return 0
        if ns.cmd == 'calc':
            units = dict(u.split('=', 1) for u in ns.unit)
            res = core.compute(ns.id, _pairs(ns.inputs), solve, units)
            print(json.dumps(res, indent=1)) if ns.json else _print_result(res)
            return 0
        if ns.cmd == 'plot':
            res = core.plot(ns.id, _pairs(ns.inputs), ns.plot, solve)
            print(json.dumps(res, indent=1))
            return 0
        if ns.cmd == 'sweep':
            res = core.sweep(ns.id, _pairs(ns.inputs), ns.var, ns.start, ns.stop, n=ns.n, solve_for=solve,
                             log=ns.log)
            if ns.json:
                print(json.dumps(res, indent=1))
            else:
                cols = [res['var']] + res['unknowns']
                print('  '.join(f'{c:>14}' for c in cols) + '   (SI units)')
                for row in res['rows']:
                    print('  '.join(f'{_fmt(row[c]):>14}' for c in cols))
                if res['skipped']:
                    print(f"{res['skipped']} point(s) had no admissible solution")
            return 0
        if ns.cmd == 'verify':
            rep = oracles.verify(ns.section)
            rt = {}
            if ns.roundtrip:
                for cid, c in core.registry()['calcs'].items():
                    if ns.section and c.section != ns.section:
                        continue
                    fails = {k: v for k, v in core.roundtrip(c).items() if v.startswith('FAIL')}
                    if fails:
                        rt[cid] = fails
            if ns.json:
                print(json.dumps({'oracles': rep, 'roundtrip_failures': rt}, indent=1))
            else:
                for case in rep['cases']:
                    if not case['ok']:
                        print(f"FAIL {case['calc']}: {case['detail']}")
                for cid in rep['uncovered']:
                    print(f'NO ORACLE {cid}')
                for cid, f in rt.items():
                    print(f'ROUNDTRIP {cid}: {f}')
                print(f"oracles passed {rep['passed']} failed {rep['failed']} uncovered {len(rep['uncovered'])}"
                      + (f"; round-trip failures {len(rt)}" if ns.roundtrip else ''))
            return 0 if not rep['failed'] and not rep['uncovered'] and not rt else 1
    except core.MechError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    return 2


def main(argv):
    """Re-exec under configured science environment when the science stack is missing, then dispatch."""
    if importlib.util.find_spec('sympy') is None or importlib.util.find_spec('pint') is None:
        if os.environ.get('ENGINEERING_SIM_MECH_WORKER'):
            print('error: sympy/pint unavailable even under configured science environment', file=sys.stderr)
            return 2
        if not VENV_PYTHON.is_file():
            print('error: science toolkit interpreter (configured science environment) is missing', file=sys.stderr)
            return 2
        env = dict(os.environ, ENGINEERING_SIM_MECH_WORKER='1')
        return subprocess.call([str(VENV_PYTHON), '-B', '-m', 'engineering_sim.science.mech_cli'] + list(argv),
                               cwd=str(ROOT), env=env)
    return _dispatch(list(argv))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

# DESCRIPTION: Runs the independent numeric oracles in tests/fixtures/mech_oracles/<section>.json
#              against the calculators (shared by the test suite, the mech_verify op and the CLI).
"""Oracle file format (one per section):

{"section": "statics",
 "cases": [{"calc": "statics.belt_friction",
            "inputs": {"T2": "100 N", "mu": 0.3, "theta": "180 deg"},
            "solve_for": ["T1"],                      # optional; default = calculator outputs
            "expect": {"T1": "256.63 N"},             # value with unit, or bare number in SI unit
            "rel_tol": 0.001,                         # optional, default 0.002
            "source": {"kind": "hand" | "cited",
                       "derivation": "T1 = 100*exp(0.3*pi) = 256.63 N",   # required for hand
                       "ref": "..."}}]}                                   # required for cited

Oracles are typed from the physics (hand substitution) or a worked example; they are NEVER produced
by running mechcalc. A 'cited' oracle recalled rather than read from a fetched source must say so in
its ref ("recalled; not re-read") so reviewers know its evidentiary weight.
"""
import json
import math
from pathlib import Path

from engineering_sim.science.mech import core

ROOT = Path(__file__).resolve().parents[3]
ORACLE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'oracles'


def load(section=None):
    files = sorted(ORACLE_DIR.glob('*.json'))
    out = {}
    for f in files:
        if section and f.stem != section:
            continue
        out[f.stem] = json.loads(f.read_text())
    return out


def check_case(case):
    """Return (ok, detail) for one oracle case."""
    calc = core.get(case['calc'])
    src = case.get('source') or {}
    if src.get('kind') not in ('hand', 'cited'):
        return False, 'source.kind must be hand or cited'
    if src['kind'] == 'hand' and not src.get('derivation'):
        return False, 'hand oracle needs a derivation'
    if src['kind'] == 'cited' and not src.get('ref'):
        return False, 'cited oracle needs a ref'
    try:
        res = core.compute(case['calc'], case.get('inputs'), case.get('solve_for'))
    except core.MechError as exc:
        return False, f'compute failed: {exc}'
    tol = float(case.get('rel_tol', 0.002))
    fails = []
    for name, expected in case['expect'].items():
        rec = res['results'].get(name)
        if rec is None:
            fails.append(f'{name}: not in results')
            continue
        got = rec['value']
        if isinstance(expected, (list, dict)) or not isinstance(got, (int, float)):
            if not _struct_close(got, expected, tol):
                fails.append(f'{name}: got {got} expected {expected}')
            continue
        exp_si = core.to_si(expected, calc.var(name))
        if not math.isclose(got, exp_si, rel_tol=tol, abs_tol=_abs_tol(case, exp_si)):
            fails.append(f'{name}: got {got:.6g} {rec["unit"]} expected {exp_si:.6g}')
    return (not fails), '; '.join(fails) or 'ok'


def _abs_tol(case, expected):
    """Relative comparison only, unless the expected value is exactly zero.

    A fixed SI abs_tol (formerly 1e-9) silently accepted any result for small SI quantities — wear
    coefficients, film thicknesses, compressibility — so an Archard oracle expecting 9e-10 passed at 2e-11
   . A case may still set "abs_tol" explicitly (e.g. round-off at a free surface).
    """
    if 'abs_tol' in case:
        return float(case['abs_tol'])
    return 1e-12 if expected == 0 else 0.0


def _struct_close(got, exp, tol):
    if isinstance(exp, dict):
        return isinstance(got, dict) and all(k in got and _struct_close(got[k], v, tol) for k, v in exp.items())
    if isinstance(exp, list):
        return isinstance(got, list) and len(got) == len(exp) and all(_struct_close(g, e, tol) for g, e in zip(got, exp))
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        return math.isclose(got, exp, rel_tol=tol, abs_tol=1e-12 if exp == 0 else 0.0)
    return got == exp


def verify(section=None):
    """Run every oracle (optionally one section); report per case plus coverage of calculators."""
    reg = core.registry()
    report = {'cases': [], 'uncovered': [], 'passed': 0, 'failed': 0}
    covered = set()
    for sec, data in load(section).items():
        for case in data.get('cases', []):
            covered.add(case['calc'])
            try:
                ok, detail = check_case(case)
            except core.MechError as exc:
                ok, detail = False, str(exc)
            report['cases'].append({'section': sec, 'calc': case['calc'], 'ok': ok, 'detail': detail,
                                    'source_kind': (case.get('source') or {}).get('kind')})
            report['passed' if ok else 'failed'] += 1
    for cid, c in reg['calcs'].items():
        if (section is None or c.section == section) and cid not in covered:
            report['uncovered'].append(cid)
    return report

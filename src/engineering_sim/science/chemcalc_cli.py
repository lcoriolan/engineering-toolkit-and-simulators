# DESCRIPTION: Task J: Chemical Calculator CLI — run/list/show/verify/export, re-execing

"""Command-line front end for the Chemical Calculator harness.

Entry point: ``the command-line front end`` dispatches ``python3 harness.py chemcalc ...`` here
before argparse. This module runs under the studio interpreter (stdlib only) and
re-executes itself with ``python -B -m engineering_sim.science.chemcalc_cli``
when RDKit is not importable, so the actual work always runs in the venv.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from engineering_sim.science.paths import VENV_PYTHON  # noqa: E402  (the single interpreter definition)


def _print_run(recipe, run_meta, result):
    counts = result['counts']
    by_distance = ', '.join('%s:%s' % (d, counts['by_distance'][d])
                            for d in sorted(counts['by_distance'], key=int))
    print('recipe_id     %s' % recipe['recipe_id'])
    print('start         %s  (k = %s)' % (recipe['start'], recipe['k']))
    print('result_sha256 %s' % run_meta['result_sha256'])
    print('wall_time_s   %s' % run_meta['wall_time_s'])
    print('counts        generated=%s valence_valid=%s returned=%s known=%s hypothetical=%s' %
          (counts['generated'], counts['valence_valid'], counts['returned'],
           counts['known'], counts['hypothetical']))
    print('by_distance   %s' % by_distance)


def _dispatch(args):
    parser = argparse.ArgumentParser(prog='chemcalc', description='science toolkit Chemical Calculator harness')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_run = sub.add_parser('run', help='run a nearest-valid-compound search')
    p_run.add_argument('--start', default='H2O')
    p_run.add_argument('--k', type=int, default=1000)
    p_run.add_argument('--elements', help='comma-separated element symbols')
    p_run.add_argument('--max-charge', type=int, default=1)
    p_run.add_argument('--no-radicals', action='store_true')
    p_run.add_argument('--max-heavy-atoms', type=int, default=12)
    p_run.add_argument('--replay', action='store_true', help='reuse a saved snapshot (no network)')

    sub.add_parser('list', help='list stored runs, newest first')

    p_show = sub.add_parser('show', help='show a stored run')
    p_show.add_argument('recipe_id')
    p_show.add_argument('--top', type=int, default=20)

    p_verify = sub.add_parser('verify', help='recompute offline and compare the result hash')
    p_verify.add_argument('recipe_id')

    p_export = sub.add_parser('export', help='write a portable, verifiable zip')
    p_export.add_argument('recipe_id')

    ns = parser.parse_args(args)

    from engineering_sim.science import chemcalc

    if ns.cmd == 'run':
        fields = {'start': ns.start, 'k': ns.k, 'max_charge': ns.max_charge,
                  'allow_radicals': not ns.no_radicals, 'max_heavy_atoms': ns.max_heavy_atoms}
        if ns.elements:
            fields['elements'] = [e.strip() for e in ns.elements.split(',') if e.strip()]
        out = chemcalc.run(fields, replay=ns.replay)
        _print_run(out['recipe'], out['run'], out['result'])
        return 0

    if ns.cmd == 'list':
        rows = chemcalc.list_runs()
        if not rows:
            print('no runs yet')
            return 0
        for r in rows:
            print('%s  start=%s  k=%s  returned=%s  known=%s  %s' %
                  (r['recipe_id'][:12], r['start'], r['k'], r['returned'], r['known'], r['created'] or ''))
        return 0

    if ns.cmd == 'show':
        data = chemcalc.get(ns.recipe_id)
        recipe = data['recipe']
        result = data['result']
        print('start %s  k=%s  recipe_id %s' % (recipe['start'], recipe['k'], recipe['recipe_id']))
        print('start %s (%s)  cid=%s' % (result['start']['smiles'], result['start']['formula'],
                                          result['start']['cid'].get('value') if result['start'].get('cid') else '-'))
        print('%-5s %-18s %-12s %s %s %s %s' % ('rank', 'smiles', 'formula', 'd', 'sim', 'known', 'cid'))
        for n in result['neighbours'][:ns.top]:
            cid = n['cid'].get('value') if n.get('cid') else '-'
            print('%-5d %-18s %-12s %d %s %s %s' %
                  (n['rank'], n['smiles'], n['formula'], n['distance'],
                   n['similarity']['value'], n['known'], cid))
        return 0

    if ns.cmd == 'verify':
        v = chemcalc.verify(ns.recipe_id)
        print('ok=%s' % v['ok'])
        print('expected_sha256   %s' % v['expected_sha256'])
        print('recomputed_sha256 %s' % v['recomputed_sha256'])
        return 0 if v['ok'] else 1

    if ns.cmd == 'export':
        e = chemcalc.export(ns.recipe_id)
        print('zip_path %s' % e['zip_path'])
        print('bytes    %s' % e['bytes'])
        print('sha256   %s' % e['sha256'])
        return 0

    return 2


def main(argv):
    """Parse the global --data flag, re-exec under the venv if needed, then dispatch."""
    args = list(argv)
    if '--data' in args:
        i = args.index('--data')
        if i + 1 >= len(args):
            print('error: --data needs a directory', file=sys.stderr)
            return 2
        os.environ['ENGINEERING_SIM_DATA'] = str(Path(args[i + 1]).expanduser())
        args = args[:i] + args[i + 2:]

    if importlib.util.find_spec('rdkit') is None:
        if os.environ.get('ENGINEERING_SIM_CHEMCALC_WORKER'):
            print('error: RDKit is unavailable even under the configured science environment interpreter', file=sys.stderr)
            return 2
        if not VENV_PYTHON.is_file():
            print('error: science toolkit interpreter (configured science environment) is missing', file=sys.stderr)
            return 2
        cmd = [str(VENV_PYTHON), '-B', '-m', 'engineering_sim.science.chemcalc_cli'] + args
        env = dict(os.environ, ENGINEERING_SIM_CHEMCALC_WORKER='1')
        return subprocess.call(cmd, cwd=str(ROOT), env=env)

    try:
        return _dispatch(args)
    except (ValueError, KeyError, OSError) as exc:
        print('error: %s' % exc, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

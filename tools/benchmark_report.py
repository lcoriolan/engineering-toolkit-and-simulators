"""Reproduce the calculator evidence; expected values are read, never regenerated.
Run from the repository root: python tools/benchmark_report.py --output benchmarks
"""
import argparse
import dataclasses
import importlib.metadata as metadata
import inspect
import json
import math
import platform
from pathlib import Path
from engineering_sim.science.mech import core, oracles


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if hasattr(value, 'item'): return clean(value.item())
    return value


def compare(actual, expected, relative_tolerance, absolute_tolerance):
    if isinstance(expected, dict):
        return {k: compare(actual.get(k), v, relative_tolerance, 1e-12 if v == 0 else 0.0) for k, v in expected.items()}
    if isinstance(expected, list):
        return [compare(g, e, relative_tolerance, 1e-12 if e == 0 else 0.0) for g, e in zip(actual, expected)] if len(actual)==len(expected) else {'passed':False, 'reason':'length mismatch'}
    if isinstance(expected, (int,float)) and isinstance(actual,(int,float)):
        error = abs(actual-expected)
        return {'absolute_error':error, 'relative_error':error/abs(expected) if expected else None,
                'allowed_absolute_error':max(absolute_tolerance, relative_tolerance*abs(expected)),
                'passed':math.isclose(actual,expected,rel_tol=relative_tolerance,abs_tol=absolute_tolerance)}
    return {'passed':actual==expected}


def passed(row):
    if isinstance(row,dict): return row['passed'] if 'passed' in row else all(passed(x) for x in row.values())
    return all(passed(x) for x in row)


def category(case):
    ref = str(case.get('source',{})).lower()
    if 'coolprop' in ref or 'cantera' in ref: return 'backend regression; not independent physical validation'
    if case['source']['kind']=='hand': return 'analytical reference (independently specified expected value)'
    return 'reference landmark; original citation confidence retained in source; not a certified external benchmark'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='benchmarks');args=parser.parse_args()
    destination=Path(args.output);(destination/'calculators').mkdir(parents=True,exist_ok=True)
    versions={'python':platform.python_version()}
    for name in ['engineering-toolkit-and-simulators','numpy','scipy','sympy','pint','CoolProp','cantera','rdkit','warp-lang','ngsolve','netgen-mesher','iapws','pyscf','openmm','pycalphad','biopython','primer3-py','tblite','ase','gemmi','meeko','vina','pymatgen','scikit-fem']:
        try: versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError: pass
    (destination/'backend-versions.json').write_text(json.dumps(versions,indent=2)+'\n')
    registry=core.registry(); summaries=[];total=0;failures=0; covered=set()
    for section,data in oracles.load().items():
        report={'section':section,'command':f'python tools/benchmark_report.py --output {args.output}',
                'test':f'python -m pytest tests/test_mechanics_oracles.py -k {section}',
                'backend_versions':'../backend-versions.json','calculators':{},'cases':[]}
        for index,case in enumerate(data['cases']):
            calc=core.get(case['calc']);covered.add(calc.id)
            if calc.id not in report['calculators']:
                procedure=None
                if calc.procedure:
                    procedure={'python':inspect.getsource(calc.procedure),
                        'source':str(Path(inspect.getsourcefile(calc.procedure)).name),
                        'line':inspect.getsourcelines(calc.procedure)[1]}
                report['calculators'][calc.id]={'title':calc.title,'variables':[dataclasses.asdict(v) for v in calc.vars],
                    'equations':calc.equations,'procedure':procedure,'outputs':calc.outputs,
                    'assumptions':calc.assumptions,'method_references':calc.refs,'synthetic_example':calc.example}
            result=core.compute(calc.id,case.get('inputs'),case.get('solve_for'))
            comparisons={};expected_si={};actual={}
            for key,expected in case['expect'].items():
                got=result['results'][key];actual[key]=got
                expected_si[key]=expected if isinstance(expected,(list,dict)) or not isinstance(got['value'],(int,float)) else core.to_si(expected,calc.var(key))
                tol=float(case.get('rel_tol',.002));exp=expected_si[key]
                absolute=float(case.get('abs_tol',1e-12 if isinstance(exp,(int,float)) and exp==0 else 0.0))
                comparisons[key]=compare(got['value'],exp,tol,absolute)
            ok=passed(comparisons);total+=1;failures+=not ok
            report['cases'].append({'id':f'{section}-{index+1:03d}','calculator':calc.id,'classification':category(case),
                'inputs':case.get('inputs',{}),'solve_for':case.get('solve_for',calc.outputs),
                'expected_as_specified':case['expect'],'expected_SI':expected_si,'actual_SI':actual,
                'relative_tolerance':float(case.get('rel_tol',.002)),
                'comparison':comparisons,'passed':ok,'source':case['source']})
        report['passed']=sum(c['passed'] for c in report['cases']);report['failed']=len(report['cases'])-report['passed']
        (destination/'calculators'/f'{section}.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
        summaries.append({'section':section,'calculators':len(report['calculators']),'cases':len(report['cases']),'passed':report['passed'],'failed':report['failed']})
        print(section,report['passed'],report['failed'],flush=True)
    summary={'cases':total,'passed':total-failures,'failed':failures,'calculators':len(covered),'sections':summaries,
        'uncovered_calculators':sorted(set(registry['calcs'])-covered),'interpretation':'Numerical verification against declared references, not broad physical validation.'}
    (destination/'calculator-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Calculator benchmark results','',
        'Expected values are maintained separately from the calculation implementation. Each linked JSON contains every governing equation or numerical procedure, variable meaning and SI unit, assumptions, synthetic inputs, reference result, actual result, absolute and relative error, tolerance, and source confidence. Backend versions are in [backend-versions.json](backend-versions.json).',
        '',f'**{total-failures}/{total} cases passed; {len(covered)} calculators in {len(summaries)} families.**',
        '', 'Most references are hand substitutions into governing equations. Recalled textbook landmarks remain labelled; same-backend fluid/combustion values are regression checks. These results are not 1,232 independently measured physical validations.',
        '', 'Reproduce with `python tools/benchmark_report.py --output benchmarks`. See [methodology](../docs/methodology.md) for primary sources and interpretation.',
        '', '| Family | Calculators | Cases | Passed | Failed |','|---|---:|---:|---:|---:|']
    for row in summaries: lines.append(f"| [{row['section']}](calculators/{row['section']}.json) | {row['calculators']} | {row['cases']} | {row['passed']} | {row['failed']} |")
    (destination/'README.md').write_text('\n'.join(lines)+'\n')
    if failures or summary['uncovered_calculators']: raise SystemExit(1)

if __name__=='__main__':main()

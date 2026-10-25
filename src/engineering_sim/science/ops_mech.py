# DESCRIPTION: science toolkit ``mech`` area: six generic ops over the declarative mechanical-engineering
#              calculator registry (engineering_sim/science/mech/). Standard-library imports only at module level.
"""science toolkit ``mech`` area.

api.py imports this file under the standard-library studio interpreter to read TIMEOUTS, so the
science package (sympy/pint/scipy via engineering_sim.science.mech.core) is imported inside each handler, which
runs in the configured science environment worker. Calculators are added by writing a section module, never by adding
ops here.
"""

# Sections refused over the HTTP op path until a security review clears them. Direct core calls (gate tests)
# are unaffected so authors can keep fixing them. 2026-09-15: user-expression parsers in these sections used
# sympy parse_expr/sympify, which evaluate Python (confirmed code execution in review of brief 10); they must
# move to core.safe_expr before removal from this list.
QUARANTINE = {
}


def _guard(calc_id):
    section = str(calc_id or '').split('.', 1)[0]
    if section in QUARANTINE:
        raise ValueError(f'mechcalc section {section!r} is temporarily disabled: {QUARANTINE[section]}')


TIMEOUTS = {
    'mech_list': 60,
    'mech_describe': 60,
    'mech_calc': 90,      # symbolic solve (1.5 s limit per unknown set) + numeric fallback scans
    'mech_sweep': 300,    # up to 400 solves
    'mech_plot': 300,
    'mech_verify': 900,   # every oracle in every section
}


def _mech_list(args):
    from engineering_sim.science.mech import core
    return core.catalog(query=args.get('query'), section_key=args.get('section'))


def _mech_describe(args):
    _guard(args.get('id'))
    from engineering_sim.science.mech import core
    return core.describe(args.get('id', ''))


def _mech_calc(args):
    _guard(args.get('id'))
    from engineering_sim.science.mech import core
    return core.compute(args.get('id', ''), args.get('inputs') or {}, args.get('solve_for') or None,
                        args.get('out_units') or {})


def _mech_sweep(args):
    _guard(args.get('id'))
    from engineering_sim.science.mech import core
    return core.sweep(args.get('id', ''), args.get('inputs') or {}, args.get('var', ''),
                      args.get('start'), args.get('stop'), n=args.get('n', 50),
                      solve_for=args.get('solve_for') or None, log=bool(args.get('log')),
                      out_units=args.get('out_units') or {})


def _mech_plot(args):
    _guard(args.get('id'))
    from engineering_sim.science.mech import core
    return core.plot(args.get('id', ''), args.get('inputs') or {}, args.get('plot', ''),
                     solve_for=args.get('solve_for') or None)


def _mech_verify(args):
    from engineering_sim.science.mech import oracles
    return oracles.verify(section=args.get('section'))


OPS = {
    'mech_list': _mech_list,
    'mech_describe': _mech_describe,
    'mech_calc': _mech_calc,
    'mech_sweep': _mech_sweep,
    'mech_plot': _mech_plot,
    'mech_verify': _mech_verify,
}

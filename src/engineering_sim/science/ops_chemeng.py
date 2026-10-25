# DESCRIPTION: Chemical Engineering (chemeng) area ops: Cantera kinetics/reactors/flames,
#              CoolProp properties and VLE, flash and process balances.
"""Chemical Engineering (chemeng) extension area.

Module level is standard-library only (api.py imports it under the studio
interpreter to read TIMEOUTS). Heavy imports (Cantera, CoolProp, SciPy) happen
inside each handler, which the worker runs under configured science environment.
"""

# Per-op timeouts (seconds, <= 900). The flame solve is the slowest; the reactor
# runs and the equilibrium/state lookups are seconds or less.
TIMEOUTS = {
    'chemeng_mechanisms': 60,
    'chemeng_ignition': 600,
    'chemeng_equilibrate': 120,
    'chemeng_flame': 900,
    'chemeng_rates': 120,
    'chemeng_bench': 300,
    'chemeng_fluids': 60,
    'chemeng_state': 60,
    'chemeng_saturation': 60,
    'chemeng_envelope': 120,
    'chemeng_bubble_dew': 60,
    'chemeng_mix': 60,
    'chemeng_flash': 60,
    'chemeng_reactor': 60,
    'chemeng_lmtd': 60,
}


def _req(args, key):
    """Return args[key] or raise a clear error for a missing required argument."""
    if not isinstance(args, dict) or key not in args or args[key] is None:
        raise ValueError('Missing required argument %r.' % key)
    return args[key]


def _mechanisms(args):
    from engineering_sim.science.chemeng_kinetics import mechanisms
    return mechanisms()


def _ignition(args):
    from engineering_sim.science.chemeng_kinetics import ignition
    return ignition(args.get('mech', 'gri30'), _req(args, 'mixture'),
                    _req(args, 'T0_K'), _req(args, 'P_bar'), _req(args, 't_end_s'),
                    args.get('reactor', 'const_pressure'))


def _equilibrate(args):
    from engineering_sim.science.chemeng_kinetics import equilibrate
    return equilibrate(args.get('mech', 'gri30'), _req(args, 'mixture'),
                       _req(args, 'T'), _req(args, 'P'), args.get('constraint', 'TP'))


def _flame(args):
    from engineering_sim.science.chemeng_kinetics import flame_speed
    return flame_speed(args.get('mech', 'gri30'), _req(args, 'fuel'), _req(args, 'oxidizer'),
                       _req(args, 'phi'), _req(args, 'T0_K'), _req(args, 'P_bar'))


def _rates(args):
    from engineering_sim.science.chemeng_kinetics import rate_screen
    return rate_screen(_req(args, 'mixture'), _req(args, 'T_K'), _req(args, 'P_bar'),
                       args.get('mech', 'gri30'))


def _bench(args):
    from engineering_sim.science.chemeng_kinetics import bench_kinetics
    return bench_kinetics(_req(args, 'items'), _req(args, 'T_K'), _req(args, 'P_bar'),
                          _req(args, 't_s'), args.get('mech', 'gri30'))


def _fluids(args):
    from engineering_sim.science.chemeng_props import fluids
    return fluids()


def _state(args):
    from engineering_sim.science.chemeng_props import state
    return state(_req(args, 'fluid'), _req(args, 'inputs'))


def _saturation(args):
    from engineering_sim.science.chemeng_props import saturation
    return saturation(_req(args, 'fluid'), T_K=args.get('T_K'), P_Pa=args.get('P_Pa'))


def _envelope(args):
    from engineering_sim.science.chemeng_props import phase_envelope
    return phase_envelope(_req(args, 'fluid'))


def _bubble_dew(args):
    from engineering_sim.science.chemeng_props import mixture_bubble_dew
    return mixture_bubble_dew(_req(args, 'components'), _req(args, 'fractions'), _req(args, 'P_Pa'))


def _mix(args):
    from engineering_sim.science.chemeng_balance import stream_mix
    return stream_mix(_req(args, 'streams'))


def _flash(args):
    from engineering_sim.science.chemeng_balance import flash_rachford_rice
    return flash_rachford_rice(_req(args, 'z'), K=args.get('K'),
                               T_K=args.get('T_K'), P_Pa=args.get('P_Pa'))


def _reactor(args):
    from engineering_sim.science.chemeng_balance import reactor_sizing
    return reactor_sizing(args.get('kind', 'CSTR'), _req(args, 'k'), args.get('order', 1),
                          _req(args, 'X'), _req(args, 'F_A0'), _req(args, 'C_A0'))


def _lmtd(args):
    from engineering_sim.science.chemeng_balance import lmtd
    return lmtd(_req(args, 'Th_in'), _req(args, 'Th_out'), _req(args, 'Tc_in'),
                _req(args, 'Tc_out'), args.get('flow', 'counter'))


OPS = {
    'chemeng_mechanisms': _mechanisms,
    'chemeng_ignition': _ignition,
    'chemeng_equilibrate': _equilibrate,
    'chemeng_flame': _flame,
    'chemeng_rates': _rates,
    'chemeng_bench': _bench,
    'chemeng_fluids': _fluids,
    'chemeng_state': _state,
    'chemeng_saturation': _saturation,
    'chemeng_envelope': _envelope,
    'chemeng_bubble_dew': _bubble_dew,
    'chemeng_mix': _mix,
    'chemeng_flash': _flash,
    'chemeng_reactor': _reactor,
    'chemeng_lmtd': _lmtd,
}

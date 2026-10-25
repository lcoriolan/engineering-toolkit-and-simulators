# DESCRIPTION: The ONLY bridge from mechcalc section procedures to existing science toolkit science modules
#              (CoolProp fluid states, humid air, LMTD, Hall–Petch, elastic bounds, Cantera equilibrium).
#              Sections must call these instead of re-implementing property layers (no redundant tooling).
"""Reuse adapters. Every function returns plain SI floats (records from the wrapped modules are unwrapped
with cite.num) so section procedures stay simple; the wrapped module keeps owning the physics/data.

Wrapped (do not duplicate elsewhere in mechcalc):
  engineering_sim.science.chemeng_props   CoolProp pure-fluid / mixture states and saturation
  engineering_sim.science.chemeng_balance lmtd
  engineering_sim.science.materials_props hall_petch, rule_of_mixtures (as elastic_bounds)
  engineering_sim.science.chemeng_kinetics equilibrate (Cantera; adiabatic flame temperature with constraint='HP')
Owned here because no science toolkit module provides it yet:
  humid_air                     CoolProp HAPropsSI (ASHRAE RP-1485 real-gas moist air)
"""
import math

from engineering_sim.science import cite
from engineering_sim.science.mech.core import MechError

# chemeng_props output keys -> short SI names used by mechcalc procedures.
_STATE_KEYS = {'T_K': 'T', 'P_Pa': 'P', 'rho_kg_m3': 'rho', 'h_J_kg': 'h', 's_J_kg_K': 's',
               'cp_J_kg_K': 'cp', 'cv_J_kg_K': 'cv', 'mu_Pa_s': 'mu', 'k_W_m_K': 'k'}
_INPUT_KEYS = {'T': 'T_K', 'P': 'P_Pa', 'Q': 'Q', 'h': 'H', 's': 'S', 'rho': 'D'}


def fluid_state(fluid, **two):
    """State of a CoolProp fluid from exactly two of T (K), P (Pa), Q (0..1), h (J/kg), s (J/kg/K), rho.

    Returns {'T','P','rho','h','s','cp','cv','mu','k','phase','quality'} in SI. Transport properties are
    None when CoolProp has no model for the fluid/phase (e.g. inside the two-phase dome).
    """
    from engineering_sim.science import chemeng_props
    if len(two) != 2 or any(k not in _INPUT_KEYS for k in two):
        raise MechError(f'fluid_state needs exactly two of {sorted(_INPUT_KEYS)}; got {sorted(two)}')
    try:
        rec = chemeng_props.state(fluid, {_INPUT_KEYS[k]: float(v) for k, v in two.items()})
    except ValueError as exc:
        # Transport properties fail inside the dome; retry is the wrapped module's concern, so surface it.
        raise MechError(str(exc))
    out = {short: cite.num(rec[key]) for key, short in _STATE_KEYS.items() if key in rec}
    out['phase'], out['quality'] = rec.get('phase'), rec.get('quality')
    return out


def saturation(fluid, T=None, P=None):
    """Saturation point by T (K) or P (Pa): {'T','P','liquid': {...}, 'vapour': {...}} in SI."""
    from engineering_sim.science import chemeng_props
    try:
        rec = chemeng_props.saturation(fluid, T_K=T, P_Pa=P)
    except ValueError as exc:
        raise MechError(str(exc))
    side = lambda d: {short: cite.num(d[key]) for key, short in _STATE_KEYS.items() if key in d}
    return {'T': cite.num(rec['T_sat']), 'P': cite.num(rec['P_sat']),
            'liquid': side(rec['liquid']), 'vapour': side(rec['vapour'])}


def humid_air(output, **three):
    """Moist-air property via CoolProp HAPropsSI (SI names: T, B wet bulb, D dew point, R relative humidity
    0..1, W humidity ratio kg/kg, H enthalpy J/kg dry air, V m³/kg dry air, P total pressure Pa).

    Exactly three inputs, one of which must be P. Real-gas ASHRAE RP-1485 model (not ideal-gas psychrometrics).
    """
    import CoolProp.CoolProp as CP
    if len(three) != 3 or 'P' not in three:
        raise MechError('humid_air needs exactly three inputs including total pressure P')
    args = []
    for k, v in three.items():
        args += [k, float(v)]
    try:
        return float(CP.HAPropsSI(output, *args))
    except Exception as exc:
        raise MechError(f'HAPropsSI({output}) failed: {exc}')


def lmtd(Th_in, Th_out, Tc_in, Tc_out, flow='counter'):
    """Log-mean temperature difference (K) — delegates to chemeng_balance.lmtd."""
    from engineering_sim.science import chemeng_balance
    try:
        return cite.num(chemeng_balance.lmtd(Th_in, Th_out, Tc_in, Tc_out, flow)['LMTD'])
    except ValueError as exc:
        raise MechError(str(exc))


def hall_petch(sigma0_Pa, k_Pa_sqrt_m, d_m):
    """Hall–Petch yield strength in Pa — delegates to materials_props.hall_petch (which works in MPa)."""
    from engineering_sim.science import materials_props
    r = materials_props.hall_petch(sigma0_Pa / 1e6, k_Pa_sqrt_m / 1e6, d_m)
    if 'error' in r:
        raise MechError(r['error'])
    return cite.num(r['yield_strength']) * 1e6


def elastic_bounds(K1, G1, K2, G2, f1):
    """Two-phase effective-modulus bounds in Pa — delegates to materials_props.rule_of_mixtures (GPa API).

    Returns {'voigt': {'K','G'}, 'reuss': {...}, 'hs_upper': {...}, 'hs_lower': {...}}. Phase 1 volume
    fraction f1; isotropic phases. The fibre-composite relations E1 = Vf·Ef + Vm·Em and
    1/E2 = Vf/Ef + Vm/Em are simple section equations, not a duplicate of this bound calculation.
    """
    from engineering_sim.science import materials_props
    r = materials_props.rule_of_mixtures({'a': {'K': K1 / 1e9, 'G': G1 / 1e9}, 'b': {'K': K2 / 1e9, 'G': G2 / 1e9}},
                                         [f1, 1.0 - f1])
    if 'error' in r:
        raise MechError(r['error'])
    bounds = r.get('bounds', r)
    return {name: {m: cite.num(bounds[name][m]) * 1e9 for m in ('K', 'G')}
            for name in ('voigt', 'reuss', 'hs_upper', 'hs_lower') if name in bounds}


def equilibrium(mech, mixture, T, P_Pa, constraint='TP'):
    """Cantera chemical equilibrium — delegates to chemeng_kinetics.equilibrate (P converted Pa→bar).

    mixture is {species: moles}, e.g. {'CH4': 1, 'O2': 2, 'N2': 7.52}. constraint 'HP' from the reactant
    temperature gives the adiabatic flame temperature (key 'T', K). Returns the wrapped
    result with its records unwrapped where they are plain values.
    """
    from engineering_sim.science import chemeng_kinetics
    try:
        r = chemeng_kinetics.equilibrate(mech, mixture, T, P_Pa / 1e5, constraint)
    except ValueError as exc:
        raise MechError(str(exc))
    return {k: (cite.num(v) if cite.is_record(v) else v) for k, v in r.items()}


def g0():
    """Standard gravity 9.80665 m/s² (exact by definition, CGPM 1901)."""
    return 9.80665


def finite(x, name='value'):
    """Raise MechError unless x is a finite float (use in procedures before returning)."""
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        raise MechError(f'{name} is not finite')
    return float(x)

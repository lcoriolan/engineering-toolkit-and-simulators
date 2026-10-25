# DESCRIPTION: Chemical-engineering fluid properties on CoolProp 8.0.0: pure-fluid
#              state points, saturation, phase envelope and HEOS bubble/dew mixtures.
"""Fluid thermophysical properties (CoolProp 8.0.0).

Pure-fluid state points, saturation (T/P), the T-P saturation envelope to the
critical point, and bubble/dew points of HEOS-supported mixtures. Every reported
quantity is a ``cite`` record (kind ``estimated``, method naming CoolProp and the
reference equation of state the fluid uses).

Units are SI: T [K], P [Pa], density [kg/m3], specific enthalpy [J/kg], specific
entropy [J/(kg K)], cp/cv [J/(kg K)], dynamic viscosity [Pa s], thermal
conductivity [W/(m K)]. The page converts to the units it displays.
"""
from __future__ import annotations

import numpy as np
import CoolProp.CoolProp as CP

from engineering_sim.science import cite

METHOD = 'CoolProp 8.0.0'
MAX_ENVELOPE_POINTS = 200

# Public input keys -> CoolProp input names. Exactly two are required for a state.
INPUT_MAP = {'T_K': 'T', 'P_Pa': 'P', 'Q': 'Q', 'H': 'H', 'S': 'S', 'D': 'D'}

# Outputs returned by state()/saturation(): key -> (CoolProp name, unit, label).
OUTPUTS = [
    ('T_K', 'T', 'K'),
    ('P_Pa', 'P', 'Pa'),
    ('rho_kg_m3', 'D', 'kg/m3'),
    ('h_J_kg', 'H', 'J/kg'),
    ('s_J_kg_K', 'S', 'J/(kg K)'),
    ('cp_J_kg_K', 'C', 'J/(kg K)'),
    ('cv_J_kg_K', 'O', 'J/(kg K)'),
    ('mu_Pa_s', 'V', 'Pa s'),
    ('k_W_m_K', 'L', 'W/(m K)'),
]


def _eos(fluid):
    """Reference equation of state for a pure fluid, from CoolProp's own metadata."""
    try:
        return CP.get_fluid_param_string(fluid, 'BibTeX-EOS') or None
    except Exception:
        return None


def _cite(value, unit, fluid, note=None):
    """One ``estimated`` record naming CoolProp and the fluid's reference EOS."""
    eos = _eos(fluid)
    method = METHOD if not eos else '%s, %s' % (METHOD, eos)
    return cite.value(value, unit, 'estimated', method=method, note=note)


def fluids():
    """Sorted list of the pure fluids and mixtures CoolProp can model."""
    names = sorted(CP.FluidsList())
    return {'count': len(names), 'fluids': names}


def _inputs(args):
    """Validate and extract exactly two state-defining inputs from a key dict."""
    if not isinstance(args, dict):
        raise ValueError('inputs must be a dict with exactly two state properties.')
    present = {k: v for k, v in args.items() if v is not None}
    unknown = set(present) - set(INPUT_MAP)
    if unknown:
        raise ValueError('Unknown state input(s): %s. Use two of %s.'
                         % (', '.join(sorted(unknown)), ', '.join(sorted(INPUT_MAP))))
    if len(present) != 2:
        raise ValueError('Exactly two state properties are required; got %d: %s.'
                         % (len(present), ', '.join(sorted(present)) or 'none'))
    pairs = [(INPUT_MAP[k], float(v)) for k, v in sorted(present.items())]
    return pairs


def _phase(fluid, n1, v1, n2, v2):
    """Phase label (liquid/gas/twophase/supercritical/...) from CoolProp."""
    try:
        return CP.PhaseSI(n1, v1, n2, v2, fluid)
    except Exception:
        return 'unknown'


def _quality(fluid, phase, n1, v1, n2, v2):
    """Vapour quality: 0 liquid / 1 gas / 0..1 two-phase / None supercritical."""
    if phase == 'twophase':
        try:
            return float(CP.PropsSI('Q', n1, v1, n2, v2, fluid))
        except Exception:
            return None
    if phase == 'liquid':
        return 0.0
    if phase == 'gas':
        return 1.0
    return None


def state(fluid, inputs):
    """Full property set at a thermodynamic state defined by two of T/P/Q/H/S/D.

    ``inputs`` keys are ``T_K``, ``P_Pa``, ``Q``, ``H`` (J/kg), ``S`` (J/(kg K)),
    ``D`` (kg/m3); exactly two are required.
    """
    pairs = _inputs(inputs)
    (n1, v1), (n2, v2) = pairs

    def prop(name):
        try:
            return float(CP.PropsSI(name, n1, v1, n2, v2, fluid))
        except Exception as exc:
            raise ValueError('CoolProp cannot compute %s for %r at %s=%g, %s=%g: %s'
                             % (name, fluid, n1, v1, n2, v2, exc))

    phase = _phase(fluid, n1, v1, n2, v2)
    result = {'fluid': fluid, 'phase': phase,
              'quality': _quality(fluid, phase, n1, v1, n2, v2)}
    for key, cname, unit in OUTPUTS:
        result[key] = _cite(prop(cname), unit, fluid)
    return result


def saturation(fluid, T_K=None, P_Pa=None):
    """Saturated liquid/vapour properties at a given temperature or pressure.

    Exactly one of ``T_K`` / ``P_Pa`` defines the saturation point; the other is
    computed by CoolProp (quality 0). Liquid and vapour properties are returned
    side by side.
    """
    if (T_K is None) == (P_Pa is None):
        raise ValueError('Provide exactly one of T_K or P_Pa.')
    if T_K is not None:
        T = float(T_K)
        P = float(CP.PropsSI('P', 'T', T, 'Q', 0.0, fluid))
    else:
        P = float(P_Pa)
        T = float(CP.PropsSI('T', 'P', P, 'Q', 0.0, fluid))

    def props(q):
        out = {}
        for key, cname, unit in OUTPUTS:
            out[key] = _cite(float(CP.PropsSI(cname, 'T', T, 'Q', q, fluid)), unit, fluid)
        return out

    return {
        'fluid': fluid,
        'T_sat': _cite(T, 'K', fluid, note='saturation temperature'),
        'P_sat': _cite(P, 'Pa', fluid, note='saturation pressure'),
        'liquid': props(0.0),
        'vapour': props(1.0),
    }


def phase_envelope(fluid):
    """T-P saturation curve from the triple point to the critical point (<= 200 pts).

    Points are plain ``[T_K, P_Pa]`` pairs for plotting; the critical and triple
    points are also returned as cite records.
    """
    try:
        Tcrit = float(CP.PropsSI('Tcrit', '', 0.0, '', 0.0, fluid))
        Pcrit = float(CP.PropsSI('Pcrit', '', 0.0, '', 0.0, fluid))
        Ttriple = float(CP.PropsSI('Ttriple', '', 0.0, '', 0.0, fluid))
    except Exception as exc:
        raise ValueError('CoolProp has no critical/triple point data for %r: %s' % (fluid, exc))

    Ts = np.linspace(Ttriple + 0.5, Tcrit - 0.05, MAX_ENVELOPE_POINTS - 1)
    points = []
    for t in Ts:
        try:
            points.append([float(t), float(CP.PropsSI('P', 'T', float(t), 'Q', 0.0, fluid))])
        except Exception:
            continue  # skip points where the EOS will not converge near the ends
    points.append([Tcrit, Pcrit])

    return {
        'fluid': fluid,
        'T_crit': _cite(Tcrit, 'K', fluid, note='critical temperature'),
        'P_crit': _cite(Pcrit, 'Pa', fluid, note='critical pressure'),
        'T_triple': _cite(Ttriple, 'K', fluid, note='triple-point temperature'),
        'points': points,
    }


def mixture_bubble_dew(components, fractions, P_Pa):
    """Bubble and dew temperatures of a HEOS mixture at a fixed pressure.

    ``components`` is a list of CoolProp fluid names; ``fractions`` their mole
    fractions (must sum to 1). Unsupported pairs are refused with CoolProp's reason.
    """
    from CoolProp.CoolProp import AbstractState

    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError('components must be a non-empty list of CoolProp fluid names.')
    if len(components) != len(fractions):
        raise ValueError('components and fractions must have the same length.')
    total = sum(float(f) for f in fractions)
    if not 0.999 <= total <= 1.001:
        raise ValueError('mole fractions must sum to 1; got %g.' % total)
    P_Pa = float(P_Pa)
    fluid_string = '&'.join(str(c) for c in components)
    try:
        as_state = AbstractState('HEOS', fluid_string)
        as_state.set_mole_fractions([float(f) for f in fractions])
    except Exception as exc:
        raise ValueError('CoolProp HEOS does not support the mixture %s: %s'
                         % (fluid_string, exc))
    try:
        as_state.update(CP.PQ_INPUTS, P_Pa, 0.0)
        T_bubble = float(as_state.T())
        x = [float(as_state.mole_fractions_liquid()[i]) for i in range(len(components))]
        as_state.update(CP.PQ_INPUTS, P_Pa, 1.0)
        T_dew = float(as_state.T())
        y = [float(as_state.mole_fractions_vapor()[i]) for i in range(len(components))]
    except Exception as exc:
        raise ValueError('Bubble/dew calculation failed for %s at %g Pa: %s'
                         % (fluid_string, P_Pa, exc))

    method = '%s, HEOS mixture' % METHOD
    return {
        'components': list(components),
        'fractions': [float(f) for f in fractions],
        'P_Pa': _cite(P_Pa, 'Pa', components[0], note='bubble/dew pressure'),
        'T_bubble': cite.value(T_bubble, 'K', 'estimated', method=method,
                               note='bubble point (first vapour), fixed P'),
        'T_dew': cite.value(T_dew, 'K', 'estimated', method=method,
                            note='dew point (last liquid), fixed P'),
        'x_bubble': {components[i]: x[i] for i in range(len(components))},
        'y_dew': {components[i]: y[i] for i in range(len(components))},
    }

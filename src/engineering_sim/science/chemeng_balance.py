# DESCRIPTION: Chemical-engineering mass/energy balances (stdlib + scipy): stream
#              mixing, Rachford-Rice flash, isothermal reactor sizing, heat duty, LMTD.
"""Process-engineering balances: mixing, flash, reactor sizing, heat exchange.

Standard library + SciPy. CoolProp is imported lazily inside ``stream_mix`` only
when a stream names a CoolProp fluid (constant-cp streams need no heavy imports).

Kinds follow the module contract: exact design equations are ``derived`` (formula
in ``method``); model-based quantities (constant-cp mixing, Raoult K-values) are
``estimated`` and their assumptions are stated in the notes.
"""
from __future__ import annotations

import math

from engineering_sim.science import cite

R = 8.314462618  # J/(mol K), CODATA 2018


# ---- stream mixing -------------------------------------------------------------
def stream_mix(streams):
    """Adiabatic mixing of process streams: mass balance + energy balance.

    Each stream is a dict:
      name   (str)               — label
      m_dot  (float, kg/s)       — mass flow
      T_K    (float)             — inlet temperature
      cp     (float, J/(kg K))   — constant specific heat, OR
      fluid  (str)               — CoolProp fluid name (cp evaluated at the inlet T)
      P_Pa   (float)             — pressure for the CoolProp cp lookup (default 1e5)
      species (dict {name: mass fraction}) — for the component mass balance

    The mixing temperature uses a constant-cp energy balance,
        T_mix = sum(m_dot_i * cp_i * T_i) / sum(m_dot_i * cp_i),
    with cp_i either the given constant or CoolProp's cp at the inlet temperature
    (not iterated to the mix temperature — a first-order estimate).
    """
    if not isinstance(streams, (list, tuple)) or not streams:
        raise ValueError('streams must be a non-empty list of stream dicts.')
    total_m = 0.0
    num = 0.0
    cp_sum = 0.0
    per_stream = []
    species = {}
    for s in streams:
        name = s.get('name', '')
        m_dot = float(s['m_dot'])
        T = float(s['T_K'])
        if m_dot <= 0.0:
            raise ValueError('stream %r: m_dot must be positive.' % name)
        if not 1.0 <= T <= 6000.0:
            raise ValueError('stream %r: T_K must be within [1, 6000] K.' % name)
        if 'cp' in s and s['cp'] is not None:
            cp = float(s['cp'])
            cp_source = 'constant cp'
        elif s.get('fluid'):
            import CoolProp.CoolProp as CP
            P = float(s.get('P_Pa', 1e5))
            cp = float(CP.PropsSI('C', 'T', T, 'P', P, str(s['fluid'])))
            cp_source = 'CoolProp cp(%s) at inlet T' % s['fluid']
        else:
            raise ValueError('stream %r: give either cp (J/(kg K)) or fluid (CoolProp name).' % name)
        if cp <= 0.0:
            raise ValueError('stream %r: cp must be positive.' % name)
        total_m += m_dot
        num += m_dot * cp * T
        cp_sum += m_dot * cp
        for sp, frac in (s.get('species') or {}).items():
            species[sp] = species.get(sp, 0.0) + m_dot * float(frac)
        per_stream.append({'name': name, 'm_dot': m_dot, 'T_K': T, 'cp': cp, 'cp_source': cp_source})

    cp_mix = cp_sum / total_m
    T_mix = num / cp_sum
    mixed_species = {sp: cite.value(v / total_m, '-', 'derived',
                                    method='mass-fraction weighted average') for sp, v in species.items()} or None

    return {
        'm_dot_total': cite.value(total_m, 'kg/s', 'derived', method='sum of stream mass flows'),
        'T_mix': cite.value(T_mix, 'K', 'estimated',
                            method='constant-cp energy balance: T_mix = sum(m cp T)/sum(m cp)',
                            note='fluid cp is evaluated at the inlet temperature (first-order estimate)'),
        'cp_mix': cite.value(cp_mix, 'J/(kg K)', 'derived', method='mass-weighted average cp'),
        'species': mixed_species,
        'streams': per_stream,
        'notes': [
            'Adiabatic mixing; no heat of mixing and no phase change are modelled.',
            'Constant-cp energy balance; CoolProp cp (where used) is evaluated at the inlet '
            'temperature, so T_mix is a first-order estimate.',
        ],
    }


# ---- flash ---------------------------------------------------------------------
# Antoine data for the Raoult path. Only water is bundled (engineering_sim.science.volatility
# carries its NIST Antoine fit); other components are refused rather than guessed.
_ANTOINE_COMPONENTS = {'water': None, 'h2o': None}  # value filled from volatility at call time


def _raoult_K(components, T_K, P_Pa):
    """K_i = P_sat_i(T)/P from Antoine data where available (Raoult's law, ideal gas)."""
    from engineering_sim.science import volatility

    K = {}
    for comp in components:
        if comp.strip().lower() not in _ANTOINE_COMPONENTS:
            raise ValueError('No Antoine data for component %r; the Raoult path only '
                             'bundles water. Supply K-values directly instead.' % comp)
        record = {'antoine': volatility.WATER_ANTOINE}
        p_sat_bar = cite.num(volatility.vapor_pressure(record, T_K))
        K[comp] = p_sat_bar * 1e5 / P_Pa
    return K


def flash_rachford_rice(z, K=None, T_K=None, P_Pa=None):
    """Two-phase flash: vapour fraction V and the liquid (x) / vapour (y) compositions.

    Rachford-Rice equation
        sum_i z_i (K_i - 1) / (1 + (K_i - 1) V) = 0
    solved for V in [0, 1]. K-values may be given directly (``K`` dict) or computed
    from Raoult's law with Antoine vapour pressures (``T_K``, ``P_Pa``; only water
    is bundled). ``z`` is the feed mole-fraction dict (normalised here).
    """
    from scipy.optimize import brentq

    if not isinstance(z, dict) or not z:
        raise ValueError('z must be a non-empty {component: mole fraction} dict.')
    total = sum(float(v) for v in z.values())
    if total <= 0.0:
        raise ValueError('z mole fractions must sum to a positive number.')
    z = {k: float(v) / total for k, v in z.items()}
    components = sorted(z.keys())

    if K is None:
        if T_K is None or P_Pa is None:
            raise ValueError('Provide K-values, or T_K and P_Pa for the Raoult path.')
        K = _raoult_K(components, float(T_K), float(P_Pa))
    else:
        if set(K.keys()) != set(z.keys()):
            raise ValueError('K must have exactly the same components as z.')
        K = {k: float(K[k]) for k in components}
    for k, v in K.items():
        if v <= 0.0:
            raise ValueError('K-value for %r must be positive.' % k)

    def f(V):
        return sum(z[c] * (K[c] - 1.0) / (1.0 + (K[c] - 1.0) * V) for c in components)

    f0 = f(0.0)  # = sum z_i (K_i - 1)
    f1 = f(1.0)  # = sum z_i (K_i - 1)/K_i
    if f0 <= 0.0:
        V, state = 0.0, 'all liquid (subcooled)'
    elif f1 >= 0.0:
        V, state = 1.0, 'all vapour (superheated)'
    else:
        V = float(brentq(f, 0.0, 1.0, xtol=1e-12))
        state = 'two-phase'

    x = {c: z[c] / (1.0 + (K[c] - 1.0) * V) for c in components}
    y = {c: K[c] * z[c] / (1.0 + (K[c] - 1.0) * V) for c in components}

    return {
        'V': cite.value(V, '-', 'derived', method='Rachford-Rice (V in [0,1])', note=state),
        'phase': state,
        'x': {c: cite.value(x[c], '-', 'derived', method='x_i = z_i/(1+(K_i-1)V)') for c in components},
        'y': {c: cite.value(y[c], '-', 'derived', method='y_i = K_i x_i') for c in components},
        'K': K,
        'notes': [
            'Rachford-Rice assumes ideal liquid/vapour equilibrium at the given K-values; '
            'K from Raoult is ideal-solution + ideal-gas (Antoine vapour pressure).',
        ],
    }


# ---- reactor sizing ------------------------------------------------------------
def reactor_sizing(kind='CSTR', k=None, order=1, X=None, F_A0=None, C_A0=None):
    """Isothermal reactor volume for a CSTR or PFR, plus the Damkoehler number.

    Design equations for rate = k * C_A^n = k * C_A0^n * (1-X)^n:
      CSTR  V = F_A0 X / [k C_A0^n (1-X)^n]
      PFR   V = F_A0/(k C_A0^n) * { -ln(1-X)              (n = 1)
                                     [(1-X)^(1-n) - 1]/(n-1)  (n != 1) }
    Da = (inlet rate) * V / F_A0; tau = V C_A0 / F_A0.

    Units: F_A0 [mol/s], C_A0 [mol/m3], k [(mol/m3)^(1-n) / s], V [m3].
    """
    if k is None or X is None or F_A0 is None or C_A0 is None:
        raise ValueError('k, X, F_A0 and C_A0 are all required.')
    kind = (kind or 'CSTR').upper()
    if kind not in ('CSTR', 'PFR'):
        raise ValueError("kind must be 'CSTR' or 'PFR', got %r." % kind)
    k = float(k)
    order = float(order)
    X = float(X)
    F_A0 = float(F_A0)
    C_A0 = float(C_A0)
    if k <= 0.0 or F_A0 <= 0.0 or C_A0 <= 0.0 or order <= 0.0:
        raise ValueError('k, order, F_A0 and C_A0 must all be positive.')
    if not 0.0 < X < 1.0:
        raise ValueError('X (conversion) must be within (0, 1), got %r.' % X)

    rate0 = k * C_A0 ** order  # inlet volumetric reaction rate, mol/(m3 s)
    if kind == 'CSTR':
        V = F_A0 * X / (k * C_A0 ** order * (1.0 - X) ** order)
        formula = 'V = F_A0 X / [k C_A0^n (1-X)^n]'
    else:
        if abs(order - 1.0) < 1e-12:
            V = F_A0 / (k * C_A0) * (-math.log(1.0 - X))
        else:
            V = F_A0 / (k * C_A0 ** order) * ((1.0 - X) ** (1.0 - order) - 1.0) / (order - 1.0)
        formula = 'V = F_A0/(k C_A0^n) * integral dX/(1-X)^n'
    Da = rate0 * V / F_A0
    tau = V * C_A0 / F_A0

    return {
        'kind': kind,
        'V': cite.value(V, 'm3', 'derived', method=formula),
        'Da': cite.value(Da, '-', 'derived', method='Da = (inlet rate) * V / F_A0'),
        'tau': cite.value(tau, 's', 'derived', method='tau = V * C_A0 / F_A0 (space time)'),
        'notes': [
            'Isothermal, single irreversible reaction A -> products, rate = k C_A^n; '
            'no pressure drop, no volume change, steady state.',
        ],
    }


# ---- heat exchange -------------------------------------------------------------
def heat_duty(m_dot, cp, dT):
    """Sensible heat duty Q = m_dot * cp * dT [W]."""
    m_dot, cp, dT = float(m_dot), float(cp), float(dT)
    return {
        'Q': cite.value(m_dot * cp * dT, 'W', 'derived', method='Q = m_dot * cp * dT'),
        'notes': ['Sensible heat only; no latent heat or heat loss.'],
    }


def lmtd(Th_in, Th_out, Tc_in, Tc_out, flow='counter'):
    """Log-mean temperature difference for a heat exchanger.

    counter:  dT1 = Th_in - Tc_out,  dT2 = Th_out - Tc_in
    parallel: dT1 = Th_in - Tc_in,   dT2 = Th_out - Tc_out
    LMTD = (dT1 - dT2) / ln(dT1 / dT2).
    """
    Th_in, Th_out, Tc_in, Tc_out = map(float, (Th_in, Th_out, Tc_in, Tc_out))
    flow = (flow or 'counter').lower()
    if flow not in ('counter', 'parallel'):
        raise ValueError("flow must be 'counter' or 'parallel', got %r." % flow)
    if flow == 'counter':
        dT1 = Th_in - Tc_out
        dT2 = Th_out - Tc_in
    else:
        dT1 = Th_in - Tc_in
        dT2 = Th_out - Tc_out
    if dT1 <= 0.0 or dT2 <= 0.0:
        raise ValueError('temperature approaches must be positive (dT1=%g, dT2=%g).' % (dT1, dT2))
    value = dT1 if abs(dT1 - dT2) < 1e-12 else (dT1 - dT2) / math.log(dT1 / dT2)
    return {
        'LMTD': cite.value(value, 'K', 'derived',
                           method='LMTD = (dT1 - dT2) / ln(dT1/dT2)', note=flow + ' flow'),
        'dT1': cite.value(dT1, 'K', 'derived', method='end temperature approach 1'),
        'dT2': cite.value(dT2, 'K', 'derived', method='end temperature approach 2'),
        'notes': ['Single-pass, constant specific heat, no heat loss.'],
    }

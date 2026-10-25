# DESCRIPTION: Thermodynamics section: ideal-gas and real-fluid state properties, specific heats, quality
#              and two-phase mixtures, boundary work, first-law balances (closed system, control volume,
#              steady-flow devices, transient filling/discharging), thermal and Carnot efficiency, COP,
#              entropy generation and isentropic efficiency, exergy and second-law efficiency, gas
#              mixtures, Clapeyron and Clausius-Clapeyron, and the Joule-Thomson coefficient. Real-fluid
#              states go through common.fluid_state / common.saturation (CoolProp).
"""Thermodynamics calculators.

Conventions:
- Absolute temperature in K; temperature differences in K (delta_degC). Specific properties are per
  unit mass (J/kg, m^3/kg, J/(kg K)); extensive closed-system quantities use J, m^3, J/K.
- Sign conventions: heat Q into a system and work W out of a system are positive (first law Q - W = dU).
  Entropy generation s_gen >= 0 for any real process.
- Ideal-gas relations use constant cp, cv with cp - cv = R and gamma = cp/cv.
- Real-fluid states (fluid_state_TP, fluid_state_TQ, fluid_saturation_dome, joule_thomson) call
  common.fluid_state / common.saturation (CoolProp); the fluid is named as text (Water, R134a, Air, ...).

Relations are the standard ones in engineering-thermodynamics texts (e.g. Cengel & Boles,
*Thermodynamics: An Engineering Approach*; Moran et al., *Fundamentals of Engineering Thermodynamics*;
Sonntag et al., *Fundamentals of Thermodynamics*). They are cited by relation name, recalled rather than
re-read, so no page numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts, common

SECTION = section('thermo', 'Thermodynamics', order=270,
                  summary='Ideal-gas and real-fluid states, energy and entropy balances, cycles, '
                          'exergy, gas mixtures, Clapeyron and Joule-Thomson.')

PI = math.pi


# ---------------------------------------------------------------------------------------------------
# Procedures and chart builders
# ---------------------------------------------------------------------------------------------------
def _fluid_name(v):
    """CoolProp fluid name from the text input (e.g. 'Water')."""
    if not isinstance(v, str) or not v.strip():
        raise MechError('fluid: give the CoolProp fluid name as text, e.g. Water')
    return v.strip()


def _fluid_state_TP(values):
    """Real-fluid state at (T, P): density, enthalpy, entropy, internal energy, cp, cv."""
    fluid = _fluid_name(values['fluid'])
    st = common.fluid_state(fluid, T=values['T'], P=values['P'])
    P = st['P']
    return {'rho': st['rho'], 'h': st['h'], 's': st['s'], 'u': st['h'] - P / st['rho'],
            'cp': st['cp'], 'cv': st['cv']}


def _fluid_state_TQ(values):
    """Real-fluid two-phase state at (T, quality Q): saturated pressure and mixture properties."""
    fluid = _fluid_name(values['fluid'])
    st = common.fluid_state(fluid, T=values['T'], Q=values['Q'])
    P, rho = st['P'], st['rho']
    return {'h': st['h'], 'P': P, 'rho': rho, 's': st['s'], 'u': st['h'] - P / rho}


def _joule_thomson(values):
    """Joule-Thomson coefficient mu = (dT/dP)_h by central finite difference on h (CoolProp states).

    mu = -(dh/dP)_T / (dh/dT)_P, with (dh/dP)_T and (dh/dT)_P evaluated by central differences on
    common.fluid_state; the denominator is approximately cp. Stated as a numerical finite difference.
    The four stencil states must share one single phase, otherwise a phase boundary is being straddled
    and the finite difference is meaningless (raises MechError).
    """
    fluid = _fluid_name(values['fluid'])
    T, P = values['T'], values['P']
    dT, dP = 0.5, 1000.0
    s_pp = common.fluid_state(fluid, T=T, P=P + dP)
    s_pm = common.fluid_state(fluid, T=T, P=P - dP)
    s_tp = common.fluid_state(fluid, T=T + dT, P=P)
    s_tm = common.fluid_state(fluid, T=T - dT, P=P)
    phases = {s_pp['phase'], s_pm['phase'], s_tp['phase'], s_tm['phase']}
    if len(phases) != 1 or next(iter(phases)) in ('twophase', 'unknown', None):
        raise MechError('Joule-Thomson coefficient needs a single-phase state; the +/-dT/dP stencil at '
                        f'(T = {T:g} K, P = {P:g} Pa) crosses a phase boundary '
                        f'({sorted(map(str, phases))}). Move the state away from the saturation curve.')
    dh_dP = (s_pp['h'] - s_pm['h']) / (2.0 * dP)
    dh_dT = (s_tp['h'] - s_tm['h']) / (2.0 * dT)
    return {'mu': -dh_dP / dh_dT, 'cp': dh_dT}


def _saturation_dome(values):
    """Saturation curve points from common.saturation, for a T-s / P-h chart."""
    fluid = _fluid_name(values['fluid'])
    rng = values['T_range']
    if not isinstance(rng, list) or len(rng) != 2:
        raise MechError('T_range: give [T_min (K), T_max (K)]')
    T_min, T_max = float(rng[0]), float(rng[1])
    if T_min >= T_max:
        raise MechError('T_range needs T_min < T_max, inside the fluid saturation range')
    N = 40
    Ts = [T_min + (T_max - T_min) * i / (N - 1) for i in range(N)]
    Ps, sfs, sgs, hfs, hgs = [], [], [], [], []
    for T in Ts:
        sat = common.saturation(fluid, T=T)
        Ps.append(sat['P'])
        sfs.append(sat['liquid']['s'])
        sgs.append(sat['vapour']['s'])
        hfs.append(sat['liquid']['h'])
        hgs.append(sat['vapour']['h'])
    return {'P_min': Ps[0], 'P_max': Ps[-1],
            'dome': {'T': Ts, 'P': Ps, 's_f': sfs, 's_g': sgs, 'h_f': hfs, 'h_g': hgs}}


def _boundary_work_isothermal(values):
    """Boundary work of an isothermal ideal-gas process W = P1 V1 ln(V2/V1) (procedure: numeric inversion)."""
    P1, V1, V2 = values['P1'], values['V1'], values['V2']
    return {'W': P1 * V1 * math.log(V2 / V1)}


def _exergy(values):
    """Specific flow exergy psi = (h - h0) - T0 (s - s0) + V^2/2 + g z, with g = common.g0()."""
    h, h0, T0, s, s0, V, z = (values[k] for k in ('h', 'h0', 'T0', 's', 's0', 'V', 'z'))
    return {'psi': (h - h0) - T0 * (s - s0) + 0.5 * V * V + common.g0() * z}


def _ts_chart(values):
    """T-s diagram from the saturation-dome points."""
    d = values['dome']
    return charts.chart('T-s diagram', 's [J/(kg K)]', 'T [K]',
                        [charts.series('saturated liquid', d['s_f'], d['T']),
                         charts.series('saturated vapour', d['s_g'], d['T'])])


def _ph_chart(values):
    """P-h (log P) diagram from the saturation-dome points."""
    d = values['dome']
    return charts.chart('P-h diagram', 'h [J/kg]', 'P [Pa]',
                        [charts.series('saturated liquid', d['h_f'], d['P']),
                         charts.series('saturated vapour', d['h_g'], d['P'])], ylog=True)


# Reusable declarations -----------------------------------------------------------------------------
def gamma_(desc='specific-heat ratio cp/cv'):
    return V('gamma', '', desc, lo=1.0, hi=2.0, label='γ')


def R_(desc='specific gas constant'):
    return V('R', 'J/(kg*K)', desc, lo=1.0, hi=1e4, label='R')


def fluid_(desc='CoolProp fluid name, e.g. Water, R134a, Air'):
    return V('fluid', '', desc, kind='text')


CALCS = [
    # ---- basic properties ------------------------------------------------------------------------
    Calc('ideal_gas_law', 'Ideal gas law (P v = R T)',
         vars=[V('P', 'Pa', 'pressure', lo=0.0, hi=1e8), V('v', 'm^3/kg', 'specific volume', lo=0.0, hi=1e3),
               R_(), V('T', 'K', 'absolute temperature', lo=0.0, hi=1e4)],
         equations=['P*v = R*T'], outputs=['P'],
         example={'v': 0.847, 'R': 287.0, 'T': 288.15},
         keywords=['equation of state', 'specific volume', 'density'],
         assumptions=['Ideal gas. Density rho = 1/v.'],
         refs=['Ideal gas law P v = R T']),

    Calc('compressibility_factor', 'Compressibility factor (Z = P v / (R T))',
         vars=[V('Z', '', 'compressibility factor', lo=0.0, hi=1e2),
               V('P', 'Pa', 'pressure', lo=0.0, hi=1e8), V('v', 'm^3/kg', 'specific volume', lo=0.0, hi=1e3),
               R_(), V('T', 'K', 'absolute temperature', lo=0.0, hi=1e4)],
         equations=['Z = P*v/(R*T)'], outputs=['Z'],
         example={'P': 10e6, 'v': 0.005, 'R': 287.0, 'T': 300.0},
         keywords=['real gas', 'deviation'],
         assumptions=['Z = 1 for an ideal gas; departures measure real-gas behaviour.'],
         refs=['Z = P v / (R T)']),

    Calc('specific_heats', 'Specific heats from R and gamma',
         vars=[V('cp', 'J/(kg*K)', 'specific heat at constant pressure', lo=1.0, hi=1e5),
               V('cv', 'J/(kg*K)', 'specific heat at constant volume', lo=1.0, hi=1e5), R_(), gamma_()],
         equations=['cp = gamma*R/(gamma - 1)', 'cv = R/(gamma - 1)'], outputs=['cp', 'cv'],
         example={'R': 287.0, 'gamma': 1.4},
         keywords=['cp', 'cv', 'mayer'],
         assumptions=['Ideal gas, constant specific heats: cp - cv = R and gamma = cp/cv.'],
         refs=['cp = gamma R/(gamma-1); cv = R/(gamma-1)']),

    Calc('ideal_gas_energy', 'Ideal-gas internal energy and enthalpy changes',
         vars=[V('delta_u', 'J/kg', 'change in specific internal energy', signed=True, label='Δu'),
               V('delta_h', 'J/kg', 'change in specific enthalpy', signed=True, label='Δh'),
               V('cv', 'J/(kg*K)', 'specific heat at constant volume', lo=1.0, hi=1e5),
               V('cp', 'J/(kg*K)', 'specific heat at constant pressure', lo=1.0, hi=1e5),
               V('delta_T', 'K', 'temperature change', signed=True, label='ΔT')],
         equations=['delta_u = cv*delta_T', 'delta_h = cp*delta_T'], outputs=['delta_u', 'delta_h'],
         example={'cv': 718.0, 'cp': 1005.0, 'delta_T': 100.0},
         keywords=['internal energy', 'enthalpy', 'temperature change'],
         assumptions=['Ideal gas, constant specific heats. Enter delta_T in K or delta_degC.'],
         refs=['du = cv dT; dh = cp dT (ideal gas)']),

    Calc('ideal_gas_entropy', 'Ideal-gas entropy change (constant specific heats)',
         vars=[V('delta_s', 'J/(kg*K)', 'change in specific entropy', signed=True, label='Δs'),
               V('cp', 'J/(kg*K)', 'specific heat at constant pressure', lo=1.0, hi=1e5),
               R_(), V('T1', 'K', 'initial temperature', lo=0.0, hi=1e4),
               V('T2', 'K', 'final temperature', lo=0.0, hi=1e4),
               V('P1', 'Pa', 'initial pressure', lo=0.0, hi=1e8),
               V('P2', 'Pa', 'final pressure', lo=0.0, hi=1e8)],
         equations=['delta_s = cp*log(T2/T1) - R*log(P2/P1)'], outputs=['delta_s'],
         example={'cp': 1005.0, 'R': 287.0, 'T1': 300.0, 'T2': 600.0, 'P1': 100e3, 'P2': 200e3},
         keywords=['second law', 'entropy change'],
         assumptions=['Ideal gas, constant specific heats (second T-ds / Maxwell relation, integrated).'],
         refs=['ds = cp dT/T - R dP/P (ideal gas)']),

    # ---- real fluids (common.fluid_state / common.saturation) ------------------------------------
    Calc('fluid_state_TP', 'Real-fluid state at given T and P',
         vars=[fluid_(), V('T', 'K', 'temperature', lo=50.0, hi=1500.0),
               V('P', 'Pa', 'pressure', lo=1e4, hi=1e7),
               V('rho', 'kg/m^3', 'density', lo=0.0, hi=1e4),
               V('h', 'J/kg', 'specific enthalpy (signed: negative at low T)', signed=True),
               V('s', 'J/(kg*K)', 'specific entropy', signed=True),
               V('u', 'J/kg', 'specific internal energy (signed: negative at low T)', signed=True),
               V('cp', 'J/(kg*K)', 'cp', lo=1.0, hi=1e6), V('cv', 'J/(kg*K)', 'cv', lo=1.0, hi=1e6)],
         procedure=_fluid_state_TP, outputs=['rho', 'h', 's', 'u', 'cp', 'cv'],
         example={'fluid': 'Air', 'T': 300.0, 'P': 101325.0},
         keywords=['coolprop', 'steam tables', 'refrigerant', 'equation of state'],
         assumptions=['Full CoolProp state from T and P (u = h - P/rho). For a liquid the density varies '
                      'only weakly with T; a gas example is used so every input is invertible.'],
         refs=['CoolProp reference equation of state (via common.fluid_state)']),

    Calc('fluid_state_TQ', 'Real-fluid two-phase state at given T and quality',
         vars=[fluid_(), V('T', 'K', 'saturation temperature (inside the fluid saturation range)', lo=50.0, hi=700.0),
               V('Q', '', 'quality (vapour mass fraction, 0..1)', lo=0.0, hi=1.0),
               V('h', 'J/kg', 'mixture specific enthalpy', signed=True),
               V('P', 'Pa', 'saturation pressure', lo=0.0, hi=1e8),
               V('rho', 'kg/m^3', 'mixture density', lo=0.0, hi=1e4),
               V('s', 'J/(kg*K)', 'mixture specific entropy', signed=True),
               V('u', 'J/kg', 'mixture specific internal energy', signed=True)],
         procedure=_fluid_state_TQ, outputs=['h', 'P', 'rho', 's', 'u'],
         example={'fluid': 'Water', 'T': 373.15, 'Q': 0.5},
         keywords=['quality', 'wet vapour', 'saturated mixture'],
         assumptions=['Two-phase state from saturation temperature T and quality Q via common.fluid_state '
                      '(CoolProp). h = h_f + Q h_fg.'],
         refs=['Lever rule h = h_f + Q (h_g - h_f) via common.fluid_state']),

    Calc('mixture_quality', 'Two-phase quality from a specific-volume lever rule',
         vars=[V('x', '', 'quality', lo=0.0, hi=1.0),
               V('v', 'm^3/kg', 'mixture specific volume', lo=0.0, hi=1e3),
               V('v_f', 'm^3/kg', 'saturated-liquid specific volume', lo=0.0, hi=1e3),
               V('v_g', 'm^3/kg', 'saturated-vapour specific volume', lo=0.0, hi=1e3)],
         equations=['x = (v - v_f)/(v_g - v_f)'], outputs=['x'],
         example={'v': 0.5, 'v_f': 0.001044, 'v_g': 1.6940},
         keywords=['lever rule', 'wet steam'],
         assumptions=['v_f and v_g are the saturated liquid and vapour values at the mixture pressure or '
                      'temperature (from tables or fluid_state_TQ). The same lever rule applies to h, s and u.'],
         refs=['x = (v - v_f)/(v_g - v_f)']),

    # ---- boundary work ---------------------------------------------------------------------------
    Calc('boundary_work_isobaric', 'Boundary work of a constant-pressure process',
         vars=[V('W', 'J', 'boundary work (positive = expansion)', signed=True),
               V('P', 'Pa', 'constant pressure', lo=0.0, hi=1e8),
               V('V1', 'm^3', 'initial volume', lo=0.0, hi=1e4),
               V('V2', 'm^3', 'final volume', lo=0.0, hi=1e4)],
         equations=['W = P*(V2 - V1)'], outputs=['W'],
         example={'P': 2e5, 'V1': 0.1, 'V2': 0.3},
         keywords=['isobaric', 'expansion', 'compression'],
         refs=['W = P (V2 - V1)']),

    Calc('boundary_work_isothermal', 'Boundary work of an isothermal ideal-gas process',
         vars=[V('W', 'J', 'boundary work (positive = expansion)', signed=True),
               V('P1', 'Pa', 'initial pressure', lo=0.0, hi=1e8),
               V('V1', 'm^3', 'initial volume', lo=0.0, hi=1e4),
               V('V2', 'm^3', 'final volume', lo=0.0, hi=1e4)],
         procedure=_boundary_work_isothermal, outputs=['W'],
         example={'P1': 100e3, 'V1': 0.5, 'V2': 1.0},
         keywords=['isothermal', 'ideal gas'],
         assumptions=['Ideal gas, constant temperature (W = P1 V1 ln(V2/V1) = m R T ln(V2/V1)). Inverting for '
                      'V1 has two roots (the other is on the compression side); the root nearest the example '
                      'is returned.'],
         refs=['W = P1 V1 ln(V2/V1)']),

    Calc('boundary_work_polytropic', 'Boundary work of a polytropic process (P V^n = const)',
         vars=[V('W', 'J', 'boundary work', signed=True),
               V('P1', 'Pa', 'initial pressure', lo=0.0, hi=1e8),
               V('V1', 'm^3', 'initial volume', lo=0.0, hi=1e4),
               V('P2', 'Pa', 'final pressure', lo=0.0, hi=1e8),
               V('V2', 'm^3', 'final volume', lo=0.0, hi=1e4),
               V('n', '', 'polytropic index (n != 1)', lo=0.0, hi=5.0)],
         equations=['W = (P2*V2 - P1*V1)/(1 - n)'], outputs=['W'],
         example={'P1': 100e3, 'V1': 0.1, 'P2': 400e3, 'V2': 0.035, 'n': 1.3},
         keywords=['polytropic', 'pv^n'],
         assumptions=['P V^n = constant. n = 1 is the isothermal case (use boundary_work_isothermal); '
                      'n = 0 isobaric; n = gamma isentropic (ideal gas).'],
         refs=['W = (P2 V2 - P1 V1)/(1 - n)']),

    # ---- first law -------------------------------------------------------------------------------
    Calc('first_law_closed', 'First law for a closed system (Q - W = dU)',
         vars=[V('Q', 'J', 'heat into the system', signed=True),
               V('W', 'J', 'work out of the system', signed=True),
               V('delta_U', 'J', 'change in internal energy', signed=True, label='ΔU')],
         equations=['Q = W + delta_U'], outputs=['Q'],
         example={'W': 50e3, 'delta_U': 25e3},
         keywords=['energy balance', 'closed system'],
         assumptions=['Q in and W out are positive.'],
         refs=['Q - W = dU']),

    Calc('cv_energy_balance', 'Steady-flow energy balance per unit mass (control volume)',
         vars=[V('q', 'J/kg', 'heat transfer per unit mass', signed=True),
               V('w', 'J/kg', 'shaft work per unit mass (positive = out)', signed=True),
               V('h1', 'J/kg', 'inlet enthalpy', signed=True),
               V('h2', 'J/kg', 'outlet enthalpy', signed=True),
               V('V1', 'm/s', 'inlet velocity', lo=0.0, hi=1e4),
               V('V2', 'm/s', 'outlet velocity', lo=0.0, hi=1e4),
               V('g', 'm/s^2', 'gravitational acceleration (9.80665 m/s^2)', lo=0.0, hi=1e3),
               V('z1', 'm', 'inlet elevation', signed=True),
               V('z2', 'm', 'outlet elevation', signed=True)],
         equations=['q = w + (h2 - h1) + (V2**2 - V1**2)/2 + g*(z2 - z1)'], outputs=['q'],
         example={'w': 100e3, 'h1': 3.0e6, 'h2': 2.9e6, 'V1': 10.0, 'V2': 50.0, 'g': 9.80665, 'z1': 2.0, 'z2': 7.0},
         keywords=['control volume', 'steady flow', 'energy equation'],
         assumptions=['Steady flow, one inlet and one outlet, per unit mass. Neglect a term by setting its '
                      'pair equal (V1 = V2 or z1 = z2).'],
         refs=['q - w = (h2 - h1) + (V2^2 - V1^2)/2 + g (z2 - z1)']),

    Calc('nozzle_diffuser', 'Nozzle/diffuser exit velocity (adiabatic, no work)',
         vars=[V('V2', 'm/s', 'exit velocity', lo=0.0, hi=1e5),
               V('V1', 'm/s', 'inlet velocity', lo=0.0, hi=1e5),
               V('h1', 'J/kg', 'inlet enthalpy', signed=True, lo=-1e8, hi=1e8),
               V('h2', 'J/kg', 'exit enthalpy', signed=True, lo=-1e8, hi=1e8)],
         equations=['V2 = sqrt(V1**2 + 2*(h1 - h2))'], outputs=['V2'],
         example={'V1': 30.0, 'h1': 3000e3, 'h2': 2800e3},
         keywords=['nozzle', 'diffuser', 'kinetic energy'],
         assumptions=['Adiabatic, no shaft work, no elevation change: V2 = sqrt(V1^2 + 2(h1 - h2)). A nozzle '
                      'lowers h to raise V; a diffuser raises h to lower V.'],
         refs=['h1 + V1^2/2 = h2 + V2^2/2']),

    Calc('turbine_work', 'Turbine work output (first law)',
         vars=[V('w_out', 'J/kg', 'turbine work output per unit mass', lo=0.0, hi=1e8),
               V('h1', 'J/kg', 'inlet enthalpy', signed=True, lo=-1e8, hi=1e8),
               V('h2', 'J/kg', 'exit enthalpy', signed=True, lo=-1e8, hi=1e8)],
         equations=['w_out = h1 - h2'], outputs=['w_out'],
         example={'h1': 3200e3, 'h2': 2400e3},
         keywords=['expansion', 'power'],
         assumptions=['Adiabatic turbine, kinetic and potential changes neglected.'],
         refs=['w_out = h1 - h2 (turbine)']),

    Calc('compressor_work', 'Compressor/pump work input (first law)',
         vars=[V('w_in', 'J/kg', 'work input per unit mass', lo=0.0, hi=1e8),
               V('h1', 'J/kg', 'inlet enthalpy', signed=True, lo=-1e8, hi=1e8),
               V('h2', 'J/kg', 'exit enthalpy', signed=True, lo=-1e8, hi=1e8)],
         equations=['w_in = h2 - h1'], outputs=['w_in'],
         example={'h1': 300e3, 'h2': 500e3},
         keywords=['pump', 'compression'],
         assumptions=['Adiabatic compressor/pump, kinetic and potential changes neglected.'],
         refs=['w_in = h2 - h1 (compressor/pump)']),

    Calc('throttle', 'Throttling (adiabatic) valve: enthalpy is constant',
         vars=[V('h1', 'J/kg', 'upstream enthalpy', signed=True, lo=-1e8, hi=1e8),
               V('h2', 'J/kg', 'downstream enthalpy', signed=True, lo=-1e8, hi=1e8)],
         equations=['h2 = h1'], outputs=['h2'],
         example={'h1': 2800e3},
         keywords=['throttle', 'isenthalpic', 'expansion valve'],
         assumptions=['Adiabatic throttle, no work, kinetic and potential changes neglected: h2 = h1.'],
         refs=['h2 = h1 (throttling)']),

    Calc('mixing_chamber', 'Adiabatic mixing chamber outlet enthalpy',
         vars=[V('h_mix', 'J/kg', 'outlet mixture enthalpy', signed=True, lo=-1e8, hi=1e8),
               V('mdot1', 'kg/s', 'mass flow of stream 1', lo=0.0, hi=1e4),
               V('h1', 'J/kg', 'enthalpy of stream 1', signed=True, lo=-1e8, hi=1e8),
               V('mdot2', 'kg/s', 'mass flow of stream 2', lo=0.0, hi=1e4),
               V('h2', 'J/kg', 'enthalpy of stream 2', signed=True, lo=-1e8, hi=1e8)],
         equations=['h_mix = (mdot1*h1 + mdot2*h2)/(mdot1 + mdot2)'], outputs=['h_mix'],
         example={'mdot1': 1.0, 'h1': 500e3, 'mdot2': 2.0, 'h2': 200e3},
         keywords=['mixing', 'feedwater heater'],
         assumptions=['Adiabatic mixing, no work, kinetic and potential changes neglected; mass balance '
                      'm3 = mdot1 + mdot2.'],
         refs=['mdot1 h1 + mdot2 h2 = (mdot1 + mdot2) h_mix']),

    # ---- transient -------------------------------------------------------------------------------
    Calc('transient_filling', 'Adiabatic charging of an evacuated rigid tank from a line',
         vars=[V('T2', 'K', 'final tank temperature', lo=0.0, hi=1e4), gamma_(),
               V('T_line', 'K', 'supply-line temperature', lo=0.0, hi=1e4)],
         equations=['T2 = gamma*T_line'], outputs=['T2'],
         example={'gamma': 1.4, 'T_line': 300.0},
         keywords=['charging', 'unsteady', 'filling'],
         assumptions=['Initially evacuated rigid tank filled adiabatically from an ideal-gas line at '
                      'constant T_line, cp/cv constant: the inflowing enthalpy shows up as internal energy, '
                      'so u2 = h_line and T2 = gamma T_line.'],
         refs=['Adiabatic charging of an evacuated tank: T2 = gamma T_line (landmark)']),

    Calc('transient_discharge', 'Reversible adiabatic blowdown of a rigid tank',
         vars=[V('T2', 'K', 'final temperature of the gas remaining in the tank', lo=0.0, hi=1e4),
               V('T1', 'K', 'initial temperature', lo=0.0, hi=1e4),
               V('P1', 'Pa', 'initial pressure', lo=0.0, hi=1e8),
               V('P2', 'Pa', 'final pressure', lo=0.0, hi=1e8), gamma_()],
         equations=['T2 = T1*(P2/P1)**((gamma - 1)/gamma)'], outputs=['T2'],
         example={'T1': 300.0, 'P1': 800e3, 'P2': 100e3, 'gamma': 1.4},
         keywords=['blowdown', 'unsteady', 'discharge'],
         assumptions=['Slow (reversible, adiabatic) discharge: the gas remaining in the rigid tank expands '
                      'isentropically, T2 = T1 (P2/P1)^((gamma-1)/gamma).'],
         refs=['Isentropic expansion of the gas remaining in the tank']),

    # ---- cycles and second law -------------------------------------------------------------------
    Calc('thermal_efficiency', 'Thermal efficiency',
         vars=[V('eta_th', '', 'thermal efficiency', hi=1.0, label='ηth'),
               V('W_net', 'J', 'net work output', lo=0.0, hi=1e9),
               V('Q_in', 'J', 'heat input', lo=0.0, hi=1e9)],
         equations=['eta_th = W_net/Q_in'], outputs=['eta_th'],
         example={'W_net': 400e3, 'Q_in': 1000e3},
         keywords=['heat engine', 'cycle efficiency'],
         refs=['eta_th = W_net/Q_in']),

    Calc('carnot_efficiency', 'Carnot efficiency',
         vars=[V('eta_carnot', '', 'Carnot efficiency', hi=1.0, label='ηc'),
               V('T_L', 'K', 'low (sink) temperature', lo=0.0, hi=1e4),
               V('T_H', 'K', 'high (source) temperature', lo=0.0, hi=1e4)],
         equations=['eta_carnot = 1 - T_L/T_H'], outputs=['eta_carnot'],
         example={'T_L': 300.0, 'T_H': 600.0},
         keywords=['carnot', 'reversible', 'maximum efficiency'],
         assumptions=['Reversible cycle between T_H and T_L.'],
         refs=['Carnot efficiency eta = 1 - T_L/T_H']),

    Calc('cop', 'Coefficient of performance (refrigerator and heat pump)',
         vars=[V('COP_ref', '', 'refrigerator COP (Q_L/W_in)'), V('COP_hp', '', 'heat-pump COP (Q_H/W_in)'),
               V('Q_L', 'J', 'heat removed from the cold space', lo=0.0, hi=1e9),
               V('Q_H', 'J', 'heat delivered to the hot space', lo=0.0, hi=1e9),
               V('W_in', 'J', 'work input', lo=0.0, hi=1e9)],
         equations=['COP_ref = Q_L/W_in', 'COP_hp = Q_H/W_in'], outputs=['COP_ref', 'COP_hp'],
         example={'Q_L': 2000e3, 'Q_H': 2500e3, 'W_in': 500e3},
         keywords=['refrigerator', 'heat pump'],
         assumptions=['COP_ref = Q_L/W_in; COP_hp = Q_H/W_in. The first law gives Q_H = Q_L + W_in for any '
                      'cycle (not only a reversible one), so COP_hp = COP_ref + 1 holds generally.'],
         refs=['COP_ref = Q_L/W_in; COP_hp = Q_H/W_in']),

    Calc('cop_carnot', 'Carnot COPs of a refrigerator and heat pump',
         vars=[V('COP_ref_max', '', 'Carnot refrigerator COP T_L/(T_H - T_L)'),
               V('COP_hp_max', '', 'Carnot heat-pump COP T_H/(T_H - T_L)'),
               V('T_L', 'K', 'low (cold) temperature', lo=0.0, hi=1e4),
               V('T_H', 'K', 'high (hot) temperature', lo=0.0, hi=1e4)],
         equations=['COP_ref_max = T_L/(T_H - T_L)', 'COP_hp_max = T_H/(T_H - T_L)'], outputs=['COP_ref_max', 'COP_hp_max'],
         example={'T_L': 268.15, 'T_H': 308.15},
         keywords=['carnot', 'refrigerator', 'heat pump'],
         assumptions=['Reversible cycle.'],
         refs=['Carnot COPs: T_L/(T_H - T_L) and T_H/(T_H - T_L)']),

    Calc('entropy_generation', 'Entropy generation (per unit mass)',
         vars=[V('s_gen', 'J/(kg*K)', 'entropy generation', signed=True),
               V('s1', 'J/(kg*K)', 'initial specific entropy', signed=True),
               V('s2', 'J/(kg*K)', 'final specific entropy', signed=True),
               V('q', 'J/kg', 'heat transfer into the system per unit mass', signed=True),
               V('T_b', 'K', 'boundary temperature at which q is transferred', lo=0.0, hi=1e4)],
         equations=['s_gen = (s2 - s1) - q/T_b'], outputs=['s_gen'],
         example={'s1': 1000.0, 's2': 1500.0, 'q': 100e3, 'T_b': 400.0},
         keywords=['second law', 'irreversibility', 'entropy balance'],
         assumptions=['Closed system: s_gen = (s2 - s1) - q/T_b >= 0 for a real process (negative means the '
                      'process is impossible).'],
         refs=['Entropy balance: s_gen = (s2 - s1) - q/T_b']),

    Calc('isentropic_efficiency_turbine', 'Turbine isentropic efficiency',
         vars=[V('eta_t', '', 'turbine isentropic efficiency', hi=1.0, label='ηt'),
               V('h1', 'J/kg', 'inlet enthalpy', signed=True, lo=-1e8, hi=1e8),
               V('h2a', 'J/kg', 'actual exit enthalpy', signed=True, lo=-1e8, hi=1e8),
               V('h2s', 'J/kg', 'isentropic exit enthalpy', signed=True, lo=-1e8, hi=1e8)],
         equations=['eta_t = (h1 - h2a)/(h1 - h2s)'], outputs=['eta_t'],
         example={'h1': 3200e3, 'h2a': 2400e3, 'h2s': 2200e3},
         keywords=['expansion', 'second law efficiency'],
         assumptions=['Expansion (h1 > h2a > h2s): eta_t = actual work / isentropic work.'],
         refs=['Turbine isentropic efficiency eta_t = (h1 - h2a)/(h1 - h2s)']),

    Calc('isentropic_efficiency_compressor', 'Compressor isentropic efficiency',
         vars=[V('eta_c', '', 'compressor isentropic efficiency', hi=1.0, label='ηc'),
               V('h1', 'J/kg', 'inlet enthalpy', signed=True, lo=-1e8, hi=1e8),
               V('h2s', 'J/kg', 'isentropic exit enthalpy', signed=True, lo=-1e8, hi=1e8),
               V('h2a', 'J/kg', 'actual exit enthalpy', signed=True, lo=-1e8, hi=1e8)],
         equations=['eta_c = (h2s - h1)/(h2a - h1)'], outputs=['eta_c'],
         example={'h1': 300e3, 'h2s': 500e3, 'h2a': 550e3},
         keywords=['compression', 'second law efficiency'],
         assumptions=['Compression (h1 < h2s < h2a): eta_c = isentropic work / actual work.'],
         refs=['Compressor isentropic efficiency eta_c = (h2s - h1)/(h2a - h1)']),

    Calc('exergy', 'Specific flow exergy (with kinetic and potential terms)',
         vars=[V('psi', 'J/kg', 'specific flow exergy', signed=True, label='ψ'),
               V('h', 'J/kg', 'specific enthalpy', signed=True),
               V('h0', 'J/kg', 'dead-state enthalpy', signed=True),
               V('T0', 'K', 'dead-state temperature', lo=0.0, hi=1e4),
               V('s', 'J/(kg*K)', 'specific entropy', signed=True),
               V('s0', 'J/(kg*K)', 'dead-state entropy', signed=True),
               V('V', 'm/s', 'velocity (0 if negligible)', lo=0.0, hi=1e5),
               V('z', 'm', 'elevation above the dead state (0 if negligible)', signed=True)],
         procedure=_exergy, outputs=['psi'],
         example={'h': 3000e3, 'h0': 100e3, 'T0': 298.15, 's': 7000.0, 's0': 360.0, 'V': 0.0, 'z': 0.0},
         keywords=['availability', 'exergy'],
         assumptions=['Specific flow exergy psi = (h - h0) - T0 (s - s0) + V^2/2 + g z, with the potential '
                      'term using standard gravity g = 9.80665 m/s^2 (common.g0()). Set V = 0 and z = 0 to '
                      'drop the kinetic/potential terms.'],
         refs=['psi = (h - h0) - T0 (s - s0) + V^2/2 + g z']),

    Calc('exergy_destruction', 'Exergy destruction (Gouy-Stodola)',
         vars=[V('x_destroyed', 'J/kg', 'exergy destroyed per unit mass', lo=0.0, hi=1e8),
               V('T0', 'K', 'dead-state temperature', lo=0.0, hi=1e4),
               V('s_gen', 'J/(kg*K)', 'entropy generation', lo=0.0, hi=1e5)],
         equations=['x_destroyed = T0*s_gen'], outputs=['x_destroyed'],
         example={'T0': 298.15, 's_gen': 200.0},
         keywords=['irreversibility', 'gouy-stodola'],
         refs=['Exergy destroyed = T0 s_gen (Gouy-Stodola)']),

    Calc('second_law_efficiency', 'Second-law efficiency',
         vars=[V('eta_II', '', 'second-law efficiency', hi=1.0, label='ηII'),
               V('x_destroyed', 'J/kg', 'exergy destroyed', lo=0.0, hi=1e8),
               V('x_supplied', 'J/kg', 'exergy supplied', lo=0.0, hi=1e8)],
         equations=['eta_II = 1 - x_destroyed/x_supplied'], outputs=['eta_II'],
         example={'x_destroyed': 30e3, 'x_supplied': 100e3},
         keywords=['exergetic efficiency', 'availability'],
         assumptions=['eta_II = 1 - (exergy destroyed)/(exergy supplied).'],
         refs=['Second-law efficiency eta_II = 1 - x_destroyed/x_supplied']),

    # ---- mixtures --------------------------------------------------------------------------------
    Calc('partial_pressure', 'Partial pressure of a component (Dalton)',
         vars=[V('p_i', 'Pa', 'partial pressure of component i', lo=0.0, hi=1e8),
               V('y_i', '', 'mole fraction of component i', lo=0.0, hi=1.0),
               V('P', 'Pa', 'total mixture pressure', lo=0.0, hi=1e8)],
         equations=['p_i = y_i*P'], outputs=['p_i'],
         example={'y_i': 0.21, 'P': 101325.0},
         keywords=['dalton', 'partial pressure'],
         refs=["Dalton's law: p_i = y_i P"]),

    Calc('gas_mixture', 'Gas-mixture mole/mass fractions and molar mass',
         vars=[V('mf1', '', 'mass fraction of component 1', lo=0.0, hi=1.0),
               V('M_mix', 'kg/kmol', 'mixture molar mass', lo=1.0, hi=1e3),
               V('y1', '', 'mole fraction of component 1', lo=0.0, hi=1.0),
               V('y2', '', 'mole fraction of component 2', lo=0.0, hi=1.0),
               V('M1', 'kg/kmol', 'molar mass of component 1', lo=1.0, hi=1e3),
               V('M2', 'kg/kmol', 'molar mass of component 2', lo=1.0, hi=1e3)],
         equations=['mf1 = y1*M1/(y1*M1 + y2*M2)', 'M_mix = y1*M1 + y2*M2'],
         outputs=['mf1', 'M_mix'], example={'y1': 0.21, 'y2': 0.79, 'M1': 32.0, 'M2': 28.0},
         keywords=['mole fraction', 'mass fraction', 'molar mass'],
         assumptions=['Binary mixture. mf1 = (y1 M1)/(y1 M1 + y2 M2); M_mix = y1 M1 + y2 M2. The mixture gas '
                      'constant is R_mix = 8314.5/M_mix [J/(kg K)].'],
         refs=['Mole/mass fraction conversion and mixture molar mass']),

    # ---- Maxwell relations, Clapeyron, Joule-Thomson --------------------------------------------
    Calc('clapeyron', 'Clapeyron equation (saturation pressure slope)',
         vars=[V('dPdT', 'Pa/K', 'saturation pressure-temperature slope', lo=0.0, hi=1e9),
               V('s_g', 'J/(kg*K)', 'saturated-vapour entropy', lo=0.0, hi=1e5),
               V('s_f', 'J/(kg*K)', 'saturated-liquid entropy', lo=0.0, hi=1e5),
               V('v_g', 'm^3/kg', 'saturated-vapour specific volume', lo=0.0, hi=1e3),
               V('v_f', 'm^3/kg', 'saturated-liquid specific volume', lo=0.0, hi=1e3)],
         equations=['dPdT = (s_g - s_f)/(v_g - v_f)'], outputs=['dPdT'],
         example={'s_g': 7354.1, 's_f': 1307.2, 'v_g': 1.6940, 'v_f': 0.001044},
         keywords=['clausius-clapeyron', 'saturation', 'maxwell'],
         assumptions=['This is the Maxwell relation (dP/dT)_v = (ds/dv)_T evaluated along the saturation '
                      'curve, equal to h_fg/(T v_fg). The four Maxwell relations are '
                      '(dT/dv)_s = -(dP/ds)_v, (dT/dP)_s = (dv/ds)_P, (dP/dT)_v = (ds/dv)_T, '
                      '(dv/dT)_P = -(ds/dP)_T.'],
         refs=['Clapeyron equation / Maxwell relation along saturation']),

    Calc('clausius_clapeyron', 'Clausius-Clapeyron equation (ideal-gas vapour)',
         vars=[V('P2', 'Pa', 'vapour pressure at T2', lo=0.0, hi=1e8),
               V('P1', 'Pa', 'vapour pressure at T1', lo=0.0, hi=1e8),
               V('h_fg', 'J/kg', 'latent heat of vaporisation', signed=True, lo=-1e7, hi=1e7),
               R_(), V('T1', 'K', 'temperature at state 1', lo=0.0, hi=1e4),
               V('T2', 'K', 'temperature at state 2', lo=0.0, hi=1e4)],
         equations=['P2 = P1*exp((h_fg/R)*(1/T1 - 1/T2))'], outputs=['P2'],
         example={'P1': 101325.0, 'h_fg': 2.256e6, 'R': 461.5, 'T1': 373.15, 'T2': 383.15},
         keywords=['vapour pressure', 'latent heat'],
         assumptions=['Ideal-gas vapour with v_g >> v_f and constant h_fg.'],
         refs=['Clausius-Clapeyron: ln(P2/P1) = (h_fg/R)(1/T1 - 1/T2)']),

    Calc('joule_thomson', 'Joule-Thomson coefficient (finite difference on enthalpy)',
         vars=[fluid_(), V('T', 'K', 'temperature', lo=250.0, hi=500.0),
               V('P', 'Pa', 'pressure', lo=1e4, hi=1e6),
               V('mu', 'K/Pa', 'Joule-Thomson coefficient (dT/dP)_h', signed=True, label='μ'),
               V('cp', 'J/(kg*K)', 'cp from the enthalpy finite difference', lo=1.0, hi=1e6)],
         procedure=_joule_thomson, outputs=['mu', 'cp'],
         example={'fluid': 'Air', 'T': 300.0, 'P': 101325.0},
         keywords=['throttling', 'inversion temperature', 'isenthalpic'],
         assumptions=['mu = (dT/dP)_h = -(dh/dP)_T/(dh/dT)_P, evaluated by central finite difference on '
                      'common.fluid_state (CoolProp) with dT = 0.5 K and dP = 1000 Pa, so it is a numerical '
                      'derivative, stated as such. mu > 0 cools on throttling; mu < 0 heats.'],
         refs=['Joule-Thomson coefficient by finite difference on h (CoolProp states)']),

    Calc('fluid_saturation_dome', 'Saturation curve (T-s / P-h dome) from common.saturation',
         vars=[fluid_(), V('T_range', '', 'saturation temperature range [T_min (K), T_max (K)]', kind='json'),
               V('P_min', 'Pa', 'saturation pressure at T_min', lo=0.0, hi=1e8),
               V('P_max', 'Pa', 'saturation pressure at T_max', lo=0.0, hi=1e8),
               V('dome', '', 'arrays T, P, s_f, s_g, h_f, h_g (all SI) along the saturation curve', kind='json')],
         procedure=_saturation_dome, outputs=['P_min', 'P_max', 'dome'],
         example={'fluid': 'Water', 'T_range': [300.0, 600.0]},
         plots=[{'id': 'ts', 'title': 'T-s diagram', 'fn': _ts_chart},
                {'id': 'ph', 'title': 'P-h diagram', 'fn': _ph_chart}],
         keywords=['steam tables', 'saturation curve', 'dome', 'ts', 'ph'],
         assumptions=['Saturated liquid and vapour states from common.saturation (CoolProp) at 40 evenly '
                      'spaced temperatures in [T_min, T_max]; T_max must be below the critical temperature.'],
         refs=['CoolProp saturation curve (via common.saturation)']),
]

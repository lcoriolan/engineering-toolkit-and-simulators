# DESCRIPTION: Power and refrigeration cycles section: air-standard Otto/Diesel/Dual/Stirling/Brayton,
#              gas-turbine modifications (regeneration, intercooling, reheat), jet propulsion, Rankine
#              (ideal, reheat, regenerative), combined/cogeneration, vapour-compression refrigeration
#              (single-stage, cascade, multistage), absorption and gas refrigeration.
"""Power and refrigeration cycle calculators.

Sign/unit conventions: absolute temperatures in K, pressures in Pa, specific enthalpies in J/kg,
specific entropies in J/(kg·K), specific works and heats in J/kg, cycle efficiencies dimensionless
(0..1). Air-standard cycles use ideal gas with constant specific-heat ratio γ = cp/cv; closed forms are
quoted per unit mass. Rankine and vapour-compression cycles are procedures that read state points from
``engineering_sim.science.mech.common`` (CoolProp / IAPWS-95, REFPROP-equivalent) — no steam or refrigerant tables
are typed into this file.

Reference relations are the standard ones in thermodynamics texts (Cengel & Boles, *Thermodynamics:
An Engineering Approach*; Moran & Shapiro, *Fundamentals of Engineering Thermodynamics*). They are cited
by relation name, recalled rather than re-read, so no page or equation numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import common

SECTION = section('cycles', 'Power and refrigeration cycles', order=280,
                  summary='Air-standard Otto/Diesel/Dual/Stirling/Brayton cycles, gas-turbine '
                          'modifications, jet propulsion, Rankine and reheat/regenerative variants, '
                          'combined/cogeneration, and vapour-compression/absorption/gas refrigeration.')

PI = math.pi


def _gamma_(desc="specific-heat ratio cp/cv"):
    return V('gamma', '', desc, lo=1.0, label='γ')


def _refrigeration_tons(values):
    """Cooling capacity in watts from tons of refrigeration (1 ton = 3.51685 kW, ASHRAE)."""
    return {'Q_dot': float(values['tons']) * 3516.85}


# ---------------------------------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------------------------------
def _otto_full(values):
    """Air-standard Otto cycle per unit mass of working fluid (constant cv, cp).

    States: 1 BDC before compression, 2 TDC after isentropic compression, 3 after constant-volume heat
    addition q_in, 4 after isentropic expansion back to V1. Returns temperatures (K), pressures (Pa),
    net work (J/kg), thermal efficiency and mean effective pressure (Pa).
    """
    P1, T1, r = values['P1'], values['T1'], values['r']
    gamma, R = values['gamma'], values['R']
    q_in = values['q_in']
    cv = R / (gamma - 1.0)
    if r <= 1.0:
        raise MechError('r: compression ratio must be > 1')
    v1 = R * T1 / P1
    v2 = v1 / r
    T2 = T1 * r ** (gamma - 1.0)
    P2 = P1 * r ** gamma
    T3 = T2 + q_in / cv
    P3 = P2 * T3 / T2
    T4 = T3 / r ** (gamma - 1.0)
    P4 = P3 / r ** gamma
    q_out = cv * (T4 - T1)
    w_net = q_in - q_out
    eta = w_net / q_in
    mep = w_net / (v1 - v2)
    return {'T2': T2, 'T3': T3, 'T4': T4, 'P2': P2, 'P3': P3, 'P4': P4,
            'q_out': q_out, 'w_net': w_net, 'eta': eta, 'mep': mep}


def _reheat_turbine(values):
    """Two-stage ideal-gas turbine with reheat to T_max between the stages.

    a = (γ−1)/γ; T2s = T_max (P_int/P1)^a is the exit of stage 1, T4s = T_max (P2/P_int)^a of stage 2
    (both reheated to T_max at entry). w_single is the work of one turbine expanding P1 → P2 with no
    reheat, so w_extra is the work gained by reheating. Equal stage pressure ratios (P_int = sqrt(P1 P2))
    maximise w_two for a given T_max.
    """
    T_max, P1, P2, P_int = values['T_max'], values['P1'], values['P2'], values['P_int']
    gamma, cp = values['gamma'], values['cp']
    if not P1 > P_int > P2:
        raise MechError('need P1 > P_int > P2 (intermediate pressure between high and low)')
    a = (gamma - 1.0) / gamma
    T2s = T_max * (P_int / P1) ** a
    T4s = T_max * (P2 / P_int) ** a
    w_stage1 = cp * (T_max - T2s)
    w_stage2 = cp * (T_max - T4s)
    w_two = w_stage1 + w_stage2
    w_single = cp * T_max * (1.0 - (P2 / P1) ** a)
    return {'T2s': T2s, 'T4s': T4s, 'w_stage1': w_stage1, 'w_stage2': w_stage2,
            'w_two': w_two, 'w_single': w_single, 'w_extra': w_two - w_single}


def _vc_states(fluid, P_evap, P_cond):
    """Ideal vapour-compression states for one stage (isentropic compressor, throttling valve).

    Returns dict with sat_evap (saturation at evaporator), sat_cond (saturation at condenser),
    h1 = saturated-vapour enthalpy at P_evap, s1, h3 = saturated-liquid enthalpy at P_cond,
    h4 = h3 (throttling), h2s = isentropic-compression outlet enthalpy at P_cond. All SI.
    """
    sat_e = common.saturation(fluid, P=P_evap)
    sat_c = common.saturation(fluid, P=P_cond)
    h1, s1 = sat_e['vapour']['h'], sat_e['vapour']['s']
    h3 = sat_c['liquid']['h']
    h2s = common.fluid_state(fluid, P=P_cond, s=s1)['h']
    return {'sat_e': sat_e, 'sat_c': sat_c, 'h1': h1, 's1': s1, 'h3': h3, 'h4': h3, 'h2s': h2s}


def _vapor_compression(values):
    """Single-stage vapour-compression refrigeration cycle.

    States: 1 saturated vapour at the evaporator, 2 compressor outlet (isentropic to P_cond), 3 saturated
    liquid at the condenser, 4 throttled to the evaporator (h4 = h3). Refrigeration effect per kg is
    q_L = h1 − h4; compressor work per kg is w_c = h2s − h1 (ideal). With the mass-flow rate m_dot the
    capacity is Q_dot = m_dot q_L and the compressor power W_dot = m_dot w_c; COP = q_L / w_c.
    """
    fluid = values['fluid']
    P_evap, P_cond, m_dot = values['P_evap'], values['P_cond'], values['m_dot']
    if not P_cond > P_evap:
        raise MechError('P_cond must exceed P_evap')
    s = _vc_states(fluid, P_evap, P_cond)
    q_L = s['h1'] - s['h4']
    w_c = s['h2s'] - s['h1']
    if w_c <= 0.0:
        raise MechError('compressor work is not positive; check the pressure levels')
    return {'h1': s['h1'], 'h2': s['h2s'], 'h3': s['h3'], 'h4': s['h4'],
            'T_evap': s['sat_e']['T'], 'T_cond': s['sat_c']['T'],
            'q_L': q_L, 'w_c': w_c, 'cop': q_L / w_c,
            'Q_dot': m_dot * q_L, 'W_dot': m_dot * w_c}


def _turbine_inlet(fluid, P_boiler, T_boiler):
    """Turbine-inlet state: saturated vapour at T_sat(P_boiler) or superheated steam at T_boiler.

    Returns (h, s, T_sat). Raises MechError when T_boiler is meaningfully below the saturation
    temperature (a subcooled/two-phase turbine inlet) and when P_boiler has no saturation point
    (above the fluid critical pressure — supercritical cycles are not modelled).
    """
    try:
        sat_b = common.saturation(fluid, P=P_boiler)
    except MechError as exc:
        raise MechError(f'P_boiler has no saturation point (supercritical?): {exc}')
    T_sat = sat_b['T']
    tol = 1e-3 * T_sat  # forgiving window so a rounded T_sat input still selects saturated vapour
    if T_boiler < T_sat - tol:
        raise MechError(f'turbine inlet {T_boiler:.3g} K is below the saturation temperature '
                        f'{T_sat:.3g} K at P_boiler; use T_sat for saturated vapour or a higher '
                        f'temperature for superheated steam')
    if T_boiler <= T_sat + tol:
        return sat_b['vapour']['h'], sat_b['vapour']['s'], T_sat
    st = common.fluid_state(fluid, P=P_boiler, T=T_boiler)
    return st['h'], st['s'], T_sat


def _rankine(values):
    """Ideal (or actual) Rankine cycle from CoolProp state points.

    States: 1 saturated liquid at P_cond, 2 pump outlet (h2 = h1 + v1 ΔP / η_pump), 3 turbine inlet —
    saturated vapour at P_boiler when T_boiler ≈ T_sat(P_boiler), otherwise superheated to T_boiler —,
    4 turbine outlet (h4 = h3 − η_turb (h3 − h4s), h4s at P_cond with s = s3). Returns the state
    enthalpies, quality at the turbine outlet, works, heats, and thermal efficiency.
    """
    fluid = values['fluid']
    P_cond, P_boiler = values['P_cond'], values['P_boiler']
    T_boiler, eta_turb, eta_pump = values['T_boiler'], values['eta_turb'], values['eta_pump']
    if not P_boiler > P_cond:
        raise MechError('P_boiler must exceed P_cond')
    sat_c = common.saturation(fluid, P=P_cond)
    h1 = sat_c['liquid']['h']
    v1 = 1.0 / sat_c['liquid']['rho']
    T_cond = sat_c['T']
    h3, s3, T_sat_b = _turbine_inlet(fluid, P_boiler, T_boiler)
    w_pump = v1 * (P_boiler - P_cond) / eta_pump
    h2 = h1 + w_pump
    h4s = common.fluid_state(fluid, P=P_cond, s=s3)['h']
    w_turb = eta_turb * (h3 - h4s)
    h4 = h3 - w_turb
    hf, hg = sat_c['liquid']['h'], sat_c['vapour']['h']
    x4 = (h4 - hf) / (hg - hf)
    q_in = h3 - h2
    q_out = h4 - h1
    w_net = w_turb - w_pump
    eta = w_net / q_in
    return {'T_cond': T_cond, 'T_sat_boiler': T_sat_b, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4,
            'x4': x4, 'w_pump': w_pump, 'w_turb': w_turb, 'w_net': w_net,
            'q_in': q_in, 'q_out': q_out, 'eta': eta}


def _rankine_reheat(values):
    """Reheat Rankine cycle.

    1 sat. liquid at P_cond → 2 pump to P_boiler → 3 superheat to T_boiler at P_boiler → 4 HP turbine to
    P_reheat (s constant) → 5 reheat to T_reheat at P_reheat → 6 LP turbine to P_cond → 1. The HP/LP
    turbines share one isentropic efficiency η_turb; the pump uses η_pump. Returns enthalpy at every
    state, works, heats, LP-turbine-exit quality x6, and thermal efficiency.
    """
    fluid = values['fluid']
    P_cond, P_boiler, P_reheat = values['P_cond'], values['P_boiler'], values['P_reheat']
    T_boiler, T_reheat = values['T_boiler'], values['T_reheat']
    eta_turb, eta_pump = values['eta_turb'], values['eta_pump']
    if not P_boiler > P_reheat > P_cond:
        raise MechError('need P_boiler > P_reheat > P_cond')
    sat_c = common.saturation(fluid, P=P_cond)
    h1 = sat_c['liquid']['h']
    v1 = 1.0 / sat_c['liquid']['rho']
    sat_r = common.saturation(fluid, P=P_reheat)
    if T_reheat < sat_r['T'] - 1e-3 * sat_r['T']:
        raise MechError(f'reheat temperature {T_reheat:.3g} K is below the saturation temperature '
                        f'{sat_r["T"]:.3g} K at P_reheat; reheat must be saturated vapour or superheated')
    h3, s3, _ = _turbine_inlet(fluid, P_boiler, T_boiler)
    w_pump = v1 * (P_boiler - P_cond) / eta_pump
    h2 = h1 + w_pump
    h4s = common.fluid_state(fluid, P=P_reheat, s=s3)['h']
    h4 = h3 - eta_turb * (h3 - h4s)
    st5 = common.fluid_state(fluid, P=P_reheat, T=T_reheat)
    h5, s5 = st5['h'], st5['s']
    h6s = common.fluid_state(fluid, P=P_cond, s=s5)['h']
    h6 = h5 - eta_turb * (h5 - h6s)
    hf, hg = sat_c['liquid']['h'], sat_c['vapour']['h']
    x6 = (h6 - hf) / (hg - hf)
    w_turb = (h3 - h4) + (h5 - h6)
    q_in = (h3 - h2) + (h5 - h4)
    w_net = w_turb - w_pump
    eta = w_net / q_in
    return {'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'h5': h5, 'h6': h6, 'x6': x6,
            'w_pump': w_pump, 'w_turb': w_turb, 'w_net': w_net, 'q_in': q_in, 'eta': eta}


def _rankine_regenerative(values):
    """Regenerative Rankine with one open feedwater heater (OFWH).

    1 sat. liquid at P_cond → 2 pump I to P_fwh → mixed with extraction flow y at P_fwh → 3 sat. liquid at
    P_fwh → 4 pump II to P_boiler → 5 turbine inlet (sat. vapour at P_boiler or superheated to T_boiler) →
    6 extraction at P_fwh, 7 turbine exit at P_cond. The extraction mass fraction y follows from the OFWH
    energy balance y h6 + (1−y) h2 = h3. The turbine isentropic efficiency η_turb is applied to both
    stages: h6 = h5 − η_turb(h5 − h6s), s6 = s(P_fwh, h6), then h7 = h6 − η_turb(h6 − h7s) with
    h7s = h(P_cond, s6). Works are per unit mass through the boiler/turbine.
    """
    fluid = values['fluid']
    P_cond, P_fwh, P_boiler = values['P_cond'], values['P_fwh'], values['P_boiler']
    T_boiler, eta_turb, eta_pump = values['T_boiler'], values['eta_turb'], values['eta_pump']
    if not P_boiler > P_fwh > P_cond:
        raise MechError('need P_boiler > P_fwh > P_cond')
    sat_c = common.saturation(fluid, P=P_cond)
    h1 = sat_c['liquid']['h']
    v1 = 1.0 / sat_c['liquid']['rho']
    sat_f = common.saturation(fluid, P=P_fwh)
    h3 = sat_f['liquid']['h']
    v3 = 1.0 / sat_f['liquid']['rho']
    h5, s5, _ = _turbine_inlet(fluid, P_boiler, T_boiler)
    # pump I (1 -> 2, per kg through the condenser), pump II (3 -> 4, per kg through the boiler)
    h2 = h1 + v1 * (P_fwh - P_cond) / eta_pump
    h4 = h3 + v3 * (P_boiler - P_fwh) / eta_pump
    # extraction (5 -> 6) at P_fwh and LP stage (6 -> 7) at P_cond, both with the turbine efficiency.
    h6s = common.fluid_state(fluid, P=P_fwh, s=s5)['h']
    h6 = h5 - eta_turb * (h5 - h6s)
    s6 = common.fluid_state(fluid, P=P_fwh, h=h6)['s']
    h7s = common.fluid_state(fluid, P=P_cond, s=s6)['h']
    h7 = h6 - eta_turb * (h6 - h7s)
    if not h6 > h2:
        raise MechError('extraction enthalpy is not above the feedwater enthalpy; check pressures')
    y = (h3 - h2) / (h6 - h2)
    w_p1 = (1.0 - y) * v1 * (P_fwh - P_cond) / eta_pump
    w_p2 = v3 * (P_boiler - P_fwh) / eta_pump
    w_turb = (h5 - h6) + (1.0 - y) * (h6 - h7)
    q_in = h5 - h4
    w_net = w_turb - w_p1 - w_p2
    eta = w_net / q_in
    return {'y': y, 'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'h5': h5, 'h6': h6, 'h7': h7,
            'w_pump': w_p1 + w_p2, 'w_turb': w_turb, 'w_net': w_net, 'q_in': q_in, 'eta': eta}


def _cascade(values):
    """Two-stage cascade refrigeration cycle (one working fluid in both stages).

    Low stage evaporates at P_evap and condenses at P_cascade; high stage evaporates at P_cascade and
    condenses at P_cond. The cascade exchanger balances heat: ṁ_high / ṁ_low = (h2 − h3)_low / (h1 − h4)_high.
    COP and specific heats/works are per unit mass of low-stage refrigerant.
    """
    fluid = values['fluid']
    P_evap, P_cascade, P_cond = values['P_evap'], values['P_cascade'], values['P_cond']
    if not P_cond > P_cascade > P_evap:
        raise MechError('need P_cond > P_cascade > P_evap')
    lo = _vc_states(fluid, P_evap, P_cascade)
    hi = _vc_states(fluid, P_cascade, P_cond)
    q_L = lo['h1'] - lo['h4']                      # refrigeration effect per kg low-stage
    w_low = lo['h2s'] - lo['h1']
    w_high = hi['h2s'] - hi['h1']
    m_ratio = (lo['h2s'] - lo['h3']) / (hi['h1'] - hi['h4'])   # ṁ_high / ṁ_low
    w_total = w_low + m_ratio * w_high
    if w_total <= 0.0:
        raise MechError('cascade work is not positive; check the pressure levels')
    return {'q_L': q_L, 'w_low': w_low, 'w_high': w_high, 'm_ratio': m_ratio,
            'w_total': w_total, 'cop': q_L / w_total,
            'T_evap': lo['sat_e']['T'], 'T_cascade': lo['sat_c']['T'], 'T_cond': hi['sat_c']['T']}


def _multistage(values):
    """Two-stage vapour-compression with a flash chamber (multistage refrigeration).

    1 sat. vapour at P_evap → 2 isentropic compression to P_int; the condenser liquid (5, sat. liquid at
    P_cond) is throttled to P_int (state 6, quality x6) and separated in the flash chamber into sat.
    vapour (3) and sat. liquid (7 → evaporator). Vapour flows mix (m1 h2 + m3 h3 = m2 h8) before the
    second-stage compression 8 → 9 to P_cond. COP and works are per unit mass through the evaporator.
    """
    fluid = values['fluid']
    P_evap, P_int, P_cond = values['P_evap'], values['P_int'], values['P_cond']
    if not P_cond > P_int > P_evap:
        raise MechError('need P_cond > P_int > P_evap')
    sat_e = common.saturation(fluid, P=P_evap)
    sat_i = common.saturation(fluid, P=P_int)
    sat_c = common.saturation(fluid, P=P_cond)
    h1, s1 = sat_e['vapour']['h'], sat_e['vapour']['s']
    h3 = sat_i['vapour']['h']
    h7 = sat_i['liquid']['h']
    h5 = sat_c['liquid']['h']
    h2 = common.fluid_state(fluid, P=P_int, s=s1)['h']      # stage 1 outlet
    h6 = h5                                                  # throttle 5 -> 6 (h constant)
    hg_i, hf_i = sat_i['vapour']['h'], sat_i['liquid']['h']
    x6 = (h6 - hf_i) / (hg_i - hf_i)                        # quality after throttle
    # per kg through evaporator: m3/m1 = x6/(1−x6)
    m3 = x6 / (1.0 - x6)
    h8 = (h2 + m3 * h3) / (1.0 + m3)                        # mixed stream into stage 2
    h9 = common.fluid_state(fluid, P=P_cond, s=common.fluid_state(fluid, P=P_int, h=h8)['s'])['h']
    q_L = h1 - h7
    w_c1 = h2 - h1
    w_c2 = (1.0 + m3) * (h9 - h8)                           # per kg through evaporator
    w_total = w_c1 + w_c2
    cop = q_L / w_total
    return {'x6': x6, 'm3': m3, 'h1': h1, 'h2': h2, 'h8': h8, 'h9': h9,
            'q_L': q_L, 'w_c1': w_c1, 'w_c2': w_c2, 'w_total': w_total, 'cop': cop,
            'T_evap': sat_e['T'], 'T_int': sat_i['T'], 'T_cond': sat_c['T']}


CALCS = [
    # ---- fundamental / air-standard heat engines ------------------------------------------------
    # Carnot efficiency and Carnot COPs are provided by thermo.carnot_efficiency / thermo.cop_carnot
    # (not duplicated here).
    Calc('otto_cycle', 'Otto cycle thermal efficiency',
         vars=[V('eta', '', 'thermal efficiency', hi=1.0, label='η'),
               V('r', '', 'compression ratio', lo=1.0), _gamma_()],
         equations=['eta = 1 - 1/r**(gamma - 1)'], outputs=['eta'],
         example={'r': 8.0, 'gamma': 1.4},
         keywords=['otto', 'spark ignition', 'si', 'compression ratio', 'air standard'],
         assumptions=['Air-standard Otto cycle: reversible adiabatic compression/expansion, constant-volume '
                      'heat addition/rejection, constant cp and cv. Efficiency depends only on r and γ.'],
         refs=['Otto air-standard efficiency η = 1 − r^(1−γ) (recalled)']),

    Calc('otto_mep', 'Air-standard Otto cycle: states, work, efficiency and mean effective pressure',
         vars=[V('P1', 'Pa', 'pressure at start of compression (BDC)'),
               V('T1', 'K', 'temperature at start of compression (BDC)'),
               V('r', '', 'compression ratio', lo=1.0), _gamma_(),
               V('R', 'J/kg/K', 'specific gas constant of the working fluid'),
               V('q_in', 'J/kg', 'heat added per unit mass (constant volume)'),
               V('T2', 'K', 'temperature after compression'), V('T3', 'K', 'temperature after heat addition'),
               V('T4', 'K', 'temperature after expansion'),
               V('P2', 'Pa', 'pressure after compression'), V('P3', 'Pa', 'pressure after heat addition'),
               V('P4', 'Pa', 'pressure after expansion'),
               V('q_out', 'J/kg', 'heat rejected per unit mass'), V('w_net', 'J/kg', 'net work per unit mass'),
               V('eta', '', 'thermal efficiency', hi=1.0, label='η'),
               V('mep', 'Pa', 'mean effective pressure W_net/(V1 − V2)')],
         procedure=_otto_full,
         outputs=['T2', 'T3', 'T4', 'P2', 'P3', 'P4', 'q_out', 'w_net', 'eta', 'mep'],
         example={'P1': 100e3, 'T1': 300.0, 'r': 8.0, 'gamma': 1.4, 'R': 287.0, 'q_in': 750e3},
         keywords=['otto', 'mep', 'air standard', 'spark ignition'],
         assumptions=['Air standard (ideal gas, constant cp/cv), reversible adiabatic compression/expansion, '
                      'constant-volume heat addition. MEP = W_net/(V1 − V2) with V1 = R T1 / P1 per unit mass.'],
         refs=['Air-standard Otto cycle state relations (recalled)']),

    Calc('diesel_cycle', 'Diesel cycle thermal efficiency',
         vars=[V('eta', '', 'thermal efficiency', hi=1.0, label='η'),
               V('r', '', 'compression ratio', lo=1.0),
               V('r_c', '', 'cutoff ratio V3/V2', lo=1.0, label='rc'), _gamma_()],
         equations=['eta = 1 - (1/r**(gamma - 1))*((r_c**gamma - 1)/(gamma*(r_c - 1)))'],
         outputs=['eta'], example={'r': 20.0, 'r_c': 2.0, 'gamma': 1.4},
         keywords=['diesel', 'compression ignition', 'ci', 'cutoff ratio', 'air standard'],
         assumptions=['Air-standard Diesel cycle: constant-pressure heat addition (cutoff ratio r_c), '
                      'reversible adiabatic compression/expansion. r_c = 1 recovers the Otto efficiency, '
                      'but the closed form is singular there (r_c must exceed 1; use otto_cycle for r_c = 1).'],
         refs=['Diesel air-standard efficiency (recalled)']),

    Calc('dual_cycle', 'Dual (limited-pressure) cycle thermal efficiency',
         vars=[V('eta', '', 'thermal efficiency', hi=1.0, label='η'),
               V('r', '', 'compression ratio', lo=1.0),
               V('r_c', '', 'cutoff ratio V4/V3', lo=1.0, label='rc'),
               V('r_p', '', 'pressure ratio P3/P2 during constant-volume heat addition', lo=1.0, label='rp'),
               _gamma_()],
         equations=['eta = 1 - (r_p*r_c**gamma - 1)/(r**(gamma - 1)*((r_p - 1) + gamma*r_p*(r_c - 1)))'],
         outputs=['eta'], example={'r': 18.0, 'r_c': 1.5, 'r_p': 1.5, 'gamma': 1.4},
         keywords=['dual', 'limited pressure', 'mixed cycle', 'air standard'],
         assumptions=['Air-standard dual cycle: constant-volume then constant-pressure heat addition. '
                      'r_p = 1 gives the Diesel cycle; r_c = 1 gives the Otto cycle.'],
         refs=['Dual air-standard efficiency (recalled)']),

    # A Stirling cycle with a perfect regenerator reaches the Carnot bound (thermo.carnot_efficiency),
    # so no separate calculator is provided here.

    Calc('brayton_cycle', 'Brayton (gas-turbine) cycle thermal efficiency',
         vars=[V('eta', '', 'thermal efficiency', hi=1.0, label='η'),
               V('rp', '', 'pressure ratio P2/P1', lo=1.0, label='rp'), _gamma_()],
         equations=['eta = 1 - 1/rp**((gamma - 1)/gamma)'], outputs=['eta'],
         example={'rp': 10.0, 'gamma': 1.4},
         keywords=['brayton', 'gas turbine', 'jet engine', 'air standard', 'pressure ratio'],
         assumptions=['Air-standard Brayton cycle: isentropic compressor/turbine, constant-pressure heat '
                      'addition/rejection, constant cp/cv. Efficiency depends only on rp and γ.'],
         refs=['Brayton air-standard efficiency η = 1 − rp^(−(γ−1)/γ) (recalled)']),

    Calc('brayton_bwr', 'Brayton cycle back work ratio',
         vars=[V('bwr', '', 'back work ratio W_c/W_t', hi=1.0),
               V('T2', 'K', 'compressor outlet temperature'), V('T4', 'K', 'turbine outlet temperature'),
               V('T1', 'K', 'compressor inlet temperature'), V('T3', 'K', 'turbine inlet temperature'),
               V('rp', '', 'pressure ratio', lo=1.0), _gamma_()],
         equations=['T2 = T1*rp**((gamma - 1)/gamma)', 'T4 = T3/rp**((gamma - 1)/gamma)',
                    'bwr = (T2 - T1)/(T3 - T4)'],
         outputs=['T2', 'T4', 'bwr'], example={'T1': 300.0, 'T3': 1200.0, 'rp': 10.0, 'gamma': 1.4},
         keywords=['brayton', 'back work ratio', 'compressor work', 'turbine work'],
         assumptions=['Ideal (isentropic) compressor and turbine with equal mass flow and specific heat, '
                      'so bwr = cp(T2 − T1) / cp(T3 − T4) cancels cp.'],
         refs=['Back work ratio of the air-standard Brayton cycle']),

    # ---- gas-turbine modifications -----------------------------------------------------------------
    Calc('regenerator_effectiveness', 'Regenerator effectiveness (gas-turbine recuperation)',
         vars=[V('eps', '', 'regenerator effectiveness', hi=1.0, label='ε'),
               V('T5', 'K', 'air temperature leaving the regenerator (into combustor)'),
               V('T2', 'K', 'air temperature leaving the compressor (into regenerator)'),
               V('T4', 'K', 'exhaust temperature leaving the turbine (into regenerator)')],
         equations=['eps = (T5 - T2)/(T4 - T2)'], outputs=['eps'],
         example={'T5': 800.0, 'T2': 490.0, 'T4': 860.0},
         keywords=['regenerator', 'recuperator', 'effectiveness', 'heat exchanger'],
         assumptions=['Effectiveness is the actual air temperature rise divided by the maximum possible '
                      'rise (to the turbine-exhaust temperature). ε = 1 is the ideal regenerator.'],
         refs=['Heat-exchanger effectiveness ε = (T5 − T2)/(T4 − T2)']),

    Calc('intercooler_optimum', 'Optimum intermediate pressure for two-stage compression with intercooling',
         vars=[V('P2', 'Pa', 'optimum intermediate (intercooler) pressure'),
               V('P1', 'Pa', 'inlet (first-stage suction) pressure'),
               V('P3', 'Pa', 'final (second-stage discharge) pressure')],
         equations=['P2 = sqrt(P1*P3)'], outputs=['P2'], example={'P1': 100e3, 'P3': 900e3},
         keywords=['intercooling', 'multistage', 'compressor', 'minimum work'],
         assumptions=['Two ideal-gas stages with perfect intercooling to the inlet temperature; equal '
                      'pressure ratios across the stages minimise the total compression work.'],
         refs=['Minimum-work split for multistage compression with intercooling, P2 = √(P1 P3) (recalled)']),

    Calc('reheat_turbine', 'Two-stage gas turbine with reheat',
         vars=[V('T_max', 'K', 'turbine inlet (and reheat) temperature'),
               V('P1', 'Pa', 'high (turbine inlet) pressure'), V('P2', 'Pa', 'low (turbine exit) pressure'),
               V('P_int', 'Pa', 'intermediate (reheat) pressure'), _gamma_(),
               V('cp', 'J/kg/K', 'constant-pressure specific heat of the gas'),
               V('T2s', 'K', 'first-stage outlet temperature'), V('T4s', 'K', 'second-stage outlet temperature'),
               V('w_stage1', 'J/kg', 'first-stage work'), V('w_stage2', 'J/kg', 'second-stage work'),
               V('w_two', 'J/kg', 'total work with reheat'), V('w_single', 'J/kg', 'work of one turbine without reheat'),
               V('w_extra', 'J/kg', 'extra work gained by reheating')],
         procedure=_reheat_turbine,
         outputs=['T2s', 'T4s', 'w_stage1', 'w_stage2', 'w_two', 'w_single', 'w_extra'],
         example={'T_max': 1400.0, 'P1': 1.2e6, 'P2': 100e3, 'P_int': 350e3, 'gamma': 1.4, 'cp': 1100.0},
         keywords=['reheat', 'gas turbine', 'two-stage', 'multistage turbine'],
         assumptions=['Ideal gas, isentropic stages, reheating restores the gas to T_max between stages. '
                      'Equal stage pressure ratios (P_int = √(P1 P2)) maximise the reheat work for a given T_max.'],
         refs=['Isentropic turbine-stage work; reheat work gain (recalled)']),

    # ---- jet propulsion ---------------------------------------------------------------------------
    Calc('jet_thrust', 'Jet engine thrust (momentum + pressure term)',
         vars=[V('F', 'N', 'net thrust (negative when V_e < V_in, e.g. a deployed reverser)', signed=True),
               V('m_dot', 'kg/s', 'mass flow rate of air (and fuel) through the engine'),
               V('V_e', 'm/s', 'exit (jet) velocity relative to the engine'),
               V('V_in', 'm/s', 'inlet (flight) velocity'),
               V('P_e', 'Pa', 'nozzle exit static pressure'), V('P_amb', 'Pa', 'ambient static pressure'),
               V('A_e', 'm^2', 'nozzle exit area')],
         equations=['F = m_dot*(V_e - V_in) + (P_e - P_amb)*A_e'], outputs=['F'],
         example={'m_dot': 80.0, 'V_e': 600.0, 'V_in': 250.0, 'P_e': 60e3, 'P_amb': 40e3, 'A_e': 0.35},
         keywords=['jet', 'thrust', 'turbojet', 'nozzle', 'momentum'],
         assumptions=['Steady one-dimensional flow; fuel mass flow included in m_dot. For a matched nozzle '
                      '(P_e = P_amb) only the momentum term remains. Static takeoff: V_in = 0.'],
         refs=['Momentum equation for jet thrust, F = ṁ(V_e − V_in) + (P_e − P_a)A_e']),

    # Ideal (Froude) propulsive efficiency is provided by turbomachinery.froude_efficiency.

    # ---- Rankine steam cycles ----------------------------------------------------------------------
    Calc('rankine_cycle', 'Rankine cycle: states, work, heat and efficiency',
         vars=[V('fluid', '', 'working fluid', kind='text', choices=['Water']),
               V('P_cond', 'Pa', 'condenser pressure'), V('P_boiler', 'Pa', 'boiler pressure'),
               V('T_boiler', 'K', 'turbine-inlet temperature (≥ T_sat(P_boiler); equal to T_sat gives saturated vapour)'),
               V('eta_turb', '', 'turbine isentropic efficiency', hi=1.0),
               V('eta_pump', '', 'pump isentropic efficiency', hi=1.0),
               V('T_cond', 'K', 'condenser saturation temperature'), V('T_sat_boiler', 'K', 'boiler saturation temperature'),
               V('h1', 'J/kg', 'condenser-outlet enthalpy'), V('h2', 'J/kg', 'pump-outlet enthalpy'),
               V('h3', 'J/kg', 'turbine-inlet enthalpy'), V('h4', 'J/kg', 'turbine-outlet enthalpy'),
               V('x4', '', 'steam quality at turbine outlet'),
               V('w_pump', 'J/kg', 'pump work per unit mass'), V('w_turb', 'J/kg', 'turbine work per unit mass'),
               V('w_net', 'J/kg', 'net work per unit mass'),
               V('q_in', 'J/kg', 'heat added per unit mass'), V('q_out', 'J/kg', 'heat rejected per unit mass'),
               V('eta', '', 'thermal efficiency', hi=1.0, label='η')],
         procedure=_rankine,
         outputs=['T_cond', 'T_sat_boiler', 'h1', 'h2', 'h3', 'h4', 'x4', 'w_pump', 'w_turb', 'w_net',
                  'q_in', 'q_out', 'eta'],
         example={'fluid': 'Water', 'P_cond': 10e3, 'P_boiler': 10e6, 'T_boiler': 873.15,
                  'eta_turb': 0.85, 'eta_pump': 0.85},
         keywords=['rankine', 'steam power', 'steam turbine', 'quality', 'condenser', 'boiler'],
         assumptions=['Steam properties from CoolProp (IAPWS-95). Ideal pump work uses the saturated-liquid '
                      'specific volume (incompressible). x4 > 1 means the turbine exit is still superheated.'],
         refs=['Rankine cycle energy balances on pump, boiler, turbine and condenser (recalled)']),

    Calc('rankine_reheat', 'Reheat Rankine cycle',
         vars=[V('fluid', '', 'working fluid', kind='text', choices=['Water']),
               V('P_cond', 'Pa', 'condenser pressure'), V('P_boiler', 'Pa', 'boiler (HP turbine inlet) pressure'),
               V('P_reheat', 'Pa', 'reheat (LP turbine inlet) pressure'),
               V('T_boiler', 'K', 'HP turbine inlet temperature (≥ T_sat; equal to T_sat gives saturated vapour)'),
               V('T_reheat', 'K', 'reheat temperature'),
               V('eta_turb', '', 'turbine isentropic efficiency (both stages)', hi=1.0),
               V('eta_pump', '', 'pump isentropic efficiency', hi=1.0),
               V('h1', 'J/kg', 'condenser-outlet enthalpy'), V('h2', 'J/kg', 'pump-outlet enthalpy'),
               V('h3', 'J/kg', 'HP turbine-inlet enthalpy'), V('h4', 'J/kg', 'HP turbine-outlet enthalpy'),
               V('h5', 'J/kg', 'LP turbine-inlet (reheat) enthalpy'), V('h6', 'J/kg', 'LP turbine-outlet enthalpy'),
               V('x6', '', 'steam quality at LP turbine outlet'),
               V('w_pump', 'J/kg', 'pump work'), V('w_turb', 'J/kg', 'total turbine work'),
               V('w_net', 'J/kg', 'net work'), V('q_in', 'J/kg', 'total heat added'), V('eta', '', 'thermal efficiency', hi=1.0)],
         procedure=_rankine_reheat,
         outputs=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'x6', 'w_pump', 'w_turb', 'w_net', 'q_in', 'eta'],
         example={'fluid': 'Water', 'P_cond': 10e3, 'P_boiler': 15e6, 'P_reheat': 2.0e6,
                  'T_boiler': 873.15, 'T_reheat': 773.15, 'eta_turb': 0.85, 'eta_pump': 0.85},
         keywords=['reheat', 'rankine', 'steam', 'two-stage turbine', 'moisture'],
         assumptions=['Steam properties from CoolProp (IAPWS-95). One turbine efficiency for both stages. '
                      'Reheat raises the average heat-addition temperature and reduces exhaust moisture.'],
         refs=['Reheat Rankine cycle energy balances (recalled)']),

    Calc('rankine_regenerative', 'Regenerative Rankine cycle with one open feedwater heater',
         vars=[V('fluid', '', 'working fluid', kind='text', choices=['Water']),
               V('P_cond', 'Pa', 'condenser pressure'), V('P_fwh', 'Pa', 'feedwater-heater (extraction) pressure'),
               V('P_boiler', 'Pa', 'boiler pressure'),
               V('T_boiler', 'K', 'turbine-inlet temperature (≥ T_sat; equal to T_sat gives saturated vapour)'),
               V('eta_turb', '', 'turbine isentropic efficiency', hi=1.0),
               V('eta_pump', '', 'pump isentropic efficiency (both pumps)', hi=1.0),
               V('y', '', 'extraction (feedwater-heater) mass fraction'),
               V('h1', 'J/kg', 'condenser-outlet enthalpy'), V('h2', 'J/kg', 'pump-I outlet enthalpy'),
               V('h3', 'J/kg', 'feedwater-heater outlet (sat. liquid) enthalpy'),
               V('h4', 'J/kg', 'pump-II outlet enthalpy'), V('h5', 'J/kg', 'turbine-inlet enthalpy'),
               V('h6', 'J/kg', 'extraction enthalpy'), V('h7', 'J/kg', 'turbine-outlet enthalpy'),
               V('w_pump', 'J/kg', 'total pump work'), V('w_turb', 'J/kg', 'turbine work'),
               V('w_net', 'J/kg', 'net work'), V('q_in', 'J/kg', 'heat added'), V('eta', '', 'thermal efficiency', hi=1.0)],
         procedure=_rankine_regenerative,
         outputs=['y', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'w_pump', 'w_turb', 'w_net', 'q_in', 'eta'],
         example={'fluid': 'Water', 'P_cond': 10e3, 'P_fwh': 800e3, 'P_boiler': 15e6,
                  'T_boiler': 873.15, 'eta_turb': 0.85, 'eta_pump': 0.85},
         keywords=['regenerative', 'rankine', 'feedwater heater', 'open fwh', 'extraction'],
         assumptions=['One open (mixing) feedwater heater; y solves y h6 + (1−y) h2 = h3. Works are per unit '
                      'mass through the boiler/turbine; the condenser flow is (1 − y).'],
         refs=['Open-feedwater-heater energy balance for regenerative Rankine (recalled)']),

    # ---- combined / cogeneration ------------------------------------------------------------------
    Calc('combined_cycle', 'Combined gas-steam cycle efficiency',
         vars=[V('eta_cc', '', 'combined-cycle efficiency', hi=1.0, label='ηcc'),
               V('eta_gt', '', 'gas-turbine (topping) efficiency', hi=1.0),
               V('eta_st', '', 'steam (bottoming) cycle efficiency', hi=1.0)],
         equations=['eta_cc = eta_gt + eta_st - eta_gt*eta_st'], outputs=['eta_cc'],
         example={'eta_gt': 0.40, 'eta_st': 0.35},
         keywords=['combined cycle', 'gas steam', 'hrsg', 'ccgt', 'topping', 'bottoming'],
         assumptions=['The bottoming cycle recovers the topping-cycle exhaust, so its heat input is the '
                      'fraction (1 − η_gt) of the fuel heat; η_cc = η_gt + η_st(1 − η_gt).'],
         refs=['Combined-cycle efficiency superposition (recalled)']),

    Calc('cogeneration', 'Cogeneration utilization factor',
         vars=[V('eps_u', '', 'utilization (energy) factor', hi=1.0, label='εu'),
               V('W_net', 'W', 'net power output'), V('Q_use', 'W', 'useful heat recovered'),
               V('Q_in', 'W', 'fuel heat input rate')],
         equations=['eps_u = (W_net + Q_use)/Q_in'], outputs=['eps_u'],
         example={'W_net': 10e6, 'Q_use': 20e6, 'Q_in': 40e6},
         keywords=['cogeneration', 'chp', 'utilization factor', 'energy efficiency'],
         assumptions=['Utilization factor counts both the net work and the useful heat against the fuel input; '
                      'it can approach 1 but is not the thermal efficiency.'],
         refs=['Cogeneration utilization factor εu = (W_net + Q_use)/Q_in']),

    # ---- refrigeration ------------------------------------------------------------------------------
    Calc('vapor_compression', 'Vapour-compression refrigeration: capacity, COP and compressor power',
         vars=[V('fluid', '', 'refrigerant', kind='text', choices=['R134a', 'R410A', 'R22', 'R717', 'R1234ze(E)']),
               V('P_evap', 'Pa', 'evaporator pressure'), V('P_cond', 'Pa', 'condenser pressure'),
               V('m_dot', 'kg/s', 'refrigerant mass flow rate'),
               V('h1', 'J/kg', 'evaporator-outlet (compressor inlet) enthalpy'),
               V('h2', 'J/kg', 'compressor-outlet enthalpy'), V('h3', 'J/kg', 'condenser-outlet enthalpy'),
               V('h4', 'J/kg', 'evaporator-inlet (throttled) enthalpy'),
               V('T_evap', 'K', 'evaporator saturation temperature'), V('T_cond', 'K', 'condenser saturation temperature'),
               V('q_L', 'J/kg', 'refrigeration effect per unit mass'), V('w_c', 'J/kg', 'compressor work per unit mass'),
               V('cop', '', 'coefficient of performance'), V('Q_dot', 'W', 'cooling capacity'),
               V('W_dot', 'W', 'compressor power input')],
         procedure=_vapor_compression,
         outputs=['h1', 'h2', 'h3', 'h4', 'T_evap', 'T_cond', 'q_L', 'w_c', 'cop', 'Q_dot', 'W_dot'],
         example={'fluid': 'R134a', 'P_evap': 200e3, 'P_cond': 800e3, 'm_dot': 0.05},
         keywords=['vapour compression', 'refrigerator', 'heat pump', 'cop', 'capacity', 'compressor'],
         assumptions=['Ideal cycle: isentropic compressor, isenthalpic throttling, saturated vapour at the '
                      'compressor inlet and saturated liquid at the condenser outlet. Refrigerant states from CoolProp.'],
         refs=['Vapour-compression cycle energy balances (recalled)']),

    Calc('refrigeration_tons', 'Refrigeration capacity in tons',
         vars=[V('Q_dot', 'W', 'cooling capacity'),
               V('tons', '', 'cooling capacity in tons of refrigeration')],
         procedure=_refrigeration_tons, outputs=['Q_dot'],
         example={'tons': 5.0},
         keywords=['ton', 'refrigeration', 'capacity', 'btu'],
         assumptions=['1 ton of refrigeration = 12 000 Btu/h = 211 kJ/min = 3.51685 kW (ASHRAE definition).'],
         refs=['Definition of the ton of refrigeration (ASHRAE; recalled)']),

    Calc('cascade_refrigeration', 'Two-stage cascade refrigeration cycle',
         vars=[V('fluid', '', 'refrigerant (both stages)', kind='text', choices=['R134a', 'R410A', 'R22', 'R717', 'R1234ze(E)']),
               V('P_evap', 'Pa', 'low-stage evaporator pressure'), V('P_cascade', 'Pa', 'cascade (intermediate) pressure'),
               V('P_cond', 'Pa', 'high-stage condenser pressure'),
               V('q_L', 'J/kg', 'refrigeration effect per kg low-stage refrigerant'),
               V('w_low', 'J/kg', 'low-stage compressor work'), V('w_high', 'J/kg', 'high-stage compressor work'),
               V('m_ratio', '', 'high/low-stage mass-flow ratio'), V('w_total', 'J/kg', 'total work per kg low-stage refrigerant'),
               V('cop', '', 'coefficient of performance'),
               V('T_evap', 'K', 'evaporator temperature'), V('T_cascade', 'K', 'cascade-exchanger temperature'),
               V('T_cond', 'K', 'condenser temperature')],
         procedure=_cascade,
         outputs=['q_L', 'w_low', 'w_high', 'm_ratio', 'w_total', 'cop', 'T_evap', 'T_cascade', 'T_cond'],
         example={'fluid': 'R134a', 'P_evap': 100e3, 'P_cascade': 400e3, 'P_cond': 1.2e6},
         keywords=['cascade', 'two-stage', 'refrigeration', 'low temperature'],
         assumptions=['Two ideal vapour-compression stages in series sharing one working fluid; the cascade '
                      'exchanger balances the low-stage condenser load with the high-stage evaporator load. '
                      'Different fluids per stage are not modelled here.'],
         refs=['Cascade refrigeration cycle energy balances (recalled)']),

    Calc('multistage_refrigeration', 'Two-stage vapour-compression refrigeration with a flash chamber',
         vars=[V('fluid', '', 'refrigerant', kind='text', choices=['R134a', 'R410A', 'R22', 'R717', 'R1234ze(E)']),
               V('P_evap', 'Pa', 'evaporator pressure'), V('P_int', 'Pa', 'flash-chamber (intermediate) pressure'),
               V('P_cond', 'Pa', 'condenser pressure'),
               V('x6', '', 'quality after the condenser-liquid throttle'),
               V('m3', '', 'flash-vapour mass per unit mass through the evaporator'),
               V('h1', 'J/kg', 'evaporator-outlet enthalpy'), V('h2', 'J/kg', 'first-stage outlet enthalpy'),
               V('h8', 'J/kg', 'mixed second-stage inlet enthalpy'), V('h9', 'J/kg', 'second-stage outlet enthalpy'),
               V('q_L', 'J/kg', 'refrigeration effect per kg through evaporator'),
               V('w_c1', 'J/kg', 'first-stage work'), V('w_c2', 'J/kg', 'second-stage work (per kg through evaporator)'),
               V('w_total', 'J/kg', 'total work'), V('cop', '', 'coefficient of performance'),
               V('T_evap', 'K', 'evaporator temperature'), V('T_int', 'K', 'flash-chamber temperature'),
               V('T_cond', 'K', 'condenser temperature')],
         procedure=_multistage,
         outputs=['x6', 'm3', 'h1', 'h2', 'h8', 'h9', 'q_L', 'w_c1', 'w_c2', 'w_total', 'cop',
                  'T_evap', 'T_int', 'T_cond'],
         example={'fluid': 'R134a', 'P_evap': 150e3, 'P_int': 400e3, 'P_cond': 1.0e6},
         keywords=['multistage', 'flash chamber', 'two-stage compression', 'refrigeration'],
         assumptions=['Ideal isentropic compressors and isenthalpic throttles; the flash chamber separates the '
                      'throttled condenser liquid into saturated vapour (to the second stage) and saturated '
                      'liquid (to the evaporator).'],
         refs=['Two-stage vapour-compression with flash chamber (recalled)']),

    Calc('absorption_refrigeration', 'Absorption refrigeration COP',
         vars=[V('cop', '', 'coefficient of performance'),
               V('Q_L', 'W', 'cooling capacity (heat absorbed at the evaporator)'),
               V('Q_gen', 'W', 'heat input to the generator')],
         equations=['cop = Q_L/Q_gen'], outputs=['cop'],
         example={'Q_L': 5.0e3, 'Q_gen': 8.0e3},
         keywords=['absorption', 'ammonia water', 'lithium bromide', 'heat driven'],
         assumptions=['Heat-driven cycle (LiBr–water or ammonia–water): the compressor is replaced by a '
                      'generator/absorber pair, so the useful input is generator heat Q_gen, not shaft work.'],
         refs=['Absorption-cycle COP = Q_L/Q_gen']),

    Calc('gas_refrigeration', 'Gas (reversed-Brayton) refrigeration cycle COP',
         vars=[V('cop', '', 'coefficient of performance'),
               V('rp', '', 'pressure ratio', lo=1.0, label='rp'), _gamma_()],
         equations=['cop = 1/(rp**((gamma - 1)/gamma) - 1)'], outputs=['cop'],
         example={'rp': 4.0, 'gamma': 1.4},
         keywords=['gas refrigeration', 'reversed brayton', 'air cycle', 'aircraft'],
         assumptions=['Air-standard reversed Brayton cycle: isentropic compressor/turbine and constant-pressure '
                      'heat exchange, so COP = 1/(rp^((γ−1)/γ) − 1).'],
         refs=['Reversed-Brayton (gas) refrigeration COP (recalled)']),
]

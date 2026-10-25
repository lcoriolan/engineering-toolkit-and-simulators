# DESCRIPTION: Turbomachinery section: pump power and efficiency, NPSH (available via common.saturation),
#              cavitation number, specific and suction specific speed, affinity laws, system curve and
#              operating point, Euler equation and velocity triangles, degree of reaction, turbine power
#              and specific speed, compressor work and isentropic efficiency, fan laws and total pressure,
#              wind-turbine power and Betz limit, tip-speed ratio, propeller thrust and efficiency.
"""Turbomachinery calculators.

Conventions:
- Incompressible pump/turbine/fan relations use g = 9.80665 m/s^2 (enter it as an input; standard gravity).
- Angular speed omega is declared rad/s (enter rad/s, rpm or rps). Specific speeds here are the
  dimensionless forms: pump N_s = omega sqrt(Q)/(g H)^(3/4); turbine N_s = omega sqrt(P/rho)/(g H)^(5/4).
- The Euler equation is written in the work-input (pump/compressor) convention: w = u2 vu2 - u1 vu1 is
  positive when work is added to the fluid; a turbine reverses the sign.
- Velocity triangles use alpha = absolute flow angle and beta = relative flow angle, both measured from
  the tangential (blade-speed) direction; Vm is the meridional (through-flow) component, Vu the
  tangential component, V = W + U.
- NPSH available computes the vapour pressure with common.saturation (CoolProp) at the liquid temperature.

Relations are the standard ones in turbomachinery and fluid-machinery texts (e.g. Dixon & Hall, *Fluid
Mechanics and Thermodynamics of Turbomachinery*; Munson et al., *Fundamentals of Fluid Mechanics*;
Saravanamuttoo et al., *Gas Turbine Theory*). They are cited by relation name, recalled rather than
re-read, so no page numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts, common

SECTION = section('turbomachinery', 'Turbomachinery', order=260,
                  summary='Pumps, turbines, compressors, fans, wind turbines and propellers: power, '
                          'efficiency, NPSH, specific speed, affinity laws, operating point, velocity '
                          'triangles, Euler equation, Betz limit.')

PI = math.pi


# ---------------------------------------------------------------------------------------------------
# Procedures and chart builders
# ---------------------------------------------------------------------------------------------------
def _fluid_name(v):
    """CoolProp fluid name from the text input (e.g. 'Water')."""
    if not isinstance(v, str) or not v.strip():
        raise MechError('fluid: give the CoolProp fluid name as text, e.g. Water')
    return v.strip()


def _npsh_available(values):
    """Net positive suction head available, with vapour pressure from common.saturation."""
    fluid = _fluid_name(values['fluid'])
    T, p_s, z_s, h_loss, rho, g = (values[k] for k in ('T', 'p_s', 'z_s', 'h_loss', 'rho', 'g'))
    sat = common.saturation(fluid, T=T)
    p_vap = sat['P']
    npsh = (p_s - p_vap) / (rho * g) + z_s - h_loss
    return {'NPSH_A': npsh, 'p_vap': p_vap}


def _velocity_triangle(values):
    """Velocity triangle from blade speed U, absolute speed V and absolute flow angle alpha (from tangential)."""
    U, V, alpha = values['U'], values['V'], values['alpha']
    Vu = V * math.cos(alpha)
    Vm = V * math.sin(alpha)
    Wu = Vu - U
    W = math.hypot(Vm, Wu)
    beta = math.atan2(Vm, Wu)
    # Wu is listed first: it depends on all three inputs, so every input is recoverable by inversion.
    return {'Wu': Wu, 'Vm': Vm, 'Vu': Vu, 'W': W, 'beta': beta}


def _operating_point(values):
    """Pump/system operating point: intersect a tabulated pump curve with H = H_static + K Q^2.

    The intersection must lie inside the tabulated range [Q_min, Q_max] of the supplied pump curve;
    otherwise the pump curve is not defined at the duty point and this raises (no flat-end clamping or
    extrapolation beyond the last row).
    """
    curve = values['pump_curve']
    H_static, K = values['H_static'], values['K']
    if not isinstance(curve, list) or len(curve) < 2:
        raise MechError('pump_curve: give at least two [Q (m^3/s), H (m)] rows')
    pts = sorted((float(r[0]), float(r[1])) for r in curve)
    Qs = [p[0] for p in pts]
    Hs = [p[1] for p in pts]
    Q_min, Q_max = Qs[0], Qs[-1]

    def H_pump(Q):
        if Q < Q_min or Q > Q_max:
            return None
        for i in range(len(Qs) - 1):
            if Q <= Qs[i + 1]:
                f = (Q - Qs[i]) / (Qs[i + 1] - Qs[i])
                return Hs[i] + f * (Hs[i + 1] - Hs[i])
        return Hs[-1]

    def f(Q):
        Hp = H_pump(Q)
        return None if Hp is None else Hp - (H_static + K * Q * Q)

    def op(Q):
        return {'Q_op': Q, 'H_op': H_static + K * Q * Q}

    # Sign-change scan over the tabulated intervals only (the curve is undefined outside them).
    fvals = [f(q) for q in Qs]
    if any(v == 0.0 for v in fvals):
        return op(Qs[next(i for i, v in enumerate(fvals) if v == 0.0)])
    lo = None
    for i in range(len(Qs) - 1):
        if fvals[i] is not None and fvals[i + 1] is not None and fvals[i] * fvals[i + 1] < 0.0:
            lo, hi = Qs[i], Qs[i + 1]
            break
    if lo is None:
        raise MechError(f'no operating point inside the pump-curve range [{Q_min:g}, {Q_max:g}] m^3/s '
                        '(the pump and system curves do not cross there; the duty point lies outside the '
                        'supplied curve)')
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return op(0.5 * (lo + hi))


def _velocity_triangle_chart(values):
    """Velocity triangle: U (blade), W (relative), V (absolute) drawn tip-to-tail."""
    U, Vu, Vm = values['U'], values['Vu'], values['Vm']
    return charts.chart('Velocity triangle', 'tangential [m/s]', 'meridional [m/s]',
                        [charts.series('U (blade)', [0.0, U], [0.0, 0.0]),
                         charts.series('W (relative)', [U, Vu], [0.0, Vm]),
                         charts.series('V (absolute)', [0.0, Vu], [0.0, Vm])],
                        markers=[charts.marker(0.0, 0.0, 'origin'),
                                 charts.marker(Vu, Vm, 'V tip')], equal=True)


def _operating_point_chart(values):
    """Pump curve vs system curve with the operating point marked."""
    curve, H_static, K = values['pump_curve'], values['H_static'], values['K']
    Q_op, H_op = values['Q_op'], values['H_op']
    Qs = [float(r[0]) for r in curve]
    Hs = [float(r[1]) for r in curve]
    qmax = max(Qs) * 1.05
    sys_qs = [qmax * i / 60.0 for i in range(61)]
    sys_hs = [H_static + K * q * q for q in sys_qs]
    return charts.chart('Pump vs system', 'Q [m^3/s]', 'H [m]',
                        [charts.series('pump curve', Qs, Hs),
                         charts.series('system curve', sys_qs, sys_hs)],
                        markers=[charts.marker(Q_op, H_op, 'operating point')])


# Reusable declarations -----------------------------------------------------------------------------
def g_(desc='gravitational acceleration (9.80665 m/s^2 standard)'):
    return V('g', 'm/s^2', desc, lo=0.0, hi=1e3, label='g')


def rho_(desc='fluid density'):
    return V('rho', 'kg/m^3', desc, lo=0.0, hi=1e4, label='ρ')


def omega_(name='omega', desc='angular speed (enter rad/s, rpm or rps)'):
    return V(name, 'rad/s', desc, lo=0.0, hi=1e4, label='ω')


CALCS = [
    # ---- pumps -----------------------------------------------------------------------------------
    Calc('pump_power', 'Pump hydraulic power, shaft power and efficiency',
         vars=[V('P_hyd', 'W', 'hydraulic (fluid) power rho g Q H', lo=0.0, hi=1e9),
               V('P_shaft', 'W', 'shaft (input) power', lo=0.0, hi=1e9),
               rho_(), g_(), V('Q', 'm^3/s', 'volumetric flow rate', lo=0.0, hi=1e3),
               V('H', 'm', 'pump head', lo=0.0, hi=1e5), V('eta', '', 'pump efficiency', hi=1.0, label='η')],
         equations=['P_hyd = rho*g*Q*H', 'P_shaft = P_hyd/eta'], outputs=['P_hyd', 'P_shaft'],
         example={'rho': 1000.0, 'g': 9.80665, 'Q': 0.05, 'H': 20.0, 'eta': 0.75},
         keywords=['water horsepower', 'brake horsepower', 'head rise'],
         assumptions=['eta = P_hyd/P_shaft is the overall pump efficiency. H is the total head added to the '
                      'fluid across the pump.'],
         refs=['P_hyd = rho g Q H; eta = P_hyd/P_shaft']),

    Calc('npsh_available', 'Net positive suction head available (NPSHa)',
         vars=[V('fluid', '', 'CoolProp fluid name, e.g. Water', kind='text'),
               V('T', 'K', 'liquid temperature at the pump suction', lo=274.0, hi=640.0),
               V('p_s', 'Pa', 'absolute pressure at the free surface', lo=0.0, hi=1e8),
               V('z_s', 'm', 'static suction head (positive if the liquid level is above the pump)', signed=True),
               V('h_loss', 'm', 'friction + minor loss head in the suction line', lo=0.0, hi=1e4),
               rho_('liquid density'), g_(),
               V('NPSH_A', 'm', 'net positive suction head available', signed=True),
               V('p_vap', 'Pa', 'vapour pressure of the liquid at T', lo=0.0, hi=1e8)],
         procedure=_npsh_available, outputs=['NPSH_A', 'p_vap'],
         example={'fluid': 'Water', 'T': 300.0, 'p_s': 101325.0, 'z_s': 3.0, 'h_loss': 1.5,
                  'rho': 997.0, 'g': 9.80665},
         keywords=['cavitation', 'suction head', 'vapor pressure'],
         assumptions=['NPSHa = (p_s - p_vap)/(rho g) + z_s - h_loss. The vapour pressure p_vap is computed '
                      'from common.saturation (CoolProp) at the liquid temperature T, which must lie in the '
                      'fluid saturation range. Density rho is entered by the user (a liquid at that state).'],
         refs=['NPSH available = (p_s - p_vap)/(rho g) + z_s - h_loss']),

    Calc('npsh_margin', 'NPSH margin (cavitation check against required NPSH)',
         vars=[V('NPSH_A', 'm', 'NPSH available', lo=0.0, hi=1e4),
               V('NPSH_R', 'm', 'NPSH required (pump data from the manufacturer)', lo=0.0, hi=1e4),
               V('margin', 'm', 'margin NPSHa - NPSHr (must stay positive)', signed=True)],
         equations=['margin = NPSH_A - NPSH_R'], outputs=['margin'],
         example={'NPSH_A': 5.2, 'NPSH_R': 3.1},
         keywords=['cavitation', 'suction'],
         assumptions=['NPSH_R is a pump characteristic obtained from the manufacturer curve, not computed '
                      'here. Keep margin > 0 (with a safety allowance) to avoid cavitation.'],
         refs=['NPSH margin = NPSHa - NPSHr']),

    Calc('cavitation_number', 'Cavitation number',
         vars=[V('sigma', '', 'cavitation number', lo=0.0, hi=1e3, label='σ'),
               V('p', 'Pa', 'local static pressure', lo=0.0, hi=1e8),
               V('p_vap', 'Pa', 'vapour pressure of the liquid', lo=0.0, hi=1e8),
               rho_(), V('V', 'm/s', 'characteristic (approach) velocity', lo=0.0, hi=1e4)],
         equations=['sigma = (p - p_vap)/(0.5*rho*V**2)'], outputs=['sigma'],
         example={'p': 200e3, 'p_vap': 2339.0, 'rho': 1000.0, 'V': 10.0},
         keywords=['cavitation index', 'thoma'],
         assumptions=['sigma = (p - p_vap)/(0.5 rho V^2); cavitation is expected as sigma approaches the '
                      'critical (incipient) value for the geometry. p_vap may be taken from npsh_available '
                      'or the thermo section.'],
         refs=['Cavitation number (Thoma) sigma = (p - p_vap)/(1/2 rho V^2)']),

    Calc('specific_speed', 'Pump specific speed (dimensionless)',
         vars=[V('N_s', '', 'specific speed omega sqrt(Q)/(g H)^(3/4)', lo=0.0, hi=1e2),
               omega_(), V('Q', 'm^3/s', 'flow rate at best efficiency point', lo=0.0, hi=1e3),
               g_(), V('H', 'm', 'head per stage at best efficiency point', lo=0.0, hi=1e5)],
         equations=['N_s = omega*sqrt(Q)/(g*H)**(3/4)'], outputs=['N_s'],
         example={'omega': 1500 * 2 * PI / 60, 'Q': 0.1, 'g': 9.80665, 'H': 15.0},
         keywords=['shape number', 'impeller selection'],
         assumptions=['Dimensionless specific speed: omega in rad/s, Q in m^3/s, H in m. The traditional '
                      'US value is N_s,US = N_s x 2733 (n in rpm, Q in gpm, H in ft). Low N_s favours '
                      'centrifugal, high N_s axial impellers.'],
         refs=['Dimensionless specific speed omega sqrt(Q)/(g H)^(3/4)']),

    Calc('suction_specific_speed', 'Suction specific speed (dimensionless)',
         vars=[V('S', '', 'suction specific speed omega sqrt(Q)/(g NPSH_R)^(3/4)', lo=0.0, hi=1e3),
               omega_(), V('Q', 'm^3/s', 'flow rate at best efficiency point', lo=0.0, hi=1e3),
               g_(), V('NPSH_R', 'm', 'required NPSH at best efficiency point', lo=0.0, hi=1e4)],
         equations=['S = omega*sqrt(Q)/(g*NPSH_R)**(3/4)'], outputs=['S'],
         example={'omega': 1500 * 2 * PI / 60, 'Q': 0.1, 'g': 9.80665, 'NPSH_R': 2.5},
         keywords=['cavitation', 'suction performance'],
         assumptions=['Same dimensionless form as specific_speed but referenced to NPSH_R; low S implies a '
                      'cavitation-prone design.'],
         refs=['Suction specific speed omega sqrt(Q)/(g NPSH_R)^(3/4)']),

    Calc('affinity_laws', 'Pump affinity laws / similarity (speed, diameter and density scaling)',
         vars=[V('Q1', 'm^3/s', 'flow at machine 1', lo=0.0, hi=1e3), V('Q2', 'm^3/s', 'flow at machine 2', lo=0.0, hi=1e3),
               V('H1', 'm', 'head at machine 1', lo=0.0, hi=1e5), V('H2', 'm', 'head at machine 2', lo=0.0, hi=1e5),
               V('P1', 'W', 'power at machine 1', lo=0.0, hi=1e9), V('P2', 'W', 'power at machine 2', lo=0.0, hi=1e9),
               omega_('omega1', 'speed 1'), omega_('omega2', 'speed 2'),
               V('D1', 'm', 'impeller diameter of machine 1', lo=0.0, hi=1e2),
               V('D2', 'm', 'impeller diameter of machine 2', lo=0.0, hi=1e2),
               V('rho1', 'kg/m^3', 'fluid density of machine 1', lo=0.0, hi=1e4, label='ρ1'),
               V('rho2', 'kg/m^3', 'fluid density of machine 2', lo=0.0, hi=1e4, label='ρ2')],
         equations=['Q2 = Q1*(omega2/omega1)*(D2/D1)**3',
                    'H2 = H1*(omega2/omega1)**2*(D2/D1)**2',
                    'P2 = P1*(rho2/rho1)*(omega2/omega1)**3*(D2/D1)**5'],
         outputs=['Q2', 'H2', 'P2'], example={'Q1': 0.1, 'H1': 20.0, 'P1': 30e3,
                                              'omega1': 1450 * 2 * PI / 60, 'omega2': 1750 * 2 * PI / 60,
                                              'D1': 0.3, 'D2': 0.3, 'rho1': 1000.0, 'rho2': 1000.0},
         keywords=['scaling', 'variable speed', 'similarity'],
         assumptions=['Geometrically similar machines; efficiency assumed constant. Q ~ N D^3, H ~ N^2 D^2, '
                      'P ~ rho N^3 D^5. Impeller trimming (cutting D in the same housing) is a separate '
                      'approximation with slightly different exponents, not this geometric-similarity law.'],
         refs=['Pump affinity laws (Q ~ N D^3, H ~ N^2 D^2, P ~ rho N^3 D^5)']),

    # ---- system and operating point --------------------------------------------------------------
    Calc('system_curve', 'System curve (static head plus friction)',
         vars=[V('H_sys', 'm', 'system head', lo=0.0, hi=1e5), V('H_static', 'm', 'static (elevation + pressure) head', lo=0.0, hi=1e5),
               V('K', 's^2/m^5', 'system resistance coefficient', lo=0.0, hi=1e9),
               V('Q', 'm^3/s', 'flow rate', lo=0.0, hi=1e3)],
         equations=['H_sys = H_static + K*Q**2'], outputs=['H_sys'],
         example={'H_static': 20.0, 'K': 500.0, 'Q': 0.1},
         keywords=['pipe resistance', 'friction head'],
         assumptions=['Quadratic (turbulent) system curve; K lumps pipe friction and minor losses for a given '
                      'pipe network.'],
         refs=['H_sys = H_static + K Q^2']),

    Calc('operating_point', 'Pump/system operating point (curve intersection)',
         vars=[V('pump_curve', '', 'pump head curve [[Q (m^3/s), H (m)], ...] (decreasing H with Q)', kind='json'),
               V('H_static', 'm', 'static head of the system', lo=0.0, hi=1e5),
               V('K', 's^2/m^5', 'system resistance coefficient', lo=0.0, hi=1e9),
               V('Q_op', 'm^3/s', 'operating flow rate', lo=0.0, hi=1e3),
               V('H_op', 'm', 'operating head', lo=0.0, hi=1e5)],
         procedure=_operating_point, outputs=['Q_op', 'H_op'],
         example={'pump_curve': [[0.0, 50.0], [0.05, 45.0], [0.1, 35.0], [0.15, 20.0]],
                  'H_static': 20.0, 'K': 500.0},
         plots=[{'id': 'pump_system', 'title': 'Pump vs system', 'fn': _operating_point_chart}],
         keywords=['duty point', 'intersection'],
         assumptions=['Linear interpolation of the tabulated pump curve; quadratic system curve. The duty '
                      'point must lie inside the supplied pump-curve range [Q_min, Q_max]; if the curves '
                      'cross beyond it the calculator raises rather than extrapolate. For pumps in series '
                      'add heads at equal Q, in parallel add flows at equal H (see pumps_series_parallel).'],
         refs=['Operating point = intersection of pump and system curves']),

    Calc('pumps_series_parallel', 'Identical pumps in series and parallel',
         vars=[V('H_series', 'm', 'combined head for n pumps in series (same Q)', lo=0.0, hi=1e5),
               V('Q_parallel', 'm^3/s', 'combined flow for n pumps in parallel (same H)', lo=0.0, hi=1e3),
               V('n_pumps', '', 'number of identical pumps', integer=True, lo=1.0, hi=20.0),
               V('H1', 'm', 'single-pump head', lo=0.0, hi=1e5), V('Q1', 'm^3/s', 'single-pump flow', lo=0.0, hi=1e3)],
         equations=['H_series = n_pumps*H1', 'Q_parallel = n_pumps*Q1'], outputs=['H_series', 'Q_parallel'],
         example={'n_pumps': 3.0, 'H1': 15.0, 'Q1': 0.08},
         keywords=['staging', 'manifold'],
         assumptions=['Identical pumps: series adds head at equal flow, parallel adds flow at equal head. The '
                      'actual combined operating point is the intersection with the system curve (operating_point).'],
         refs=['Series and parallel combination of identical pumps']),

    # ---- Euler equation and velocity triangles ---------------------------------------------------
    Calc('euler_turbomachine', 'Euler turbomachine equation (specific work and ideal head)',
         vars=[V('w_euler', 'J/kg', 'specific work (positive = added to the fluid)', signed=True),
               V('H_euler', 'm', 'ideal head w/g', signed=True),
               V('u1', 'm/s', 'blade speed at inlet', lo=0.0, hi=1e4),
               V('vu1', 'm/s', 'tangential component of absolute velocity at inlet', signed=True),
               V('u2', 'm/s', 'blade speed at outlet', lo=0.0, hi=1e4),
               V('vu2', 'm/s', 'tangential component of absolute velocity at outlet', signed=True), g_()],
         equations=['w_euler = u2*vu2 - u1*vu1', 'H_euler = w_euler/g'], outputs=['w_euler', 'H_euler'],
         example={'u1': 20.0, 'vu1': 4.0, 'u2': 40.0, 'vu2': 30.0, 'g': 9.80665},
         keywords=['energy transfer', 'whirl', 'velocity triangle'],
         assumptions=['Pump/compressor convention: w > 0 when work is added. Power = mdot * w_euler. For a '
                      'turbine the fluid does work (w < 0) and the sign is reversed.'],
         refs=['Euler turbomachine equation w = u2 vu2 - u1 vu1']),

    Calc('velocity_triangle', 'Velocity triangle (absolute, relative and blade-speed vectors)',
         vars=[V('U', 'm/s', 'blade (tangential) speed', lo=0.0, hi=1e4),
               V('V', 'm/s', 'absolute velocity magnitude', lo=0.0, hi=1e4),
               V('alpha', 'rad', 'absolute flow angle from the tangential direction', lo=0.0, hi=PI / 2, label='α'),
               V('Wu', 'm/s', 'tangential component of the relative velocity', signed=True),
               V('Vm', 'm/s', 'meridional (through-flow) component', lo=0.0, hi=1e4),
               V('Vu', 'm/s', 'tangential component of the absolute velocity', signed=True),
               V('W', 'm/s', 'relative velocity magnitude', lo=0.0, hi=1e4),
               V('beta', 'rad', 'relative flow angle from the tangential direction', signed=True, label='β')],
         procedure=_velocity_triangle, outputs=['Wu', 'Vm', 'Vu', 'W', 'beta'],
         example={'U': 30.0, 'V': 50.0, 'alpha': math.radians(40.0)},
         plots=[{'id': 'triangle', 'title': 'Velocity triangle', 'fn': _velocity_triangle_chart}],
         keywords=['blade angle', 'relative flow', 'meridional'],
         assumptions=['V = U + W. Vu = V cos(alpha), Vm = V sin(alpha), Wu = Vu - U, '
                      'beta = atan2(Vm, Wu). alpha, beta measured from the tangential (U) direction; the '
                      'angle used in the Euler equation is the tangential component Vu.'],
         refs=['Velocity triangle V = U + W; Vu = V cos(alpha), Vm = V sin(alpha)']),

    Calc('degree_of_reaction', 'Degree of reaction of a turbomachine stage',
         vars=[V('R', '', 'degree of reaction', lo=0.0, hi=1.0, label='R'),
               V('h1', 'J/kg', 'static enthalpy at rotor inlet', signed=True, lo=-1e8, hi=1e8),
               V('h2', 'J/kg', 'static enthalpy at rotor outlet', signed=True, lo=-1e8, hi=1e8),
               V('h3', 'J/kg', 'static enthalpy at stage outlet', signed=True, lo=-1e8, hi=1e8)],
         equations=['R = (h2 - h1)/(h3 - h1)'], outputs=['R'],
         example={'h1': 3.0e6, 'h2': 3.5e6, 'h3': 4.0e6},
         keywords=['impulse', 'reaction turbine', 'stage'],
         assumptions=['Degree of reaction R = (static enthalpy change across the rotor)/(static enthalpy '
                      'change across the whole stage). Compressor convention (rotor first): station 1 = '
                      'rotor inlet, 2 = rotor outlet, 3 = stage (stator) outlet, so R = (h2 - h1)/(h3 - h1). '
                      'Turbine form (stator first): 1 = stage inlet, 2 = rotor inlet, 3 = rotor outlet, so '
                      'R = (h2 - h3)/(h1 - h3). An impulse stage has R = 0; an ideal 50% reaction stage has '
                      'R = 0.5. For an incompressible machine replace h with p/(rho g).'],
         refs=['Degree of reaction R = (h2 - h1)/(h3 - h1) (compressor); R = (h2 - h3)/(h1 - h3) (turbine)']),

    # ---- turbines --------------------------------------------------------------------------------
    Calc('turbine_power', 'Hydraulic turbine power and efficiency',
         vars=[V('P', 'W', 'turbine shaft (output) power', lo=0.0, hi=1e9),
               rho_(), g_(), V('Q', 'm^3/s', 'flow rate', lo=0.0, hi=1e3),
               V('H', 'm', 'available head', lo=0.0, hi=1e5), V('eta', '', 'turbine efficiency', hi=1.0, label='η')],
         equations=['P = rho*g*Q*H*eta'], outputs=['P'],
         example={'rho': 1000.0, 'g': 9.80665, 'Q': 2.0, 'H': 120.0, 'eta': 0.88},
         keywords=['hydro', 'pelton', 'francis', 'kaplan'],
         assumptions=['eta is the overall turbine efficiency (P = eta rho g Q H).'],
         refs=['Turbine power P = eta rho g Q H']),

    Calc('turbine_specific_speed', 'Turbine specific speed (dimensionless)',
         vars=[V('N_s', '', 'specific speed omega sqrt(P/rho)/(g H)^(5/4)', lo=0.0, hi=1e2),
               omega_(), V('P', 'W', 'turbine power', lo=0.0, hi=1e9),
               rho_(), g_(), V('H', 'm', 'net head', lo=0.0, hi=1e5)],
         equations=['N_s = omega*sqrt(P/rho)/(g*H)**(5/4)'], outputs=['N_s'],
         example={'omega': 300 * 2 * PI / 60, 'P': 1.0e6, 'rho': 1000.0, 'g': 9.80665, 'H': 60.0},
         keywords=['pelton', 'francis', 'kaplan', 'runner selection'],
         assumptions=['Dimensionless turbine specific speed. Impulse (Pelton) machines have low N_s, '
                      'Francis medium, Kaplan (axial) high; the exact selection ranges are manufacturer data.'],
         refs=['Turbine specific speed omega sqrt(P/rho)/(g H)^(5/4)']),

    Calc('pelton_optimum', 'Pelton wheel optimum bucket speed (u = V/2)',
         vars=[V('u', 'm/s', 'bucket (pitch-circle) speed for maximum power', lo=0.0, hi=1e4),
               V('V', 'm/s', 'jet velocity', lo=0.0, hi=1e4)],
         equations=['u = V/2'], outputs=['u'],
         example={'V': 60.0},
         keywords=['pelton', 'impulse turbine', 'bucket speed'],
         assumptions=['For a frictionless Pelton wheel the power is maximised when the bucket speed is half '
                      'the jet speed, u = V/2; real turbines run slightly below this ideal (u ~ 0.46 V).'],
         refs=['Pelton optimum bucket speed u = V/2 (ideal)']),

    # ---- compressors -----------------------------------------------------------------------------
    Calc('compressor', 'Compressor isentropic work, outlet temperature and efficiency',
         vars=[V('T2s', 'K', 'isentropic outlet temperature', lo=0.0, hi=1e4),
               V('w_s', 'J/kg', 'isentropic specific work', lo=0.0, hi=1e7),
               V('eta_c', '', 'isentropic efficiency', hi=1.0, label='ηc'),
               V('cp', 'J/(kg*K)', 'specific heat at constant pressure', lo=1.0, hi=1e5),
               V('gamma', '', 'specific-heat ratio cp/cv', lo=1.0, hi=2.0, label='γ'),
               V('T1', 'K', 'inlet total temperature', lo=0.0, hi=1e4),
               V('P2', 'Pa', 'outlet total pressure', lo=0.0, hi=1e8),
               V('P1', 'Pa', 'inlet total pressure', lo=0.0, hi=1e8),
               V('w_actual', 'J/kg', 'actual specific work', lo=0.0, hi=1e7)],
         equations=['T2s = T1*(P2/P1)**((gamma - 1)/gamma)',
                    'w_s = cp*(T2s - T1)',
                    'eta_c = w_s/w_actual'],
         outputs=['T2s', 'w_s', 'eta_c'], example={'cp': 1005.0, 'gamma': 1.4, 'T1': 300.0,
                                                   'P2': 800e3, 'P1': 100e3, 'w_actual': 245000.0},
         keywords=['isentropic efficiency', 'pressure ratio', 'axial', 'centrifugal'],
         assumptions=['Ideal gas, constant cp and gamma. eta_c = w_s/w_actual, with w_s the isentropic work '
                      'between the same pressure levels. Real-gas behaviour can be evaluated with the thermo '
                      'section (common.fluid_state).'],
         refs=['Isentropic compression: T2s = T1 (P2/P1)^((gamma-1)/gamma); w_s = cp (T2s - T1); '
               'eta_c = w_s/w_actual']),

    # ---- fans ------------------------------------------------------------------------------------
    Calc('fan_laws', 'Fan laws (speed scaling for flow, pressure and power)',
         vars=[V('Q1', 'm^3/s', 'flow at speed 1', lo=0.0, hi=1e3), V('Q2', 'm^3/s', 'flow at speed 2', lo=0.0, hi=1e3),
               V('dp1', 'Pa', 'pressure rise at speed 1', lo=0.0, hi=1e6),
               V('dp2', 'Pa', 'pressure rise at speed 2', lo=0.0, hi=1e6),
               V('P1', 'W', 'power at speed 1', lo=0.0, hi=1e6), V('P2', 'W', 'power at speed 2', lo=0.0, hi=1e6),
               omega_('omega1', 'speed 1'), omega_('omega2', 'speed 2')],
         equations=['Q2 = Q1*(omega2/omega1)', 'dp2 = dp1*(omega2/omega1)**2', 'P2 = P1*(omega2/omega1)**3'],
         outputs=['Q2', 'dp2', 'P2'], example={'Q1': 2.0, 'dp1': 500.0, 'P1': 2.0e3,
                                               'omega1': 1000 * 2 * PI / 60, 'omega2': 1200 * 2 * PI / 60},
         keywords=['fan affinity', 'variable speed', 'blower'],
         assumptions=['Same fan at different speeds, density unchanged: Q ~ N, dp ~ N^2, P ~ N^3.'],
         refs=['Fan laws (Q ~ N, dp ~ N^2, P ~ N^3)']),

    Calc('fan_total_pressure', 'Fan total pressure from static and velocity pressure',
         vars=[V('p_t', 'Pa', 'total pressure', lo=0.0, hi=1e6),
               V('p_s', 'Pa', 'static pressure', lo=0.0, hi=1e6),
               rho_(), V('V', 'm/s', 'air velocity (at the measuring plane)', lo=0.0, hi=1e4)],
         equations=['p_t = p_s + 0.5*rho*V**2'], outputs=['p_t'],
         example={'p_s': 400.0, 'rho': 1.2, 'V': 20.0},
         keywords=['velocity pressure', 'dynamic pressure', 'static pressure'],
         assumptions=['Total pressure = static + dynamic (velocity) pressure. The fan total pressure rise is '
                      'the difference of p_t across the fan.'],
         refs=['p_t = p_s + 1/2 rho V^2']),

    # ---- wind turbines and propellers ------------------------------------------------------------
    Calc('wind_turbine_power', 'Wind turbine power (power coefficient)',
         vars=[V('P', 'W', 'turbine power', lo=0.0, hi=1e9),
               rho_('air density'), V('A', 'm^2', 'rotor swept area', lo=0.0, hi=1e6),
               V('V', 'm/s', 'wind speed', lo=0.0, hi=1e3),
               V('Cp', '', 'power coefficient (<= 16/27 by Betz)', lo=0.0, hi=1.0)],
         equations=['P = 0.5*rho*A*V**3*Cp'], outputs=['P'],
         example={'rho': 1.225, 'A': 500.0, 'V': 10.0, 'Cp': 0.45},
         keywords=['power coefficient', 'rated power'],
         assumptions=['Cp is the power coefficient (fraction of the wind power captured); the Betz limit '
                      'caps Cp at 16/27 ~ 0.593 (see betz_limit).'],
         refs=['P = 1/2 rho A V^3 Cp']),

    Calc('betz_limit', 'Betz limit: maximum extractable wind power',
         vars=[V('P_betz', 'W', 'maximum (Betz-limited) power', lo=0.0, hi=1e9),
               rho_('air density'), V('A', 'm^2', 'rotor swept area', lo=0.0, hi=1e6),
               V('V', 'm/s', 'wind speed', lo=0.0, hi=1e3)],
         equations=['P_betz = (16/27)*0.5*rho*A*V**3'], outputs=['P_betz'],
         example={'rho': 1.225, 'A': 500.0, 'V': 10.0},
         keywords=['betz coefficient', 'actuator disc'],
         assumptions=['Betz (1919) limit: a frictionless actuator disc can extract at most 16/27 of the wind '
                      'power; no real turbine reaches it.'],
         refs=['Betz limit Cp,max = 16/27 (landmark)']),

    Calc('tip_speed_ratio', 'Tip speed ratio',
         vars=[V('TSR', '', 'tip speed ratio omega R / V', lo=0.0, hi=1e2),
               omega_(), V('R', 'm', 'rotor radius', lo=0.0, hi=1e3),
               V('V', 'm/s', 'wind speed', lo=0.0, hi=1e3)],
         equations=['TSR = omega*R/V'], outputs=['TSR'],
         example={'omega': 20 * 2 * PI / 60, 'R': 25.0, 'V': 10.0},
         keywords=['lambda', 'wind turbine design'],
         assumptions=['Tip speed ratio lambda = omega R / V. Modern three-blade turbines peak around '
                      'lambda ~ 6-8 (design value).'],
         refs=['TSR = omega R / V']),

    Calc('propeller', 'Propeller thrust and efficiency (momentum theory)',
         vars=[V('T', 'N', 'thrust', lo=0.0, hi=1e8), V('eta_p', '', 'propulsive efficiency', hi=1.0, label='ηp'),
               V('mdot', 'kg/s', 'mass flow rate through the propeller disc', lo=0.0, hi=1e4),
               V('V_e', 'm/s', 'exit (slipstream) velocity', lo=0.0, hi=1e4),
               V('V_0', 'm/s', 'freestream (flight) velocity', lo=0.0, hi=1e4),
               V('P_shaft', 'W', 'shaft power input', lo=0.0, hi=1e9)],
         equations=['T = mdot*(V_e - V_0)', 'eta_p = T*V_0/P_shaft'], outputs=['T', 'eta_p'],
         example={'mdot': 20.0, 'V_e': 60.0, 'V_0': 40.0, 'P_shaft': 20e3},
         keywords=['actuator disc', 'aircraft propeller', 'propulsive'],
         assumptions=['Ideal (actuator-disc) propeller: thrust = momentum change T = mdot (V_e - V_0); '
                      'propulsive efficiency eta_p = T V_0 / P_shaft. Requires V_e >= V_0.'],
         refs=['Propeller momentum theory: T = mdot (V_e - V_0); eta_p = T V_0 / P_shaft']),

    Calc('froude_efficiency', 'Ideal (Froude) propulsive efficiency',
         vars=[V('eta_ideal', '', 'ideal propulsive efficiency', hi=1.0, label='ηideal'),
               V('V_e', 'm/s', 'exit (slipstream) velocity', lo=0.0, hi=1e4),
               V('V_0', 'm/s', 'freestream (flight) velocity', lo=0.0, hi=1e4)],
         equations=['eta_ideal = 2*V_0/(V_0 + V_e)'], outputs=['eta_ideal'],  # = 2/(1 + V_e/V_0); this form is finite at V_0 = 0
         example={'V_e': 60.0, 'V_0': 40.0},
         keywords=['actuator disc', 'froude efficiency', 'propulsive'],
         assumptions=['Ideal actuator-disc (Froude) propulsive efficiency eta_ideal = 2/(1 + V_e/V_0). '
                      'Requires V_e >= V_0; static operation (V_0 = 0) gives zero propulsive efficiency.'],
         refs=['Froude propulsive efficiency 2/(1 + V_e/V_0)']),
]

# DESCRIPTION: Compressible flow section: Mach number and Mach angle, isentropic stagnation and area
#              relations, choked mass flow, normal and oblique shocks, Prandtl-Meyer expansion, Fanno
#              and Rayleigh flow, nozzle exit velocity and thrust. Ideal-gas, constant-gamma relations
#              are declarative equations; inverse area-Mach and the oblique-shock theta-beta-M inverse
#              (weak/strong branch choice) are procedures.
"""Compressible flow calculators (calorically perfect ideal gas, constant specific-heat ratio gamma).

Conventions and validity limits (read before using these relations):
- Ideal gas, constant gamma and R, isentropic unless a shock/friction/heat path is named. gamma is the
  ratio of specific heats cp/cv; the gas constant R is the specific value (J/(kg K)), and cp = gamma R/(gamma-1).
- Mach number M = V/a with a = sqrt(gamma R T). Angles are radians internally (mu Mach angle, beta shock
  angle, theta deflection, nu Prandtl-Meyer) and enter as rad or deg.
- Stagnation properties are the isentropic (adiabatic, reversible) stagnation values.
- Normal-shock ratios are across a stationary normal shock with upstream Mach M1 >= 1; M2 <= 1.
- Fanno (adiabatic, constant-area, frictional) and Rayleigh (frictionless, constant-area, heat addition)
  relations are referenced to the sonic (*) state; f_D L*/D is the sonic length parameter written with the
  Darcy friction factor f_D (= 4 f with f the Fanning factor), consistent with the fluid_dynamics section.
- These are one-dimensional, steady, ideal-gas relations. High-temperature (thermally perfect) and
  real-gas effects are out of scope; use the thermo section for real-fluid states.

Reference relations are the standard ones in compressible-flow texts (e.g. Anderson, *Modern Compressible
Flow*; White, *Fluid Mechanics*; Zucker & Biblarz, *Fundamentals of Gas Dynamics*). They are cited by
relation name, recalled rather than re-read, so no page numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('compressible', 'Compressible flow', order=250,
                  summary='Mach number, stagnation and isentropic area relations, choked flow, normal '
                          'and oblique shocks, Prandtl-Meyer, Fanno and Rayleigh flow, nozzles and thrust.')

PI = math.pi


# ---------------------------------------------------------------------------------------------------
# Shared math (procedures and chart builders)
# ---------------------------------------------------------------------------------------------------
def _area_ratio(M, gamma):
    """Isentropic A/A* at Mach M (>= 0). Returns inf when M -> 0."""
    if M == 0.0:
        return float('inf')
    return (1.0 / M) * ((2.0 / (gamma + 1.0)) * (1.0 + (gamma - 1.0) / 2.0 * M ** 2)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def _theta_from_beta(M, beta, gamma):
    """Oblique-shock deflection angle theta (rad) for upstream Mach M, shock angle beta."""
    s = math.sin(beta)
    num = M * M * s * s - 1.0
    den = M * M * (gamma + math.cos(2.0 * beta)) + 2.0
    return math.atan((2.0 / math.tan(beta)) * num / den)


def _area_mach(A_Astar, gamma):
    """Mach numbers whose A/A* equals A_Astar: (M_sub < 1, M_super > 1). Requires A_Astar >= 1."""
    if A_Astar < 1.0:
        raise MechError('A/A* must be >= 1 for a real Mach number')
    if abs(A_Astar - 1.0) < 1e-12:
        return 1.0, 1.0
    # Subsonic root: A/A* falls from inf (M -> 0) to 1 (M = 1).
    lo, hi = 1e-9, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _area_ratio(mid, gamma) > A_Astar:
            lo = mid
        else:
            hi = mid
    M_sub = 0.5 * (lo + hi)
    # Supersonic root: A/A* rises from 1 (M = 1) upward; grow an upper bound.
    lo, hi = 1.0, 1.5
    while _area_ratio(hi, gamma) < A_Astar and hi < 1e6:
        hi *= 2.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if _area_ratio(mid, gamma) < A_Astar:
            lo = mid
        else:
            hi = mid
    M_super = 0.5 * (lo + hi)
    return M_sub, M_super


def _beta_weak_strong(M, theta, gamma):
    """Oblique-shock angles beta (weak, strong) for upstream Mach M, deflection theta (rad)."""
    if M < 1.0:
        raise MechError('an oblique shock needs a supersonic upstream (M >= 1)')
    if theta <= 0.0:
        raise MechError('theta (deflection) must be positive for an attached oblique shock')
    mu = math.asin(1.0 / M)
    # Theta peaks inside [mu, pi/2]; find the peak by a golden-style scan to locate the two roots.
    betas = [mu + (PI / 2.0 - mu) * i / 400.0 for i in range(401)]
    thetas = [_theta_from_beta(M, b, gamma) for b in betas]
    imax = max(range(len(thetas)), key=lambda i: thetas[i])
    theta_max = thetas[imax]
    if theta > theta_max * (1.0 + 1e-9):
        raise MechError(f'theta {math.degrees(theta):.3g} deg exceeds the maximum deflection '
                        f'{math.degrees(theta_max):.3g} deg for M = {M:.4g}, gamma = {gamma:.4g}')

    def bisect(lo, hi, rising):
        # theta rises with beta on [mu, beta_max] and falls on [beta_max, pi/2].
        flo, fhi = _theta_from_beta(M, lo, gamma) - theta, _theta_from_beta(M, hi, gamma) - theta
        for _ in range(120):
            mid = 0.5 * (lo + hi)
            fm = _theta_from_beta(M, mid, gamma) - theta
            if (fm < 0.0) == rising:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    bmax = betas[imax]
    beta_weak = bisect(mu, bmax, True)
    beta_strong = bisect(bmax, PI / 2.0, False)
    return beta_weak, beta_strong


def _shock_ratios(M1n, gamma):
    """Normal-shock property ratios across M1n (normal component): (M2n, P2/P1, rho2/rho1, T2/T1)."""
    g = gamma
    num = 2.0 * g * M1n ** 2 - (g - 1.0)
    M2n = math.sqrt(((g - 1.0) * M1n ** 2 + 2.0) / num)
    P2_P1 = num / (g + 1.0)
    rho2_rho1 = (g + 1.0) * M1n ** 2 / ((g - 1.0) * M1n ** 2 + 2.0)
    T2_T1 = P2_P1 / rho2_rho1
    return M2n, P2_P1, rho2_rho1, T2_T1


def _area_ratio_inverse(values):
    """Inverse area-Mach: both Mach numbers whose isentropic A/A* equals the given value."""
    gamma = values['gamma']
    A_Astar = values['A_Astar']
    if gamma <= 1.0:
        raise MechError('gamma must be > 1')
    M_sub, M_super = _area_mach(A_Astar, gamma)
    return {'M_sub': M_sub, 'M_super': M_super}


def _oblique_shock(values):
    """Oblique shock from (M, theta, gamma): weak and strong shock angles plus downstream properties."""
    M, theta, gamma = values['M'], values['theta'], values['gamma']
    beta_w, beta_s = _beta_weak_strong(M, theta, gamma)
    out = {}
    for tag, beta in (('weak', beta_w), ('strong', beta_s)):
        M1n = M * math.sin(beta)
        M2n, P2_P1, rho2_rho1, T2_T1 = _shock_ratios(M1n, gamma)
        out[f'beta_{tag}'] = beta
        out[f'M2_{tag}'] = M2n / math.sin(beta - theta)
        out[f'P2_P1_{tag}'] = P2_P1
        out[f'T2_T1_{tag}'] = T2_T1
        out[f'rho2_rho1_{tag}'] = rho2_rho1
    return out


def _area_ratio_chart(values):
    """A/A* vs M on log-log axes for the given gamma (M from 0.1 to 10)."""
    gamma = values['gamma']
    Ms = [10 ** (x / 100.0) for x in range(-100, 101)]
    AAs = [_area_ratio(M, gamma) for M in Ms]
    return charts.chart('A/A* vs Mach', 'M', 'A/A*',
                        [charts.series('A/A*', Ms, AAs)], xlog=True, ylog=True)


def _theta_beta_m_chart(values):
    """theta-beta-M curve for the given upstream M and gamma."""
    M, gamma = values['M'], values['gamma']
    if M <= 1.0:
        raise MechError('theta-beta-M needs M > 1')
    mu = math.asin(1.0 / M)
    betas = [mu + (PI / 2.0 - mu) * i / 200.0 for i in range(1, 201)]
    thetas = [_theta_from_beta(M, b, gamma) for b in betas]
    return charts.chart('theta-beta-M', 'beta [deg]', 'theta [deg]',
                        [charts.series('theta vs beta',
                                       [math.degrees(b) for b in betas],
                                       [math.degrees(t) for t in thetas])])


# Reusable declarations -----------------------------------------------------------------------------
def gamma_(desc="specific-heat ratio cp/cv"):
    return V('gamma', '', desc, lo=1.0, hi=2.0, label='γ')


def R_(desc='specific gas constant'):
    return V('R', 'J/(kg*K)', desc, lo=1.0, hi=1e4, label='R')


def M_(name='M', desc='Mach number', lo=0.0, hi=50.0):
    return V(name, '', desc, lo=lo, hi=hi)


CALCS = [
    # ---- Mach number and Mach angle --------------------------------------------------------------
    Calc('speed_of_sound', 'Speed of sound (a = sqrt(gamma R T))',
         vars=[V('a', 'm/s', 'speed of sound', lo=0.0, hi=1e4), gamma_(), R_(),
               V('T', 'K', 'static temperature', lo=0.0, hi=1e4)],
         equations=['a = sqrt(gamma*R*T)'], outputs=['a'],
         example={'gamma': 1.4, 'R': 287.0, 'T': 288.15},
         keywords=['sonic velocity', 'acoustic speed'],
         assumptions=['Ideal gas, constant gamma. For air at sea level (gamma = 1.4, R = 287 J/(kg K), '
                      'T = 15 degC) this gives a ~ 340 m/s.'],
         refs=['a = sqrt(gamma R T) (ideal gas)']),

    Calc('mach_number', 'Mach number (M = V/a)',
         vars=[M_(), V('V', 'm/s', 'flow velocity', lo=0.0, hi=1e5),
               V('a', 'm/s', 'local speed of sound', lo=0.0, hi=1e4)],
         equations=['M = V/a'], outputs=['M'], example={'V': 680.0, 'a': 340.0},
         keywords=['velocity ratio', 'subsonic', 'supersonic'],
         refs=['M = V/a']),

    Calc('mach_angle', 'Mach angle (mu = asin(1/M))',
         vars=[V('mu', 'rad', 'Mach angle', lo=0.0, hi=PI / 2, label='μ'), M_('M', 'Mach number (>= 1)', lo=1.0, hi=50.0)],
         equations=['mu = asin(1/M)'], outputs=['mu'], example={'M': 2.0},
         keywords=['mach cone', 'zone of silence'],
         assumptions=['Only defined for M >= 1 (supersonic); mu is the half-angle of the Mach cone.'],
         refs=['sin(mu) = 1/M']),

    # ---- stagnation and isentropic area ----------------------------------------------------------
    Calc('stagnation_ratios', 'Isentropic stagnation property ratios',
         vars=[V('T0_T', '', 'T0/T', lo=1.0, hi=1e3), V('P0_P', '', 'P0/P', lo=1.0, hi=1e6),
               V('rho0_rho', '', 'rho0/rho', lo=1.0, hi=1e6), M_(), gamma_()],
         equations=['T0_T = 1 + (gamma - 1)/2*M**2',
                    'P0_P = (1 + (gamma - 1)/2*M**2)**(gamma/(gamma - 1))',
                    'rho0_rho = (1 + (gamma - 1)/2*M**2)**(1/(gamma - 1))'],
         outputs=['T0_T', 'P0_P', 'rho0_rho'], example={'M': 2.0, 'gamma': 1.4},
         keywords=['total temperature', 'total pressure', 'pilot'],
         assumptions=['Isentropic (adiabatic, reversible) stagnation. Multiply by the static value to get '
                      'the stagnation value: T0 = T (T0/T), etc.'],
         refs=['Isentropic stagnation relations T0/T = 1 + (gamma-1)/2 M^2, P0/P = (...)^(gamma/(gamma-1))']),

    Calc('isentropic_area_ratio', 'Isentropic area ratio A/A*',
         vars=[V('A_Astar', '', 'area ratio A/A*', lo=1.0, hi=1e6), M_(), gamma_()],
         equations=['A_Astar = (1/M)*((2/(gamma + 1))*(1 + (gamma - 1)/2*M**2))**((gamma + 1)/(2*(gamma - 1)))'],
         outputs=['A_Astar'], example={'M': 2.0, 'gamma': 1.4},
         plots=[{'id': 'area_mach', 'title': 'A/A* vs Mach', 'fn': _area_ratio_chart}],
         keywords=['converging-diverging', 'de laval', 'nozzle area'],
         assumptions=['A* is the throat (sonic) area. Two Mach numbers share one A/A* > 1; the subsonic root '
                      'is returned when inverting unless the example favours the supersonic one (use '
                      'area_ratio_inverse for both branches).'],
         refs=['Isentropic area-Mach relation (recalled form)']),

    Calc('area_ratio_inverse', 'Inverse area-Mach: both Mach numbers for a given A/A*',
         vars=[V('A_Astar', '', 'area ratio A/A*', lo=1.0, hi=1e6), gamma_(),
               V('M_sub', '', 'subsonic Mach number (<= 1)', lo=0.0, hi=1.0),
               V('M_super', '', 'supersonic Mach number (>= 1)', lo=1.0, hi=1e3)],
         procedure=_area_ratio_inverse, outputs=['M_sub', 'M_super'],
         example={'A_Astar': 1.6875, 'gamma': 1.4},
         keywords=['nozzle design', 'two solutions'],
         assumptions=['A/A* >= 1; A/A* = 1 gives M = 1. Solves by bisection on the isentropic area-Mach '
                      'relation, so it is only as accurate as that ideal-gas model.'],
         refs=['Inverse of the isentropic area-Mach relation (bisection)']),

    Calc('choked_mass_flow', 'Choked (sonic throat) mass flow rate',
         vars=[V('mdot', 'kg/s', 'mass flow rate', lo=0.0, hi=1e4),
               V('A_star', 'm^2', 'throat area', lo=0.0, hi=1e2),
               V('P0', 'Pa', 'stagnation pressure', lo=0.0, hi=1e8),
               V('T0', 'K', 'stagnation temperature', lo=0.0, hi=1e4), gamma_(), R_()],
         equations=['mdot = A_star*P0/sqrt(T0)*sqrt(gamma/R)*(2/(gamma + 1))**((gamma + 1)/(2*(gamma - 1)))'],
         outputs=['mdot'], example={'A_star': 0.01, 'P0': 5e5, 'T0': 300.0, 'gamma': 1.4, 'R': 287.0},
         keywords=['sonic', 'critical', 'throat', 'discharge coefficient'],
         assumptions=['Choked flow at a sonic throat (M = 1). At the choked condition mdot is the maximum '
                      'for the given stagnation state; gamma = 1.4 gives the constant '
                      '(2/(gamma+1))^((gamma+1)/(2(gamma-1))) sqrt(gamma/R).'],
         refs=['mdot = A* P0 / sqrt(T0) * sqrt(gamma/R) * (2/(gamma+1))^((gamma+1)/(2(gamma-1))) (recalled form)']),

    # ---- shocks and expansion --------------------------------------------------------------------
    Calc('normal_shock', 'Normal shock property ratios',
         vars=[V('M2', '', 'downstream Mach number', lo=0.0, hi=1.0),
               V('P2_P1', '', 'pressure ratio P2/P1', lo=1.0, hi=1e6),
               V('rho2_rho1', '', 'density ratio rho2/rho1', lo=1.0, hi=1e2),
               V('T2_T1', '', 'temperature ratio T2/T1', lo=1.0, hi=1e4),
               V('p02_p01', '', 'stagnation (total) pressure ratio P02/P01', lo=0.0, hi=1.0),
               M_('M1', 'upstream Mach number (>= 1)', lo=1.0, hi=50.0), gamma_()],
         equations=['M2 = sqrt(((gamma - 1)*M1**2 + 2)/(2*gamma*M1**2 - (gamma - 1)))',
                    'P2_P1 = (2*gamma*M1**2 - (gamma - 1))/(gamma + 1)',
                    'rho2_rho1 = (gamma + 1)*M1**2/((gamma - 1)*M1**2 + 2)',
                    'T2_T1 = P2_P1/rho2_rho1',
                    'p02_p01 = ((gamma + 1)*M1**2/(2 + (gamma - 1)*M1**2))**(gamma/(gamma - 1))'
                    '*((gamma + 1)/(2*gamma*M1**2 - (gamma - 1)))**(1/(gamma - 1))'],
         outputs=['M2', 'P2_P1', 'rho2_rho1', 'T2_T1', 'p02_p01'], example={'M1': 2.0, 'gamma': 1.4},
         keywords=['shock strength', 'supersonic to subsonic'],
         assumptions=['Stationary normal shock; the upstream is supersonic (M1 >= 1) and the downstream '
                      'subsonic (M2 <= 1). Entropy rises across the shock, so the stagnation pressure ratio '
                      'P02/P01 <= 1.'],
         refs=['Rankine-Hugoniot normal-shock relations (recalled forms)']),

    Calc('oblique_shock_theta', 'Oblique shock: deflection angle from M and shock angle (theta-beta-M)',
         vars=[V('theta', 'rad', 'flow deflection angle', lo=0.0, hi=PI / 2, label='θ'),
               V('beta', 'rad', 'shock (wave) angle', lo=0.0, hi=PI / 2, label='β'),
               M_(), gamma_()],
         equations=['theta = atan(2/tan(beta)*(M**2*sin(beta)**2 - 1)/(M**2*(gamma + cos(2*beta)) + 2))'],
         outputs=['theta'], example={'M': 2.0, 'beta': math.radians(40.0), 'gamma': 1.4},
         keywords=['wedge', 'ramp', 'deflection'],
         assumptions=['Supersonic upstream. A real shock needs M sin(beta) >= 1 (beta >= Mach angle); the '
                      'relation then gives theta from 0 at the Mach angle back to 0 at a normal shock, with a '
                      'maximum in between.'],
         refs=['theta-beta-M relation, tan(theta) = 2 cot(beta) (M^2 sin^2(beta) - 1)/(M^2(gamma + cos 2beta) + 2)']),

    Calc('oblique_shock', 'Oblique shock: weak and strong shock angles and downstream properties',
         vars=[M_(), V('theta', 'rad', 'flow deflection angle', lo=0.0, hi=PI / 2, label='θ'), gamma_(),
               V('beta_weak', 'rad', 'weak shock angle', lo=0.0, hi=PI / 2, label='βw'),
               V('beta_strong', 'rad', 'strong shock angle', lo=0.0, hi=PI / 2, label='βs'),
               V('M2_weak', '', 'downstream Mach, weak branch', lo=0.0, hi=1e3),
               V('M2_strong', '', 'downstream Mach, strong branch', lo=0.0, hi=1.0),
               V('P2_P1_weak', '', 'P2/P1, weak branch', lo=1.0, hi=1e6),
               V('P2_P1_strong', '', 'P2/P1, strong branch', lo=1.0, hi=1e6),
               V('T2_T1_weak', '', 'T2/T1, weak branch', lo=1.0, hi=1e4),
               V('T2_T1_strong', '', 'T2/T1, strong branch', lo=1.0, hi=1e4),
               V('rho2_rho1_weak', '', 'rho2/rho1, weak branch', lo=1.0, hi=1e2),
               V('rho2_rho1_strong', '', 'rho2/rho1, strong branch', lo=1.0, hi=1e2)],
         procedure=_oblique_shock, outputs=['beta_weak', 'beta_strong', 'M2_weak', 'M2_strong',
                                            'P2_P1_weak', 'P2_P1_strong', 'T2_T1_weak', 'T2_T1_strong',
                                            'rho2_rho1_weak', 'rho2_rho1_strong'],
         example={'M': 2.0, 'theta': math.radians(20.0), 'gamma': 1.4},
         plots=[{'id': 'theta_beta_m', 'title': 'theta-beta-M', 'fn': _theta_beta_m_chart}],
         keywords=['wedge', 'attached shock', 'strong solution', 'detached'],
         assumptions=['Attached oblique shock (theta below the detachment maximum; otherwise the shock is '
                      'detached and this calculator raises). Weak solution is the one usually observed; the '
                      'strong solution keeps M2 < 1. Downstream values use the normal shock on M1n = M sin(beta).'],
         refs=['theta-beta-M weak/strong solution; normal-shock relations on the normal component']),

    Calc('prandtl_meyer', 'Prandtl-Meyer expansion function',
         vars=[V('nu', 'rad', 'Prandtl-Meyer angle', lo=0.0, hi=PI, label='ν'),
               M_('M', 'Mach number (>= 1)', lo=1.0, hi=50.0), gamma_()],
         equations=['nu = sqrt((gamma + 1)/(gamma - 1))*atan(sqrt((gamma - 1)/(gamma + 1)*(M**2 - 1))) '
                    '- atan(sqrt(M**2 - 1))'],
         outputs=['nu'], example={'M': 2.0, 'gamma': 1.4},
         keywords=['expansion fan', 'supersonic corner'],
         assumptions=['nu(M) = 0 at M = 1 and rises to nu_max = pi/2 (sqrt((gamma+1)/(gamma-1)) - 1) as '
                      'M -> inf. Across a centered expansion fan the turning is nu2 - nu1.'],
         refs=['Prandtl-Meyer function (recalled form)']),

    # ---- Fanno and Rayleigh ----------------------------------------------------------------------
    Calc('fanno_flow', 'Fanno flow (adiabatic constant-area duct with friction)',
         vars=[V('T_Tstar', '', 'T/T*', lo=1e-6, hi=1e2), V('P_Pstar', '', 'P/P*', lo=1e-6, hi=1e4),
               V('rho_rhostar', '', 'rho/rho*', lo=1e-6, hi=1e4), V('P0_P0star', '', 'P0/P0*', lo=1e-6, hi=1e4),
               V('fDL_D', '', 'f_D L*/D (Darcy-friction sonic length parameter)', lo=0.0, hi=100.0),
               M_('M', 'Mach number', lo=0.01, hi=20.0), gamma_()],
         equations=['T_Tstar = (gamma + 1)/(2 + (gamma - 1)*M**2)',
                    'P_Pstar = (1/M)*sqrt((gamma + 1)/(2 + (gamma - 1)*M**2))',
                    'rho_rhostar = (1/M)*sqrt((2 + (gamma - 1)*M**2)/(gamma + 1))',
                    'P0_P0star = (1/M)*((2 + (gamma - 1)*M**2)/(gamma + 1))**((gamma + 1)/(2*(gamma - 1)))',
                    'fDL_D = (1 - M**2)/(gamma*M**2) + (gamma + 1)/(2*gamma)*log((gamma + 1)*M**2/(2 + (gamma - 1)*M**2))'],
         outputs=['T_Tstar', 'P_Pstar', 'rho_rhostar', 'P0_P0star', 'fDL_D'],
         example={'M': 0.5, 'gamma': 1.4},
         keywords=['pipe friction', 'duct', 'choking length'],
         assumptions=['Adiabatic, constant-area duct with constant Darcy friction factor f_D; * is the sonic '
                      '(M = 1) reference state. The friction parameter f_D L*/D uses the Darcy factor '
                      '(f_D = 4 f, with f the Fanning factor); the duct length needed to reach M = 1 from '
                      'state M is L* = (f_D L*/D) D / f_D.'],
         refs=['Fanno flow reference ratios (recalled forms)']),

    Calc('rayleigh_flow', 'Rayleigh flow (frictionless constant-area duct with heat addition)',
         vars=[V('T_Tstar', '', 'T/T*', lo=1e-6, hi=1e2), V('P_Pstar', '', 'P/P*', lo=1e-6, hi=1e2),
               V('rho_rhostar', '', 'rho/rho*', lo=1e-6, hi=1e4), V('T0_T0star', '', 'T0/T0*', lo=1e-6, hi=1e2),
               V('P0_P0star', '', 'P0/P0*', lo=1e-6, hi=1e4),
               M_('M', 'Mach number', lo=0.01, hi=20.0), gamma_()],
         equations=['T_Tstar = (gamma + 1)**2*M**2/(1 + gamma*M**2)**2',
                    'P_Pstar = (gamma + 1)/(1 + gamma*M**2)',
                    'rho_rhostar = (1 + gamma*M**2)/((gamma + 1)*M**2)',
                    'T0_T0star = (gamma + 1)*M**2*(2 + (gamma - 1)*M**2)/(1 + gamma*M**2)**2',
                    'P0_P0star = (gamma + 1)/(1 + gamma*M**2)*((2 + (gamma - 1)*M**2)/(gamma + 1))**(gamma/(gamma - 1))'],
         outputs=['T_Tstar', 'P_Pstar', 'rho_rhostar', 'T0_T0star', 'P0_P0star'],
         example={'M': 0.5, 'gamma': 1.4},
         keywords=['heat addition', 'combustion chamber', 'thermal choking'],
         assumptions=['Frictionless, constant-area duct with heat addition; * is the sonic reference. Heat '
                      'addition drives M toward 1 (thermal choking): subsonic flow speeds up, supersonic '
                      'slows down.'],
         refs=['Rayleigh flow reference ratios (recalled forms)']),

    # ---- nozzle and thrust -----------------------------------------------------------------------
    Calc('nozzle_exit_velocity', 'Nozzle exit velocity (isentropic expansion from stagnation)',
         vars=[V('V_e', 'm/s', 'exit velocity', lo=0.0, hi=1e5),
               V('Te', 'K', 'exit static temperature', lo=0.0, hi=1e4),
               V('T0', 'K', 'stagnation temperature', lo=0.0, hi=1e4),
               V('P_e', 'Pa', 'exit static pressure', lo=0.0, hi=1e8),
               V('P0', 'Pa', 'stagnation pressure', lo=0.0, hi=1e8), gamma_(), R_()],
         equations=['Te = T0*(P_e/P0)**((gamma - 1)/gamma)',
                    'V_e = sqrt(2*gamma/(gamma - 1)*R*(T0 - Te))'],
         outputs=['V_e', 'Te'], example={'T0': 300.0, 'P_e': 101325.0, 'P0': 5e5, 'gamma': 1.4, 'R': 287.0},
         keywords=['isentropic nozzle', 'exhaust', 'rocket'],
         assumptions=['Isentropic, adiabatic expansion from stagnation; P_e <= P0 (choked or unchoked as '
                      'given). Equivalent to V_e = sqrt(2 gamma/(gamma-1) R T0 (1 - (P_e/P0)^((gamma-1)/gamma))).'],
         refs=['Isentropic nozzle exit velocity from the steady-flow energy equation']),

    Calc('nozzle_thrust', 'Rocket/nozzle thrust (momentum plus pressure term)',
         vars=[V('F', 'N', 'thrust', lo=0.0, hi=1e8),
               V('mdot', 'kg/s', 'mass flow rate', lo=0.0, hi=1e4),
               V('V_e', 'm/s', 'exit velocity', lo=0.0, hi=1e5),
               V('P_e', 'Pa', 'exit static pressure', lo=0.0, hi=1e8),
               V('P_a', 'Pa', 'ambient (back) pressure', lo=0.0, hi=1e8),
               V('A_e', 'm^2', 'exit area', lo=0.0, hi=1e2)],
         equations=['F = mdot*V_e + (P_e - P_a)*A_e'], outputs=['F'],
         example={'mdot': 50.0, 'V_e': 2000.0, 'P_e': 101325.0, 'P_a': 0.0, 'A_e': 0.5},
         keywords=['rocket', 'jet engine', 'specific impulse'],
         assumptions=['Steady one-dimensional exit plane; the pressure term (P_e - P_a) A_e vanishes for a '
                      'perfectly expanded nozzle (P_e = P_a).'],
         refs=['F = mdot V_e + (P_e - P_a) A_e (steady momentum equation)']),
]

# DESCRIPTION: Fluid dynamics section: continuity, Bernoulli and energy head, momentum and angular
#              momentum, dimensionless numbers, laminar pipe flow, Darcy friction factor (laminar,
#              Colebrook, Swamee–Jain, Haaland, Moody chart), head losses, hydraulic diameter, boundary
#              layers, drag/lift, terminal velocity, flow meters, stream function/vorticity/circulation,
#              water hammer, Buckingham π and pipe networks (Hardy Cross).
"""Fluid dynamics calculators.

Conventions: incompressible steady flow unless stated; head is energy per unit weight in metres;
angles in radians. Friction factor covers laminar (64/Re), Colebrook (iterated), Swamee–Jain and Haaland
closed forms with a Moody chart generated from Colebrook. Pipe networks use the Hardy Cross loop method.
Minor-loss K and discharge coefficients are user inputs (they come from tables).

Relations are the standard ones in fluid-mechanics texts (Munson, Young & Okiishi; White; Fox & McDonald).
Cited by relation name, recalled rather than re-read. The friction-factor correlations are shared with
engineering_sim/operating_points.py through engineering_sim/correlations.py (one implementation, stdlib only).
"""
import math

import sympy as sp

from engineering_sim import correlations
from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts, common

SECTION = section('fluid_dynamics', 'Fluid dynamics', order=240,
                  summary='Continuity, Bernoulli, momentum, dimensionless numbers, pipe flow and friction '
                          'factor, head loss, boundary layers, drag/lift, flow meters, water hammer, '
                          'Buckingham π and pipe networks.')

PI = math.pi
G0 = common.g0()  # 9.80665 m/s², used in procedures and to pre-fill example inputs


# ---- friction-factor helpers ----------------------------------------------------------------------
# The correlations themselves live in engineering_sim/correlations.py (stdlib, shared with engineering_sim/operating_points.py).
def _friction_factor(values):
    """Darcy friction factor: laminar 64/Re, Colebrook (iterated), Swamee–Jain and Haaland."""
    Re = values['Re']
    rel = values['rel_rough']
    if Re <= 0:
        raise MechError('Re must be positive')
    if rel < 0:
        raise MechError('relative roughness must be >= 0')
    return {'f_colebrook': correlations.darcy_colebrook(Re, rel),
            'f_swamee_jain': correlations.darcy_swamee_jain(Re, rel),
            'f_haaland': correlations.darcy_haaland(Re, rel),
            'f_laminar': correlations.darcy_laminar(Re)}


def _moody_chart(values):
    """Moody diagram spec: laminar 64/Re plus Colebrook curves for several ε/D, log–log."""
    re_lam = [600, 800, 1000, 1500, 2000, 2300]
    re_turb = [4000, 6000, 1e4, 2e4, 4e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 5e7, 1e8]
    rels = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001, 0.0]
    series = [charts.series('laminar (64/Re)', re_lam, [64.0 / r for r in re_lam])]
    for rel in rels:
        series.append(charts.series(f'ε/D = {rel:g}', re_turb, [correlations.darcy_colebrook(r, rel) for r in re_turb]))
    return charts.chart('Moody chart (Colebrook)', 'Reynolds number Re', 'Darcy friction factor f',
                        series, xlog=True, ylog=True)


# ---- other procedures ----------------------------------------------------------------------------
def _momentum_bend(values):
    """Force of a steady fluid stream on a pipe bend/nozzle/vane (2-D, horizontal plane).

    inlet/outlet are [v (m/s), p (Pa, gauge), A (m²), theta (deg from +x)]; rho (kg/m³). The force is
    that exerted BY the fluid ON the bend: F = p1 A1 e1 − p2 A2 e2 + mdot (v1 e1 − v2 e2).
    """
    rho = values['rho']
    v1, p1, A1, th1 = map(float, values['inlet'])
    v2, p2, A2, th2 = map(float, values['outlet'])
    if A1 <= 0 or A2 <= 0:
        raise MechError('areas must be positive')
    # Steady incompressible flow: mass continuity requires A1·v1 = A2·v2.
    if abs(A1 * v1 - A2 * v2) > 0.01 * max(abs(A1 * v1), abs(A2 * v2), 1e-300):
        raise MechError('steady flow needs A1·v1 ≈ A2·v2 (continuity); these differ by more than 1 %')
    e1x, e1y = math.cos(math.radians(th1)), math.sin(math.radians(th1))
    e2x, e2y = math.cos(math.radians(th2)), math.sin(math.radians(th2))
    mdot = rho * A1 * v1
    Fx = p1 * A1 * e1x - p2 * A2 * e2x + mdot * (v1 * e1x - v2 * e2x)
    Fy = p1 * A1 * e1y - p2 * A2 * e2y + mdot * (v1 * e1y - v2 * e2y)
    return {'Fx': Fx, 'Fy': Fy, 'F': math.hypot(Fx, Fy), 'theta_F': math.atan2(Fy, Fx), 'mdot': mdot}


def _boundary_layer(values):
    """Flat-plate boundary layer: laminar (Blasius) and turbulent (1/7 power law) thicknesses and C_f."""
    rho, U, x, mu = values['rho'], values['U'], values['x'], values['mu']
    if x <= 0 or mu <= 0:
        raise MechError('x and mu must be positive')
    Re_x = rho * U * x / mu
    if Re_x <= 0:
        raise MechError('Re_x must be positive (check the sign of U)')
    sx = math.sqrt(Re_x)
    delta_lam = 5.0 * x / sx
    dstar_lam = 1.7208 * x / sx
    theta_lam = 0.664 * x / sx
    cf_lam = 0.664 / sx
    cf_lam_mean = 1.328 / sx
    re15 = Re_x ** 0.2
    delta_turb = 0.37 * x / re15
    return {'Re_x': Re_x, 'delta_lam': delta_lam, 'delta_star_lam': dstar_lam, 'theta_lam': theta_lam,
            'delta_turb': delta_turb, 'delta_star_turb': delta_turb / 8.0, 'theta_turb': 7.0 * delta_turb / 72.0,
            'cf_lam': cf_lam, 'cf_turb': 0.0592 / re15,
            'cf_lam_mean': cf_lam_mean, 'cf_turb_mean': 0.074 / re15}


def _buckingham(values):
    """Buckingham π theorem: dimensionless groups from the dimension matrix via sympy nullspace.

    matrix: rows = fundamental dimensions (M, L, T, ...), columns = physical quantities, integer
    exponents. The first `rank` columns must be the repeating variables (independent). Each π group is
    one non-repeating variable times powers of the repeating variables: solve D_r·a = −d_j for column j.
    """
    mat = values['matrix']
    if not isinstance(mat, list) or not mat or not all(isinstance(r, list) for r in mat):
        raise MechError('matrix: give a list of rows (dimensions) of integer exponents')
    try:
        M = sp.Matrix([[int(x) for x in row] for row in mat])
    except (TypeError, ValueError):
        raise MechError('matrix entries must be integers')
    m, n = M.shape
    rank = M.rank()
    if rank == 0:
        raise MechError('rank 0: the quantities are already dimensionless')
    if rank != m:
        raise MechError(f'rank {rank} < {m} dimensions: remove redundant dimension rows')
    if n - rank < 1:
        raise MechError('nullity 0: the variables are dimensionally independent')
    Dr = M[:, :rank]
    if Dr.det() == 0:
        raise MechError('the first r columns (repeating variables) must be linearly independent; reorder them')
    groups = []
    for j in range(rank, n):
        a = Dr.solve(-M[:, j])
        vec = [0] * n
        for i in range(rank):
            x = a[i]
            vec[i] = int(x) if x.is_Integer else float(x)
        vec[j] = 1
        groups.append(vec)
    return {'pi_groups': groups, 'n_pi': len(groups), 'rank': int(rank)}


def _series_pipes(values):
    """Total head loss for pipes in series (one flow rate): h = Σ (f L/D + K) V²/(2g)."""
    Q = values['Q']
    if Q < 0:
        raise MechError('Q must be >= 0')
    h_total = 0.0
    h_each = []
    for row in values['pipes']:
        if len(row) != 4:
            raise MechError('pipes: each row must be [L (m), D (m), f, K]')
        L, D, f, K = map(float, row)
        if D <= 0:
            raise MechError('pipe diameter must be positive')
        A = PI * D * D / 4.0
        vel = Q / A
        h = (f * L / D + K) * vel * vel / (2.0 * G0)
        h_each.append(h)
        h_total += h
    return {'h_total': h_total, 'h_each': h_each}


def _pipe_network(values):
    """Pipe network by the Hardy Cross loop method.

    pipes: [[id, L (m), D (m), f, K, Q0 (m³/s)], ...]; initial flows Q0 must satisfy junction
    continuity. loops: [[[id, dir], ...], ...] with dir +1/−1 relative to each pipe's positive flow.
    Iterates ΔQ = −Σ(dr·R·Q|Q|) / Σ(2·R·|Q|) around each loop until |ΔQ| < 1e-9 m³/s.
    """
    pipes = values['pipes']
    loops = values['loops']
    idx, L, D, f, K, Q = {}, [], [], [], [], []
    for row in pipes:
        if len(row) != 6:
            raise MechError('pipes: each row must be [id, L (m), D (m), f, K, Q0 (m³/s)]')
        pid = row[0]
        idx[pid] = len(Q)
        L.append(float(row[1])); D.append(float(row[2])); f.append(float(row[3]))
        K.append(float(row[4])); Q.append(float(row[5]))
    n = len(Q)
    if n == 0:
        raise MechError('pipes: at least one pipe is required')
    A = [PI * d * d / 4.0 for d in D]
    R = [(f[i] * L[i] / D[i] + K[i]) / (2.0 * G0 * A[i] * A[i]) for i in range(n)]

    def resolve(pid):
        if pid not in idx:
            raise MechError(f'unknown pipe id {pid!r} in a loop')
        return idx[pid]

    it, maxcorr = 0, 0.0
    for it in range(1, 201):
        maxcorr = 0.0
        for loop in loops:
            num = den = 0.0
            entries = []
            for entry in loop:
                pid, dr = entry[0], (1.0 if float(entry[1]) > 0 else -1.0)
                i = resolve(pid)
                entries.append((i, dr))
                num += dr * R[i] * Q[i] * abs(Q[i])
                den += 2.0 * R[i] * abs(Q[i])
            if den == 0:
                continue
            dQ = -num / den
            maxcorr = max(maxcorr, abs(dQ))
            for i, dr in entries:
                Q[i] += dr * dQ
        if maxcorr < 1e-9:
            break
    if maxcorr >= 1e-9:
        raise MechError(f'Hardy Cross did not converge in 200 iterations (last |ΔQ| = {maxcorr:.3g} m³/s); '
                        'check the network definition and the initial flows')
    flows = [[pipes[i][0], Q[i], R[i] * Q[i] * abs(Q[i])] for i in range(n)]
    return {'flows': flows, 'iterations': it, 'max_correction': maxcorr}


CALCS = [
    # ---- conservation and energy ------------------------------------------------------------------
    Calc('continuity', 'Continuity: volumetric and mass flow rate (Q = A·v)',
         vars=[V('Q', 'm^3/s', 'volumetric flow rate'), V('mdot', 'kg/s', 'mass flow rate'),
               V('A', 'm^2', 'cross-sectional area'), V('v', 'm/s', 'mean velocity'),
               V('rho', 'kg/m^3', 'density', label='ρ')],
         equations=['Q = A*v', 'mdot = rho*Q'], outputs=['Q', 'mdot'],
         example={'A': 0.02, 'v': 5.0, 'rho': 1000.0},
         keywords=['flow rate', 'mass flow', 'discharge'],
         refs=['Continuity: Q = A·v; ṁ = ρQ']),

    Calc('bernoulli_ideal', 'Bernoulli equation (ideal, along a streamline)',
         vars=[V('p1', 'Pa', 'pressure at point 1'), V('p2', 'Pa', 'pressure at point 2'),
               V('v1', 'm/s', 'velocity at point 1'), V('v2', 'm/s', 'velocity at point 2'),
               V('z1', 'm', 'elevation of point 1'), V('z2', 'm', 'elevation of point 2'),
               V('rho', 'kg/m^3', 'density', label='ρ'), V('g', 'm/s^2', 'gravitational acceleration')],
         equations=['p1 + rho*v1**2/2 + rho*g*z1 = p2 + rho*v2**2/2 + rho*g*z2'], outputs=['p2'],
         example={'p1': 101325.0, 'v1': 2.0, 'v2': 4.0, 'z1': 2.0, 'z2': 5.0, 'rho': 1000.0, 'g': G0},
         keywords=['bernoulli', 'streamline', 'energy'],
         assumptions=['Steady, incompressible, inviscid flow along a streamline; no shaft work or losses '
                      '(for those, use the energy equation).'],
         refs=['Bernoulli: p + ½ρv² + ρgz = const']),

    Calc('energy_equation', 'Energy equation in head form (with losses and pump/turbine head)',
         vars=[V('H1', 'm', 'total head at section 1'), V('H2', 'm', 'total head at section 2'),
               V('h_L', 'm', 'head loss between sections (friction + minor)'),
               V('p1', 'Pa', 'pressure at section 1'), V('p2', 'Pa', 'pressure at section 2'),
               V('v1', 'm/s', 'velocity at section 1'), V('v2', 'm/s', 'velocity at section 2'),
               V('z1', 'm', 'elevation of section 1'), V('z2', 'm', 'elevation of section 2'),
               V('rho', 'kg/m^3', 'density', label='ρ'), V('g', 'm/s^2', 'gravitational acceleration'),
               V('h_p', 'm', 'head added by a pump'), V('h_t', 'm', 'head extracted by a turbine')],
         equations=['H1 = p1/(rho*g) + v1**2/(2*g) + z1', 'H2 = p2/(rho*g) + v2**2/(2*g) + z2',
                    'H2 = H1 + h_p - h_t - h_L'],
         outputs=['H1', 'H2', 'h_L'],
         example={'p1': 200000.0, 'p2': 100000.0, 'v1': 2.0, 'v2': 3.0, 'z1': 2.0, 'z2': 7.0,
                  'rho': 1000.0, 'g': G0, 'h_p': 5.0, 'h_t': 2.0},
         keywords=['head', 'pump', 'turbine', 'energy equation', 'head loss'],
         assumptions=['Steady, incompressible flow between two sections with uniform properties; the kinetic-'
                      'energy correction factor α = 1. h_L bundles all friction and minor losses.'],
         refs=['Extended Bernoulli / energy equation in head form']),

    Calc('jet_force', 'Force of a jet on a fixed vane (F = ρQv(1 − cos β))',
         vars=[V('F', 'N', 'force on the vane in the jet direction'),
               V('rho', 'kg/m^3', 'fluid density', label='ρ'), V('Q', 'm^3/s', 'jet flow rate'),
               V('v', 'm/s', 'jet velocity'), V('beta', 'rad', 'deflection angle of the jet', hi=PI, label='β')],
         equations=['F = rho*Q*v*(1 - cos(beta))'], outputs=['F'],
         example={'rho': 1000.0, 'Q': 0.05, 'v': 20.0, 'beta': PI},
         keywords=['jet', 'vane', 'impulse', 'pelton'],
         assumptions=['Fixed vane, steady jet deflected through angle β from its original direction with no '
                      'speed loss. β = 90° (flat plate) gives F = ρQv; β = 180° (cup) gives F = 2ρQv.'],
         refs=['Momentum of a deflected jet']),

    Calc('momentum_bend', 'Force of a fluid stream on a pipe bend, nozzle or vane (2-D)',
         vars=[V('rho', 'kg/m^3', 'density', label='ρ'),
               V('inlet', '', '[v (m/s), p (Pa, gauge), A (m²), theta (deg from +x)]', kind='json'),
               V('outlet', '', '[v (m/s), p (Pa, gauge), A (m²), theta (deg from +x)]', kind='json'),
               V('Fx', 'N', 'x component of the force of the fluid on the bend', signed=True),
               V('Fy', 'N', 'y component of the force of the fluid on the bend', signed=True),
               V('F', 'N', 'resultant force magnitude'), V('theta_F', 'rad', 'direction of F from +x', signed=True),
               V('mdot', 'kg/s', 'mass flow rate (from the inlet)')],
         procedure=_momentum_bend, outputs=['Fx', 'Fy', 'F', 'theta_F', 'mdot'],
         example={'rho': 1000.0, 'inlet': [5, 150000, 0.02, 0], 'outlet': [5, 130000, 0.02, 90]},
         keywords=['momentum', 'pipe bend', 'nozzle', 'reaction force'],
         assumptions=['Steady flow; the force reported is that exerted BY the fluid ON the bend (the anchoring '
                      'force is equal and opposite). Steady flow requires ρ·A1·v1 = ρ·A2·v2; mdot uses the '
                      'inlet values. Gauge pressures.'],
         refs=['Control-volume momentum, F = p1A1e1 − p2A2e2 + ṁ(v1e1 − v2e2)']),

    Calc('turbine_torque', 'Euler turbomachine torque (T = ṁ(r₂v_t2 − r₁v_t1))',
         vars=[V('T', 'N*m', 'torque on the fluid (negative for a turbine)', signed=True), V('mdot', 'kg/s', 'mass flow rate'),
               V('r1', 'm', 'inlet radius'), V('r2', 'm', 'outlet radius'),
               V('vt1', 'm/s', 'tangential velocity component at inlet', signed=True),
               V('vt2', 'm/s', 'tangential velocity component at outlet', signed=True)],
         equations=['T = mdot*(r2*vt2 - r1*vt1)'], outputs=['T'],
         example={'mdot': 2.0, 'r1': 0.1, 'r2': 0.3, 'vt1': 5.0, 'vt2': 15.0},
         keywords=['euler turbine equation', 'turbomachine', 'angular momentum'],
         assumptions=['T is the torque the fluid exerts on the rotor (Euler turbomachine equation). For a '
                      'turbine the fluid loses angular momentum (r₂v_t2 < r₁v_t1), so T is negative; the shaft '
                      'delivers the equal-and-opposite torque to the load. For a pump the signs reverse.'],
         refs=['Euler turbomachine (angular momentum) equation']),

    Calc('sprinkler_torque', 'Rotating sprinkler arm torque (T = ρQr(v_j − ωr))',
         vars=[V('T', 'N*m', 'resisting torque', signed=True),
               V('rho', 'kg/m^3', 'density', label='ρ'), V('Q', 'm^3/s', 'flow rate'),
               V('r', 'm', 'arm radius (jet radius)'), V('v_j', 'm/s', 'jet velocity relative to the arm'),
               V('omega', 'rad/s', 'angular speed of the arm', label='ω')],
         equations=['T = rho*Q*r*(v_j - omega*r)'], outputs=['T'],
         example={'rho': 1000.0, 'Q': 0.001, 'r': 0.2, 'v_j': 10.0, 'omega': 20.0},
         keywords=['sprinkler', 'lawn', 'angular momentum', 'reaction'],
         assumptions=['Two opposed jets at radius r; T is the torque needed to hold the rotation speed ω '
                      '(T = 0 is the free-running speed ω = v_j/r).'],
         refs=['Angular momentum of a rotating sprinkler']),

    # ---- dimensionless numbers --------------------------------------------------------------------
    Calc('reynolds_number', 'Reynolds number (Re = ρvL/μ)',
         vars=[V('Re', '', 'Reynolds number'), V('rho', 'kg/m^3', 'density', label='ρ'),
               V('v', 'm/s', 'velocity'), V('L', 'm', 'characteristic length'),
               V('mu', 'Pa*s', 'dynamic viscosity', label='μ')],
         equations=['Re = rho*v*L/mu'], outputs=['Re'],
         example={'rho': 1000.0, 'v': 1.0, 'L': 0.01, 'mu': 0.001},
         keywords=['reynolds', 'inertia', 'viscous'],
         refs=['Re = ρvL/μ']),

    Calc('froude_number', 'Froude number (Fr = v/√(gL))',
         vars=[V('Fr', '', 'Froude number'), V('v', 'm/s', 'velocity'), V('g', 'm/s^2', 'gravitational acceleration'),
               V('L', 'm', 'characteristic length')],
         equations=['Fr = v/sqrt(g*L)'], outputs=['Fr'],
         example={'v': 5.0, 'g': G0, 'L': 1.0},
         keywords=['froude', 'free surface', 'open channel'],
         refs=['Fr = v/√(gL)']),

    Calc('mach_number', 'Mach number (Ma = v/c)',
         vars=[V('Ma', '', 'Mach number'), V('v', 'm/s', 'flow velocity'), V('c', 'm/s', 'local speed of sound')],
         equations=['Ma = v/c'], outputs=['Ma'],
         example={'v': 170.0, 'c': 340.0},
         keywords=['mach', 'compressible', 'sonic'],
         refs=['Ma = v/c']),

    Calc('weber_number', 'Weber number (We = ρv²L/σ)',
         vars=[V('We', '', 'Weber number'), V('rho', 'kg/m^3', 'density', label='ρ'),
               V('v', 'm/s', 'velocity'), V('L', 'm', 'characteristic length'),
               V('sigma', 'N/m', 'surface tension', label='σ')],
         equations=['We = rho*v**2*L/sigma'], outputs=['We'],
         example={'rho': 1000.0, 'v': 2.0, 'L': 0.01, 'sigma': 0.0728},
         keywords=['weber', 'surface tension', 'droplet'],
         refs=['We = ρv²L/σ']),

    Calc('euler_number', 'Euler number (Eu = Δp/(ρv²))',
         vars=[V('Eu', '', 'Euler number'), V('dp', 'Pa', 'pressure difference', label='Δp'),
               V('rho', 'kg/m^3', 'density', label='ρ'), V('v', 'm/s', 'velocity')],
         equations=['Eu = dp/(rho*v**2)'], outputs=['Eu'],
         example={'dp': 4000.0, 'rho': 1000.0, 'v': 2.0},
         keywords=['euler number', 'pressure coefficient'],
         refs=['Eu = Δp/(ρv²)']),

    Calc('strouhal_number', 'Strouhal number (St = fL/v)',
         vars=[V('St', '', 'Strouhal number'), V('f', 'Hz', 'shedding (cyclic) frequency'),
               V('L', 'm', 'characteristic length'), V('v', 'm/s', 'velocity')],
         equations=['St = f*L/v'], outputs=['St'],
         example={'f': 10.0, 'L': 0.05, 'v': 2.0},
         keywords=['strouhal', 'vortex shedding', 'frequency'],
         assumptions=['f is a cyclic frequency (Hz); angular frequencies must be converted first.'],
         refs=['St = fL/v']),

    # ---- pipe flow --------------------------------------------------------------------------------
    Calc('hagen_poiseuille', 'Laminar pipe flow (Hagen–Poiseuille) and velocity profile',
         vars=[V('Q', 'm^3/s', 'volumetric flow rate'), V('u_avg', 'm/s', 'mean velocity'),
               V('u_max', 'm/s', 'centreline velocity'),
               V('dp', 'Pa', 'pressure drop over length L', label='Δp'),
               V('D', 'm', 'pipe diameter'), V('mu', 'Pa*s', 'dynamic viscosity', label='μ'), V('L', 'm', 'pipe length')],
         equations=['Q = pi*D**4*dp/(128*mu*L)', 'u_avg = Q/(pi*D**2/4)', 'u_max = 2*u_avg'],
         outputs=['Q', 'u_avg', 'u_max'],
         example={'dp': 1000.0, 'D': 0.01, 'mu': 0.001, 'L': 1.0},
         keywords=['hagen-poiseuille', 'laminar', 'parabolic profile'],
         assumptions=['Fully developed laminar flow (Re < 2300); parabolic profile u(r) = u_max(1 − (r/R)²) '
                      'with u_max = 2·u_avg at the centreline.'],
         refs=['Hagen–Poiseuille: Q = πΔpD⁴/(128μL)']),

    Calc('friction_factor', 'Darcy friction factor: laminar, Colebrook, Swamee–Jain, Haaland',
         vars=[V('Re', '', 'Reynolds number'), V('rel_rough', '', 'relative roughness ε/D', label='ε/D'),
               V('f_colebrook', '', 'Colebrook friction factor (iterated)'),
               V('f_swamee_jain', '', 'Swamee–Jain friction factor'),
               V('f_haaland', '', 'Haaland friction factor'), V('f_laminar', '', 'laminar friction factor 64/Re')],
         procedure=_friction_factor, outputs=['f_colebrook', 'f_swamee_jain', 'f_haaland', 'f_laminar'],
         example={'Re': 1e5, 'rel_rough': 0.001},
         plots=[{'id': 'moody', 'title': 'Moody chart', 'fn': _moody_chart}],
         keywords=['moody', 'colebrook', 'swamee-jain', 'haaland', 'darcy', 'roughness'],
         assumptions=['Use f_laminar for Re < 2300 and f_colebrook (or the closed forms) for Re > 4000; '
                      '2300 < Re < 4000 is transitional and uncertain. Below Re = 2300 the Colebrook/Swamee–Jain/'
                      'Haaland correlations are turbulent relations and are misleading — use f_laminar there. '
                      'Swamee–Jain is valid for 5000 < Re < 1e8 and 1e-6 < ε/D < 0.01; Haaland is an explicit fit '
                      'accurate to ~2% of Colebrook.'],
         refs=['Colebrook–White equation; Swamee–Jain (1976); Haaland (1983) explicit approximation']),

    Calc('head_loss', 'Major and minor head losses and equivalent length',
         vars=[V('h_f', 'm', 'major (friction) head loss'), V('h_m', 'm', 'minor (fitting) head loss'),
               V('h_total', 'm', 'total head loss'), V('L_eq', 'm', 'equivalent length of the fitting (K·D/f)'),
               V('f', '', 'Darcy friction factor'), V('L', 'm', 'pipe length'), V('D', 'm', 'pipe diameter'),
               V('v', 'm/s', 'mean velocity'), V('g', 'm/s^2', 'gravitational acceleration'),
               V('K', '', 'minor-loss coefficient (fitting table)')],
         equations=['h_f = f*L*v**2/(2*g*D)', 'h_m = K*v**2/(2*g)', 'h_total = h_f + h_m', 'L_eq = K*D/f'],
         outputs=['h_f', 'h_m', 'h_total', 'L_eq'],
         example={'f': 0.02, 'L': 100.0, 'D': 0.05, 'v': 2.0, 'g': G0, 'K': 2.0},
         keywords=['darcy-weisbach', 'minor loss', 'equivalent length', 'fittings'],
         assumptions=['K is the tabulated minor-loss coefficient for the fitting (user input); equivalent '
                      'length L_eq gives the same loss as the fitting at the same f.'],
         refs=['Darcy–Weisbach h_f = f(L/D)v²/(2g); h_m = Kv²/(2g)']),

    Calc('hydraulic_diameter', 'Hydraulic diameter for non-circular ducts (D_h = 4A/P)',
         vars=[V('D_h', 'm', 'hydraulic diameter'), V('A', 'm^2', 'flow cross-sectional area'),
               V('P', 'm', 'wetted perimeter')],
         equations=['D_h = 4*A/P'], outputs=['D_h'],
         example={'A': 0.01, 'P': 0.4},
         keywords=['duct', 'wetted perimeter', 'non-circular'],
         assumptions=['D_h substituted into Re and h_f relations approximates non-circular ducts (exact for a '
                      'full circular pipe: D_h = D).'],
         refs=['D_h = 4A/P']),

    Calc('series_pipes', 'Pipes in series: total head loss',
         vars=[V('pipes', '', 'rows [L (m), D (m), f, K] for each pipe in series', kind='json'),
               V('Q', 'm^3/s', 'common flow rate through all pipes'),
               V('h_total', 'm', 'total head loss'), V('h_each', '', 'head loss of each pipe [m]', kind='json')],
         procedure=_series_pipes, outputs=['h_total', 'h_each'],
         example={'pipes': [[50, 0.05, 0.02, 0.5], [30, 0.04, 0.02, 1.0]], 'Q': 0.002},
         keywords=['series', 'pipes in series'],
         assumptions=['Same flow rate through every pipe; losses add. Parallel pipes and general networks use '
                      'the pipe-network (Hardy Cross) calculator.'],
         refs=['Series pipe losses: h = Σ (f L/D + K) V²/(2g)']),

    Calc('pipe_network', 'Pipe network by the Hardy Cross loop method',
         vars=[V('pipes', '', 'rows [id, L (m), D (m), f, K, Q0 (m³/s)]; Q0 satisfy junction continuity', kind='json'),
               V('loops', '', '[[[id, dir], ...], ...] with dir +1/−1 around each loop', kind='json'),
               V('flows', '', '[[id, Q (m³/s), head loss (m)], ...]', kind='json'),
               V('iterations', '', 'number of Hardy Cross iterations', integer=True),
               V('max_correction', 'm^3/s', 'largest loop flow correction of the last iteration')],
         procedure=_pipe_network, outputs=['flows', 'iterations', 'max_correction'],
         example={'pipes': [['p1', 100, 0.2, 0.02, 0, 0.3], ['p2', 150, 0.15, 0.02, 0, 0.2]],
                  'loops': [[['p1', 1], ['p2', -1]]]},
         keywords=['hardy cross', 'network', 'parallel pipes', 'distribution'],
         assumptions=['Hardy Cross loop balancing with fixed friction factors f and minor-loss K; initial flows '
                      'must satisfy continuity at every junction (the corrections preserve it). Converges to '
                      'Σh = 0 around each loop. f may be taken from the friction-factor calculator.'],
         refs=['Hardy Cross method (1936) for pipe networks']),

    # ---- external flow ----------------------------------------------------------------------------
    Calc('boundary_layer', 'Flat-plate boundary layer: thickness, displacement and momentum thickness',
         vars=[V('rho', 'kg/m^3', 'density', label='ρ'), V('U', 'm/s', 'freestream velocity'),
               V('x', 'm', 'distance from the leading edge'), V('mu', 'Pa*s', 'dynamic viscosity', label='μ'),
               V('Re_x', '', 'local Reynolds number ρUx/μ'),
               V('delta_lam', 'm', 'laminar boundary-layer thickness (99%)'),
               V('delta_star_lam', 'm', 'laminar displacement thickness'),
               V('theta_lam', 'm', 'laminar momentum thickness'),
               V('delta_turb', 'm', 'turbulent boundary-layer thickness (1/7 power law)'),
               V('delta_star_turb', 'm', 'turbulent displacement thickness'),
               V('theta_turb', 'm', 'turbulent momentum thickness'),
               V('cf_lam', '', 'laminar local skin-friction coefficient'),
               V('cf_turb', '', 'turbulent local skin-friction coefficient'),
               V('cf_lam_mean', '', 'laminar mean skin-friction coefficient 1.328/√Re_x'),
               V('cf_turb_mean', '', 'turbulent mean skin-friction coefficient 0.074/Re_x^0.2')],
         procedure=_boundary_layer,
         outputs=['Re_x', 'delta_lam', 'delta_star_lam', 'theta_lam', 'delta_turb', 'delta_star_turb',
                  'theta_turb', 'cf_lam', 'cf_turb', 'cf_lam_mean', 'cf_turb_mean'],
         example={'rho': 1000.0, 'U': 1.0, 'x': 0.1, 'mu': 0.001},
         keywords=['boundary layer', 'blasius', 'displacement thickness', 'momentum thickness'],
         assumptions=['Laminar: exact Blasius flat-plate solution (δ = 5.0x/√Re_x, δ* = 1.7208x/√Re_x, '
                      'θ = 0.664x/√Re_x, local C_f = 0.664/√Re_x, mean C_f = 1.328/√Re_x). Turbulent: '
                      '1/7-power-law approximation (δ = 0.37x/Re_x^0.2, δ* = δ/8, θ = 7δ/72, local '
                      'C_f = 0.0592/Re_x^0.2, mean C_f = 0.074/Re_x^0.2) for a smooth plate, Re_x up to ~1e7; '
                      'constants vary slightly between texts. Transition near Re_x ≈ 5e5.'],
         refs=['Blasius flat-plate solution; 1/7-power-law turbulent boundary layer']),

    Calc('drag_lift', 'Dynamic pressure, drag, lift and skin-friction force',
         vars=[V('q', 'Pa', 'dynamic pressure ρV²/2'), V('F_D', 'N', 'drag force'), V('F_L', 'N', 'lift force'),
               V('F_sf', 'N', 'skin-friction drag force'),
               V('C_D', '', 'drag coefficient'), V('C_L', '', 'lift coefficient'), V('C_f', '', 'skin-friction coefficient'),
               V('rho', 'kg/m^3', 'density', label='ρ'), V('V', 'm/s', 'freestream velocity'),
               V('A', 'm^2', 'reference (frontal/planform) area'), V('A_wet', 'm^2', 'wetted area for skin friction')],
         equations=['q = rho*V**2/2', 'F_D = C_D*q*A', 'F_L = C_L*q*A', 'F_sf = C_f*q*A_wet'],
         outputs=['q', 'F_D', 'F_L', 'F_sf'],
         example={'C_D': 0.3, 'C_L': 0.5, 'C_f': 0.003, 'rho': 1.225, 'V': 30.0, 'A': 2.0, 'A_wet': 10.0},
         keywords=['drag', 'lift', 'skin friction', 'dynamic pressure'],
         assumptions=['C_D, C_L and C_f are user inputs (from tables/data for the body shape and Re); A is the '
                      'reference area appropriate to each coefficient. C_f is the mean (integrated) skin-friction '
                      'coefficient over the wetted area.'],
         refs=['Drag/lift: F = C·½ρV²A']),

    Calc('terminal_velocity', 'Terminal velocity of a falling body (v_t = √(2mg/(ρC_D A)))',
         vars=[V('v_t', 'm/s', 'terminal velocity'), V('m', 'kg', 'mass'), V('g', 'm/s^2', 'gravitational acceleration'),
               V('rho', 'kg/m^3', 'fluid density', label='ρ'), V('C_D', '', 'drag coefficient'),
               V('A', 'm^2', 'frontal area')],
         equations=['v_t = sqrt(2*m*g/(rho*C_D*A))'], outputs=['v_t'],
         example={'m': 70.0, 'g': G0, 'rho': 1.225, 'C_D': 1.0, 'A': 0.7},
         keywords=['terminal velocity', 'falling', 'settling', 'skydiver'],
         assumptions=['Drag balances weight (buoyancy neglected); constant C_D.'],
         refs=['Terminal velocity from force balance mg = ½ρv²C_D A']),

    # ---- flow measurement -------------------------------------------------------------------------
    Calc('pitot_tube', 'Pitot tube velocity (v = √(2Δp/ρ))',
         vars=[V('v', 'm/s', 'flow velocity'), V('dp', 'Pa', 'dynamic pressure (stagnation − static)', label='Δp'),
               V('rho', 'kg/m^3', 'fluid density', label='ρ')],
         equations=['v = sqrt(2*dp/rho)'], outputs=['v'],
         example={'dp': 500.0, 'rho': 1.225},
         keywords=['pitot', 'pitot-static', 'stagnation'],
         assumptions=['Incompressible flow; the stagnation point brings the fluid to rest isentropically.'],
         refs=['Bernoulli: v = √(2Δp/ρ)']),

    Calc('obstruction_flowmeter', 'Orifice / venturi / nozzle flow rate (Q = C_d A₂√(2Δp/(ρ(1−β⁴))))',
         vars=[V('Q', 'm^3/s', 'flow rate'), V('C_d', '', 'discharge coefficient (0.6…1.0, table)'),
               V('A_2', 'm^2', 'throat (minimum) area'), V('dp', 'Pa', 'pressure difference across the meter', label='Δp'),
               V('rho', 'kg/m^3', 'fluid density', label='ρ'), V('beta', '', 'diameter ratio d/D', hi=1.0, label='β')],
         equations=['Q = C_d*A_2*sqrt(2*dp/(rho*(1 - beta**4)))'], outputs=['Q'],
         example={'C_d': 0.98, 'A_2': 0.0005, 'dp': 20000.0, 'rho': 1000.0, 'beta': 0.5},
         keywords=['orifice', 'venturi', 'nozzle', 'flow meter', 'discharge coefficient'],
         assumptions=['C_d accounts for losses and the vena-contracta (user input from meter tables); '
                      'β = d/D is the diameter ratio.'],
         refs=['Obstruction-meter flow equation (Bernoulli + continuity)']),

    Calc('rotameter', 'Rotameter (variable-area meter) flow rate',
         vars=[V('Q', 'm^3/s', 'flow rate'), V('C_d', '', 'float discharge coefficient'),
               V('A_t', 'm^2', 'tube cross-sectional area'), V('A_f', 'm^2', 'float frontal area'),
               V('g', 'm/s^2', 'gravitational acceleration'), V('V_f', 'm^3', 'float volume'),
               V('rho_f', 'kg/m^3', 'float density'), V('rho', 'kg/m^3', 'fluid density', label='ρ')],
         equations=['Q = C_d*(A_t - A_f)*sqrt(2*g*V_f*(rho_f - rho)/(rho*A_f))'], outputs=['Q'],
         example={'C_d': 0.95, 'A_t': 0.0003, 'A_f': 0.0002, 'g': G0, 'V_f': 5e-6, 'rho_f': 8000.0, 'rho': 1000.0},
         keywords=['rotameter', 'variable area', 'float'],
         assumptions=['Float held up by drag against its buoyant weight; C_d is a user input.'],
         refs=['Rotameter force balance Q = C_d (A_t − A_f) √(2gV_f(ρ_f − ρ)/(ρA_f))']),

    # ---- stream function, vorticity, circulation ---------------------------------------------------
    Calc('point_vortex', 'Point vortex: stream function and velocity potential',
         vars=[V('psi', 'm^2/s', 'stream function', signed=True, label='ψ'),
               V('phi', 'm^2/s', 'velocity potential', signed=True, label='φ'),
               V('Gamma', 'm^2/s', 'circulation (vortex strength)', label='Γ'), V('r', 'm', 'radial distance from the vortex'),
               V('r0', 'm', 'reference radius where ψ = 0'), V('theta', 'rad', 'angle about the vortex', hi=2 * PI, label='θ')],
         equations=['psi = -Gamma/(2*pi)*log(r/r0)', 'phi = Gamma/(2*pi)*theta'], outputs=['psi', 'phi'],
         example={'Gamma': 2 * PI, 'r': 2.0, 'r0': 1.0, 'theta': 0.5},
         keywords=['stream function', 'velocity potential', 'irrotational vortex'],
         assumptions=['2-D point vortex at the origin; tangential velocity v_θ = Γ/(2πr) (see the circulation '
                      'calculator) and vorticity is zero everywhere except at the origin (free vortex). With the '
                      'stream function defined so that v_θ = −∂ψ/∂r, ψ = −(Γ/2π) ln(r/r0), fixed by ψ = 0 at '
                      'r = r0; ψ is negative for r > r0 and positive for r < r0 (both signs are physical).'],
         refs=['Point-vortex stream function and potential, ψ = −(Γ/2π) ln(r/r0)']),

    Calc('forced_vortex', 'Forced (solid-body rotation) vortex: velocity and vorticity',
         vars=[V('v_theta', 'm/s', 'tangential velocity at radius r'), V('zeta', 'rad/s', 'vorticity (2ω)', label='ζ'),
               V('omega', 'rad/s', 'angular speed', label='ω'), V('r', 'm', 'radius')],
         equations=['v_theta = omega*r', 'zeta = 2*omega'], outputs=['v_theta', 'zeta'],
         example={'omega': 5.0, 'r': 0.2},
         keywords=['forced vortex', 'solid body rotation', 'vorticity'],
         assumptions=['Solid-body rotation: vorticity ζ = 2ω is uniform (radian is dimensionless, so 1 rad/s = 1 s⁻¹).'],
         refs=['Vorticity of solid-body rotation, ζ = 2Ω']),

    Calc('circulation', 'Circulation (Γ = 2πr·v_θ)',
         vars=[V('Gamma', 'm^2/s', 'circulation', label='Γ'), V('r', 'm', 'radius'),
               V('v_theta', 'm/s', 'tangential velocity at radius r')],
         equations=['Gamma = 2*pi*r*v_theta'], outputs=['Gamma'],
         example={'r': 0.2, 'v_theta': 5.0},
         keywords=['circulation', 'vortex', 'kutta'],
         refs=['Circulation Γ = 2πr·v_θ for an axisymmetric flow']),

    # ---- water hammer, dimensional analysis --------------------------------------------------------
    Calc('water_hammer', 'Water hammer pressure rise (Joukowsky)',
         vars=[V('c', 'm/s', 'wave (pressure) speed'), V('dp_wh', 'Pa', 'pressure change (rise positive)', signed=True, label='Δp'),
               V('rho', 'kg/m^3', 'fluid density', label='ρ'), V('K', 'Pa', 'fluid bulk modulus'),
               V('E', 'Pa', 'pipe-wall elastic modulus'), V('D', 'm', 'pipe diameter'), V('t', 'm', 'pipe wall thickness'),
               V('dV', 'm/s', 'velocity change (sudden closure → negative)', signed=True)],
         equations=['c = sqrt(K/(rho*(1 + K*D/(E*t))))', 'dp_wh = -rho*c*dV'], outputs=['c', 'dp_wh'],
         example={'rho': 1000.0, 'K': 2.2e9, 'E': 200e9, 'D': 0.1, 't': 0.005, 'dV': -1.0},
         keywords=['joukowsky', 'water hammer', 'surge', 'pipe'],
         assumptions=['Sudden velocity change dV; elastic pipe (a rigid pipe is the limit E → ∞, c = √(K/ρ)). '
                      'Joukowsky Δp = −ρ c dV is positive (a rise) for a sudden closure (dV < 0) and negative '
                      '(a drop) for a sudden opening (dV > 0).'],
         refs=['Joukowsky: Δp = −ρ c dV; elastic-pipe wave speed']),

    Calc('buckingham_pi', 'Buckingham π theorem: dimensionless groups from a dimension matrix',
         vars=[V('matrix', '', 'rows = fundamental dimensions (M, L, T, …), columns = quantities, integer exponents; '
                              'first `rank` columns are the repeating variables', kind='json'),
               V('pi_groups', '', 'list of π groups (exponent vectors over the columns)', kind='json'),
               V('n_pi', '', 'number of π groups', integer=True), V('rank', '', 'rank of the dimension matrix', integer=True)],
         procedure=_buckingham, outputs=['pi_groups', 'n_pi', 'rank'],
         example={'matrix': [[0, 0, 1, 1, 1], [1, 1, -3, 1, -1], [-1, 0, 0, -2, -1]]},
         keywords=['buckingham', 'pi theorem', 'dimensional analysis', 'similitude'],
         assumptions=['The dimension matrix rows are the fundamental dimensions and columns the quantities; the '
                      'first r columns must be independent repeating variables. Each π group = one non-repeating '
                      'quantity times powers of the repeating variables (the nullspace of the matrix).'],
         refs=['Buckingham π theorem (1914); nullspace of the dimension matrix']),
]

# DESCRIPTION: Heat transfer section: conduction and thermal resistances, composite walls, overall U,
#              heat generation, critical insulation, fins, lumped and one-term transient conduction,
#              semi-infinite solids, 2-D shape factors and finite differences, dimensionless groups,
#              forced and natural convection correlations, boiling and condensation, thermal radiation
#              (blackbody, band fractions, exchange, shields) and heat-exchanger sizing.
"""Heat transfer calculators.

Sign conventions: heat rate q is positive in the direction of decreasing temperature (from surface 1 to
surface 2, from T_s to T_inf, from the hot to the cold side). Radiation temperatures are absolute (K).

Relations are the standard ones in heat-transfer texts (e.g. Incropera & DeWitt, *Fundamentals of Heat
and Mass Transfer*; Çengel, *Heat Transfer*). They are cited by relation name, recalled rather than
re-read; correlation constants from charts/tables (C, m, n, Kt, C_sf, ...) are user inputs whose ``desc``
names the source table. Boiling/condensation fluid properties (h_fg, ρ_l, ρ_v, σ, μ_l, c_p,l, Pr_l) are
user inputs — the humid-air/LMTD adapters in common.py are used where they apply (heat exchangers).
"""
import math

from scipy import integrate, optimize, special

from engineering_sim import correlations
from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import common

SECTION = section('heat_transfer', 'Heat transfer', order=320,
                  summary='Conduction, resistances, fins, transient conduction, convection correlations, '
                          'boiling/condensation, thermal radiation and heat exchangers.')

PI = math.pi
G0 = common.g0()  # 9.80665 m/s² (pre-fill for buoyancy/bubble examples)

# Fundamental radiation constants (CODATA 2018 / standard values, recalled).
SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann constant, W/(m²·K⁴)
C1_RAD = 3.741771852e-16    # first radiation constant 2πhc², W·m²
C2_RAD = 1.438776877e-2     # second radiation constant hc/k, m·K
WIEN_B = 2.897771955e-3     # Wien displacement constant, m·K


# ---- procedures ------------------------------------------------------------------------------------
def _fin_straight(values):
    """Straight rectangular fin with a convective tip (exact solution).

    P = 2(w + t), A_c = w·t, m = sqrt(hP/(kA_c)). The exact convective-tip heat rate is
    Q = sqrt(hPkA_c)·θ_b·(sinh(mL) + (h/mk)cosh(mL)) / (cosh(mL) + (h/mk)sinh(mL)); efficiency
    η = Q/(h A_f θ_b) with A_f = P·L + A_c (side area plus tip face); effectiveness
    ε = Q/(h A_c θ_b); tip temperature from the exact profile θ_tip/θ_b = 1/(cosh(mL) + (h/mk)sinh(mL)).
    """
    L, t, w = values['L'], values['t'], values['w']
    k, h = values['k'], values['h']
    Tb, Tinf = values['T_b'], values['T_inf']
    if min(L, t, w) <= 0:
        raise MechError('fin length, thickness and width must be positive')
    if k <= 0 or h <= 0:
        raise MechError('k and h must be positive')
    P = 2.0 * (w + t)
    Ac = w * t
    m = math.sqrt(h * P / (k * Ac))
    N = h / (m * k)                       # dimensionless (h/mk)
    mL = m * L
    sinh_mL = math.sinh(mL)
    cosh_mL = math.cosh(mL)
    Q = math.sqrt(h * P * k * Ac) * (Tb - Tinf) * (sinh_mL + N * cosh_mL) / (cosh_mL + N * sinh_mL)
    Af = P * L + Ac                       # side area + tip face (all fin surfaces)
    eta = Q / (h * Af * (Tb - Tinf))
    eps = Q / (h * Ac * (Tb - Tinf))
    Ttip = Tinf + (Tb - Tinf) / (cosh_mL + N * sinh_mL)
    return {'Q_fin': Q, 'm': m, 'eta': eta, 'eps': eps, 'T_tip': Ttip}


def _one_term_plane(values):
    """First-term (Fo > 0.2) transient solution for a plane wall of half-thickness L in convection.

    Eigenvalue ζ1 solves ζ tan ζ = Bi on (0, π/2); C1 = 4 sin ζ1/(2ζ1 + sin 2ζ1). Centre and surface
    temperatures follow θ = C1 exp(-ζ1² Fo) cos(ζ1 x/L).
    """
    Ti, Tinf = values['T_i'], values['T_inf']
    L, k, h = values['L'], values['k'], values['h']
    alpha, t = values['alpha'], values['t']
    if min(L, k, h, alpha, t) <= 0:
        raise MechError('L, k, h, alpha and t must be positive')
    Bi = h * L / k
    zeta = optimize.brentq(lambda z: z * math.tan(z) - Bi, 1e-9, PI / 2 - 1e-9)
    C1 = 4.0 * math.sin(zeta) / (2.0 * zeta + math.sin(2.0 * zeta))
    Fo = alpha * t / L ** 2
    theta_c = C1 * math.exp(-zeta ** 2 * Fo)
    theta_s = theta_c * math.cos(zeta)
    warns = []
    if Fo < 0.2:
        warns.append(f'Fo = {Fo:.3g} < 0.2: the one-term approximation is inaccurate (at Bi = 1, Fo = 0.05 '
                     'it can even exceed the initial temperature); use more series terms')
    return {'T_center': Tinf + (Ti - Tinf) * theta_c,
            'T_surface': Tinf + (Ti - Tinf) * theta_s,
            'zeta1': zeta, 'C1': C1, 'Bi': Bi, 'Fo': Fo, '_warnings': warns}


def _one_term_cylinder(values):
    """First-term (Fo > 0.2) transient solution for an infinite cylinder of radius r0 in convection.

    Eigenvalue ζ1 solves ζ J1(ζ)/J0(ζ) = Bi on (0, 2.4048); C1 = 2 J1(ζ)/(ζ (J0² + J1²)). Centre and
    surface temperatures follow θ = C1 exp(−ζ1² Fo) J0(ζ r/r0).
    """
    Ti, Tinf = values['T_i'], values['T_inf']
    r0, k, h = values['r0'], values['k'], values['h']
    alpha, t = values['alpha'], values['t']
    if min(r0, k, h, alpha, t) <= 0:
        raise MechError('r0, k, h, alpha and t must be positive')
    Bi = h * r0 / k
    j0_first = special.jn_zeros(0, 1)[0]     # first zero of J0 ≈ 2.4048
    zeta = optimize.brentq(lambda z: z * special.j1(z) / special.j0(z) - Bi, 1e-9, j0_first - 1e-9)
    C1 = 2.0 * special.j1(zeta) / (zeta * (special.j0(zeta) ** 2 + special.j1(zeta) ** 2))
    Fo = alpha * t / r0 ** 2
    theta_c = C1 * math.exp(-zeta ** 2 * Fo)
    theta_s = theta_c * special.j0(zeta)
    warns = []
    if Fo < 0.2:
        warns.append(f'Fo = {Fo:.3g} < 0.2: the one-term approximation is inaccurate; use more series terms')
    return {'T_center': Tinf + (Ti - Tinf) * theta_c,
            'T_surface': Tinf + (Ti - Tinf) * theta_s,
            'zeta1': zeta, 'C1': C1, 'Bi': Bi, 'Fo': Fo, '_warnings': warns}


def _one_term_sphere(values):
    """First-term (Fo > 0.2) transient solution for a sphere of radius r0 in convection.

    Eigenvalue ζ1 solves 1 − ζ cot ζ = Bi on (0, π); C1 = 4 (sin ζ − ζ cos ζ)/(2ζ − sin 2ζ). Centre and
    surface temperatures follow θ = C1 exp(−ζ1² Fo) (sin(ζ r/r0)/(ζ r/r0)).
    """
    Ti, Tinf = values['T_i'], values['T_inf']
    r0, k, h = values['r0'], values['k'], values['h']
    alpha, t = values['alpha'], values['t']
    if min(r0, k, h, alpha, t) <= 0:
        raise MechError('r0, k, h, alpha and t must be positive')
    Bi = h * r0 / k
    zeta = optimize.brentq(lambda z: 1.0 - z / math.tan(z) - Bi, 1e-9, PI - 1e-9)
    C1 = 4.0 * (math.sin(zeta) - zeta * math.cos(zeta)) / (2.0 * zeta - math.sin(2.0 * zeta))
    Fo = alpha * t / r0 ** 2
    theta_c = C1 * math.exp(-zeta ** 2 * Fo)
    theta_s = theta_c * math.sin(zeta) / zeta
    warns = []
    if Fo < 0.2:
        warns.append(f'Fo = {Fo:.3g} < 0.2: the one-term approximation is inaccurate; use more series terms')
    return {'T_center': Tinf + (Ti - Tinf) * theta_c,
            'T_surface': Tinf + (Ti - Tinf) * theta_s,
            'zeta1': zeta, 'C1': C1, 'Bi': Bi, 'Fo': Fo, '_warnings': warns}


def _finite_difference_2d(values):
    """Steady 2-D conduction (Laplace) nodal temperatures by Gauss-Seidel on a square grid.

    Dirichlet temperatures on all four edges; the interior is iterated until the largest change is below
    1e-9 K. Grid is [nodes_x, nodes_y] with 3..50 nodes per side (interior nodes are the unknowns).
    """
    grid = values['grid']
    if not (isinstance(grid, list) and len(grid) == 2):
        raise MechError('grid: give [nodes_x, nodes_y]')
    nx, ny = int(grid[0]), int(grid[1])
    if not (3 <= nx <= 51 and 3 <= ny <= 51):
        raise MechError('grid: 3..51 nodes per side')
    if nx % 2 == 0 or ny % 2 == 0:
        raise MechError('grid: node counts must be odd so the centre node is well defined')
    Ttop, Tbot, Tleft, Tright = values['T_top'], values['T_bottom'], values['T_left'], values['T_right']
    T = [[0.0] * ny for _ in range(nx)]
    for i in range(nx):
        T[i][0] = Tbot
        T[i][ny - 1] = Ttop
    for j in range(ny):
        T[0][j] = Tleft
        T[nx - 1][j] = Tright
    T[0][0] = 0.5 * (Tbot + Tleft)
    T[0][ny - 1] = 0.5 * (Ttop + Tleft)
    T[nx - 1][0] = 0.5 * (Tbot + Tright)
    T[nx - 1][ny - 1] = 0.5 * (Ttop + Tright)
    avg = 0.25 * (Ttop + Tbot + Tleft + Tright)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            T[i][j] = avg
    iters = 0
    converged = False
    for iters in range(1, 50000):
        maxdiff = 0.0
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                new = 0.25 * (T[i - 1][j] + T[i + 1][j] + T[i][j - 1] + T[i][j + 1])
                d = abs(new - T[i][j])
                T[i][j] = new
                if d > maxdiff:
                    maxdiff = d
        if maxdiff < 1e-9:
            converged = True
            break
    T_center = T[nx // 2][ny // 2]
    warns = []
    if not converged:
        warns.append('Gauss–Seidel did not converge to 1e-9 K within 50000 iterations')
    return {'T_center': T_center, 'T_grid': [row[:] for row in T], 'iterations': iters, '_warnings': warns}


def _band_fraction(values):
    """Blackbody band fraction F(λ1→λ2) = ∫E_bλ dλ / (σ T⁴) by numeric integration of Planck's law."""
    T = values['T']
    l1, l2 = values['lambda1'], values['lambda2']
    if T <= 0 or l1 <= 0 or l2 <= l1:
        raise MechError('need T > 0 and 0 < lambda1 < lambda2')
    C1, C2 = C1_RAD, C2_RAD

    def spec(lam):
        x = C2 / (lam * T)
        if x > 700.0:                       # exp overflows; contribution is negligible
            return 0.0
        return C1 / (lam ** 5 * math.expm1(x))

    total, _ = integrate.quad(spec, l1, l2, limit=200)
    return {'F': total / (SIGMA_SB * T ** 4)}


def _plane_wall_series(values):
    """Composite plane wall: conduction + contact + convection resistances in series (per unit area).

    layers = [[L, k, R_c], ...] from the hot side; R_c is the contact resistance AFTER that layer.
    Temperatures are reported from the hot surface through each interface to the cold surface.
    """
    layers = values['layers']
    if not isinstance(layers, list) or not layers:
        raise MechError('layers: give [[L, k, R_c], ...] from the hot side')
    rows = []
    for row in layers:
        if len(row) != 3:
            raise MechError('layers: each row must be [L (m), k (W/m·K), R_c (m²·K/W)]')
        L, k, Rc = float(row[0]), float(row[1]), float(row[2])
        if L <= 0 or k <= 0 or Rc < 0:
            raise MechError('layers: need L > 0, k > 0 and R_c >= 0')
        rows.append((L, k, Rc))
    h1, h2 = values['h1'], values['h2']
    T1, T2 = values['T_inf1'], values['T_inf2']
    if h1 <= 0 or h2 <= 0:
        raise MechError('h1 and h2 must be positive')
    R_total = 1.0 / h1 + 1.0 / h2
    for L, k, Rc in rows:
        R_total += L / k + Rc
    q = (T1 - T2) / R_total
    temps = [T1 - q / h1]                  # hot surface
    for L, k, Rc in rows:
        temps.append(temps[-1] - q * (L / k))       # cold side of the layer
        if Rc > 0:
            temps.append(temps[-1] - q * Rc)        # after the contact resistance
    return {'q': q, 'R_total': R_total, 'T_interfaces': temps}


def _effectiveness_counter(values):
    """Counter-flow effectiveness ε = (1 − e^(−NTU(1−Cr)))/(1 − Cr e^(−NTU(1−Cr))), with the Cr = 1 limit
    ε = NTU/(1 + NTU).

    A procedure (rather than an equation) so the inverse of Cr is found by 1-D root finding: the closed
    form for Cr involves the W_−1 branch of the Lambert W function, which the symbolic solver resolves to
    the wrong (W_0) branch.
    """
    NTU, Cr = values['NTU'], values['Cr']
    if Cr < 0 or Cr > 1.0:
        raise MechError('Cr must satisfy 0 <= Cr <= 1')
    if abs(1.0 - Cr) < 1e-9:
        return {'eps': NTU / (1.0 + NTU)}
    a = math.exp(-NTU * (1.0 - Cr))
    return {'eps': (1.0 - a) / (1.0 - Cr * a)}


def _lmtd(values):
    """Log-mean temperature difference via common.lmtd (the shared adapter)."""
    return {'LMTD': common.lmtd(values['Th_in'], values['Th_out'], values['Tc_in'], values['Tc_out'],
                                flow=values['flow'])}


def _critical_insulation(values):
    """Critical insulation radius: r_cr = k/h for a cylinder, 2k/h for a sphere."""
    k, h = values['k'], values['h']
    if k <= 0 or h <= 0:
        raise MechError('k and h must be positive')
    n = 1.0 if values['geometry'] == 'cylinder' else 2.0
    return {'r_cr': n * k / h}


def _lumped_capacitance(values):
    """Lumped-capacitance transient temperature, time constant and Biot number.

    Bi = h L_c / k with L_c = V/A_s; the model is valid for Bi < 0.1, so a larger Biot number is reported
    as a warning rather than silently returning a suspect temperature.
    """
    Ti, Tinf = values['T_i'], values['T_inf']
    t = values['t']
    rho, V, cp = values['rho'], values['V'], values['c_p']
    h, As, k = values['h'], values['A_s'], values['k']
    if k <= 0:
        raise MechError('k must be positive')
    tau = rho * V * cp / (h * As)
    T = Tinf + (Ti - Tinf) * math.exp(-t / tau)
    Bi = h * (V / As) / k
    warns = []
    if Bi >= 0.1:
        warns.append(f'Bi = {Bi:.3g} >= 0.1: internal temperature gradients are not negligible; the '
                     'lumped-capacitance result is approximate')
    return {'T': T, 'tau': tau, 'Bi': Bi, '_warnings': warns}


def _dittus_boelter(values):
    """Dittus–Boelter turbulent internal-flow correlation.

    Nu = 0.023 Re^0.8 Pr^n with n = 0.4 (heating) or 0.3 (cooling); h = Nu k/D. Validity Re > 10000 and
    0.6 < Pr < 160 is checked and reported as warnings (not enforced) so off-range use is visible.
    """
    Re, Pr = values['Re'], values['Pr']
    k, D = values['k'], values['D']
    n = 0.4 if values['mode'] == 'heating' else 0.3
    Nu = correlations.nusselt_dittus_boelter(Re, Pr, n)
    h = Nu * k / D
    warns = []
    if Re < 10000:
        warns.append(f'Re = {Re:.3g} < 10000: Dittus–Boelter is a turbulent-flow relation (Re > 10000); '
                     'use a laminar correlation below Re = 2300')
    if not (0.6 < Pr < 160):
        warns.append(f'Pr = {Pr:.3g} outside the 0.6–160 validity range of Dittus–Boelter')
    return {'Nu': Nu, 'h': h, '_warnings': warns}


def _laminar_nu(values):
    """Fully developed laminar internal-flow Nusselt number (constant, independent of Re)."""
    return {'Nu': 3.66 if values['mode'] == 'const_T' else 4.36}


CALCS = [
    # ---- steady conduction and resistances ---------------------------------------------------------
    Calc('conduction_plane', "Fourier's law of conduction through a plane wall (q = kAΔT/L)",
         vars=[V('q', 'W', 'heat rate from surface 1 (T1) to surface 2 (T2)', signed=True),
               V('k', 'W/(m*K)', 'thermal conductivity'), V('A', 'm^2', 'area normal to the flow'),
               V('T1', 'K', 'temperature of surface 1 (hot side)'),
               V('T2', 'K', 'temperature of surface 2 (cold side)'), V('L', 'm', 'wall thickness')],
         equations=['q = k*A*(T1 - T2)/L'], outputs=['q'],
         example={'k': 0.78, 'A': 10.0, 'T1': 293.15, 'T2': 268.15, 'L': 0.2},
         keywords=['conduction', 'fourier', 'thermal conductivity', 'heat rate'],
         assumptions=['One-dimensional steady conduction, constant k; positive when T1 > T2.'],
         refs=["Fourier's law q = kAΔT/L"]),

    Calc('thermal_resistance_plane', 'Conduction resistance of a plane wall (R = L/(kA))',
         vars=[V('R', 'K/W', 'thermal resistance'), V('L', 'm', 'wall thickness'),
               V('k', 'W/(m*K)', 'thermal conductivity'), V('A', 'm^2', 'area')],
         equations=['R = L/(k*A)'], outputs=['R'],
         example={'L': 0.2, 'k': 0.78, 'A': 10.0},
         keywords=['resistance', 'conduction', 'wall'],
         refs=['R_cond = L/(kA)']),

    Calc('thermal_resistance_cylinder', 'Conduction resistance of a cylindrical shell (R = ln(r2/r1)/(2πkL))',
         vars=[V('R', 'K/W', 'thermal resistance'), V('r1', 'm', 'inner radius'), V('r2', 'm', 'outer radius'),
               V('k', 'W/(m*K)', 'thermal conductivity'), V('L', 'm', 'cylinder length')],
         equations=['R = log(r2/r1)/(2*pi*k*L)'], outputs=['R'],
         example={'r1': 0.05, 'r2': 0.06, 'k': 15.0, 'L': 1.0},
         keywords=['pipe', 'radial conduction', 'cylinder'],
         assumptions=['Radial conduction through a pipe/tube wall, r2 > r1.'],
         refs=['R_cyl = ln(r2/r1)/(2πkL)']),

    Calc('thermal_resistance_sphere', 'Conduction resistance of a spherical shell (R = (1/r1 − 1/r2)/(4πk))',
         vars=[V('R', 'K/W', 'thermal resistance'), V('r1', 'm', 'inner radius'), V('r2', 'm', 'outer radius'),
               V('k', 'W/(m*K)', 'thermal conductivity')],
         equations=['R = (1/r1 - 1/r2)/(4*pi*k)'], outputs=['R'],
         example={'r1': 0.2, 'r2': 0.25, 'k': 0.5},
         keywords=['sphere', 'radial conduction', 'tank'],
         assumptions=['Radial conduction through a spherical shell, r2 > r1.'],
         refs=['R_sph = (1/r1 − 1/r2)/(4πk)']),

    Calc('series_resistance', 'Total resistance of resistances in series (R = R1 + R2 + R3)',
         vars=[V('R', 'K/W', 'total resistance'), V('R1', 'K/W', 'first resistance'),
               V('R2', 'K/W', 'second resistance'), V('R3', 'K/W', 'third resistance (0 if unused)')],
         equations=['R = R1 + R2 + R3'], outputs=['R'],
         example={'R1': 0.05, 'R2': 0.12, 'R3': 0.03},
         keywords=['series', 'composite', 'resistance'],
         assumptions=['Resistances in series add; chain more of these for more layers.'],
         refs=['R_total = Σ R_i']),

    Calc('contact_resistance', 'Thermal contact resistance from a contact conductance (R_c = 1/(h_c A))',
         vars=[V('R_c', 'K/W', 'contact resistance'), V('h_c', 'W/(m^2*K)', 'contact conductance'),
               V('A', 'm^2', 'apparent contact area')],
         equations=['R_c = 1/(h_c*A)'], outputs=['R_c'],
         example={'h_c': 10000.0, 'A': 0.01},
         keywords=['contact conductance', 'interface', 'gap'],
         assumptions=['h_c is a tabulated contact conductance for the material pair, finish and interface '
                      'pressure (user input, table); it spans ~2000 (air gap) to ~25000 W/m²K (ground metal).'],
         refs=['R_c = 1/(h_c A)']),

    Calc('plane_wall_series', 'Composite plane wall: series layers with convection and contact resistances',
         vars=[V('layers', '', 'rows [L (m), k (W/m·K), R_c (m²·K/W)] per layer from the hot side; '
                               'R_c is the contact resistance after that layer (0 if none)', kind='json'),
               V('h1', 'W/(m^2*K)', 'convection coefficient on the hot side'),
               V('h2', 'W/(m^2*K)', 'convection coefficient on the cold side'),
               V('T_inf1', 'K', 'hot-side fluid temperature'), V('T_inf2', 'K', 'cold-side fluid temperature'),
               V('q', 'W/m^2', 'heat flux (hot to cold, per unit area)', signed=True),
               V('R_total', 'm^2*K/W', 'total resistance per unit area'),
               V('T_interfaces', '', 'temperatures [K] from the hot surface through each interface to the '
                                     'cold surface', kind='json')],
         procedure=_plane_wall_series,
         outputs=['q', 'R_total', 'T_interfaces'],
         example={'layers': [[0.1, 0.78, 0.0], [0.2, 0.026, 0.01], [0.05, 0.1, 0.0]],
                  'h1': 25.0, 'h2': 8.0, 'T_inf1': 293.15, 'T_inf2': 268.15},
         keywords=['composite wall', 'series', 'insulation', 'contact resistance'],
         assumptions=['One-dimensional steady conduction per unit area; convection resistances 1/h1 and 1/h2 '
                      'and the layer conduction/contact resistances add in series.'],
         refs=['q = (T_inf1 − T_inf2) / Σ R; R = L/k + R_c + 1/h']),

    Calc('overall_U_plane', 'Overall heat-transfer coefficient U for a plane wall (with fouling)',
         vars=[V('U', 'W/(m^2*K)', 'overall heat-transfer coefficient'),
               V('h_i', 'W/(m^2*K)', 'inside convection coefficient'), V('h_o', 'W/(m^2*K)', 'outside convection coefficient'),
               V('t', 'm', 'wall thickness'), V('k', 'W/(m*K)', 'wall thermal conductivity'),
               V('R_fi', 'm^2*K/W', 'inside fouling factor'), V('R_fo', 'm^2*K/W', 'outside fouling factor')],
         equations=['U = 1/(1/h_i + R_fi + t/k + R_fo + 1/h_o)'], outputs=['U'],
         example={'h_i': 2000.0, 'h_o': 200.0, 't': 0.002, 'k': 15.0, 'R_fi': 0.0002, 'R_fo': 0.0001},
         keywords=['overall heat transfer', 'fouling factor', 'u-value'],
         assumptions=['Plane wall with equal inside/outside areas; fouling factors R_f are tabulated (TEMA) '
                      'user inputs. For cylinders use per-area resistances with the appropriate areas.'],
         refs=['1/U = 1/h_i + R_fi + t/k + R_fo + 1/h_o']),

    # ---- internal generation, critical radius, fins -------------------------------------------------
    Calc('heat_gen_plane', 'Plane wall with uniform volumetric heat generation',
         vars=[V('T_s', 'K', 'surface temperature'), V('T_max', 'K', 'centre-line (maximum) temperature'),
               V('qdot', 'W/m^3', 'volumetric heat generation rate', label='q″′'),
               V('L', 'm', 'half-thickness of the wall'), V('k', 'W/(m*K)', 'thermal conductivity'),
               V('T_inf', 'K', 'ambient fluid temperature'), V('h', 'W/(m^2*K)', 'convection coefficient')],
         equations=['T_s = T_inf + qdot*L/h', 'T_max = T_s + qdot*L**2/(2*k)'], outputs=['T_s', 'T_max'],
         example={'qdot': 5e6, 'L': 0.02, 'k': 20.0, 'T_inf': 293.15, 'h': 100.0},
         keywords=['heat generation', 'nuclear fuel', 'electric heating'],
         assumptions=['Symmetric wall (same convection on both faces); steady state, constant k.'],
         refs=['T_s = T_inf + q″′L/h; T_max − T_s = q″′L²/(2k)']),

    Calc('critical_insulation', 'Critical insulation radius for a cylinder or sphere (r_cr = n·k/h)',
         vars=[V('geometry', '', 'geometry', kind='text', choices=['cylinder', 'sphere']),
               V('r_cr', 'm', 'critical radius'), V('k', 'W/(m*K)', 'insulation thermal conductivity'),
               V('h', 'W/(m^2*K)', 'external convection coefficient')],
         procedure=_critical_insulation, outputs=['r_cr'],
         example={'geometry': 'cylinder', 'k': 0.05, 'h': 10.0},
         keywords=['critical radius', 'insulation', 'pipe', 'sphere'],
         assumptions=['Critical radius r_cr = k/h for a cylinder and 2k/h for a sphere. Below r_cr, '
                      'adding insulation INCREASES heat loss.'],
         refs=['r_cr = k/h (cylinder); r_cr = 2k/h (sphere)']),

    Calc('fin_straight', 'Straight rectangular fin: efficiency, effectiveness, heat rate and tip temperature',
         vars=[V('Q_fin', 'W', 'heat transfer from one fin', signed=True),
               V('m', '1/m', 'fin parameter sqrt(hP/(kA_c))'),
               V('eta', '', 'fin efficiency', hi=1.0, label='η'),
               V('eps', '', 'fin effectiveness ε = Q_fin/(h A_c (T_b − T_inf))'),
               V('T_tip', 'K', 'tip temperature (convective-tip profile)'),
               V('L', 'm', 'fin length'), V('t', 'm', 'fin thickness'), V('w', 'm', 'fin width'),
               V('k', 'W/(m*K)', 'fin thermal conductivity'), V('h', 'W/(m^2*K)', 'convection coefficient'),
               V('T_b', 'K', 'base temperature'), V('T_inf', 'K', 'ambient temperature')],
         procedure=_fin_straight, outputs=['Q_fin', 'm', 'eta', 'eps', 'T_tip'],
         example={'L': 0.05, 't': 0.005, 'w': 0.1, 'k': 200.0, 'h': 50.0, 'T_b': 400.0, 'T_inf': 300.0},
         keywords=['fin', 'extended surface', 'efficiency', 'effectiveness'],
         assumptions=['Rectangular fin with a CONVECTIVE tip (exact solution, not the corrected-length '
                      'approximation); P = 2(w + t), A_c = w·t, and the exposed area A_f = P·L + A_c. A fin '
                      'is worthwhile when ε > 2.'],
         refs=['Convective-tip fin: Q = √(hPkA_c) θ_b (sinh mL + (h/mk) cosh mL)/(cosh mL + (h/mk) sinh mL)']),

    # ---- transient conduction -----------------------------------------------------------------------
    Calc('lumped_capacitance', 'Lumped-capacitance transient temperature, time constant and Biot number',
         vars=[V('T', 'K', 'temperature at time t'), V('tau', 's', 'thermal time constant ρVc_p/(hA_s)', label='τ'),
               V('Bi', '', 'Biot number h(V/A_s)/k'),
               V('T_i', 'K', 'initial temperature'), V('T_inf', 'K', 'ambient (fluid) temperature'),
               V('t', 's', 'elapsed time'), V('rho', 'kg/m^3', 'density', label='ρ'),
               V('V', 'm^3', 'volume'), V('c_p', 'J/(kg*K)', 'specific heat'), V('h', 'W/(m^2*K)', 'convection coefficient'),
               V('A_s', 'm^2', 'surface area'), V('k', 'W/(m*K)', 'solid thermal conductivity')],
         procedure=_lumped_capacitance, outputs=['T', 'tau', 'Bi'],
         no_invert={'k': 'affects only the Biot number (T and tau do not depend on k)'},
         example={'T_i': 400.0, 'T_inf': 300.0, 't': 60.0, 'rho': 7800.0, 'V': 0.0001, 'c_p': 480.0,
                  'h': 50.0, 'A_s': 0.02, 'k': 45.0},
         keywords=['lumped', 'biot number', 'transient', 'time constant', 'quenching'],
         assumptions=['Valid when Bi = hL_c/k < 0.1 (negligible internal gradients), L_c = V/A_s; a larger '
                      'Biot number is reported as a warning.'],
         refs=['Lumped capacitance: T = T_inf + (T_i − T_inf) e^(−t/τ), τ = ρVc_p/(hA_s); Bi = hL_c/k']),

    Calc('one_term_plane', 'One-term transient solution for a plane wall (Fo > 0.2)',
         vars=[V('T_center', 'K', 'centre temperature at time t'), V('T_surface', 'K', 'surface temperature at time t'),
               V('zeta1', 'rad', 'first eigenvalue ζ1 (root of ζ tan ζ = Bi)', label='ζ1'),
               V('C1', '', 'first-series coefficient C1'), V('Bi', '', 'Biot number hL/k'),
               V('Fo', '', 'Fourier number αt/L²'),
               V('T_i', 'K', 'initial (uniform) temperature'), V('T_inf', 'K', 'ambient temperature'),
               V('L', 'm', 'half-thickness of the slab'), V('k', 'W/(m*K)', 'thermal conductivity'),
               V('h', 'W/(m^2*K)', 'convection coefficient'), V('alpha', 'm^2/s', 'thermal diffusivity', label='α'),
               V('t', 's', 'elapsed time')],
         procedure=_one_term_plane, outputs=['T_center', 'T_surface', 'zeta1', 'C1', 'Bi', 'Fo'],
         example={'T_i': 500.0, 'T_inf': 300.0, 'L': 0.05, 'k': 20.0, 'h': 100.0, 'alpha': 8.0e-6, 't': 100.0},
         keywords=['heisler', 'one-term approximation', 'plane wall', 'eigenvalue', 'transient'],
         assumptions=['First-term approximation valid for Fo > 0.2; a smaller Fo is reported as a warning. '
                      'Uniform initial temperature, constant h. ζ1 is found by root-finding ζ tan ζ = Bi '
                      '(no recalled table values).'],
         refs=['One-term series solution of the transient plane-wall conduction problem']),

    Calc('one_term_cylinder', 'One-term transient solution for an infinite cylinder (Fo > 0.2)',
         vars=[V('T_center', 'K', 'centre temperature at time t'), V('T_surface', 'K', 'surface temperature at time t'),
               V('zeta1', 'rad', 'first eigenvalue ζ1 (root of ζ J1(ζ)/J0(ζ) = Bi)', label='ζ1'),
               V('C1', '', 'first-series coefficient C1'), V('Bi', '', 'Biot number h·r0/k'),
               V('Fo', '', 'Fourier number αt/r0²'),
               V('T_i', 'K', 'initial (uniform) temperature'), V('T_inf', 'K', 'ambient temperature'),
               V('r0', 'm', 'cylinder radius'), V('k', 'W/(m*K)', 'thermal conductivity'),
               V('h', 'W/(m^2*K)', 'convection coefficient'), V('alpha', 'm^2/s', 'thermal diffusivity', label='α'),
               V('t', 's', 'elapsed time')],
         procedure=_one_term_cylinder, outputs=['T_center', 'T_surface', 'zeta1', 'C1', 'Bi', 'Fo'],
         example={'T_i': 500.0, 'T_inf': 300.0, 'r0': 0.02, 'k': 20.0, 'h': 1000.0, 'alpha': 8.0e-6, 't': 20.0},
         keywords=['heisler', 'one-term approximation', 'cylinder', 'eigenvalue', 'transient', 'bessel'],
         assumptions=['First-term approximation valid for Fo > 0.2; a smaller Fo is reported as a warning. '
                      'Uniform initial temperature, constant h, long cylinder (end effects negligible). ζ1 is '
                      'found by root-finding ζ J1(ζ)/J0(ζ) = Bi with the Bessel functions J0, J1.'],
         refs=['One-term series solution of the transient infinite-cylinder conduction problem']),

    Calc('one_term_sphere', 'One-term transient solution for a sphere (Fo > 0.2)',
         vars=[V('T_center', 'K', 'centre temperature at time t'), V('T_surface', 'K', 'surface temperature at time t'),
               V('zeta1', 'rad', 'first eigenvalue ζ1 (root of 1 − ζ cot ζ = Bi)', label='ζ1'),
               V('C1', '', 'first-series coefficient C1'), V('Bi', '', 'Biot number h·r0/k'),
               V('Fo', '', 'Fourier number αt/r0²'),
               V('T_i', 'K', 'initial (uniform) temperature'), V('T_inf', 'K', 'ambient temperature'),
               V('r0', 'm', 'sphere radius'), V('k', 'W/(m*K)', 'thermal conductivity'),
               V('h', 'W/(m^2*K)', 'convection coefficient'), V('alpha', 'm^2/s', 'thermal diffusivity', label='α'),
               V('t', 's', 'elapsed time')],
         procedure=_one_term_sphere, outputs=['T_center', 'T_surface', 'zeta1', 'C1', 'Bi', 'Fo'],
         example={'T_i': 500.0, 'T_inf': 300.0, 'r0': 0.02, 'k': 20.0, 'h': 1000.0, 'alpha': 8.0e-6, 't': 20.0},
         keywords=['heisler', 'one-term approximation', 'sphere', 'eigenvalue', 'transient'],
         assumptions=['First-term approximation valid for Fo > 0.2; a smaller Fo is reported as a warning. '
                      'Uniform initial temperature, constant h. ζ1 is found by root-finding 1 − ζ cot ζ = Bi.'],
         refs=['One-term series solution of the transient sphere conduction problem']),

    Calc('semi_infinite', 'Semi-infinite solid under a sudden surface temperature change',
         vars=[V('T', 'K', 'temperature at depth x and time t'), V('q_s', 'W/m^2', 'surface heat flux', signed=True),
               V('T_i', 'K', 'initial (uniform) temperature'), V('T_s', 'K', 'imposed surface temperature'),
               V('x', 'm', 'depth from the surface'), V('alpha', 'm^2/s', 'thermal diffusivity', label='α'),
               V('t', 's', 'elapsed time'), V('k', 'W/(m*K)', 'thermal conductivity')],
         equations=['T = T_i + (T_s - T_i)*erfc(x/(2*sqrt(alpha*t)))',
                    'q_s = k*(T_s - T_i)/sqrt(pi*alpha*t)'],
         outputs=['T', 'q_s'],
         example={'T_i': 300.0, 'T_s': 400.0, 'x': 0.02, 'alpha': 1.0e-6, 't': 60.0, 'k': 1.5},
         keywords=['semi-infinite', 'error function', 'transient', 'ground'],
         assumptions=['Constant surface temperature T_s applied at t = 0 to a semi-infinite body; '
                      'erfc is the complementary error function.'],
         refs=['Semi-infinite solid: T = T_i + (T_s − T_i) erfc(x/(2√(αt))); q_s = k(T_s−T_i)/√(παt)']),

    # ---- 2-D conduction -----------------------------------------------------------------------------
    Calc('shape_factor_cylinder', '2-D conduction shape factor: buried horizontal cylinder',
         vars=[V('S', 'm', 'conduction shape factor'), V('q', 'W', 'heat rate', signed=True),
               V('L', 'm', 'cylinder length'), V('D', 'm', 'cylinder diameter'), V('z', 'm', 'burial depth (to centreline)'),
               V('k', 'W/(m*K)', 'medium thermal conductivity'), V('dT', 'K', 'temperature difference T1 − T2')],
         equations=['S = 2*pi*L/acosh(2*z/D)', 'q = S*k*dT'], outputs=['S', 'q'],
         example={'L': 10.0, 'D': 0.5, 'z': 1.0, 'k': 1.4, 'dT': 30.0},
         keywords=['shape factor', 'buried pipe', '2-d conduction'],
         assumptions=['Isothermal cylinder in a semi-infinite medium; the exact form S = 2πL/acosh(2z/D) is '
                      'valid for z > D/2 (fully buried).'],
         refs=['Buried cylinder shape factor S = 2πL/acosh(2z/D)']),

    Calc('shape_factor_sphere', '2-D conduction shape factor: buried sphere',
         vars=[V('S', 'm', 'conduction shape factor'), V('q', 'W', 'heat rate', signed=True),
               V('r', 'm', 'sphere radius'), V('z', 'm', 'burial depth (to centre)'),
               V('k', 'W/(m*K)', 'medium thermal conductivity'), V('dT', 'K', 'temperature difference T1 − T2')],
         equations=['S = 4*pi*r/(1 - r/(2*z))', 'q = S*k*dT'], outputs=['S', 'q'],
         example={'r': 0.5, 'z': 1.5, 'k': 1.4, 'dT': 30.0},
         keywords=['shape factor', 'buried tank', '2-d conduction'],
         assumptions=['Isothermal sphere in a semi-infinite medium; valid for z > r.'],
         refs=['Buried sphere shape factor S = 4πr/(1 − r/(2z))']),

    Calc('finite_difference_2d', '2-D steady conduction nodal temperatures (Gauss-Seidel)',
         vars=[V('grid', '', 'grid size [nodes_x, nodes_y] (odd, 3..51 per side)', kind='json'),
               V('T_top', 'K', 'top-edge temperature'), V('T_bottom', 'K', 'bottom-edge temperature'),
               V('T_left', 'K', 'left-edge temperature'), V('T_right', 'K', 'right-edge temperature'),
               V('T_center', 'K', 'temperature at the centre node'),
               V('T_grid', '', 'full temperature matrix [[T, ...], ...] in K', kind='json'),
               V('iterations', '', 'Gauss-Seidel iterations to convergence', integer=True)],
         procedure=_finite_difference_2d, outputs=['T_center', 'T_grid', 'iterations'],
         example={'grid': [9, 9], 'T_top': 373.15, 'T_bottom': 293.15, 'T_left': 293.15, 'T_right': 293.15},
         keywords=['finite difference', 'laplace', 'nodal', 'steady conduction', 'gauss-seidel'],
         assumptions=['Uniform k (Laplace equation), Dirichlet boundary temperatures, square grid with ODD '
                      'node counts (so the centre node is unique), converged to 1e-9 K per node (a hit '
                      'iteration cap is reported as a warning).'],
         refs=['Five-point finite-difference discretisation of ∇²T = 0 solved by Gauss-Seidel iteration']),

    # ---- dimensionless groups -----------------------------------------------------------------------
    Calc('prandtl_number', 'Prandtl number (Pr = μc_p/k)',
         vars=[V('Pr', '', 'Prandtl number μc_p/k'),
               V('mu', 'Pa*s', 'dynamic viscosity', label='μ'),
               V('c_p', 'J/(kg*K)', 'specific heat'), V('k', 'W/(m*K)', 'thermal conductivity')],
         equations=['Pr = mu*c_p/k'], outputs=['Pr'],
         example={'mu': 1.8e-5, 'c_p': 1005.0, 'k': 0.026},
         keywords=['prandtl', 'dimensionless', 'viscosity'],
         assumptions=['Reynolds number is computed separately by fluid_dynamics.reynolds_number.'],
         refs=['Pr = μc_p/k = ν/α']),

    Calc('nusselt_from_h', 'Nusselt number from a heat-transfer coefficient (Nu = hL/k)',
         vars=[V('Nu', '', 'Nusselt number'), V('h', 'W/(m^2*K)', 'heat-transfer coefficient'),
               V('L', 'm', 'characteristic length'), V('k', 'W/(m*K)', 'fluid thermal conductivity')],
         equations=['Nu = h*L/k'], outputs=['Nu'],
         example={'h': 50.0, 'L': 0.1, 'k': 0.026},
         keywords=['nusselt', 'convection', 'dimensionless'],
         assumptions=['Characteristic length matches the one used in the correlation (plate length, '
                      'diameter D, etc.).'],
         refs=['Nu = hL/k']),

    Calc('grashof_rayleigh', 'Grashof and Rayleigh numbers (natural convection)',
         vars=[V('Gr', '', 'Grashof number gβΔTL³/ν²'), V('Ra', '', 'Rayleigh number Gr·Pr'),
               V('g', 'm/s^2', 'gravitational acceleration'), V('beta', '1/K', 'thermal expansion coefficient', label='β'),
               V('dT', 'K', 'temperature difference (magnitude) |T_s − T_inf|'),
               V('L', 'm', 'characteristic length'), V('nu', 'm^2/s', 'kinematic viscosity', label='ν'),
               V('Pr', '', 'Prandtl number')],
         equations=['Gr = g*beta*dT*L**3/nu**2', 'Ra = Gr*Pr'], outputs=['Gr', 'Ra'],
         example={'g': G0, 'beta': 3.0e-3, 'dT': 30.0, 'L': 0.3, 'nu': 1.6e-5, 'Pr': 0.7},
         keywords=['grashof', 'rayleigh', 'natural convection', 'buoyancy'],
         assumptions=['ν is the KINEMATIC viscosity; β = 1/T for an ideal gas.'],
         refs=['Gr = gβΔTL³/ν²; Ra = Gr·Pr']),

    Calc('biot_fourier', 'Biot and Fourier numbers (transient conduction)',
         vars=[V('Bi', '', 'Biot number hL/k'), V('Fo', '', 'Fourier number αt/L²'),
               V('h', 'W/(m^2*K)', 'convection coefficient'), V('L', 'm', 'characteristic length'),
               V('k', 'W/(m*K)', 'solid thermal conductivity'), V('alpha', 'm^2/s', 'thermal diffusivity', label='α'),
               V('t', 's', 'time')],
         equations=['Bi = h*L/k', 'Fo = alpha*t/L**2'], outputs=['Bi', 'Fo'],
         example={'h': 50.0, 'L': 0.05, 'k': 20.0, 'alpha': 8.0e-6, 't': 100.0},
         keywords=['biot', 'fourier', 'transient', 'lumped capacitance'],
         assumptions=['For a solid use L_c = V/A_s; lumped-capacitance holds for Bi < 0.1.'],
         refs=['Bi = hL/k; Fo = αt/L²']),

    Calc('peclet_stanton', 'Peclet and Stanton numbers',
         vars=[V('Pe', '', 'Peclet number Re·Pr'), V('St', '', 'Stanton number Nu/(Re·Pr)'),
               V('Re', '', 'Reynolds number'), V('Pr', '', 'Prandtl number'), V('Nu', '', 'Nusselt number')],
         equations=['Pe = Re*Pr', 'St = Nu/(Re*Pr)'], outputs=['Pe', 'St'],
         example={'Re': 5000.0, 'Pr': 0.7, 'Nu': 30.0},
         keywords=['peclet', 'stanton', 'dimensionless'],
         assumptions=['St = Nu/(Re Pr) = h/(ρ V c_p).'],
         refs=['Pe = Re·Pr; St = Nu/(Re·Pr)']),

    # ---- forced convection correlations -------------------------------------------------------------
    Calc('forced_flat_plate', 'Forced convection over a flat plate (power-law correlation)',
         vars=[V('Nu', '', 'average Nusselt number'), V('h', 'W/(m^2*K)', 'average heat-transfer coefficient'),
               V('Re', '', 'Reynolds number (based on plate length)'), V('Pr', '', 'Prandtl number'),
               V('C', '', 'correlation constant C (table)'), V('m', '', 'Reynolds exponent m (table)'),
               V('n', '', 'Prandtl exponent n (table)'), V('k', 'W/(m*K)', 'fluid thermal conductivity'),
               V('L', 'm', 'plate length')],
         equations=['Nu = C*Re**m*Pr**n', 'h = Nu*k/L'], outputs=['Nu', 'h'],
         example={'Re': 5e5, 'Pr': 0.7, 'C': 0.037, 'm': 0.8, 'n': 1.0 / 3.0, 'k': 0.026, 'L': 1.0},
         keywords=['flat plate', 'forced convection', 'correlation'],
         assumptions=['C, m, n are correlation constants from the textbook table for the geometry and '
                      'Reynolds range (laminar AVERAGE C=0.664 m=0.5; turbulent average C=0.037 m=0.8, n=1/3). '
                      'For a plate with both laminar and turbulent regions use flat_plate_mixed.'],
         refs=['Nu = C Re^m Pr^n (flat-plate correlation, constants from table)']),

    Calc('flat_plate_mixed', 'Average Nusselt number over a flat plate with mixed boundary layers',
         vars=[V('Nu', '', 'average Nusselt number'), V('h', 'W/(m^2*K)', 'average heat-transfer coefficient'),
               V('Re', '', 'Reynolds number (based on plate length)'), V('Pr', '', 'Prandtl number'),
               V('k', 'W/(m*K)', 'fluid thermal conductivity'), V('L', 'm', 'plate length')],
         equations=['Nu = (0.037*Re**0.8 - 871)*Pr**(1/3)', 'h = Nu*k/L'],
         outputs=['Nu', 'h'],
         example={'Re': 2e6, 'Pr': 0.7, 'k': 0.026, 'L': 1.0},
         keywords=['flat plate', 'forced convection', 'mixed boundary layer', 'turbulent'],
         assumptions=['Average Nusselt number for a flat plate with transition at Re_c = 5e5; valid '
                      '5e5 < Re < 1e8, 0.6 < Pr < 60, isothermal plate, properties at the film temperature.'],
         refs=['Mixed-boundary-layer flat plate: Nu = (0.037 Re^0.8 − 871) Pr^⅓']),

    Calc('cylinder_churchill_bernstein', 'Forced convection over a circular cylinder (Churchill–Bernstein)',
         vars=[V('Nu', '', 'average Nusselt number'), V('h', 'W/(m^2*K)', 'average heat-transfer coefficient'),
               V('Re', '', 'Reynolds number (based on D)'), V('Pr', '', 'Prandtl number'),
               V('k', 'W/(m*K)', 'fluid thermal conductivity'), V('D', 'm', 'cylinder diameter')],
         equations=['Nu = 0.3 + 0.62*Re**0.5*Pr**(1/3)/(1 + (0.4/Pr)**(2/3))**0.25*(1 + (Re/282000)**(5/8))**(4/5)',
                    'h = Nu*k/D'],
         outputs=['Nu', 'h'],
         example={'Re': 40000.0, 'Pr': 0.7, 'k': 0.026, 'D': 0.02},
         keywords=['cylinder', 'churchill-bernstein', 'cross flow', 'forced convection'],
         assumptions=['Valid for Re·Pr > 0.2; all properties at the film temperature.'],
         refs=['Churchill & Bernstein (1977) cylinder correlation (closed form, recalled)']),

    Calc('sphere_whitaker', 'Forced convection over a sphere (Whitaker)',
         vars=[V('Nu', '', 'average Nusselt number'), V('h', 'W/(m^2*K)', 'average heat-transfer coefficient'),
               V('Re', '', 'Reynolds number (based on D)'), V('Pr', '', 'Prandtl number'),
               V('mu_ratio', '', 'viscosity ratio μ/μ_s (≈1 for gases)'), V('k', 'W/(m*K)', 'fluid thermal conductivity'),
               V('D', 'm', 'sphere diameter')],
         equations=['Nu = 2 + (0.4*Re**0.5 + 0.06*Re**(2/3))*Pr**0.4*mu_ratio**0.25', 'h = Nu*k/D'],
         outputs=['Nu', 'h'],
         example={'Re': 10000.0, 'Pr': 0.7, 'mu_ratio': 1.0, 'k': 0.026, 'D': 0.02},
         keywords=['sphere', 'whitaker', 'forced convection'],
         assumptions=['Whitaker (1972); valid 3.5 < Re < 7.6e4 and 0.71 < Pr < 380; properties at T_inf '
                      'except μ_s at the surface.'],
         refs=['Whitaker sphere correlation (recalled)']),

    Calc('tube_bank_zukauskas', 'Forced convection over a tube bank (Žukauskas)',
         vars=[V('Nu', '', 'average Nusselt number'), V('h', 'W/(m^2*K)', 'average heat-transfer coefficient'),
               V('Re', '', 'Reynolds number (based on D and V_max)'), V('Pr', '', 'Prandtl number'),
               V('C', '', 'Žukauskas constant C (table)'), V('m', '', 'Žukauskas exponent m (table)'),
               V('Pr_s', '', 'Prandtl number at the surface temperature'),
               V('k', 'W/(m*K)', 'fluid thermal conductivity'), V('D', 'm', 'tube diameter')],
         equations=['Nu = C*Re**m*Pr**0.36*(Pr/Pr_s)**0.25', 'h = Nu*k/D'], outputs=['Nu', 'h'],
         example={'Re': 5000.0, 'Pr': 0.7, 'C': 0.27, 'm': 0.63, 'Pr_s': 0.71, 'k': 0.026, 'D': 0.02},
         keywords=['tube bank', 'zukauskas', 'heat exchanger', 'cross flow'],
         assumptions=['C and m are the Žukauskas tube-bank constants for the arrangement (inline/staggered) '
                      'and Reynolds range (user inputs, table), for a bank with many rows (N_L ≥ 20); the '
                      'row-number correction factor C2 for short banks is NOT included here.'],
         refs=['Žukauskas tube-bank correlation (constants from table)']),

    Calc('internal_entrance_length', 'Hydrodynamic entrance lengths for internal flow',
         vars=[V('L_h_lam', 'm', 'laminar entrance length 0.05·Re·D'), V('L_h_turb', 'm', 'turbulent entrance length ~10·D'),
               V('Re', '', 'Reynolds number (based on D)'), V('D', 'm', 'tube diameter')],
         equations=['L_h_lam = 0.05*Re*D', 'L_h_turb = 10*D'], outputs=['L_h_lam', 'L_h_turb'],
         example={'Re': 1500.0, 'D': 0.02},
         keywords=['entrance length', 'developing flow', 'internal flow'],
         assumptions=['Fully developed when L > L_h; laminar Re < 2300.'],
         refs=['L_h,lam = 0.05 Re D; L_h,turb ≈ 10 D']),

    Calc('thermal_entrance_length', 'Thermal entrance length for laminar internal flow',
         vars=[V('L_t_lam', 'm', 'laminar thermal entrance length 0.05·Re·Pr·D'),
               V('Re', '', 'Reynolds number (based on D)'), V('Pr', '', 'Prandtl number'),
               V('D', 'm', 'tube diameter')],
         equations=['L_t_lam = 0.05*Re*Pr*D'], outputs=['L_t_lam'],
         example={'Re': 1500.0, 'Pr': 0.7, 'D': 0.02},
         keywords=['thermal entrance length', 'developing flow', 'internal flow'],
         assumptions=['Thermal entrance length for laminar flow (Pr > 1 liquids develop thermally more '
                      'slowly than hydrodynamically); for turbulent flow L_t ≈ L_h.'],
         refs=['L_t,lam = 0.05 Re Pr D']),

    Calc('dittus_boelter', 'Turbulent internal flow (Dittus–Boelter)',
         vars=[V('mode', '', 'heating (n = 0.4) or cooling (n = 0.3) of the fluid', kind='text',
                 choices=['heating', 'cooling']),
               V('Nu', '', 'average Nusselt number'), V('h', 'W/(m^2*K)', 'average heat-transfer coefficient'),
               V('Re', '', 'Reynolds number (based on D)'), V('Pr', '', 'Prandtl number'),
               V('k', 'W/(m*K)', 'fluid thermal conductivity'), V('D', 'm', 'tube diameter')],
         procedure=_dittus_boelter, outputs=['h', 'Nu'],
         example={'mode': 'heating', 'Re': 20000.0, 'Pr': 5.0, 'k': 0.6, 'D': 0.02},
         keywords=['dittus-boelter', 'internal flow', 'turbulent', 'pipe'],
         assumptions=['Valid for Re > 10000, 0.6 < Pr < 160 and L/D > 10; moderate temperature differences. '
                      'Off-range Re/Pr are reported as warnings. For accuracy over wide ranges use Gnielinski '
                      '(not implemented).'],
         refs=['Dittus–Boelter (1930) Nu = 0.023 Re^0.8 Pr^n, n = 0.4 heating / 0.3 cooling']),

    Calc('laminar_fully_developed', 'Fully developed laminar internal-flow Nusselt number',
         vars=[V('mode', '', 'boundary condition', kind='text', choices=['const_T', 'const_q']),
               V('Nu', '', 'fully developed Nusselt number')],
         procedure=_laminar_nu, outputs=['Nu'],
         example={'mode': 'const_T'},
         keywords=['laminar', 'fully developed', 'constant wall temperature', 'constant heat flux', 'pipe'],
         assumptions=['Fully developed laminar flow in a circular tube: Nu = 3.66 for a constant surface '
                      'temperature and Nu = 4.36 for a constant surface heat flux (independent of Re, Pr).'],
         refs=['Fully developed laminar circular-tube Nu: 3.66 (const T), 4.36 (const q)']),

    # ---- natural convection -------------------------------------------------------------------------
    Calc('natural_convection_powerlaw', 'Natural convection (power-law Nu = C·Ra^n)',
         vars=[V('Nu', '', 'average Nusselt number'), V('h', 'W/(m^2*K)', 'average heat-transfer coefficient'),
               V('Ra', '', 'Rayleigh number'), V('C', '', 'correlation constant C (table)'), V('n', '', 'exponent n (table)'),
               V('k', 'W/(m*K)', 'fluid thermal conductivity'), V('L', 'm', 'characteristic length')],
         equations=['Nu = C*Ra**n', 'h = Nu*k/L'], outputs=['Nu', 'h'],
         example={'Ra': 1e8, 'C': 0.59, 'n': 0.25, 'k': 0.026, 'L': 0.3},
         keywords=['natural convection', 'free convection', 'correlation'],
         assumptions=['C and n from the textbook table for the geometry and Ra range (vertical plate laminar '
                      'C=0.59 n=1/4; turbulent C=0.10 n=1/3). Properties at the film temperature.'],
         refs=['Nu = C Ra^n (natural-convection correlation, constants from table)']),

    Calc('vertical_plate_churchill_chu', 'Natural convection on a vertical plate (Churchill–Chu)',
         vars=[V('Nu', '', 'average Nusselt number'), V('h', 'W/(m^2*K)', 'average heat-transfer coefficient'),
               V('Ra', '', 'Rayleigh number (based on plate height)'), V('Pr', '', 'Prandtl number'),
               V('k', 'W/(m*K)', 'fluid thermal conductivity'), V('L', 'm', 'plate height')],
         equations=['Nu = (0.825 + 0.387*Ra**(1/6)/(1 + (0.492/Pr)**(9/16))**(8/27))**2', 'h = Nu*k/L'],
         outputs=['Nu', 'h'],
         example={'Ra': 1e9, 'Pr': 0.7, 'k': 0.026, 'L': 0.5},
         keywords=['vertical plate', 'churchill-chu', 'natural convection'],
         assumptions=['Churchill–Chu (1975) for an isothermal vertical plate; valid over the full Ra range '
                      '(laminar and turbulent) for 0 < Pr.'],
         refs=['Churchill & Chu vertical-plate correlation (recalled)']),

    # ---- boiling and condensation -------------------------------------------------------------------
    Calc('nucleate_boiling_rohsenow', 'Nucleate pool boiling heat flux (Rohsenow)',
         vars=[V('q', 'W/m^2', 'boiling heat flux'), V('mu_l', 'Pa*s', 'saturated-liquid dynamic viscosity', label='μl'),
               V('h_fg', 'J/kg', 'latent heat of vaporisation'), V('g', 'm/s^2', 'gravitational acceleration'),
               V('rho_l', 'kg/m^3', 'saturated-liquid density', label='ρl'), V('rho_v', 'kg/m^3', 'saturated-vapour density', label='ρv'),
               V('sigma', 'N/m', 'surface tension', label='σ'), V('cp_l', 'J/(kg*K)', 'saturated-liquid specific heat'),
               V('dT_e', 'K', 'wall superheat ΔT_e = T_s − T_sat', label='ΔTe'),
               V('C_sf', '', 'Rohsenow surface–fluid coefficient (table)'), V('Pr_l', '', 'liquid Prandtl number'),
               V('n', '', 'Rohsenow exponent (1.0 water, 1.7 others)')],
         equations=['q = mu_l*h_fg*(g*(rho_l - rho_v)/sigma)**0.5*(cp_l*dT_e/(C_sf*h_fg*Pr_l**n))**3'],
         outputs=['q'],
         example={'mu_l': 2.8e-4, 'h_fg': 2.26e6, 'g': G0, 'rho_l': 958.0, 'rho_v': 0.6, 'sigma': 0.0589,
                  'cp_l': 4217.0, 'dT_e': 10.0, 'C_sf': 0.013, 'Pr_l': 1.75, 'n': 1.0},
         keywords=['pool boiling', 'rohsenow', 'nucleate boiling', 'superheat'],
         assumptions=['Nucleate pool boiling; C_sf and n are the surface–fluid coefficients (table: '
                      'water–polished copper C_sf ≈ 0.013, n = 1.0). Fluid properties at the saturation '
                      'temperature.'],
         refs=['Rohsenow (1952) nucleate pool boiling correlation (coefficients from table)']),

    Calc('critical_heat_flux_zuber', 'Critical (peak) pool boiling heat flux (Zuber)',
         vars=[V('q_max', 'W/m^2', 'critical heat flux'), V('C', '', 'Zuber constant (0.131 large flat plate)'),
               V('h_fg', 'J/kg', 'latent heat of vaporisation'), V('rho_v', 'kg/m^3', 'saturated-vapour density', label='ρv'),
               V('sigma', 'N/m', 'surface tension', label='σ'), V('g', 'm/s^2', 'gravitational acceleration'),
               V('rho_l', 'kg/m^3', 'saturated-liquid density', label='ρl')],
         equations=['q_max = C*h_fg*rho_v**0.5*(sigma*g*(rho_l - rho_v))**0.25'], outputs=['q_max'],
         example={'C': 0.131, 'h_fg': 2.26e6, 'rho_v': 0.6, 'sigma': 0.0589, 'g': G0, 'rho_l': 958.0},
         keywords=['critical heat flux', 'chf', 'burnout', 'zuber', 'peak flux'],
         assumptions=['Saturated pool boiling on a large horizontal flat plate: the theoretical Zuber '
                      'constant is π/24 ≈ 0.131; Lienhard–Dhir recommend C = 0.149 for large flat plates. '
                      'Other geometries have different C.'],
         refs=['Zuber (1959) critical heat flux q_max = C h_fg ρ_v^½ (σg(ρ_l − ρ_v))^¼; C = π/24 ≈ 0.131 '
               '(Lienhard–Dhir: 0.149)']),

    Calc('film_condensation_vertical', 'Laminar film condensation on a vertical plate (Nusselt)',
         vars=[V('h', 'W/(m^2*K)', 'average condensation heat-transfer coefficient'),
               V('q', 'W', 'condensation heat rate', signed=True),
               V('g', 'm/s^2', 'gravitational acceleration'), V('rho_l', 'kg/m^3', 'liquid density', label='ρl'),
               V('rho_v', 'kg/m^3', 'vapour density', label='ρv'), V('h_fg', 'J/kg', 'modified latent heat h_fg′'),
               V('k_l', 'W/(m*K)', 'liquid thermal conductivity'), V('mu_l', 'Pa*s', 'liquid dynamic viscosity', label='μl'),
               V('T_sat', 'K', 'saturation temperature'), V('T_s', 'K', 'surface (wall) temperature'),
               V('L', 'm', 'plate height'), V('A', 'm^2', 'plate area')],
         equations=['h = 0.943*(g*rho_l*(rho_l - rho_v)*h_fg*k_l**3/(mu_l*(T_sat - T_s)*L))**0.25',
                    'q = h*A*(T_sat - T_s)'],
         outputs=['h', 'q'],
         example={'g': G0, 'rho_l': 958.0, 'rho_v': 0.6, 'h_fg': 2.32e6, 'k_l': 0.68, 'mu_l': 2.8e-4,
                  'T_sat': 373.15, 'T_s': 353.15, 'L': 0.5, 'A': 0.5},
         keywords=['condensation', 'film', 'nusselt', 'vertical plate'],
         assumptions=['Laminar film condensation on a vertical plate; use h_fg′ = h_fg + 0.68 c_p,l (T_sat − T_s) '
                      'to account for liquid subcooling. Re < 1800 for laminar film.'],
         refs=['Nusselt (1916) laminar film condensation h = 0.943[gρ_l(ρ_l−ρ_v)h_fg′k_l³/(μ_lΔTL)]^¼']),

    # ---- thermal radiation --------------------------------------------------------------------------
    Calc('blackbody_emission', 'Blackbody emissive power (Stefan–Boltzmann)',
         vars=[V('E_b', 'W/m^2', 'total emissive power'), V('sigma_sb', 'W/(m^2*K^4)', 'Stefan–Boltzmann constant'),
               V('T', 'K', 'absolute surface temperature')],
         equations=['E_b = sigma_sb*T**4'], outputs=['E_b'],
         example={'sigma_sb': SIGMA_SB, 'T': 1000.0},
         keywords=['blackbody', 'stefan-boltzmann', 'emissive power'],
         assumptions=['σ = 5.670374419e-8 W/(m²·K⁴) (CODATA); T in kelvin.'],
         refs=['Stefan–Boltzmann law E_b = σT⁴']),

    Calc('wien_displacement', "Wien's displacement law",
         vars=[V('lambda_max', 'm', 'wavelength of peak emission', label='λmax'), V('b', 'm*K', "Wien's constant"),
               V('T', 'K', 'absolute temperature')],
         equations=['lambda_max = b/T'], outputs=['lambda_max'],
         example={'b': WIEN_B, 'T': 5800.0},
         keywords=['wien', 'peak wavelength', 'blackbody', 'displacement'],
         assumptions=["b = 2.897771955e-3 m·K (CODATA); gives the peak of E_bλ (sun ≈ 0.5 µm at 5800 K)."],
         refs=["Wien's displacement law λ_max T = b"]),

    Calc('planck_spectral', 'Planck spectral (monochromatic) blackbody emissive power',
         vars=[V('E_bl', 'W/m^3', 'spectral emissive power E_bλ(λ,T)', lo=1e-20, hi=1e30, label='E_bλ'),
               V('C1', 'W*m^2', 'first radiation constant 2πhc²', lo=1e-20, hi=1e-10),
               V('C2', 'm*K', 'second radiation constant hc/k', lo=1e-4, hi=1e-1),
               V('lam', 'm', 'wavelength', lo=1e-9, hi=1e-3, label='λ'),
               V('T', 'K', 'absolute temperature', lo=100.0, hi=20000.0)],
         equations=['E_bl = C1/(lam**5*(exp(C2/(lam*T)) - 1))'], outputs=['E_bl'],
         example={'C1': C1_RAD, 'C2': C2_RAD, 'lam': 1.0e-6, 'T': 5800.0},
         keywords=['planck', 'spectral', 'blackbody', 'monochromatic'],
         assumptions=['C1 = 3.741771852e-16 W·m², C2 = 1.438776877e-2 m·K (CODATA).'],
         refs=['Planck distribution E_bλ = C1/(λ⁵(e^(C2/λT) − 1))']),

    Calc('band_fraction', 'Blackbody band fraction F(λ1→λ2) by numerical integration of Planck',
         vars=[V('F', '', 'fraction of total emission in [λ1, λ2]', hi=1.0),
               V('T', 'K', 'absolute temperature', lo=50.0, hi=20000.0),
               V('lambda1', 'm', 'lower wavelength', lo=0.1e-6, hi=100e-6, label='λ1'),
               V('lambda2', 'm', 'upper wavelength', lo=0.1e-6, hi=100e-6, label='λ2')],
         procedure=_band_fraction, outputs=['F'],
         example={'T': 500.0, 'lambda1': 8.0e-6, 'lambda2': 12.0e-6},
         keywords=['band fraction', 'blackbody function', 'planck integration'],
         assumptions=['F = ∫_λ1^λ2 E_bλ dλ / (σT⁴), integrated numerically (no tabulated F(0→λT) values).'],
         refs=['Blackbody radiation function F = ∫E_bλ dλ/(σT⁴)']),

    Calc('surface_radiation_props', 'Surface radiation properties: reflectivity, absorptivity, transmissivity',
         vars=[V('tau', '', 'transmissivity τ'), V('alpha', '', 'absorptivity α = ε (gray surface)'),
               V('rho', '', 'reflectivity ρ'), V('epsilon', '', 'emissivity ε', label='ε')],
         equations=['tau = 1 - rho - alpha', 'alpha = epsilon'], outputs=['tau', 'alpha'],
         example={'rho': 0.1, 'epsilon': 0.6},
         keywords=['emissivity', 'absorptivity', 'reflectivity', 'transmissivity', 'kirchhoff'],
         assumptions=['Energy balance α + ρ + τ = 1; Kirchhoff α = ε for a gray surface in local equilibrium. '
                      'Opaque surface: τ = 0.'],
         refs=['α + ρ + τ = 1; Kirchhoff α = ε']),

    Calc('parallel_plates_exchange', 'Radiation exchange between two large parallel plates',
         vars=[V('q', 'W/m^2', 'net radiation heat flux from surface 1 to 2', signed=True),
               V('sigma_sb', 'W/(m^2*K^4)', 'Stefan–Boltzmann constant'), V('T1', 'K', 'temperature of surface 1'),
               V('T2', 'K', 'temperature of surface 2'), V('eps1', '', 'emissivity of surface 1', label='ε1'),
               V('eps2', '', 'emissivity of surface 2', label='ε2')],
         equations=['q = sigma_sb*(T1**4 - T2**4)/(1/eps1 + 1/eps2 - 1)'], outputs=['q'],
         example={'sigma_sb': SIGMA_SB, 'T1': 800.0, 'T2': 300.0, 'eps1': 0.8, 'eps2': 0.6},
         keywords=['parallel plates', 'radiation', 'gray surfaces', 'view factor'],
         assumptions=['Two infinite parallel gray plates (view factor F12 = 1); net flux from 1 to 2.'],
         refs=['q = σ(T1⁴ − T2⁴)/(1/ε1 + 1/ε2 − 1)']),

    Calc('small_body_enclosure', 'Radiation from a small body in a large isothermal enclosure',
         vars=[V('q', 'W', 'net radiation heat rate (body to enclosure)', signed=True),
               V('eps', '', 'body emissivity', label='ε'), V('sigma_sb', 'W/(m^2*K^4)', 'Stefan–Boltzmann constant'),
               V('A', 'm^2', 'body surface area'), V('T1', 'K', 'body temperature'), V('T2', 'K', 'enclosure temperature')],
         equations=['q = eps*sigma_sb*A*(T1**4 - T2**4)'], outputs=['q'],
         example={'eps': 0.7, 'sigma_sb': SIGMA_SB, 'A': 0.5, 'T1': 500.0, 'T2': 300.0},
         keywords=['enclosure', 'radiation', 'small body', 'gray'],
         assumptions=['Small convex body (A1 ≪ A2, F12 = 1) inside a large enclosure treated as black.'],
         refs=['q = ε σ A (T1⁴ − T2⁴)']),

    Calc('two_surface_enclosure', 'Radiation exchange between two gray surfaces (general enclosure)',
         vars=[V('q12', 'W', 'net radiation from surface 1 to surface 2', signed=True),
               V('sigma_sb', 'W/(m^2*K^4)', 'Stefan–Boltzmann constant'), V('T1', 'K', 'temperature of surface 1'),
               V('T2', 'K', 'temperature of surface 2'), V('eps1', '', 'emissivity of surface 1', label='ε1'),
               V('eps2', '', 'emissivity of surface 2', label='ε2'), V('A1', 'm^2', 'area of surface 1'),
               V('A2', 'm^2', 'area of surface 2'), V('F12', '', 'view factor from surface 1 to 2', hi=1.0)],
         equations=['q12 = sigma_sb*(T1**4 - T2**4)/((1 - eps1)/(eps1*A1) + 1/(A1*F12) + (1 - eps2)/(eps2*A2))'],
         outputs=['q12'],
         example={'sigma_sb': SIGMA_SB, 'T1': 800.0, 'T2': 400.0, 'eps1': 0.8, 'eps2': 0.5, 'A1': 1.0, 'A2': 2.0, 'F12': 0.9},
         keywords=['radiation network', 'two-surface enclosure', 'view factor'],
         assumptions=['Two diffuse-gray surfaces forming an enclosure; F12 is the view factor (user input, '
                      'from geometry tables or the reciprocity relation).'],
         refs=['Two-surface enclosure: q12 = σ(T1⁴−T2⁴)/[(1−ε1)/(ε1A1) + 1/(A1F12) + (1−ε2)/(ε2A2)]']),

    Calc('radiation_shield', 'Radiation shield between parallel plates (equal emissivities)',
         vars=[V('q', 'W/m^2', 'net radiation heat flux with shields', signed=True),
               V('sigma_sb', 'W/(m^2*K^4)', 'Stefan–Boltzmann constant'), V('T1', 'K', 'temperature of surface 1'),
               V('T2', 'K', 'temperature of surface 2'), V('eps', '', 'emissivity (all surfaces equal)', label='ε'),
               V('n', '', 'number of shields', integer=True)],
         equations=['q = sigma_sb*(T1**4 - T2**4)/((n + 1)*(2/eps - 1))'], outputs=['q'],
         example={'sigma_sb': SIGMA_SB, 'T1': 800.0, 'T2': 300.0, 'eps': 0.2, 'n': 1.0},
         keywords=['radiation shield', 'superinsulation', 'parallel plates'],
         assumptions=['n thin shields between two large parallel plates, all surfaces with the same '
                      'emissivity ε; n = 0 gives the unshielded result and each shield cuts the flux by 1/(n+1).'],
         refs=['Shielded parallel plates: q = σ(T1⁴−T2⁴)/[(n+1)(2/ε − 1)]']),

    Calc('view_factor_reciprocity', 'View factor reciprocity (A1 F12 = A2 F21)',
         vars=[V('F21', '', 'view factor from surface 2 to 1', hi=1.0),
               V('A1', 'm^2', 'area of surface 1'), V('F12', '', 'view factor from surface 1 to 2', hi=1.0),
               V('A2', 'm^2', 'area of surface 2')],
         equations=['F21 = A1*F12/A2'], outputs=['F21'],
         example={'A1': 1.0, 'F12': 0.9, 'A2': 2.0},
         keywords=['view factor', 'reciprocity', 'radiation'],
         assumptions=['Reciprocity between any two surfaces of an enclosure.'],
         refs=['Reciprocity relation A_i F_ij = A_j F_ji']),

    # ---- heat exchangers ----------------------------------------------------------------------------
    Calc('heat_capacity_ratio', 'Heat capacity ratio (C_r = C_min/C_max)',
         vars=[V('Cr', '', 'heat capacity ratio C_min/C_max', hi=1.0),
               V('C_min', 'W/K', 'minimum heat capacity rate'), V('C_max', 'W/K', 'maximum heat capacity rate')],
         equations=['Cr = C_min/C_max'], outputs=['Cr'],
         example={'C_min': 400.0, 'C_max': 800.0},
         keywords=['heat capacity ratio', 'effectiveness', 'ntu'],
         assumptions=['C = m_dot c_p for each stream; 0 ≤ Cr ≤ 1.'],
         refs=['C_r = C_min/C_max']),

    Calc('ntu', 'Number of transfer units (NTU = UA/C_min)',
         vars=[V('NTU', '', 'number of transfer units'), V('U', 'W/(m^2*K)', 'overall heat-transfer coefficient'),
               V('A', 'm^2', 'heat-transfer area'), V('C_min', 'W/K', 'minimum heat capacity rate')],
         equations=['NTU = U*A/C_min'], outputs=['NTU'],
         example={'U': 1000.0, 'A': 2.0, 'C_min': 400.0},
         keywords=['ntu', 'effectiveness', 'heat exchanger'],
         refs=['NTU = UA/C_min']),

    Calc('effectiveness_counter', 'Effectiveness of a counter-flow heat exchanger',
         vars=[V('eps', '', 'effectiveness', hi=1.0, label='ε'), V('NTU', '', 'number of transfer units'),
               V('Cr', '', 'heat capacity ratio (0 ≤ Cr ≤ 1)', hi=1.0)],
         procedure=_effectiveness_counter, outputs=['eps'],
         example={'NTU': 2.0, 'Cr': 0.5},
         keywords=['counterflow', 'effectiveness', 'ntu'],
         assumptions=['Counter-flow exchanger; Cr = 1 gives ε = NTU/(1 + NTU).'],
         refs=['Counter-flow effectiveness ε = (1 − e^(−NTU(1−Cr)))/(1 − Cr e^(−NTU(1−Cr))); ε = NTU/(1+NTU) at Cr = 1']),

    Calc('effectiveness_parallel', 'Effectiveness of a parallel-flow heat exchanger',
         vars=[V('eps', '', 'effectiveness', hi=1.0, label='ε'), V('NTU', '', 'number of transfer units'),
               V('Cr', '', 'heat capacity ratio', hi=1.0)],
         equations=['eps = (1 - exp(-NTU*(1 + Cr)))/(1 + Cr)'], outputs=['eps'],
         example={'NTU': 2.0, 'Cr': 0.5},
         keywords=['parallel flow', 'effectiveness', 'ntu'],
         assumptions=['Parallel-flow exchanger.'],
         refs=['Parallel-flow effectiveness ε = (1 − e^(−NTU(1+Cr)))/(1 + Cr)']),

    Calc('effectiveness_phase_change', 'Effectiveness with a phase-change stream (C_r = 0)',
         vars=[V('eps', '', 'effectiveness', hi=1.0, label='ε'), V('NTU', '', 'number of transfer units')],
         equations=['eps = 1 - exp(-NTU)'], outputs=['eps'],
         example={'NTU': 1.5},
         keywords=['boiling', 'condensing', 'phase change', 'effectiveness'],
         assumptions=['One stream boils or condenses (C_max → ∞, Cr = 0): valid for any flow arrangement.'],
         refs=['ε = 1 − e^(−NTU) for Cr = 0']),

    Calc('exchanger_duty', 'Heat-exchanger heat rate from effectiveness (q = ε C_min ΔT_max)',
         vars=[V('q', 'W', 'actual heat-transfer rate', signed=True), V('eps', '', 'effectiveness', hi=1.0, label='ε'),
               V('C_min', 'W/K', 'minimum heat capacity rate'), V('Th_in', 'K', 'hot inlet temperature'),
               V('Tc_in', 'K', 'cold inlet temperature')],
         equations=['q = eps*C_min*(Th_in - Tc_in)'], outputs=['q'],
         example={'eps': 0.7, 'C_min': 400.0, 'Th_in': 373.15, 'Tc_in': 293.15},
         keywords=['heat exchanger', 'duty', 'effectiveness-ntu'],
         assumptions=['ΔT_max = Th_in − Tc_in (the maximum possible temperature difference).'],
         refs=['q = ε C_min (Th_in − Tc_in)']),

    Calc('heat_exchanger_lmtd', 'Log-mean temperature difference (LMTD)',
         vars=[V('Th_in', 'K', 'hot inlet temperature'), V('Th_out', 'K', 'hot outlet temperature'),
               V('Tc_in', 'K', 'cold inlet temperature'), V('Tc_out', 'K', 'cold outlet temperature'),
               V('flow', '', 'flow arrangement', kind='text', choices=['counter', 'parallel']),
               V('LMTD', 'K', 'log-mean temperature difference')],
         procedure=_lmtd, outputs=['LMTD'],
         example={'Th_in': 383.15, 'Th_out': 323.15, 'Tc_in': 298.15, 'Tc_out': 338.15, 'flow': 'counter'},
         keywords=['lmtd', 'log mean temperature difference', 'heat exchanger'],
         assumptions=['Single-pass, constant specific heat, no heat loss (via common.lmtd). Multi-pass and '
                      'cross-flow exchangers apply a correction factor F to the counter-flow LMTD.'],
         refs=['LMTD = (ΔT1 − ΔT2)/ln(ΔT1/ΔT2) via common.lmtd']),

    Calc('heat_exchanger_sizing', 'Heat-exchanger sizing (Q = U A F ΔT_lm)',
         vars=[V('Q', 'W', 'heat-transfer rate', signed=True), V('U', 'W/(m^2*K)', 'overall heat-transfer coefficient'),
               V('A', 'm^2', 'heat-transfer area'), V('F', '', 'LMTD correction factor (≤1, chart)'),
               V('LMTD', 'K', 'log-mean temperature difference (counter-flow)')],
         equations=['Q = U*A*F*LMTD'], outputs=['Q'],
         example={'U': 1000.0, 'A': 2.0, 'F': 0.95, 'LMTD': 34.0},
         keywords=['heat exchanger area', 'sizing', 'correction factor', 'lmtd'],
         assumptions=['F is the LMTD correction factor from exchanger charts (F = 1 for counter-flow); '
                      'solve for A to size the exchanger. Add fouling via the overall-U calculator.'],
         refs=['Q = U A F ΔT_lm']),

    # ---- mass transfer ------------------------------------------------------------------------------
    Calc('fick_diffusion', "Fick's law: steady diffusion of species A through a slab",
         vars=[V('m_dot', 'kg/s', 'mass diffusion rate of species A', signed=True),
               V('D_AB', 'm^2/s', 'binary diffusion coefficient'), V('A', 'm^2', 'diffusion area'),
               V('rho_A1', 'kg/m^3', 'concentration of A on side 1'), V('rho_A2', 'kg/m^3', 'concentration of A on side 2'),
               V('L', 'm', 'slab thickness')],
         equations=['m_dot = D_AB*A*(rho_A1 - rho_A2)/L'], outputs=['m_dot'],
         example={'D_AB': 2.5e-5, 'A': 0.01, 'rho_A1': 0.02, 'rho_A2': 0.0, 'L': 0.01},
         keywords=['fick', 'diffusion', 'mass transfer', 'concentration'],
         assumptions=['Steady one-dimensional diffusion with a linear concentration profile; ρ_A1 > ρ_A2.'],
         refs=["Fick's law m_dot = D_AB A Δρ_A / L"]),

    Calc('schmidt_number', 'Schmidt number (Sc = ν/D_AB)',
         vars=[V('Sc', '', 'Schmidt number'), V('nu', 'm^2/s', 'kinematic viscosity', label='ν'),
               V('D_AB', 'm^2/s', 'binary diffusion coefficient')],
         equations=['Sc = nu/D_AB'], outputs=['Sc'],
         example={'nu': 1.5e-5, 'D_AB': 2.5e-5},
         keywords=['schmidt', 'mass transfer', 'dimensionless'],
         refs=['Sc = ν/D_AB']),

    Calc('sherwood_number', 'Sherwood number (Sh = h_m L/D_AB)',
         vars=[V('Sh', '', 'Sherwood number'), V('h_m', 'm/s', 'mass-transfer coefficient'),
               V('L', 'm', 'characteristic length'), V('D_AB', 'm^2/s', 'binary diffusion coefficient')],
         equations=['Sh = h_m*L/D_AB'], outputs=['Sh'],
         example={'h_m': 0.01, 'L': 0.1, 'D_AB': 2.5e-5},
         keywords=['sherwood', 'mass transfer', 'dimensionless'],
         refs=['Sh = h_m L/D_AB']),

    Calc('sherwood_correlation', 'Convective mass-transfer correlation (Sh = C Re^m Sc^n)',
         vars=[V('Sh', '', 'average Sherwood number'), V('Re', '', 'Reynolds number'),
               V('Sc', '', 'Schmidt number'), V('C', '', 'correlation constant C (table)'),
               V('m', '', 'Reynolds exponent m (table)'), V('n', '', 'Schmidt exponent n (table)')],
         equations=['Sh = C*Re**m*Sc**n'], outputs=['Sh'],
         example={'Re': 50000.0, 'Sc': 0.6, 'C': 0.037, 'm': 0.8, 'n': 1.0 / 3.0},
         keywords=['mass transfer correlation', 'heat-mass analogy', 'sherwood'],
         assumptions=['Heat–mass analogy form; C, m, n from the mass-transfer table for the geometry.'],
         refs=['Sh = C Re^m Sc^n (mass-transfer analogy)']),
]
